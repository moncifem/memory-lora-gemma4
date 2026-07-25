#!/usr/bin/env python3
"""Test a generated adapter the way it was TRAINED: repo Q&A.

`diagnose_head.py` scores raw repo text, which is the wrong distribution for a
head trained on "Q: <question>\\nA:" -> answer. It is the right tool for
detecting an adapter that is inert or collapsed (bad at everything), but a
working head specializes on the QA format and can look worse on plain text
while being dramatically better at the task it was trained for.

This scores the trained format, on the same model instance, adapter on vs off:
  * loss on held-out QA pairs for the repo (if the repo is in the corpus)
  * greedy generations for hand-written questions, base vs adapted

Usage:
    python test_repo_qa.py --job <jobId> --checkpoint ../../runs/h200_run/head.best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "engine"))
import config  # noqa: E402
from generate_and_merge import (TARGET_MODULES, load_head,  # noqa: E402
                                _lora_modules_by_spec_name)
from memory_lora.core import DEFAULT_ROOT_PREFIX, get_module_specs  # noqa: E402

QUESTIONS = [
    "What is the core purpose of this repository?",
    "What is the main entry point or primary public API of this project?",
    "How is this project's source code organized?",
    "What testing framework and conventions does this repository use?",
    "How is this project built, packaged, or deployed?",
]


@torch.no_grad()
def gen(model, tok, prompt: str, device: str, max_new: int = 40) -> str:
    enc = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    txt = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return txt.split("\nQ:")[0].strip()


@torch.no_grad()
def qa_loss(model, tok, pairs, device: str) -> float:
    tot, ntok = 0.0, 0
    for p in pairs:
        prefix, target = p["prefix"], p["target"]
        pid = tok(prefix, add_special_tokens=False)["input_ids"]
        tid = tok(target, add_special_tokens=False)["input_ids"]
        if not tid:
            continue
        ids = torch.tensor([pid + tid], device=device)
        labels = torch.tensor([[-100] * len(pid) + tid], device=device)
        out = model(input_ids=ids, labels=labels)
        n = len(tid)
        tot += float(out.loss) * n
        ntok += n
    return tot / max(ntok, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--checkpoint", default=str(config.DEFAULT_CHECKPOINT))
    ap.add_argument("--qna-path", default="data/qna/aligned6_qna.jsonl")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-qa", type=int, default=20)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    ws = config.workspace(args.job)
    emb = np.load(ws / "embedding.npy").astype("float32")
    repo_url = json.loads((ws / "status.json").read_text()).get("repo_url", "?")
    device = config.resolve_device(args.device)

    head, cfg, alpha = load_head(Path(args.checkpoint))
    with torch.no_grad():
        head_out = head(torch.from_numpy(emb).unsqueeze(0))

    print(f"repo: {repo_url}\ncheckpoint: {args.checkpoint}\n", flush=True)
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForImageTextToText.from_pretrained(
        config.BASE_MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    specs = get_module_specs(base, TARGET_MODULES, root_prefix=DEFAULT_ROOT_PREFIX)
    type_of = {s.full_name: s.type for s in specs}
    model = get_peft_model(base, LoraConfig(
        r=cfg["rank"], lora_alpha=alpha,
        target_modules=[s.full_name for s in specs],
        lora_dropout=0.0, bias="none"))
    mods = _lora_modules_by_spec_name(model)
    with torch.no_grad():
        for sp in specs:
            m = mods.get(sp.full_name)
            if m is None:
                continue
            t = type_of[sp.full_name]
            m.lora_A["default"].weight.copy_(head_out["A"][t][0].float())
            m.lora_B["default"].weight.copy_(head_out["B"][t][0].float())
    model.to(device)
    model.eval()

    # 1. Held-out QA loss in the trained format, if this repo is in the corpus.
    doc_id = repo_url.replace("https://github.com/", "").rstrip("/")
    pairs = []
    qp = Path(args.qna_path)
    if qp.exists():
        with open(qp) as f:
            for line in f:
                d = json.loads(line)
                if d.get("doc_id") == doc_id:
                    pairs.append(d)
                    if len(pairs) >= args.max_qa:
                        break
    if pairs:
        with model.disable_adapter():
            lb = qa_loss(model, tok, pairs, device)
        la = qa_loss(model, tok, pairs, device)
        print(f"QA loss on {len(pairs)} pairs (the TRAINED format):")
        print(f"  base    {lb:.4f}")
        print(f"  adapted {la:.4f}   delta {la-lb:+.4f}  "
              f"{'HELPS' if la < lb else 'HURTS'}\n", flush=True)
    else:
        print(f"(no QA rows for doc_id '{doc_id}' — generation comparison only)\n",
              flush=True)

    # 2. Generation comparison.
    for q in QUESTIONS:
        prompt = f"Q: {q}\nA:"
        with model.disable_adapter():
            b = gen(model, tok, prompt, device)
        a = gen(model, tok, prompt, device)
        print(f"Q: {q}")
        print(f"  base:    {b[:200]}")
        print(f"  adapted: {a[:200]}\n", flush=True)


if __name__ == "__main__":
    main()
