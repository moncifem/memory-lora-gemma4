#!/usr/bin/env python3
"""Evaluate a trained Memory-LoRA head: recall EM/EditSim on held-out QA,
split into in-corpus (ir_test) and cross-corpus (cr_val/cr_test), plus a
manual spot-check comparing the LoRA-adapted model against the bare base
model on hand-picked Code2LoRA-paper questions (proof the adapter, not
general pretraining, is doing the recall).

Forked from Code2LoRA's evaluation metrics (EM after whitespace collapsing
+ trailing-punctuation removal with relaxed prefix matching; EditSim via
difflib.SequenceMatcher).

Usage:
    python scripts/eval_memory_lora.py --checkpoint runs/full1/head.best.pt
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
from memory_lora.data_paths import EMBEDDINGS_DIR, QNA_DIR  # noqa: E402
from memory_lora.core import (  # noqa: E402
    MemoryLoRAHead,
    DEFAULT_ROOT_PREFIX,
    get_module_specs,
    inject_lora_weights,
    load_doc_rows,
    load_qna_rows,
    replace_with_lora,
)

DEFAULT_MODEL = "google/gemma-4-E2B"
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "gate_proj", "down_proj",
]

SPOT_CHECK_QUESTIONS = [
    ("What LoRA rank does Code2LoRA's static hypernetwork use?", "16"),
    ("How many trainable parameters does Code2LoRA-Static have?", "720 million"),
    ("What is the cross-repo exact match of Code2LoRA-Static on the static track?", "63.8%"),
    ("How many Python repositories are in RepoPeftBench?", "604"),
    ("What is the base LLM used in Code2LoRA's experiments?", "Qwen2.5-Coder-1.5B"),
]


def normalize(s: str) -> str:
    s = s.strip().rstrip(".,;:!?")
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def exact_match(pred: str, target: str) -> bool:
    p, t = normalize(pred), normalize(target)
    return p == t or p.startswith(t) or t.startswith(p)


def edit_sim(pred: str, target: str) -> float:
    return difflib.SequenceMatcher(None, normalize(pred), normalize(target)).ratio()


@torch.no_grad()
def generate(base_model, tokenizer, prefix: str, device, max_new_tokens: int = 12) -> str:
    """Greedy-decode the answer, then truncate at the first newline.

    Without this the model frequently keeps going past the answer into a
    hallucinated ``\\nQ: <next question>`` continuation (base models without
    an EOS-triggering chat template rarely stop cleanly on a bare
    completion prompt); comparing the *untruncated* string against the gold
    answer would mark an otherwise-correct short answer wrong just because
    of what it rambled into afterward.
    """
    enc = tokenizer(prefix, return_tensors="pt").to(device)
    out = base_model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    gen_ids = out[0][enc["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.split("\n")[0]


def load_head_and_model(checkpoint: Path, model_name: str, target_modules: List[str],
                         root_prefix: str, device: torch.device, dtype: torch.dtype,
                         attn_implementation: str):
    # weights_only=False: these checkpoints carry the run's config/args dicts,
    # not just tensors, and torch>=2.6 defaults the strict unpickler on --
    # which rejects them ("Unsupported operand"). They are produced by this
    # project's own training script, so loading them fully is intended.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=dtype, attn_implementation=attn_implementation,
    ).to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    specs = get_module_specs(base_model, target_modules, root_prefix=root_prefix)
    rank = ckpt["config"]["rank"]
    alpha = ckpt["args"].get("alpha", 32.0)
    replace_with_lora(base_model, specs, rank=rank, alpha=alpha)

    head = MemoryLoRAHead(
        input_dim=ckpt["config"]["input_dim"],
        type_dims={k: tuple(v) for k, v in ckpt["config"]["type_dims"].items()},
        hidden_dim=ckpt["config"]["hidden_dim"],
        rank=rank,
    ).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    return base_model, head, specs, tokenizer


def eval_suite(base_model, head, specs, tokenizer, doc_rows, qnas_by_doc, device,
                max_qna_per_doc: int = 20) -> Dict[str, float]:
    n_em, n_total, sum_editsim = 0, 0, 0.0
    for dr in doc_rows:
        pairs = qnas_by_doc.get(dr.doc_id, [])[:max_qna_per_doc]
        if not pairs:
            continue
        ctx = torch.from_numpy(dr.doc_embedding).to(device).unsqueeze(0)
        head_out = head(ctx)
        inject_lora_weights(base_model, specs, head_out, batch_index=0)
        for p in pairs:
            pred = generate(base_model, tokenizer, p["prefix"], device)
            target = p["target"]
            if exact_match(pred, target):
                n_em += 1
            sum_editsim += edit_sim(pred, target)
            n_total += 1
    return {
        "em": n_em / max(n_total, 1),
        "editsim": sum_editsim / max(n_total, 1),
        "n": n_total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--embeddings-path", default=str(EMBEDDINGS_DIR / "doc_embeddings.parquet"))
    ap.add_argument("--qna-path", default=str(QNA_DIR / "qna.jsonl"))
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    ap.add_argument("--root-prefix", default=DEFAULT_ROOT_PREFIX)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--suites", nargs="+", default=["cr_val", "cr_test", "ir_test"])
    ap.add_argument("--max-qna-per-doc", type=int, default=20)
    ap.add_argument("--limit-docs", type=int, default=0,
                     help="Random-sample at most N docs per suite (fixed seed) "
                          "for a fast estimate -- cr_test has thousands of real "
                          "repos, far too many to greedy-generate on CPU.")
    ap.add_argument("--skip-spot-check", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    base_model, head, specs, tokenizer = load_head_and_model(
        Path(args.checkpoint), args.model_name, args.target_modules,
        args.root_prefix, device, dtype, args.attn_implementation,
    )

    all_docs = load_doc_rows(Path(args.embeddings_path))
    all_qnas = load_qna_rows(Path(args.qna_path))
    qnas_by_doc_all = {}
    qnas_held_out_by_doc = {}
    for q in all_qnas:
        qnas_by_doc_all.setdefault(q.doc_id, []).append({"prefix": q.prefix, "target": q.target})
        if q.qna_split == "held_out":
            qnas_held_out_by_doc.setdefault(q.doc_id, []).append({"prefix": q.prefix, "target": q.target})
    train_docs = [d for d in all_docs if d.split == "train"]

    results: Dict[str, Any] = {}
    for suite in args.suites:
        if suite in ("cr_val", "cr_test"):
            rows, q_by_doc = [d for d in all_docs if d.split == suite], qnas_by_doc_all
        elif suite == "ir_test":
            rows, q_by_doc = train_docs, qnas_held_out_by_doc
        else:
            continue
        if args.limit_docs and len(rows) > args.limit_docs:
            import random as _r
            rows = _r.Random(3407).sample(rows, args.limit_docs)
        print(f"Evaluating {suite} ({len(rows)} docs) ...", flush=True)
        m = eval_suite(base_model, head, specs, tokenizer, rows, q_by_doc, device,
                        max_qna_per_doc=args.max_qna_per_doc)
        results[suite] = m
        print(f"  {suite}: EM={m['em']:.3f} EditSim={m['editsim']:.3f} n={m['n']}", flush=True)

    print(json.dumps(results, indent=2))

    if not args.skip_spot_check:
        print("\n=== Spot check: base model vs. LoRA-adapted, Code2LoRA paper facts ===", flush=True)
        paper_doc = next((d for d in all_docs if d.doc_id == "code2lora_paper"), None)
        if paper_doc is None:
            print("  [skip] code2lora_paper doc not found in embeddings", flush=True)
        else:
            ctx = torch.from_numpy(paper_doc.doc_embedding).to(device).unsqueeze(0)
            head_out = head(ctx)
            for q, gold in SPOT_CHECK_QUESTIONS:
                prefix = f"Q: {q}\nA:"
                # base: zero out LoRA (A=B=None) by re-wrapping without injection
                for sp in specs:
                    named = dict(base_model.named_modules())
                    named[sp.full_name].A = None
                    named[sp.full_name].B = None
                base_pred = generate(base_model, tokenizer, prefix, device)
                inject_lora_weights(base_model, specs, head_out, batch_index=0)
                adapted_pred = generate(base_model, tokenizer, prefix, device)
                print(f"Q: {q}")
                print(f"  gold:    {gold}")
                print(f"  base:    {base_pred!r}")
                print(f"  adapted: {adapted_pred!r}")
                print()


if __name__ == "__main__":
    main()
