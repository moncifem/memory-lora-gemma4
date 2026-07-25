#!/usr/bin/env python3
"""Show concrete held-out examples: for a few cr_test docs (both REAL repos
and synthetic), inject the hypernetwork-generated adapter and print
question -> gold vs base-model vs adapted-model prediction, so we can SEE
what the eval actually measured. CPU-only."""
from __future__ import annotations
import json, random, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
from memory_lora.core import (MemoryLoRAHead, get_module_specs, replace_with_lora,
                               inject_lora_weights, load_doc_rows, load_qna_rows, DEFAULT_ROOT_PREFIX)
from transformers import AutoModelForImageTextToText, AutoTokenizer
TM = ["q_proj","k_proj","v_proj","o_proj","up_proj","gate_proj","down_proj"]

@torch.no_grad()
def gen(model, tok, prefix, n=12):
    enc = tok(prefix, return_tensors="pt")
    out = model.generate(**enc, max_new_tokens=n, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0]

def main():
    ckpt = torch.load("runs/full_real_v4/head.best.pt", map_location="cpu")
    tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    print("loading base model on CPU...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained("google/gemma-4-E2B",
        torch_dtype=torch.float32, attn_implementation="eager", low_cpu_mem_usage=True)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    specs = get_module_specs(model, TM, root_prefix=DEFAULT_ROOT_PREFIX)
    replace_with_lora(model, specs, rank=ckpt["config"]["rank"], alpha=ckpt["args"].get("alpha",32.0))
    head = MemoryLoRAHead(input_dim=ckpt["config"]["input_dim"],
        type_dims={k:tuple(v) for k,v in ckpt["config"]["type_dims"].items()},
        hidden_dim=ckpt["config"]["hidden_dim"], rank=ckpt["config"]["rank"])
    head.load_state_dict(ckpt["state_dict"]); head.eval()

    docs = load_doc_rows("data/embeddings/combined_embeddings.parquet")
    qnas = load_qna_rows("data/qna/combined_qna.jsonl")
    by_doc = {}
    for q in qnas:
        if q.split == "cr_test": by_doc.setdefault(q.doc_id, []).append(q)
    doc_by_id = {d.doc_id: d for d in docs}

    real = [d for d in by_doc if "@" in d and d in doc_by_id]      # real repos
    synth = [d for d in by_doc if "@" not in d and d in doc_by_id] # synthetic
    rng = random.Random(1)
    picks = rng.sample(real, min(4,len(real))) + rng.sample(synth, min(2,len(synth)))
    named = dict(model.named_modules())

    for doc_id in picks:
        d = doc_by_id[doc_id]
        kind = "REAL REPO" if "@" in doc_id else "SYNTHETIC"
        ctx = torch.from_numpy(d.doc_embedding).unsqueeze(0)
        head_out = head(ctx)
        pairs = by_doc[doc_id][:2]
        print(f"\n========== [{kind}] {doc_id} ==========", flush=True)
        for q in pairs:
            for sp in specs: named[sp.full_name].A=None; named[sp.full_name].B=None
            base = gen(model, tok, q.prefix)
            inject_lora_weights(model, specs, head_out, batch_index=0)
            adapt = gen(model, tok, q.prefix)
            ok = adapt.strip().rstrip('.').lower().startswith(q.target.strip().rstrip('.').lower()) or \
                 q.target.strip().lower().startswith(adapt.strip().lower())
            print(f"  Q: {q.prefix.strip()[:90].replace(chr(10),' ')}", flush=True)
            print(f"     gold={q.target.strip()!r}  base={base.strip()!r}  adapted={adapt.strip()!r}  {'OK' if ok else 'X'}", flush=True)

if __name__ == "__main__":
    main()
