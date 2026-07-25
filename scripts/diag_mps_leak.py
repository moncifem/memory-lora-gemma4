#!/usr/bin/env python3
"""Isolated MPS memory-leak diagnostic.

Runs a minimal train-step loop on REAL code prefixes and prints
system-available memory after every single operation, aborting the moment
it crosses a hard floor. Tests one variable at a time (gradient
checkpointing on/off, empty_cache on/off) so we can pinpoint what actually
leaks, without the full training harness in the way.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import psutil, torch, torch.nn.functional as F
HERE = Path(__file__).resolve().parent; REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
from memory_lora.core import (MemoryLoRAHead, get_module_specs, discover_module_types_and_dims,
                               inject_lora_weights, replace_with_lora, DEFAULT_ROOT_PREFIX)
from transformers import AutoModelForImageTextToText, AutoTokenizer

TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","up_proj","gate_proj","down_proj"]

def avail_gb(): return psutil.virtual_memory().available / 1e9

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--empty-cache-every", type=int, default=0, help="0=never")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--floor-gb", type=float, default=22.0)
    ap.add_argument("--train", action="store_true", help="do backward (else forward-only)")
    args = ap.parse_args()
    device = torch.device("mps")

    print(f"[cfg] grad_ckpt={args.grad_checkpoint} empty_cache_every={args.empty_cache_every} "
          f"train={args.train} seq={args.seq_len} mb={args.micro_batch}", flush=True)
    print(f"[mem] start avail={avail_gb():.1f}GB", flush=True)

    tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        "google/gemma-4-E2B", torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    print(f"[mem] after model load avail={avail_gb():.1f}GB", flush=True)
    if args.grad_checkpoint:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("  gradient checkpointing ON", flush=True)

    specs = get_module_specs(model, TARGET_MODULES, root_prefix=DEFAULT_ROOT_PREFIX)
    type_dims = discover_module_types_and_dims(specs)
    replace_with_lora(model, specs, rank=16, alpha=32.0)
    head = MemoryLoRAHead(input_dim=2048, type_dims=type_dims, hidden_dim=512, rank=16).to(device)
    print(f"[mem] after head+lora avail={avail_gb():.1f}GB", flush=True)

    # real prefixes
    prefixes, targets = [], []
    with open(REPO_ROOT / "data/qna/real_code2lora_qna.jsonl") as f:
        for line in f:
            d = json.loads(line); prefixes.append(d["prefix"]); targets.append(d["target"])
            if len(prefixes) >= args.steps * args.micro_batch: break
    ctx = torch.randn(1, 2048, device=device)

    for step in range(args.steps):
        if avail_gb() < args.floor_gb:
            print(f"[ABORT] avail={avail_gb():.1f}GB < floor {args.floor_gb} at step {step}", flush=True)
            break
        i0 = step * args.micro_batch
        ps = prefixes[i0:i0+args.micro_batch]; ts = targets[i0:i0+args.micro_batch]
        ids_list, lab_list = [], []
        for p, t in zip(ps, ts):
            tids = tok(t + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
            pids = tok(p, add_special_tokens=False)["input_ids"]
            budget = max(8, args.seq_len - len(tids)); pids = pids[-budget:]
            ids = pids + tids; lab = [-100]*len(pids) + list(tids)
            ids_list.append(torch.tensor(ids)); lab_list.append(torch.tensor(lab))
        L = args.seq_len
        pad = tok.pad_token_id or 0
        def lp(x, v): return F.pad(x, (L-x.size(0),0), value=v) if x.size(0)<=L else x[-L:]
        input_ids = torch.stack([lp(t,pad) for t in ids_list]).to(device)
        labels = torch.stack([lp(t,-100) for t in lab_list]).to(device)
        attn = (input_ids != pad).long().to(device)
        head_out = head(ctx); inject_lora_weights(model, specs, head_out, batch_index=0)
        if args.train:
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            (out.loss).backward()
            head.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss_val = out.loss.item()
        del head_out, out
        if args.empty_cache_every and step % args.empty_cache_every == 0:
            torch.mps.empty_cache()
        print(f"[step {step}] loss={loss_val:.3f} avail={avail_gb():.1f}GB", flush=True)
    print(f"[done] final avail={avail_gb():.1f}GB", flush=True)

if __name__ == "__main__":
    main()
