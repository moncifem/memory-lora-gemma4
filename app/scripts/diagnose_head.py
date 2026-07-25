#!/usr/bin/env python3
"""Diagnose WHY a generated adapter helps or hurts, in one model load.

Compares, on the same repo text and the same in-memory model:

  none          - adapter disabled (baseline)
  <checkpoint>  - adapter emitted by each hypernetwork checkpoint given
  random        - random A/B matched to the trained adapter's scale
  zero-B        - B set to zero (delta is exactly 0; sanity check that the
                  injection path is wired correctly -- must equal `none`)

Why the controls matter: if a trained adapter hurts, that is either (a) the
adapter is applied incorrectly, or (b) the hypernetwork is undertrained and
emitting noise. `zero-B` isolates (a): if it does not exactly reproduce the
baseline, the injection path is broken. `random` calibrates (b): a trained head
that is no better than random noise of the same magnitude has not learned the
repo->adapter mapping yet.

Usage:
    python diagnose_head.py --job <jobId> \
        --checkpoints ../runs/sixview_v2/head.best.pt ../runs/sixview_v1/head.best.pt
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
from ab_test_adapter import repo_texts, loss_on  # noqa: E402


def set_adapter(lora_modules, specs, type_of, A_by_type, B_by_type):
    with torch.no_grad():
        for sp in specs:
            mod = lora_modules.get(sp.full_name)
            if mod is None:
                continue
            t = type_of[sp.full_name]
            mod.lora_A["default"].weight.copy_(A_by_type[t].float())
            mod.lora_B["default"].weight.copy_(B_by_type[t].float())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--checkpoints", nargs="+",
                    default=[str(config.DEFAULT_CHECKPOINT)])
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    ws = config.workspace(args.job)
    emb = np.load(ws / "embedding.npy").astype("float32")
    texts = repo_texts(ws, args.max_chars)
    device = config.resolve_device(args.device)

    # Head outputs per checkpoint (cheap: head is small, model isn't loaded yet)
    conditions: dict[str, tuple[dict, dict]] = {}
    rank = alpha = None
    for ck in args.checkpoints:
        head, cfg, a = load_head(Path(ck))
        with torch.no_grad():
            out = head(torch.from_numpy(emb).unsqueeze(0))
        name = Path(ck).parent.name + "/" + Path(ck).stem
        conditions[name] = ({k: v[0] for k, v in out["A"].items()},
                            {k: v[0] for k, v in out["B"].items()})
        rank, alpha = cfg["rank"], a

    # Controls derived from the first checkpoint's scale
    first = next(iter(conditions.values()))
    rnd_A, rnd_B, zero_A, zero_B = {}, {}, {}, {}
    g = torch.Generator().manual_seed(3407)
    for t, A in first[0].items():
        B = first[1][t]
        rnd_A[t] = torch.randn(A.shape, generator=g) * A.std()
        rnd_B[t] = torch.randn(B.shape, generator=g) * B.std()
        zero_A[t] = A.clone()
        zero_B[t] = torch.zeros_like(B)
    conditions["random(matched scale)"] = (rnd_A, rnd_B)
    conditions["zero-B (must == none)"] = (zero_A, zero_B)

    print(f"loading {config.BASE_MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForImageTextToText.from_pretrained(
        config.BASE_MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    specs = get_module_specs(base, TARGET_MODULES, root_prefix=DEFAULT_ROOT_PREFIX)
    type_of = {sp.full_name: sp.type for sp in specs}
    model = get_peft_model(base, LoraConfig(
        r=rank, lora_alpha=alpha, target_modules=[s.full_name for s in specs],
        lora_dropout=0.0, bias="none"))
    lora_modules = _lora_modules_by_spec_name(model)
    model.to(device)
    model.eval()
    print("ready.\n", flush=True)

    def mean_loss(disabled: bool) -> float:
        tot = 0.0
        for _, text in texts:
            if disabled:
                with model.disable_adapter():
                    tot += loss_on(model, tok, text, device)
            else:
                tot += loss_on(model, tok, text, device)
        return tot / len(texts)

    baseline = mean_loss(disabled=True)
    print(f"{'condition':<28} {'mean loss':>10} {'vs none':>10}")
    print("-" * 50)
    print(f"{'none (base model)':<28} {baseline:10.4f} {0.0:+10.4f}")
    for name, (A, B) in conditions.items():
        set_adapter(lora_modules, specs, type_of, A, B)
        m = mean_loss(disabled=False)
        print(f"{name[:28]:<28} {m:10.4f} {m - baseline:+10.4f}")
    print("\nlower is better; `zero-B` must match `none` exactly for the "
          "injection path to be considered correct.")


if __name__ == "__main__":
    main()
