#!/usr/bin/env python3
"""Steps 2-3 of the serving pipeline: run the hypernetwork on a repo's
embedding to emit a LoRA adapter, then merge it into the frozen base model.

The trained ``MemoryLoRAHead`` outputs ONE (A, B) pair per shape-qualified
module *type* (e.g. ``q_proj_1536x2048``), shared across every transformer
layer of that shape -- exactly how it was applied during training
(``inject_lora_weights``). Its update is

    delta = (alpha / rank) * (x @ A^T) @ B^T,   A:[r,in]  B:[out,r]

which is *identically* PEFT's LoRA convention (lora_A:[r,in], lora_B:[out,r],
scaling = lora_alpha / r). So we materialize a standard PEFT adapter by
copying the shared (A, B) of each type onto every concrete Linear of that
shape, save it (usable directly by vLLM ``--enable-lora``), then
``merge_and_unload`` into the base weights and save a self-contained merged
model that vLLM can serve with no adapter plumbing at all.

Usage:
    python generate_and_merge.py \
        --embedding emb.npy --checkpoint runs/sixview_v2/head.best.pt \
        --adapter-out out/adapter --merged-out out/merged [--no-merge]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import config  # noqa: F401  (sets sys.path to the training repo root)
from memory_lora.core import (
    MemoryLoRAHead,
    DEFAULT_ROOT_PREFIX,
    get_module_specs,
)

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "up_proj", "gate_proj", "down_proj"]

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16,
           "float32": torch.float32}


def _assert_real_checkpoint(path: Path) -> None:
    """Fail with an actionable message when the file is a git-LFS pointer.

    A pointer is ~130 bytes of text starting "version https://git-lfs...".
    Handed to torch.load it surfaces as `Unsupported operand 118` or
    `invalid load key, 'v'` -- opcode 'v', the first byte of "version" -- which
    reads like a corrupt checkpoint rather than a file that was never fetched.
    Any `GIT_LFS_SKIP_SMUDGE=1` clone or pull leaves checkpoints in this state.
    """
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    if path.stat().st_size < 5000:
        head = path.read_bytes()[:64]
        if head.startswith(b"version https://git-lfs"):
            raise RuntimeError(
                f"{path} is a git-LFS pointer, not the checkpoint "
                f"({path.stat().st_size} bytes). Fetch it with:\n"
                f"    git lfs pull --include=\"{path}\""
            )


def load_head(checkpoint: Path):
    _assert_real_checkpoint(Path(checkpoint))
    # weights_only=False: these checkpoints carry the run's config/args dicts,
    # not just tensors, and torch>=2.6 defaults the strict unpickler on --
    # which rejects them ("Unsupported operand"). They are produced by this
    # project's own training script, so loading them fully is intended.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    head = MemoryLoRAHead(
        input_dim=cfg["input_dim"],
        type_dims={k: tuple(v) for k, v in cfg["type_dims"].items()},
        hidden_dim=cfg["hidden_dim"],
        rank=cfg["rank"],
    )
    # strict=False so checkpoints predating input standardization still load;
    # their identity mean/std reproduces the original (un-standardized) forward.
    head.load_state_dict(ckpt["state_dict"], strict=False)
    head.eval()
    alpha = float(ckpt["args"].get("alpha", 32.0))
    return head, cfg, alpha


def _lora_modules_by_spec_name(model) -> dict:
    """Map each original module path -> its PEFT LoRA wrapper, in one pass.

    PEFT prefixes the original names with ``base_model.model.``, so we strip
    that prefix rather than suffix-scanning per spec (which would be quadratic
    over ~200 specs x ~1000 modules)."""
    out = {}
    for name, mod in model.named_modules():
        if not hasattr(mod, "lora_A"):
            continue
        key = name
        for prefix in ("base_model.model.", "base_model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        out[key] = mod
    return out


def generate_and_merge(embedding_path: Path, checkpoint: Path,
                       adapter_out: Path, merged_out: Path | None,
                       device: str, save_dtype: torch.dtype | None = torch.bfloat16,
                       load_dtype: torch.dtype = torch.bfloat16) -> dict:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    device = config.resolve_device(device)
    t0 = time.time()

    head, cfg, alpha = load_head(checkpoint)
    emb = np.load(embedding_path).astype("float32")
    if emb.shape[-1] != cfg["input_dim"]:
        raise ValueError(
            f"embedding dim {emb.shape[-1]} != head input_dim {cfg['input_dim']}"
        )
    ctx = torch.from_numpy(emb).unsqueeze(0)
    with torch.no_grad():
        head_out = head(ctx)  # {"A": {type:[1,r,in]}, "B": {type:[1,out,r]}}

    # Load in bf16 rather than fp32: the base model is ~5B params, so fp32
    # would need ~20GB resident plus another ~10GB for the bf16 copy at save
    # time. bf16 is also precise enough here -- the generated delta has RMS
    # ~2e-3 against weights of ~2e-2, roughly 25x bf16's resolution at that
    # magnitude, so the adapter survives the merge intact.
    print(f"[generate] loading base model {config.BASE_MODEL} ({load_dtype}) ...",
          flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForImageTextToText.from_pretrained(
        config.BASE_MODEL, torch_dtype=load_dtype, low_cpu_mem_usage=True,
    )
    base.eval()

    # Discover the exact Linear modules the head was trained to adapt and use
    # their full names as PEFT targets: full-name exact matching keeps the
    # vision_tower / audio_tower (which also contain q_proj etc.) untouched.
    specs = get_module_specs(base, TARGET_MODULES, root_prefix=DEFAULT_ROOT_PREFIX)
    type_of = {sp.full_name: sp.type for sp in specs}
    target_names = [sp.full_name for sp in specs]

    lora_cfg = LoraConfig(
        r=cfg["rank"],
        lora_alpha=alpha,
        target_modules=target_names,
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)

    # Copy each type's shared (A, B) onto every concrete module of that shape.
    injected, missing = 0, []
    A_by_type, B_by_type = head_out["A"], head_out["B"]
    lora_modules = _lora_modules_by_spec_name(model)
    with torch.no_grad():
        for sp in specs:
            mod = lora_modules.get(sp.full_name)
            if mod is None:
                missing.append(sp.full_name)
                continue
            t = type_of[sp.full_name]
            A = A_by_type[t][0].to(torch.float32)  # [r, in]
            B = B_by_type[t][0].to(torch.float32)  # [out, r]
            mod.lora_A["default"].weight.copy_(A)
            mod.lora_B["default"].weight.copy_(B)
            injected += 1
    if not injected:
        raise RuntimeError("no LoRA modules were injected; check target_modules")
    print(f"[generate] injected {injected} modules "
          f"({len(missing)} missing) in {time.time()-t0:.1f}s", flush=True)

    adapter_out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_out))
    tokenizer.save_pretrained(str(adapter_out))
    print(f"[generate] saved PEFT adapter -> {adapter_out}", flush=True)

    result = {
        "ok": True,
        "injected": injected,
        "missing": len(missing),
        "rank": cfg["rank"],
        "alpha": alpha,
        "adapter_out": str(adapter_out),
        "base_model": config.BASE_MODEL,
    }

    if merged_out is not None:
        # Snapshot one target weight pre-merge so we can prove the merge applied
        # exactly the adapter's delta (and not, say, silently no-op'd).
        probe = specs[0]
        probe_mod = lora_modules[probe.full_name]
        w_before = probe_mod.base_layer.weight.detach().clone().float()
        t_probe = type_of[probe.full_name]
        expected_delta = (alpha / cfg["rank"]) * (
            B_by_type[t_probe][0].float() @ A_by_type[t_probe][0].float()
        )

        print("[merge] merging adapter into base weights ...", flush=True)
        merged = model.merge_and_unload()

        w_after = dict(merged.named_modules())[probe.full_name].weight.detach().float()
        actual_delta = w_after - w_before
        # Relative error against the delta's own magnitude: an absolute
        # tolerance would be meaningless across load dtypes (bf16 rounds the
        # stored weight itself at ~1e-4 near these magnitudes).
        rel = (actual_delta - expected_delta).norm().item() / max(
            expected_delta.norm().item(), 1e-12)
        moved = actual_delta.abs().max().item()
        result["verify"] = {
            "module": probe.full_name,
            "relative_error_vs_expected": rel,
            "max_abs_weight_change": moved,
            "expected_delta_fro": expected_delta.norm().item(),
            "ok": bool(rel < 0.05 and moved > 0),
        }
        print(f"[merge] verify {probe.full_name}: weight moved by max "
              f"{moved:.3e}; delta matches hypernetwork output to "
              f"{rel*100:.2f}% relative error", flush=True)
        merged_out.mkdir(parents=True, exist_ok=True)
        # Store in bf16 regardless of the load dtype: it halves the ~10GB
        # per-job artifact and is what the serving runtime loads anyway. The
        # verify above reports how much of the adapter survived the round trip.
        if save_dtype is not None:
            merged = merged.to(save_dtype)
        merged.save_pretrained(str(merged_out), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_out))
        result["merged_out"] = str(merged_out)
        print(f"[merge] saved merged model -> {merged_out}", flush=True)

    result["seconds"] = round(time.time() - t0, 1)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--checkpoint", default=str(config.DEFAULT_CHECKPOINT))
    ap.add_argument("--adapter-out", required=True)
    ap.add_argument("--merged-out", default="")
    ap.add_argument("--no-merge", action="store_true",
                    help="only emit the LoRA adapter; skip merging into base")
    ap.add_argument("--device", default=config.DEVICE)
    ap.add_argument("--load-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="dtype the base model is loaded/merged in")
    ap.add_argument("--save-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="dtype of the saved merged model")
    args = ap.parse_args()

    merged_out = None if (args.no_merge or not args.merged_out) else Path(args.merged_out)
    res = generate_and_merge(
        Path(args.embedding), Path(args.checkpoint),
        Path(args.adapter_out), merged_out, args.device,
        save_dtype=_DTYPES[args.save_dtype], load_dtype=_DTYPES[args.load_dtype],
    )
    print(json.dumps(res))


if __name__ == "__main__":
    main()
