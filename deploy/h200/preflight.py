#!/usr/bin/env python3
"""Pre-flight validation for a paid single-GPU training run.

Runs BEFORE the real training and fails loudly on anything that would waste
GPU-hours. Every check here corresponds to something that has actually gone
wrong in this project.

    python3 deploy/h200/preflight.py            # validate + auto-tune batch size
    python3 deploy/h200/preflight.py --quick    # skip the timed throughput probe

Checks, in order (cheapest first, so it fails fast):
  1. GPU present, expected VRAM, bf16 support
  2. Data files materialized (git-LFS pointers are ~130 bytes, not data)
  3. Embedding/QA integrity: dims, splits, NaNs, doc_id join coverage
  4. Embeddings are NOT already standardized (would double-apply)
  5. Base model loads, and the LoRA target modules are discoverable
  6. Head builds, standardization measurably de-correlates the input
  7. Largest micro-batch that fits in VRAM (auto-tune)
  8. Measured throughput -> steps achievable in the training budget
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

FAIL: list[str] = []
WARN: list[str] = []


def ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m {msg}", flush=True)


def bad(msg: str) -> None:
    FAIL.append(msg)
    print(f"  \033[31m✗ {msg}\033[0m", flush=True)


def warn(msg: str) -> None:
    WARN.append(msg)
    print(f"  \033[33m! {msg}\033[0m", flush=True)


def check_gpu(min_vram_gb: float) -> dict:
    print("\n[1] GPU")
    if not torch.cuda.is_available():
        bad("CUDA not available — this must run on the GPU instance")
        return {}
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    ok(f"{name}, {vram:.0f} GB VRAM, torch {torch.__version__}, CUDA {torch.version.cuda}")
    if vram < min_vram_gb:
        warn(f"VRAM {vram:.0f}GB < expected {min_vram_gb}GB — lower --lm-micro-batch")
    if not torch.cuda.is_bf16_supported():
        bad("bf16 unsupported — training config assumes bf16")
    else:
        ok("bf16 supported")
    if torch.cuda.device_count() > 1:
        warn(f"{torch.cuda.device_count()} GPUs visible; this script trains on ONE")
    return {"name": name, "vram_gb": vram}


def check_data(emb_path: Path, qna_path: Path) -> dict:
    print("\n[2] Data files")
    info: dict = {}
    for p in (emb_path, qna_path):
        if not p.exists():
            bad(f"missing: {p}")
            return info
        sz = p.stat().st_size
        if sz < 5000:
            bad(f"{p.name} is {sz}B — this is a git-LFS POINTER, not data. "
                f"Run: git lfs pull")
            return info
        ok(f"{p.name}: {sz/1e6:.1f} MB")

    print("\n[3] Data integrity")
    import pyarrow.parquet as pq
    t = pq.read_table(emb_path)
    dim = len(t.column("doc_embedding")[0].as_py())
    splits: dict[str, int] = {}
    for s in t.column("split").to_pylist():
        splits[s] = splits.get(s, 0) + 1
    ok(f"embeddings: {t.num_rows} rows, dim {dim}, splits {splits}")
    info["dim"] = dim
    info["n_docs"] = t.num_rows
    info["splits"] = splits

    if dim != 12288:
        bad(f"embedding dim {dim} != 12288 (6 views x 2048)")
    for need in ("train", "cr_val", "cr_test"):
        if splits.get(need, 0) == 0:
            bad(f"split '{need}' is empty — eval/gating needs it")

    E = np.array(t.column("doc_embedding").to_pylist(), dtype="float32")
    if not np.isfinite(E).all():
        bad(f"{(~np.isfinite(E)).sum()} non-finite values in embeddings")
    else:
        ok("embeddings all finite")
    zero_rows = int((np.linalg.norm(E, axis=1) == 0).sum())
    if zero_rows:
        bad(f"{zero_rows} all-zero embedding rows")

    # Guard against feeding already-standardized embeddings (double-applying the
    # transform would silently destroy the conditioning signal).
    mu, sd = E.mean(), E.std()
    if abs(mu) < 0.05 and abs(sd - 1.0) < 0.15:
        bad(f"embeddings look ALREADY standardized (mean {mu:.3f}, std {sd:.3f}) "
            f"— the head standardizes internally; do not pre-normalize")
    else:
        ok(f"embeddings raw as expected (mean {mu:.3f}, std {sd:.3f})")

    # The DC component this project's failure was traced to.
    shared = (np.linalg.norm(E.mean(0)) ** 2) / (np.linalg.norm(E, axis=1) ** 2).mean()
    k = min(300, len(E))
    Ek = E[:k].astype(np.float64)
    En = Ek / np.linalg.norm(Ek, axis=1, keepdims=True)
    C = En @ En.T
    iu = np.triu_indices(k, 1)
    ok(f"shared DC energy {100*shared:.0f}%, raw pairwise cosine {C[iu].mean():.3f} "
       f"(standardization is what removes this)")

    print("\n[4] QA join coverage")
    doc_ids = set(t.column("doc_id").to_pylist())
    split_of = dict(zip(t.column("doc_id").to_pylist(), t.column("split").to_pylist()))
    n_qa = 0
    per_split: dict[str, int] = {}
    orphan = 0
    tlens = []
    with open(qna_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            n_qa += 1
            did = d.get("doc_id")
            if did not in doc_ids:
                orphan += 1
            else:
                s = split_of[did]
                per_split[s] = per_split.get(s, 0) + 1
            if len(tlens) < 5000:
                tlens.append(len(d.get("target", "")))
    ok(f"QA rows: {n_qa}, per split {per_split}, mean target {np.mean(tlens):.0f} chars")
    info["n_qa"] = n_qa
    if orphan:
        warn(f"{orphan} QA rows reference doc_ids absent from embeddings (ignored at train time)")
    train_docs_with_qa = per_split.get("train", 0)
    if train_docs_with_qa == 0:
        bad("no QA rows joined to train docs — training would have zero examples")
    for need in ("cr_val", "cr_test"):
        if per_split.get(need, 0) == 0:
            bad(f"no QA rows for '{need}' — the baseline gate cannot be computed")
    return info


def check_model_and_head(model_name: str, emb_path: Path, dim: int,
                         hidden: int, rank: int, alpha: float, device: str):
    print("\n[5] Base model + LoRA targets")
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from memory_lora.core import (DEFAULT_ROOT_PREFIX, MemoryLoRAHead,
                                  discover_module_types_and_dims,
                                  get_module_specs, replace_with_lora)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    ok(f"base model loaded in {time.time()-t0:.0f}s")
    specs = get_module_specs(model, ["q_proj", "k_proj", "v_proj", "o_proj",
                                     "up_proj", "gate_proj", "down_proj"],
                             root_prefix=DEFAULT_ROOT_PREFIX)
    if not specs:
        bad("no LoRA target modules discovered — root_prefix/model mismatch")
        return None, None, None, None
    type_dims = discover_module_types_and_dims(specs)
    ok(f"{len(specs)} target modules, {len(type_dims)} shape-qualified types")

    print("\n[6] Head + standardization")
    import pyarrow.parquet as pq
    t = pq.read_table(emb_path)
    E = np.array(t.column("doc_embedding").to_pylist(), dtype="float32")
    tr = [i for i, s in enumerate(t.column("split").to_pylist()) if s == "train"]
    head = MemoryLoRAHead(input_dim=dim, type_dims=type_dims,
                          hidden_dim=hidden, rank=rank)
    if not hasattr(head, "fit_input_stats"):
        bad("MemoryLoRAHead has no fit_input_stats — you are on OLD code that "
            "collapses to one adapter for every repo. git pull.")
        return None, None, None, None
    head.fit_input_stats(torch.from_numpy(E[tr]))
    n_params = sum(p.numel() for p in head.parameters())
    ok(f"head built: {n_params/1e6:.0f}M params (hidden_dim={hidden}, rank={rank})")

    x = torch.from_numpy(E[:24])
    xs = (x - head.input_mean) / head.input_std

    def cosmean(X):
        Xn = F.normalize(X, dim=1)
        C = Xn @ Xn.T
        iu = torch.triu_indices(len(X), len(X), 1)
        return float(C[iu[0], iu[1]].mean())
    raw_c, std_c = cosmean(x), cosmean(xs)
    if std_c < raw_c - 0.2:
        ok(f"standardization de-correlates input: cosine {raw_c:.3f} -> {std_c:.3f}")
    else:
        bad(f"standardization ineffective: cosine {raw_c:.3f} -> {std_c:.3f}")

    replace_with_lora(model, specs, rank=rank, alpha=alpha)
    model.to(device)
    head.to(device)
    return model, head, specs, tok


def autotune_batch(model, head, specs, tok, emb_path, device, seq_len,
                   candidates=(64, 48, 32, 24, 16, 8, 4)) -> int:
    print("\n[7] Auto-tune micro-batch (largest that fits)")
    from memory_lora.core import inject_lora_weights
    import pyarrow.parquet as pq
    t = pq.read_table(emb_path)
    e = np.array(t.column("doc_embedding")[0].as_py(), dtype="float32")
    ctx = torch.from_numpy(e).to(device).unsqueeze(0)
    best = 0
    for b in candidates:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            head_out = head(ctx)
            inject_lora_weights(model, specs, head_out, batch_index=0)
            ids = torch.randint(0, 1000, (b, seq_len), device=device)
            labels = ids.clone()
            out = model(input_ids=ids, labels=labels,
                        attention_mask=torch.ones_like(ids))
            out.loss.backward()
            peak = torch.cuda.max_memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            ok(f"micro-batch {b}: fits, peak {peak:.0f} GB / {total:.0f} GB")
            best = b
            head.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            break
        except torch.cuda.OutOfMemoryError:
            print(f"    micro-batch {b}: OOM", flush=True)
            head.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            warn(f"micro-batch {b}: {type(e).__name__}: {e}")
            break
    if not best:
        bad("no micro-batch size fit in VRAM")
    return best


def throughput(model, head, specs, emb_path, device, batch, seq_len,
               budget_hours: float, n_iters: int = 6) -> None:
    print("\n[8] Throughput -> what fits in the budget")
    from memory_lora.core import inject_lora_weights
    import pyarrow.parquet as pq
    t = pq.read_table(emb_path)
    e = np.array(t.column("doc_embedding")[0].as_py(), dtype="float32")
    ctx = torch.from_numpy(e).to(device).unsqueeze(0)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-5)
    ids = torch.randint(0, 1000, (batch, seq_len), device=device)
    attn = torch.ones_like(ids)
    for i in range(n_iters):
        if i == 2:  # warmup done
            torch.cuda.synchronize()
            t0 = time.time()
        head_out = head(ctx)
        inject_lora_weights(model, specs, head_out, batch_index=0)
        loss = model(input_ids=ids, labels=ids, attention_mask=attn).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    per_step = (time.time() - t0) / (n_iters - 2)
    steps = int(budget_hours * 3600 / per_step)
    ok(f"{per_step:.2f}s/step at micro-batch {batch} "
       f"-> ~{steps} steps in {budget_hours}h")
    if steps < 2000:
        warn(f"only ~{steps} steps in budget — consider fewer QA per doc or "
             f"a shorter seq_len")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="data/embeddings/aligned6_embeddings.parquet")
    ap.add_argument("--qna", default="data/qna/aligned6_qna.jsonl")
    ap.add_argument("--model-name", default="models/gemma-4-E2B")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--min-vram-gb", type=float, default=130.0)
    ap.add_argument("--budget-hours", type=float, default=1.5)
    ap.add_argument("--quick", action="store_true",
                    help="skip the timed throughput probe")
    ap.add_argument("--data-only", action="store_true",
                    help="validate data/config without a GPU (run on your laptop "
                         "before provisioning, so bad data is caught for free)")
    args = ap.parse_args()

    import os
    os.chdir(REPO_ROOT)
    print("=" * 62)
    print("  PRE-FLIGHT — validating before spending GPU time")
    print("=" * 62)

    gpu = {} if args.data_only else check_gpu(args.min_vram_gb)
    if args.data_only:
        print("\n[1] GPU  — skipped (--data-only)")
    info = check_data(Path(args.embeddings), Path(args.qna))
    if FAIL:
        finish(None)

    if args.data_only:
        print("\n[5-8] model/GPU checks — skipped (--data-only)")
        finish(None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, head, specs, tok = check_model_and_head(
        args.model_name, Path(args.embeddings), info["dim"],
        args.hidden, args.rank, args.alpha, device)
    if FAIL or model is None:
        finish(None)

    batch = 0
    if not args.quick and device == "cuda":
        batch = autotune_batch(model, head, specs, tok,
                               Path(args.embeddings), device, args.seq_len)
        if batch:
            throughput(model, head, specs, Path(args.embeddings), device,
                       batch, args.seq_len, args.budget_hours)
    finish(batch)


def finish(batch: int | None) -> None:
    print("\n" + "=" * 62)
    if FAIL:
        print(f"  \033[31mFAILED — {len(FAIL)} blocking problem(s). Do NOT start training.\033[0m")
        for f in FAIL:
            print(f"    - {f}")
        sys.exit(1)
    if WARN:
        print(f"  {len(WARN)} warning(s):")
        for w in WARN:
            print(f"    - {w}")
    print("  \033[32mALL CHECKS PASSED\033[0m")
    if batch:
        print(f"\n  Recommended:  MICRO_BATCH={batch} bash deploy/h200/train_h200.sh")
    print("=" * 62)
    sys.exit(0)


if __name__ == "__main__":
    main()
