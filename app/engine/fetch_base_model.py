#!/usr/bin/env python3
"""Pull the base Gemma model from the Hugging Face Hub into a local directory.

The base model is NOT part of the training repo — it is fetched separately
from the official ``google/gemma-4-E2B`` repository. This downloads a
self-contained snapshot (weights + tokenizer + config) into
``<repo_root>/models/<name>`` so the adapter-merge and serving steps load from
disk with no network dependency.

Why not plain ``snapshot_download``: on multi-GB shards its connection
routinely stalls part-way through — throughput drops to zero while the socket
stays open, so no timeout fires and the download hangs indefinitely. The CDN
itself is fine (raw range requests sustain ~10 MB/s). So large files are pulled
with ``curl``, which can both *detect* a stall (``--speed-limit``/
``--speed-time``) and *resume* from the partial file (``-C -``); small metadata
files still go through ``huggingface_hub``.

The model is public, so no token is normally required; set ``HF_TOKEN`` if you
hit rate limits or the repo later becomes gated.

Usage:
    python fetch_base_model.py                 # pull config.BASE_MODEL_ID
    python fetch_base_model.py --repo google/gemma-4-E2B
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import config

# Files at or above this size are pulled with curl instead of huggingface_hub.
LARGE_FILE_BYTES = 200 * 1024 * 1024


def curl_download(url: str, dest: Path, token: str = "",
                  retries: int = 50, min_bytes_per_s: int = 65536,
                  stall_seconds: int = 20) -> bool:
    """Download ``url`` -> ``dest`` with resume and stall detection.

    ``--speed-limit``/``--speed-time`` make curl abort when throughput stays
    below ``min_bytes_per_s`` for ``stall_seconds`` — turning the silent hang
    into a non-zero exit that the retry loop resumes from with ``-C -``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        cmd = [
            "curl", "-L", "--fail", "-C", "-",
            "--speed-limit", str(min_bytes_per_s),
            "--speed-time", str(stall_seconds),
            "--retry", "5", "--retry-delay", "3", "--retry-all-errors",
            "-o", str(dest), url,
        ]
        if token:
            cmd[1:1] = ["-H", f"Authorization: Bearer {token}"]
        have = dest.stat().st_size if dest.exists() else 0
        print(f"[fetch] curl attempt {attempt}/{retries} "
              f"(have {have/1e9:.2f} GB) {dest.name}", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc == 0:
            return True
        # curl exits 33 when the server can't do a ranged resume and 416 when
        # the range is already satisfied — both mean "nothing more to fetch".
        if rc == 33 or rc == 36:
            print(f"[fetch] curl rc={rc}; treating partial as complete-check",
                  flush=True)
        time.sleep(min(15, 2 * attempt))
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=config.BASE_MODEL_ID)
    ap.add_argument("--dest", default="", help="target dir (default models/<name>)")
    ap.add_argument("--token", default="", help="HF token (else env / anonymous)")
    ap.add_argument("--retries", type=int, default=50)
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_url, snapshot_download

    token = args.token or os.environ.get("HF_TOKEN", "")
    dest = Path(args.dest) if args.dest else config.local_model_dir(args.repo)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] pulling {args.repo} -> {dest}", flush=True)
    t0 = time.time()

    api = HfApi()
    info = api.model_info(args.repo, files_metadata=True, token=token or None)
    sizes = {s.rfilename: (s.size or 0) for s in info.siblings}
    skip_ext = (".gguf", ".onnx", ".tflite", ".task")

    large = [f for f, sz in sizes.items()
             if sz >= LARGE_FILE_BYTES and not f.endswith(skip_ext)]
    small = [f for f, sz in sizes.items()
             if sz < LARGE_FILE_BYTES and not f.endswith(skip_ext)]

    # 1. small metadata files via huggingface_hub (fast, no stall risk)
    if small:
        snapshot_download(
            repo_id=args.repo, local_dir=str(dest), token=token or None,
            allow_patterns=small,
        )
        print(f"[fetch] {len(small)} metadata files ok", flush=True)

    # 2. large shards via curl (resumable, stall-detecting)
    for fname in large:
        target = dest / fname
        want = sizes[fname]
        if target.exists() and target.stat().st_size == want:
            print(f"[fetch] {fname} already complete ({want/1e9:.2f} GB)", flush=True)
            continue
        url = hf_hub_url(repo_id=args.repo, filename=fname)
        ok = curl_download(url, target, token=token, retries=args.retries)
        got = target.stat().st_size if target.exists() else 0
        if not ok or got != want:
            print(f"[fetch] FAILED {fname}: got {got} of {want} bytes",
                  file=sys.stderr)
            sys.exit(1)
        print(f"[fetch] {fname} complete ({got/1e9:.2f} GB)", flush=True)

    # Drop huggingface_hub's staging dir; the snapshot is self-contained now.
    shutil.rmtree(dest / ".cache", ignore_errors=True)

    has_config = (dest / "config.json").exists()
    weights = list(dest.glob("*.safetensors")) + list(dest.glob("*.bin"))
    print(json.dumps({
        "ok": has_config and bool(weights),
        "repo": args.repo,
        "dest": str(dest),
        "config_json": has_config,
        "weight_files": len(weights),
        "total_gb": round(sum(p.stat().st_size for p in dest.rglob("*")
                              if p.is_file()) / 1e9, 2),
        "seconds": round(time.time() - t0, 1),
    }))
    if not (has_config and weights):
        print("[fetch] WARNING: snapshot missing config.json or weights",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
