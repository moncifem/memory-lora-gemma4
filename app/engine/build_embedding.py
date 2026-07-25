#!/usr/bin/env python3
"""Step 1 of the serving pipeline: turn a cloned repo on disk into the
12288-d 6-view embedding the trained hypernetwork expects.

Reuses the EXACT same view extraction + frozen Qwen encoder used to build the
training set (``scripts/build_repo_multiview.py``), so the embedding an unseen
repo gets here is distributed identically to what the head was trained on.

Usage:
    python build_embedding.py --repo /path/to/clone --out /path/to/emb.npy
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import config  # noqa: F401  (sets sys.path to the training repo root)
from memory_lora.encoder import load_encoder, embed_document
# Reuse the training-time view extractor verbatim.
from scripts.build_repo_multiview import extract_views, summarize_views_text, VIEWS


def build_embedding(repo: Path, device: str) -> tuple[np.ndarray, dict]:
    device = config.resolve_device(device)
    enc_model, enc_tok = load_encoder(device=device)

    views = extract_views(repo)
    vecs = []
    for vk in VIEWS:
        secs = views[vk] or [("empty", "none")]
        vv = embed_document(
            secs, enc_model, enc_tok, device,
            chunk_tokens=2048, chunk_overlap=128, batch_size=2,
        )
        vecs.append(
            vv.numpy().astype("float32") if vv is not None
            else np.zeros(2048, "float32")
        )
    full = np.concatenate(vecs)  # 6 * 2048 = 12288
    return full, summarize_views_text(views)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to the cloned repo")
    ap.add_argument("--out", required=True, help="output .npy for the 12288-d vector")
    ap.add_argument("--out-views", default="", help="optional .json of per-view summaries")
    ap.add_argument("--device", default=config.DEVICE)
    args = ap.parse_args()

    t0 = time.time()
    full, view_text = build_embedding(Path(args.repo), args.device)
    np.save(args.out, full)
    if args.out_views:
        Path(args.out_views).write_text(json.dumps(view_text, indent=2))
    print(json.dumps({
        "ok": True,
        "dim": int(full.shape[0]),
        "seconds": round(time.time() - t0, 1),
        "out": args.out,
    }))


if __name__ == "__main__":
    main()
