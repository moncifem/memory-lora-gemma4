#!/usr/bin/env python3
"""Embed every document in data/docs/documents.jsonl with the frozen
Qwen3-Embedding-0.6B encoder (memory_lora/encoder.py) and write
data/embeddings/doc_embeddings.parquet.

Mirrors Code2LoRA's ``create_dataset/build_repo_state_embeddings_shard.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
from memory_lora.data_paths import DOCS_DIR, EMBEDDINGS_DIR, ensure_dirs  # noqa: E402
from memory_lora.encoder import DEFAULT_EMBED_MODEL, embed_document, load_encoder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--chunk-tokens", type=int, default=4096)
    ap.add_argument("--chunk-overlap", type=int, default=512)
    args = ap.parse_args()

    ensure_dirs()
    docs_path = DOCS_DIR / "documents.jsonl"
    out_path = EMBEDDINGS_DIR / "doc_embeddings.parquet"

    device = args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    print(f"Loading encoder {args.embed_model} on {device} ...", flush=True)
    model, tokenizer = load_encoder(args.embed_model, device=device)

    docs = [json.loads(l) for l in docs_path.open()]
    print(f"{len(docs)} documents to embed", flush=True)

    rows = []
    for d in tqdm(docs):
        sections = [(s["name"], s["text"]) for s in d["sections"]]
        vec = embed_document(
            sections, model, tokenizer, device,
            chunk_tokens=args.chunk_tokens, chunk_overlap=args.chunk_overlap,
        )
        if vec is None:
            print(f"  [warn] no embedding for {d['doc_id']}, skipping", flush=True)
            continue
        rows.append({
            "doc_id": d["doc_id"],
            "doc_version": d["doc_version"],
            "split": d["split"],
            "category": d["category"],
            "doc_embedding": vec.numpy().astype("float32").tolist(),
        })

    table = pa.table({
        "doc_id": [r["doc_id"] for r in rows],
        "doc_version": [r["doc_version"] for r in rows],
        "split": [r["split"] for r in rows],
        "category": [r["category"] for r in rows],
        "doc_embedding": [r["doc_embedding"] for r in rows],
    })
    pq.write_table(table, out_path)
    dim = len(rows[0]["doc_embedding"]) if rows else 0
    print(f"Wrote {len(rows)} embeddings (dim={dim}) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
