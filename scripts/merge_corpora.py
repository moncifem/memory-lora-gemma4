#!/usr/bin/env python3
"""Merge the real Code2LoRA/RepoPeftBench corpus (73,849 real repo-commit
docs, 443,798 real assertion-completion QnAs) with our synthetic corpus
(211 docs: the Code2LoRA paper + coding-agent-harness + agile/Jira +
general fact-sheets) into one combined training set for the hypernetwork.

Output:
    data/embeddings/combined_embeddings.parquet
    data/qna/combined_qna.jsonl

Usage:
    python scripts/merge_corpora.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
from memory_lora.data_paths import EMBEDDINGS_DIR, QNA_DIR, ensure_dirs  # noqa: E402


def merge_embeddings() -> int:
    synthetic_path = EMBEDDINGS_DIR / "doc_embeddings.parquet"
    real_path = EMBEDDINGS_DIR / "real_code2lora_embeddings.parquet"
    out_path = EMBEDDINGS_DIR / "combined_embeddings.parquet"

    tables = []
    for p, label in [(synthetic_path, "synthetic"), (real_path, "real")]:
        if not p.exists():
            print(f"  [skip] {p} not found", flush=True)
            continue
        t = pq.read_table(p)
        print(f"  {label}: {t.num_rows} rows, columns={t.column_names}", flush=True)
        tables.append(t.select(["doc_id", "doc_version", "split", "category", "doc_embedding"]))

    combined = pa.concat_tables(tables)
    pq.write_table(combined, out_path)
    print(f"Wrote {combined.num_rows} combined embeddings -> {out_path}", flush=True)
    return combined.num_rows


def merge_qna() -> int:
    synthetic_path = QNA_DIR / "qna.jsonl"
    real_path = QNA_DIR / "real_code2lora_qna.jsonl"
    out_path = QNA_DIR / "combined_qna.jsonl"

    n = 0
    with out_path.open("w") as out:
        for p, label in [(synthetic_path, "synthetic"), (real_path, "real")]:
            if not p.exists():
                print(f"  [skip] {p} not found", flush=True)
                continue
            count = 0
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    out.write(line + "\n")
                    count += 1
                    n += 1
            print(f"  {label}: {count} QnA rows", flush=True)
    print(f"Wrote {n} combined QnA pairs -> {out_path}", flush=True)
    return n


def main() -> None:
    ensure_dirs()
    print("Merging embeddings...", flush=True)
    n_docs = merge_embeddings()
    print("\nMerging QnA...", flush=True)
    n_qna = merge_qna()
    print(f"\nCombined corpus: {n_docs} documents, {n_qna} QnA pairs.", flush=True)


if __name__ == "__main__":
    main()
