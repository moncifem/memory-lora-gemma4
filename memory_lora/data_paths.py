#!/usr/bin/env python3
"""Local data path resolution for Memory-LoRA.

Unlike Code2LoRA's ``data_paths.py`` (which lazily ``snapshot_download``s from
the ``code2lora/`` HF org), everything here is generated locally via
OpenRouter and never leaves the machine unless the user chooses to publish
it -- so this just resolves to ``<repo_root>/data/``.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"

DOCS_DIR = DATA_ROOT / "docs"          # raw generated documents (jsonl)
EMBEDDINGS_DIR = DATA_ROOT / "embeddings"  # doc embeddings parquet
QNA_DIR = DATA_ROOT / "qna"            # recall QA pairs parquet
CACHE_DIR = DATA_ROOT / "openrouter_cache"  # raw OpenRouter responses, keyed by prompt hash
RUNS_DIR = REPO_ROOT / "runs"          # training checkpoints + metrics


def ensure_dirs() -> None:
    for d in (DOCS_DIR, EMBEDDINGS_DIR, QNA_DIR, CACHE_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)


__all__ = [
    "REPO_ROOT", "DATA_ROOT", "DOCS_DIR", "EMBEDDINGS_DIR", "QNA_DIR",
    "CACHE_DIR", "RUNS_DIR", "ensure_dirs",
]
