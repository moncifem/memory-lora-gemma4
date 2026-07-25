#!/usr/bin/env python3
"""Step A of the side recall-test: embed THIS repo (gemma4-hack) with the
frozen encoder in a short-lived, low-memory process, save the 2048-d vector
to disk, and exit -- so the big base-model process (Step B) never has the
encoder resident at the same time. CPU-only to avoid competing with the
live MPS training run."""
from __future__ import annotations
import sys, gc
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
from memory_lora.encoder import load_encoder, embed_document
from memory_lora.codegraph import extract_repo_graph_sections, extract_repo_dependency_summary

def main():
    skip = {".git","venv","__pycache__","data","runs",".mypy_cache","scratchpad"}
    print("Extracting codegraph sections from this repo...", flush=True)
    graph_sections = extract_repo_graph_sections(REPO_ROOT, max_files=60, skip_dirs=skip)
    dep = extract_repo_dependency_summary(REPO_ROOT, skip_dirs=skip)
    sections = list(graph_sections)
    if dep:
        sections.append(("dependency_graph", dep))
    # also add a few raw key files for content signal
    for rel in ["memory_lora/core.py", "memory_lora/encoder.py", "scripts/train_memory_lora.py",
                 "memory_lora/codegraph.py", "requirements.txt"]:
        p = REPO_ROOT / rel
        if p.exists():
            sections.append((f"raw:{rel}", p.read_text(errors="ignore")))
    print(f"  {len(sections)} sections", flush=True)

    print("Loading encoder on CPU...", flush=True)
    model, tok = load_encoder(device="cpu")
    vec = embed_document(sections, model, tok, "cpu", chunk_tokens=2048, chunk_overlap=256, batch_size=2)
    del model, tok; gc.collect()
    if vec is None:
        print("FAILED: no embedding", flush=True); return
    out = REPO_ROOT / "runs" / "this_repo_emb.npy"
    np.save(out, vec.numpy().astype("float32"))
    print(f"Saved repo embedding {tuple(vec.shape)} -> {out}", flush=True)

if __name__ == "__main__":
    main()
