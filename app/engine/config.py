#!/usr/bin/env python3
"""Shared configuration + path resolution for the Memory-LoRA serving app.

The engine lives in ``app/engine`` but reuses the training repo's
``memory_lora`` package and the trained hypernetwork checkpoints under
``runs/``. Everything is resolved relative to the training repo root so the
Next.js app can shell out to these scripts from anywhere.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# app/engine/config.py  ->  app/engine -> app -> <repo_root>
ENGINE_DIR = Path(__file__).resolve().parent
APP_DIR = ENGINE_DIR.parent
REPO_ROOT = APP_DIR.parent

# Make the training package importable (memory_lora.*, scripts.*).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Where per-repo build artifacts live (embeddings, adapters, merged models,
# status.json). One subdirectory per job id. Kept out of the training repo's
# git history via app/.gitignore.
WORKSPACES_DIR = Path(os.environ.get("MLORA_WORKSPACES", APP_DIR / ".workspaces"))

# The trained hypernetwork checkpoint. Prefers the newest run that actually
# beats the frozen baseline -- sixview_v2 is kept last because it scored WORSE
# than no adapter at all (and worse than random noise), so defaulting to it
# silently produced a demo that made the model worse.
def _default_checkpoint() -> Path:
    env = os.environ.get("MLORA_CHECKPOINT")
    if env:
        return Path(env)
    for run in ("all_lora", "h200_run", "local_stdfix", "sixview_v2"):
        p = REPO_ROOT / "runs" / run / "head.best.pt"
        if p.exists():
            return p
    return REPO_ROOT / "runs" / "sixview_v2" / "head.best.pt"


DEFAULT_CHECKPOINT = _default_checkpoint()

# The base model is NOT bundled in this repo — it is pulled separately from the
# official Hugging Face repository (see engine/fetch_base_model.py) into
# ``models/<name>``. BASE_MODEL_ID is the HF id; BASE_MODEL resolves to the
# local clone when present so serving/merging need no network.
BASE_MODEL_ID = os.environ.get("MLORA_BASE_MODEL", "google/gemma-4-E2B")
MODELS_DIR = Path(os.environ.get("MLORA_MODELS_DIR", REPO_ROOT / "models"))


def local_model_dir(repo_id: str) -> Path:
    """Local snapshot directory for a HF repo id (``org/name`` -> models/name)."""
    return MODELS_DIR / repo_id.split("/")[-1]


def resolve_base_model() -> str:
    """Prefer a fully-materialized local clone; fall back to the HF id so
    transformers downloads on first use."""
    d = local_model_dir(BASE_MODEL_ID)
    if (d / "config.json").exists() and (
        any(d.glob("*.safetensors")) or any(d.glob("*.bin"))
    ):
        return str(d)
    return BASE_MODEL_ID


BASE_MODEL = resolve_base_model()

# Compute device for the local (embedding + head + merge) steps.
DEVICE = os.environ.get("MLORA_DEVICE", "mps")

# Port the served OpenAI-compatible engine (vLLM or the transformers fallback)
# listens on. The Next.js app proxies /v1/* here.
SERVE_PORT = int(os.environ.get("MLORA_SERVE_PORT", "8000"))


def workspace(job_id: str) -> Path:
    d = WORKSPACES_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_device(requested: str) -> str:
    """Fall back to CPU when MPS/CUDA is unavailable so the engine never
    crashes on a machine without the requested accelerator."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested
