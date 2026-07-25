#!/usr/bin/env bash
# One-time setup on a Brev H200 SXM5 (141GB) instance.
#
#   bash deploy/h200/setup.sh
#
# Installs Python deps, pulls the frozen encoder + base model, and verifies the
# GPU is visible. Safe to re-run: every step is idempotent.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== [1/5] GPU check ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
  echo "!! no GPU visible — this must run on the H200 instance, not locally" >&2
  exit 1
}

echo "=== [2/5] Python deps ==="
python3 -m pip install --upgrade pip
# torch with CUDA is preinstalled on Brev images; only install it if missing so
# we never downgrade the image's CUDA-matched build.
python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
  echo "installing torch (cu124)"
  python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu124
}
# requirements-h200.txt deliberately omits the torch pin so we never replace
# the image's CUDA-matched build (see that file's header).
python3 -m pip install -r deploy/h200/requirements-h200.txt
# FlashAttention-2: big win on H200 for the frozen forward pass. Optional —
# training falls back to sdpa if the build fails.
python3 -m pip install flash-attn --no-build-isolation 2>/dev/null \
  && echo "flash-attn installed" \
  || echo "flash-attn unavailable — will use sdpa (still fine)"

echo "=== [3/5] base model ==="
if [ -f models/gemma-4-E2B/config.json ] && ls models/gemma-4-E2B/*.safetensors >/dev/null 2>&1; then
  echo "already present ($(du -sh models/gemma-4-E2B | cut -f1))"
else
  python3 app/engine/fetch_base_model.py
fi

echo "=== [4/5] frozen encoder (only needed to embed NEW repos) ==="
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-Embedding-0.6B")
print("encoder cached")
PY

echo "=== [5/6] training data (targeted LFS pull) ==="
# `git lfs pull` with no filter fetches ~10 GB of raw corpora (commitpack,
# RepoPeftBench, combined_qna) that this training run never reads. Training
# needs ~0.14 GB. On a per-hour GPU that difference is billed download time.
git lfs pull --include="data/embeddings/aligned6_embeddings.parquet,data/qna/aligned6_qna.jsonl"

echo "=== [6/6] data check ==="
python3 - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq, collections, sys
emb = Path("data/embeddings/aligned6_embeddings.parquet")
qna = Path("data/qna/aligned6_qna.jsonl")
missing = [str(p) for p in (emb, qna) if not p.exists()]
if missing:
    print("!! MISSING (upload these from your Mac):", missing); sys.exit(1)
t = pq.read_table(emb)
print(f"embeddings: {t.num_rows} rows, dim {len(t.column('doc_embedding')[0].as_py())}, "
      f"splits {dict(collections.Counter(t.column('split').to_pylist()))}")
print(f"qna: {sum(1 for _ in open(qna))} rows")
PY

echo
echo "setup complete —  next:  python3 deploy/h200/preflight.py"
