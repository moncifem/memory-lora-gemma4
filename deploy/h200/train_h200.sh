#!/usr/bin/env bash
# Train the Memory-LoRA hypernetwork on a single H200 SXM5 (141 GB HBM3e).
#
#   bash deploy/h200/preflight.py     # ALWAYS run this first
#   GATE=1 bash deploy/h200/train_h200.sh    # ~20 min go/no-go
#   bash deploy/h200/train_h200.sh           # full run
#
# Exploiting 141 GB
# -----------------
# One training step = ONE document plus its sampled QA pairs. So raising
# --lm-micro-batch alone stops helping once the batch exceeds
# --max-qna-per-doc: the extra capacity has nothing to put in it. To actually
# use the card, BOTH are raised together (32/32 here vs 12/2 on the Mac), which
# means each step sees 32 QA pairs for a document in a single forward instead of
# six sequential passes of 2.
#
# Memory at these settings is dominated by the head and its optimizer state,
# not the frozen base model:
#   frozen Gemma bf16          ~10 GB
#   head fp32 (hidden 512)      ~3 GB
#   AdamW moments (2x)          ~6 GB
#   gradients                   ~3 GB
#   activations (32x512)      ~10-30 GB
# ...so ~60 GB of the 141 GB is headroom. Gradient checkpointing is OFF because
# we are not memory-bound; that alone is ~30% more throughput.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT="${OUT:-h200_run}"
MICRO_BATCH="${MICRO_BATCH:-32}"
MAX_QNA="${MAX_QNA:-32}"
LR="${LR:-2e-4}"
HIDDEN="${HIDDEN:-512}"
ATTN="${ATTN:-sdpa}"
SEQ="${SEQ:-512}"
EXTRA=()

if [ "${GATE:-0}" = "1" ]; then
  # Gate: ~20 min. Sole purpose is to answer "is the adapter beating the frozen
  # baseline, and is it repo-conditional?" BEFORE committing the budget.
  OUT="${OUT}_gate"
  EXTRA+=(--epochs 1 --limit-train-docs 400 --eval-every-steps 100
          --limit-eval-docs 30 --max-hours 0.4)
  echo ">>> GATE RUN (~20 min) -> runs/$OUT"
else
  EXTRA+=(--epochs 400 --eval-every-steps 400 --limit-eval-docs 60
          --max-hours "${MAX_HOURS:-1.5}")
  echo ">>> FULL RUN -> runs/$OUT   (budget ${MAX_HOURS:-1.5}h)"
fi

[ -n "${RESUME:-}" ] && { EXTRA+=(--resume-from "$RESUME"); echo ">>> resuming from $RESUME"; }

mkdir -p "runs/$OUT"
set -x
python3 scripts/train_memory_lora.py \
  --embeddings-path data/embeddings/aligned6_embeddings.parquet \
  --qna-path data/qna/aligned6_qna.jsonl \
  --output-dir "$OUT" \
  --model-name models/gemma-4-E2B \
  --device cuda \
  --dtype bfloat16 \
  --attn-implementation "$ATTN" \
  --head-hidden-dim "$HIDDEN" \
  --lr "$LR" \
  --rank 16 --alpha 32 \
  --max-qna-per-doc "$MAX_QNA" \
  --lm-micro-batch "$MICRO_BATCH" \
  --max-seq-len "$SEQ" --fixed-seq-len \
  --eval-suites cr_val cr_test \
  --no-gradient-checkpointing \
  --checkpoint-every-steps 100 \
  --checkpoint-every-minutes 10 \
  --log-every-iters 25 \
  --min-available-gb 8 \
  "${EXTRA[@]}" \
  2>&1 | tee -a "runs/$OUT/train.log"
