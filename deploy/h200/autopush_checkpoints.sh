#!/usr/bin/env bash
# Periodically push the best checkpoint to the HF remote while training runs.
#
#   nohup bash deploy/h200/autopush_checkpoints.sh h200_run 1500 > /tmp/autopush.log 2>&1 &
#         └ run dir under runs/                    └ interval seconds (default 1500 = 25 min)
#
# WHY: digitalocean_H200_sxm5 is NOT stoppable — the only way to stop billing is
# `brev delete`, which destroys the disk. Anything not pushed is gone. Waiting to
# push until the run finishes means an instance failure at minute 80 loses
# everything. This pushes as training progresses, so the worst case is losing one
# interval of work instead of the whole run.
#
# Only pushes when the checkpoint's hash actually changed, so an unchanged
# `best` (i.e. no eval improvement) costs no bandwidth.
set -uo pipefail

RUN="${1:-h200_run}"
INTERVAL="${2:-1500}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CKPT="runs/$RUN/head.best.pt"
LAST_HASH=""

echo "[autopush] watching $CKPT every ${INTERVAL}s"
git lfs track "runs/$RUN/*.pt" >/dev/null 2>&1 || true

while true; do
  sleep "$INTERVAL"

  if [ ! -f "$CKPT" ]; then
    echo "[autopush] $(date -u +%H:%M:%S) no checkpoint yet"
    continue
  fi

  # Cheap change detection: size+mtime, then hash only if that moved.
  SIG="$(stat -c '%s-%Y' "$CKPT" 2>/dev/null || stat -f '%z-%m' "$CKPT")"
  if [ "$SIG" = "$LAST_HASH" ]; then
    echo "[autopush] $(date -u +%H:%M:%S) unchanged, skipping"
    continue
  fi
  LAST_HASH="$SIG"

  echo "[autopush] $(date -u +%H:%M:%S) pushing $CKPT ($(du -h "$CKPT" | cut -f1))"
  git add .gitattributes "$CKPT" "runs/$RUN/metrics.jsonl" 2>/dev/null
  if git diff --cached --quiet; then
    echo "[autopush] nothing staged"
    continue
  fi
  git commit -q -m "autopush: $RUN checkpoint @ $(date -u +%Y-%m-%dT%H:%M:%SZ)" || true
  # Rebase on any concurrent remote change rather than failing the push.
  git pull --rebase -q origin main 2>/dev/null || true
  if git push -q origin main 2>&1; then
    echo "[autopush] pushed OK"
  else
    echo "[autopush] PUSH FAILED — checkpoint still only on this instance"
  fi
done
