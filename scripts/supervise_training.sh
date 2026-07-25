#!/bin/bash
# Self-healing supervisor for sixview_v2 training -- runs INDEFINITELY.
# Watches the training process; if it dies for ANY reason (crash, OOM, clean
# early-stop, epoch-done, max-hours), it relaunches from the newest checkpoint.
# It NEVER stops training on its own. It exits only when explicitly told to,
# via the kill-switch file:  runs/STOP_TRAINING
# Runs independently (nohup+disown) so it survives the CLI session ending.
set -u
cd /Users/moncif/gemma4-hack

LOG=runs/sixview_v2_train.log
SUP=runs/supervisor.log
STOP=runs/STOP_TRAINING
PATTERN="train_memory_lora.py --output-dir sixview_v2"
RELAUNCH_MAX_HOURS=12       # each launch runs long; supervisor relaunches on any exit
last_launch=0
fastfail=0

log() { echo "$(date '+%F %T'): $*" >> "$SUP"; }

launch() {
  local ckpt="$1"
  log "RELAUNCH from $ckpt (max-hours $RELAUNCH_MAX_HOURS)"
  nohup ./venv/bin/python scripts/train_memory_lora.py --output-dir sixview_v2 \
    --resume-from "$ckpt" \
    --embeddings-path data/embeddings/aligned6_embeddings.parquet \
    --qna-path data/qna/aligned6_qna.jsonl --epochs 1000 --max-hours "$RELAUNCH_MAX_HOURS" \
    --checkpoint-every-steps 50 --checkpoint-every-minutes 30 --epoch-ckpt-every 5 \
    --eval-every-steps 300 --limit-eval-docs 40 --max-seq-len 512 --fixed-seq-len \
    --max-qna-per-doc 12 --lm-micro-batch 2 --device mps --no-gradient-checkpointing \
    --rank 16 --head-hidden-dim 128 --head-dropout 0.1 --weight-decay 0.05 \
    --early-stop-patience 100000 --lr 8e-5 --lr-total-steps 9000 --min-available-gb 5 \
    >> "$LOG" 2>&1 &
  disown
  last_launch=$(date +%s)
}

valid_ckpt(){ ./venv/bin/python -c "import torch,sys;torch.load(sys.argv[1],map_location='cpu')" "$1" 2>/dev/null; }
newest_ckpt() {
  # newest-first, but skip any CORRUPT checkpoint (integrity-validated with
  # torch.load). A checkpoint killed mid-write can be truncated; this prevents
  # a crash-loop by falling through to the next intact one.
  local c
  for c in $(ls -t runs/sixview_v2/head.latest.pt runs/sixview_v2/head.t*m.pt \
                   runs/sixview_v2/head.ep*.pt runs/sixview_v2/head.best.pt \
                   runs/sixview_v2/head.snapshot.pt 2>/dev/null) \
           runs/sixview_v1/head.best.pt; do
    if [ -f "$c" ] && valid_ckpt "$c"; then echo "$c"; return; fi
  done
  echo runs/sixview_v1/head.best.pt
}

log "supervisor (INDEFINITE) started; stop only via touch $STOP; adopting existing process if alive"

while true; do
  if [ -f "$STOP" ]; then
    log "STOP file present ($STOP); supervisor exiting WITHOUT relaunch (deliberate stop)"
    exit 0
  fi
  if ! pgrep -f "$PATTERN" >/dev/null 2>&1; then
    now=$(date +%s)
    if [ "$last_launch" -gt 0 ] && [ $((now - last_launch)) -lt 120 ]; then
      # died within 120s of our relaunch => checkpoint likely corrupt/mid-write,
      # or a config error. Fall back to progressively safer checkpoints.
      fastfail=$((fastfail + 1))
      if [ "$fastfail" -ge 3 ]; then ckpt=runs/sixview_v1/head.best.pt
      else ckpt="$(newest_ckpt)"; fi   # newest_ckpt already skips corrupt files
      log "FAST-FAIL #$fastfail (died in $((now-last_launch))s); fallback ckpt=$ckpt"
    else
      fastfail=0
      ckpt="$(newest_ckpt)"
      log "training DOWN; newest ckpt=$ckpt"
    fi
    sleep 5
    launch "$ckpt"
    sleep 40   # let it load/start before re-checking
  else
    fastfail=0
  fi
  sleep 45
done
