#!/bin/bash
# Self-healing supervisor for sixview_v4 (enlarged 2066-repo dataset incl. batch2
# novel repos + enhanced QA). Warm-starts from v3's best checkpoint on first
# launch, then resumes from its own newest VALID checkpoint. Runs INDEFINITELY;
# stop only via: touch runs/STOP_TRAINING. Independent (nohup+disown).
set -u
cd /Users/moncif/gemma4-hack

OUT=sixview_v4
LOG=runs/${OUT}_train.log
SUP=runs/supervisor_v4.log
STOP=runs/STOP_TRAINING
PATTERN="train_memory_lora.py --output-dir ${OUT}"
WARM=runs/sixview_v2/head.best.pt      # v3 best (cr_val 2.5995) warm start
RELAUNCH_MAX_HOURS=12
last_launch=0; fastfail=0

log(){ echo "$(date '+%F %T'): $*" >> "$SUP"; }
valid_ckpt(){ ./venv/bin/python -c "import torch,sys;torch.load(sys.argv[1],map_location='cpu')" "$1" 2>/dev/null; }
newest_ckpt() {
  local c
  for c in $(ls -t runs/${OUT}/head.latest.pt runs/${OUT}/head.t*m.pt \
                   runs/${OUT}/head.ep*.pt runs/${OUT}/head.best.pt 2>/dev/null) \
           "$WARM" runs/sixview_v1/head.best.pt; do
    if [ -f "$c" ] && valid_ckpt "$c"; then echo "$c"; return; fi
  done
  echo "$WARM"
}

launch() {
  local ckpt="$1"
  log "RELAUNCH from $ckpt (max-hours $RELAUNCH_MAX_HOURS)"
  nohup ./venv/bin/python scripts/train_memory_lora.py --output-dir "$OUT" \
    --resume-from "$ckpt" \
    --embeddings-path data/embeddings/aligned6_embeddings.parquet \
    --qna-path data/qna/aligned6_qna.jsonl --epochs 1000 --max-hours "$RELAUNCH_MAX_HOURS" \
    --checkpoint-every-steps 50 --checkpoint-every-minutes 30 --epoch-ckpt-every 5 \
    --eval-every-steps 300 --limit-eval-docs 40 --max-seq-len 512 --fixed-seq-len \
    --max-qna-per-doc 12 --lm-micro-batch 2 --device mps --no-gradient-checkpointing \
    --rank 16 --head-hidden-dim 128 --head-dropout 0.1 --weight-decay 0.05 \
    --early-stop-patience 100000 --lr 8e-5 --lr-total-steps 12000 --min-available-gb 5 \
    >> "$LOG" 2>&1 &
  disown
  last_launch=$(date +%s)
}

log "supervisor_v4 (INDEFINITE) started; dataset=2066 repos; warm=$WARM; stop via touch $STOP"
while true; do
  if [ -f "$STOP" ]; then log "STOP file present; exiting WITHOUT relaunch"; exit 0; fi
  if ! pgrep -f "$PATTERN" >/dev/null 2>&1; then
    now=$(date +%s)
    if [ "$last_launch" -gt 0 ] && [ $((now - last_launch)) -lt 120 ]; then
      fastfail=$((fastfail + 1))
      if [ "$fastfail" -ge 3 ]; then ckpt="$WARM"; else ckpt="$(newest_ckpt)"; fi
      log "FAST-FAIL #$fastfail (died in $((now-last_launch))s); fallback ckpt=$ckpt"
    else
      fastfail=0; ckpt="$(newest_ckpt)"; log "training DOWN; newest ckpt=$ckpt"
    fi
    sleep 5; launch "$ckpt"; sleep 40
  else
    fastfail=0
  fi
  sleep 45
done
