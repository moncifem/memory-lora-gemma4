#!/bin/bash
# Independent: at 08:00 push all the NEW data + current best checkpoint to HF.
# Runs via nohup so it survives the CLI session ending. Reads HF token from .env.
set -u
cd /Users/moncif/gemma4-hack
PL=runs/push8am.log
log(){ echo "$(date '+%F %T'): $*" >> "$PL"; }

TARGET=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-07-25 08:00:00" +%s 2>/dev/null)
NOW=$(date +%s)
SEC=$(( TARGET - NOW )); [ "$SEC" -lt 0 ] && SEC=0
log "sleeping ${SEC}s until 08:00 to push new data"
sleep "$SEC"

log "08:00 reached -- staging new data + checkpoints for HF push"
export HF_TOKEN=$(grep -E "^HF_API_KEY=" .env | head -1 | cut -d= -f2-)

# Safely snapshot live checkpoints (size-stable) so we never commit a half-written file
snap(){
  local src="$1" dst="$2"
  [ -f "$src" ] || return 1
  local s1 s2 s3
  s1=$(stat -f%z "$src"); cp "$src" "$dst"; s2=$(stat -f%z "$dst"); s3=$(stat -f%z "$src")
  if [ "$s1" != "$s2" ] || [ "$s2" != "$s3" ]; then sleep 3; cp "$src" "$dst"; fi
}
snap runs/sixview_v2/head.best.pt   runs/sixview_v2/head.best.pt
snap runs/sixview_v2/head.latest.pt runs/sixview_v2/head.push_latest.pt

# Stage the new/updated data (complete dataset), current checkpoints, logs
git add -f \
  data/embeddings/aligned6_embeddings.parquet \
  data/embeddings/multiview_embeddings.parquet \
  data/qna/aligned6_qna.jsonl \
  data/qna/repo_scoped_qa.jsonl \
  data/docs/multiview_sources.jsonl data/docs/*.jsonl \
  data/multilang_repo_list.txt \
  runs/sixview_v2/head.best.pt \
  runs/sixview_v2/head.push_latest.pt \
  runs/sixview_v2/metrics.jsonl \
  runs/sixview_v2/tb \
  runs/switch.log runs/supervisor.log \
  scripts/supervise_training.sh scripts/orchestrate_switch.sh scripts/push_at_8am.sh \
  2>> "$PL"

git commit -q -m "8am snapshot: complete ~1650-repo dataset + sixview_v2 checkpoints

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" >> "$PL" 2>&1
log "committed; pushing to HF (LFS) ..."

git remote set-url origin "https://moncefem:${HF_TOKEN}@huggingface.co/moncefem/memory-lora-gemma4"
git push origin main >> "$PL" 2>&1
ec=$?
git remote set-url origin "https://huggingface.co/moncefem/memory-lora-gemma4"
log "PUSH_EXIT=$ec (remote scrubbed)"
