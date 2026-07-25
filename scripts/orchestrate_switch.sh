#!/bin/bash
# Independent orchestrator: when the final QA finishes and the complete dataset
# is good, assemble it and switch training onto it WASTING NO TIME -- the kill is
# timed to fire immediately after a checkpoint save, so the relaunch resumes from
# the freshest weights. The always-on supervisor performs the actual relaunch
# (it already reads data/.../aligned6_* which we overwrite here with the complete
# set), so we only need to assemble + time the kill.
set -u
cd /Users/moncif/gemma4-hack
SW=runs/switch.log
log(){ echo "$(date '+%F %T'): $*" >> "$SW"; }

EMB=data/embeddings/aligned6_embeddings.parquet
QNA=data/qna/aligned6_qna.jsonl
TRAIN_PAT="train_memory_lora.py --output-dir sixview_v2"

log "orchestrator started"

# 1) wait for the final QA generation to finish
while pgrep -f "generate_repo_scoped_qa" >/dev/null 2>&1; do sleep 15; done
log "final QA finished"

# 2) report coverage
./venv/bin/python - >> "$SW" 2>&1 <<'PY'
import json
srcs={json.loads(l)['doc_id'] for l in open('data/docs/multiview_sources.jsonl')}
qa={json.loads(l)['doc_id'] for l in open('data/qna/repo_scoped_qa.jsonl')}
print(f"coverage: emb={len(srcs)} qa={len(qa)} missing={len(srcs-qa)}")
PY

# 3) assemble the COMPLETE aligned dataset (overwrites aligned6_* in place;
#    the running trainer already holds its data in memory, so this is safe)
log "assembling complete aligned dataset ..."
./venv/bin/python scripts/assemble_6view_dataset.py >> "$SW" 2>&1

# 4) verify the assembled dataset is bigger/good before switching
REPOS=$(./venv/bin/python -c "import pyarrow.parquet as pq;print(pq.read_table('$EMB').num_rows)" 2>/dev/null || echo 0)
QCOUNT=$(wc -l < "$QNA" 2>/dev/null | tr -d ' ')
log "assembled: repos=$REPOS qa=$QCOUNT (was 1058 repos / 8540 qa)"
if [ "${REPOS:-0}" -lt 1400 ]; then
  log "ABORT SWITCH: assembled repos ($REPOS) < 1400 -- keeping current dataset, NOT switching."
  exit 1
fi

# 5) TIMED KILL: wait for the next checkpoint save (head.latest.pt mtime bump),
#    then kill immediately so the resume loses ~0 steps.
log "waiting for next checkpoint save to time the kill (zero wasted steps) ..."
m0=$(stat -f %m "runs/sixview_v2/head.latest.pt" 2>/dev/null || echo 0)
killed=0
for i in $(seq 1 900); do   # up to ~15 min safety
  m1=$(stat -f %m "runs/sixview_v2/head.latest.pt" 2>/dev/null || echo 0)
  if [ "$m1" != "$m0" ] && [ "$m1" != "0" ]; then
    step=$(grep -oE "step[0-9]+" runs/sixview_v2_train.log | tail -1)
    log "checkpoint just saved (mtime bumped) at $step; KILLING training now to switch dataset"
    pkill -f "$TRAIN_PAT"
    killed=1
    break
  fi
  sleep 1
done
if [ "$killed" = 0 ]; then
  log "no checkpoint save seen in 15min; killing anyway (latest.pt is still recent)"
  pkill -f "$TRAIN_PAT"
fi

log "training killed; supervisor will relaunch on COMPLETE dataset ($REPOS repos) from head.latest.pt within ~45s"
# 6) confirm the relaunch actually happened on the new data
sleep 90
if pgrep -f "$TRAIN_PAT" >/dev/null 2>&1; then
  log "SWITCH OK: training is running again (complete dataset, $REPOS repos)"
else
  log "WARN: training not detected 90s after kill -- supervisor should relaunch; will self-heal"
fi
log "orchestrator done"
