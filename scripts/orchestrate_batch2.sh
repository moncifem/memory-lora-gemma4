#!/bin/bash
# Independent batch2 pipeline: wait for the novel-repo CPU build to finish, then
# run the ENHANCED QA pass on exactly those repos. Never touches training.
# Leaves everything staged for a next-day assemble (does NOT re-assemble aligned6).
set -u
cd /Users/moncif/gemma4-hack
BL=runs/build_batch2.log
QL=runs/qa_batch2.log
OL=runs/batch2_orchestrator.log
log(){ echo "$(date '+%F %T'): $*" >> "$OL"; }

log "batch2 orchestrator started; waiting for build to finish"
# wait for the build process to end
while pgrep -f "build_repo_multiview.py --repos-file data/batch2_novel_repos.txt" >/dev/null 2>&1; do
  sleep 30
done
log "build finished: $(grep -oE 'Done\. [0-9]+ repos.*' "$BL" | tail -1)"

# run enhanced QA on just the novel batch repos (resume-safe, appends to main qna)
log "launching enhanced QA (debug/security/perf/concurrency/migration)"
./venv/bin/python scripts/generate_repo_scoped_qa_v2.py \
  --sources data/docs/multiview_sources.jsonl \
  --only-repos data/batch2_novel_repos.txt \
  --model google/gemma-4-31b-it --workers 10 >> "$QL" 2>&1
log "enhanced QA done: $(grep -oE 'Done\. [0-9]+ novel repos.*' "$QL" | tail -1)"
log "batch2 dataset READY for next-day assemble (novel repos in main pool; enhanced QA appended)"
