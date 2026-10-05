#!/bin/bash
# Row 208 E7: wait_modal_jobs.sh plus the pull-back: when a detached Modal job list's local client exits, download every tag in the job
# list from the ai-exp-results volume and ingest it (results/, evals/runs.jsonl, adapters), with timestamps. Run as a tracked background
# task right after the launch (Bash run_in_background), so its completion notifies the session:
#   scripts/wait_and_ingest.sh r209 logs/r209_launch.log
# Adapters still need `just push-models` and a commit afterwards (models/adapters must be DVC-added before staging).
set -u
cd "$(dirname "$0")/.."
jobs="$1"; log="$2"
pat="scripts/modal_jobs/${jobs}.json"
echo "== ${jobs}: watching from $(date -u +%H:%M:%S) UTC"
while pgrep -f -- "$pat" >/dev/null; do sleep 30; done
echo "== ${jobs}: client exited $(date -u +%H:%M:%S) UTC"
grep -E "top-1|R6_all=|last exit|Error:|Traceback" "$log" | grep -v Warning | cut -c1-180 | tail -20
tags=$(python3 -c "import json; print(' '.join(j['tag'] for j in json.load(open('$pat'))))")
got=""
for t in $tags; do
  if modal volume get ai-exp-results "$t" modal_out/ --force >/dev/null 2>&1; then got="$got $t"; else echo "   not on the volume: $t"; fi
done
[ -n "$got" ] && uv run python scripts/ingest_modal.py $got 2>&1 | grep -v Bytecode | tail -12
echo "== ${jobs}: ingested ($got) $(date -u +%H:%M:%S) UTC"
