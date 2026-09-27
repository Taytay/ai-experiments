#!/bin/bash
# Wait for a detached Modal job list's local client to exit, then print the scores and exits from its log.
# Run this FILE as the background task (Bash run_in_background: `scripts/wait_modal_jobs.sh r83 <scratchpad>/r83.log`); never an inline
# `until ! pgrep -f "modal_jobs/r83.json" ...` loop: the task's own shell carries that pattern in its command line, pgrep matches it,
# and the loop never ends (2026-09-27: eleven such watchers sat "running" for hours after their jobs finished). See CLAUDE.md.
# usage: scripts/wait_modal_jobs.sh <jobs-file basename, e.g. r83> <log of the `modal run --detach` client>
set -u
jobs="$1"; log="$2"
pat="modal_app.py --jobs scripts/modal_jobs/${jobs}.json"
while pgrep -f -- "$pat" >/dev/null; do sleep 30; done
echo "== ${jobs}: client exited"
grep -E "top-1|R6_all=|last exit|Error:|Traceback" "$log" | grep -v Warning | cut -c1-180 | tail -20
