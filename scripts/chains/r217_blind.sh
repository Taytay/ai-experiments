#!/bin/bash
# Row 217: does blind_v2 (another generator, scripts/blind_budgets.py) rank the late-interaction models as the owner's budget did (row 216:
# a5 64.6 > c0 60.9 there; c0 ahead on our own synthetic households)? If so it is the transfer test arms are chosen on.
set -eu
cd "$(dirname "$0")/../.."
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1
F='Failed to load|warn|Loading|FutureWarning|Bytecode|UNEXPECTED|position_ids|Notes|LOAD REPORT|^Key|^---|Writing model|deprecated'
while pgrep -f "chains/r217_query[.]sh" >/dev/null; do sleep 20; done  # one GPU job at a time (RAM)
echo "start $(date -u +%H:%M:%S)"
READ=blind2 ARMS=${ARMS:-li_r211_c0,li_r210_a5,li_r210_a1,li_r210_a0,li_r217_q1,li_r217_q2,li_r217_q3} uv run python scripts/li_decider.py read 2>&1 | grep --line-buffered -vE "$F"
echo "== r217 blind done $(date -u +%H:%M:%S)"
