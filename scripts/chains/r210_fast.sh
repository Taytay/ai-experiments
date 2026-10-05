#!/bin/bash
# Row 210 speed check (owner, 2026-10-05: "if you were to retrain now, tell me how fast it would go after this fix?"): arm a1 retrained on
# the local 3090 with 8 household windows per step and the same data (~83k transactions: 380 steps x ~220), timed, then read on the same
# 50 households to confirm it matches the Modal a1 (75.0 / 62.5 first-time). Waits for the row 202 re-score (RAM).
set -eu
cd "$(dirname "$0")/../.."
while pgrep -f "chains/r202_agree.sh" >/dev/null; do sleep 20; done
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1
F='Failed to load|warn|Loading|FutureWarning|Bytecode|UNEXPECTED|position_ids|Notes|LOAD REPORT|^Key|^---|Writing model|deprecated'
echo "start $(date -u +%H:%M:%S)"
ARM=a1g8 CTX=1 GROUPS=8 STEPS=380 uv run python scripts/li_decider.py train 2>&1 | grep --line-buffered -vE "$F"
echo "trained $(date -u +%H:%M:%S)"
ARMS=a1g8 TEST_SEEDS=100000-100049 uv run python scripts/li_decider.py read 2>&1 | grep --line-buffered -vE "$F"
echo "== r210 fast done $(date -u +%H:%M:%S)"
