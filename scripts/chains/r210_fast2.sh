#!/bin/bash
# Row 210 speed check, part 2: 8 household windows per step trained a1 16x faster (63 s) but lost 1.5 / 2.5 (73.5 / 60.0 against 75.0 / 62.5):
# 8x fewer updates. Two fixes on the same data: a higher learning rate (1.5e-4, 380 steps) and more steps (1,000 at 5e-5, ~2.6x the data).
set -eu
cd "$(dirname "$0")/../.."
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1
F='Failed to load|warn|Loading|FutureWarning|Bytecode|UNEXPECTED|position_ids|Notes|LOAD REPORT|^Key|^---|Writing model|deprecated'
echo "start $(date -u +%H:%M:%S)"
ARM=a1g8lr CTX=1 GROUPS=8 STEPS=380 LR=1.5e-4 uv run python scripts/li_decider.py train 2>&1 | grep -vE "$F"
ARM=a1g8s1k CTX=1 GROUPS=8 STEPS=1000 uv run python scripts/li_decider.py train 2>&1 | grep -vE "$F"
echo "trained $(date -u +%H:%M:%S)"
ARMS=a1g8lr,a1g8s1k TEST_SEEDS=100000-100049 uv run python scripts/li_decider.py read 2>&1 | grep -vE "$F"
echo "== r210 fast2 done $(date -u +%H:%M:%S)"
