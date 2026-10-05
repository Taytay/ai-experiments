#!/bin/bash
# Row 210 speed check, part 3: updates mattered more than data per update (1,000 steps x 8 windows 74.2 < 3,000 x 1 window 75.0); keep
# 3,000 steps with 8 windows (8x the data, ~9 min on the 3090) and see whether it beats Modal's a1. Prepare is cached from this run on.
set -eu
cd "$(dirname "$0")/../.."
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1
F='Failed to load|warn|Loading|FutureWarning|Bytecode|UNEXPECTED|position_ids|Notes|LOAD REPORT|^Key|^---|Writing model|deprecated'
echo "start $(date -u +%H:%M:%S)"
ARM=a1g8s3k CTX=1 GROUPS=8 STEPS=3000 uv run python scripts/li_decider.py train 2>&1 | grep -vE "$F"
echo "trained $(date -u +%H:%M:%S)"
ARMS=a1g8s3k TEST_SEEDS=100000-100049 uv run python scripts/li_decider.py read 2>&1 | grep -vE "$F"
echo "== r210 fast3 done $(date -u +%H:%M:%S)"
