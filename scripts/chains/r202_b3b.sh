#!/bin/bash
# Row 202 b3b: b3's random-negative pool with the anchor's and positive's payee masked (b3 lost 13 points: the shared world's pool is full
# of the same payee strings filed by other households), then read beside b1 on the same 50 households.
set -eu
cd "$(dirname "$0")/../.."
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1
F='Failed to load|warn|Loading|FutureWarning|Bytecode|UNEXPECTED|position_ids|Notes|LOAD REPORT|^Key|^---|Writing model|deprecated'
echo "start $(date -u +%H:%M:%S)"
ARM=b3b LOSS=infonce,dedup,pool PAYEE_MASK=1 uv run python scripts/hist_train2.py train 2>&1 | grep --line-buffered -vE "$F"
TEST_SEEDS=100000-100049 ENCS=hist_r202_b3b uv run python scripts/hist_train2.py read 2>&1 | grep --line-buffered -vE "$F"
echo "== r202 b3b done $(date -u +%H:%M:%S)"
