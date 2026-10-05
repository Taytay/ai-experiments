#!/bin/bash
# Row 202: encoder training fixes from the Pinterest papers (scripts/hist_train2.py), one seed per arm on the local GPU, then kNN and MaxSim
# on 50 held-out v4 households against hist_knn_v1. Arms: b0 InfoNCE in the plain loop (hist_knn_v1's loss), b1 + dedup and payee cap,
# b2 + logQ, b3 + random-negative pool, b4 household softmax alone, b5 InfoNCE + dedup + household softmax.
set -eu
cd "$(dirname "$0")/../.."
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1
F='Failed to load|warn|Loading|FutureWarning|Bytecode|UNEXPECTED|position_ids|Notes|LOAD REPORT|^Key|^---|Writing model|deprecated'
echo "start $(date -u +%H:%M:%S)"
for a in "b0 infonce" "b1 infonce,dedup" "b2 infonce,dedup,logq" "b3 infonce,dedup,pool" "b4 hs" "b5 infonce,dedup,hs"; do
  set -- $a
  [ -f models/encoders/hist_r202_$1/model.safetensors ] && { echo "== $1 exists"; continue; }
  ARM=$1 LOSS=$2 uv run python scripts/hist_train2.py train 2>&1 | grep --line-buffered -vE "$F"
done
TEST_SEEDS=100000-100049 ENCS=hist_knn_v1,hist_r202_b0,hist_r202_b1,hist_r202_b2,hist_r202_b3,hist_r202_b4,hist_r202_b5 \
  uv run python scripts/hist_train2.py read 2>&1 | grep --line-buffered -vE "$F"
echo "== r202 done $(date -u +%H:%M:%S)"
