#!/bin/bash
# Row 204: several prototypes per category (scripts/hist_proto.py, training-free): Ward clusters per category as medoids and centroids,
# medoids + time-decayed importance (lambda 0, 0.01, 0.05), farthest-point K 3 and 5, against MaxSim over all filings, on 50 held-out
# v4 households with hist_knn_v1. Log: logs/r204_proto.log.
set -eu
cd "$(dirname "$0")/../.."
mkdir -p logs
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1
F='Failed to load|warn|Loading|FutureWarning|Bytecode|UNEXPECTED|position_ids|Notes|LOAD REPORT|^Key|^---|deprecated'
{
  echo "start $(date -u +%H:%M:%S)"
  ENC=hist_knn_v1 TEST_SEEDS=100000-100049 LAMBDAS=0,0.01,0.05 K=3,5 uv run python scripts/hist_proto.py 2>&1 | grep --line-buffered -vE "$F"
  echo "== r204 done $(date -u +%H:%M:%S)"
} 2>&1 | tee logs/r204_proto.log
