#!/bin/bash
# Row 213 (a)-(b): EVoC layers against row 189's k-means 64 over the shared v4 test world's category texts (crowd households 200000-200699 +
# held-out 100000-100099, two_tower_v1, embedded on CPU), intrinsic (purposes) and the nearest-earlier-payee proxy on held-out 100000-100049.
set -eu
cd "$(dirname "$0")/../.."
mkdir -p logs
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 CROWD_KEY=v2 SPLIT=test SEEDS=200000-200699,100000-100099 TEST_SEEDS=100000-100049 DEVICE=cpu
echo "start $(date -u +%H:%M:%S)"
uv run --with evoc==0.3.1 --with matplotlib python scripts/evoc_clusters.py 2>&1 | grep -vE "Bytecode|Installed|Downloading|Downloaded" | tee logs/r213_evoc.log
echo "== r213 done $(date -u +%H:%M:%S)"
