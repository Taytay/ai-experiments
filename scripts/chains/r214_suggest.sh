#!/bin/bash
# Row 214 (exploratory, synthetic): suggest a category scheme per held-out household (100000-100049, shared v4 test world) by clustering
# its payees (history-encoder name vectors, row 213's crowd vectors over k-means-64 clusters, amount/rhythm), EVoC layers and oracle-k
# k-means / agglomerative against the household's real categories; treemap JSON of household 100003 to data/interim/r214/. CPU only.
set -eu
cd "$(dirname "$0")/../.."
mkdir -p logs
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 CROWD_KEY=v2 TEST_SEEDS=100000-100049 DEVICE=cpu CUDA_VISIBLE_DEVICES= TREE_SEED=100003
echo "start $(date -u +%H:%M:%S)"
uv run --with evoc==0.3.1 --with matplotlib python scripts/suggest_categories.py 2>&1 | grep -vE "Bytecode|Installed|Downloading|Downloaded" | tee logs/r214_suggest.log
echo "== r214 done $(date -u +%H:%M:%S)"
