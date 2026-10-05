#!/bin/bash
# Row 189 data: category clusters across households (two-tower category side, k-means 64) and per-string cluster counts for the test
# world (crowd households 200000-200699 + the held-out 100000-100099) and the training world (0-799); then the held-out test sets with
# similar-payee rows chosen by character n-grams (SIM_SRC=ngram) or by behaviour (SIM_SRC=behav, fused with n-grams like SIM=2).
set -eu
cd "$(dirname "$0")/../.."
export SHARED_WORLD=1 GROUPNAMES=1 TEST_LAYOUT=split SIM=2
SPLIT=test SEEDS=200000-200699,100000-100099 OUT=realstyle_crowdclus_test.json uv run python scripts/build_crowd_clusters.py | grep -v "^  [0-9]"
SPLIT=train SEEDS=0-799 OUT=realstyle_crowdclus_train.json uv run python scripts/build_crowd_clusters.py | grep -v "^  [0-9]"
SIM_SRC=ngram TEST_OUT=realstyle_v3g_ngram_test.json uv run python scripts/build_realstyle.py test 100 60 | tail -1
SIM_SRC=behav CROWD_CLUS=realstyle_crowdclus_test.json TEST_OUT=realstyle_v3g_behav_test.json uv run python scripts/build_realstyle.py test 100 60 | tail -1
TEST_SPLIT=train TEST_SEED0=300000 SIM_SRC=ngram TEST_OUT=realstyle_v3g_seen_ngram_test.json uv run python scripts/build_realstyle.py test 100 60 | tail -1
TEST_SPLIT=train TEST_SEED0=300000 SIM_SRC=behav CROWD_CLUS=realstyle_crowdclus_train.json TEST_OUT=realstyle_v3g_seen_behav_test.json uv run python scripts/build_realstyle.py test 100 60 | tail -1
echo "== r189 data done"
