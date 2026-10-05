#!/bin/bash
# Rows 192 and 193 data, v4 world (REALSTYLE_V4=1, CROWD_KEY=v2). Row 192: category clusters across households (two-tower category side,
# k-means 64) for the test world (crowd households 200000-200699 + held-out 100000-100099) and the training world (0-799); held-out test
# sets with similar-payee rows by character n-grams (SIM_SRC=ngram) or by behaviour (SIM_SRC=behav), each without and with the crowd
# line. Row 193: a training file with the crowd line (40% dropped) and behavioural similar rows (realstyle_v4gcdb_train.jsonl).
set -eu
cd "$(dirname "$0")/../.."
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 CROWD_KEY=v2 TEST_LAYOUT=split SIM=2
SPLIT=test SEEDS=200000-200699,100000-100099 OUT=realstyle_crowdclus_v4_test.json uv run python scripts/build_crowd_clusters.py | grep -v "^  [0-9]"
SPLIT=train SEEDS=0-799 OUT=realstyle_crowdclus_v4_train.json uv run python scripts/build_crowd_clusters.py | grep -v "^  [0-9]"
for src in ngram behav; do
  SIM_SRC=$src CROWD_CLUS=realstyle_crowdclus_v4_test.json TEST_OUT=realstyle_v4g_${src}_test.json uv run python scripts/build_realstyle.py test 100 60 | tail -1
  SIM_SRC=$src CROWD_CLUS=realstyle_crowdclus_v4_test.json CROWD=realstyle_crowd_v4_test.json TEST_OUT=realstyle_v4gc_${src}_test.json uv run python scripts/build_realstyle.py test 100 60 | tail -1
  TEST_SPLIT=train TEST_SEED0=300000 SIM_SRC=$src CROWD_CLUS=realstyle_crowdclus_v4_train.json TEST_OUT=realstyle_v4g_seen_${src}_test.json uv run python scripts/build_realstyle.py test 100 60 | tail -1
  TEST_SPLIT=train TEST_SEED0=300000 SIM_SRC=$src CROWD_CLUS=realstyle_crowdclus_v4_train.json CROWD=realstyle_crowd_v4_train.json TEST_OUT=realstyle_v4gc_seen_${src}_test.json uv run python scripts/build_realstyle.py test 100 60 | tail -1
done
SIM_SRC=behav CROWD_CLUS=realstyle_crowdclus_v4_train.json CROWD=realstyle_crowd_v4_train.json CROWD_DROP=0.4 TRAIN_OUT=realstyle_v4gcdb_train.jsonl uv run python scripts/build_realstyle.py train 400 100 | tail -1
echo "== r192 data done"
