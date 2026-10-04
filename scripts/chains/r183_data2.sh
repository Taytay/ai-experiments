#!/bin/bash
# Row 183 data, after the parser fix: households with crowd lines, and the test sets: new-world held-out households (test merchants,
# crowd from the test world) and same-world held-out households (training merchants, crowd from the training households).
set -eu
cd "$(dirname "$0")/../.."
export SHARED_WORLD=1 GROUPNAMES=1 TEST_LAYOUT=split
CROWD=realstyle_crowd_train.json TRAIN_OUT=realstyle_v3gc_train.jsonl uv run python scripts/build_realstyle.py train 400 100 | tail -1
TEST_OUT=realstyle_v3g_test.json uv run python scripts/build_realstyle.py test 100 60 | tail -1
CROWD=realstyle_crowd_test.json TEST_OUT=realstyle_v3gc_test.json uv run python scripts/build_realstyle.py test 100 60 | tail -1
TEST_SPLIT=train TEST_SEED0=300000 TEST_OUT=realstyle_v3g_seen_test.json uv run python scripts/build_realstyle.py test 100 60 | tail -1
TEST_SPLIT=train TEST_SEED0=300000 CROWD=realstyle_crowd_train.json TEST_OUT=realstyle_v3gc_seen_test.json uv run python scripts/build_realstyle.py test 100 60 | tail -1
echo "== r183 data done"
