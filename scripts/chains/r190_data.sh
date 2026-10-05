#!/bin/bash
# Row 190: the v4 shared world, rebuilt with both fixes (REALSTYLE_V4=1: order codes re-coded from render_v2's recorded Reference;
# CROWD_KEY=v2: crowd keys drop only code-like tokens). New file names only; v3 files are untouched.
set -eu
cd "$(dirname "$0")/../.."
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 CROWD_KEY=v2 TEST_LAYOUT=split
SEEDS=0-799 OUT=realstyle_crowd_v4_train.json uv run python scripts/build_crowd.py | tail -1
SPLIT=test SEEDS=100000-100099 OUT=realstyle_crowd_v4_test_a.json uv run python scripts/build_crowd.py | tail -1
SPLIT=test SEEDS=200000-200699 OUT=realstyle_crowd_v4_test_b.json uv run python scripts/build_crowd.py | tail -1
uv run python - <<'PY'
import json
from ai_experiments.paths import PROCESSED
a, b = (json.loads((PROCESSED / f"realstyle_crowd_v4_test_{x}.json").read_text()) for x in "ab")
for k, v in b["keys"].items():
    for n, hs in v.items():
        a["keys"].setdefault(k, {}).setdefault(n, []).extend(hs)
a["households"] += b["households"]
(PROCESSED / "realstyle_crowd_v4_test.json").write_text(json.dumps(a)); print(len(a["keys"]), "test crowd keys from", len(a["households"]), "households")
for x in "ab":
    (PROCESSED / f"realstyle_crowd_v4_test_{x}.json").unlink()
PY
TRAIN_OUT=realstyle_v4g_train.jsonl uv run python scripts/build_realstyle.py train 400 100 | tail -1
CROWD=realstyle_crowd_v4_train.json CROWD_DROP=0.4 TRAIN_OUT=realstyle_v4gcd_train.jsonl uv run python scripts/build_realstyle.py train 400 100 | tail -1
CROWD=realstyle_crowd_v4_train.json CROWD_DROP=0.4 CROWD_HH_DROP=0.3 TRAIN_OUT=realstyle_v4gcdh_train.jsonl uv run python scripts/build_realstyle.py train 400 100 | tail -1
TEST_OUT=realstyle_v4g_test.json uv run python scripts/build_realstyle.py test 100 60 | tail -1
CROWD=realstyle_crowd_v4_test.json TEST_OUT=realstyle_v4gc_test.json uv run python scripts/build_realstyle.py test 100 60 | tail -1
TEST_SPLIT=train TEST_SEED0=300000 TEST_OUT=realstyle_v4g_seen_test.json uv run python scripts/build_realstyle.py test 100 60 | tail -1
TEST_SPLIT=train TEST_SEED0=300000 CROWD=realstyle_crowd_v4_train.json TEST_OUT=realstyle_v4gc_seen_test.json uv run python scripts/build_realstyle.py test 100 60 | tail -1
echo "== r190 data done"
