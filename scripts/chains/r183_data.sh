#!/bin/bash
# Row 183 data: a shared synthetic world (SHARED_WORLD=1), its crowd tables, households with and without "Others filed this payee as"
# lines (group names on), and the held-out households' test sets in the split layout.
set -eu
cd "$(dirname "$0")/../.."
export SHARED_WORLD=1 GROUPNAMES=1
SEEDS=0-799 OUT=realstyle_crowd_train.json uv run python scripts/build_crowd.py | tail -1
SPLIT=test SEEDS=100000-100099 OUT=realstyle_crowd_test_a.json uv run python scripts/build_crowd.py | tail -1
SPLIT=test SEEDS=200000-200699 OUT=realstyle_crowd_test_b.json uv run python scripts/build_crowd.py | tail -1
uv run python - <<'PY'
import json
from ai_experiments.paths import PROCESSED
a, b = (json.loads((PROCESSED / f"realstyle_crowd_test_{x}.json").read_text()) for x in "ab")
for k, v in b["keys"].items():
    for n, hs in v.items():
        a["keys"].setdefault(k, {}).setdefault(n, []).extend(hs)
a["households"] += b["households"]
(PROCESSED / "realstyle_crowd_test.json").write_text(json.dumps(a)); print(len(a["keys"]), "test crowd keys from", len(a["households"]), "households")
PY
TRAIN_OUT=realstyle_v3g_train.jsonl uv run python scripts/build_realstyle.py train 400 100 | tail -1
CROWD=realstyle_crowd_train.json TRAIN_OUT=realstyle_v3gc_train.jsonl uv run python scripts/build_realstyle.py train 400 100 | tail -1
TEST_LAYOUT=split TEST_OUT=realstyle_v3g_test.json uv run python scripts/build_realstyle.py test 100 60 | tail -1
TEST_LAYOUT=split CROWD=realstyle_crowd_test.json TEST_OUT=realstyle_v3gc_test.json uv run python scripts/build_realstyle.py test 100 60 | tail -1
echo "== r183 data done"
