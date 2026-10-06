#!/bin/bash
# Row 218: decider's training lessons on the late-interaction model, on row 217's chosen query (QARGS, e.g. "NB=12 AGO=1 REC=8 QFMT=rows
# MAXLEN=512"). From row 210's a1, 3,000 steps each, one seed: h1 1,000 training households (decider's recipe saw every v4 training
# household; built once by r218_data.sh); n1 category-name augmentation (decider's RENAME 0.5); hn both. Read on 50 held-out v4 households
# and on blind_v2's 250 users as budgets (the transfer test, row 217) in one process each.
set -eu
cd "$(dirname "$0")/../.."
while pgrep -f "chains/r21[78]_(query|blind|data)[.]sh" >/dev/null; do sleep 20; done
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 ROW=218 CTX=1 STEPS=3000 BASE=models/encoders/li_r210_a1 AGO=1
export ${QARGS:?set QARGS to row 217's chosen query settings}
F='Failed to load|warn|Loading|FutureWarning|Bytecode|UNEXPECTED|position_ids|Notes|LOAD REPORT|^Key|^---|Writing model|deprecated'
echo "start $(date -u +%H:%M:%S) ($QARGS)"
for a in "h1 1000 0" "n1 200 0.5" "hn 1000 0.5"; do
  set -- $a
  [ -f models/encoders/li_r218_$1/li_config.json ] && { echo "== $1 exists"; continue; }
  ARM=$1 HOUSEHOLDS=$2 RENAME=$3 uv run python scripts/li_decider.py train 2>&1 | grep --line-buffered -vE "$F"
  echo "trained $1 $(date -u +%H:%M:%S)"
done
A=li_r218_h1,li_r218_n1,li_r218_hn
ARMS=$A TEST_SEEDS=100000-100049 uv run python scripts/li_decider.py read 2>&1 | grep --line-buffered -vE "$F"
READ=blind2 ARMS=$A uv run python scripts/li_decider.py read 2>&1 | grep --line-buffered -vE "$F"
echo "== r218 done $(date -u +%H:%M:%S)"
