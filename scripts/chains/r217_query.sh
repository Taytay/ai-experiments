#!/bin/bash
# Row 217: decider's history slice in the late-interaction query. From row 210's a1, 3,000 steps each on the local 3090, as row 211's c0
# (the control: a1 + 3,000 steps with the old query, 75.7 / 63.4 on these households). q1: 12 nearest earlier filings, each with "N d ago";
# q2: q1 + the household's 8 most recent filings (newest first); q3: q2 written once per category. Then one read process for all three
# on the same 50 held-out households (neighbour lists cached per budget in data/interim/li_nb).
set -eu
cd "$(dirname "$0")/../.."
while pgrep -f "chains/r216_agree.sh" >/dev/null; do sleep 20; done
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 ROW=217 CTX=1 STEPS=3000 BASE=models/encoders/li_r210_a1
F='Failed to load|warn|Loading|FutureWarning|Bytecode|UNEXPECTED|position_ids|Notes|LOAD REPORT|^Key|^---|Writing model|deprecated'
echo "start $(date -u +%H:%M:%S)"
for a in "q1 12 0 rows 352" "q2 12 8 rows 512" "q3 12 8 group 416"; do
  set -- $a
  [ -f models/encoders/li_r217_$1/li_config.json ] && { echo "== $1 exists"; continue; }
  ARM=$1 NB=$2 AGO=1 REC=$3 QFMT=$4 MAXLEN=$5 uv run python scripts/li_decider.py train 2>&1 | grep --line-buffered -vE "$F"
  echo "trained $1 $(date -u +%H:%M:%S)"
done
ARMS=li_r217_q1,li_r217_q2,li_r217_q3 TEST_SEEDS=100000-100049 uv run python scripts/li_decider.py read 2>&1 | grep --line-buffered -vE "$F"
echo "== r217 done $(date -u +%H:%M:%S)"
