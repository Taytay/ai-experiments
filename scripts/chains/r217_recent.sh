#!/bin/bash
# Row 217, second round (2026-10-06): the first arms (q1-q3) trained and read with today's visible categories only, so no trip purchase
# was ever trained on or scored (trip categories are hidden after the trip; 5.5-7% of decider's items). With OPTS=recent (visible + any
# category filed in the past year, decider's rule), from row 210's a1, 3,000 steps each: p0 c0's query (the control), p1 12 nearest + "ago",
# p2 p1 + 8 most recent, p3 p2 per category. One read process per set: 50 held-out v4 households (with the trip column), blind_v2 budgets.
set -eu
cd "$(dirname "$0")/../.."
while pgrep -f "chains/r217_(query|blind)[.]sh" >/dev/null; do sleep 20; done
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 ROW=217 CTX=1 STEPS=3000 BASE=models/encoders/li_r210_a1 OPTS=recent
F='Failed to load|warn|Loading|FutureWarning|Bytecode|UNEXPECTED|position_ids|Notes|LOAD REPORT|^Key|^---|Writing model|deprecated'
echo "start $(date -u +%H:%M:%S)"
for a in "p0 5 0 0 rows 96" "p1 12 1 0 rows 352" "p2 12 1 8 rows 512" "p3 12 1 8 group 416"; do
  set -- $a
  [ -f models/encoders/li_r217_$1/li_config.json ] && { echo "== $1 exists"; continue; }
  ARM=$1 NB=$2 AGO=$3 REC=$4 QFMT=$5 MAXLEN=$6 uv run python scripts/li_decider.py train 2>&1 | grep --line-buffered -vE "$F"
  echo "trained $1 $(date -u +%H:%M:%S)"
done
A=li_r211_c0,li_r217_p0,li_r217_p1,li_r217_p2,li_r217_p3
ARMS=$A TEST_SEEDS=100000-100049 uv run python scripts/li_decider.py read 2>&1 | grep --line-buffered -vE "$F"
READ=blind2 ARMS=$A uv run python scripts/li_decider.py read 2>&1 | grep --line-buffered -vE "$F"
echo "== r217 recent done $(date -u +%H:%M:%S)"
