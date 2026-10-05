#!/bin/bash
# Row 211: interaction layers on the late-interaction decision model, from row 210's a1 (query with the nearest earlier filings, MaxSim over
# per-category documents), 3,000 steps each on the local 3090 (row 210: updates, not data per update, set accuracy). c0 = a1 trained 3,000
# more steps without new layers (the control: more training alone); x = cross-attention to each option's tokens; xm = + memory tokens from
# the nearest earlier filings; xml = + a listwise layer over the options. Read on the same 50 held-out households as rows 202 and 210.
set -eu
cd "$(dirname "$0")/../.."
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 ROW=211 CTX=1 STEPS=3000 BASE=models/encoders/li_r210_a1
F='Failed to load|warn|Loading|FutureWarning|Bytecode|UNEXPECTED|position_ids|Notes|LOAD REPORT|^Key|^---|Writing model|deprecated'
echo "start $(date -u +%H:%M:%S)"
for a in "c0 " "x x" "xm xm" "xml xml"; do
  set -- $a
  [ -f models/encoders/li_r211_$1/li_config.json ] && { echo "== $1 exists"; continue; }
  ARM=$1 INTERACT=${2:-} uv run python scripts/li_decider.py train 2>&1 | grep -vE "$F"
  echo "trained $1 $(date -u +%H:%M:%S)"
done
ARMS=li_r211_c0,li_r211_x,li_r211_xm,li_r211_xml TEST_SEEDS=100000-100049 uv run python scripts/li_decider.py read 2>&1 | grep -vE "$F"
echo "== r211 done $(date -u +%H:%M:%S)"
