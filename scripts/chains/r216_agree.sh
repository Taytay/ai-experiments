#!/bin/bash
# Row 216 (owner, 2026-10-05: "Continue work by comparing to decider"): the late-interaction decision models (row 211's c0, row 210's a5)
# on decider's own items with scripts/hist_agree.py (ENCS=li_r*: one reader "LI", over decider's options): 100 held-out v4 households
# (decider G4 r190 seeds 0 and 1 on realstyle_v4g_ngram_test), then the owner's budget once (decider r190-g s0/s1 on items_grp_sim2;
# aggregates only). One process: households and decider's scores are read once for both models.
set -eu
cd "$(dirname "$0")/../.."
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 TEST_SEEDS=100000-100099
F='Failed to load|warn|Loading|it/s\]|example/s|FutureWarning|Bytecode|UNEXPECTED|position_ids|Notes|LOAD REPORT|^Key|^---'
echo "start $(date -u +%H:%M:%S)"
ENCS=${ENCS:-li_r211_c0,li_r210_a5} OWNER_ENCS=${OWNER_ENCS:-li_r211_c0,li_r210_a5} BUDGET=${BUDGET:-$(cat ~/.config/ynab/budget_id)} \
  uv run python scripts/hist_agree.py 2>&1 | grep --line-buffered -vE "$F"
echo "== r216 agree done $(date -u +%H:%M:%S)"
