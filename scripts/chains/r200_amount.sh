#!/bin/bash
# Row 200: amounts as numbers for the history encoders (scripts/hist_amount.py): after encoder B (no amount in the text) has trained;
# settings chosen on 10 held-out synthetic households, applied to the owner's budget (private; aggregates only).
set -eu
cd "$(dirname "$0")/../.."
until grep -q "hist_knn_noamt_v1" logs/r200_train_noamt.log 2>/dev/null; do sleep 30; done
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 BUDGET=${BUDGET:-$(cat ~/.config/ynab/budget_id)} TEST_SEEDS=100000-100009
uv run python scripts/hist_amount.py 2>&1 | grep -vE 'Failed to load|warn|Loading|it/s\]|example/s|FutureWarning'
echo "== r200 done"
