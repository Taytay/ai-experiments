#!/bin/bash
# Row 199: decider + history encoders with settings chosen on synthetic households, applied to the owner's budget (private; aggregates only).
set -eu
cd "$(dirname "$0")/../.."
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 BUDGET=${BUDGET:-$(cat ~/.config/ynab/budget_id)}
uv run python scripts/hist_fuse.py 2>&1 | grep -vE 'Failed to load|warn|Loading|it/s\]|example/s|UNEXPECTED|Notes:|can be ignored|FutureWarning'
echo "== r199 done"
