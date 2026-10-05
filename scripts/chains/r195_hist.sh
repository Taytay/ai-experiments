#!/bin/bash
# Row 195: more history-aware readers (scripts/hist_encoder2.py): TransAct-style query history (5 most recent + 5 nearest, with ages),
# transaction-level MaxSim, token-level ColBERT; after row 194 (needs its encoder 1); local GPU; held-out households and the owner's
# budget (private; aggregates only), fused with decider G4 (r190, seeds 0 and 1).
set -eu
cd "$(dirname "$0")/../.."
until grep -q "r194 done" logs/r194_hist.log 2>/dev/null; do sleep 60; done
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 BUDGET=96c06c41-f26f-4a44-98ed-eaba2f471e1e
F='Failed to load|warn|Loading|it/s\]|example/s|UNEXPECTED|Notes:|can be ignored'
uv run python scripts/hist_encoder2.py train4 2>&1 | grep -vE "$F" | tail -3
uv run python scripts/hist_encoder2.py train6 2>&1 | grep -vE "$F" | tail -3
uv run python scripts/hist_encoder2.py read 2>&1 | grep -vE "$F"
READ=budget FUSE=r190-g-s0,r190-g-s1 uv run python scripts/hist_encoder2.py read 2>&1 | grep -vE "$F"
echo "== r195 done"
