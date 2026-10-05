#!/bin/bash
# Row 202 on decider's own items (owner, 2026-10-05: "Can we re-score this row to get a better comparison?"): scripts/hist_agree.py per
# encoder on 100 held-out v4 households (decider G4 r190 seeds 0 and 1 on realstyle_v4g_ngram_test), then the owner's budget for the old
# encoder and b5 (chosen on the 50-household read: best on first-time payees). Waits for the b3b run (one GPU job at a time; RAM).
set -eu
cd "$(dirname "$0")/../.."
while pgrep -f "chains/r202_b3b.sh" >/dev/null; do sleep 20; done
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 TEST_SEEDS=100000-100099
F='Failed to load|warn|Loading|it/s\]|example/s|FutureWarning|Bytecode'
echo "start $(date -u +%H:%M:%S)"
for e in hist_knn_v1 hist_r202_b0 hist_r202_b1 hist_r202_b2 hist_r202_b3 hist_r202_b3b hist_r202_b4 hist_r202_b5; do
  echo "=== $e synthetic"; OUT1=$e uv run python scripts/hist_agree.py 2>&1 | grep --line-buffered -vE "$F"
done
for e in hist_knn_v1 hist_r202_b5; do
  echo "=== $e owner"; OUT1=$e BUDGET=96c06c41-f26f-4a44-98ed-eaba2f471e1e uv run python scripts/hist_agree.py 2>&1 | grep --line-buffered -vE "$F"
done
echo "== r202 agree done $(date -u +%H:%M:%S)"
