#!/bin/bash
# Row 217 on the owner's budget (owner, 2026-10-06: "Can we use this best model on my budget before we start moving on?"): p2 (the best
# arm on synthetic and blind_v2), with c0 and a5 re-scored under the corrected options (OPTS=span; row 216's 60.9 / 64.6 could not score
# the 7% of items filed to categories hidden today), on decider's own items (hist_agree, OWNER_ONLY=1, SPAN_AFTER=365 so decider's
# options are a subset); then p2 on every transaction of the budget with its suggestion lists. Local GPU, private: aggregates only.
set -eu
cd "$(dirname "$0")/../.."
while pgrep -f "chains/r217_recent[.]sh" >/dev/null; do sleep 20; done
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 OPTS=span RCHUNK=32
F='Failed to load|warn|Loading|it/s\]|example/s|FutureWarning|Bytecode|UNEXPECTED|position_ids|Notes|LOAD REPORT|^Key|^---'
B=96c06c41-f26f-4a44-98ed-eaba2f471e1e
echo "start $(date -u +%H:%M:%S)"
OWNER_ONLY=1 ENCS=li_r217_p2,li_r211_c0,li_r210_a5 BUDGET=$B uv run python scripts/hist_agree.py 2>&1 | grep --line-buffered -vE "$F"
BUDGET=$B READ=budget ARMS=li_r217_p2 uv run python scripts/li_decider.py read 2>&1 | grep --line-buffered -vE "$F"
echo "== r217 owner done $(date -u +%H:%M:%S)"
