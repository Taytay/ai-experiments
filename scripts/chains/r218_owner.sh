#!/bin/bash
# Row 218 finalists on the owner's budget, once, before choosing row 222's recipe (owner's budget disagreed with the synthetic sets on
# row 217): decider's items (hist_agree OWNER_ONLY, SPAN_AFTER=365). ENCS from the environment (o1, mdo, m0 first; m1-m3 when read).
set -eu
cd "$(dirname "$0")/../.."
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 OPTS=span RCHUNK=32 MCHUNK=32
F='Failed to load|warn|Loading|it/s\]|example/s|FutureWarning|Bytecode|UNEXPECTED|position_ids|Notes|LOAD REPORT|^Key|^---'
echo "start $(date -u +%H:%M:%S) ($ENCS)"
OWNER_ONLY=1 BUDGET=96c06c41-f26f-4a44-98ed-eaba2f471e1e uv run ${UVWITH:-} python scripts/hist_agree.py 2>&1 | grep --line-buffered -vE "$F"
echo "== r218 owner done $(date -u +%H:%M:%S)"
