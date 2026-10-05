#!/bin/bash
# Row 194: history-aware encoders (scripts/hist_encoder.py): train approach 1 (transaction <-> transaction; also read as 2, prototypes),
# then approach 3 (query with its 5 nearest earlier transactions), on the local GPU; read on 50 held-out households and on the owner's
# budget (private; aggregates only), fused with decider G4 (r190, seeds 0 and 1).
set -eu
cd "$(dirname "$0")/../.."
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 BUDGET=96c06c41-f26f-4a44-98ed-eaba2f471e1e
F='Failed to load|warn|Loading|it/s\]|example/s'
uv run python scripts/hist_encoder.py train1 2>&1 | grep -vE "$F" | tail -3
uv run python scripts/hist_encoder.py train3 2>&1 | grep -vE "$F" | tail -3
uv run python scripts/hist_encoder.py read 2>&1 | grep -vE "$F"
READ=budget FUSE=r190-g-s0,r190-g-s1 uv run python scripts/hist_encoder.py read 2>&1 | grep -vE "$F"
echo "== r194 done"
