#!/bin/bash
# Rows 194-195 fusion re-read on the owner's budget (private; aggregates only): the first reads matched decider's options against the
# wrong items file (plain names against "Group: Name" labels; 11 of 163 options matched), so every fused row was invalid.
set -eu
cd "$(dirname "$0")/../.."
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 BUDGET=96c06c41-f26f-4a44-98ed-eaba2f471e1e READ=budget FUSE=r190-g-s0,r190-g-s1
F='Failed to load|warn|Loading|it/s\]|example/s|UNEXPECTED|Notes:|can be ignored|FutureWarning'
uv run python scripts/hist_encoder.py read 2>&1 | grep --line-buffered -vE "$F"
uv run python scripts/hist_encoder2.py read 2>&1 | grep --line-buffered -vE "$F"
echo "== r195 fuse done"
