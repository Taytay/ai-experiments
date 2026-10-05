#!/bin/bash
# Row 201: where decider and the history encoders disagree (scripts/hist_agree.py), 100 synthetic households (hist_fast: GPU readers, cached) then the owner's budget
# (private; aggregates only). Local GPU.
set -eu
cd "$(dirname "$0")/../.."
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 BUDGET=96c06c41-f26f-4a44-98ed-eaba2f471e1e TEST_SEEDS=100000-100099
echo "start $(date -u +%H:%M:%S)"; uv run python scripts/hist_agree.py 2>&1 | grep -vE 'Failed to load|warn|Loading|it/s\]|example/s|FutureWarning|Bytecode'
echo "== r201 done $(date -u +%H:%M:%S)"
