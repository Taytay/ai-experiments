#!/bin/bash
# Row 231 (owner, 2026-10-06: "Were we training decider on v5, or some other dataset?"): decider G4's real-style share was v4 households
# (realstyle_v4g_train.jsonl, r190_data.sh); the encoder gained +1.5 on the owner's budget from v5 households (§197). The same episodes
# from v5 households (RS_V5=1), same counts and layout, for decider at matched effort.
set -eu
cd "$(dirname "$0")/../.."
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 RS_V5=1 CROWD_KEY=v2 TEST_LAYOUT=split
echo "start $(date -u +%H:%M:%S)"
TRAIN_OUT=realstyle_v5g_train.jsonl uv run python scripts/build_realstyle.py train 400 100 | tail -1
echo "== r231 data done $(date -u +%H:%M:%S)"
