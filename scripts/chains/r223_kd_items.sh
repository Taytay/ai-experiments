#!/bin/bash
# Row 223 (distilling decider into the late-interaction model; ColBERT-Zero 2602.16609 and LITE 2406.17968: supervised contrastive then KD):
# decider items on 300 training-world households decider G4 never trained on (it trained on seeds 0-399; row 192's "seen" sets use
# 300000-300099), seeds 301000-301299, 100 items each, the G4 layout (n-gram similar rows, no crowd line), for decider G4 to score as the
# teacher. Local CPU, once.
set -eu
cd "$(dirname "$0")/../.."
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 CROWD_KEY=v2 TEST_LAYOUT=split SIM=2
echo "start $(date -u +%H:%M:%S)"
TEST_SPLIT=train TEST_SEED0=301000 SIM_SRC=ngram CROWD_CLUS=realstyle_crowdclus_v4_train.json TEST_OUT=realstyle_v4g_kd_train.json \
  uv run python scripts/build_realstyle.py test 300 100 | tail -2
echo "== r223 items done $(date -u +%H:%M:%S)"
