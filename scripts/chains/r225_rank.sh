#!/bin/bash
# Row 225: do v5 households (RS_V5=1, scripts/realness_gap.py) rank the late-interaction arms as the owner's budget did? One read process,
# all arms (households and neighbours built once). Owner's budget (decider's items, OPTS=span): c0 63.3, a5 66.8, p2 59.5, o1 66.9,
# mdo 67.7, m0 66.7, e0 68.6, ekmv 68.9 / 68.2, kd0 67.8, h15 69.0 / 68.8.
set -eu
cd "$(dirname "$0")/../.."
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 RS_V5=1 OPTS=span RCHUNK=32 TEST_SEEDS=${TEST_SEEDS:-100000-100009}
A=${ARMS:-li_r217_p2,li_r211_c0,li_r210_a5,li_r218_o1,li_r218_m0,li_r223_kd0,li_r222_e0,li_r222_ekmv,li_r221_h15,li_r218_mdo}
F='Failed to load|warn|Loading|it/s\]|example/s|FutureWarning|Bytecode|UNEXPECTED|position_ids|Notes|LOAD REPORT|^Key|^---'
echo "start $(date -u +%H:%M:%S) ($A)"
ARMS=$A uv run python scripts/li_decider.py read 2>&1 | grep --line-buffered -vE "$F"
echo "== r225 rank done $(date -u +%H:%M:%S)"
