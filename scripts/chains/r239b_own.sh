#!/usr/bin/env bash
# Row 239 (b): owner's-budget reads for the blend, locally (the synthetic reads run on Modal, r239b.json)
cd "$(dirname "$0")/../.."
env SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 RS_V5=1 OPTS=span PARTS=own BUDGET="$1" \
  uv run --with "sentence-transformers>=6.1.0" --with "transformers>=5.18" --with torch==2.13.0 --with torchvision==0.28.0 \
  python scripts/blend_eval.py > logs/r239b_own.log 2>&1
grep -E "^li_|Traceback|Error" logs/r239b_own.log | cut -c1-200
