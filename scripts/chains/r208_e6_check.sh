#!/bin/bash
# Row 208 E6 check (local 3090): the same 3 training steps (MICRO 4, seed 0) as one batch and as 2 length-sorted groups; the step losses
# must match up to float order. Adapters go to throwaway names (RUN_TAG e6check*) and are deleted afterwards; tracker rows are reverted.
set -eu
cd "$(dirname "$0")/../.."
UV='uv run --frozen --with transformers==5.17.0 --with flash-linear-attention --with peft>=0.21 --with torch==2.13.0 --with torchvision==0.28.0 python'
for S in 1 2; do
  echo "== MICRO_SPLIT=$S $(date -u +%H:%M:%S)"
  env DBEP=0.5 FOLD=0 RENAME=0.5 MICRO=4 STEPS=3 ALL_LABELS=1 AUX_LM=1 LABELS=rand255 MISLEAD=mislead_v1 ALT=0.1 ALT_SOFT=1 LOOKUP=0.1 \
    OVERRIDE=0.1 EVFREE=0.1 EVFREE_MODE=soft MODEL=Mapika/decider-4b EMPTY=20 LAYOUT=labelled_shots SEED=0 RUN_TAG=e6check$S OTHERS=0.5 \
    REALSTYLE=0.25 REALSTYLE_FILE=realstyle_v4g_train.jsonl MICRO_SPLIT=$S $UV scripts/exp_decider_finetune.py 2>&1 | grep -E "step|padded|tokens|Error" | tail -6
done
rm -rf models/adapters/*e6check*
git checkout -- evals/runs.jsonl 2>/dev/null || true
echo "== r208 e6 check done $(date -u +%H:%M:%S)"
