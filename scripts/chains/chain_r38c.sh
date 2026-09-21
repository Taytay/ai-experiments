#!/bin/bash
# PLAN row 38 (INFRA-2), third chain: how much the 4-bit default moved the back-filled scores of bf16-trained adapters. The same
# scripts, now defaulting to bf16 (LOAD_4BIT=0), on the base and arm C: items2 (section 33's ARC / MMLU / induction reads),
# graph4 (section 42), the LRE probe (section 40). Waits for chain_r38b. Results land beside the 4-bit ones under a _bf16 tag.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs
while ! grep -q 'CHAIN DONE' logs/gpu38b.log 2>/dev/null; do sleep 120; done
C=curriculum_Qwen2.5-3B_C_p200_lora; G=logs/gpu38c
run() { log=$1; shift; echo "start $log $(date +%H:%M)" >> $G.log; "$@" > ${G}_$log.log 2>&1; echo "end $log $(date +%H:%M) exit $? $(grep -ci 'error\|traceback' ${G}_$log.log) errors" >> $G.log; }
run E1_items2_base_bf16 env RUN_TAG=bf16 uv run python scripts/exp_items_v2.py base
run E2_items2_C_bf16 env RUN_TAG=bf16 uv run python scripts/exp_items_v2.py $C
run E3_graph4_C_bf16 env RUN_TAG=bf16 uv run python scripts/exp_graph4.py $C
run E4_graph4_base_bf16 env RUN_TAG=bf16 uv run python scripts/exp_graph4.py base
run E5_lre_C_bf16 env RUN_TAG=bf16 FORMAT=trained uv run python scripts/exp_lre.py $C
echo "CHAIN DONE $(date +%H:%M)" >> $G.log
