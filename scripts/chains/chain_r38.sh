#!/bin/bash
# PLAN row 38 (INFRA-2): the transformers + peft trainer (TRAINER=hf) against the unsloth one, same seed, and both scorers on both
# adapters. Waits for chain_r37 to finish first.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs
while ! grep -q 'CHAIN DONE' logs/gpu37.log 2>/dev/null; do sleep 120; done
AD=models/adapters; CAT=categoriser_Qwen2.5-3B-Instruct; G=logs/gpu38
run() { log=$1; shift; echo "start $log $(date +%H:%M)" >> $G.log; "$@" > ${G}_$log.log 2>&1; echo "end $log $(date +%H:%M) exit $? $(grep -ci 'error\|traceback' ${G}_$log.log) errors" >> $G.log; }
run D1_hf_none env TRAINER=hf uv run python scripts/exp_categoriser.py llm none
[ -d $AD/${CAT}_none_hf_lora ] && run D2_hf_none_score env CONDS=noctx uv run python scripts/exp_real6.py llm ${CAT}_none_hf_lora
[ -d $AD/${CAT}_none_hf_lora ] && run D3_hf_none_hfscore env CONDS=noctx SCORER=hf uv run python scripts/exp_real6.py llm ${CAT}_none_hf_lora
run D4_hfscore_unsloth_none env CONDS=noctx SCORER=hf uv run python scripts/exp_real6.py llm ${CAT}_none_lora
run D5_hf_ret env TRAINER=hf uv run python scripts/exp_categoriser.py llm ret
[ -d $AD/${CAT}_ret_hf_lora ] && run D6_hf_ret_score env CONDS=ctx uv run python scripts/exp_real6.py llm ${CAT}_ret_hf_lora
echo "CHAIN DONE $(date +%H:%M)" >> $G.log
