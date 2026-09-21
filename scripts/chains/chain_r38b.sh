#!/bin/bash
# PLAN row 38 (INFRA-2), second chain: after scripts/check_unsloth_base.py showed that unsloth's loader defaults to the NF4 4-bit base
# (load_in_4bit=True), the bf16 numbers the row needs: the peft-trained record-in-prompt adapter under the bf16 scorer, and the
# instruct base itself at bf16 on REAL-6 (against the 4-bit base of section 37).
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs
AD=models/adapters; CAT=categoriser_Qwen2.5-3B-Instruct; INS=Qwen/Qwen2.5-3B-Instruct; G=logs/gpu38b
run() { log=$1; shift; echo "start $log $(date +%H:%M)" >> $G.log; "$@" > ${G}_$log.log 2>&1; echo "end $log $(date +%H:%M) exit $? $(grep -ci 'error\|traceback' ${G}_$log.log) errors" >> $G.log; }
[ -d $AD/${CAT}_ret_hf_lora ] && run D7_hf_ret_hfscore env CONDS=ctx SCORER=hf uv run python scripts/exp_real6.py llm ${CAT}_ret_hf_lora
run D8_base_bf16 env CONDS=noctx,ctx SCORER=hf MODEL=$INS uv run python scripts/exp_real6.py llm base
echo "CHAIN DONE $(date +%H:%M)" >> $G.log
