#!/bin/bash
# PLAN row 41 (REAL-9): the 24 shots chosen per query by a rule (ai_experiments.real6_shots: recent, nearest, transact, cluster)
# instead of the frozen stratified block, on the LLM arms. Everything QLoRA on the 4-bit base and scored on it (LOAD_4BIT=1).
#   H  the untrained instruct base with the rule's shots, without and with the record
#   I  the fixed-shot adapters of section 38 (seed 0: no DB, record in prompt) read with the rule's shots at test only
#   J  adapters trained with the rule's shots (SHOTS=<rule> in exp_categoriser.py; 24 shots per training query by the same rule
#      over the DB_ONLY-free history), scored with the rule's shots
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs
AD=models/adapters; CAT=categoriser_Qwen2.5-3B-Instruct; INS=Qwen/Qwen2.5-3B-Instruct; G=logs/gpu41
export LOAD_4BIT=1
run() { log=$1; shift; echo "start $log $(date +%H:%M)" >> $G.log; "$@" > ${G}_$log.log 2>&1; rc=$?; echo "end $log $(date +%H:%M) exit $rc $(grep -i 'error\|traceback' ${G}_$log.log | grep -vc torchao) errors" >> $G.log; }  # rc first: a $? after $(date) reads date's status

for r in recent nearest transact cluster; do
  run H_base_$r env SHOTS=$r CONDS=noctx,ctx MODEL=$INS uv run python scripts/exp_real6.py llm base
done
for r in recent nearest transact cluster; do
  run I_none_$r env SHOTS=$r CONDS=noctx uv run python scripts/exp_real6.py llm ${CAT}_none_lora
  run I_ret_$r env SHOTS=$r CONDS=ctx uv run python scripts/exp_real6.py llm ${CAT}_ret_lora
done
for r in recent nearest transact cluster; do
  A=${CAT}_none_shots${r}_lora
  [ -d $AD/$A ] || run J_train_none_$r env SHOTS=$r uv run python scripts/exp_categoriser.py llm none
  [ -d $AD/$A ] && run J_score_none_$r env CONDS=noctx uv run python scripts/exp_real6.py llm $A
  A=${CAT}_ret_shots${r}_lora
  [ -d $AD/$A ] || run J_train_ret_$r env SHOTS=$r uv run python scripts/exp_categoriser.py llm ret
  [ -d $AD/$A ] && run J_score_ret_$r env CONDS=ctx uv run python scripts/exp_real6.py llm $A
done
echo "CHAIN DONE $(date +%H:%M)" >> $G.log
