#!/bin/bash
# PLAN row 39 (REAL-8): the parametric exposure curve and the chat template. Everything at the precision of sections 37 to 43
# (LOAD_4BIT=1: QLoRA on the NF4 base, scored on it), so the numbers sit beside Tables 38.1, 43.4 and 44.1.
#   F block, the chat template (real6.chat_prompt: the prompt up to the query line as the user turn, "Category:" opening the assistant
#     turn): the instruct base with and without the record, then the no-DB and record-in-prompt categorisers trained and scored in it.
#   G block, the exposure curve: parametric injection at 50% for 800 steps (6.7 passes over the 960 DB texts) and 1,600 steps
#     (13.3 passes), three seeds each, scored on REAL-6 without a record and on the ARC / MMLU / ICL items (exp_items_v2).
#     Section 38's points: 200 steps at 30% (1 pass) and 400 at 50% (3.3 passes; seeds 0 to 2 in section 43).
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs
AD=models/adapters; CAT=categoriser_Qwen2.5-3B-Instruct; INS=Qwen/Qwen2.5-3B-Instruct; G=logs/gpu39
export LOAD_4BIT=1
run() { log=$1; shift; echo "start $log $(date +%H:%M)" >> $G.log; "$@" > ${G}_$log.log 2>&1; echo "end $log $(date +%H:%M) exit $? $(grep -ci 'error\|traceback' ${G}_$log.log) errors" >> $G.log; }

run F1_base_chat env CHAT=1 CONDS=noctx,ctx MODEL=$INS uv run python scripts/exp_real6.py llm base
[ -d $AD/${CAT}_none_chat_lora ] || run F2_train_none_chat env CHAT=1 uv run python scripts/exp_categoriser.py llm none
[ -d $AD/${CAT}_none_chat_lora ] && run F3_score_none_chat env CONDS=noctx uv run python scripts/exp_real6.py llm ${CAT}_none_chat_lora
[ -d $AD/${CAT}_ret_chat_lora ] || run F4_train_ret_chat env CHAT=1 uv run python scripts/exp_categoriser.py llm ret
[ -d $AD/${CAT}_ret_chat_lora ] && run F5_score_ret_chat env CONDS=ctx uv run python scripts/exp_real6.py llm ${CAT}_ret_chat_lora

for x in 4 8; do
  steps=$((200 * x))
  for s in 0 1 2; do
    A=${CAT}_param_x${x}_s${s}_lora
    [ -d $AD/$A ] || run G_train_x${x}_s$s env SEED=$s STEPS=$steps DB_FRAC=0.5 RUN_TAG=x${x}_s$s uv run python scripts/exp_categoriser.py llm param
    [ -d $AD/$A ] && run G_score_x${x}_s$s env CONDS=noctx uv run python scripts/exp_real6.py llm $A
    [ -d $AD/$A ] && run G_items2_x${x}_s$s env MODEL=$INS uv run python scripts/exp_items_v2.py $A
  done
done
echo "CHAIN DONE $(date +%H:%M)" >> $G.log
