#!/bin/bash
# PLAN row 37 (REAL-7): A the ambiguous fact DB, B the retrieved record end to end, C seeds 1 and 2 of the SFT arms.
# Resumed on 2026-09-20 after the owner's stop: A1 (train ret_amb) and A2 (score it, ctx + ret1) finished on 2026-09-19 and are commented out.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs
AD=models/adapters; CAT=categoriser_Qwen2.5-3B-Instruct; INS=Qwen/Qwen2.5-3B-Instruct; G=logs/gpu37
run() { log=$1; shift; echo "start $log $(date +%H:%M)" >> $G.log; "$@" > ${G}_$log.log 2>&1; echo "end $log $(date +%H:%M) exit $? $(grep -ci 'error\|traceback' ${G}_$log.log) errors" >> $G.log; }
# A: the ambiguous DB
# run A1_train_ret_amb env REAL6_DB=amb uv run python scripts/exp_categoriser.py llm ret
# [ -d $AD/${CAT}_ret_amb_lora ] && run A2_score_ret_amb env REAL6_DB=amb CONDS=ctx,ret1 uv run python scripts/exp_real6.py llm ${CAT}_ret_amb_lora
run A3_score_ret_on_amb env REAL6_DB=amb CONDS=ctx uv run python scripts/exp_real6.py llm ${CAT}_ret_lora
run A4_score_base_amb env REAL6_DB=amb CONDS=ctx MODEL=$INS uv run python scripts/exp_real6.py llm base
run A5_enc_amb env REAL6_DB=amb ENC_CTX=1 uv run python scripts/exp_real6.py encoder categoriser_bge_none
run A6_train_param_x2_amb env REAL6_DB=amb STEPS=400 DB_FRAC=0.5 RUN_TAG=x2 uv run python scripts/exp_categoriser.py llm param
[ -d $AD/${CAT}_param_x2_amb_lora ] && run A7_score_param_x2_amb env CONDS=noctx uv run python scripts/exp_real6.py llm ${CAT}_param_x2_amb_lora
[ -d $AD/${CAT}_param_x2_amb_lora ] && run A8_items2_param_x2_amb env MODEL=$INS uv run python scripts/exp_items_v2.py ${CAT}_param_x2_amb_lora
# B: the retrieved record end to end (disjoint DB)
run B1_ret1_sft_ret env CONDS=ret1 uv run python scripts/exp_real6.py llm ${CAT}_ret_lora
run B2_ret1_base env CONDS=ret1 MODEL=$INS uv run python scripts/exp_real6.py llm base
run B3_enc_ret1 env ENC_CTX=ret uv run python scripts/exp_real6.py encoder categoriser_bge_none
# C: seeds 1 and 2 of the three SFT arms
for s in 1 2; do
  run C_none_s$s env SEED=$s RUN_TAG=s$s uv run python scripts/exp_categoriser.py llm none
  [ -d $AD/${CAT}_none_s${s}_lora ] && run C_none_s${s}_score env CONDS=noctx uv run python scripts/exp_real6.py llm ${CAT}_none_s${s}_lora
  run C_ret_s$s env SEED=$s RUN_TAG=s$s uv run python scripts/exp_categoriser.py llm ret
  [ -d $AD/${CAT}_ret_s${s}_lora ] && run C_ret_s${s}_score env CONDS=ctx uv run python scripts/exp_real6.py llm ${CAT}_ret_s${s}_lora
  run C_x2_s$s env SEED=$s STEPS=400 DB_FRAC=0.5 RUN_TAG=x2_s$s uv run python scripts/exp_categoriser.py llm param
  [ -d $AD/${CAT}_param_x2_s${s}_lora ] && run C_x2_s${s}_score env CONDS=noctx uv run python scripts/exp_real6.py llm ${CAT}_param_x2_s${s}_lora
done
echo "CHAIN DONE $(date +%H:%M)" >> $G.log
