#!/bin/bash
# PLAN row 47 (REAL-8 amended by REAL-13): is the 200-step no-DB categoriser under-trained? The no-DB SFT at 400 steps (the history
# exposure of section 45's 6.7-pass parametric arm: 800 steps at 50%), 800 steps (the 13.3-pass arm's; three seeds) and 1,600 steps,
# scored on REAL-6 without a record (read in REPORT.md 48's corrected groups) and on the ARC / MMLU / ICL items (exp_items_v2).
# QLoRA on the 4-bit base and scored on it (LOAD_4BIT=1), as sections 37 to 48.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs
AD=models/adapters; CAT=categoriser_Qwen2.5-3B-Instruct; INS=Qwen/Qwen2.5-3B-Instruct; G=logs/gpu47
export LOAD_4BIT=1
run() { log=$1; shift; echo "start $log $(date +%H:%M)" >> $G.log; "$@" > ${G}_$log.log 2>&1; rc=$?; echo "end $log $(date +%H:%M) exit $rc $(grep -i 'error\|traceback' ${G}_$log.log | grep -vc torchao) errors" >> $G.log; }  # rc first: a $? after $(date) reads date's status

for spec in 800:0 400:0 800:1 800:2 1600:0; do
  steps=${spec%:*}; s=${spec#*:}
  A=${CAT}_none_st${steps}_s${s}_lora
  [ -d $AD/$A ] || run K_train_st${steps}_s$s env SEED=$s STEPS=$steps RUN_TAG=st${steps}_s$s uv run python scripts/exp_categoriser.py llm none
  [ -d $AD/$A ] && run K_score_st${steps}_s$s env CONDS=noctx uv run python scripts/exp_real6.py llm $A
  [ -d $AD/$A ] && run K_items2_st${steps}_s$s env MODEL=$INS uv run python scripts/exp_items_v2.py $A
done
echo "CHAIN DONE $(date +%H:%M)" >> $G.log
