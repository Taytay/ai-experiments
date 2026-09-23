#!/bin/bash
# PLAN row 56 (TRAIN-11): training efficiency of the no-DB categoriser. All runs seed 0, scored on all 1,179 REAL-6 items without a
# record and on the ARC / MMLU / ICL items, each adapter on the base precision it was trained on (REPORT.md 44).
#   M1  plain recipe, 200 steps, bf16 base (LOAD_4BIT=0)          -> the precision comparison against row 47's 4-bit 200-step s0
#   M2  all-label loss, 100 / 200 / 400 steps, 4-bit base          -> against row 47's 4-bit curve (200 / 400 / 800 / 1,600)
#   M3  all-label loss, 200 steps, bf16 base                       -> both changes together
# Runs from the worktree (its own models/adapters and evals db) while the main checkout holds row 42's uncommitted results.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs
AD=models/adapters; CAT=categoriser_Qwen2.5-3B-Instruct; INS=Qwen/Qwen2.5-3B-Instruct; G=logs/gpu56
run() { log=$1; shift; echo "start $log $(date +%H:%M)" >> $G.log; "$@" > ${G}_$log.log 2>&1; rc=$?; echo "end $log $(date +%H:%M) exit $rc $(grep -i 'error\|traceback' ${G}_$log.log | grep -vc torchao) errors" >> $G.log; }  # rc first: a $? after $(date) reads date's status

run S_smoke_alllab env SMOKE=1 LOAD_4BIT=1 ALL_LABELS=1 uv run python scripts/exp_categoriser.py llm none  # first GPU use of both paths
run S_smoke_bf16 env SMOKE=1 LOAD_4BIT=0 RUN_TAG=bf16 uv run python scripts/exp_categoriser.py llm none

# spec: name : steps : all_labels : load_in_4bit : run_tag
for spec in bf16plain:200:0:0:bf16 alllab100:100:1:1:st100 alllab200:200:1:1: alllab400:400:1:1:st400 bf16alllab:200:1:0:bf16; do
  IFS=: read -r name steps al q4 tag <<< "$spec"
  A=${CAT}_none${tag:+_$tag}$([ $al = 1 ] && echo _alllab)_lora
  [ -d $AD/$A ] || run M_train_$name env LOAD_4BIT=$q4 ALL_LABELS=$al STEPS=$steps RUN_TAG=$tag uv run python scripts/exp_categoriser.py llm none
  [ -d $AD/$A ] && run M_score_$name env LOAD_4BIT=$q4 CONDS=noctx uv run python scripts/exp_real6.py llm $A
  [ -d $AD/$A ] && run M_items2_$name env LOAD_4BIT=$q4 MODEL=$INS uv run python scripts/exp_items_v2.py $A
done
echo "CHAIN DONE $(date +%H:%M)" >> $G.log
