#!/bin/bash
# 2026-09-25, owner: "I trust your judgement. The GPU is available." Row 42's rename question asked on the all-label recipe of row 56,
# the record arm's rename folds, the answer-weighted all-label loss, and seeds for the all-label recipe on 4-bit and bf16. From MAIN.
#   A  no-DB, all-label + rename augmentation (RENAME=0.5), four held-out folds, 200 steps, 4-bit
#   B  record arm, rename augmentation, four held-out folds, 200 steps, 4-bit (row 42's planned record folds)
#   C  no-DB, all-label with the final answer weighted to half of each sequence's loss (ANS_WEIGHT=0.5), four held-out folds
#   D  all-label, all 20 users, 200 steps, seeds 1 and 2, on 4-bit and on bf16 (seed 0 of each is in part 1)
#   E  the one old-recipe rename adapter already trained (none_st800_f0_ren50) scored on its five users
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs
AD=models/adapters; CAT=categoriser_Qwen2.5-3B-Instruct; INS=Qwen/Qwen2.5-3B-Instruct; G=logs/gpu56d
run() { log=$1; shift; echo "start $log $(date +%H:%M)" >> $G.log; "$@" > ${G}_$log.log 2>&1; rc=$?; echo "end $log $(date +%H:%M) exit $rc $(grep -i 'error\|traceback' ${G}_$log.log | grep -vc torchao) errors" >> $G.log; }  # rc first: a $? after $(date) reads date's status
export LOAD_4BIT=1

run S_smoke_aw env SMOKE=1 ALL_LABELS=1 ANS_WEIGHT=0.5 uv run python scripts/exp_categoriser.py llm none
run S_smoke_ren_alllab env SMOKE=1 FOLD=0 RENAME=0.5 ALL_LABELS=1 uv run python scripts/exp_categoriser.py llm none
grep -q "S_smoke_aw .* exit 0 0 errors" $G.log && grep -q "S_smoke_ren_alllab .* exit 0 0 errors" $G.log || { echo "SMOKE FAILED, stopping $(date +%H:%M)" >> $G.log; exit 1; }

for f in 0 1 2 3; do
  A=${CAT}_none_f${f}_ren50_alllab_lora
  [ -d $AD/$A ] || run A_train_f$f env FOLD=$f RENAME=0.5 ALL_LABELS=1 STEPS=200 uv run python scripts/exp_categoriser.py llm none
  [ -d $AD/$A ] && [ ! -f results/real6_$A.json ] && run A_score_f$f env CONDS=noctx uv run python scripts/exp_real6.py llm $A
done
for f in 0 1 2 3; do
  A=${CAT}_ret_f${f}_ren50_lora
  [ -d $AD/$A ] || run B_train_f$f env FOLD=$f RENAME=0.5 STEPS=200 uv run python scripts/exp_categoriser.py llm ret
  [ -d $AD/$A ] && [ ! -f results/real6_$A.json ] && run B_score_f$f env CONDS=ctx uv run python scripts/exp_real6.py llm $A
done
for f in 0 1 2 3; do
  A=${CAT}_none_f${f}_alllab_aw50_lora
  [ -d $AD/$A ] || run C_train_f$f env FOLD=$f ALL_LABELS=1 ANS_WEIGHT=0.5 STEPS=200 uv run python scripts/exp_categoriser.py llm none
  [ -d $AD/$A ] && [ ! -f results/real6_$A.json ] && run C_score_f$f env CONDS=noctx uv run python scripts/exp_real6.py llm $A
done
# spec: load_in_4bit : run_tag : seed
for spec in 1:s1:1 1:s2:2 0:bf16_s1:1 0:bf16_s2:2; do
  IFS=: read -r q4 tag seed <<< "$spec"
  A=${CAT}_none_${tag}_alllab_lora
  [ -d $AD/$A ] || run D_train_$tag env LOAD_4BIT=$q4 SEED=$seed ALL_LABELS=1 STEPS=200 RUN_TAG=$tag uv run python scripts/exp_categoriser.py llm none
  [ -d $AD/$A ] && [ ! -f results/real6_$A.json ] && run D_score_$tag env LOAD_4BIT=$q4 CONDS=noctx uv run python scripts/exp_real6.py llm $A
  [ -d $AD/$A ] && [ ! -f results/items2_$A.json ] && run D_items2_$tag env LOAD_4BIT=$q4 MODEL=$INS uv run python scripts/exp_items_v2.py $A
done
A=${CAT}_none_st800_f0_ren50_lora
[ -f results/real6_$A.json ] || run E_score_old_ren_f0 env CONDS=noctx uv run python scripts/exp_real6.py llm $A
echo "CHAIN DONE $(date +%H:%M)" >> $G.log
