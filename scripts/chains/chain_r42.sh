#!/bin/bash
# PLAN row 42 (REAL-10): held-out users. Four folds of five users (fold = user id mod 4); for each fold the no-DB and the
# record-in-prompt SFT trained on the other fifteen users, plain and with rename augmentation (RENAME=0.5: each category name
# replaced by a fresh coined word with probability 0.5 per episode), scored on the fold's five users only (exp_real6.py reads
# the fold from the adapter name). Every item is then scored by an adapter that never saw its user. QLoRA on the 4-bit base.
# ST (training steps) is set from row 47's answer; default the section 38 recipe's 200.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs
AD=models/adapters; CAT=categoriser_Qwen2.5-3B-Instruct; G=logs/gpu42; ST=${ST:-200}
export LOAD_4BIT=1
run() { log=$1; shift; echo "start $log $(date +%H:%M)" >> $G.log; "$@" > ${G}_$log.log 2>&1; rc=$?; echo "end $log $(date +%H:%M) exit $rc $(grep -i 'error\|traceback' ${G}_$log.log | grep -vc torchao) errors" >> $G.log; }  # rc first: a $? after $(date) reads date's status
TAG=$([ "$ST" = 200 ] && echo "" || echo "_st$ST")

for f in 0 1 2 3; do
  for db in none ret; do
    cond=$([ $db = none ] && echo noctx || echo ctx)
    for ren in 0 0.5; do
      sfx=$([ $ren = 0 ] && echo "" || echo "_ren50")
      A=${CAT}_${db}${TAG}_f${f}${sfx}_lora
      [ -d $AD/$A ] || run L_train_${db}_f${f}${sfx} env FOLD=$f RENAME=$ren STEPS=$ST RUN_TAG=${TAG#_} uv run python scripts/exp_categoriser.py llm $db
      [ -d $AD/$A ] && run L_score_${db}_f${f}${sfx} env CONDS=$cond uv run python scripts/exp_real6.py llm $A
    done
  done
done
echo "CHAIN DONE $(date +%H:%M)" >> $G.log
