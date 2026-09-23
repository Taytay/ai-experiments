#!/bin/bash
# Row 56 part 3 (TRAIN-11): does the all-label loss's gain survive held-out users? Row 42's four folds (user id mod 4) with ALL_LABELS=1
# at 200 steps on the 4-bit base, each scored on its five held-out users; compared with row 42's plain folds at 800 steps (69.1 on the
# first 15 users) and the all-label adapter trained on all 20 (80.4). Starts when seq_r56.sh writes SEQUENCE DONE. From the worktree.
cd "$(dirname "$0")/../.." || exit 1
AD=models/adapters; CAT=categoriser_Qwen2.5-3B-Instruct; G=logs/gpu56
export LOAD_4BIT=1
run() { log=$1; shift; echo "start $log $(date +%H:%M)" >> $G.log; "$@" > ${G}_$log.log 2>&1; rc=$?; echo "end $log $(date +%H:%M) exit $rc $(grep -i 'error\|traceback' ${G}_$log.log | grep -vc torchao) errors" >> $G.log; }
while ! grep -q 'SEQUENCE DONE' $G.log 2>/dev/null; do sleep 60; done
for f in 0 1 2 3; do
  A=${CAT}_none_f${f}_alllab_lora
  [ -d $AD/$A ] || run H_train_alllab_f$f env FOLD=$f ALL_LABELS=1 STEPS=200 uv run python scripts/exp_categoriser.py llm none
  [ -d $AD/$A ] && run H_score_alllab_f$f env CONDS=noctx uv run python scripts/exp_real6.py llm $A
done
echo "CHAIN DONE (part 3) $(date +%H:%M)" >> $G.log
