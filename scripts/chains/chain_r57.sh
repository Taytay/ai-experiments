#!/bin/bash
# PLAN row 57 (REAL-15): the fact DB as supervised decisions. All-label, 200 steps, 4-bit, row 42's four held-out-user folds; each
# adapter scored without the record on its five held-out users. The no-DB folds (none_f<k>_alllab) are row 56's.
#   E   database episodes: DBEP=0.5
#   P   prose records with the category: DB=param DB_CAT=1 DB_FRAC=0.5
#   PE  both: DB=param DB_CAT=1 DB_FRAC=0.25 DBEP=0.5
# Runs from the worktree (plan-57 code) once chain_r56d.sh in the main checkout writes CHAIN DONE.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs
AD=models/adapters; CAT=categoriser_Qwen2.5-3B-Instruct; G=logs/gpu57; MAIN=/home/taytay/projects/YNAB/ai-experiments
run() { log=$1; shift; echo "start $log $(date +%H:%M)" >> $G.log; "$@" > ${G}_$log.log 2>&1; rc=$?; echo "end $log $(date +%H:%M) exit $rc $(grep -i 'error\|traceback' ${G}_$log.log | grep -vc torchao) errors" >> $G.log; }  # rc first: a $? after $(date) reads date's status
export LOAD_4BIT=1
[ -n "$NOWAIT" ] || while ! grep -q 'CHAIN DONE' $MAIN/logs/gpu56d.log 2>/dev/null; do sleep 60; done
uv run evals rebuild > logs/rebuild57.log 2>&1
run S_smoke_dbep env SMOKE=1 FOLD=0 ALL_LABELS=1 DBEP=0.5 uv run python scripts/exp_categoriser.py llm none
run S_smoke_dbcat env SMOKE=1 FOLD=0 ALL_LABELS=1 DB_CAT=1 DB_FRAC=0.5 uv run python scripts/exp_categoriser.py llm param
grep -q "S_smoke_dbep .* exit 0 0 errors" $G.log && grep -q "S_smoke_dbcat .* exit 0 0 errors" $G.log || { echo "SMOKE FAILED, stopping $(date +%H:%M)" >> $G.log; exit 1; }

# spec: name : db : extra env (comma-separated) : adapter suffix after _f<k>
for spec in E:none:DBEP=0.5:_alllab_dbep50 P:param:DB_CAT=1,DB_FRAC=0.5:_alllab_dbcat PE:param:DB_CAT=1,DB_FRAC=0.25,DBEP=0.5:_alllab_dbep50_dbcat; do
  IFS=: read -r name db extra sfx <<< "$spec"
  for f in 0 1 2 3; do
    A=${CAT}_${db}_f${f}${sfx}_lora
    [ -d $AD/$A ] || run ${name}_train_f$f env FOLD=$f ALL_LABELS=1 STEPS=200 ${extra//,/ } uv run python scripts/exp_categoriser.py llm $db
    [ -d $AD/$A ] && [ ! -f results/real6_$A.json ] && run ${name}_score_f$f env CONDS=noctx uv run python scripts/exp_real6.py llm $A
  done
done
echo "CHAIN DONE $(date +%H:%M)" >> $G.log
