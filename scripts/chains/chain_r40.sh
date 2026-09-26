#!/bin/bash
# PLAN row 40 (BASE-6): FastFit, GLiClass and a logistic-regression head on the REAL-6 cells (scripts/exp_real6_fewshot.py), one model
# per user from the 24 shots or the whole history, without and with the merchant's record on the statement (CTX=1).
#   A  logistic regression on frozen bge embeddings (minutes), C = 1, 10, 100
#   B  FastFit per user: bge encoder plain and with the record; mpnet (FastFit's own default) plain
#   C  GLiClass untrained (names only / 24 shots as examples), then fine-tuned across users, plain and with the record
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs
AD=models/adapters; G=logs/gpu40
run() { log=$1; shift; echo "start $log $(date +%H:%M)" >> $G.log; "$@" > ${G}_$log.log 2>&1; echo "end $log $(date +%H:%M) exit $? $(grep -ci 'error\|traceback' ${G}_$log.log) errors" >> $G.log; }

run A1_logreg env CTX=0 uv run python scripts/exp_real6_fewshot.py logreg
run A2_logreg_ctx env CTX=1 uv run python scripts/exp_real6_fewshot.py logreg
for c in 10 100; do  # sklearn's C=1 underfits unit-norm embeddings (added after the first pass)
  run A3_logreg_c$c env CTX=0 LOGREG_C=$c uv run python scripts/exp_real6_fewshot.py logreg
  run A4_logreg_c${c}_ctx env CTX=1 LOGREG_C=$c uv run python scripts/exp_real6_fewshot.py logreg
done

run B1_fastfit_bge env CTX=0 ENC=bge uv run python scripts/exp_real6_fewshot.py fastfit
run B2_fastfit_bge_ctx env CTX=1 ENC=bge uv run python scripts/exp_real6_fewshot.py fastfit
run B3_fastfit_mpnet env CTX=0 ENC=mpnet uv run python scripts/exp_real6_fewshot.py fastfit

run C1_gliclass env CTX=0 uv run python scripts/exp_real6_fewshot.py gliclass base
run C2_gliclass_ctx env CTX=1 uv run python scripts/exp_real6_fewshot.py gliclass base
[ -d $AD/gliclass_ft ] || run C3_gliclass_train env CTX=0 uv run python scripts/exp_real6_fewshot.py gliclass train
[ -d $AD/gliclass_ft ] && run C4_gliclass_ft env CTX=0 uv run python scripts/exp_real6_fewshot.py gliclass gliclass_ft
[ -d $AD/gliclass_ft_ctx ] || run C5_gliclass_train_ctx env CTX=1 uv run python scripts/exp_real6_fewshot.py gliclass train
[ -d $AD/gliclass_ft_ctx ] && run C6_gliclass_ft_ctx env CTX=1 uv run python scripts/exp_real6_fewshot.py gliclass gliclass_ft_ctx
echo "CHAIN DONE $(date +%H:%M)" >> $G.log
