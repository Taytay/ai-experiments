#!/bin/bash
# Owner, 2026-09-23: settle the precision question (4-bit vs bf16) before row 42's rename folds finish. Order:
#   1. wait for the row 42 training step that was running when chain_r42.sh was stopped (its adapter is kept; the chain skips it)
#   2. row 56 part 1 from the worktree: smokes, bf16 200 and 800 steps (seed 0), the all-label runs
#   3. row 42's chain resumed from the main checkout (skips finished adapters, scores the paused fold)
#   4. row 56 part 2 from the worktree: bf16 800 steps seeds 1 and 2, the record arm at bf16 200 steps seed 0
WT=/home/taytay/projects/Taytay/ai-experiments-wt06; MAIN=/home/taytay/projects/Taytay/ai-experiments
while kill -0 "$1" 2>/dev/null; do sleep 30; done   # $1: pid of the running row 42 python
cd $WT && uv run evals rebuild > logs/rebuild56.log 2>&1
$WT/scripts/chains/chain_r56.sh
echo "resumed after row 56 part 1 $(date +%H:%M)" >> $MAIN/logs/gpu42.log
$MAIN/scripts/chains/chain_r42.sh
SPECS="bf16st800s1:800:0:0:st800_bf16_s1:1 bf16st800s2:800:0:0:st800_bf16_s2:2" $WT/scripts/chains/chain_r56.sh
cd $WT && env LOAD_4BIT=0 RUN_TAG=bf16 bash -c 'echo "start R_ret_bf16 $(date +%H:%M)" >> logs/gpu56.log; uv run python scripts/exp_categoriser.py llm ret > logs/gpu56_R_train_ret_bf16.log 2>&1; LOAD_4BIT=0 CONDS=ctx uv run python scripts/exp_real6.py llm categoriser_Qwen2.5-3B-Instruct_ret_bf16_lora > logs/gpu56_R_score_ret_bf16.log 2>&1; echo "end R_ret_bf16 $(date +%H:%M) SEQUENCE DONE" >> logs/gpu56.log'
