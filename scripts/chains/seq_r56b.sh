#!/bin/bash
# 2026-09-23 17:40, after row 56 part 1: the all-label loss beat 1,600 plain steps in 100, so its held-out-user check (part 3) goes
# first; row 42's rename folds (old recipe) and the bf16 seeds follow. $1: pid of the re-score running when the order changed.
WT=/home/taytay/projects/Taytay/ai-experiments-wt06; MAIN=/home/taytay/projects/Taytay/ai-experiments
while kill -0 "$1" 2>/dev/null; do sleep 20; done
NOWAIT=1 $WT/scripts/chains/chain_r56c.sh
echo "resumed after row 56 part 3 $(date +%H:%M)" >> $MAIN/logs/gpu42.log
$MAIN/scripts/chains/chain_r42.sh
SPECS="bf16st800s1:800:0:0:st800_bf16_s1:1 bf16st800s2:800:0:0:st800_bf16_s2:2" $WT/scripts/chains/chain_r56.sh
cd $WT && env LOAD_4BIT=0 RUN_TAG=bf16 bash -c 'echo "start R_ret_bf16 $(date +%H:%M)" >> logs/gpu56.log; uv run python scripts/exp_categoriser.py llm ret > logs/gpu56_R_train_ret_bf16.log 2>&1; LOAD_4BIT=0 CONDS=ctx uv run python scripts/exp_real6.py llm categoriser_Qwen2.5-3B-Instruct_ret_bf16_lora > logs/gpu56_R_score_ret_bf16.log 2>&1; echo "end R_ret_bf16 $(date +%H:%M) SEQUENCE DONE" >> logs/gpu56.log'
