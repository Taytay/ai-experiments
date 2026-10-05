#!/bin/bash
# Row 205: the crowd line inside the encoder (scripts/hist_crowd.py; OmniSearchSage's board titles, references/papers/2404.16260).
# Arms, one seed each (screen): crowd = row 202's b1 recipe (infonce + dedup, hist_train2.train) on texts enriched with the top 3 crowd
# category names ("<payee> | $amt | Wd | others: Groceries, Household"; this household left out, CROWD_KEY=v2 v4 tables); ngram = the hash
# n-gram payee tower (unigrams, bigrams, char trigrams -> 2^18 buckets -> EmbeddingBag mean 384-d) on b1's loss and triplets. Read with
# kNN and MaxSim on 50 held-out v4 households beside hist_r202_b1 (plain text, and zero-shot on the crowd text), plus the tower's kNN
# summed with b1's. Needs a GPU and models/encoders/hist_r202_b1 (just pull / Modal ENCODERS_FROM). Log: logs/r205_crowd.log.
set -eu
cd "$(dirname "$0")/../.."
mkdir -p logs
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 CROWD_KEY=v2
F='Failed to load|warn|Loading|FutureWarning|Bytecode|UNEXPECTED|position_ids|Notes|LOAD REPORT|^Key|^---|Writing model|deprecated'
{
  echo "start $(date -u +%H:%M:%S)"
  [ -f models/encoders/hist_r205_crowd/model.safetensors ] && echo "== crowd exists" || \
    ARM=crowd LOSS=infonce,dedup uv run python scripts/hist_crowd.py train_crowd 2>&1 | grep --line-buffered -vE "$F"
  [ -f models/encoders/hist_r205_ngram/tower.safetensors ] && echo "== ngram exists" || \
    uv run python scripts/hist_crowd.py train_ngram 2>&1 | grep --line-buffered -vE "$F"
  TEST_SEEDS=100000-100049 \
    ENCS=hist_r205_crowd,hist_r205_crowd:plain,hist_r205_ngram,hist_r202_b1,hist_r202_b1:crowd \
    FUSE=hist_r205_ngram+hist_r202_b1,hist_r205_ngram+hist_r205_crowd \
    uv run python scripts/hist_crowd.py read 2>&1 | grep --line-buffered -vE "$F"
  echo "== r205 done $(date -u +%H:%M:%S)"
} 2>&1 | tee logs/r205_crowd.log
