#!/bin/bash
# Row 212 (c), knowledge pretraining: bank string <-> kind text contrastive stage (scripts/knowledge_stage.py), then the history stage from
# it with the b1 recipe (hist_train2.py, LOSS=infonce,dedup), read on 50 held-out v4 households beside b1 (same recipe from plain
# bge-small) and against decider on the decider items (hist_agree.py). GPU; one seed (no-duplicate-work rule: a second seed only if c1
# wins or lands within noise of b1).
# Needs: hist_train2.train to skip open_licence for a local BASE, as li_decider.py does ("if not Path(H.BASE).exists(): open_licence(H.BASE)");
# without it huggingface_hub rejects models/encoders/know_r212_c1 as a repo id (HFValidationError). The check below stops the chain until then.
set -eu
cd "$(dirname "$0")/../.."
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1
F='Failed to load|warn|Loading|FutureWarning|Bytecode|UNEXPECTED|position_ids|Notes|LOAD REPORT|^Key|^---|Writing model|deprecated'
grep -q 'Path(H.BASE).exists()' scripts/hist_train2.py || { echo "hist_train2.py still calls open_licence on a local BASE: add the guard first"; exit 1; }
echo "start $(date -u +%H:%M:%S)"

# 1. pairs, full size (CPU, DuckDB over the Overture files; 48k merchants x 2 renderings + up to 100k places + brands)
[ -f data/interim/knowledge_pairs_v1.jsonl ] || uv run --with duckdb python scripts/build_knowledge_pairs.py

# 2. knowledge stage: one epoch, batch 256, symmetric loss, 2% of merchants held out for the kind-retrieval read
[ -f models/encoders/know_r212_c1/know_config.json ] || \
  ARM=c1 PAIRS_FILE=data/interim/knowledge_pairs_v1.jsonl uv run python scripts/knowledge_stage.py train 2>&1 | grep -vE "$F"

# 3. history stage from it, the b1 recipe -> models/encoders/hist_r202_r212_c1
[ -d models/encoders/hist_r202_r212_c1 ] || \
  BASE=models/encoders/know_r212_c1 ARM=r212_c1 LOSS=infonce,dedup uv run python scripts/hist_train2.py train 2>&1 | grep -vE "$F"

# 4. did knowledge get in, and does it survive the history stage? kind retrieval on the held-out pairs
ENCS=BAAI/bge-small-en-v1.5,know_r212_c1,hist_r202_b1,hist_r202_r212_c1 PAIRS_FILE=data/interim/knowledge_pairs_v1.jsonl \
  uv run python scripts/knowledge_stage.py eval 2>&1 | grep -vE "$F"

# 5. the 50-household read (kNN and MaxSim, all and first-time payee) beside b1
[ -d models/encoders/hist_r202_b1 ] || uv run dvc pull models/encoders/hist_r202_b1.dvc
ENCS=hist_r202_b1,hist_r202_r212_c1 TEST_SEEDS=100000-100049 uv run python scripts/hist_train2.py read 2>&1 | grep -vE "$F"

# 6. against decider on the decider items: where the knowledge-pretrained encoder is right and decider is not (b1 for reference)
for enc in hist_r202_b1 hist_r202_r212_c1; do
  echo "== hist_agree $enc"
  OUT1=$enc TEST_SEEDS=100000-100049 uv run python scripts/hist_agree.py 2>&1 | grep -vE "$F"
done

# Option, not run: mix instead of stage (MIX = fraction of steps that are b1 history batches), read as in 5:
#   ARM=c2 MIX=0.5 EPOCHS=1 uv run python scripts/knowledge_stage.py train
echo "== r212 (c) done $(date -u +%H:%M:%S)"
echo "then: just push-models (know_r212_c1, hist_r202_r212_c1), commit the .dvc files with the logs"
