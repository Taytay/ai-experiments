#!/usr/bin/env bash
# PLAN step 154 (owner, 2026-10-01): strands-decider's v19 training corpus and evaluation sets, built by strands' own recipe stages
# (training/recipe.sh build fetch multistep generated adequacy) at the pinned commit, for training decider's one-slot readout on the
# same data (exp_corpus_slot.py). No API calls: the generated rows are committed in their repo; public sets are downloaded (the short-task
# classification sets from the Hub, ContractNLI, MuSiQue, HelpSteer2) and checked against their data/SHA256SUMS by their recipe.
# Licences (owner, 2026-10-01): the full corpus for a research comparison only; models trained on it are never shipped or used on real data.
# Writes data/external/strands_corpus/ (gitignored; tracked by DVC once built).
# usage: uv run --with "strands-decider[train]==0.1.0" --with transformers==5.17.0 bash scripts/build_strands_corpus.sh
set -euo pipefail
SHA=f91487ab8f7e4b4967ae57e46b8d90e91e67d616
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT="$ROOT/data/external/strands_corpus"
SRC=/tmp/strands-decider-$SHA
if [ ! -d "$SRC" ]; then
  curl -fsSL "https://github.com/strands-labs/strands-decider/archive/$SHA.tar.gz" | tar xz -C /tmp
fi
cd "$SRC"
export PY=$(command -v python)
bash training/recipe.sh build fetch multistep generated adequacy
mkdir -p "$OUT"
for f in train_v5 holdout_v5_norule multistep_v14 multistep_v14_eval generated_v16 generated_v16_eval generated_v18 generated_v18_eval \
         adequacy_hs2 adequacy_hs2_eval adequacy_gen adequacy_gen_eval; do
  cp "data/$f.jsonl" "$OUT/"
done
cp data/synthetic/replay_v14_multistep.jsonl data/SHA256SUMS data/sources.md "$OUT/"
(cd "$OUT" && wc -l ./*.jsonl && sha256sum ./*.jsonl > MANIFEST.sha256)
