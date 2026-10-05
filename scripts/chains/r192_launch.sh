#!/bin/bash
# Rows 192 and 193 launch, after r192_data.sh: DVC-add the new v4 data, commit, launch r192_193.json detached.
set -eu
cd "$(dirname "$0")/../.."
until grep -q "r192 data done" logs/r192_data.log 2>/dev/null; do sleep 30; done
F="data/processed/realstyle_crowdclus_v4_test.json data/processed/realstyle_crowdclus_v4_train.json data/processed/realstyle_v4gcdb_train.jsonl"
for src in ngram behav; do for c in g gc; do for s in "" _seen; do F="$F data/processed/realstyle_v4${c}${s}_${src}_test.json"; done; done; done
uv run dvc add $F | tail -1
uv run dvc push $(for f in $F; do echo $f.dvc; done) | tail -1
git add data/processed/.gitignore data/processed/*.dvc scripts/modal_jobs/r192_193.json scripts/modal_jobs/make_r192_r193.py scripts/chains/r192_data.sh scripts/chains/r192_launch.sh
git commit -qm "Rows 192-193: v4 category clusters, behavioural / n-gram test sets, crowd + behavioural training file (DVC); job list r192_193

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JVRPTTkLk2ASQqD5TUEPg3"
modal run --detach scripts/modal_app.py --jobs scripts/modal_jobs/r192_193.json
