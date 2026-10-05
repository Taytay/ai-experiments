#!/bin/bash
# Row 190 launch, after the owner-budget scoring that uploads data/processed has finished (DVC's hardlinks count as modifying files
# Modal is uploading): DVC-add the v4 data, commit, launch r190.json detached.
set -eu
cd "$(dirname "$0")/../.."
until grep -q "r183 owner scoring done" logs/r188_owner.log 2>/dev/null; do sleep 30; done
F="data/processed/realstyle_crowd_v4_train.json data/processed/realstyle_crowd_v4_test.json data/processed/realstyle_v4g_train.jsonl data/processed/realstyle_v4gcd_train.jsonl data/processed/realstyle_v4gcdh_train.jsonl data/processed/realstyle_v4g_test.json data/processed/realstyle_v4gc_test.json data/processed/realstyle_v4g_seen_test.json data/processed/realstyle_v4gc_seen_test.json"
uv run dvc add $F | tail -1
uv run dvc push $(for f in $F; do echo $f.dvc; done) | tail -1
git add data/processed/.gitignore data/processed/*.dvc scripts/modal_jobs/r190.json scripts/chains/r190_launch.sh
git commit -qm "Row 190: v4 shared world data (both fixes; DVC); job list r190 (G4, GCD4, GCDH4, two seeds)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JVRPTTkLk2ASQqD5TUEPg3"
modal run --detach scripts/modal_app.py --jobs scripts/modal_jobs/r190.json
