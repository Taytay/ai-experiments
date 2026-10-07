#!/bin/bash
# Row 235: the shared prep job first (r235-prep, SAVE_DATA), then the three cold-start arms reading its caches (DATA_FROM=...,r235-prep).
set -u
cd "$(dirname "$0")/../.."
scripts/wait_modal_jobs.sh r235_prep logs/r235_prep_launch.log | tail -3
modal volume ls ai-exp-results r235-prep/data/interim 2>&1 | head -5
echo "== arms $(date -u +%H:%M:%S)"
modal run --detach scripts/modal_app.py --jobs scripts/modal_jobs/r235.json > logs/r235_launch.log 2>&1
echo "== launched $(date -u +%H:%M:%S)"
