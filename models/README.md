# models/

Trained weights. Everything here except this file is gitignored (3.8 GB at last count).

- `adapters/curriculum_<model>_<arm>_lora/`: LoRA adapters saved by `scripts/exp_curriculum.py`.
  Re-score one without retraining with `EVAL_ONLY=1`.
- `adapters/universe_<model>[_lr...]_lora/`: adapters from the earlier `scripts/exp_universe_ladder.py`.
- `adapters/minilm-unsloth/`: the all-MiniLM-L6-v2 fine-tune from the original driver check.

Each adapter's provenance (commit, config, metrics) is in `evals/runs.jsonl`; the JSON results
that go with it are in `results/`.
