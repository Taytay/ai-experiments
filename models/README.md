# models/

Trained weights, versioned with DVC. Git holds only the pointer file `adapters.dvc` (a hash of
the whole `adapters/` tree); the bytes live in the DVC remote `dstore`, a plain folder at
`D:\repos\dvc\ai-experiments` (`/mnt/d/repos/dvc/ai-experiments` from WSL). About 3.8 GB.

```
uv run dvc pull                 # after a fresh clone or checkout: fetch the adapters this commit expects
uv run dvc add models/adapters  # after training: re-hash the tree (autostage puts the .dvc file in git)
uv run dvc push                 # copy new blobs to D:; then commit the .dvc file
uv run dvc status -c            # anything local that is not on the remote?
```

On a machine where D: is mounted elsewhere, override the path without touching the shared
config: `uv run dvc remote modify --local dstore url /mnt/d/repos/dvc/ai-experiments`
(written to `.dvc/config.local`, which is gitignored).

Contents of `adapters/`:

- `curriculum_<model>_<arm>_lora/`: LoRA adapters saved by `scripts/exp_curriculum.py`.
  Re-score one without retraining with `EVAL_ONLY=1`.
- `universe_<model>[_lr...]_lora/`: adapters from the earlier `scripts/exp_universe_ladder.py`.
- `minilm-unsloth/`: the all-MiniLM-L6-v2 fine-tune from the original driver check.

Each adapter is one fp32 safetensors file of about 457 MB (rank 64 on every linear layer of a
3B model). Provenance (commit, config, metrics) is in `evals/runs.jsonl`; the JSON results that
go with each adapter are in `results/`.
