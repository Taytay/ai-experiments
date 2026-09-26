# models/

Trained weights, versioned with DVC. Git holds one pointer file per adapter,
`adapters/<name>.dvc`; the bytes live in the DVC remote `dstore`, a plain folder at
`D:\repos\dvc\ai-experiments` (`/mnt/d/repos/dvc/ai-experiments` from WSL). All adapters
together are around 300 GB, so a checkout keeps only the ones it is using: pull an adapter when
a run needs it and drop it afterwards. D: is the cold store; the local disk is a working set.

The remote's path is deliberately absent from the shared `.dvc/config`, because it differs by
OS. A fresh clone fails every `dvc` command with `config file error: expected 'url' for
dictionary value @ data['remote']['dstore']` until you set it once, locally:

```
uv run dvc remote modify --local dstore url D:\repos\dvc\ai-experiments          # Windows
uv run dvc remote modify --local dstore url /mnt/d/repos/dvc/ai-experiments      # WSL
```

That writes `.dvc/config.local`, which is gitignored; `.dvc/config.local.example` shows the
result. The same instructions are in a comment in `.dvc/config` itself, and `just setup` (or
`just dvc-remote`) runs the right line for the OS it is invoked from.

```
just pull NAME [NAME...]    # fetch adapters from D:     (uv run dvc pull models/adapters/NAME)
just push-models            # after training: dvc add each adapter dir present, then dvc push;
                            #   autostage puts the new .dvc files in git, then commit them
just drop-all               # push, then delete local adapter dirs and .dvc/cache to free disk
just pull-all               # everything this commit tracks (hundreds of GB; rarely wanted)
uv run dvc status -c        # anything local that is not on the remote?
```

`.dvc/config` sets `cache.type = hardlink,copy`, so a pulled adapter is stored once on disk
(cache and `models/adapters/` share the bytes). Hardlinked files are read-only: write a new
adapter directory rather than editing files in an existing one. Deleting only the adapter
directory does not free space while `.dvc/cache` still holds the blobs; `just drop-all` removes
both.

Before this layout, a single `models/adapters.dvc` tracked the whole tree. Older commits still
have it; `uv run dvc pull` on one of those fetches that commit's full tree.

Contents of `adapters/`:

- `curriculum_<model>_<arm>_lora/`: LoRA adapters saved by `scripts/exp_curriculum.py`.
  Re-score one without retraining with `EVAL_ONLY=1`.
- `universe_<model>[_lr...]_lora/`: adapters from the earlier `scripts/exp_universe_ladder.py`.
- `minilm-unsloth/`: the all-MiniLM-L6-v2 fine-tune from the original driver check.

Each adapter is one fp32 safetensors file of about 457 MB (rank 64 on every linear layer of a
3B model). Provenance (commit, config, metrics) is in `evals/runs.jsonl`; the JSON results that
go with each adapter are in `results/`.
