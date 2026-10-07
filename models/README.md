# models/

Trained weights, versioned with DVC. Every trained model goes to DVC (owner's rule: keep every
adapter, full model, encoder and checkpoint; never drop one before it is pushed). Git holds one
pointer file per model directory, `<name>.dvc`, and a matching line in that folder's
`.gitignore`; the bytes live in the DVC remote `dstore`, a plain folder at
`D:\repos\dvc\ai-experiments` (`/mnt/d/repos/dvc/ai-experiments` from WSL). Together they run
to hundreds of GB, so a checkout keeps only what it is using: pull a model when a run needs it and
drop it afterwards. D: is the cold store; the local disk is a working set.

| Folder | What | Tracked |
|---|---|---|
| `adapters/` | LoRA adapters (peft or unsloth) and, from the earlier rows, some small full fine-tunes; about 650 on 2026-10-07 | one `adapters/<name>.dvc` each |
| `encoders/` | encoders trained as readers: late-interaction and two-tower models, history encoders, knowledge-stage encoders; about 130 on 2026-10-07 | one `encoders/<name>.dvc` each |
| `smoke/` | outputs of `just smoke` and other `SMOKE=1` runs: plumbing checks, not results | gitignored, never pushed |

Provenance (commit, config, metrics) is in `evals/runs.jsonl`; the JSON results that go with
each model are in `results/`. Models trained on Modal come back in `modal_out/<tag>/` and are
copied here by `scripts/ingest_modal.py`.

## Names

The name says which script wrote it and with what settings; the suffixes are that script's env
knobs, so read the script's docstring for them.

`adapters/`:

- `decider_<model>_<episode suffix>_lora`: decider-0.8B / 2B / 4B / 35B-A3B fine-tunes from
  `scripts/exp_decider_finetune.py` (Qwen3.5-based, peft); `h100bf16st<steps>` gives the
  hardware, precision and step count, `s1` a second seed. The current main LLM line.
- `categoriser_<base>_<db arm>_..._lora`: the categorisers from `scripts/exp_categoriser.py`
  (Qwen2.5-3B / 14B, Qwen3.5; `none` / `param` / `ret` fact-DB arms, `f<n>` folds, `h100` Modal
  runs). Research records.
- `slot_<model>_*`, `corpusslot_*`, `kev_*`: the one-slot readout on other bases and the
  decision-model comparisons.
- `encmask_*`, `encgli_*`, `encmbi_*`, `hops_*`, `mlm_*`: encoder option scorers (Ettin, ModernBERT,
  GLiClass, Laya and others) from the encoder rows.
- `embed_*`, `retriever_*`, `fastfit_*`, `gliclass_*`: embedding and retrieval models from the
  embedding block and REAL-5 rows.
- `curriculum_<model>_<arm>_lora`, `universe_*`, `encoder_*`, `edit_*`, `strands_*`, `minilm-unsloth`:
  the knowledge-injection rows on the species universe and merchant set (`exp_curriculum.py`,
  `exp_universe_ladder.py`, `exp_encoder.py`, the knowledge-editing row, the original driver
  check). Re-score a curriculum adapter without retraining with `EVAL_ONLY=1`, on the precision it
  was trained on (`LOAD_4BIT`).

`encoders/`:

- `li_r<row>_<arm>[_s1]`: late-interaction and two-tower models from `scripts/li_decider.py`,
  named by PLAN row and arm (encoder, `proj.pt`, `scale.pt`). `li_r227_fcr` is fcr (Ettin-32M);
  `li_r236_g2cos*` and `li_r239_g2h` are EmbeddingGemma 2 two-towers.
- `hist_*`: history encoders (kNN, ColBERT, context) from `scripts/hist_encoder.py`, `hist_encoder2.py`,
  `hist_train2.py` and row 202.
- `know_r212_*`: encoders after a knowledge and alias stage (merchant kinds; `scripts/knowledge_stage.py`).
- `two_tower_v1`, `payee_enc_v1`, `mlm_r222_*`: the first two-tower (`scripts/two_tower.py`), the
  payee encoder (`scripts/train_payee_encoder.py`), and a masked-token pretraining run (`scripts/mlm_stage.py`).

## Fetching and pushing

The remote's path is deliberately absent from the shared `.dvc/config`, because it differs by OS.
A fresh clone fails every `dvc` command with `config file error: expected 'url' for dictionary
value @ data['remote']['dstore']` until it is set once, locally:

```
uv run dvc remote modify --local dstore url D:\repos\dvc\ai-experiments          # Windows
uv run dvc remote modify --local dstore url /mnt/d/repos/dvc/ai-experiments      # WSL
```

That writes the gitignored `.dvc/config.local` (`.dvc/config.local.example` shows the result);
`just setup` (or `just dvc-remote`) runs the right line for its OS.

```
just pull NAME [NAME...]                    # adapters only: runs `dvc pull NAME` inside models/adapters/
uv run dvc pull models/encoders/NAME.dvc    # an encoder (just pull does not look in encoders/)
just push-models                            # dvc add every dir under adapters/ and encoders/, then dvc push;
                                            #   autostage stages the .dvc files; commit them
just drop-all                               # push, then delete local adapter dirs and .dvc/cache
just pull-all                               # everything this commit tracks (hundreds of GB; rarely wanted)
uv run dvc status -c                        # anything local that is not on the remote?
```

Run `just push-models` before staging anything under `models/`: staging a model folder first puts
the weights in git. `push-models` covers `encoders/` since 2026-10-07, when 18 encoders were found
only in `modal_out/`. `drop-all` deletes adapter directories only; remove a pushed encoder
directory by hand if the disk needs it.

`.dvc/config` sets `cache.type = hardlink,copy`, so a pulled model is stored once on disk (cache
and `models/` share the bytes). Hardlinked files are read-only: write a new directory rather than
editing files in an existing one. Deleting only the model directory does not free space while
`.dvc/cache` still holds the blobs; `just drop-all` removes both.

Before the per-model layout, a single `models/adapters.dvc` tracked the whole tree. Older commits
still have it; `uv run dvc pull` on one of those fetches that commit's full tree.
