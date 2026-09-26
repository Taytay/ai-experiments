# EasyEdit driver and settings for PLAN step 19 (BASE-2)

Copies of the files that live in the EasyEdit checkout beside this repo (`../EasyEdit`, its own `uv` environment with
`ai_experiments` installed `--no-deps`; `overrides.txt` pins pyyaml 6.0.2), kept here so the run is reproducible:

- `ai_exp_edit.py`: the driver (`.venv/bin/python ai_exp_edit.py MEMIT|AlphaEdit type|all [--n N]`), one `batch_edit` in
  a single chunk with `sequential_edit=True`; saves the edited model under `models/adapters/edit_<alg>_<facts>_qwen2.5-3b`
  and `results/edit_<alg>_<facts>.json`. `COV_RIDGE=<x>` adds a ridge to MEMIT's solve and tags the names `_ridge<x>`;
  `SMOKE_DIR` redirects a `--n` smoke's model outside the adapters directory.
- `MEMIT_qwen2.5-3b.yaml`, `AlphaEdit_qwen2.5-3b.yaml`: go to `hparams/MEMIT/qwen2.5-3b.yaml` and `hparams/AlphaEdit/qwen2.5-3b.yaml`.
  `lm_head_module` is `model.embed_tokens` because Qwen2.5-3B ties its output head to the embeddings.
- `easyeditor.patch`: the library change (`cov_ridge` hyperparameter on the MEMIT solve), applied with `git apply`.

The covariance statistics (`data/stats/Qwen2.5-3B/wikitext_stats/*.npz`, 20,000 WikiText samples, about 20 minutes) and
AlphaEdit's null-space projections (`null_space_project.pt`, one SVD per layer) are cached in the checkout, not copied.
