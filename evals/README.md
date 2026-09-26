# evals: experiment tracking with provenance

A small, dependency-free tracker. Every experiment run records **what** was run
(experiment name, model, full config dict, script path + sha256), **at which code
state** (git commit, branch, dirty flag), **where** (host, user, Python/torch/CUDA/GPU,
library versions) and **what came out** (metrics per condition, artifact paths + sha256).

## Files

This folder holds the tracker's data; the code is the `ai_experiments.evals` package.

| path | tracked in git? | purpose |
|---|---|---|
| `src/ai_experiments/evals/tracker.py` | yes | `Run` context manager + SQLite schema |
| `src/ai_experiments/evals/__main__.py` | yes | CLI, installed as the `evals` command |
| `evals/runs.jsonl` | **yes** | diff-friendly mirror, one JSON line per run, rewritten on every finish. This is the record of truth in the repo. |
| `evals/LEADERBOARD.md` | yes | regenerated with `leaderboard`; every finished run x condition x metric |
| `evals/runs.db` | no (gitignored) | SQLite working copy. Rebuild anywhere with `uv run evals rebuild` |

## Instrumenting an experiment

```python
from ai_experiments.evals.tracker import Run

with Run("universe_ladder", model="Qwen/Qwen2.5-3B",
         config=dict(steps=600, lr=2e-4, method="lora", lora_r=64)) as run:
    ...
    run.log({"L1_recall": 71.2, "L3_induct_type_nonsense": 40.0}, condition="lora")
    run.log({"loss": 0.31}, condition="lora", step=300)   # optional step for curves
    run.artifact("results/universe_Qwen2.5-3B.json")
```

Only numeric values are stored as metrics; strings belong in `config`. If the block
raises, the run is marked `failed` with the exception text. The run id is printed at
start (`[evals] run 20260913-151609-207526 ... commit=eb845a6*`); a `*` means the
working tree had uncommitted changes to anything other than the tracker's own outputs (`evals/runs.jsonl`, `evals/LEADERBOARD.md`, `results/`), i.e. code or data provenance is approximate. Commit first.

## CLI

`uv run evals` (the same as `uv run python -m ai_experiments.evals`):

```
uv run evals list [--exp NAME] [-n 20]
uv run evals show RUN_ID_PREFIX
uv run evals compare EXPERIMENT [--metric PREFIX] [--condition COND] [--show-cfg lr steps]
uv run evals leaderboard              # writes evals/LEADERBOARD.md  (also: just leaderboard)
uv run evals import-json results/x.json --exp NAME --model M --commit SHA --config '{...}'
uv run evals export | rebuild         # db -> jsonl | jsonl -> db
```

`compare` is the workhorse: one row per (run, condition), one column per metric,
with the chosen config keys inlined, so hyperparameter sweeps read as a table.

## Conventions

- **experiment** = a question ("merchant_knowledge_injection", "universe_ladder"), not a script.
  Different hyperparameters or models are different *runs* of the same experiment.
- **condition** = the arm inside a run ("base", "incontext", "ft_aug_lr1e-05", "lora_ctx").
- Metric names are stable across runs so `compare` lines up columns. Accuracies are
  percentages; `*_margin` are probabilities; `ppl_*` are perplexities.
- Result JSONs in `results/` remain the raw artifacts; the tracker stores their sha256.

## Why not DoltLite / MLflow / W&B

- **DoltLite** (dolthub/doltlite, beta Aug 2026) is the natural fit: SQLite with branches,
  diffs and merges, same `sqlite3_*` API. Its Python wheel (`doltlite` 0.50.10) ships only
  `libdoltlite.so`; there is no Windows library, and at the time Smart App Control on this
  machine blocked unsigned native DLLs. Both objections have since weakened (Smart App Control
  is off, and the repo now also runs from WSL2 where the `.so` would load), so DoltLite is an
  option again. `tracker.connect()` is the single place to swap in `doltlite.connect()`; the
  schema and CLI need no edits.
- **MLflow** requires pyarrow (blocked at the time); **W&B** is cloud-hosted. Neither adds
  much over a 300-line tracker for one-GPU experiments, and the JSONL mirror gives
  git-native review of results alongside the code that produced them.
