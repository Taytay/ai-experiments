# ai-experiments

Started as a check that this machine's NVIDIA driver and CUDA stack could fine-tune a small
embedding model with unsloth. It now holds a set of experiments on **injecting new knowledge and
terminology into small language models** (Qwen2.5-0.5B/3B, all-MiniLM-L6-v2) and **extracting it
again** in new task formats, on two synthetic databases: 120 fictional merchants with spending
categories, and a 160-species creature universe.

## Where to start

**`PLAN.md`** is the work queue: an ordered table of steps with status, and the procedure for
doing one. To continue the project, do the first `todo` row. The rest is reference:

1. `reports/REPORT.md`: what was run and what it showed. `reports/improvements.html` is the
   same story with charts and worked examples.
2. `reports/QUESTIONS.md`: defines every question ID the queue refers to (EVAL-1, DATA-5,
   TRAIN-5, ...) with the reviewer's evidence and the proposed experiment.
3. `references/SURVEY.md`: what 34 papers say about each of those IDs. Per-paper summaries are
   under `references/papers/`.

## Layout

The layout follows the Cookiecutter Data Science convention: what we produced is in `reports/`,
what we collected is in `references/`, code is split into a library and entry points.

```
PLAN.md          work queue and per-step status (start here)
CLAUDE.md        one-job-per-location table and the working rules for agent sessions
NOTES.md         machine and environment history
data/            no raw data; processed/ holds frozen item sets (see data/README.md)
src/             library: universe.py, merchants.py (data generators), icl_suite.py
scripts/         entry points: exp_*.py experiments, bench_throughput.py, table generators,
                 and the original driver-check scripts (check_gpu.py, finetune_embed.py)
evals/           run tracker (runs.jsonl is the record of truth) and LEADERBOARD.md
results/         raw JSON and logs from every run
models/          LoRA adapters and fine-tuned weights, versioned with DVC (see models/README.md)
reports/         REPORT.md, improvements.html, QUESTIONS.md
references/      papers/ (text, metadata, summaries; PDFs not tracked), SURVEY.md,
                 and two first-day notes (lit_review.md, frameworks.md)
```

## Running

```
uv sync
uv run dvc pull                             # fetch the trained adapters (3.8 GB) from D:\repos\dvc\ai-experiments
uv run python scripts/exp_curriculum.py C   # ARM [model] [steps] [lr]; SMOKE=1 for a quick check
uv run python -m evals leaderboard          # regenerate evals/LEADERBOARD.md
```

Large files are not in git. The DVC remote is a plain folder on the D: drive, one folder per
repo under `D:\repos\dvc\`; from WSL the same folder is `/mnt/d/repos/dvc/ai-experiments`, set
once per WSL checkout with `uv run dvc remote modify --local dstore url /mnt/d/repos/dvc/ai-experiments`.

Scripts add `src/` to `sys.path` themselves, so they run from any working directory. The repo
runs natively on Windows 11 or from WSL2 (Ubuntu) on the same machine; both see the RTX 3090.
Each checkout needs its own `uv sync` and `uv run dvc pull`. Always use `uv run python`, not
`python`. On Windows, also set `PYTHONIOENCODING=utf-8` when printing dataset text. See
`CLAUDE.md` for the working rules that apply to agent sessions and `NOTES.md` for the
environment history.
