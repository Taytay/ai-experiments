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
justfile         task runner: setup, doctor, pull, push-models, smoke, leaderboard
bootstrap.sh     fresh machine: install uv and just if missing, then `just setup` and
bootstrap.ps1    `just doctor` (WSL/Ubuntu and Windows respectively)
src/ai_experiments/
                 the library, installed editable: universe.py, merchants.py (data generators),
                 icl_suite.py, items.py (frozen eval sets), scoring.py (per-item option scores),
                 paths.py (repo locations), evals/ (run tracker + CLI)
scripts/         entry points: exp_*.py experiments, rescore.py (score any adapter on the frozen
                 sets), bench_throughput.py, table generators, doctor.py, and the original
                 driver-check scripts (check_gpu.py, finetune_embed.py)
data/            no raw data; processed/ holds frozen item sets (see data/README.md)
evals/           tracker data: runs.jsonl (the record of truth) and LEADERBOARD.md
results/         raw JSON and logs from every run; per_item/ has every option's log-prob per item
models/          LoRA adapters and fine-tuned weights, versioned with DVC (see models/README.md)
reports/         REPORT.md, improvements.html, QUESTIONS.md
references/      papers/ (text, metadata, summaries; PDFs not tracked), SURVEY.md,
                 and two first-day notes (lit_review.md, frameworks.md)
.claude/         settings.json: environment for agent sessions (UTF-8)
```

## Running

On a fresh machine, run the bootstrap for your OS once: `./bootstrap.sh` on WSL/Ubuntu,
`powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1` on Windows. It installs `uv` and
`just` if missing, then runs `just setup` and `just doctor`. With those two tools present you
can skip it and use the recipes directly.

```
just setup                                  # once per checkout: uv sync, DVC remote for this OS, dvc pull (3.8 GB)
just doctor                                 # check machine and checkout; --gpu adds the VRAM-spill test
just smoke                                  # plumbing check: arm C, 2 training steps, tiny eval; not recorded in the tracker
uv run python scripts/exp_curriculum.py C   # ARM [model] [steps] [lr]
uv run evals leaderboard                    # regenerate evals/LEADERBOARD.md
just                                        # list every recipe
```

Without `just`, setup is three commands: `uv sync`; the one `uv run dvc remote modify --local`
line for your OS from the comment in `.dvc/config`; `uv run dvc pull`.

Large files are not in git. The DVC remote is a plain folder on the D: drive, one folder per
repo under `D:\repos\dvc\`; from WSL the same folder is `/mnt/d/repos/dvc/ai-experiments`. The
path is not in the shared DVC config because it differs by OS; `just setup` writes the right
one to the gitignored `.dvc/config.local` (details in `models/README.md`).

The library is the `ai_experiments` package under `src/`, installed editable by `uv sync`, so
scripts import it and run from any working directory. Output is UTF-8 whatever the console
codec (the package reconfigures stdout on import; `.claude/settings.json` sets `PYTHONUTF8` for
agent sessions). `.gitattributes` keeps every text file LF on both sides. The repo runs natively
on Windows 11 or from WSL2 (Ubuntu) on the same machine; both see the RTX 3090, and each
checkout needs its own `just setup`. Always use `uv run python`, not `python`. See `CLAUDE.md`
for the working rules that apply to agent sessions and `NOTES.md` for the environment history.
