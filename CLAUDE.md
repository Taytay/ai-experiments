# CLAUDE.md

Project: experiments on injecting new knowledge and terminology into small LLMs and embedding
models, and extracting it again (multiple choice, in-context label induction, prototype
classification). One RTX 3090 (24 GB) in a Windows 11 machine, driven either natively or from
WSL2 (Ubuntu 26.04); `uv` venv, unsloth. The original driver
check the repo started as is done; the research is what the repo is for now.

## Start here

**`PLAN.md`** is the work queue and the only file that changes as work gets done. "Do the next
step" means: open `PLAN.md`, take the first `todo` row whose Needs are done, and follow the
procedure at the top of that file. Do not read anything else first.

## One job per location

Layout follows Cookiecutter Data Science: `reports/` is what we produced, `references/` is what
we collected, `src/` is library code, `scripts/` is entry points.

| Location | Job | Changes when |
|---|---|---|
| `PLAN.md` | Ordered queue, status per step, log | every step |
| `reports/QUESTIONS.md` | Defines each question ID (what was seen, proposed experiment) | a new question appears, or a `Status:` line is added |
| `reports/REPORT.md` | Results write-up; new results go in new numbered subsections at the end | a step finishes |
| `references/SURVEY.md` | What 34 papers say about each ID | more papers are read |
| `references/papers/<id>/summary.md` | One paper each; `INDEX.md` lists them by thread | a paper is read |
| `src/ai_experiments/` | The library, installed editable by `uv sync`: `universe.py`, `merchants.py` (synthetic data and eval items), `icl_suite.py`, `items.py` (the frozen eval sets and their hashes), `scoring.py` (per-item, per-option log-prob records), `paths.py` (repo locations), `evals/` (run tracker and its CLI) | the datasets, item builders or tracker change |
| `scripts/` | Runnable experiments and table generators; they `import ai_experiments` and run from any directory | a new experiment |
| `justfile` | Task runner: `just setup`, `just doctor`, `just push-models`, `just smoke`; `just` lists them | a routine changes |
| `data/processed/` | Frozen item sets (`ladder_v1.json` etc., plain and `_morph` universes), versioned and immutable; hashes go into the tracker config; `uv run python -m ai_experiments.items check` says whether the generators still reproduce them | a new item-set version is frozen |
| `results/` | Raw JSON and logs, one file per run and arm; `per_item/` has one JSONL per run and condition with every option's log-probs (`ai_experiments.scoring`) | every run |
| `models/` | Adapters and fine-tuned weights, tracked by DVC (`adapters.dvc` in git, bytes at `D:\repos\dvc\ai-experiments`) | every training run |
| `evals/` | Tracker data: `runs.jsonl` (the record), `LEADERBOARD.md`; CLI is `uv run evals` | every run |
| `NOTES.md` | Machine and environment history | the environment changes |

`reports/improvements.html` is an illustrated copy of the report as of section 8.
`references/lit_review.md` and `references/frameworks.md` are first-day notes, superseded by
`SURVEY.md`.

## Working rules for this machine

The repo runs from two checkouts on one machine: native Windows 11 and WSL2 (Ubuntu 26.04).
They share the GPU, the driver, everything in git, and the DVC remote on D:. They do not share
`.venv/`, `models/adapters/`, `hf_cache/` or `evals/runs.db`. See `NOTES.md` for the history.

Both sides:

- Once per checkout: `just setup` (runs `uv sync`, writes this OS's DVC remote path to the
  gitignored `.dvc/config.local`, then `dvc pull`). Without `just`, the three commands are in
  `README.md`. If any `dvc` command fails with `expected 'url' for dictionary value`, the
  remote step has not been done on this checkout.
- `just doctor` checks the machine and checkout (venv, package, UTF-8, Store alias, poppler,
  line endings, DVC remote, GPU, torchvision build matching torch so unsloth imports); `just doctor --gpu` also tests whether the driver spills VRAM
  to system RAM. Run it when something looks off, and after changing a machine setting.
- Run Python with `uv run python ...`, never a bare `python`.
- unsloth's `FastLanguageModel.from_pretrained` defaults to `load_in_4bit=True` (the NF4 4-bit base). Always pass
  `load_in_4bit=` explicitly (the scripts read `LOAD_4BIT`), and score an adapter on the precision it was trained on:
  a bf16 adapter on the 4-bit base (or the reverse) changes a fifth of the predictions (REPORT.md section 44).
- Before an `EVAL_ONLY=1` re-score, check the adapter exists under `models/adapters/` on this
  side; if not, `just pull` (see `models/README.md`).
- Subagents cannot write report files; have them return text and write it from the main session.
- The Read tool cannot open PDFs. Extract first: `pdftotext -layout x.pdf paper.txt`. That
  needs poppler, which is optional and per machine: `just pdf-tools` installs it, `just doctor`
  says whether it is present on this side.
- Paper APIs (arXiv, Semantic Scholar) rate-limit hard. One sequential process only, never in
  parallel, never from subagents. `references/papers/*/paper.txt` already holds every paper read
  so far. The `research-papers` skill defaults to `docs/papers`; pass `--dest references/papers`.
- Do not commit PDFs or TeX archives under `references/papers/` (gitignored); text and summaries only.
- Commit code before a long run so the tracker records a clean hash. Otherwise commit only when asked.
- After a training run: `just push-models` (`dvc add models/adapters` then `dvc push`), then
  commit the updated `models/adapters.dvc` with the results.
- The driver can spill VRAM to system memory and turn an OOM into a 3x slowdown
  (`reports/REPORT.md` section 7). `just doctor --gpu` tests it; unless that check is OK on
  this side, watch step times, not just whether the run finishes.

Windows side only:

- Write files with the Write/Edit tools. Bash heredocs fail there (`ENAMETOOLONG`, quote errors).
- Console output is UTF-8 without any setting: `.claude/settings.json` sets `PYTHONUTF8=1` for
  agent sessions, and importing `ai_experiments` reconfigures stdout for everything else.
