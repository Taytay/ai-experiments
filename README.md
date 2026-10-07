# ai-experiments

Research on **per-user transaction categorisation** with small LLMs and encoders: one shared model
for over a million users, each with a personal category scheme, that auto-files the extremely
confident transactions and suggests up to three categories for the rest. The owner's framing
(2026-09-26) is to understand the science, not to tune a product; product use only weights the
metrics. Three questions:

- **Q1** how LLMs and encoders work as multiple-choice categorisers;
- **Q2** the best way to inject knowledge (fact databases about merchants);
- **Q3** inferring meaningless or unseen category names from examples, in training and in the prompt.

The repo started on 2026-09-12 as a check that this machine's driver could fine-tune a small
embedding model with unsloth, then spent its first weeks on synthetic knowledge injection
(a fictional species universe and merchant set, Qwen2.5). Those results stay in the report as
research records.

## Where to start

1. **`PLAN.md`**: the work queue. Its "Current state" block says where the work stands, the best
   readers so far, which branch is the top, and what comes next. "Do the next step" means the
   first `todo` row whose Needs are done, by the procedure under it.
2. **`reports/review_2026-10-07/index.html`**: the review of rows 1-239 (what is known, how sure,
   recommended experiments, next steps). Read it before the report.
3. **`reports/REPORT.md`**: every result, one numbered section per step; the index at its top is
   built from `reports/report_titles.tsv` by `scripts/report_index.py`. §1 Executive summary
   covers up to §154; the review covers the rest. Cite sections as "§N short title".
4. `reports/QUESTIONS.md` defines each question ID (EVAL-1, REAL-27, MODEL-25, ...);
   `references/SURVEY.md` and `references/papers/` hold the literature.

## Main model lines (2026-10-07)

All read with the same scorecard (`ai_experiments.scorecard`) on held-out synthetic households,
the blind sets and, once per finished row, the owner's own budget (a development set now; see
PLAN's current state for the numbers).

- **decider-4B** (`Mapika/decider-4b`, Apache-2.0, on Qwen3.5-4B): a LoRA adapter trained on the
  categoriser's episodes by `scripts/exp_decider_finetune.py`, one answer slot over labelled
  options; the most accurate single reader. Recipe in §107, §113, §123; v5 households §202.
- **Late-interaction encoders** (`scripts/li_decider.py`): every category a short document of its
  recent filings, the transaction the query, MaxSim scoring trained as a decision. The best is
  fcr (`li_r227_fcr`, Ettin-encoder-32M with the crowd line, §198), about 1.9 ms a read.
- **EmbeddingGemma 2 two-tower** (`li_decider.py` with `HYBRID=2`, models `li_r236_g2cos*`,
  `li_r239_g2h`): one vector each side, cosine; the default Gemma recipe (§210, §212).

Older lines (Qwen2.5 categorisers, curriculum and universe adapters, bge / MiniLM encoders,
GLiClass, Clef, Ettin-1B option scorer) are closed; the report says why.

## Setup

On a fresh machine, `./bootstrap.sh` (WSL/Ubuntu) or
`powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1` (Windows) installs `uv` and `just`
if missing and runs `just setup` and `just doctor`.

```
just setup              # once per checkout: uv sync, then this OS's DVC remote path into .dvc/config.local
just doctor             # check machine and checkout; --gpu adds the VRAM-spill test
just pull NAME          # fetch one adapter (models/adapters/NAME) from the DVC remote
uv run dvc pull models/encoders/NAME.dvc   # fetch one encoder
just push-models        # after training: dvc add every adapter and encoder dir present, dvc push
just                    # list every recipe
```

Without `just`: `uv sync`, then the one `uv run dvc remote modify --local dstore url ...` line
for your OS from the comment in `.dvc/config`. The DVC remote is a plain folder,
`D:\repos\dvc\ai-experiments` (`/mnt/d/repos/dvc/ai-experiments` from WSL); models are pulled on
demand, never all at once (`models/README.md`). Always `uv run python`, never a bare `python`.
Scripts import the editable `ai_experiments` package and run from any directory.

## Where GPU work runs

- **Modal** (since 2026-09-25): `scripts/modal_app.py` runs this repo's scripts unchanged on an
  H100 (LLMs, Ettin-400M with FlashAttention), an L40S (small encoders) or CPU only. Job lists go
  in `scripts/modal_jobs/<row>.json` and launch with `--detach`; `scripts/wait_and_ingest.sh`
  watches them and `scripts/ingest_modal.py` brings results back. Every launch's cost is in
  `reports/modal_costs.md`.
- **The local RTX 3090** (WSL2 or native Windows): small, short jobs when free (smoke tests,
  small encoders), and every read of the owner's real budget.

The details (overlays for Qwen3.5 and EmbeddingGemma 2, GPU choice per model, watching jobs,
licences, seeds) are in `CLAUDE.md`, "GPU work: Modal".

## Layout

Cookiecutter Data Science: `reports/` is what we produced, `references/` what we collected,
`src/ai_experiments/` the library, `scripts/` the entry points (experiments, table generators,
`scripts/chains/` run scripts, `scripts/modal_jobs/` job lists). `data/processed/` holds frozen
item sets, `results/` raw JSON and per-item scores, `evals/` the run tracker (`runs.jsonl`,
`LEADERBOARD.md`, `uv run evals`), `models/` adapters and encoders under DVC. The full
one-job-per-location table and every working rule are in **`CLAUDE.md`**; `NOTES.md` is the
machine and environment history.

## The owner's real data

The owner's own YNAB budget is read on this machine only. It never goes into git, logs, Modal
volumes or reports: no rows, payee names or ids. Reports carry aggregates only, and names only as
generic kinds. The access token and budget id live outside the repo under `~/.config/ynab/`
(0600), and the cache and per-item scores under `~/.local/share/ynab-real-eval/`. The code that
reads it is listed in PLAN's current state.
