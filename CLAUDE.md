# CLAUDE.md

Project: experiments on injecting new knowledge and terminology into small LLMs and embedding
models, and extracting it again (multiple choice, in-context label induction, prototype
classification). One RTX 3090 (24 GB) in a Windows 11 machine, driven either natively or from
WSL2 (Ubuntu 26.04); `uv` venv, unsloth. The original driver
check the repo started as is done; the research is what the repo is for now.

## Start here

**`PLAN.md`** is the work queue and the only file that changes as work gets done. Its "Current state" block at the top says where the work stands, which branch is the top of the PR stack, which
checkout holds what, and what is in flight. "Do the next
step" means: open `PLAN.md`, take the first `todo` row whose Needs are done, and follow the
procedure at the top of that file. Do not read anything else first.

## One job per location

Layout follows Cookiecutter Data Science: `reports/` is what we produced, `references/` is what
we collected, `src/` is library code, `scripts/` is entry points.

| Location | Job | Changes when |
|---|---|---|
| `PLAN.md` | Ordered queue, status per step, log | every step |
| `reports/QUESTIONS.md` | Defines each question ID (what was seen, proposed experiment) | a new question appears, or a `Status:` line is added |
| `reports/REPORT.md` | Results write-up; new results go in new numbered subsections at the end. Cite a section as "§N short title" (e.g. "§179 Pinterest research applicability"), never a bare number: section numbers are not PLAN row numbers. A new section adds a line to `reports/report_titles.tsv`, then `uv run python scripts/report_index.py` rebuilds the index at the top (PLAN rows read from the Result column) | a step finishes |
| `references/SURVEY.md` | What 34 papers say about each ID | more papers are read |
| `references/papers/<id>/summary.md` | One paper each; `INDEX.md` lists them by thread | a paper is read |
| `src/ai_experiments/` | The library, installed editable by `uv sync`: `universe.py`, `merchants.py` (synthetic data and eval items), `icl_suite.py`, `items.py` (the frozen eval sets and their hashes), `scoring.py` (per-item, per-option log-prob records), `paths.py` (repo locations), `evals/` (run tracker and its CLI) | the datasets, item builders or tracker change |
| `scripts/` | Runnable experiments and table generators; they `import ai_experiments` and run from any directory | a new experiment |
| `justfile` | Task runner: `just setup`, `just doctor`, `just push-models`, `just smoke`; `just` lists them | a routine changes |
| `data/processed/` | Frozen item sets (`ladder_v1.json` etc., plain and `_morph` universes), versioned and immutable; hashes go into the tracker config; `uv run python -m ai_experiments.items check` says whether the generators still reproduce them | a new item-set version is frozen |
| `data/processed/` (large files) | Item sets over ~20 MB are tracked by DVC, like the adapters (owner, 2026-09-29): `uv run dvc add data/processed/<file>` (writes `<file>.dvc` and a line in `data/processed/.gitignore`), `uv run dvc push`, commit the `.dvc` file; fetch with `uv run dvc pull data/processed/<file>.dvc`. Not Git LFS | a large set is added |
| `results/` | Raw JSON and logs, one file per run and arm; `per_item/` has one JSONL per run and condition with every option's log-probs (`ai_experiments.scoring`) | every run |
| `models/` | Adapters and fine-tuned weights, tracked by DVC (one `adapters/<name>.dvc` per adapter in git, bytes at `D:\repos\dvc\ai-experiments`; pulled on demand) | every training run |
| `evals/` | Tracker data: `runs.jsonl` (the record), `LEADERBOARD.md`; CLI is `uv run evals` | every run |
| `NOTES.md` | Machine and environment history | the environment changes |

`reports/improvements.html` is an illustrated copy of the report as of section 8.
`references/lit_review.md` and `references/frameworks.md` are first-day notes, superseded by
`SURVEY.md`.

## GPU work: Modal (from 2026-09-25)

The owner moved all GPU work to Modal (workspace `ynab`, shared with colleagues: touch nothing but this project's app and volumes).
The local 3090 may take small, short GPU jobs when free (owner, 2026-10-04: "I'm okay to use local GPU sometimes when we can!"): smoke tests, small encoders, embedding private strings; long trainings and parallel batches stay on Modal. The rules below about it still hold.

- Client: `uv tool install modal` (a uv tool, not a project dependency), then `modal setup`. Modal's agent skill is in
  `.claude/skills/modal` with its docs bundled (Modal's sample Docker token there is replaced by a placeholder: GitHub push protection).
- `scripts/modal_app.py` (app `ai-experiments-training`): builds the image from `uv.lock`, runs this repo's scripts unchanged on one H100,
  streams their output, and writes every file they create or change to the volume `ai-exp-results` under the job's tag; model
  downloads persist in `ai-exp-hf-cache`. One job: `modal run scripts/modal_app.py --tag T --env "K=V,..." --cmd "cmd1 ;; cmd2"`.
  Many in parallel (at most 8 containers): write a JSON list of `{tag, env, cmds}` to `scripts/modal_jobs/<row>.json`, commit it, then
  `modal run --detach scripts/modal_app.py --jobs scripts/modal_jobs/<row>.json` (`--detach`: the jobs survive the local client dying, as when WSL crashed on 2026-09-26). `ADAPTERS_FROM=<tag,...>` in a job's env copies adapters trained
  by earlier jobs into the container (scoring-only jobs). The container clock is UTC.
- Watching a job list (2026-09-27: eleven watchers sat "running" for hours after their jobs finished and their results went unnoticed):
  - start the watcher as a script file, as a tracked background task: `scripts/wait_and_ingest.sh r83 logs/r83_launch.log` (it also pulls every tag back and ingests it when the client exits; `scripts/wait_modal_jobs.sh` only waits). Never an
    inline `until ! pgrep -f "modal_jobs/r83.json" ...` loop: the task's own shell carries the pattern in its command line, so pgrep
    matches the watcher itself and it never exits;
  - one watcher per job list, started right after the launch; when it reports, ingest at once;
  - before saying what is running, check the truth rather than the task list: `ps -eo pid,etime,args | grep "modal run"` for local
    clients and `modal container list` for containers (this app's only); TaskStop any watcher whose jobs are gone;
  - a `--detach` job keeps running if the local client dies; then `modal app logs` or the volume (`modal volume ls ai-exp-results <tag>`)
    tell whether it finished.
- Bring results back: `modal volume get ai-exp-results <tag> modal_out/` (gitignored), then `uv run python scripts/ingest_modal.py <tag> ...`
  (copies results, unions `evals/runs.jsonl` by run_id, copies adapters); stage `results/` and `evals/` only, run `just push-models` before
  staging anything under `models/adapters/` (staging the adapter folders first puts the weights in git), then commit the `.dvc` files.
  By hand: copy `results/` into the branch, union the
  job's `evals/runs.jsonl` rows into ours by `run_id`, copy adapters into `models/adapters/`, `just push-models`, commit the new `.dvc` files, `just drop-all`.
- Open licences only (owner, 2026-09-26): a model, its base and any code must be Apache-2.0, MIT, BSD or CC-BY (`ai_experiments.licences.open_licence`).
  Qwen2.5-3B-Instruct is under the Qwen Research licence: new trained work uses Qwen3.5 (`LLM_BASE=Qwen/Qwen3.5-2B` or `-4B`; it runs through
  transformers + peft, `TRAINER=hf`, with `uv run --with transformers==5.17.0 --with flash-linear-attention --with "peft>=0.21" --with torch==2.13.0 --with torchvision==0.28.0`:
  torch 2.13 brings Triton 3.7.1, which flash-linear-attention needs to train on Hopper). Without fla its DeltaNet layers run in pure
  PyTorch at a fifth of the speed; fla tunes a kernel per sequence length, so Qwen3.5 batch lengths are rounded up to 64
  (`scripts/bench_train_step.py`: warm, Qwen3.5-2B trains at ~14k tokens/s, as fast as Qwen2.5-3B in the same loop).
- Defaults for new runs: bf16 base (`LOAD_4BIT=0`), `MICRO=16` (one 16-sequence pass per step), all-label loss for the no-DB
  categoriser (`ALL_LABELS=1`), `RUN_TAG=h100...` so Modal adapters never collide with 3090 ones; compare arms only within one
  hardware and precision setting. About $0.60 to $0.80 per train-and-score job on the H100.
- New decider rows build their job list with `scripts/modal_jobs/make_jobs.py <spec>` (a spec in `scripts/modal_jobs/specs/`: arms, the row's
  reads, variant reads per arm, drills on or off, seeds, read-only arms); it takes adapter names from the trainer itself and prints the
  estimated cost per job before anything launches. History-encoder readers (kNN, MaxSim) run through `scripts/hist_fast.py` (GPU, cached). Synthetic households are cached on disk by `two_tower.households` (`data/interim/hh_cache`, keyed by the generator's code, data and env; `HH_CACHE=0` rebuilds) and `hist_train2.py` caches its mined triplets: scripts that build households should go through it. `hist_agree.py` reads several encoders in one run (`ENCS`).
- Do not duplicate work (owner, 2026-10-05: "Let's not duplicate work here! That's just silly."): screen a new recipe or arm with one seed;
  train the second seed only for arms that win or land within seed noise of the best (1-2 points on the owner's budget), and call
  nothing a gain on the owner's budget before two seeds agree. Each job reads only the test sets its row compares (not every set
  from earlier rows); second seeds skip the drills and blind sets unless the row is about them. Synthetic reads may run on the
  local 3090 when it is free, leaving Modal to train.
- Cost of every Modal job (owner, 2026-10-05): `scripts/modal_app.py` records each launch (app id, PLAN rows from `PLAN_ROWS` or the job-list
  name, job tags, minutes) in `reports/modal_launches.jsonl`; `uv run python scripts/modal_costs.py` joins it with Modal's billing report into
  `reports/modal_costs.md` (per row, per launch, per job by minutes). Every REPORT section whose step used Modal ends with a **Cost** line from
  `modal_costs.py --rows N` (local-GPU steps say $0). Billing is reported for complete days, so run it the day after.
- Report every result with the scorecard (`ai_experiments.scorecard`: top-1, top-3, calibrated bits, auto-file coverage, skill over
  the no-model cascade) on held-out users, and add a blind strong-reader ceiling (`scripts/blind_ceiling.py`) for a new item set.

## Branches, PRs and models

- **One branch and PR per row** (owner, 2026-10-01): each PLAN row, or round of tests, gets its own branch off the current top and its own
  PR, added to the top of GitHub stack #24 with `gh stack link <every PR already in the stack, in stack order> <new-branch>`: every member, merged or not, from #5 up (the order is the stack's, not numeric: #20 sits after #23); leave one out and gh stack refuses ("this would remove ... from the stack") and prints the current list to copy (PR numbers push nothing;
  only the new branch is pushed). Commit a row's results on its own branch before starting the next row. `PLAN.md`'s current state lists
  the stack.
- **Merge forward only** (owner, after two PRs were auto-closed by a force push): lower branches reach upper ones by merging; never rebase,
  never force-push, never delete a branch. `evals/runs.jsonl` conflicts on every merge-forward: resolve as the union of rows by `run_id`.
- **Keep every trained model** (owner): every adapter, full model, encoder and checkpoint goes to DVC (`just push-models`, commit the
  `.dvc` files); `just drop-all` only frees the local disk after the push.
- **Table generators live in `scripts/`**, never the scratchpad (it was wiped by a reboot once).
- **Verify before routing around** (owner, 2026-09-27): benchmark a slowdown, or read the library source for a "not supported", before
  building a workaround.
- **Judge on the blind sets.** REAL-6 / REAL-7 and the behaviour sets were built with the training generators; they are diagnostics.
  Choices are read on blind_v1 / v2 (blind_v3 is sealed for a final read). A set built with the training data hides overfitting (REPORT
  103, 151).

## Working rules for this machine

The repo runs from two checkouts on one machine: native Windows 11 and WSL2 (Ubuntu 26.04).
They share the GPU, the driver, everything in git, and the DVC remote on D:. They do not share
`.venv/`, `models/adapters/`, `hf_cache/` or `evals/runs.db`. See `NOTES.md` for the history.

Both sides:

- Once per checkout: `just setup` (runs `uv sync`, writes this OS's DVC remote path to the
  gitignored `.dvc/config.local`). Adapters are not pulled; fetch the ones a run needs with
  `just pull NAME`. Without `just`, the two commands are in `README.md`. If any `dvc` command fails with `expected 'url' for dictionary value`, the
  remote step has not been done on this checkout.
- If `dvc pull` of the whole `models/adapters` fails (the full set is hundreds of GB and filled the C: drive once), pull only the adapter you need: `uv run dvc pull models/adapters/<name>`.
- `just doctor` checks the machine and checkout (venv, package, UTF-8, Store alias, poppler,
  line endings, DVC remote, GPU, torchvision build matching torch so unsloth imports); `just doctor --gpu` also tests whether the driver spills VRAM
  to system RAM. Run it when something looks off, and after changing a machine setting.
- Run Python with `uv run python ...`, never a bare `python`.
- unsloth's `FastLanguageModel.from_pretrained` defaults to `load_in_4bit=True` (the NF4 4-bit base). Always pass
  `load_in_4bit=` explicitly (the scripts read `LOAD_4BIT`), and score an adapter on the precision it was trained on:
  a bf16 adapter on the 4-bit base (or the reverse) changes a fifth of the predictions (REPORT.md section 44).
- Before an `EVAL_ONLY=1` re-score, check the adapter exists under `models/adapters/` on this
  side; if not, `just pull NAME` (see `models/README.md`).
- Subagents cannot write report files; have them return text and write it from the main session.
- The Read tool cannot open PDFs. Extract first: `pdftotext -layout x.pdf paper.txt`. That
  needs poppler, which is optional and per machine: `just pdf-tools` installs it, `just doctor`
  says whether it is present on this side.
- Paper APIs (arXiv, Semantic Scholar) rate-limit hard. One sequential process only, never in
  parallel, never from subagents. `references/papers/*/paper.txt` already holds every paper read
  so far. The `research-papers` skill (`.claude/skills/research-papers`, copied from the owner's taytays_stuff 2026-10-05) writes to `references/papers/` by default and keeps arXiv TeX source (`paper_flat.tex`, gitignored).
- Do not commit PDFs or TeX archives under `references/papers/` (gitignored); text and summaries only.
- Commit code before a long run so the tracker records a clean hash. Otherwise commit only when asked.
- After a training run: `just push-models` (`dvc add` on each adapter dir present, then
  `dvc push`), then commit the new or changed `models/adapters/*.dvc` with the results.
  `just drop-all` frees the disk afterwards (the C: drive filled up once from local adapters).
- The driver can spill VRAM to system memory and turn an OOM into a 3x slowdown
  (`reports/REPORT.md` section 7). `just doctor --gpu` tests it; unless that check is OK on
  this side, watch step times, not just whether the run finishes.

Windows side only:

- Write files with the Write/Edit tools. Bash heredocs fail there (`ENAMETOOLONG`, quote errors).
- Console output is UTF-8 without any setting: `.claude/settings.json` sets `PYTHONUTF8=1` for
  agent sessions, and importing `ai_experiments` reconfigures stdout for everything else.
