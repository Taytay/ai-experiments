# CLAUDE.md

Project: experiments on injecting new knowledge and terminology into small LLMs and embedding
models, and extracting it again (multiple choice, in-context label induction, prototype
classification). One RTX 3090 (24 GB), Windows 11, `uv` venv, unsloth. The original driver
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
| `src/` | `universe.py`, `merchants.py` (synthetic data and eval items), `icl_suite.py` | the datasets or item builders change |
| `scripts/` | Runnable experiments and table generators; each adds `src/` to `sys.path` | a new experiment |
| `data/processed/` | Frozen item sets, versioned, hash recorded in the tracker | an item set is frozen |
| `results/` | Raw JSON and logs, one file per run and arm | every run |
| `models/` | Adapters and fine-tuned weights, gitignored | every training run |
| `evals/` | Run tracker (`runs.jsonl` is the record), `LEADERBOARD.md` | every run |
| `NOTES.md` | Machine and environment history | the environment changes |

`reports/improvements.html` is an illustrated copy of the report as of section 8.
`references/lit_review.md` and `references/frameworks.md` are first-day notes, superseded by
`SURVEY.md`.

## Working rules for this machine

- Run Python with `uv run python ...`; plain `python` is a Microsoft Store alias.
- Write files with the Write/Edit tools. Bash heredocs fail here (`ENAMETOOLONG`, quote errors).
- Subagents cannot write report files; have them return text and write it from the main session.
- The Read tool cannot open PDFs (no `pdftoppm`). Extract first: `pdftotext -layout x.pdf paper.txt`.
- Paper APIs (arXiv, Semantic Scholar) rate-limit hard. One sequential process only, never in
  parallel, never from subagents. `references/papers/*/paper.txt` already holds every paper read
  so far. The `research-papers` skill defaults to `docs/papers`; pass `--dest references/papers`.
- Do not commit PDFs or TeX archives under `references/papers/` (gitignored); text and summaries only.
- Set `PYTHONIOENCODING=utf-8` when printing dataset text; the console codec is cp1252.
- Commit code before a long run so the tracker records a clean hash. Otherwise commit only when asked.
