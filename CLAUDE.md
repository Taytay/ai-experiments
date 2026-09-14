# CLAUDE.md

Project: experiments on injecting new knowledge and terminology into small LLMs and embedding
models, and extracting it again (multiple choice, in-context label induction, prototype
classification). One RTX 3090 (24 GB), Windows 11, `uv` venv, unsloth. The original driver
check that named the repo is done; the research is what the repo is for now.

## Where things are

| What | Where |
|---|---|
| Main write-up of results | `report/REPORT.md`; illustrated version `report/improvements.html` |
| **Open questions and next steps** (stable IDs, status per item) | `report/QUESTIONS.md` |
| **Priority order** for those next steps, with literature reasons | `report/SURVEY.md` section 2 |
| Literature: 34 paper summaries, full text, index by thread | `docs/papers/` (`INDEX.md`) |
| Experiment scripts | `experiments/` (`universe.py`, `merchants.py`, `exp_curriculum.py`, ...) |
| Results, logs, JSON | `results/` |
| Run tracker and leaderboard | `evals/` (`LEADERBOARD.md`, `evals/README.md`) |
| Environment history and gotchas | `NOTES.md` |

Start any new experiment by picking an ID from `QUESTIONS.md` (tier order in `SURVEY.md`),
and when it is done add a `Status:` line under that item pointing at the commit or result file.

## Working rules for this machine

- Run Python with `uv run python ...`; plain `python` is a Microsoft Store alias.
- Write files with the Write/Edit tools. Bash heredocs fail here (`ENAMETOOLONG`, quote errors).
- Subagents cannot write report files; have them return text and write it from the main session.
- The Read tool cannot open PDFs (no `pdftoppm`). Extract first: `pdftotext -layout x.pdf paper.txt`.
- Paper APIs (arXiv, Semantic Scholar) rate-limit hard. One sequential process only, never in
  parallel, never from subagents. `docs/papers/*/paper.txt` already holds every paper read so far.
- Do not commit PDFs or TeX archives under `docs/papers/` (gitignored); text and summaries only.
- Set `PYTHONIOENCODING=utf-8` when printing dataset text; the console codec is cp1252.
- Commit only when asked.
