# gpu-unsloth-check

Started as a check that this machine's NVIDIA driver and CUDA stack could fine-tune a small
embedding model with unsloth. It now holds a set of experiments on **injecting new knowledge and
terminology into small language models** (Qwen2.5-0.5B/3B, all-MiniLM-L6-v2) and **extracting it
again** in new task formats, on two synthetic databases: 120 fictional merchants with spending
categories, and a 160-species creature universe.

## Read these, in order

1. `report/REPORT.md`: what was run and what it showed. `report/improvements.html` is the same
   story with charts and worked examples.
2. `report/QUESTIONS.md`: the open questions and planned experiments, one stable ID each
   (EVAL-1, DATA-5, TRAIN-5, ...), with a status line once addressed. **This is the next-steps
   list.**
3. `report/SURVEY.md`: a 34-paper literature survey mapped onto those IDs, ending with the
   priority order in four tiers. Per-paper summaries are under `docs/papers/`.

## Layout

- `experiments/`: data generators (`universe.py`, `merchants.py`), training and evaluation
  scripts (`exp_curriculum.py`, `exp_knowledge_injection.py`, `exp_universe_embed.py`, ...).
- `results/`: logs and JSON from every run.
- `evals/`: run tracker and `LEADERBOARD.md`.
- `docs/papers/`: extracted text, metadata and summaries of the papers read (PDFs not tracked).
- `NOTES.md`: environment setup history and Windows-specific gotchas.

## Running

```
uv sync
uv run python experiments/exp_curriculum.py C   # ARM [model] [steps] [lr]; SMOKE=1 for a quick check
```

Windows notes: use `uv run python`, not `python`; set `PYTHONIOENCODING=utf-8` when printing
dataset text. See `CLAUDE.md` for the working rules that apply to agent sessions.
