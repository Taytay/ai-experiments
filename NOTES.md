# NOTES: machine and environment history

## 2026-09-12: GPU / unsloth readiness check

Goal: confirm the NVIDIA driver on this machine is good enough to fine-tune a
small embeddings model with unsloth.

## Hardware / driver (observed 2026-09-12)

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090, 24 GiB, WDDM mode |
| Driver | 591.86 (Windows driver 32.0.15.9186, dated 2026-01-19) |
| Max CUDA runtime supported by driver | 13.1 |
| CUDA toolkit on disk | v12.3 (not needed; torch wheels bundle their own runtime) |
| WSL | Ubuntu 18.04 only at the time (too old for current torch/triton), so native Windows was used. Superseded 2026-09-14: a WSL2 Ubuntu 26.04 distro now exists and sees the GPU (see below). |

## Steps

1. Install `uv` via winget, create a Python 3.12 venv in `.venv/`.
2. `scripts/check_gpu.py` - torch sees the GPU and runs fp32/fp16/bf16 kernels.
3. Install unsloth + deps, run a tiny embeddings fine-tune (`scripts/finetune_embed.py`).

## Results so far

- `scripts/check_gpu.py`: PASS. torch 2.11.0+cu128 sees the RTX 3090 (sm_86), 22.8 GiB free,
  bf16 supported, fp32/fp16/bf16 matmuls agree with CPU reference. The driver is fine.
- `uv add unsloth sentence-transformers datasets`: installed unsloth 2026.9.4,
  triton-windows 3.8.0.post28, xformers 0.0.35.

## Blocker: Windows Smart App Control (not the driver)

`import unsloth` fails with:

    ImportError: DLL load failed while importing libtriton:
    An Application Control policy has blocked this file.

Diagnosis:

- `HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy\VerifiedAndReputablePolicyState = 1`
  -> Smart App Control is ON. Machine is not domain/Entra/MDM joined, so this is
  the consumer Smart App Control feature, not a corporate WDAC policy.
- Event log `Microsoft-Windows-CodeIntegrity/Operational` IDs 3077/3033 show
  python.exe blocked from loading, because they are unsigned and have no reputation:
  - `.venv\Lib\site-packages\triton\_C\libtriton.pyd`  (needed by unsloth, always imported)
  - `.venv\Lib\site-packages\pyarrow\arrow_compute.dll` and `pyarrow\lib` (needed by `datasets`,
    which sentence-transformers imports)
- torch, xformers, tokenizers, safetensors load fine (also unsigned, but Microsoft's
  reputation service knows them).

Fix options:

1. Turn Smart App Control off: Windows Security > App & browser control >
   Smart App Control settings > Off. This is one-way; it cannot be re-enabled
   without reinstalling Windows. Requires no other changes; unsloth then works natively.
2. Keep Smart App Control on and run unsloth inside WSL2 with a current Ubuntu
   (24.04). The existing Ubuntu 18.04 distro is too old. GPU passthrough works via
   the same 591.86 driver.

## Research + experiments (2026-09-12, later)

Results: `reports/REPORT.md`. Work queue and next step: `PLAN.md` (repo root). Question
definitions: `reports/QUESTIONS.md`; literature per question: `references/SURVEY.md`; paper
summaries: `references/papers/`. `references/frameworks.md` and `references/lit_review.md` are earlier notes.
Experiments in `scripts/`, outputs in `results/`. All ran with transformers + torch
only (Smart App Control still on), so unsloth/sentence-transformers trainers remain untested here.

## 2026-09-13: Smart App Control turned off by user

`VerifiedAndReputablePolicyState = 0`. triton 3.8.0, pyarrow 25.0.1, datasets 4.3.0,
sentence-transformers 6.0.1 and unsloth 2026.9.4 all import. `uv run python scripts/finetune_embed.py`
(unsloth `FastSentenceTransformer` + `SentenceTransformerTrainer`, torch.compile fast encoder
path) trains all-MiniLM-L6-v2 on the RTX 3090 end to end. Original question answered: the
driver stack is fully usable for unsloth fine-tuning of embedding models, natively on Windows.
Gotcha: sentence-transformers 6.0 rejects `dataset_num_proc` in `SentenceTransformerTrainingArguments`
even though unsloth's Windows docs recommend it (that flag belongs to TRL's `SFTConfig`).

## 2026-09-14: second environment, WSL2 (Ubuntu 26.04)

The repo can now be run from either side of the same machine: natively on Windows 11 or from
WSL2. Both see the same RTX 3090 through the same 591.86 driver. Observed from WSL:

| Item | Value |
| --- | --- |
| Distro | Ubuntu 26.04.1 LTS, kernel 6.18.33.2-microsoft-standard-WSL2 |
| GPU | `nvidia-smi` reports the RTX 3090, 24576 MiB, driver 591.86 (Windows driver passed through) |
| uv | 0.12.13 at `~/.local/bin/uv` |
| Checkout | `/home/taytay/projects/Taytay/ai-experiments`, separate from the Windows clone |
| Not yet done | unsloth import on WSL untested; `pdftotext` (poppler-utils) not installed |

What the two checkouts share and do not share:

- Shared through git: code, `results/*.json`, `evals/runs.jsonl`, `data/processed/`, reports.
- Not shared (gitignored, per checkout): `.venv/`, `models/adapters/`, `hf_cache/`,
  `evals/runs.db` (rebuild with `uv run python -m evals rebuild`, see `evals/README.md`).
- Adapters are versioned with DVC (commit 043db8a); the remote is a folder on D:, reachable from
  both sides, and each checkout runs `uv run dvc pull` to materialise them.

Same day, later, from WSL: `uv sync` built the venv from the Windows-generated lockfile without
changes (Linux resolves `triton` 3.6.0 where Windows has `triton-windows`). torch 2.11.0+cu128
reports `cuda.is_available()` true on the RTX 3090. The shared `.dvc/config` originally stored
the remote as `D:\repos\dvc\ai-experiments`, which Linux cannot open; with a warm cache
`dvc pull` then says "Everything is up to date" without ever reaching the remote. So the url
was removed from the shared config: every clone now fails loudly (`expected 'url' for
dictionary value @ data['remote']['dstore']`) until it runs the one-line
`dvc remote modify --local` command for its OS, spelled out in a comment in `.dvc/config`.
With `/mnt/d/repos/dvc/ai-experiments` set locally, `uv run dvc pull` fetched 37 files (3.8 GB,
ten adapter directories) over the 9p mount and `uv run dvc status -c` reports cache and remote
in sync.

Windows-only gotchas that do not apply on WSL: the `python` Store alias, cp1252 console
(`PYTHONIOENCODING`), Bash heredoc failures, Smart App Control, WDDM system-memory fallback.
Linux paths in `runs.jsonl` `argv` will look different from the Windows ones already recorded;
the tracker's `env.platform` field says which side a run came from.

## 2026-09-14: ergonomics pass and machine-setting audit

Repo changes so the docs carry fewer caveats: `src/ai_experiments` is an installable package
(no `sys.path` edits in scripts), `.gitattributes` forces LF, `.claude/settings.json` sets
`PYTHONUTF8=1` for agent sessions, the package reconfigures stdout to UTF-8 on import, a
`justfile` wraps setup and the DVC routine, and `scripts/doctor.py` (`just doctor`) checks the
machine. `just` 1.58.0 was installed on the Windows side with winget.

The first `just doctor --gpu` on the Windows side found the `python3.exe` Store alias still on,
no `pdftoppm`, and the driver spilling past VRAM into system RAM (a 125% allocation
succeeded). The same day the user turned the Store aliases off and set the NVIDIA Control Panel
CUDA Sysmem Fallback Policy to "Prefer No Sysmem Fallback" on both the Windows and WSL sides;
the test now gets the wanted OutOfMemoryError. The WDDM caveat in `reports/REPORT.md`
section 7 describes runs made before that change.

This file records what changed and why; `just doctor` reports the current state of a machine,
so it is not tracked here.

Git state: every tracked file is LF in the index; 80 files sit as CRLF in the Windows working
tree (written by editors, normalised on add). Harmless, and a fresh checkout is all LF.

## 2026-09-17: unattended Windows Update restart

The System event log shows `MoUsoCoreWorker.exe` starting a planned restart at 06:57 ("Operating
System: Service pack"), the machine shutting down at 06:58 and not starting again until 09:35, then
a second update restart (TrustedInstaller) at 09:36. No unexpected-shutdown event (41 / 6008), so
it was orderly, but anything on the GPU at 06:57 would have been killed. Nothing was: the row 18
follow-up queue ended at 01:17, the MEMIT smoke at 01:42, and the last commit and DVC push at 01:43.
For overnight chains, pause updates or set Windows Update's active hours to cover the run; check the
log with `powershell.exe Get-WinEvent -FilterHashtable @{LogName='System'; Id=41,1074,6008}`.

## 2026-09-21: unsloth loads the 4-bit base unless told otherwise

`FastLanguageModel.from_pretrained(name, dtype=torch.bfloat16)` returns the NF4 4-bit model (`unsloth/<name>-unsloth-bnb-4bit`,
bitsandbytes `Linear4bit` layers, 2.25 GiB for Qwen2.5-3B against 5.85 GiB in bf16) because its `load_in_4bit` default is
True; `dtype` only sets the compute type. `scripts/check_unsloth_base.py` prints what a load returns. `exp_curriculum.py` and the
older scripts passed the flag; six scripts written from 2026-09-18 on did not (REPORT.md section 44 lists them and the effect).
An adapter's `adapter_config.json` records the base it was trained on (`unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit` for the
categorisers), which is how it was noticed: the peft-trained bf16 adapter loaded through unsloth landed on the 4-bit base and
agreed with its bf16 scoring on only 81% of the predictions. Rule: pass `load_in_4bit` explicitly everywhere (`LOAD_4BIT` in the
scripts) and score on the precision the adapter was trained on.

## 2026-09-27: Qwen3.5 and the decision models on Modal

- Qwen3.5 (hybrid: three Gated DeltaNet layers to one attention layer) needs flash-linear-attention (fla); without it transformers
  runs the DeltaNet layers in pure PyTorch (a fifth of the speed, and more memory). fla refuses to train on Hopper with Triton in
  [3.4, 3.7.1) (wrong gradients, fla #640); its tilelang backend needs nvcc, which the Modal image lacks. Two working setups:
  `uv run --with transformers==5.17.0 --with flash-linear-attention --with "peft>=0.21" --with torch==2.13.0 --with torchvision==0.28.0`
  (torch 2.13 brings Triton 3.7.1; a bare `--with flash-linear-attention` pulls torch 2.14 against the project's torchvision 0.26 and
  breaks every import), or unsloth in the project env (its zoo vendors fla with a tile that avoids #640; take `tok.tokenizer` from the
  processor it returns for these vision-language checkpoints).
- fla tunes a kernel per sequence length: pad batches to a multiple of 64 (scripts/bench_train_step.py; warm, Qwen3.5-2B trains at
  ~14k tokens/s, a 200-step run without the rounding averaged 4-7k).
- Decision-1.0 removed its code from its model repos on 2026-09-27; scripts/exp_decision_models.py loads it from Sol-2B revision
  60ea30a4 (same prompt version as the weights). kev is fetched as a GitHub archive at a pinned commit.
- Every model, base and package must have an open licence (ai_experiments.licences.open_licence).

## 2026-09-25 (recorded 2026-10-07): Modal is where GPU work runs

The owner moved all GPU work to Modal (workspace `ynab`, shared with colleagues: this project touches only its own app,
`ai-experiments-training`, and volumes `ai-exp-results` and `ai-exp-hf-cache`). `scripts/modal_app.py` builds the image from `uv.lock`
and runs the repo's scripts unchanged; the client is a uv tool (`uv tool install modal`, then `modal setup`), not a project dependency.
Job lists launch with `--detach` so they survive the local client dying (WSL crashed on 2026-09-26 with jobs in flight). Results come
back with `modal volume get` into the gitignored `modal_out/` and `scripts/ingest_modal.py`. The container clock is UTC. Launches and
their cost are recorded in `reports/modal_launches.jsonl` and `reports/modal_costs.md` (from 2026-10-05). How-to: `CLAUDE.md`,
"GPU work: Modal"; first results in REPORT §54 Modal experiment infrastructure.

## 2026-10-04: the local 3090 is back for small jobs

The owner: "I'm okay to use local GPU sometimes when we can!" The RTX 3090 takes small, short jobs when free (smoke tests, small
encoders, synthetic reads) and every read of the owner's real budget, which stays on this machine. Long trainings and parallel batches
stay on Modal. The older rules still hold: one GPU job at a time, `just doctor --gpu` for the VRAM spill.

## 2026-10-06: GPU type per model on Modal

- Small encoders (bge-small, Ettin-32M late-interaction models) go on an L40S (a per-job `"gpu": "L40S"` in a job list): 2.3x the local
  3090 at about $0.18 per 3,000 steps (§188 The cheapest Modal GPU for the small encoder models).
- ModernBERT and Ettin on an H100 need FlashAttention from the Hub (`ATTN=kernels-community/flash-attn2@main`, with
  `uv run --frozen --with "kernels<0.11" ...`). With the default sdpa they run about 7x slower there (2.7 s a step against 0.35-0.41),
  which was the whole of §188's "H100 is 23x slower" (§200 Speed for longer queries and larger encoders). Ettin-32M stays on the L40S
  (flash is no faster there); Ettin-400M goes to the H100 with flash (half the time of the L40S at the same cost).
- CPU-bound jobs (clustering, household and data builds) take `--gpu cpu` (8 cores, no GPU billed); row 204 once held an H100 for two
  hours of CPU work.

## 2026-10-07: EmbeddingGemma 2 needs a newer transformers than unsloth allows

- EmbeddingGemma 2 needs transformers 5.18 or later and sentence-transformers 6.1, beyond the project lock. It runs as a uv overlay:
  `uv run --with "sentence-transformers>=6.1.0" --with "transformers>=5.18" --with torch==2.13.0 --with torchvision==0.28.0 python
  scripts/li_decider.py ...` (5.19 resolved at the time; §210 EmbeddingGemma 2 as the encoder).
- unsloth's releases (and main, 2026-10-07) cap transformers at 5.17. `uv run --with` and `UV_OVERRIDE` cannot lift the cap; a script's
  own `[tool.uv] override-dependencies` can. `scripts/unsloth_run.py` (`uv run --script scripts/unsloth_run.py scripts/li_decider.py
  train`, with `LOADER=unsloth`) pins transformers to the commit unsloth's EmbeddingGemma 2 notebook installs; release 5.19.0 trained
  to NaN under unsloth here (§211 EmbeddingGemma 2 under unsloth). unsloth's LoRA was no faster per step on an L40S, but it fits the
  model on the 3090 where plain transformers did not.

## 2026-10-07: py-spy needs root in WSL

py-spy cannot attach to a running process in WSL without root, so a hung read cannot be inspected from outside. Long scripts register
`faulthandler` on SIGUSR1 instead (first in `scripts/blend_eval.py`): `kill -USR1 <pid>` prints every thread's stack to the log.

## 2026-10-07: budget id out of tracked files; backup on D:

- The owner's budget id moved out of tracked files into `~/.config/ynab/budget_id` (0600), beside the token in `~/.config/ynab/token`.
  Chains read it with `BUDGET=${BUDGET:-$(cat ~/.config/ynab/budget_id)}`. Neither is ever printed or committed.
- Backup at `/mnt/d/Backup/ai-experiments_2026-10-07/`: `ai-experiments_all-refs.bundle` (a git bundle of every ref, ~520 MB) and
  `ai-experiments_folder.tar` (the whole checkout folder, ~361 GB, written at about 23 MB/s over WSL's D: mount, so about four hours;
  `tar.log` records the start and the exit code). Restore with `git clone <bundle>`, or untar.
