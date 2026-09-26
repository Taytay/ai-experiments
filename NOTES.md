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
