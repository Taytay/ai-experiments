# GPU / unsloth readiness check

Goal: confirm the NVIDIA driver on this machine is good enough to fine-tune a
small embeddings model with unsloth.

## Hardware / driver (observed 2026-09-12)

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090, 24 GiB, WDDM mode |
| Driver | 591.86 (Windows driver 32.0.15.9186, dated 2026-01-19) |
| Max CUDA runtime supported by driver | 13.1 |
| CUDA toolkit on disk | v12.3 (not needed; torch wheels bundle their own runtime) |
| WSL | Ubuntu 18.04 only (too old for current torch/triton), so native Windows is used |

## Steps

1. Install `uv` via winget, create a Python 3.12 venv in `.venv/`.
2. `check_gpu.py` - torch sees the GPU and runs fp32/fp16/bf16 kernels.
3. Install unsloth + deps, run a tiny embeddings fine-tune (`finetune_embed.py`).

## Results so far

- `check_gpu.py`: PASS. torch 2.11.0+cu128 sees the RTX 3090 (sm_86), 22.8 GiB free,
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
