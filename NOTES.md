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
