"""Run this repo's experiment scripts on a Modal GPU (PLAN row 34, INFRA-1), unchanged, and bring the outputs back.

App `ai-experiments-training` in the owner's Modal workspace. The image is the project's own environment: Python 3.12 and
`uv sync --frozen` from pyproject.toml / uv.lock (torch 2.11 + cu128, unsloth), in a cached layer that rebuilds only when the
lock file changes. The code (src/, scripts/, data/processed/, evals/) is mounted at container start and copied into a writable
tree, so a code change does not rebuild the image. Hugging Face downloads persist in the `ai-exp-hf-cache` volume. After the
commands run, every file they created or changed under results/, models/adapters/ and evals/runs.jsonl is copied to the
`ai-exp-results` volume under <tag>/, with the command log.

usage (the Modal client is a uv tool, not a project dependency: `uv tool install modal`, then `modal setup`):
  modal run scripts/modal_app.py --check                                     GPU, torch and unsloth smoke (a few cents)
  modal run scripts/modal_app.py --tag T --env "LOAD_4BIT=1,ALL_LABELS=1" \\
      --cmd "uv run --frozen python scripts/exp_categoriser.py llm none ;; uv run --frozen python scripts/exp_real6.py llm <adapter>"
  modal run scripts/modal_app.py --jobs scripts/modal_jobs/<file>.json     many jobs in parallel (at most 8 containers):
      a JSON list of {"tag": ..., "env": {...}, "cmds": [...]}; each job writes only its own <tag>/ directory
  modal volume get ai-exp-results T <local dir>                              then copy into the checkout, push-models, union runs.jsonl
GPU: one H100, 6-hour timeout per call (edit `run` to change either).
"""
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parents[1]
REPO, MNT = "/root/repo", "/mnt/repo"
APP = modal.App("ai-experiments-training")
HF = modal.Volume.from_name("ai-exp-hf-cache", create_if_missing=True)
OUT = modal.Volume.from_name("ai-exp-results", create_if_missing=True)
IGNORE = ["**/__pycache__/**", "**/*.pyc"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("gcc", "g++", "git")  # triton compiles its kernel launchers at run time
    .pip_install("uv")
    .add_local_file(LOCAL / "pyproject.toml", f"{REPO}/pyproject.toml", copy=True)
    .add_local_file(LOCAL / "uv.lock", f"{REPO}/uv.lock", copy=True)
    .add_local_file(LOCAL / "README.md", f"{REPO}/README.md", copy=True)
    .run_commands(f"cd {REPO} && UV_LINK_MODE=copy uv sync --frozen --no-install-project")
    .env({"HF_HOME": "/cache/hf", "PYTHONUTF8": "1", "UV_LINK_MODE": "copy", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(LOCAL / "src", f"{MNT}/src", ignore=IGNORE)
    .add_local_dir(LOCAL / "scripts", f"{MNT}/scripts", ignore=IGNORE)
    .add_local_dir(LOCAL / "data" / "processed", f"{MNT}/data/processed", ignore=IGNORE)
    .add_local_dir(LOCAL / "evals", f"{MNT}/evals", ignore=IGNORE + ["runs.db"])
)


def _prepare():
    """The mounted code copied into the writable repo tree beside the prebuilt .venv; the tracker db rebuilt from runs.jsonl."""
    import shutil
    import subprocess
    for d in ("src", "scripts", "data/processed", "evals"):
        shutil.copytree(f"{MNT}/{d}", f"{REPO}/{d}", dirs_exist_ok=True)
    for d in ("results/per_item", "models/adapters", "logs"):
        Path(REPO, d).mkdir(parents=True, exist_ok=True)
    subprocess.run("uv run --frozen evals rebuild", shell=True, cwd=REPO, check=True)


@APP.function(image=image, gpu="H100", timeout=900, volumes={"/cache": HF})
def gpu_check():
    import subprocess
    _prepare()
    out = subprocess.run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; "
                         "uv run --frozen python -c \"import torch, unsloth; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), "
                         "torch.cuda.get_device_name(0)); print('unsloth', unsloth.__version__)\"",
                         shell=True, cwd=REPO, capture_output=True, text=True)
    return out.stdout + out.stderr[-3000:]


def _snapshot():
    return {str(p): p.stat().st_mtime for base in ("results", "models/adapters", "evals") for p in Path(REPO, base).rglob("*") if p.is_file()}


@APP.function(image=image, gpu="H100", timeout=6 * 3600, volumes={"/cache": HF, "/out": OUT}, max_containers=8)
def run(cmds: list, env: dict, tag: str):
    import os
    import shutil
    import subprocess
    import time
    _prepare()
    for src in [s for s in env.get("ADAPTERS_FROM", "").split(",") if s]:  # adapters trained by earlier jobs, from their result directories
        shutil.copytree(f"/out/{src}/models/adapters", f"{REPO}/models/adapters", dirs_exist_ok=True)
    before = _snapshot(); t0 = time.time(); log = []
    for cmd in cmds:
        started = time.strftime("%H:%M:%S")
        print(f"### {started} {cmd}", flush=True)
        p = subprocess.Popen(cmd, shell=True, cwd=REPO, env={**os.environ, **env}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        lines = []
        for line in p.stdout:  # streamed: shows in `modal run` and the dashboard's logs as it happens
            print(line, end="", flush=True); lines.append(line)
        p.wait()
        log.append(f"### {started} exit {p.returncode}: {cmd}\n{''.join(lines)[-200000:]}")
        if p.returncode != 0:
            break
    dest = Path("/out", tag); dest.mkdir(parents=True, exist_ok=True)
    changed = [f for f, m in _snapshot().items() if before.get(f) != m]
    for f in changed:
        rel = Path(f).relative_to(REPO); (dest / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dest / rel)
    (dest / "modal_run.log").write_text("\n".join(log) + f"\n### wall {round((time.time() - t0) / 60, 1)} min, {len(changed)} files\n")
    OUT.commit(); HF.commit()
    return f"{tag}: {len(changed)} files to ai-exp-results/{tag}, {round((time.time() - t0) / 60, 1)} min; last exit {log[-1].split(':')[0]}"



@APP.local_entrypoint()
def main(cmd: str = "", env: str = "", tag: str = "", check: bool = False, jobs: str = ""):
    if check:
        print(gpu_check.remote()); return
    if jobs:  # parallel: at most 8 containers (volume commits contend beyond ~5 concurrent small ones)
        import json
        spec = json.loads(Path(jobs).read_text())
        for line in run.starmap([(j["cmds"], j.get("env", {}), j["tag"]) for j in spec], return_exceptions=True):
            print(line, flush=True)
        return
    assert cmd and tag, "--cmd and --tag are required"
    envd = dict(kv.split("=", 1) for kv in env.split(",") if kv)
    print(run.remote([c.strip() for c in cmd.split(";;") if c.strip()], envd, tag))
