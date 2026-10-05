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
GPU: one H100, 6-hour timeout per call (edit `run` to change either; `--gpu H200` overrides the GPU for one launch).
Cost ledger (owner, 2026-10-05: "record how much each job cost, and which jobs were which"): every launch appends its app id, kind,
job list or reader, PLAN rows (env PLAN_ROWS, else read from the job-list name "r192_193") and job tags to reports/modal_launches.jsonl,
and each job's minutes when it reports back; `uv run python scripts/modal_costs.py` joins that with Modal's billing report.
"""
import datetime as dt
import json
import os
import re
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parents[1]
REPO, MNT = "/root/repo", "/mnt/repo"
APP = modal.App("ai-experiments-training")
HF = modal.Volume.from_name("ai-exp-hf-cache", create_if_missing=True)
OUT = modal.Volume.from_name("ai-exp-results", create_if_missing=True)
IGNORE = ["**/__pycache__/**", "**/*.pyc"]
LEDGER = LOCAL / "reports" / "modal_launches.jsonl"


def _rows(name=""):
    """PLAN rows for the ledger: PLAN_ROWS, else from a job-list or tag name ("r192_193" -> "192-193", "r190-g-s0" -> "190")"""
    if os.environ.get("PLAN_ROWS"):
        return os.environ["PLAN_ROWS"]
    m = re.match(r"r(\d+)(?:_(\d+))?", name)
    return (m.group(1) + (f"-{m.group(2)}" if m.group(2) else "")) if m else ""


def _ledger(**rec):
    """local side only: one JSON line per launch or finished job"""
    try:
        rec = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), app_id=APP.app_id, **rec)
        with open(LEDGER, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:  # never let bookkeeping break a launch
        print(f"[ledger] not written: {e}")


def _done(line):
    """the job runner's last line "<tag>: N files to ..., M min; last exit ..." -> a ledger entry"""
    m = re.match(r"^(\S+): \d+ files to .*?, ([\d.]+) min", str(line))
    if m:
        _ledger(kind="job_done", tag=m.group(1), minutes=float(m.group(2)))

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



@APP.function(image=image, gpu="H100", timeout=3 * 3600, volumes={"/cache": HF, "/out": OUT.read_only()}, max_containers=8)
def score_private(items: list, reader: str, layout: str, adapter_from: str) -> str:
    """Owner, 2026-10-02: score private items (a real budget) without keeping them on Modal. The items arrive as the call's argument and
    the scores leave as its return value; nothing is written to a volume (results mounted read-only, only to copy the adapter in) or to
    the repo tree; the items go to scripts/real_budget_eval.py `stream` on stdin and its scores come back on stdout. Logs carry counts
    only; a failure is re-raised without its message (it could quote a transaction)."""
    import json
    import os
    import shutil
    import subprocess
    _prepare()
    if adapter_from:
        shutil.copytree(f"/out/{adapter_from}/models/adapters", f"{REPO}/models/adapters", dirs_exist_ok=True)
    cmd = ("uv run --with transformers==5.17.0 --with flash-linear-attention --with 'peft>=0.21' --with torch==2.13.0 --with torchvision==0.28.0 "
           "python scripts/real_budget_eval.py stream")
    p = subprocess.run(cmd, shell=True, cwd=REPO, input=json.dumps(items), capture_output=True, text=True,
                       env={**os.environ, "BUDGET": "private", "READER": reader, "LAYOUT": layout})
    progress = [l for l in p.stderr.splitlines() if "items/s" in l]
    print(f"{len(items)} items, exit {p.returncode}; " + (progress[-1].strip() if progress else ""), flush=True)
    if p.returncode != 0:
        kinds = [l.split(":")[0] for l in p.stderr.splitlines() if l and not l.startswith(" ") and ("Error" in l.split(":")[0] or "Exception" in l.split(":")[0])]
        raise RuntimeError(f"scoring failed: {kinds[-1] if kinds else 'exit ' + str(p.returncode)}")
    return p.stdout


@APP.function(image=image, gpu="H100", timeout=3600, volumes={"/cache": HF, "/out": OUT.read_only()})
def embed_private(names: list, reader: str, adapter_from: str) -> bytes:
    """As score_private, for payee names: real_budget_eval.py `embed_stream` (names on stdin, float16 .npy bytes back); nothing kept."""
    import json
    import os
    import shutil
    import subprocess
    _prepare()
    if adapter_from:
        shutil.copytree(f"/out/{adapter_from}/models/adapters", f"{REPO}/models/adapters", dirs_exist_ok=True)
    cmd = ("uv run --with transformers==5.17.0 --with flash-linear-attention --with 'peft>=0.21' --with torch==2.13.0 --with torchvision==0.28.0 "
           "python scripts/real_budget_eval.py embed_stream")
    p = subprocess.run(cmd, shell=True, cwd=REPO, input=json.dumps(names).encode(), capture_output=True,
                       env={**os.environ, "BUDGET": "private", "READER": reader})
    print(f"{len(names)} names, exit {p.returncode}", flush=True)
    if p.returncode != 0:
        kinds = [l.split(":")[0] for l in p.stderr.decode(errors="replace").splitlines() if l and not l.startswith(" ") and "Error" in l.split(":")[0]]
        raise RuntimeError(f"embedding failed: {kinds[-1] if kinds else 'exit ' + str(p.returncode)}")
    return p.stdout


def embed_private_call(names: list, reader: str, adapter_from: str) -> bytes:
    with APP.run():
        _ledger(kind="private_scoring", what=f"owner-budget payee embedding {reader.split('@')[-1]}", rows=_rows(reader.split("@")[-1]),
                gpu="H100", items=len(names))
        return embed_private.remote(names, reader, adapter_from)


def private_scores(items: list, reader: str, layout: str, adapter_from: str, shards: int = 8, gpu: str = "H100"):
    """Local side: whole days per shard (the per-day cached prefix needs them together), shards scored in parallel; yields score lines."""
    from collections import defaultdict
    days = defaultdict(list)
    for it in items:
        days[it["date"]].append(it)
    parts = [[] for _ in range(shards)]
    for d in sorted(days, key=lambda d: -len(days[d])):  # largest days first, each to the lightest shard
        min(parts, key=len).extend(days[d])
    with APP.run():
        import time
        t0 = time.time()
        _ledger(kind="private_scoring", what=f"owner-budget scoring {reader.split('@')[-1]}", rows=_rows(reader.split("@")[-1]),
                gpu=gpu, shards=sum(1 for p in parts if p), items=len(items))
        fn = score_private if gpu == "H100" else score_private.with_options(gpu=gpu)  # the 35B: H200 (69 GB of weights)
        for out in fn.starmap([(p, reader, layout, adapter_from) for p in parts if p]):
            yield from out.splitlines()
        _ledger(kind="job_done", tag=f"owner-budget scoring {reader.split('@')[-1]}", minutes=round((time.time() - t0) / 60, 1))


@APP.local_entrypoint()
def main(cmd: str = "", env: str = "", tag: str = "", check: bool = False, jobs: str = "", gpu: str = ""):
    global run
    if gpu:  # row 125: another GPU type for this launch (e.g. H200 for decider-35B-A3B training); the default stays one H100
        run = run.with_options(gpu=gpu)
    if check:
        print(gpu_check.remote()); return
    if jobs:  # parallel: at most 8 containers (volume commits contend beyond ~5 concurrent small ones)
        spec = json.loads(Path(jobs).read_text())
        _ledger(kind="jobs", what=Path(jobs).stem, rows=_rows(Path(jobs).stem), gpu=gpu or "H100", tags=[j["tag"] for j in spec])
        for line in run.starmap([(j["cmds"], j.get("env", {}), j["tag"]) for j in spec], return_exceptions=True):
            print(line, flush=True)
            _done(line)
        return
    assert cmd and tag, "--cmd and --tag are required"
    envd = dict(kv.split("=", 1) for kv in env.split(",") if kv)
    _ledger(kind="jobs", what=tag, rows=_rows(tag), gpu=gpu or "H100", tags=[tag])
    line = run.remote([c.strip() for c in cmd.split(";;") if c.strip()], envd, tag)
    print(line)
    _done(line)
