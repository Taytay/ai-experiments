"""Run curriculum arms back to back, each in its own process (a CUDA OOM in one arm cannot
poison the next). Logs to results/curriculum_<model>_<arm>.log.

usage: uv run python experiments/run_curriculum.py [--model Qwen/Qwen2.5-3B] [--steps 800] [--lr 1e-4] ARM [ARM ...]
"""
import subprocess
import sys
import time
from pathlib import Path

args = sys.argv[1:]
model, steps, lr = "Qwen/Qwen2.5-3B", "800", "1e-4"
while args and args[0].startswith("--"):
    k, v, args = args[0], args[1], args[2:]
    if k == "--model": model = v
    elif k == "--steps": steps = v
    elif k == "--lr": lr = v
arms = args or ["base", "A", "B", "C", "Cn", "D", "base_m", "E"]
tag = model.split("/")[-1]
here = Path(__file__).parent
for arm in arms:
    log = here.parent / "results" / f"curriculum_{tag}_{arm}.log"
    t0 = time.time()
    print(f"=== arm {arm} -> {log.name}", flush=True)
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run([sys.executable, str(here / "exp_curriculum.py"), arm, model, steps, lr],
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"    exit {rc} after {(time.time() - t0) / 60:.1f} min", flush=True)
