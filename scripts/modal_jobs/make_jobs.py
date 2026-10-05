"""PLAN step 208, E2 (owner, 2026-10-05: "Let's not duplicate work here!"; reports/workflow_review_2026-10-05.md item 3): one job-list
generator for decider rows, so a row trains and reads only what it compares and prints its cost before anything launches.

A spec (scripts/modal_jobs/specs/r<row>.json) names the row, its arms and its read sets:
  {"row": "209", "base": {...optional env overrides for every arm...},
   "arms": {"g4": {"REALSTYLE_FILE": "realstyle_v4g_train.jsonl"}, "k4": {"KNOWLEDGE": "0.4", ...}},
   "reads": ["realstyle_v4g_test"],                  read by every job (the row's primary sets only)
   "variant_reads": {"k4": ["realstyle_v4g_seen_test"]},   extra sets for named arms only
   "drills": false,                                  the blind and drill sets (blind_v1, mislead_v1, ...) only when the row is about them
   "seeds": [0],                                     screen with one seed; add 1 later for winners or close calls
   "read_only": {"g4": "r190-g"}}                    arms already trained: read with ADAPTERS_FROM=<tag>-s<seed>, no training
Adapter names come from scripts/exp_decider_finetune.py itself (imported with each arm's env), so they always match training.
The estimate: training STEPS x 5.2 s (measured 5.0-5.2 s per step on the H100), each read's median minutes from evals/runs.jsonl
(6 min when a set has no history) plus 0.5 min process start (with READER=vllm: one engine, 2.5 min, and 0.45 of the HF
minutes, measured on the H100 in row 208), 5 min container start; $4.09 per H100-hour (reports/modal_costs.md).
usage: uv run python scripts/modal_jobs/make_jobs.py scripts/modal_jobs/specs/r<row>.json   -> scripts/modal_jobs/r<row>.json + estimate
"""
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UV = ('uv run --frozen --with transformers==5.17.0 --with flash-linear-attention --with "peft>=0.21" --with torch==2.13.0 '
      '--with torchvision==0.28.0 python')
BASE = {"DBEP": "0.5", "FOLD": "0", "RENAME": "0.5", "MICRO": "16", "STEPS": "800", "ALL_LABELS": "1", "AUX_LM": "1", "LABELS": "rand255",
        "MISLEAD": "mislead_v1", "ALT": "0.1", "ALT_SOFT": "1", "LOOKUP": "0.1", "OVERRIDE": "0.1", "EVFREE": "0.1", "EVFREE_MODE": "soft",
        "MODEL": "Mapika/decider-4b", "EMPTY": "20", "LAYOUT": "labelled_shots", "OTHERS": "0.5", "REALSTYLE": "0.25",
        "REALSTYLE_FILE": "realstyle_v4g_train.jsonl",  # the r190 recipe (G4)
        "MICRO_SPLIT": "4", "READER": "vllm"}  # row 208 (§181): length-sorted groups (same model, -28% training) and vLLM reads (2.3x)
DRILLS = ["blind_v1_others", "blind_v1", "", "real6_v1_novel", "mislead_v1", "override_v1", "blind_v2_others", "blind_v2"]
PER_HOUR, STEP_S, START_MIN, DEFAULT_READ, READ_START = 4.09, 5.2, 5.0, 6.0, 0.5


def adapter_name(env):
    code = "import sys; sys.argv=['x']; import exp_decider_finetune as F; print(F.NAME)"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT / "scripts", env={**os.environ, **env}, capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f"adapter name failed for {env}: {out.stderr[-500:]}")
    return out.stdout.strip().splitlines()[-1]


def users_of(item_set):
    """USERS for a read: the users in the item set's file, else those an earlier job list used for it"""
    if not item_set:  # the default REAL-6 set reads every user
        return []
    f = ROOT / "data" / "processed" / f"{item_set}.json"
    if f.exists() and item_set.startswith("realstyle"):
        return sorted({it["user"] for it in json.loads(f.read_text())["items"] if "user" in it})
    for jl in sorted((ROOT / "scripts" / "modal_jobs").glob("r*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        for j in json.loads(jl.read_text()):
            for c in j.get("cmds", []):
                if f"ITEMS_SET={item_set} " in c:  # the newest job list that read this set decides, with or without USERS
                    return c.split(" USERS=")[1].split(" ")[0].split(",") if " USERS=" in c else []
    return []


def read_minutes():
    """median minutes per ITEMS_SET from the tracker (decision_models runs)"""
    by = defaultdict(list)
    for line in open(ROOT / "evals" / "runs.jsonl"):
        r = json.loads(line)
        if r.get("experiment") != "decision_models":
            continue
        try:
            m = (float(r["finished_at"]) - float(r["started_at"])) / 60
        except (KeyError, TypeError, ValueError):
            continue
        cfg = r.get("config", {})
        by[str(cfg.get("ITEMS_SET", cfg.get("items_set", "")))].append(m)
    return {k: statistics.median(v) for k, v in by.items() if v}


def read_cmd(adapter, item_set):
    users = users_of(item_set)
    u = f" USERS={','.join(map(str, users))}" if users else ""
    return (f"FAMILY=decider MODEL=Mapika/decider-4b ADAPTER={adapter} CONDS=noctx LABELS=rand255 LAYOUT=labelled_shots "
            f"ITEMS_SET={item_set}{u} {UV} scripts/exp_decision_models.py")


def main(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    row = spec["row"]
    mins = read_minutes()
    jobs, est = [], []
    for arm, over in spec["arms"].items():
        for seed in spec.get("seeds", [0]):
            env = {**BASE, **spec.get("base", {}), **over, "SEED": str(seed)}
            split = int(env.get("MICRO_SPLIT", "1"))  # the run tag names steps and split so no adapter overwrites another (row 190's were st800)
            env["RUN_TAG"] = f"h100bf16st{env.get('STEPS', '800')}" + (f"ms{split}" if split > 1 else "") + (f"s{seed}" if seed else "")
            name = adapter_name(env)
            sets = list(spec.get("reads", [])) + list(spec.get("variant_reads", {}).get(arm, []))
            if spec.get("drills"):
                sets += DRILLS
            tag = f"r{row}-{arm}-s{seed}"
            ro = spec.get("read_only", {}).get(arm)
            if env.get("READER") == "vllm" and sets:  # row 208: one process, one vLLM engine for all reads
                rf = ROOT / "scripts" / "modal_jobs" / "reads" / f"{tag}.json"
                rf.parent.mkdir(parents=True, exist_ok=True)
                rf.write_text(json.dumps([dict(ITEMS_SET=s, USERS=",".join(map(str, users_of(s)))) for s in sets], indent=1))
                reads = [f"FAMILY=decider MODEL=Mapika/decider-4b ADAPTER={name} CONDS=noctx LABELS=rand255 LAYOUT=labelled_shots "
                         f"READS_FILE={rf.relative_to(ROOT)} {UV} scripts/exp_decision_models.py"]
            else:
                reads = [read_cmd(name, s) for s in sets]
            cmds = ([] if ro else [f"{UV} scripts/exp_decider_finetune.py"]) + reads
            job_env = {**env, **({"ADAPTERS_FROM": f"{ro}-s{seed}"} if ro else {})}
            jobs.append(dict(tag=tag, env=job_env, cmds=cmds))
            step_s = STEP_S if split == 1 else 3.8  # row 208: 50.2 min for 800 steps with MICRO_SPLIT=4
            train_min = 0 if ro else int(env.get("STEPS", 800)) * step_s / 60
            vllm = env.get("READER") == "vllm"
            read_min = (2.5 + sum(mins.get(s, DEFAULT_READ) * 0.45 for s in sets)) if vllm else sum(mins.get(s, DEFAULT_READ) + READ_START for s in sets)
            est.append((tag, train_min, read_min, len(sets)))
    out = ROOT / "scripts" / "modal_jobs" / f"r{row}.json"
    out.write_text(json.dumps(jobs, indent=1))
    print(f"{len(jobs)} jobs -> {out}\n")
    print("| job | train min | reads | read min | est. $ |")
    print("|---|---|---|---|---|")
    total = 0.0
    for tag, tm, rm, n in est:
        cost = (tm + rm + START_MIN) / 60 * PER_HOUR
        total += cost
        print(f"| {tag} | {tm:.0f} | {n} | {rm:.0f} | {cost:.2f} |")
    print(f"\nestimated total ${total:.2f} at ${PER_HOUR}/H100-hour (at most 8 containers at once)")


if __name__ == "__main__":
    main(sys.argv[1])
