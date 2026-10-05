"""Job list r192_193.json (one list so the app's cap of 8 containers holds), derived from r190.json's commands.
Row 192: G4 and GCD4 (seeds 0, 1; adapters from r190) read the v4 test sets with n-gram or behavioural similar rows, without and with
the crowd line. Row 193: three new arms, two seeds: K4 (v4g + knowledge episodes 20k x 6.5 at 40%), CDB4 (crowd line 40% dropped +
behavioural similar rows in training, realstyle_v4gcdb), KCDB4 (both); trained, then read on the usual sets, row 190's v4 sets and
row 192's sets.
usage: uv run python scripts/modal_jobs/make_r192_r193.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
T = json.loads((HERE / "r190.json").read_text())
NEW = [f"realstyle_v4{c}{s}_{src}_test" for src in ("ngram", "behav") for c in ("g", "gc") for s in ("", "_seen")]


def read(cmd, item_set):
    """r190's read of realstyle_v4g_test (or _seen) pointed at another item set"""
    return cmd.replace("ITEMS_SET=realstyle_v4g_seen_test", f"ITEMS_SET={item_set}").replace("ITEMS_SET=realstyle_v4g_test", f"ITEMS_SET={item_set}")


def reads(job, sets):
    base = {s: next(c for c in job["cmds"] if f"ITEMS_SET=realstyle_v4g{s}_test " in c) for s in ("", "_seen")}
    return [read(base["_seen" if "_seen_" in x else ""], x) for x in sets]


jobs = []
for j in T:  # row 192: scoring only
    if j["tag"].startswith(("r190-g-", "r190-gcd-")):
        jobs.append(dict(tag=j["tag"].replace("r190", "r192"), env=dict(j["env"], ADAPTERS_FROM=j["tag"]), cmds=reads(j, NEW)))
KN = "merchant_knowledge_v1_20k6_train.jsonl"
for arm, rs, know in (("k4", "v4g", True), ("cdb4", "v4gcdb", False), ("kcdb4", "v4gcdb", True)):
    for j in (x for x in T if x["tag"].startswith("r190-g-")):
        env = dict(j["env"], REALSTYLE_FILE=f"realstyle_{rs}_train.jsonl")
        sfx = "rs25v4g"
        new = f"rs25{rs}" + ("_mk40_20k6" if know else "")
        if know:
            env.update(KNOWLEDGE="0.4", KNOWLEDGE_FILE=KN)
        cmds = [c.replace(f"{sfx}_lora", f"{new}_lora") for c in j["cmds"] + reads(j, NEW)]
        jobs.append(dict(tag=j["tag"].replace("r190-g", f"r193-{arm}"), env=env, cmds=cmds))
(HERE / "r192_193.json").write_text(json.dumps(jobs, indent=1))
for j in jobs:
    print(j["tag"], len(j["cmds"]), j["env"].get("REALSTYLE_FILE"), j["env"].get("KNOWLEDGE", ""), j["env"].get("ADAPTERS_FROM", ""))
