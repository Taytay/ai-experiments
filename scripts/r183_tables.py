"""PLAN step 183 tables: other households' filings ("Others filed this payee as: ...") against what training on other households puts in
the weights. Arms G3 (shared-world households, no crowd lines) and GC (with them), two seeds each, each read on held-out households with
and without the line, in a new world (test merchants: the weights cannot know them; crowd from 800 test-world households) and in the
training world (training merchants: other users filed them in training; crowd from the 800 training households). Rows split by the
payee being new to the household and by how many other households filed its bank string (the crowd table's count, the household left out).
PLAN step 190: V4=1 reads the v4 world (both generator fixes; crowd keys v2, set CROWD_KEY=v2) with arms G4 / GCD4 / GCDH4.
usage: [V4=1 CROWD_KEY=v2] uv run python scripts/r183_tables.py
"""
import json
import os
from collections import defaultdict

import numpy as np

from ai_experiments.paths import PROCESSED, ROOT

sys_path = __import__("sys").path
sys_path.insert(0, str(ROOT / "scripts"))
from build_crowd import crowd_key  # noqa: E402

R = ("results/per_item/real6_dm_decider_decider_decider-4b_none_h100bf16st800{seed}_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_oth50_"
     "aux100_labrand255_laylabelled_shots_ev10soft_rs25{v}_lora_{set}_labrand255_laylabelled_shots.noctx.jsonl")
ARMS = {"G3 (no crowd lines in training)": "v3g", "GC (crowd lines in training)": "v3gc", "GCD (crowd lines, 40% dropped)": "v3gcd", "GCDH (+ 30% of households with none)": "v3gcdh"}
WORLDS = {"new world (merchants unseen in training)": ("realstyle_v3g_test", "realstyle_v3gc_test", "realstyle_crowd_test.json"),
          "training world (merchants other users filed in training)": ("realstyle_v3g_seen_test", "realstyle_v3gc_seen_test", "realstyle_crowd_train.json")}
if os.environ.get("V4") == "1":
    ARMS = {"G4 (no crowd lines in training)": "v4g", "GCD4 (crowd lines, 40% dropped)": "v4gcd", "GCDH4 (+ 30% of households with none)": "v4gcdh"}
    WORLDS = {"new world (merchants unseen in training)": ("realstyle_v4g_test", "realstyle_v4gc_test", "realstyle_crowd_v4_test.json"),
              "training world (merchants other users filed in training)": ("realstyle_v4g_seen_test", "realstyle_v4gc_seen_test", "realstyle_crowd_v4_train.json")}
BUCKETS = [("0-1", 0, 1), ("2-9", 2, 9), ("10-49", 10, 49), ("50+", 50, 10 ** 9)]


def acc(path, ids):
    recs = {r["id"]: r for r in map(json.loads, open(ROOT / path))}
    xs = [recs[i]["correct"] for i in ids if i in recs]
    return 100 * np.mean(xs) if xs else float("nan")


if __name__ == "__main__":
    for world, (plain, withline, crowdf) in WORLDS.items():
        items = json.loads((PROCESSED / f"{withline}.json").read_text())["items"]
        crowd = json.loads((PROCESSED / crowdf).read_text())["keys"]

        def others(it):
            v = crowd.get(crowd_key(it["raw"]), {})
            return len(set().union(*map(set, v.values())) - {it["user"]}) if v else 0
        groups = defaultdict(list)
        for it in items:
            groups["all"].append(it["id"])
            groups["payee filed before" if it["payee_seen"] else "first-time payee"].append(it["id"])
            if not it["payee_seen"]:
                n = others(it)
                for b, lo, hi in BUCKETS:
                    if lo <= n <= hi:
                        groups[f"first-time, {b} other households"].append(it["id"])
        print(f"\n**{world}: % right first, mean of seeds 0 and 1 [each seed]**\n")
        cols = [(arm, v, s, read) for arm, v in ARMS.items() for read, s in (("no line", plain), ("crowd line", withline))]
        print("| items | n | " + " | ".join(f"{a.split(' ')[0]}, {r}" for a, _, _, r in cols) + " |")
        print("|---|---|" + "---|" * len(cols))
        for g, ids in groups.items():
            cells = []
            for _, v, s, _ in cols:
                vals = [acc(R.format(seed=seed, v=v, set=s), ids) for seed in ("", "s1")]
                cells.append(f"{np.mean(vals):.1f} [{vals[0]:.1f} / {vals[1]:.1f}]")
            print(f"| {g} | {len(ids)} | " + " | ".join(cells) + " |")
