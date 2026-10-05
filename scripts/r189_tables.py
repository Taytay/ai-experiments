"""PLAN step 189 tables: similar-payee rows chosen by behaviour (other households' filings over cross-household category clusters, fused
with character n-grams; SIM_SRC=behav) against by character n-grams alone (SIM_SRC=ngram) and against no similar rows (the household's own
rows only), read by G3 and GCD (seeds 0 and 1) without the crowd line, in the new world and the training world; rows split by the payee
being new to the household and by how many other households filed its bank string.
usage: uv run python scripts/r189_tables.py
"""
import json
import sys
from collections import defaultdict

import numpy as np

from ai_experiments.paths import PROCESSED, ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from build_crowd import crowd_key  # noqa: E402
from r183_tables import BUCKETS, R, acc  # noqa: E402

ARMS = {"G3": "v3g", "GCD": "v3gcd"}
WORLDS = {"new world": ("realstyle_v3g{}_test", "realstyle_crowd_test.json"),
          "training world": ("realstyle_v3g_seen{}_test", "realstyle_crowd_train.json")}
READS = {"own rows only": "", "n-gram similar rows": "_ngram", "behavioural similar rows": "_behav"}

if __name__ == "__main__":
    for world, (pat, crowdf) in WORLDS.items():
        items = json.loads((PROCESSED / f"{pat.format('')}.json").read_text())["items"]
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
        print(f"\n**{world}: % right first, mean of seeds 0 and 1 [each seed]; no crowd line**\n")
        cols = [(a, v, r, s) for a, v in ARMS.items() for r, s in READS.items()]
        print("| items | n | " + " | ".join(f"{a}, {r}" for a, _, r, _ in cols) + " |")
        print("|---|---|" + "---|" * len(cols))
        for g, ids in groups.items():
            cells = []
            for _, v, _, s in cols:
                vals = [acc(R.format(seed=seed, v=v, set=pat.format(s)), ids) for seed in ("", "s1")]
                cells.append(f"{np.mean(vals):.1f} [{vals[0]:.1f} / {vals[1]:.1f}]")
            print(f"| {g} | {len(ids)} | " + " | ".join(cells) + " |")
