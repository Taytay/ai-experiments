"""PLAN steps 192-193 tables, v4 world: % right first by reader arm (G4, GCD4 from row 190; K4, CDB4, KCDB4 from row 193) and by read
(similar rows: own rows only, character n-grams, behavioural two-hop; each without and with the crowd line), on held-out households in a
new world and in the training world; for all items, payees filed before, and first-time payees. Means of seeds 0 and 1 [each seed].
usage: CROWD_KEY=v2 uv run python scripts/r192_tables.py
"""
import json
import sys

import numpy as np

from ai_experiments.paths import PROCESSED, ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from r183_tables import R, acc  # noqa: E402

ARMS = {"G4": "v4g", "GCD4": "v4gcd", "K4": "v4g_mk40_20k6", "CDB4": "v4gcdb", "KCDB4": "v4gcdb_mk40_20k6"}
READS = {"own rows": "", "n-gram rows": "_ngram", "behavioural rows": "_behav"}
WORLDS = {"new world": "", "training world": "_seen"}

if __name__ == "__main__":
    for world, w in WORLDS.items():
        items = json.loads((PROCESSED / f"realstyle_v4g{w}_test.json").read_text())["items"]
        groups = {"all": [i["id"] for i in items], "payee filed before": [i["id"] for i in items if i["payee_seen"]],
                  "first-time payee": [i["id"] for i in items if not i["payee_seen"]]}
        for g, ids in groups.items():
            print(f"\n**{world}, {g} (n={len(ids)}): % right first, mean of seeds 0 and 1 [each seed]**\n")
            cols = [(r, s, line) for line in ("", "c") for r, s in READS.items()]
            print("| arm | " + " | ".join(f"{r}{', crowd line' if line else ''}" for r, _, line in cols) + " |")
            print("|---|" + "---|" * len(cols))
            for arm, v in ARMS.items():
                cells = []
                for _, s, line in cols:
                    vals = [acc(R.format(seed=seed, v=v, set=f"realstyle_v4g{line}{w}{s}_test"), ids) for seed in ("", "s1")]
                    cells.append("-" if any(np.isnan(vals)) else f"{np.mean(vals):.1f} [{vals[0]:.1f} / {vals[1]:.1f}]")
                print(f"| {arm} | " + " | ".join(cells) + " |")
