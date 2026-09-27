"""Tables for PLAN step 66 (REAL-21): does a places database taught as training episodes store facts or teach a skill? Qwen3.5-2B on
POI-1 (fold 0 held out, plain layout, rename 0.5, all-label loss) without database episodes, and with them (DBEP 0.5 over
data/processed/poi1_v1_db.json: half of POI-1's test places, chosen by a hash of the Overture id). Fold 0's test items split by
whether their place is in the database: the in-database gain over the no-database arm is facts plus skill, the held-out gain is
skill alone, and their difference is what the weights store about the particular places.

  F.1  top-1 on in-database and held-out places, with the gains over the no-database arm and their difference (a user-bootstrap
       interval on the difference), overall and by seen / unseen kind
usage: uv run python scripts/facts_skill_tables.py
"""
import json
import warnings

import numpy as np

from ai_experiments import real6_cells as RC
from ai_experiments.paths import PROCESSED

C = "real6_categoriser_Qwen3.5-2B_poi1_v1_none_{tag}_hf{db}_f0_ren50_alllab{ep}_lora_poi1_v1_hfs.noctx.jsonl"
ARMS = [("no database episodes, 800 steps", C.format(tag="h100bf16", db="", ep="")),
        ("database episodes, 800 steps", C.format(tag="h100bf16", db="_db", ep="_dbep50")),
        ("database episodes, 1,600 steps", C.format(tag="h100bf16st1600", db="_db", ep="_dbep50"))]


def ok(r):
    return int(np.argmax(r["sum_lp"])) == r["answer"]


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in json.loads((PROCESSED / "poi1_v1.json").read_text())["items"]}
    in_db = set(json.loads((PROCESSED / "poi1_v1_db.json").read_text())["in_db_items"])
    f0 = [i for i in items if items[i]["user"] % 4 == 0]
    recs = [{i: r for i, r in RC.load_recs(f).items() if i in items} for _, f in ARMS]
    base = recs[0]
    print(f"**Table F.1: POI-1 fold 0 (in the database: {sum(i in in_db for i in f0)} items; held out: {sum(i not in in_db for i in f0)}), "
          "top-1 %, and the gain over the no-database arm; facts = in-database gain minus held-out gain (user-bootstrap 95% interval)**\n")
    for sub, sel in (("all items", lambda i: True), ("seen kind", lambda i: items[i]["seen"]), ("unseen kind", lambda i: not items[i]["seen"])):
        print(f"*{sub}*\n")
        print("| arm | in database | held out | gain in database | gain held out | facts (difference) [interval] |"); print("|---|---|---|---|---|---|")
        for (lab, _), r in zip(ARMS, recs):
            if not r:
                continue
            ids = [i for i in f0 if sel(i) and i in r and i in base]
            a_in = np.mean([ok(r[i]) for i in ids if i in in_db]); a_out = np.mean([ok(r[i]) for i in ids if i not in in_db])
            b_in = np.mean([ok(base[i]) for i in ids if i in in_db]); b_out = np.mean([ok(base[i]) for i in ids if i not in in_db])
            users = sorted({items[i]["user"] for i in ids}); rng = np.random.default_rng(0); boots = []
            by_u = {u: [i for i in ids if items[i]["user"] == u] for u in users}
            for _ in range(1000):
                s = [i for u in rng.choice(users, len(users)) for i in by_u[u]]
                d = lambda subset: np.mean([ok(r[i]) - ok(base[i]) for i in subset]) if subset else 0.0  # noqa: E731
                boots.append(d([i for i in s if i in in_db]) - d([i for i in s if i not in in_db]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f"| {lab} | {100 * a_in:.1f} | {100 * a_out:.1f} | {100 * (a_in - b_in):+.1f} | {100 * (a_out - b_out):+.1f} | "
                  f"{100 * ((a_in - b_in) - (a_out - b_out)):+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] |")
        print()
