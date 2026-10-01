"""PLAN step 143 (owner, 2026-09-30: overlap of payee names and category trends between training and test users is intended; the separate
merchant database teaches the model about certain payees). For blind_v1 and blind_v2: whether each query's payee is known to training,
and how the models read each group:
  in the training merchant catalogue (REAL-6's users and the merchant database, incl. the misleading-name set)
  in the Overture business-name pool (OVDB episodes) / the kinds catalogue (income, bills, ...), not in the catalogue
  novel: in none of them
Payee names are matched on a normalised form (lower case, letters and digits only). Top-1 of decider-4B (the other-users line, three
seeds), the untrained 35B, and the recommended routing (35B for first-time payees only, REPORT 139).
usage: uv run python scripts/overlap_tables.py
"""
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments import kinds as K
from ai_experiments import transactions as T
from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
from blind2_tables import readers  # noqa: E402
from decision_tables import folds_calibrate, product  # noqa: E402


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    cat = {norm(m["name"]) for m in T.load()["merchants"]}
    for f in ("mislead_v1.json",):
        try:
            d = json.loads((PROCESSED / f).read_text())
            cat |= {norm(x["merchant"]) for x in d.get("items", []) if x.get("merchant")}
        except FileNotFoundError:
            pass
    ov = {norm(p["name"]) for p in json.loads((PROCESSED / "overture_pool_v1.json").read_text())["places"]}
    kd = {norm(p) for k in K.KINDS for p in K.KINDS[k]["payees"]}
    print("| set | payee known to training as | n | share | ideal reader | decider-4B | 35B untrained | routed system |")
    print("|---|---|---|---|---|---|---|---|")
    for v in (1, 2):
        items = {i["id"]: i for i in json.loads((PROCESSED / f"blind_v{v}_others.json").read_text())["items"]}
        ph = json.loads((PROCESSED / f"blind_v{v}_payeehist.json").read_text())
        rd = readers(v); decs, big = rd["decider-4B + other-users line"], rd["decider-35B untrained"][0]
        grp = {}
        for i, x in items.items():
            n = norm(x["merchant"])
            g = "merchant catalogue / database" if n in cat else "Overture pool" if n in ov else "kinds catalogue" if n in kd else "novel"
            if v == 1 and g in ("Overture pool", "kinds catalogue"):  # filtered out of training for blind_v1 (build_overture_pool, kinds.payees)
                g = "novel (listed, filtered from training)"
            grp[i] = g
        routed = []
        for d in decs:
            Pd = folds_calibrate(d, items); Pb = product([d, big], items)
            routed.append({i: (Pb[i] if not ph[i] else Pd[i]) for i in Pd if i in Pb})
        for g in ("merchant catalogue / database", "Overture pool", "kinds catalogue", "novel (listed, filtered from training)", "novel"):
            ids = [i for i in items if grp[i] == g]
            if not ids:
                continue
            acc = lambda recs: np.mean([100 * np.mean([int(np.argmax(r[i])) == items[i]["answer"] for i in ids if i in r]) for r in recs])  # noqa: E731
            print(f"| blind_v{v} | {g} | {len(ids)} | {100 * len(ids) / len(items):.0f}% | {100 * np.mean([items[i]['best'] == items[i]['answer'] for i in ids]):.0f} | "
                  f"{acc(decs):.1f} | {acc([big]):.1f} | {acc(routed):.1f} |")
