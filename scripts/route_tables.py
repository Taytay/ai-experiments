"""PLAN step 142 (after REPORT 138: the untrained 35B overrides the user's own recent pattern, changed minds 44 -> 5 on blind_v2): route
by payee history. Top-1 and effort (per-item rule, W = 10) on blind_v1 and blind_v2: decider alone; decider -> 35B for what decider does
not auto-file (REPORT 138); decider x 35B for every transaction; decider x 35B only for first-time payees (no earlier filing), decider
alone for known payees; by subset.
usage: uv run python scripts/route_tables.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
from blind2_tables import effort, readers, rule_of  # noqa: E402
from decision_tables import folds_calibrate, product  # noqa: E402

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("| set | system | top-1 all | first-time payees | known payees | changed mind / weekday / trip | effort W=10 | 35B reads % |")
    print("|---|---|---|---|---|---|---|---|")
    for v in (1, 2):
        items = {i["id"]: i for i in json.loads((PROCESSED / f"blind_v{v}_others.json").read_text())["items"]}
        ph = json.loads((PROCESSED / f"blind_v{v}_payeehist.json").read_text())
        rd = readers(v); decs, big = rd["decider-4B + other-users line"], rd["decider-35B untrained"][0]
        habit = [i for i, x in items.items() if x["why"] in ("changed_mind", "weekday_split", "trip")]
        first = [i for i in items if not ph[i]]; known = [i for i in items if ph[i]]
        rows = {}
        for d in decs:
            Pd = folds_calibrate(d, items); Pb = product([d, big], items); ad, ab = rule_of(Pd, 10), rule_of(Pb, 10)
            ids = [i for i in Pd if i in Pb]
            sysP = {"decider": (Pd, ad, 0.0),
                    "decider -> 35B (unsettled)": ({i: Pd[i] if ad[i] else Pb[i] for i in ids}, {i: ad[i] or ab[i] for i in ids}, np.mean([not ad[i] for i in ids])),
                    "decider x 35B (all)": (Pb, ab, 1.0),
                    "decider x 35B for first-time payees only": ({i: Pb[i] if not ph[i] else Pd[i] for i in ids}, {i: ab[i] if not ph[i] else ad[i] for i in ids}, np.mean([not ph[i] for i in ids]))}
            for n, (P, a, share) in sysP.items():
                acc = lambda sel: 100 * np.mean([int(np.argmax(P[i])) == items[i]["answer"] for i in sel if i in P])  # noqa: E731
                rows.setdefault(n, []).append([acc(list(P)), acc(first), acc(known), acc(habit), effort(P, items, a, 10), 100 * share])
        for n, r in rows.items():
            m = np.mean(r, 0)
            print(f"| blind_v{v} | {n} | {m[0]:.1f} | {m[1]:.1f} | {m[2]:.1f} | {m[3]:.1f} | {m[4]:.3f} | {m[5]:.0f} |")
