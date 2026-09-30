"""PLAN step 141 (after REPORT 137: the encoder does not transfer to blind_v2): the system without the encoder. Effort under the per-item
rule (W = 5 / 10 / 20) on blind_v1 and blind_v2: decider (+ other-users line); decider -> untrained 35B for what decider does not
auto-file; decider x 35B for all; and REPORT 126's encoder x decider -> 35B for comparison.
usage: uv run python scripts/noenc_tables.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
from blind2_tables import effort, readers, rule_of  # noqa: E402
from decision_tables import fmt, folds_calibrate, product  # noqa: E402

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("| set | system | W = 5 | W = 10 | W = 20 |"); print("|---|---|---|---|---|")
    for v in (1, 2):
        items = {i["id"]: i for i in json.loads((PROCESSED / f"blind_v{v}_others.json").read_text())["items"]}
        rd = readers(v); dec, enc, big = rd["decider-4B + other-users line"], rd["encoder (Ettin-1B)"], rd["decider-35B untrained"][0]
        res = {k: {w: [] for w in (5, 10, 20)} for k in ("decider", "decider -> 35B", "decider x 35B (all)", "encoder x decider -> 35B")}
        for d in dec:
            Pd = folds_calibrate(d, items); Pb = product([d, big], items)
            for w in (5, 10, 20):
                a1, ab = rule_of(Pd, w), rule_of(Pb, w); ids = [i for i in Pd if i in Pb]
                res["decider"][w].append(effort(Pd, items, a1, w))
                res["decider -> 35B"][w].append(effort({i: (Pd[i] if a1[i] else Pb[i]) for i in ids}, items, {i: a1[i] or ab[i] for i in ids}, w))
                res["decider x 35B (all)"][w].append(effort(Pb, items, ab, w))
            for e in enc:
                P2, P3 = product([e, d], items), product([e, d, big], items)
                for w in (5, 10, 20):
                    a2, a3 = rule_of(P2, w), rule_of(P3, w); ids = [i for i in P2 if i in P3]
                    res["encoder x decider -> 35B"][w].append(effort({i: (P2[i] if a2[i] else P3[i]) for i in ids}, items, {i: a2[i] or a3[i] for i in ids}, w))
        for k, r in res.items():
            print(f"| blind_v{v} | {k} | " + " | ".join(fmt(r[w], 3) for w in (5, 10, 20)) + " |")
