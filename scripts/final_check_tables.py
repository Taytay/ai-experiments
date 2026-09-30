"""PLAN step 145: the final synthetic check on blind_v3 (a third blind generator, read once). Written and committed BEFORE any model read
blind_v3 (pre-registered): the recommended system as fixed in PLAN.md's current state on 2026-09-30, against its baselines, by the report
card. No choice is made on these numbers.

Recommended system (REPORTs 130 to 139): decider-4B with the other-users line reads every transaction; the untrained decider-35B-A3B's
distribution is multiplied in for first-time payees only (no earlier filing of the payee); for known payees YNAB's rule's category is
first unless the model's top category is at least 0.2 more probable; the ranked list follows; five suggestions for first-time payees,
the short list (plausible categories free, others only when confident) otherwise.
Baselines: YNAB's rule alone; the no-model list (the rule, then the payee's other past categories); decider-4B (+ line) alone with the
same list rules.
Measures (per REPORT 131 / 132): right one 1st %, in the list %, suggestions shown, work saved (owner's rank scale), by subset: all,
first-time payees, known payees filed differently from the rule, the payee's usual category, new users.
usage: uv run python scripts/final_check_tables.py
"""
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments import scorecard as S
from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
from confirm_card import KMAX, card, history_lists  # noqa: E402
from decision_tables import folds_calibrate, load, product  # noqa: E402
from report_card import plausible_sets  # noqa: E402

R = "results/per_item/real6_"
GATE, LAM = 0.2, 1.0


def system_lists(P, items, plaus, rule, ph, gate=True):
    out = {}
    for i, p in P.items():
        order = [int(c) for c in np.argsort(-p)]; r = rule.get(i, -1)
        if not ph[i]:
            out[i] = order[:KMAX]; continue
        short = S.suggest(p, lam=LAM, miss=10.0, kmax=KMAX, plausible=plaus[i])
        first = order[0] if (r < 0 or (gate and order[0] != r and p[order[0]] - p[r] > GATE)) else r
        out[i] = ([first] + [c for c in short if c != first])[:KMAX]
    return out


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in json.loads((PROCESSED / "blind_v3_others.json").read_text())["items"]}
    rule = json.loads((PROCESSED / "blind_v3_ynabrule.json").read_text()); ph = json.loads((PROCESSED / "blind_v3_payeehist.json").read_text())
    plaus = plausible_sets(items)
    for i, r in rule.items():
        if r >= 0:
            plaus[i].add(r)
    decs = [load(f) for f in sorted(glob.glob(R + "dm_decider_decider_decider-4b_none_h100bf16st800*_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_oth50_aux100_labrand255_laylabelled_shots_ev10soft_lora_blind_v3_others_labrand255_laylabelled_shots.noctx.jsonl"))]
    big = load(R + "dm_decider_decider-35b-a3b_blind_v3.noctx.jsonl")
    ids = sorted(items); gold = {i: items[i]["answer"] for i in ids}
    subsets = {"all": ids, "first-time payee": [i for i in ids if not ph[i]], "filed differently from the rule": [i for i in ids if ph[i] and gold[i] != rule[i]],
               "the payee's usual": [i for i in ids if ph[i] and gold[i] == rule[i]], "new users": [i for i in ids if items[i]["why"] == "new_user"]}
    print(f"**blind_v3 (lists of at most {KMAX}; read once; ideal-reader ceiling {100 * np.mean([x['best'] == x['answer'] for x in items.values()]):.1f}%; decider seeds {len(decs)}): "
          "right one 1st % / in the list % / suggestions shown / work saved %**\n")
    print("| suggestions from | " + " | ".join(f"{k} (n={len(v)})" for k, v in subsets.items()) + " |"); print("|---|" + "---|" * len(subsets))
    srcs = {"YNAB today": [{i: ([rule[i]] if rule[i] >= 0 else []) for i in ids}], "no-model list (YNAB + payee's other past categories)": [history_lists(items, rule)]}
    Pd = [folds_calibrate(d, items) for d in decs]
    Prec = []
    for d, P in zip(decs, Pd):
        Pb = product([d, big], items)
        Prec.append({i: (Pb[i] if not ph[i] else P[i]) for i in P if i in Pb})
    srcs["decider-4B (+ line) alone, same list rules, no gate"] = [system_lists(P, items, plaus, rule, ph, gate=False) for P in Pd]
    srcs["recommended system"] = [system_lists(P, items, plaus, rule, ph) for P in Prec]
    for n, Ls in srcs.items():
        cells = []
        for sel in subsets.values():
            cs = [card(L, items, plaus, sel) for L in Ls]
            cells.append(f"{np.mean([c['first'] for c in cs]):.1f} / {np.mean([100 - c['search'] for c in cs]):.1f} / {np.mean([c['k'] for c in cs]):.1f} / {np.mean([c['saved'] for c in cs]):.0f}")
        print(f"| {n} | " + " | ".join(cells) + " |")
