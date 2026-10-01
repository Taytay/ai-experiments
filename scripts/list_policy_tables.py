"""PLAN step 139 (after REPORT 133: first-time payees have the right category in the system's top five 88% of the time, but the
suggestion rule showed short lists). Suggestion-list policies for a confirm-everything product (REPORT 131), on blind_v1 and the
ordinary stream: how often the right category is 1st / anywhere in the list, how many suggestions are shown, how many shown ones are wrong.
  current      REPORT 131: YNAB's suggestion first, then plausible categories free and others only if confident (LAM 1)
  top 5        always the model's top five (YNAB's first when it has one)
  p >= x       categories with calibrated probability at least x (at most five, at least one; YNAB's first)
  adaptive     current for payees seen before; top five for first-time payees
usage: uv run python scripts/list_policy_tables.py
"""
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
from confirm_card import lists  # noqa: E402
from decision_tables import folds_calibrate, load, product  # noqa: E402
from ensemble_tables import DEC, ENC  # noqa: E402
from report_card import plausible_sets  # noqa: E402

R = "results/per_item/real6_"


def policy(P, items, plaus, rule, ph, name):
    out = {}
    cur = lists(P, items, plaus, rule, "rule")
    for i, p in P.items():
        order = [int(c) for c in np.argsort(-p)]; r = rule.get(i, -1)
        def rule_first(L):
            return ([r] + [c for c in L if c != r])[:5] if r >= 0 else L[:5]
        if name == "current":
            out[i] = cur[i]
        elif name == "top 5":
            out[i] = rule_first(order[:5])
        elif name.startswith("p >= "):
            x = float(name.split()[-1]); L = [c for c in order[:5] if p[c] >= x] or order[:1]
            out[i] = rule_first(L)
        elif name == "adaptive":
            out[i] = cur[i] if ph[i] else rule_first(order[:5])
    return out


def stats(L, items, ids):
    g = {i: items[i]["answer"] for i in ids}
    return (100 * np.mean([L[i][0] == g[i] for i in ids]), 100 * np.mean([g[i] in L[i] for i in ids]),
            np.mean([len(L[i]) for i in ids]), np.mean([sum(c != g[i] for c in L[i]) for i in ids]))


def run(title, items, rule, ph, P):
    plaus = plausible_sets(items)
    for i, r in rule.items():
        if r >= 0 and i in plaus:
            plaus[i].add(r)
    ids = sorted(P); first = [i for i in ids if not ph[i]]; seen = [i for i in ids if ph[i]]
    print(f"\n**{title}: right one 1st % / in the list % / suggestions shown / wrong ones shown (all; first-time payees; payees seen before)**\n")
    print("| policy | all | first-time payees | payees seen before |"); print("|---|---|---|---|")
    for name in ("current", "adaptive", "p >= 0.05", "p >= 0.1", "top 5"):
        L = policy(P, items, plaus, rule, ph, name)
        cells = [" / ".join(f"{v:.1f}" if k < 2 else f"{v:.2f}" for k, v in enumerate(stats(L, items, sel))) for sel in (ids, first, seen)]
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in json.loads((PROCESSED / "blind_v1_others.json").read_text())["items"]}
    big = load(R + "dm_decider_decider-35b-a3b_blind_v1.noctx.jsonl")
    P = product([load(sorted(glob.glob(ENC))[0]), load(sorted(glob.glob(DEC))[0]), big], items)
    run("blind_v1, encoder x decider x 35B", items, json.loads((PROCESSED / "blind_v1_ynabrule.json").read_text()),
        json.loads((PROCESSED / "blind_v1_payeehist.json").read_text()), P)
    items = {i["id"]: i for i in json.loads((PROCESSED / "blind_bulk_v1.json").read_text())["items"]}
    d = sorted(glob.glob(R + "dm_decider_decider_decider-4b_none_h100bf16st800*_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_aux100_labrand255_laylabelled_shots_ev10soft_lora_blind_bulk_v1_labrand255_laylabelled_shots.noctx.jsonl"))
    e = sorted(glob.glob(R + "encmask_ettin1b_st3000_h100fresh*_f0_decnone_h100fresh*_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_odb20_short15_kinds20_ev10soft_blind_bulk_v1.noctx.jsonl"))
    P = product([load(e[0]), load(d[0])], items)
    run("the ordinary stream (blind_bulk_v1), encoder x decider", items, json.loads((PROCESSED / "blind_bulk_v1_ynabrule.json").read_text()),
        json.loads((PROCESSED / "blind_bulk_v1_payeehist.json").read_text()), P)
