"""PLAN step 135 (owner, 2026-09-30): two scores of their own.
  1. A user categorising a payee for the first time (no earlier filing of this payee; payee identity exact, from the generator).
  2. A user categorising a payee differently than before: (a) not what YNAB's rule suggests (the category in 2 of the payee's last 3,
     else the last), (b) a category never used for this payee before. Against the control: the payee's usual category (= the rule).
Scored as the confirm-everything card (REPORT 131: right one 1st 0, 2nd 1, ... 5th 4, not suggested 10; clutter 1 per implausible wrong
suggestion). Anticipation: the model's top probability on each subset against the control, and how well low confidence picks out the
"differently" cases from the usual ones (AUROC of 1 - p1).
usage: uv run python scripts/novelty_card.py [bulk]
"""
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
from confirm_card import card, history_lists, lists  # noqa: E402
from decision_tables import folds_calibrate, load, product  # noqa: E402
from ensemble_tables import DEC, ENC  # noqa: E402
from report_card import plausible_sets  # noqa: E402

R = "results/per_item/real6_"
BULK = "bulk" in sys.argv

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if BULK:
        items = {i["id"]: i for i in json.loads((PROCESSED / "blind_bulk_v1.json").read_text())["items"]}
        rule = json.loads((PROCESSED / "blind_bulk_v1_ynabrule.json").read_text()); ph = json.loads((PROCESSED / "blind_bulk_v1_payeehist.json").read_text())
        decs = [load(f) for f in sorted(glob.glob(R + "dm_decider_decider_decider-4b_none_h100bf16st800*_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_aux100_labrand255_laylabelled_shots_ev10soft_lora_blind_bulk_v1_labrand255_laylabelled_shots.noctx.jsonl"))]
        encs = [load(f) for f in sorted(glob.glob(R + "encmask_ettin1b_st3000_h100fresh*_f0_decnone_h100fresh*_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_odb20_short15_kinds20_ev10soft_blind_bulk_v1.noctx.jsonl"))]
        systems = {"decider-4B": [folds_calibrate(d, items) for d in decs], "encoder + decider": [product([e, d], items) for e in encs for d in decs]}
        title = "blind_bulk_v1 (one user's sync, every transaction in order)"
    else:
        items = {i["id"]: i for i in json.loads((PROCESSED / "blind_v1_others.json").read_text())["items"]}
        rule = json.loads((PROCESSED / "blind_v1_ynabrule.json").read_text()); ph = json.loads((PROCESSED / "blind_v1_payeehist.json").read_text())
        encs = [load(f) for f in sorted(glob.glob(ENC))]; decs = [load(f) for f in sorted(glob.glob(DEC))]
        big = load(R + "dm_decider_decider-35b-a3b_blind_v1.noctx.jsonl")
        systems = {"decider-4B (+ other users' line)": [folds_calibrate(d, items) for d in decs],
                   "encoder + decider": [product([e, d], items) for e in encs for d in decs],
                   "encoder + decider + 35B": [product([e, d, big], items) for e in encs for d in decs]}
        title = "blind_v1 (hard cases over-sampled)"
    plaus = plausible_sets(items)
    for i, r in rule.items():
        if r >= 0 and i in plaus:
            plaus[i].add(r)
    ids = sorted(items); g = {i: items[i]["answer"] for i in ids}
    subsets = {"1. first time for this payee": [i for i in ids if not ph[i]],
               "2a. differently from YNAB's rule": [i for i in ids if ph[i] and g[i] != rule[i]],
               "2b. a category never used for this payee": [i for i in ids if ph[i] and g[i] not in ph[i]],
               "control: the payee's usual category (= the rule)": [i for i in ids if ph[i] and g[i] == rule[i]]}
    print(f"**Novelty card, {title}: right one 1st % / suggested at all (1st-5th) % / score (0 best, 10 = search) per subset; mean over seeds / pairs**\n")
    print("| suggestions from | " + " | ".join(f"{k} (n={len(v)}, {100 * len(v) / len(ids):.1f}%)" for k, v in subsets.items()) + " |")
    print("|---|" + "---|" * len(subsets))
    rl = {i: ([rule[i]] if rule[i] >= 0 else []) for i in ids}
    srcs = {"YNAB today": [rl], "rule + payee's past + other users' categories (no model; NOT in YNAB today)": [history_lists(items, rule)]}
    for n, Ps in systems.items():
        srcs[n] = [lists(P, items, plaus, rule, "model") for P in Ps]
        srcs[f"YNAB first, then {n}"] = [lists(P, items, plaus, rule, "rule") for P in Ps]
    for n, Ls in srcs.items():
        cells = []
        for k, sel in subsets.items():
            cs = [card(L, items, plaus, sel) for L in Ls]
            cells.append(f"{np.mean([c['first'] for c in cs]):.1f} / {np.mean([100 - c['search'] for c in cs]):.1f} / {np.mean([c['cost'] for c in cs]):.2f}")
        print(f"| {n} | " + " | ".join(cells) + " |")
    print("\n**Anticipation: the model's top probability (calibrated) by subset, and how well low confidence separates 'differently' (2a) from the usual (control): AUROC of 1 - p1**\n")
    print("| model | " + " | ".join(f"mean p1, {k.split('.')[0] if '.' in k[:3] else k.split(':')[0]}" for k in subsets) + " | AUROC 2a vs control | AUROC 2b vs control |")
    print("|---|" + "---|" * len(subsets) + "---|---|")
    for n, Ps in systems.items():
        mp = {k: np.mean([np.mean([P[i].max() for i in sel if i in P]) for P in Ps]) for k, sel in subsets.items()}
        aucs = []
        for pos in ("2a. differently from YNAB's rule", "2b. a category never used for this payee"):
            neg = subsets["control: the payee's usual category (= the rule)"]
            aucs.append(np.mean([roc_auc_score([1] * len(subsets[pos]) + [0] * len(neg), [1 - P[i].max() for i in subsets[pos] + neg]) for P in Ps]))
        print(f"| {n} | " + " | ".join(f"{100 * v:.1f}%" for v in mp.values()) + f" | {aucs[0]:.3f} | {aucs[1]:.3f} |")
