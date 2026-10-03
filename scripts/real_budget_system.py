"""PLAN step 165 (owner, 2026-10-02: "Do #1 - confidence gate"): REPORT 141's recommended system on the owner's real budget
(real_budget_eval.py's items and scores, read from OUT outside the repo; only aggregates are printed).

The system as fixed for blind_v3 (final_check_tables.system_lists): decider-4B (the recipe) reads every transaction; for first-time
payees the untrained decider-35B-A3B's distribution is multiplied in; for known payees YNAB's rule's category is first unless the model's
top category is at least GATE (0.2) more probable; the rest of a list follows the model (scorecard.suggest, at most KMAX = 3, plausible
categories free). Two things the synthetic sets had and one real budget does not:
  calibration  the synthetic tables fit temperatures on other users' folds; with one user, each half of the transactions (by parity of
               their date-order index) is tempered with the temperature fitted on the other half
  plausible    the synthetic generator listed every item's plausible categories (for the clutter charge and the list's free entries);
               here the stand-in is the categories the owner had filed the payee under before the transaction's date, plus the rule's
Measures (confirm_card.card): right one first, in the list, suggestions shown, work saved (owner's rank scale, not suggested = 10); and
the auto-file curve of the system's first suggestion against its calibrated probability.
env: BUDGET, OUT (as real_budget_eval.py), GATE (0.2), KMAX (3), DEC (the main reader's scores), BIG (the first-time-payee reader).
usage: BUDGET=<id> uv run python scripts/real_budget_system.py
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ai_experiments import calibration as C
from ai_experiments import scorecard as S

sys.path.insert(0, str(Path(__file__).parent))
from confirm_card import card  # noqa: E402

BUDGET = os.environ["BUDGET"]
OUT = Path(os.environ.get("OUT", Path.home() / ".local" / "share" / "ynab-real-eval" / BUDGET))
GATE = float(os.environ.get("GATE", "0.2"))
KMAX = int(os.environ.get("KMAX", "3"))
LAM = 1.0
DEC = os.environ.get("DEC", "recipe_split")  # the main reader's scores (scores_<DEC>.jsonl): recipe_split | recipe_split_sim | big-recipe_split
BIG = os.environ.get("BIG", "big_split")  # the first-time-payee reader multiplied in ("" = none)


def halves(lps, gold, order):
    """{id: calibrated probabilities}: each parity half tempered with the temperature fitted on the other half."""
    out = {}
    for h in (0, 1):
        fit = [i for k, i in enumerate(order) if k % 2 != h]
        t = C.fit_temperature([np.asarray(lps[i]) for i in fit], [gold[i] for i in fit])
        out.update({i: C.softmax(np.asarray(lps[i]), t) for k, i in enumerate(order) if k % 2 == h})
    return out


def payee_of(it):
    return it["prompt"].rsplit("Transaction: ", 1)[1].split(" | ")[1]


if __name__ == "__main__":
    items = {it["id"]: it for it in json.loads((OUT / "items.json").read_text())["items"] if it["answer"] >= 0}
    dec = {r["id"]: r["lp"] for r in map(json.loads, open(OUT / f"scores_{DEC}.jsonl"))}
    bigf = OUT / f"scores_{BIG}.jsonl"
    big = {r["id"]: r["lp"] for r in map(json.loads, open(bigf))} if BIG and bigf.exists() else {}
    print(f"main reader: {DEC}; first-time payees x {BIG or 'nothing'}\n")
    order = sorted(items, key=lambda i: (items[i]["date"], i))
    gold = {i: items[i]["answer"] for i in order}
    opt = {i: {c: k for k, c in enumerate(items[i]["options"])} for i in order}
    rule = {i: opt[i].get(items[i]["rule"], -1) if items[i]["rule"] else -1 for i in order}

    # the payee's earlier categories (dated strictly before), most recent first: the plausible stand-in and the no-model list
    past, hist, day, pending = {}, defaultdict(list), None, []
    for i in order:
        if items[i]["date"] != day:
            for j in pending:
                hist[payee_of(items[j])].append(items[j]["gold"])
            day, pending = items[i]["date"], []
        seen = list(dict.fromkeys(reversed(hist[payee_of(items[i])])))
        past[i] = [opt[i][c] for c in seen if c in opt[i]]
        pending.append(i)
    known = {i: items[i]["payee_seen"] for i in order}
    plaus = {i: set(past[i]) | ({rule[i]} if rule[i] >= 0 else set()) for i in order}

    Pd = halves(dec, gold, order)
    new = [i for i in order if not known[i]]
    have_big = bool(big) and all(i in big for i in new)
    if have_big:
        Pb = halves({i: big[i] for i in new}, gold, new)
        prod = {i: np.log(np.clip(Pd[i], 1e-12, 1)) + np.log(np.clip(Pb[i], 1e-12, 1)) for i in new}
        Pp = halves(prod, gold, new)
    else:
        print(f"(35B scores missing or incomplete: {len(big)} of {len(new)} first-time payees; the system runs without it)")

    def system(P, gate=True):
        out = {}
        for i in order:
            p = P[i]; o = [int(c) for c in np.argsort(-p)]; r = rule[i]
            if not known[i]:
                out[i] = o[:KMAX]; continue
            short = S.suggest(p, lam=LAM, miss=10.0, kmax=KMAX, plausible=plaus[i])
            first = o[0] if (r < 0 or (gate and o[0] != r and p[o[0]] - p[r] > GATE)) else r
            out[i] = ([first] + [c for c in short if c != first])[:KMAX]
        return out

    lists = {
        "YNAB today": {i: ([rule[i]] if rule[i] >= 0 else []) for i in order},
        "no-model list (rule, then the payee's other past categories)": {i: ([rule[i]] if rule[i] >= 0 else []) + [c for c in past[i] if c != rule[i]][:KMAX - 1] if rule[i] >= 0 else past[i][:KMAX] for i in order},
        f"{DEC} alone, its own order": {i: S.suggest(Pd[i], lam=LAM, miss=10.0, kmax=KMAX, plausible=plaus[i]) if known[i] else [int(c) for c in np.argsort(-Pd[i])][:KMAX] for i in order},
        f"rule always first, then {DEC}'s list (final_check's no-gate row)": system(Pd, gate=False),
        f"{DEC} + the gate (rule first unless the model is 0.2 surer)": system(Pd),
    }
    if have_big:
        lists[f"recommended system (x {BIG} for first-time payees)"] = system({**Pd, **Pp})
    subsets = {"all": order, "first-time payee": new, "known payee, filed differently from the rule": [i for i in order if known[i] and gold[i] != rule[i]],
               "known payee, the rule's category": [i for i in order if known[i] and gold[i] == rule[i]]}
    citems = {i: {"answer": gold[i]} for i in order}
    print(f"**right one 1st % / in the list (at most {KMAX}) % / suggestions shown / work saved %**\n")
    print("| suggestions from | " + " | ".join(f"{k} (n={len(v)})" for k, v in subsets.items()) + " |"); print("|---|" + "---|" * len(subsets))
    for n, L in lists.items():
        cells = []
        for sel in subsets.values():
            c = card(L, citems, plaus, sel)
            cells.append(f"{c['first']:.1f} / {100 - c['search']:.1f} / {c['k']:.1f} / {c['saved']:.0f}")
        print(f"| {n} | " + " | ".join(cells) + " |")

    # auto-file: the system's first suggestion, its calibrated probability (the rule's category's probability when the rule leads)
    best = list(lists)[-1]; L = lists[best]; P = {**Pd, **(Pp if have_big else {})}
    conf = np.array([P[i][L[i][0]] for i in order]); ok = np.array([L[i][0] == gold[i] for i in order])
    print(f"\n**Auto-filing the first suggestion of: {best} (calibrated probability)**\n")
    print("| filed automatically when at least | share of transactions | right among them | share filed right overall |"); print("|---|---|---|---|")
    for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        s = conf >= t
        print(f"| {t} | {100 * s.mean():.1f}% | {100 * ok[s].mean():.1f}% | {100 * (s & ok).mean():.1f}% |")
    for prec in (0.9, 0.95, 0.98):
        cov = C.coverage_at_precision(conf, ok, prec)
        print(f"coverage at {int(prec * 100)}% precision: {100 * cov:.1f}%" if not isinstance(cov, tuple) else f"coverage at {int(prec * 100)}% precision: {cov}")
    print(f"\ncalibration (ECE, first suggestion): {C.ece(conf, ok):.3f}")
