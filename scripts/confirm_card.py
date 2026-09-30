"""PLAN step 134 (owner, 2026-09-30: "YNAB makes you confirm everything anyway. It currently suggests the last one used in 2 out of 3
transactions (or the last one used if there aren't 3 transactions in that payee's history yet)"). No auto-file: every transaction is
confirmed, so the score is where the right category lands among the suggestions (owner's scale: 1st 0, 2nd 1, ... 5th 4; not suggested 10,
the user searches), plus a clutter cost (LAM) per wrong suggestion that is not plausible (REPORT 130).

  YNAB today      the current rule (build_blind_v1 BLIND_YNAB: the category in 2 of the payee's last 3 filings, else the last one; payee
                  identity exact; none without history): one suggestion
  model           a system's calibrated list: plausible categories free, others only when confident (`scorecard.suggest`), at most 5
  rule, then model  YNAB's suggestion first (when it has one), the model's list after it
Blind_v1 (the others variant's items, same answers). By what decides the item as well.
usage: uv run python scripts/confirm_card.py [LAM]
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
from decision_tables import folds_calibrate, load, product  # noqa: E402
from ensemble_tables import DEC, ENC  # noqa: E402
from report_card import plausible_sets  # noqa: E402

BIG = "results/per_item/real6_dm_decider_decider-35b-a3b_blind_v1.noctx.jsonl"
LAM = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "bulk" else 1.0
MISS = 10.0
import os as _os
KMAX = int(_os.environ.get("KMAX", "3"))  # owner 2026-09-30: lists of at most three suggestions (KMAX=5 for the earlier tables)


def cost(shown, gold, plaus):
    base = shown.index(gold) if gold in shown else MISS
    return base + LAM * sum(c != gold and c not in plaus for c in shown)


def lists(P, items, plaus, rule, mode):
    out = {}
    for i, p in P.items():
        m = S.suggest(p, lam=LAM, miss=MISS, kmax=KMAX, plausible=plaus[i])
        if mode == "model":
            out[i] = m
        else:  # rule first, then the model's list
            r = rule.get(i, -1)
            out[i] = ([r] + [c for c in m if c != r])[:KMAX] if r >= 0 else m
    return out


def history_lists(items, rule):
    """No model: YNAB's suggestion first, then the other categories this user filed the payee under in the prompt's rows (most used
    first), then the user's categories other users file it under (blind_v1_others' line), at most five."""
    from collections import Counter as _C
    from ai_experiments import oneslot
    out = {}
    for i, it in items.items():
        opts = [o.strip() for o in it["options"]]; idx = {o: k for k, o in enumerate(opts)}
        lines = it["prompt"].split("\n"); q = [l for l in lines if l.startswith("Transaction: ")][-1][len("Transaction: "):]
        key = oneslot.payee_key(q); cnt = _C()
        for a, b in zip(lines, lines[1:]):
            if a.startswith("Transaction: ") and b.startswith("Category: ") and a != "Transaction: " + q and key and oneslot.payee_key(a[len("Transaction: "):]) == key:
                c = b[len("Category: "):].strip()
                if c in idx:
                    cnt[idx[c]] += 1
        order = [c for c, _ in cnt.most_common()]
        for l in lines:
            if l.startswith("Other users file this payee as: "):
                for part in l[len("Other users file this payee as: "):].split("), "):
                    nm = part.rsplit(" (", 1)[0].strip()
                    if nm in idx and idx[nm] not in order:
                        order.append(idx[nm])
        r = rule.get(i, -1)
        out[i] = ([r] if r >= 0 else []) + [c for c in order if c != r]
        out[i] = out[i][:KMAX]
    return out


def card(L, items, plaus, ids):
    g = {i: items[i]["answer"] for i in ids}
    c = [cost(L[i], g[i], plaus[i]) for i in ids]
    first = [bool(L[i]) and L[i][0] == g[i] for i in ids]; later = [g[i] in L[i][1:] for i in ids]; miss = [g[i] not in L[i] for i in ids]
    return dict(first=100 * np.mean(first), later=100 * np.mean(later), search=100 * np.mean(miss), k=np.mean([len(L[i]) for i in ids]),
                cost=np.mean(c), saved=100 * (1 - np.mean(c) / MISS))


if __name__ == "__main__" and "bulk" not in sys.argv:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in json.loads((PROCESSED / "blind_v1_others.json").read_text())["items"]}
    rule = json.loads((PROCESSED / "blind_v1_ynabrule.json").read_text())
    plaus = plausible_sets(items)
    for i, r in rule.items():  # YNAB's own suggestion is plausible by definition
        if r >= 0:
            plaus[i].add(r)
    encs = [load(f) for f in sorted(glob.glob(ENC))]; decs = [load(f) for f in sorted(glob.glob(DEC))]; big = load(BIG)
    systems = {"decider-4B": [folds_calibrate(d, items) for d in decs],
               "encoder + decider": [product([e, d], items) for e in encs for d in decs],
               "encoder + decider + 35B": [product([e, d, big], items) for e in encs for d in decs]}
    ids = sorted(items)
    whys = {"all": ids, "payee seen before": [i for i in ids if rule[i] >= 0], "payee new to the user": [i for i in ids if rule[i] < 0],
            "user changed their mind / new category": [i for i in ids if items[i]["why"] in ("changed_mind", "new_category")],
            "one payee, several categories (split / multi-purpose / p2p)": [i for i in ids if items[i]["why"] in ("amount_split", "multi_purpose", "weekday_split", "ambiguous_p2p", "date_night")],
            "new users (0-10 rows)": [i for i in ids if items[i]["why"] == "short_history"]}
    print(f"**Confirm-everything card, blind_v1 (clutter {LAM:g} per implausible wrong suggestion; not suggested = 10)**\n")
    print("| suggestions from | right one 1st % | right one 2nd-5th % | not suggested (search) % | suggestions shown | score (0 best) | work saved vs by hand |")
    print("|---|---|---|---|---|---|---|")
    rule_lists = {i: ([rule[i]] if rule[i] >= 0 else []) for i in ids}
    rows = {"YNAB today (2 of last 3, else last)": [card(rule_lists, items, plaus, ids)],
            "YNAB today + the payee's other past categories (no model)": [card(history_lists(items, rule), items, plaus, ids)]}
    for n, Ps in systems.items():
        rows[n] = [card(lists(P, items, plaus, rule, "model"), items, plaus, ids) for P in Ps]
        rows[f"YNAB's suggestion first, then {n}"] = [card(lists(P, items, plaus, rule, "rule"), items, plaus, ids) for P in Ps]
    for n, cs in rows.items():
        g = lambda k: np.mean([c[k] for c in cs])  # noqa: E731
        print(f"| {n} | {g('first'):.1f} | {g('later'):.1f} | {g('search'):.1f} | {g('k'):.1f} | {g('cost'):.2f} | {g('saved'):.0f}% |")
    print("\n**Right one 1st %, by kind of transaction**\n")
    print("| suggestions from | " + " | ".join(f"{w} (n={len(v)})" for w, v in whys.items()) + " |"); print("|---|" + "---|" * len(whys))
    show = [("YNAB today (2 of last 3, else last)", lambda P: rule_lists)] + [(n, lambda P: lists(P, items, plaus, rule, "model")) for n in systems]
    for n, fn in show:
        Ps = systems.get(n, [None])
        cells = []
        for w, sel in whys.items():
            cells.append(f"{np.mean([card(fn(P), items, plaus, sel)['first'] for P in Ps]):.1f}")
        print(f"| {n} | " + " | ".join(cells) + " |")


def bulk():
    """The same card on blind_bulk_v1: one user's sync, every transaction in order (the natural mix, not blind_v1's hard-case sampling);
    decider-4B (the recipe, labelled rows, two seeds) and the encoder (two seeds); no other-users line or 35B read this set."""
    items = {i["id"]: i for i in json.loads((PROCESSED / "blind_bulk_v1.json").read_text())["items"]}
    rule = json.loads((PROCESSED / "blind_bulk_v1_ynabrule.json").read_text())
    plaus = plausible_sets(items)
    for i, r in rule.items():
        if r >= 0 and i in plaus:
            plaus[i].add(r)
    R = "results/per_item/real6_"
    decs = [load(f) for f in sorted(glob.glob(R + "dm_decider_decider_decider-4b_none_h100bf16st800*_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_aux100_labrand255_laylabelled_shots_ev10soft_lora_blind_bulk_v1_labrand255_laylabelled_shots.noctx.jsonl"))]
    encs = [load(f) for f in sorted(glob.glob(R + "encmask_ettin1b_st3000_h100fresh*_f0_decnone_h100fresh*_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_odb20_short15_kinds20_ev10soft_blind_bulk_v1.noctx.jsonl"))]
    systems = {"decider-4B": [folds_calibrate(d, items) for d in decs], "encoder + decider": [product([e, d], items) for e in encs for d in decs]}
    ids = sorted(items); seen = [i for i in ids if rule[i] >= 0]; new = [i for i in ids if rule[i] < 0]
    print(f"\n**Confirm-everything card, blind_bulk_v1 (one sync per user, {len(ids)} transactions in order; payee seen before {100 * len(seen) / len(ids):.0f}%)**\n")
    print(f"| suggestions from | right one 1st % | 2nd-{KMAX}th % | search % | shown | score | work saved | 1st %, payee seen | 1st %, payee new |")
    print("|---|---|---|---|---|---|---|---|---|")
    rl = {i: ([rule[i]] if rule[i] >= 0 else []) for i in ids}
    rows = {"YNAB today": ([rl], None), "YNAB today + the payee's other past categories (no model)": ([history_lists(items, rule)], None)}
    for n, Ps in systems.items():
        rows[n] = ([lists(P, items, plaus, rule, "model") for P in Ps], None)
        rows[f"YNAB's suggestion first, then {n}"] = ([lists(P, items, plaus, rule, "rule") for P in Ps], None)
    for n, (Ls, _) in rows.items():
        cs = [card(L, items, plaus, ids) for L in Ls]; g = lambda k: np.mean([c[k] for c in cs])  # noqa: E731
        s1 = np.mean([card(L, items, plaus, seen)["first"] for L in Ls]); s2 = np.mean([card(L, items, plaus, new)["first"] for L in Ls]) if new else float("nan")
        print(f"| {n} | {g('first'):.1f} | {g('later'):.1f} | {g('search'):.1f} | {g('k'):.1f} | {g('cost'):.2f} | {g('saved'):.0f}% | {s1:.1f} | {s2:.1f} |")


if __name__ == "__main__" and "bulk" in sys.argv:
    bulk()
