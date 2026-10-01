"""Tables for PLAN step 140 (EVAL-15): the recent findings re-read on blind_v2 (a second blind test set, written by an agent from the
product brief only), beside blind_v1. For each set:
  B2.1  top-1 of each reader, and the set's ideal-reader ceiling
  B2.2  effort under the per-item rule (REPORT 118) at W = 5 / 10 / 20: decider (recipe; + other-users line), encoder x decider,
        encoder x decider -> untrained 35B for the rest, all three
  B2.3  the confirm-everything card (REPORT 131): YNAB's rule, the no-model list, the models, rule-first
  B2.4  the novelty card (REPORT 132): first-time payees, filed differently from the rule, a category never used for the payee
usage: uv run python scripts/blind2_tables.py
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
from confirm_card import card, history_lists, lists  # noqa: E402
from decision_tables import fmt, folds_calibrate, load, product  # noqa: E402
from report_card import plausible_sets  # noqa: E402

R = "results/per_item/real6_"
MID = "_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10"


def readers(v):
    """{name: [per-seed records]} for blind_v<v>."""
    b, bo = f"blind_v{v}", f"blind_v{v}_others"
    g = lambda pat: [load(f) for f in sorted(glob.glob(R + pat))]  # noqa: E731
    return {
        "decider-4B recipe": g(f"dm_decider_decider_decider-4b_none_h100bf16st800*{MID}_aux100_labrand255_laylabelled_shots_ev10soft_lora_{b}_labrand255_laylabelled_shots.noctx.jsonl"),
        "decider-4B + other-users line": g(f"dm_decider_decider_decider-4b_none_h100bf16st800*{MID}_oth50_aux100_labrand255_laylabelled_shots_ev10soft_lora_{bo}_labrand255_laylabelled_shots.noctx.jsonl"),
        "encoder (Ettin-1B)": g(f"encmask_ettin1b_st3000_h100fresh*_f0_decnone_h100fresh*{MID}_odb20_short15_kinds20_ev10soft_{b}.noctx.jsonl"),
        "decider-35B untrained": g(f"dm_decider_decider-35b-a3b_{b}.noctx.jsonl"),
        "decider-4B untrained": g(f"dm_decider_decider-4b_{b}.noctx.jsonl"),
    }


def rule_of(P, W):
    return {i: bool(S.decide(p.max(), np.sort(p)[-3:].sum(), W)) for i, p in P.items()}


def effort(P, items, a, W):
    ids = sorted(P); ok = [int(np.argmax(P[i])) == items[i]["answer"] for i in ids]
    t3 = [items[i]["answer"] in np.argsort(-P[i])[:3] for i in ids]
    return S.effort([a[i] for i in ids], ok, t3, wrong=W)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    for v in (1, 2):
        items = {i["id"]: i for i in json.loads((PROCESSED / f"blind_v{v}_others.json").read_text())["items"]}
        rule = json.loads((PROCESSED / f"blind_v{v}_ynabrule.json").read_text()); ph = json.loads((PROCESSED / f"blind_v{v}_payeehist.json").read_text())
        rd = readers(v)
        print(f"\n### blind_v{v} ({len(items)} items; ideal-reader ceiling {100 * np.mean([x['best'] == x['answer'] for x in items.values()]):.1f}%)\n")
        print("**B2.1 top-1 (mean [range] over seeds)**\n"); print("| reader | seeds | top-1 |"); print("|---|---|---|")
        for n, recs in rd.items():
            if recs:
                print(f"| {n} | {len(recs)} | {fmt([100 * np.mean([int(np.argmax(x)) == items[i]['answer'] for i, x in r.items() if i in items]) for r in recs])} |")
        dec, enc, big = rd["decider-4B + other-users line"], rd["encoder (Ettin-1B)"], rd["decider-35B untrained"]
        if not (dec and enc and big):
            print("(missing readers for the systems)"); continue
        pairs = [(e, d) for e in enc for d in dec]
        print("\n**B2.2 effort under the per-item rule (mean [range] over seed pairs)**\n"); print("| system | W = 5 | W = 10 | W = 20 |"); print("|---|---|---|---|")
        res = {k: {w: [] for w in (5, 10, 20)} for k in ("decider (+ line)", "encoder x decider", "pair -> 35B", "all three")}
        for d in dec:
            Pd = folds_calibrate(d, items)
            for w in (5, 10, 20):
                res["decider (+ line)"][w].append(effort(Pd, items, rule_of(Pd, w), w))
        for e, d in pairs:
            P2 = product([e, d], items); P3 = product([e, d, big[0]], items)
            for w in (5, 10, 20):
                a2, a3 = rule_of(P2, w), rule_of(P3, w); ids = [i for i in P2 if i in P3]
                res["encoder x decider"][w].append(effort(P2, items, a2, w))
                res["pair -> 35B"][w].append(effort({i: (P2[i] if a2[i] else P3[i]) for i in ids}, items, {i: a2[i] or a3[i] for i in ids}, w))
                res["all three"][w].append(effort(P3, items, a3, w))
        for k, r in res.items():
            print(f"| {k} | " + " | ".join(fmt(r[w], 3) for w in (5, 10, 20)) + " |")
        plaus = plausible_sets(items)
        for i, r in rule.items():
            if r >= 0 and i in plaus:
                plaus[i].add(r)
        ids = sorted(items); gold = {i: items[i]["answer"] for i in ids}
        subsets = {"all": ids, "first-time payee": [i for i in ids if not ph[i]], "differently from the rule": [i for i in ids if ph[i] and gold[i] != rule[i]],
                   "never used for the payee": [i for i in ids if ph[i] and gold[i] not in ph[i]], "the payee's usual": [i for i in ids if ph[i] and gold[i] == rule[i]]}
        systems = {"decider (+ line)": [folds_calibrate(d, items) for d in dec], "encoder x decider": [product([e, d], items) for e, d in pairs],
                   "encoder x decider x 35B": [product([e, d, big[0]], items) for e, d in pairs]}
        srcs = {"YNAB today": [{i: ([rule[i]] if rule[i] >= 0 else []) for i in ids}], "rule + payee's past + other users' categories (no model; NOT in YNAB today)": [history_lists(items, rule)]}
        for n, Ps in systems.items():
            srcs[n] = [lists(P, items, plaus, rule, "model") for P in Ps]
            srcs[f"YNAB first, then {n}"] = [lists(P, items, plaus, rule, "rule") for P in Ps]
        print("\n**B2.3 / B2.4 confirm-everything card by subset: right one 1st % / in the list % / work saved % (mean over seeds / pairs)**\n")
        print("| suggestions from | " + " | ".join(f"{k} (n={len(v_)})" for k, v_ in subsets.items()) + " |"); print("|---|" + "---|" * len(subsets))
        for n, Ls in srcs.items():
            cells = []
            for sel in subsets.values():
                cs = [card(L, items, plaus, sel) for L in Ls]
                cells.append(f"{np.mean([c['first'] for c in cs]):.1f} / {np.mean([100 - c['search'] for c in cs]):.1f} / {np.mean([c['saved'] for c in cs]):.0f}")
            print(f"| {n} | " + " | ".join(cells) + " |")
