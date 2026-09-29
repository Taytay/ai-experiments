"""Tables for PLAN step 120 (EVAL-12): a large model for the transactions the two small models do not settle. On blind_v1: the encoder
(Ettin-1B, the encoder recipe, three seeds), decider-4B (the recipe with labelled rows, two seeds) and decider-35B-A3B untrained (its own
layout, letters), each calibrated with the leave-fold-out temperature and 98% threshold (`effort_tables.calibrated`).

  ES.1  top-1 of each reader overall, by what decides the item, and where the small models agree / disagree
  ES.2  system policies (six small-model seed pairs): REPORT 115's best ("agree and either confident"), then the 35B on what is left:
        auto-file when it is confident and agrees with one small model / when it is confident alone; suggestions from the 35B or from the
        small models' product; auto-filed %, precision, effort, and the share of transactions sent to the 35B
usage: uv run python scripts/escalation_tables.py
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
from agreement_tables import DEC, ENC, fmt, load  # noqa: E402
from effort_tables import calibrated  # noqa: E402

BIG = "results/per_item/real6_dm_decider_decider-35b-a3b_blind_v1.noctx.jsonl"
SMALL0 = "results/per_item/real6_dm_decider_decider-4b_blind_v1.noctx.jsonl"


def row(items, ids, auto, pick, sugg):
    a = [auto[i] for i in ids]; ok = [pick[i] == items[i]["answer"] for i in ids]
    t3 = [items[i]["answer"] in np.argsort(-sugg[i])[:3] for i in ids]
    return 100 * np.mean(a), (100 * np.mean([o for x, o in zip(a, ok) if x]) if any(a) else float("nan")), S.effort(a, ok, t3)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in json.loads((PROCESSED / "blind_v1.json").read_text())["items"]}
    big = calibrated(load(BIG), items); zs4 = calibrated(load(SMALL0), items)
    encs = [calibrated(load(f), items) for f in sorted(glob.glob(ENC))]; decs = [calibrated(load(f), items) for f in sorted(glob.glob(DEC))]
    top = lambda c, i: int(np.argmax(c[i]["p"]))  # noqa: E731
    whys = sorted({x["why"] for x in items.values()}, key=lambda w: -sum(x["why"] == w for x in items.values()))
    print("**Table ES.1: top-1 on blind_v1 (small models: first seed; agree / disagree: the first encoder and decider seeds)**\n")
    readers = [("encoder (Ettin-1B, recipe)", encs[0]), ("decider-4B, recipe", decs[0]), ("decider-4B untrained", zs4), ("decider-35B-A3B untrained", big)]
    print("| group | n | " + " | ".join(r for r, _ in readers) + " |"); print("|---|---|" + "---|" * len(readers))
    ids_all = sorted(set(big) & set(encs[0]) & set(decs[0]))
    same = {i: top(encs[0], i) == top(decs[0], i) for i in ids_all}
    groups = [("all", ids_all), ("small models agree", [i for i in ids_all if same[i]]), ("small models disagree", [i for i in ids_all if not same[i]])]
    groups += [(w, [i for i in ids_all if items[i]["why"] == w]) for w in whys]
    for name, ids in groups:
        if len(ids) < 15:
            continue
        print(f"| {name} | {len(ids)} | " + " | ".join(f"{100 * np.mean([top(c, i) == items[i]['answer'] for i in ids]):.1f}" for _, c in readers) + " |")
    print("\n**Table ES.2: system policies on blind_v1 (mean [range] over six small-model seed pairs)**\n")
    pol = {}
    for ec in encs:
        for dc in decs:
            ids = sorted(set(ec) & set(dc) & set(big))
            ae = {i: top(ec, i) for i in ids}; ad = {i: top(dc, i) for i in ids}; ab = {i: top(big, i) for i in ids}
            prod = {}
            for i in ids:
                q = np.clip(ec[i]["p"], 1e-12, 1) * np.clip(dc[i]["p"], 1e-12, 1); prod[i] = q / q.sum()
            p3 = {}
            for i in ids:
                q = np.clip(ec[i]["p"], 1e-12, 1) * np.clip(dc[i]["p"], 1e-12, 1) * np.clip(big[i]["p"], 1e-12, 1); p3[i] = q / q.sum()
            base = {i: ae[i] == ad[i] and (ec[i]["a98"] or dc[i]["a98"]) for i in ids}
            esc = {i: not base[i] for i in ids}
            b_agree = {i: esc[i] and big[i]["a98"] and ab[i] in (ae[i], ad[i]) for i in ids}
            b_alone = {i: esc[i] and big[i]["a98"] for i in ids}
            rows = {
                "agree and either confident (REPORT 115); suggestions from the small models": ({i: base[i] for i in ids}, ad, prod, 0),
                "the same; suggestions for the rest from the 35B": ({i: base[i] for i in ids}, ad, {i: prod[i] if base[i] else big[i]["p"] for i in ids}, 1),
                "+ the 35B confident and agreeing with one small model": ({i: base[i] or b_agree[i] for i in ids}, {i: ad[i] if base[i] else ab[i] for i in ids},
                                                                          {i: prod[i] if base[i] else big[i]["p"] for i in ids}, 1),
                "+ the 35B confident alone": ({i: base[i] or b_alone[i] for i in ids}, {i: ad[i] if base[i] else ab[i] for i in ids},
                                              {i: prod[i] if base[i] else big[i]["p"] for i in ids}, 1),
                "+ the 35B, suggestions from the three-model product": ({i: base[i] or b_agree[i] for i in ids}, {i: ad[i] if base[i] else ab[i] for i in ids},
                                                                        {i: prod[i] if base[i] else p3[i] for i in ids}, 1),
                "the 35B alone at its 98% (no small models)": ({i: big[i]["a98"] for i in ids}, ab, {i: big[i]["p"] for i in ids}, 2),
            }
            for name, (a, p, sg, uses) in rows.items():
                cov, prec, eff = row(items, ids, a, p, sg)
                sent = 100 * np.mean([esc[i] for i in ids]) if uses == 1 else (100.0 if uses == 2 else 0.0)
                pol.setdefault(name, []).append((cov, prec, eff, sent))
    print("| policy | auto-filed % | precision % | effort | sent to the 35B % |"); print("|---|---|---|---|---|")
    for name, v in pol.items():
        v = np.array(v); print(f"| {name} | {fmt(v[:, 0])} | {fmt(v[:, 1])} | {fmt(v[:, 2], 3)} | {fmt(v[:, 3])} |")
