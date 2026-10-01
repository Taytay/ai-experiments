"""Tables for PLAN step 129: the system with everything that helped, scored by expected effort under the per-item rule (REPORT 118) at
W = 5 / 10 / 20 / 50 (W: what a wrong auto-file costs). On blind_v1 (decider with the other-users line reads blind_v1_others):

  decider              decider-4B alone: the recipe (labelled rows), or trained with the other-users line (OTHERS=0.5, line shown)
  pair                 the encoder (Ettin-1B) x decider: product of the calibrated distributions, recalibrated, per-item rule
  pair -> 35B          the pair's rule first; what it does not auto-file goes to the untrained decider-35B-A3B, whose distribution is
                       multiplied in (three-model product, recalibrated) and the rule applied again; suggestions from that product
  three for all        every transaction through all three (the three-model product)

Mean [range] over every encoder x decider seed pair; the share sent to the 35B beside it.
usage: uv run python scripts/system_v2_tables.py
"""
import glob
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments import scorecard as S
from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
from decision_tables import fmt, folds_calibrate, load, product  # noqa: E402

REC = "ren50_dbep50_mislead_v1_alt10s_lk10_ov10"
DEC = "results/per_item/real6_dm_decider_decider_decider-4b_none_h100bf16st800*_emp20_f0_" + REC + "{oth}_aux100_labrand255_laylabelled_shots_ev10soft_lora_{set}_labrand255_laylabelled_shots.noctx.jsonl"
ENC = f"results/per_item/real6_encmask_ettin1b_st3000_h100fresh*_f0_decnone_h100fresh*_emp20_f0_{REC}_odb20_short15_kinds20_ev10soft_blind_v1.noctx.jsonl"
BIG = os.environ.get("BIG", "results/per_item/real6_dm_decider_decider-35b-a3b_blind_v1.noctx.jsonl")
WS = (5, 10, 20, 50)


def stats(P, items, auto, W):
    ids = sorted(P); ok = np.array([int(np.argmax(P[i])) == items[i]["answer"] for i in ids])
    t3 = np.array([items[i]["answer"] in np.argsort(-P[i])[:3] for i in ids]); a = np.array([auto[i] for i in ids])
    return 100 * a.mean(), (100 * ok[a].mean() if a.any() else float("nan")), S.effort(a, ok, t3, wrong=W)


def rule(P, W):
    return {i: bool(S.decide(p.max(), np.sort(p)[-3:].sum(), W)) for i, p in P.items()}


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in json.loads((PROCESSED / "blind_v1.json").read_text())["items"]}
    encs = [load(f) for f in sorted(glob.glob(ENC))]; big = load(BIG)
    decs = {"recipe": [load(f) for f in sorted(glob.glob(DEC.format(oth="", set="blind_v1")))],
            "+ other users' line": [load(f) for f in sorted(glob.glob(DEC.format(oth="_oth50", set="blind_v1_others")))]}
    print(f"**Table SY2.1: blind_v1, effort per transaction under the per-item rule (mean [range] over seed pairs; decider seeds: "
          + ", ".join(f"{k} {len(v)}" for k, v in decs.items()) + f"; encoder seeds {len(encs)})**\n")
    print("| decider | system | " + " | ".join(f"W = {w}" for w in WS) + " | auto-filed % / precision % at W = 10 | sent to the 35B % |")
    print("|---|---|" + "---|" * len(WS) + "---|---|")
    for dname, ds in decs.items():
        res = {k: {w: [] for w in WS} for k in ("decider", "pair", "pair -> 35B", "three for all")}
        at10 = {k: [] for k in res}; sent = []
        for d in ds:
            Pd = folds_calibrate(d, items)
            for w in WS:
                s = stats(Pd, items, rule(Pd, w), w); res["decider"][w].append(s[2])
                if w == 10:
                    at10["decider"].append(s[:2])
            for e in encs:
                P2 = product([e, d], items); P3 = product([e, d, big], items)
                for w in WS:
                    a2 = rule(P2, w); a3 = rule(P3, w)
                    s = stats(P2, items, a2, w); res["pair"][w].append(s[2])
                    Pesc = {i: (P2[i] if a2[i] else P3[i]) for i in P2 if i in P3}
                    aesc = {i: a2[i] or a3[i] for i in Pesc}
                    s2 = stats(Pesc, items, aesc, w); res["pair -> 35B"][w].append(s2[2])
                    s3 = stats(P3, items, a3, w); res["three for all"][w].append(s3[2])
                    if w == 10:
                        at10["pair"].append(s[:2]); at10["pair -> 35B"].append(s2[:2]); at10["three for all"].append(s3[:2])
                        sent.append(100 * np.mean([not a2[i] for i in Pesc]))
        for k in res:
            a = np.array(at10[k]); snt = "0" if k in ("decider", "pair") else (fmt(sent) if k == "pair -> 35B" else "100")
            print(f"| {dname} | {k} | " + " | ".join(fmt(res[k][w], 3) for w in WS) + f" | {fmt(a[:, 0])} / {fmt(a[:, 1])} | {snt} |")
