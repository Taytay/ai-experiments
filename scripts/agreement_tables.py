"""Tables for PLAN step 116 (owner, 2026-09-29): using the encoder and decider together. On blind_v1, the encoder (Ettin-1B, the encoder
recipe, three seeds) and decider-4B (the recipe with labelled rows, REPORT 113, two seeds); every seed pair.

  AG.1  how often the two agree on the top answer, and top-1 when they agree / disagree
  AG.2  auto-file policies at a 98% target (every threshold chosen on the other users' folds, as the scorecard): each alone; the cascade
        (encoder first); both agree; both agree and each clears its own threshold; the product of the calibrated distributions (p_enc x
        p_dec, renormalised) with one threshold; the mean of the two. Coverage, realised precision, effort (suggestions from the combined
        distribution where there is one, else decider's)
  AG.3  is the encoder's confidence a probability? Reliability on blind_v1, raw and after the temperature fitted on the other users'
        folds: confidence bins against the accuracy in them, ECE; decider beside it
usage: uv run python scripts/agreement_tables.py
"""
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments import calibration as C
from ai_experiments import scorecard as S
from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
from effort_tables import calibrated  # noqa: E402

REC = "ren50_dbep50_mislead_v1_alt10s_lk10_ov10"
ENC = f"results/per_item/real6_encmask_ettin1b_st3000_h100fresh*_f0_decnone_h100fresh*_emp20_f0_{REC}_odb20_short15_kinds20_ev10soft_blind_v1.noctx.jsonl"
DEC = f"results/per_item/real6_dm_decider_decider_decider-4b_none_h100bf16st800*_emp20_f0_{REC}_aux100_labrand255_laylabelled_shots_ev10soft_lora_blind_v1_labrand255_laylabelled_shots.noctx.jsonl"


def load(f):
    return {r["id"]: r for r in map(json.loads, open(f))}


def fold_threshold(ids, conf, correct, users, target=0.98):
    """Per item: accepted or not, the threshold chosen on the other users' folds (user id mod 4)."""
    acc = {}
    for f in range(4):
        fit = [i for i in ids if users[i] % 4 != f]; own = [i for i in ids if users[i] % 4 == f]
        th = C.select_threshold(np.array([conf[i] for i in fit]), np.array([correct[i] for i in fit]), 1 - target)
        for i in own:
            acc[i] = th is not None and conf[i] >= th
    return acc


def policy_row(items, ids, auto, pick, sugg):
    """auto: accepted; pick: the auto-filed option; sugg: the distribution the top-3 suggestions come from."""
    a = [auto[i] for i in ids]; ok = [pick[i] == items[i]["answer"] for i in ids]
    t3 = [items[i]["answer"] in np.argsort(-sugg[i])[:3] for i in ids]
    prec = 100 * np.mean([o for x, o in zip(a, ok) if x]) if any(a) else float("nan")
    return 100 * np.mean(a), prec, S.effort(a, ok, t3)


def fmt(v, d=1):
    v = np.array([x for x in v if not np.isnan(x)])
    return "–" if not len(v) else f"{v.mean():.{d}f} [{v.min():.{d}f}, {v.max():.{d}f}]" if len(v) > 1 else f"{v.mean():.{d}f}"


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in json.loads((PROCESSED / "blind_v1.json").read_text())["items"]}
    users = {i: x["user"] for i, x in items.items()}
    encs = [load(f) for f in sorted(glob.glob(ENC))]; decs = [load(f) for f in sorted(glob.glob(DEC))]
    ag, pol = {k: [] for k in ("agree", "acc_agree", "acc_dis", "enc_right_dis", "dec_right_dis")}, {}
    for er in encs:
        ec = calibrated(er, items)
        for dr in decs:
            dc = calibrated(dr, items); ids = sorted(set(ec) & set(dc))
            pe = {i: ec[i]["p"] for i in ids}; pd = {i: dc[i]["p"] for i in ids}
            ae = {i: int(np.argmax(pe[i])) for i in ids}; ad = {i: int(np.argmax(pd[i])) for i in ids}
            y = {i: items[i]["answer"] for i in ids}; same = [ae[i] == ad[i] for i in ids]
            ag["agree"].append(100 * np.mean(same))
            ag["acc_agree"].append(100 * np.mean([ad[i] == y[i] for i, s in zip(ids, same) if s]))
            ag["acc_dis"].append(100 * np.mean([ad[i] == y[i] for i, s in zip(ids, same) if not s]))
            ag["enc_right_dis"].append(100 * np.mean([ae[i] == y[i] for i, s in zip(ids, same) if not s]))
            prod = {}
            for i in ids:
                q = np.exp(np.log(np.clip(pe[i], 1e-12, 1)) + np.log(np.clip(pd[i], 1e-12, 1))); prod[i] = q / q.sum()
            mean = {i: (pe[i] + pd[i]) / 2 for i in ids}
            rows = {}
            rows["decider alone (98%)"] = ({i: dc[i]["a98"] for i in ids}, ad, pd)
            rows["encoder alone (98%)"] = ({i: ec[i]["a98"] for i in ids}, ae, pe)
            rows["cascade: encoder at its 98%, else decider at its 98%"] = ({i: ec[i]["a98"] or dc[i]["a98"] for i in ids},
                                                                            {i: ae[i] if ec[i]["a98"] else ad[i] for i in ids}, pd)
            rows["both agree (no threshold)"] = ({i: ae[i] == ad[i] for i in ids}, ad, prod)
            rows["both agree and both clear their 98%"] = ({i: ae[i] == ad[i] and ec[i]["a98"] and dc[i]["a98"] for i in ids}, ad, prod)
            rows["both agree and either clears its 98%"] = ({i: ae[i] == ad[i] and (ec[i]["a98"] or dc[i]["a98"]) for i in ids}, ad, prod)
            for name, dist in (("product of the two distributions, one 98% threshold", prod), ("mean of the two distributions, one 98% threshold", mean)):
                top = {i: int(np.argmax(dist[i])) for i in ids}
                acc = fold_threshold(ids, {i: dist[i].max() for i in ids}, {i: top[i] == y[i] for i in ids}, users)
                rows[name] = (acc, top, dist)
            for name, (a, p, sg) in rows.items():
                pol.setdefault(name, []).append(policy_row(items, ids, a, p, sg))
    print(f"**Table AG.1: blind_v1, encoder ({len(encs)} seeds) and decider with labelled rows ({len(decs)} seeds): {len(encs) * len(decs)} pairs**\n")
    print("| agree on the top answer | decider's top-1 where they agree | where they disagree: decider right | encoder right |"); print("|---|---|---|---|")
    print(f"| {fmt(ag['agree'])} | {fmt(ag['acc_agree'])} | {fmt(ag['acc_dis'])} | {fmt(ag['enc_right_dis'])} |")
    print("\n**Table AG.2: auto-file policies (mean [range] over the pairs)**\n")
    print("| policy | auto-filed % | precision % | effort |"); print("|---|---|---|---|")
    for name, v in pol.items():
        v = np.array(v); print(f"| {name} | {fmt(v[:, 0])} | {fmt(v[:, 1])} | {fmt(v[:, 2], 3)} |")
    print("\n**Table AG.3: reliability on blind_v1 (first seed of each): the share right among answers given with confidence in each bin**\n")
    bins = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0001]
    print("| model | probabilities | " + " | ".join(f"{a:.2f}-{min(b, 1):.2f}" for a, b in zip(bins, bins[1:])) + " | ECE |"); print("|---|---|" + "---|" * (len(bins)))
    for lab, rec in (("encoder", encs[0]), ("decider", decs[0])):
        ids = sorted(i for i in rec if i in items); y = [items[i]["answer"] for i in ids]
        for kind in ("raw", "calibrated"):
            if kind == "raw":
                P = [C.softmax(np.asarray(rec[i]["sum_lp"], float)) for i in ids]
            else:
                cal = calibrated(rec, items); P = [cal[i]["p"] for i in ids]
            conf = np.array([p.max() for p in P]); ok = np.array([int(np.argmax(p)) == t for p, t in zip(P, y)])
            cells = []
            for a, b in zip(bins, bins[1:]):
                m = (conf >= a) & (conf < b)
                cells.append(f"{100 * ok[m].mean():.0f} (n={m.sum()})" if m.sum() else "–")
            print(f"| {lab} | {kind} | " + " | ".join(cells) + f" | {100 * C.ece(conf, ok):.1f} |")
