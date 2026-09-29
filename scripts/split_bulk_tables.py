"""Tables for PLAN step 118 (INFRA-3): the split layout (a sync's shared rows first, then the payee's own and similar payees' rows) against
today's layout, on blind_v1 (the same rows reordered: blind_v1_split) and on blind_bulk_v1 (one user's sync: the next 30 transactions after a
cutoff, history filed before it), for decider-4B with labelled rows: the recipe's adapters read zero-shot, and adapters trained with split
episodes (SPLIT=1). Top-1, auto-filed % / precision at the calibrated 98% threshold (leave-fold-out), effort; and the system (the encoder
and decider agree and either is confident, REPORT 115) on the bulk set.
usage: uv run python scripts/split_bulk_tables.py
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
from agreement_tables import fmt, load  # noqa: E402
from effort_tables import calibrated  # noqa: E402

REC = "ren50_dbep50_mislead_v1_alt10s_lk10_ov10"
DEC = "results/per_item/real6_dm_decider_decider_decider-4b_none_h100bf16st800{s}_emp20_f0_" + REC + "_aux100_labrand255_laylabelled_shots{split}_ev10soft_lora_{set}_labrand255_laylabelled_shots.noctx.jsonl"
ENC = "results/per_item/real6_encmask_ettin1b_st3000_h100fresh*_f0_decnone_h100fresh*_emp20_f0_" + REC + "_odb20_short15_kinds20_ev10soft_{set}.noctx.jsonl"
SEEDS = ("", "s1")


def metrics(recs, items):
    cal = calibrated(recs, items); ids = sorted(cal)
    top = {i: int(np.argmax(cal[i]["p"])) for i in ids}; ok = [top[i] == items[i]["answer"] for i in ids]
    a = [cal[i]["a98"] for i in ids]; t3 = [items[i]["answer"] in np.argsort(-cal[i]["p"])[:3] for i in ids]
    return 100 * np.mean(ok), 100 * np.mean(a), (100 * np.mean([o for x, o in zip(a, ok) if x]) if any(a) else float("nan")), S.effort(a, ok, t3)


def system(dec, enc, items):
    dc, ec = calibrated(dec, items), calibrated(enc, items); ids = sorted(set(dc) & set(ec))
    ad = {i: int(np.argmax(dc[i]["p"])) for i in ids}; ae = {i: int(np.argmax(ec[i]["p"])) for i in ids}
    a = [ad[i] == ae[i] and (dc[i]["a98"] or ec[i]["a98"]) for i in ids]; ok = [ad[i] == items[i]["answer"] for i in ids]
    t3 = []
    for i in ids:
        q = np.clip(dc[i]["p"], 1e-12, 1) * np.clip(ec[i]["p"], 1e-12, 1); t3.append(items[i]["answer"] in np.argsort(-q)[:3])
    return 100 * np.mean(a), (100 * np.mean([o for x, o in zip(a, ok) if x]) if any(a) else float("nan")), S.effort(a, ok, t3)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sets = {n: {i["id"]: i for i in json.loads((PROCESSED / f"{n}.json").read_text())["items"]}
            for n in ("blind_v1", "blind_v1_split", "blind_bulk_v1", "blind_bulk_v1_split")}
    print("**Table SB.1: decider-4B with labelled rows, by layout (mean [range] over two seeds): top-1 / auto-filed % / precision % / effort**\n")
    print("| adapter | read on | top-1 | auto-filed % | precision % | effort |"); print("|---|---|---|---|---|---|")
    for split, lab in (("", "the recipe (trained on today's layout)"), ("_split", "trained on split episodes (SPLIT=1)")):
        for st in sets:
            files = [DEC.format(s=s, split=split, set=st) for s in SEEDS]
            files = [f for f in files if Path(f).exists()]
            if not files:
                continue
            v = np.array([metrics(load(f), sets[st]) for f in files])
            print(f"| {lab} | {st} | {fmt(v[:, 0])} | {fmt(v[:, 1])} | {fmt(v[:, 2])} | {fmt(v[:, 3], 3)} |")
    print("\n**Table SB.2: the system on the bulk set (the encoder reads today's layout; decider as shown), agree and either confident**\n")
    print("| decider | read on | auto-filed % | precision % | effort |"); print("|---|---|---|---|---|")
    encs = [load(f) for f in sorted(glob.glob(ENC.format(set="blind_bulk_v1")))]
    for split, lab in (("", "the recipe"), ("_split", "split-trained")):
        for st in ("blind_bulk_v1", "blind_bulk_v1_split"):
            files = [f for f in (DEC.format(s=s, split=split, set=st) for s in SEEDS) if Path(f).exists()]
            if not files or not encs:
                continue
            v = np.array([system(load(f), e, sets[st]) for f in files for e in encs])
            print(f"| {lab} | {st} | {fmt(v[:, 0])} | {fmt(v[:, 1])} | {fmt(v[:, 2], 3)} |")
