"""Tables for PLAN step 107 (MODEL-17, REAL-19): real Overture business names as training episodes (OVDB 0.2) for decider-4B (the recipe,
two seeds) and Ettin-1B (the encoder recipe), against the same recipes without them.

  OV.1  novel_merchants_v1 (fold 0's users; real businesses in no database, obscure renderings) by name group, blind_v1 overall and on its
        new-merchant and new-user groups, blind calibrated auto-filing, REAL-6 fold 0, novel category names, misleading names in the DB
usage: uv run python scripts/overture_ep_tables.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments import real6_cells as RC

sys.path.insert(0, str(Path(__file__).parent))
import rematch_tables as R  # noqa: E402
from effort_tables import calibrated  # noqa: E402

D = "real6_dm_decider_decider_decider-4b_none_h100bf16st800{s}_emp20_f0_" + R.REC + "{o}_aux100_labrand255_ev10soft_lora_{set}_labrand255.noctx.jsonl"
E = "real6_encmask_ettin1b_st3000_h100fresh_f0_decnone_h100fresh_f0_" + R.REC + "{o}_ev10soft{set}.noctx.jsonl"
READERS = [("decider-4B, recipe", [lambda st, s=s: D.format(s=s, o="", set=st or "real6") for s in ("", "s1")]),
           ("decider-4B, recipe + Overture episodes", [lambda st, s=s: D.format(s=s, o="_odb20", set=st or "real6") for s in ("", "s1")]),
           ("Ettin-1B, encoder recipe", [lambda st: E.format(o="", set="_" + st if st else "")]),
           ("Ettin-1B, encoder recipe + Overture episodes", [lambda st: E.format(o="_odb20", set="_" + st if st else "")])]


def fmt(v):
    v = [x for x in v if x is not None and not np.isnan(x)]
    return "–" if not v else f"{np.mean(v):.1f}" + (f" [{min(v):.1f}, {max(v):.1f}]" if len(v) > 1 else "")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    nm, bl, r6, nov, mi = (R.items_of(s) for s in ("novel_merchants_v1", "blind_v1", "", "real6_v1_novel", "mislead_v1"))
    ok = lambda r, it, i: int(np.argmax(r[i]["sum_lp"])) == it[i]["answer"]  # noqa: E731
    print("**Table OV.1: real business names in training (mean [range] over seeds)**\n")
    print("| reader | novel merchants: descriptive / plain / chain | blind_v1 | blind new merchant: descriptive / chain / opaque | blind new users | blind auto-filed at 98% | REAL-6 | novel names | misleading names in the DB |")
    print("|---|---|---|---|---|---|---|---|---|")
    for lab, fs in READERS:
        c = {k: [] for k in ("nd", "np", "nc", "b", "bd", "bc", "bo", "bs", "ba", "r6", "nv", "m")}
        for f in fs:
            r = RC.load_recs(f("novel_merchants_v1"))
            if r:
                for g, k in (("descriptive", "nd"), ("plain", "np"), ("chain", "nc")):
                    c[k].append(100 * np.mean([ok(r, nm, i) for i in r if nm[i]["nm_group"] == g and nm[i]["user"] % 4 == 0]))
            r = RC.load_recs(f("blind_v1"))
            if r:
                c["b"].append(100 * np.mean([ok(r, bl, i) for i in r]))
                for w, k in (("new_merchant_descriptive", "bd"), ("new_merchant_chain", "bc"), ("new_merchant_opaque", "bo"), ("short_history", "bs")):
                    c[k].append(100 * np.mean([ok(r, bl, i) for i in r if bl[i]["why"] == w]))
                cal = calibrated(r, bl); c["ba"].append(100 * np.mean([cal[i]["a98"] for i in cal]))
            for st, it, k, f0 in (("", r6, "r6", True), ("real6_v1_novel", nov, "nv", True)):
                r = RC.load_recs(f(st))
                if r:
                    c[k].append(R.top1(r, it, f0)[0])
            r = RC.load_recs(f("mislead_v1"))
            if r:
                c["m"].append(100 * np.mean([ok(r, mi, i) for i in r if mi[i]["mkind"] == "misleading" and mi[i]["in_db"] and mi[i]["cond"] == "none"]))
        print(f"| {lab} | {fmt(c['nd'])} / {fmt(c['np'])} / {fmt(c['nc'])} | {fmt(c['b'])} | {fmt(c['bd'])} / {fmt(c['bc'])} / {fmt(c['bo'])} | {fmt(c['bs'])} | {fmt(c['ba'])} | {fmt(c['r6'])} | {fmt(c['nv'])} | {fmt(c['m'])} |")
