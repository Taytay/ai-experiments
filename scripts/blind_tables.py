"""Tables for PLAN step 100 (EVAL-11): the readers on blind_v1, a test set built by a subagent from the product description and the prompt
format only (scripts/build_blind_v1.py; 250 users, 1,500 items, the production history slice). Its `best` is the builder's own rule-based
ideal reader (ceiling 87.9); a blind Opus 5.5 reader scored 91% on a stratified 120 (the builder's ceiling on those 120: 91.7).

  BL.1  per reader: top-1, share of the ceiling, top-3, calibrated auto-file at the 98% target (temperature and threshold from the
        other users' folds), items with more than 26 options
  BL.2  top-1 by what decides the item
  BL.3  top-1 by scheme
usage: uv run python scripts/blind_tables.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments import real6_cells as RC
from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
from effort_tables import calibrated  # noqa: E402

REC = "ren50_dbep50_mislead_v1_alt10s_lk10_ov10"
FINAL = f"decider_decider-4b_none_h100bf16st800{{s}}_f0_{REC}{{r7}}_aux100_labrand26_ev10soft_lora"
DM = "real6_dm_decider_{a}_blind_v1_labrand26.noctx.jsonl"
ENC = "real6_{n}_blind_v1.noctx.jsonl"
READERS = [("decider-4B untrained", [DM.format(a="decider-4b")]),
           ("decider-4B, final recipe (row 89)", [DM.format(a=FINAL.format(s=s, r7="")) for s in ("", "s1")]),
           ("decider-4B, + dated REAL-7 slices (row 98)", [DM.format(a=FINAL.format(s=s, r7="_r7all_kind")) for s in ("", "s1")]),
           ("Granite-4.0-micro, final recipe (row 101)", [DM.format(a=f"slot_granite-4.0-micro_none_h100bf16st800_f0_{REC}_aux100_labrand26_ev10soft_lora")]),
           ("GLiClass-large, recipe episodes (row 99)", [ENC.format(n=f"encgli_large_st3000_h100fresh_f0_decnone_h100fresh_f0_{REC}")]),
           ("Ettin-encoder 1B, recipe episodes (row 99)", [ENC.format(n=f"encmask_ettin1b_st3000_h100fresh_f0_decnone_h100fresh_f0_{REC}")])]
for enc, t in (("encgli_large", "GLiClass-large"), ("encmask_ettin1b", "Ettin-encoder 1B")):  # row 102
    READERS.append((f"{t}, + soft targets (row 102)", [ENC.format(n=f"{enc}_st3000_h100fresh_f0_decnone_h100fresh_f0_{REC}_ev10soft")]))
    READERS.append((f"{t}, + soft targets + dated REAL-7 slices (row 102)", [ENC.format(n=f"{enc}_st3000_h100fresh{s}_f0_decnone_h100fresh{s}_f0_{REC}_r7all_kind_u2400n2400_ev10soft") for s in ("", "s1")]))


def fmt(v):
    return "–" if not v else f"{np.mean(v):.1f}" + (f" [{min(v):.1f}, {max(v):.1f}]" if len(v) > 1 else "")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in json.loads((PROCESSED / "blind_v1.json").read_text())["items"]}
    ceil = 100 * np.mean([x["best"] == x["answer"] for x in items.values()])
    loaded = [(lab, [r for r in (RC.load_recs(f) for f in fs) if r]) for lab, fs in READERS]
    loaded = [(lab, rs) for lab, rs in loaded if rs]
    ok = lambda r, i: int(np.argmax(r[i]["sum_lp"])) == items[i]["answer"]  # noqa: E731
    print(f"**Table BL.1: blind_v1, 250 users, 1,500 items; ceiling {ceil:.1f} (the builder's ideal reader); blind Opus 5.5 91% on a stratified 120**\n")
    print("| reader | runs | top-1 | %% of ceiling | top-3 | auto-filed at 98%% (precision) | items with > 26 options (n = %d) |" % sum(len(x["options"]) > 26 for x in items.values()))
    print("|---|---|---|---|---|---|---|")
    for lab, rs in loaded:
        t1 = [100 * np.mean([ok(r, i) for i in r]) for r in rs]
        t3 = [100 * np.mean([items[i]["answer"] in np.argsort(-np.asarray(r[i]["sum_lp"]))[:3] for i in r]) for r in rs]
        cov, prec = [], []
        for r in rs:
            cal = calibrated(r, items); a = [i for i in cal if cal[i]["a98"]]
            cov.append(100 * len(a) / len(cal)); prec.append(100 * np.mean([int(np.argmax(cal[i]["p"])) == items[i]["answer"] for i in a]) if a else float("nan"))
        big = [100 * np.mean([ok(r, i) for i in r if len(items[i]["options"]) > 26]) for r in rs]
        print(f"| {lab} | {len(rs)} | {fmt(t1)} | {100 * np.mean(t1) / ceil:.1f}% | {fmt(t3)} | {np.mean(cov):.1f} ({np.nanmean(prec):.1f}) | {fmt(big)} |")
    for key, title in (("why", "BL.2: top-1 by what decides the item"), ("scheme", "BL.3: top-1 by scheme")):
        vals = sorted({x[key] for x in items.values()}, key=lambda v: -sum(x[key] == v for x in items.values()))
        print(f"\n**Table {title} (n, ceiling)**\n")
        print("| " + key + " | n | ceiling | " + " | ".join(lab for lab, _ in loaded) + " |"); print("|---|---|---|" + "---|" * len(loaded))
        for v in vals:
            ids = [i for i, x in items.items() if x[key] == v]
            c = 100 * np.mean([items[i]["best"] == items[i]["answer"] for i in ids])
            print(f"| {v} | {len(ids)} | {c:.0f} | " + " | ".join(f"{np.mean([100 * np.mean([ok(r, i) for i in ids if i in r]) for r in rs]):.0f}" for _, rs in loaded) + " |")
