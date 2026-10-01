"""Tables for PLAN step 90: the final recipe (section 89: ALT 0.1) over all 20 REAL-6 users, each fold's model read on its own held-out
users (fold 0 from row 89 seed 0, folds 1 to 3 from row 90), against the four-fold figures of earlier sections.

  F.1  REAL-6 scorecard pooled over the four folds (thresholds and temperature leave-fold-out), novel names, real places in no DB
usage: uv run python scripts/final_folds_tables.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments import real6 as R6
from ai_experiments import real6_cells as RC
from ai_experiments import scorecard as S
from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))

P = "real6_dm_decider_decider_decider-4b_none_h100bf16st800_f{f}_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_aux100_labrand26_ev10soft_lora_{s}_labrand26.noctx.jsonl"


def pooled(s, items):
    out = {}
    for f in range(4):
        for i, r in RC.load_recs(P.format(f=f, s=s)).items():
            if i in items and items[i]["user"] % 4 == f:
                out[i] = r
    return out


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    r6 = {i["id"]: i for i in R6.load()["items"]}
    rec = pooled("real6", r6)
    folds = sorted({r6[i]["user"] % 4 for i in rec})
    sc = S.scorecard(rec, r6, fold_of=lambda u: u % 4)
    print(f"**Table F.1: the final recipe, REAL-6 over the held-out users of folds {folds} ({len(rec)} items)**\n")
    print("| set | n | top-1 [user interval] | top-3 | bits left | auto-filed at 98% (precision) | auto-filed at 95% (precision) |")
    print("|---|---|---|---|---|---|---|")
    print(f"| REAL-6 | {sc['n']} | {sc['top1']:.1f} [{sc['top1_ci'][0]}, {sc['top1_ci'][1]}] | {sc['top3']:.1f} | {sc['bits']:.2f} | "
          f"{sc['cov98']:.1f} ({sc['prec98']:.1f}) | {sc['cov95']:.1f} ({sc['prec95']:.1f}) |")
    for s, lab in (("real6_v1_novel", "novel names"), ("novel_merchants_v1_clean", "real places in no DB")):
        items = r6 if s == "real6_v1_novel" else {i["id"]: i for i in json.loads((PROCESSED / f"{s}.json").read_text())["items"]}
        r = pooled(s, items)
        if r:
            print(f"| {lab} | {len(r)} | {100 * np.mean([int(np.argmax(x['sum_lp'])) == x['answer'] for x in r.values()]):.1f} | | | | |")
    per = {f: [i for i in rec if r6[i]["user"] % 4 == f] for f in folds}
    print("\nPer fold top-1: " + ", ".join(f"fold {f} {100 * np.mean([int(np.argmax(rec[i]['sum_lp'])) == rec[i]['answer'] for i in ids]):.1f} (n={len(ids)})"
                                         for f, ids in per.items()))
