"""Table for PLAN row 76 (MODEL-13): does a hop-trained encoder transfer to the categoriser? Row 53's layout (Laya's head on
ModernBERT-large) started from ModernBERT-large, Laya's checkpoint, or a row 71 v2 hop encoder (plain options arm; labelled lines +
hop loss), read on fold 0's held-out users of REAL-6 and POI-1 after 0 / 200 / 1,500 training steps.

  T.1  top-1 [user-resampled interval] and bits left by start and training steps, REAL-6 then POI-1
usage: uv run python scripts/transfer_tables.py
"""
import json
import warnings

from ai_experiments import real6 as R6
from ai_experiments import real6_cells as RC
from ai_experiments import scorecard as S
from ai_experiments.paths import PROCESSED

STARTS = [("ModernBERT-large", "mbert"), ("Laya checkpoint", "laya"), ("hop encoder, plain", "hops-v2-options-k1-3"),
          ("hop encoder, labelled lines + hop loss", "hops-v2-options-lab-aux1-k1-3")]
STEPS = [("0 (zero-shot)", "zeroshot"), ("200", "st200"), ("1,500", "st1500")]


def cell(pat, items, users):
    recs = {i: r for i, r in RC.load_recs(pat).items() if items[i]["user"] % 4 == 0}
    if not recs:
        return "-"
    sc = S.scorecard(recs, items, users=users, fold_of=lambda u: (u // 4) % 4)
    return f"{sc['top1']:.1f} [{sc['top1_ci'][0]}, {sc['top1_ci'][1]}] / {sc['bits']:.2f}"


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    doc = json.loads((PROCESSED / "poi1_v1.json").read_text())
    for title, items, users, mid in [("REAL-6, fold 0 (5 users)", {i["id"]: i for i in R6.load()["items"]}, None, ""),
                                     ("POI-1, fold 0 (50 users)", {i["id"]: i for i in doc["items"]}, {u["user"]: u for u in doc["users"]}, "poi1_v1_")]:
        print(f"\n**Table T.1 {title}: top-1 [interval] / bits left, by the categoriser's starting point and training steps**\n")
        print("| start | " + " | ".join(f"{s} steps" for s, _ in STEPS) + " |"); print("|---" * (len(STEPS) + 1) + "|")
        for label, init in STARTS:
            print(f"| {label} | " + " | ".join(cell(f"real6_encmask_{init}_{mid}{tag}_f0.noctx.jsonl", items, users) for _, tag in STEPS) + " |")
