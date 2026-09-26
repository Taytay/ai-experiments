"""Tables for PLAN step 65 (POI-1): user categorisation of real Overture places, at 12 to 20 categories per user.

  P.1  the scorecard on fold 0's held-out users (the users every reader can be scored on): untrained 3B / 7B / 14B, two REAL-6
       categorisers as transfer, the POI-1-trained categorisers; the baselines include the kind lookup (the user's label for another
       place of the same Overture basic category: what a places database and the user's history give without a model)
  P.2  top-1 by item level (kind seen or unseen in the user's history x readable / merged / coined category name), fold 0, with blind
       Opus 5.5 on its level-stratified sample (all users)
  P.3  the untrained and transfer readers on all 200 users
usage: uv run python scripts/poi1_tables.py
"""
import json
import sys
from pathlib import Path

import numpy as np

from ai_experiments import real6_cells as RC
from ai_experiments import scorecard as S
from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
import blind_ceiling as BC  # noqa: E402

SET = "poi1_v1"
CAT = "real6_categoriser_Qwen2.5-3B-Instruct"
READERS = [("Qwen2.5-3B-Instruct, untrained", f"real6_Qwen2.5-3B-Instruct_{SET}.noctx.jsonl"),
           ("Qwen2.5-7B-Instruct, untrained", f"real6_Qwen2.5-7B-Instruct_{SET}.noctx.jsonl"),
           ("Qwen2.5-14B-Instruct, untrained", f"real6_Qwen2.5-14B-Instruct_{SET}.noctx.jsonl"),
           ("3B trained on REAL-6 (all-label), transfer", f"{CAT}_none_h100bf16_f0_alllab_lora_{SET}.noctx.jsonl"),
           ("3B trained on REAL-6 (all-label + rename), transfer", f"{CAT}_none_h100bf16_f0_ren50_alllab_lora_{SET}.noctx.jsonl"),
           ("3B trained on POI-1, 200 steps", f"{CAT}_{SET}_none_h100bf16st200_f0_alllab_lora_{SET}.noctx.jsonl"),
           ("3B trained on POI-1, 800 steps", f"{CAT}_{SET}_none_h100bf16_f0_alllab_lora_{SET}.noctx.jsonl"),
           ("3B trained on POI-1, 800 steps + rename", f"{CAT}_{SET}_none_h100bf16_f0_ren50_alllab_lora_{SET}.noctx.jsonl")]
HEAD = ("| reader | n | top-1 [interval] | top-3 | MRR | bits left | auto-file at 98%: coverage (precision) | usage prior top-1 / top-3 | "
        "kind lookup: share, top-1 where it answers | kind lookup → other users → prior, top-1 | skill top-1 over it / top-3 |")


def load():
    doc = json.loads((PROCESSED / f"{SET}.json").read_text())
    return {i["id"]: i for i in doc["items"]}, {u["user"]: u for u in doc["users"]}


def card(recs, items, users, fold0):
    sc = S.scorecard(recs, items, users=users, fold_of=(lambda u: (u // 4) % 4) if fold0 else None)
    best = max(sc["prior_top1"], sc["kind_cascade_top1"])
    sc["skill_kind"] = 100 * (sc["top1"] - best) / (100 - best)
    return sc


def row(label, sc):
    return (f"| {label} | {sc['n']} | {sc['top1']:.1f} [{sc['top1_ci'][0]}, {sc['top1_ci'][1]}] | {sc['top3']:.1f} | {sc['mrr']:.2f} | {sc['bits']:.2f} | "
            f"{sc['cov98']:.1f} ({sc['prec98']:.1f}) | {sc['prior_top1']:.1f} / {sc['prior_top3']:.1f} | {sc['kind_share']:.0f}%, {sc['kind_top1']:.1f} | "
            f"{sc['kind_cascade_top1']:.1f} | {sc['skill_kind']:.0f} / {sc['skill_top3']:.0f} |")


def table(title, items, users, only_fold0):
    print(title + "\n"); print(HEAD); print("|" + "---|" * 11)
    out = {}
    for label, pat in READERS:
        recs = RC.load_recs(pat)
        recs = {i: r for i, r in recs.items() if not only_fold0 or items[i]["user"] % 4 == 0}
        if not recs:
            continue
        out[label] = card(recs, items, users, only_fold0); print(row(label, out[label]))
    return out


if __name__ == "__main__":
    items, users = load()
    table("**Table P.1: POI-1, fold 0's held-out users (50 users): the scorecard (temperature and auto-file thresholds fitted leave-users-out "
          "within the fold's users, grouped by (id // 4) mod 4; kind lookup = the user's label for another place of the same Overture basic category)**",
          items, users, True)
    levels = sorted({i["level"] for i in items.values()})
    print("\n**Table P.2: top-1 % by level, fold 0 (seen / unseen = the place's Overture basic category is / is not in the user's history; "
          "standard / renamed / new = readable / merged / coined category name); blind Opus on its own sample (all users, 20 per level)**\n")
    print("| reader | " + " | ".join(levels) + " |"); print("|---|" + "---|" * len(levels))
    for label, pat in READERS:
        recs = {i: r for i, r in RC.load_recs(pat).items() if items[i]["user"] % 4 == 0}
        if recs:
            print(f"| {label} | " + " | ".join(f"{100 * np.mean([int(np.argmax(r['sum_lp'])) == r['answer'] for i, r in recs.items() if items[i]['level'] == lv]):.0f}" for lv in levels) + " |")
    try:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            got = BC.score_levels(SET)
        if got:
            print("| blind Opus 5.5 (sample) | " + " | ".join(f"{100 * np.mean([ok for i, ok in got.items() if items[i]['level'] == lv]):.0f} ({sum(items[i]['level'] == lv for i in got)})" for lv in levels) + " |")
    except FileNotFoundError:
        pass
    print()
    table("**Table P.3: POI-1, all 200 users (the untrained and transfer readers only)**", items, users, False)
