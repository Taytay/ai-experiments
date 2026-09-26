"""Tables for PLAN step 53 (MODEL-6): the [MASK]-per-option encoder (Laya's layout) against the 3B categoriser, fold 0's held-out users.

  M.1  REAL-6, fold 0 (5 users, 298 items): the scorecard (top-1 / top-3 / bits / auto-file) and top-1 by corrected group, with ms per item
  M.2  POI-1, fold 0 (50 users, 507 items): the same readers trained on POI-1's users
usage: uv run python scripts/encmask_tables.py
"""
import json
import warnings

import numpy as np

from ai_experiments import real6 as R6
from ai_experiments import real6_cells as RC
from ai_experiments import scorecard as S
from ai_experiments.paths import PROCESSED, ROOT

CAT = "real6_categoriser_Qwen2.5-3B-Instruct"
REAL6 = [("encoder: Laya checkpoint, untrained", "real6_encmask_laya_zeroshot_f0.noctx.jsonl", "encmask_laya_zeroshot_f0"),
         ("encoder: ModernBERT-large, 1,500 steps", "real6_encmask_mbert_st1500_f0.noctx.jsonl", "encmask_mbert_st1500_f0"),
         ("encoder: Laya init, 1,500 steps", "real6_encmask_laya_st1500_f0.noctx.jsonl", "encmask_laya_st1500_f0"),
         ("encoder: Laya init, 4,000 steps", "real6_encmask_laya_st4000_f0.noctx.jsonl", "encmask_laya_st4000_f0"),
         ("3B SFT no DB, all-label (H100 bf16)", f"{CAT}_none_h100bf16_f0_alllab_lora.noctx.jsonl", None),
         ("3B database episodes (H100 bf16)", f"{CAT}_none_h100bf16_f0_alllab_dbep50_lora.noctx.jsonl", None),
         ("encoder + record: ModernBERT-large", "real6_encmask_mbert_st1500_f0_ctx.ctx.jsonl", "encmask_mbert_st1500_f0_ctx"),
         ("encoder + record: Laya init", "real6_encmask_laya_st1500_f0_ctx.ctx.jsonl", "encmask_laya_st1500_f0_ctx"),
         ("3B + record in the prompt (H100 bf16)", f"{CAT}_ret_h100bf16_f0_lora.ctx.jsonl", None),
         ("3B + record with category (H100 bf16)", f"{CAT}_ret_h100bf16_f0_reccat_lora.ctx.jsonl", None)]
POI = [("encoder: ModernBERT-large, 1,500 steps", "real6_encmask_mbert_poi1_v1_st1500_f0.noctx.jsonl", "encmask_mbert_poi1_v1_st1500_f0"),
       ("encoder: Laya init, 1,500 steps", "real6_encmask_laya_poi1_v1_st1500_f0.noctx.jsonl", "encmask_laya_poi1_v1_st1500_f0"),
       ("3B, 200 steps", f"{CAT}_poi1_v1_none_h100bf16st200_f0_alllab_lora_poi1_v1.noctx.jsonl", None),
       ("3B, 800 steps", f"{CAT}_poi1_v1_none_h100bf16_f0_alllab_lora_poi1_v1.noctx.jsonl", None),
       ("3B, 800 steps + rename", f"{CAT}_poi1_v1_none_h100bf16_f0_ren50_alllab_lora_poi1_v1.noctx.jsonl", None)]
HEAD = "| reader | n | top-1 [interval] | top-3 | bits left | auto-file at 98%: coverage (precision) | ms per item, batched / one at a time |"


def ms(name):
    if not name:
        return "-"
    d = json.loads((ROOT / "results" / f"{name}.json").read_text()); c = d.get("noctx") or d.get("ctx")
    return f"{c['ms_per_item_batched']:.0f} / {c['ms_per_item_single']:.0f}"


def table(runs, items, users=None, groups=False):
    print(HEAD + (" " + " | ".join(RC.KINDS[:4]) + " |" if groups else "")); print("|---" * (7 + (4 if groups else 0)) + "|")
    for label, pat, name in runs:
        recs = {i: r for i, r in RC.load_recs(pat).items() if items[i]["user"] % 4 == 0}
        if not recs:
            print(f"| {label} | missing |"); continue
        sc = S.scorecard(recs, items, users=users, fold_of=lambda u: (u // 4) % 4)
        cells = ""
        if groups:
            cells = " " + " | ".join(f"{100 * np.mean([sc['_per_item'][i]['top1'] for i in sc['_per_item'] if RC.kind(i) == g]):.0f}" for g in RC.KINDS[:4]) + " |"
        print(f"| {label} | {sc['n']} | {sc['top1']:.1f} [{sc['top1_ci'][0]}, {sc['top1_ci'][1]}] | {sc['top3']:.1f} | {sc['bits']:.2f} | {sc['cov98']:.1f} ({sc['prec98']:.1f}) | {ms(name)} |" + cells)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("**Table M.1: REAL-6, fold 0's held-out users (5 users; calibration leave-users-out within them): the encoder against the 3B; "
          "top-1 by corrected group on the right**\n")
    table(REAL6, {i["id"]: i for i in R6.load()["items"]}, groups=True)
    doc = json.loads((PROCESSED / "poi1_v1.json").read_text())
    print("\n**Table M.2: POI-1, fold 0's held-out users (50 users), readers trained on POI-1's other users**\n")
    table(POI, {i["id"]: i for i in doc["items"]}, users={u["user"]: u for u in doc["users"]})
