"""Tables for PLAN steps 53 and 69 (MODEL-6, MODEL-10): the encoder option scorers (Laya's layout, GLiClass, ModernBERT-Instruct) against
the 3B categoriser, fold 0's held-out users. Besides top-k and bits: precision at 25 / 50 / 75% coverage (items taken in order of the
calibrated top probability) and AURC (mean error over all coverages; lower is better), and position bias (total variation between
where the model's picks sit in the option list and where the gold answers sit; 0 = none).

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
         ("GLiClass large, untrained", "real6_encgli_large_zeroshot_f0.noctx.jsonl", "encgli_large_zeroshot_f0"),
         ("GLiClass large, 1,500 steps", "real6_encgli_large_st1500_f0.noctx.jsonl", "encgli_large_st1500_f0"),
         ("ModernBERT-Instruct, letters, untrained", "real6_encmbi_instruct_zeroshot_f0.noctx.jsonl", "encmbi_instruct_zeroshot_f0"),
         ("ModernBERT-Instruct, letters, 1,500 steps", "real6_encmbi_instruct_st1500_f0.noctx.jsonl", "encmbi_instruct_st1500_f0"),
         ("ModernBERT-Instruct, unused-token IDs, 1,500 steps", "real6_encmbi_instruct_st1500_f0_unused.noctx.jsonl", "encmbi_instruct_st1500_f0_unused"),
         ("ModernBERT-Instruct, letters, shot labels carry the letter", "real6_encmbi_instruct_st1500_f0_shotlab.noctx.jsonl", "encmbi_instruct_st1500_f0_shotlab"),
         ("ModernBERT-Instruct, unused IDs, shot labels carry the ID", "real6_encmbi_instruct_st1500_f0_unused_shotlab.noctx.jsonl", "encmbi_instruct_st1500_f0_unused_shotlab"),
         ("ModernBERT-Instruct, letters, 5,000 steps", "real6_encmbi_instruct_st5000_f0.noctx.jsonl", "encmbi_instruct_st5000_f0"),
         ("3B SFT no DB, all-label (H100 bf16)", f"{CAT}_none_h100bf16_f0_alllab_lora.noctx.jsonl", None),
         ("3B database episodes (H100 bf16)", f"{CAT}_none_h100bf16_f0_alllab_dbep50_lora.noctx.jsonl", None),
         ("encoder + record: ModernBERT-large", "real6_encmask_mbert_st1500_f0_ctx.ctx.jsonl", "encmask_mbert_st1500_f0_ctx"),
         ("encoder + record: Laya init", "real6_encmask_laya_st1500_f0_ctx.ctx.jsonl", "encmask_laya_st1500_f0_ctx"),
         ("GLiClass large + record", "real6_encgli_large_st1500_f0_ctx.ctx.jsonl", "encgli_large_st1500_f0_ctx"),
         ("ModernBERT-Instruct + record", "real6_encmbi_instruct_st1500_f0_ctx.ctx.jsonl", "encmbi_instruct_st1500_f0_ctx"),
         ("3B + record in the prompt (H100 bf16)", f"{CAT}_ret_h100bf16_f0_lora.ctx.jsonl", None),
         ("3B + record with category (H100 bf16)", f"{CAT}_ret_h100bf16_f0_reccat_lora.ctx.jsonl", None)]
POI = [("encoder: ModernBERT-large, 1,500 steps", "real6_encmask_mbert_poi1_v1_st1500_f0.noctx.jsonl", "encmask_mbert_poi1_v1_st1500_f0"),
       ("encoder: Laya init, 1,500 steps", "real6_encmask_laya_poi1_v1_st1500_f0.noctx.jsonl", "encmask_laya_poi1_v1_st1500_f0"),
       ("GLiClass large, 1,500 steps", "real6_encgli_large_poi1_v1_st1500_f0.noctx.jsonl", "encgli_large_poi1_v1_st1500_f0"),
       ("3B, 200 steps", f"{CAT}_poi1_v1_none_h100bf16st200_f0_alllab_lora_poi1_v1.noctx.jsonl", None),
       ("3B, 800 steps", f"{CAT}_poi1_v1_none_h100bf16_f0_alllab_lora_poi1_v1.noctx.jsonl", None),
       ("3B, 800 steps + rename", f"{CAT}_poi1_v1_none_h100bf16_f0_ren50_alllab_lora_poi1_v1.noctx.jsonl", None)]
HEAD = ("| reader | n | top-1 [interval] | top-3 | bits left | precision at 25 / 50 / 75% coverage | AURC | position bias | "
        "ms per item, batched / one at a time |")


def ms(name):
    if not name:
        return "-"
    d = json.loads((ROOT / "results" / f"{name}.json").read_text()); c = d.get("noctx") or d.get("ctx")
    return f"{c['ms_per_item_batched']:.0f} / {c['ms_per_item_single']:.0f}"


def table(runs, items, users=None, groups=False):
    print(HEAD + (" " + " | ".join(RC.KINDS[:4]) + " |" if groups else "")); print("|---" * (9 + (4 if groups else 0)) + "|")
    for label, pat, name in runs:
        recs = {i: r for i, r in RC.load_recs(pat).items() if items[i]["user"] % 4 == 0}
        if not recs:
            print(f"| {label} | missing |"); continue
        sc = S.scorecard(recs, items, users=users, fold_of=lambda u: (u // 4) % 4)
        cells = ""
        if groups:
            cells = " " + " | ".join(f"{100 * np.mean([sc['_per_item'][i]['top1'] for i in sc['_per_item'] if RC.kind(i) == g]):.0f}" for g in RC.KINDS[:4]) + " |"
        per = sc["_per_item"]; order = sorted(per, key=lambda i: -per[i]["conf"]); ok = np.array([per[i]["top1"] for i in order], float)
        prec = " / ".join(f"{100 * ok[:max(1, int(c * len(ok)))].mean():.0f}" for c in (0.25, 0.5, 0.75))
        aurc = float(np.mean(1 - np.cumsum(ok) / np.arange(1, len(ok) + 1)))
        K = max(len(r["sum_lp"]) for r in recs.values()); pp = np.zeros(K); gp = np.zeros(K)
        for r in recs.values():
            pp[int(np.argmax(r["sum_lp"]))] += 1; gp[r["answer"]] += 1
        bias = 0.5 * np.abs(pp / pp.sum() - gp / gp.sum()).sum()
        print(f"| {label} | {sc['n']} | {sc['top1']:.1f} [{sc['top1_ci'][0]}, {sc['top1_ci'][1]}] | {sc['top3']:.1f} | {sc['bits']:.2f} | {prec} | {aurc:.3f} | {bias:.2f} | {ms(name)} |" + cells)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("**Table M.1: REAL-6, fold 0's held-out users (5 users; calibration leave-users-out within them): the encoder against the 3B; "
          "top-1 by corrected group on the right**\n")
    table(REAL6, {i["id"]: i for i in R6.load()["items"]}, groups=True)
    doc = json.loads((PROCESSED / "poi1_v1.json").read_text())
    print("\n**Table M.2: POI-1, fold 0's held-out users (50 users), readers trained on POI-1's other users**\n")
    table(POI, {i["id"]: i for i in doc["items"]}, users={u["user"]: u for u in doc["users"]})
