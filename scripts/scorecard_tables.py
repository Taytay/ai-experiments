"""Tables for PLAN step 63 (EVAL-9): every main REAL-6 and novel-merchant run re-read with `ai_experiments.scorecard`.

  S.1  REAL-6, held-out users (row 42's four folds): rank metrics, calibrated bits, auto-file coverage, the baseline ladder, skill
  S.2  the novel merchants (Overture, row 62), obscure renderings: the same
  S.3  the auto-file view: on items the model auto-files at 98% precision, which groups they come from (in history / category-determined / ...)
usage: uv run python scripts/scorecard_tables.py
"""
import json

from ai_experiments import real6 as R6
from ai_experiments import real6_cells as RC
from ai_experiments import scorecard as S
from ai_experiments.paths import PROCESSED

CAT = "real6_categoriser_Qwen2.5-3B-Instruct"
REAL6 = [("SFT no DB, 800 steps (3090, 4-bit)", f"{CAT}_none_st800_f?_lora.noctx.jsonl"),
         ("SFT no DB, all-label (H100, bf16)", f"{CAT}_none_h100bf16_f?_alllab_lora.noctx.jsonl"),
         ("database episodes (H100, bf16)", f"{CAT}_none_h100bf16_f?_alllab_dbep50_lora.noctx.jsonl"),
         ("record in the prompt (H100, bf16)", f"{CAT}_ret_h100bf16_f?_lora.ctx.jsonl"),
         ("record with category in the prompt (H100, bf16)", f"{CAT}_ret_h100bf16_f?_reccat_lora.ctx.jsonl"),
         ("record in the prompt + rename augmentation (3090, 4-bit)", f"{CAT}_ret_f?_ren50_lora.ctx.jsonl"),
         ("no DB, all-label + rename augmentation (3090, 4-bit)", f"{CAT}_none_f?_ren50_alllab_lora.noctx.jsonl")]
NM = [("untrained instruct, no record", "real6_Qwen2.5-3B-Instruct_novel_merchants_v1.noctx.jsonl"),
      ("untrained instruct + Overture record", "real6_Qwen2.5-3B-Instruct_novel_merchants_v1.ctx.jsonl"),
      ("SFT no DB", f"{CAT}_none_h100bf16_f?_alllab_lora_novel_merchants_v1.noctx.jsonl"),
      ("database episodes", f"{CAT}_none_h100bf16_f?_alllab_dbep50_lora_novel_merchants_v1.noctx.jsonl"),
      ("record arm + Overture record", f"{CAT}_ret_h100bf16_f?_lora_novel_merchants_v1.ctx.jsonl"),
      ("record + category arm + Overture record", f"{CAT}_ret_h100bf16_f?_reccat_lora_novel_merchants_v1.ctx.jsonl")]
HEAD = ("| run | n | top-1 [interval] | top-3 [interval] | MRR | bits left [interval] | bits gained over the usage prior | auto-file at 98%: coverage "
        "(precision) | uniform top-1 / top-3 | usage prior top-1 / top-3 | lookup → other users → prior, top-1 | skill top-1 / top-3 |")


def row(label, sc):
    return (f"| {label} | {sc['n']} | {sc['top1']:.1f} [{sc['top1_ci'][0]}, {sc['top1_ci'][1]}] | {sc['top3']:.1f} [{sc['top3_ci'][0]}, {sc['top3_ci'][1]}] | "
            f"{sc['mrr']:.2f} | {sc['bits']:.2f} [{sc['bits_ci'][0]}, {sc['bits_ci'][1]}] | {sc['info_bits_over_prior']:+.2f} | {sc['cov98']:.1f} ({sc['prec98']:.1f}) | "
            f"{sc['uniform_top1']:.1f} / {sc['uniform_top3']:.1f} | {sc['prior_top1']:.1f} / {sc['prior_top3']:.1f} | {sc['cascade_top1']:.1f} | {sc['skill_top1']:.0f} / {sc['skill_top3']:.0f} |")


def table(title, runs, items):
    print(title + "\n")
    print(HEAD); print("|" + "---|" * HEAD.count("|")[:-0] if False else "|---|---|---|---|---|---|---|---|---|---|---|---|")
    out = {}
    for label, pat in runs:
        recs = RC.load_recs(pat)
        if not recs:
            print(f"| {label} | missing |"); continue
        sc = S.scorecard(recs, items); out[label] = sc; print(row(label, sc))
    return out


if __name__ == "__main__":
    real6 = {i["id"]: i for i in R6.load()["items"]}
    nm = {i["id"]: i for i in json.loads((PROCESSED / "novel_merchants_v1.json").read_text())["items"]}
    r = table("**Table S.1: REAL-6 on held-out users, the scorecard (top-k in %; bits = -log2 p(gold) after the leave-users-out temperature; "
              "auto-file threshold chosen on other users; skill = share of the headroom over the best baseline; intervals resample users)**", REAL6, real6)
    print()
    table("**Table S.2: the novel merchants (Overture, obscure renderings), the scorecard (no merchant lookup or other-user label exists for them)**", NM, nm)
    print("\n**Table S.3: what the auto-file set (98% precision) is made of, REAL-6 held-out users (share of auto-filed items per group, and the group's share of all items)**\n")
    groups = RC.KINDS[:4]
    print("| run | " + " | ".join(groups) + " |")
    print("|---|" + "---|" * len(groups))
    for label, sc in r.items():
        per = sc["_per_item"]; auto = [i for i in per if per[i]["auto98"]]
        cells = [f"{100 * sum(RC.kind(i) == g for i in auto) / max(len(auto), 1):.0f}% of auto ({100 * sum(RC.kind(i) == g for i in per) / len(per):.0f}% of all)" for g in groups]
        print(f"| {label} | " + " | ".join(cells) + " |")
