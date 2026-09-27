"""Results as a share of the maximum achievable score (the owner, 2026-09-26; `ai_experiments.ceiling` defines the per-item ceilings).

  C.1  REAL-6: the ceiling by corrected group and overall (all users; fold 0), and the leading runs as top-1 and % of ceiling
  C.2  POI-1 (estimated ceiling), fold 0: the same
  C.3  label induction v1 / v2: the ceiling per condition and the readers as % of it
usage: uv run python scripts/ceiling_tables.py
"""
import json
import warnings

import numpy as np

from ai_experiments import ceiling as CE
from ai_experiments import real6 as R6
from ai_experiments import real6_cells as RC
from ai_experiments.paths import PROCESSED

CAT = "real6_categoriser_Qwen2.5-{s}-Instruct"
REAL6 = [("3B SFT no DB, all-label", f"{CAT.format(s='3B')}_none_h100bf16_f0_alllab_lora.noctx.jsonl"),
         ("3B database episodes", f"{CAT.format(s='3B')}_none_h100bf16_f0_alllab_dbep50_lora.noctx.jsonl"),
         ("7B database episodes", f"{CAT.format(s='7B')}_none_h100bf16_f0_alllab_dbep50_lora.noctx.jsonl"),
         ("14B database episodes", f"{CAT.format(s='14B')}_none_h100bf16_f0_alllab_dbep50_lora.noctx.jsonl"),
         ("3B + record with category", f"{CAT.format(s='3B')}_ret_h100bf16_f0_reccat_lora.ctx.jsonl"),
         ("7B + record with category", f"{CAT.format(s='7B')}_ret_h100bf16_f0_reccat_lora.ctx.jsonl"),
         ("14B + record with category", f"{CAT.format(s='14B')}_ret_h100bf16_f0_reccat_lora.ctx.jsonl"),
         ("encoder (Laya layout), no record", "real6_encmask_laya_st1500_f0.noctx.jsonl"),
         ("GLiClass-large + record", "real6_encgli_large_st1500_f0_ctx.ctx.jsonl"),
         ("encoder (Laya layout) + record", "real6_encmask_laya_st1500_f0_ctx.ctx.jsonl")]
POI = [("untrained 14B", "real6_Qwen2.5-14B-Instruct_poi1_v1.noctx.jsonl"),
       ("3B, all-label + rename", f"{CAT.format(s='3B')}_poi1_v1_none_h100bf16_f0_ren50_alllab_lora_poi1_v1.noctx.jsonl"),
       ("7B, all-label + rename", f"{CAT.format(s='7B')}_poi1_v1_none_h100bf16_f0_ren50_alllab_lora_poi1_v1.noctx.jsonl"),
       ("14B, all-label + rename", f"{CAT.format(s='14B')}_poi1_v1_none_h100bf16_f0_ren50_alllab_lora_poi1_v1.noctx.jsonl"),
       ("GLiClass-large", "real6_encgli_large_poi1_v1_st1500_f0.noctx.jsonl"),
       ("3B + kind-retrieved shots", f"{CAT.format(s='3B')}_poi1_v1_none_h100bf16_f0_ren50_alllab_lora_poi1_v1_kshots.noctx.jsonl")]
LI = [("3B", "real6_Qwen2.5-3B-Instruct_{s}.noctx.jsonl"), ("14B", "real6_Qwen2.5-14B-Instruct_{s}.noctx.jsonl"),
      ("3B rename-trained", f"{CAT.format(s='3B')}_none_h100bf16_f0_ren50_alllab_lora_{{s}}.noctx.jsonl")]


def ok(r):
    return int(np.argmax(r["sum_lp"])) == r["answer"]


def runs(title, rows, items, ceil, groups=None):
    print(title + "\n")
    gl = list(groups) if groups else []
    print("| run | n | top-1 | ceiling on the same items | % of ceiling | headroom left (points) |" + "".join(f" {g}: % of ceiling |" for g in gl))
    print("|---" * (6 + len(gl)) + "|")
    for label, pat in rows:
        recs = {i: r for i, r in RC.load_recs(pat).items() if i in items and items[i]["user"] % 4 == 0}
        if not recs:
            continue
        c = [ok(r) for r in recs.values()]; ce = [ceil[i] for i in recs]
        cells = ""
        for g in gl:
            ids = [i for i in recs if groups[g](i)]
            cells += f" {CE.share([ok(recs[i]) for i in ids], [ceil[i] for i in ids]):.0f} |" if ids else " - |"
        print(f"| {label} | {len(c)} | {100 * np.mean(c):.1f} | {100 * np.mean(ce):.1f} | {CE.share(c, ce):.1f} | {100 * (np.mean(ce) - np.mean(c)):.1f} |" + cells)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in R6.load()["items"]}; ceil = {i: CE.real6(it) for i, it in items.items()}
    print("**Table C.0: REAL-6's ceiling by corrected group (an ideal reader: the merchant's true category and the user's exact scheme, "
          "not the coin flip of a split)**\n")
    print("| group | items (all users) | ceiling, all users | items (fold 0) | ceiling, fold 0 |"); print("|---|---|---|---|---|")
    for g in list(RC.KINDS) + ["all"]:
        ids = [i for i in items if g == "all" or RC.kind(i) == g]; f0 = [i for i in ids if items[i]["user"] % 4 == 0]
        print(f"| {g} | {len(ids)} | {100 * np.mean([ceil[i] for i in ids]):.1f} | {len(f0)} | {100 * np.mean([ceil[i] for i in f0]) if f0 else float('nan'):.1f} |")
    print()
    runs("**Table C.1: REAL-6, fold 0's held-out users: top-1 as a share of the ceiling on the same items**", REAL6, items, ceil,
         {k: (lambda i, k=k: RC.kind(i) == k) for k in RC.KINDS[:4]})
    doc = json.loads((PROCESSED / "poi1_v1.json").read_text()); users = {u["user"]: u for u in doc["users"]}
    pitems = {i["id"]: i for i in doc["items"]}; pceil = {i: CE.poi1(it, users) for i, it in pitems.items()}
    print("\n**Table C.2: POI-1, fold 0. Seen kinds (the place's Overture basic category is in the user's history) have an exact ceiling of "
          "100 (a perfect places database plus the history); unseen kinds have none, so they are read as a bracket: the best reader below, 100 above**\n")
    seen = [i for i in pitems if pceil[i] is not None]
    print(f"(fold 0: {sum(pitems[i]['user'] % 4 == 0 for i in seen)} seen-kind items, {sum(pitems[i]['user'] % 4 == 0 and pceil[i] is None for i in pitems)} unseen-kind; blind Opus 5.5 read 56 on its sample)\n")
    print("| run | seen kind: top-1 = % of ceiling | seen kind, coined name | unseen kind: top-1 | all items: top-1 |"); print("|---|---|---|---|---|")
    best_unseen = 0
    for label, pat in POI:
        recs = {i: r for i, r in RC.load_recs(pat).items() if pitems[i]["user"] % 4 == 0}
        if not recs:
            continue
        sv = [ok(recs[i]) for i in recs if pceil[i] is not None]; sc = [ok(recs[i]) for i in recs if pceil[i] is not None and pitems[i]["name_type"] == "new"]
        un = [ok(recs[i]) for i in recs if pceil[i] is None]; best_unseen = max(best_unseen, 100 * np.mean(un))
        print(f"| {label} | {100 * np.mean(sv):.1f} | {100 * np.mean(sc):.1f} | {100 * np.mean(un):.1f} | {100 * np.mean([ok(r) for r in recs.values()]):.1f} |")
    print(f"\n(unseen kinds: the ceiling lies between {best_unseen:.1f}, the best reader here, and 100)")
    for s in ("label_induction_v1", "label_induction_v2"):
        li = {i["id"]: i for i in json.loads((PROCESSED / f"{s}.json").read_text())["items"]}; lc = {i: CE.label_induction(it) for i, it in li.items()}
        conds = list(dict.fromkeys(x["condition"] for x in li.values()))
        print(f"\n**Table C.3 {s}: ceiling and % of ceiling by condition**\n")
        recs = {lab: RC.load_recs(p.format(s=s)) for lab, p in LI}
        print("| condition | ceiling | " + " | ".join(f"{lab}: top-1 / % of ceiling" for lab, _ in LI) + " |"); print("|---" * (2 + len(LI)) + "|")
        for c in conds:
            ids = [i for i in li if li[i]["condition"] == c]
            cells = [f"{100 * np.mean([ok(r[i]) for i in ids]):.0f} / {CE.share([ok(r[i]) for i in ids], [lc[i] for i in ids]):.0f}" if r else "-" for r in recs.values()]
            print(f"| {c} | {100 * np.mean([lc[i] for i in ids]):.0f} | " + " | ".join(cells) + " |")
