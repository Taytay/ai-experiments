"""Tables for PLAN step 75 (EVAL-9; REPORT 64.2 hypothesis 1: REAL-6's last points are the lookup). The user's own label for a merchant
in their history (ai_experiments.scorecard.lookup: the majority label over their history rows for that merchant) placed in front of
the best REAL-6 runs: where the lookup answers, its answer; elsewhere the model's. Read against the ceiling (ai_experiments.ceiling:
94.0 over all users) overall and by corrected group (ai_experiments.real6_cells.kind), on every fold a run has (held-out users only).

  L.1  each run alone and with the lookup in front: top-1, % of ceiling, headroom left
  L.2  the same by corrected group (% of ceiling), with the lookup's share and accuracy per group
usage: uv run python scripts/lookup_cascade_tables.py
"""
import glob
import os
import warnings

import numpy as np

from ai_experiments import ceiling as CE
from ai_experiments import real6 as R6
from ai_experiments import real6_cells as RC
from ai_experiments import scorecard as S
from ai_experiments.paths import ROOT

Q25 = "real6_categoriser_Qwen2.5-{s}-Instruct"
RUNS = [("Qwen2.5-3B, database episodes", f"{Q25.format(s='3B')}_none_h100bf16_f?_alllab_dbep50_lora.noctx.jsonl"),
        ("Qwen2.5-3B + record with category", f"{Q25.format(s='3B')}_ret_h100bf16_f?_reccat_lora.ctx.jsonl"),
        ("Qwen2.5-14B, database episodes", f"{Q25.format(s='14B')}_none_h100bf16_f?_alllab_dbep50_lora.noctx.jsonl"),
        ("Qwen2.5-14B + record with category", f"{Q25.format(s='14B')}_ret_h100bf16_f?_reccat_lora.ctx.jsonl"),
        ("Qwen3.5-2B, database episodes", "real6_categoriser_Qwen3.5-2B_none_h100bf16_hf_f?_alllab_dbep50_lora_hfs.noctx.jsonl"),
        ("Qwen3.5-4B, database episodes", "real6_categoriser_Qwen3.5-4B_none_h100bf16_hf_f?_alllab_dbep50_lora_hfs.noctx.jsonl")]


def held_out(pat, items):
    """Records pooled over the folds the run has, each fold's file read on its own held-out users (user % 4 == fold)."""
    out = {}
    for f in sorted(glob.glob(str(ROOT / "results" / "per_item" / pat))):
        k = int(os.path.basename(f).split("_f")[1][0]) if "_f" in os.path.basename(f) else None
        for i, r in RC.load_recs(os.path.basename(f)).items():
            if i in items and (k is None or items[i]["user"] % 4 == k):
                out[i] = r
    return out


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in R6.load()["items"]}
    ceil = {i: CE.real6(it) for i, it in items.items()}
    lk = {i: S.lookup(it) for i, it in items.items()}
    groups = list(RC.KINDS)
    print("**Table L.1: REAL-6, held-out users of every fold a run has: the run alone and with the user's merchant lookup in front**\n")
    print("| run | folds | n | top-1 alone | % of ceiling | lookup, then the run: top-1 | % of ceiling | headroom left (points) |")
    print("|---|---|---|---|---|---|---|---|")
    table2 = []
    for label, pat in RUNS:
        recs = held_out(pat, items)
        if not recs:
            continue
        folds = sorted({items[i]["user"] % 4 for i in recs})
        ids = sorted(recs)
        alone = {i: int(np.argmax(recs[i]["sum_lp"])) == recs[i]["answer"] for i in ids}
        casc = {i: (lk[i] == items[i]["answer"]) if lk[i] is not None else alone[i] for i in ids}
        c = [ceil[i] for i in ids]
        print(f"| {label} | {','.join(map(str, folds))} | {len(ids)} | {100 * np.mean(list(alone.values())):.1f} | {CE.share(list(alone.values()), c):.1f} | "
              f"{100 * np.mean(list(casc.values())):.1f} | {CE.share(list(casc.values()), c):.1f} | {100 * (np.mean(c) - np.mean(list(casc.values()))):.1f} |")
        table2.append((label, ids, alone, casc))
    print("\n**Table L.2: % of ceiling by corrected group, alone → with the lookup in front (the group's ceiling in brackets; the lookup's share "
          "of the group's items and its accuracy where it answers)**\n")
    print("| run | " + " | ".join(groups) + " |"); print("|---|" + "---|" * len(groups))
    all_ids = sorted({i for _, ids, _, _ in table2 for i in ids})
    print("| ceiling (items) | " + " | ".join(f"{100 * np.mean([ceil[i] for i in all_ids if RC.kind(i) == g]):.0f} ({sum(RC.kind(i) == g for i in all_ids)})" for g in groups) + " |")
    print("| lookup: share, accuracy | " + " | ".join(
        f"{100 * np.mean([lk[i] is not None for i in all_ids if RC.kind(i) == g]):.0f}%, {100 * np.mean([lk[i] == items[i]['answer'] for i in all_ids if RC.kind(i) == g and lk[i] is not None] or [np.nan]):.0f}"
        for g in groups) + " |")
    for label, ids, alone, casc in table2:
        cells = []
        for g in groups:
            gi = [i for i in ids if RC.kind(i) == g]
            cells.append(f"{CE.share([alone[i] for i in gi], [ceil[i] for i in gi]):.0f} → {CE.share([casc[i] for i in gi], [ceil[i] for i in gi]):.0f}" if gi else "-")
        print(f"| {label} | " + " | ".join(cells) + " |")
