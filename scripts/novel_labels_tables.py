"""Tables for PLAN step 61 (REAL-18): how much of a reader's accuracy comes from what its category names mean. Every REAL-6 item re-scored
with each user's category names replaced by fresh coined words (data/processed/real6_v1_novel.json; same ids, options order and gold),
fold 0's held-out users, against the same reader on the real names.

  N.1  per reader: scorecard on real names and on novel names, and the drop
  N.2  the drop split by whether the user has filed the merchant before (in history / not) and by the gold name's type in REAL-6
       (a real category name, or one of REAL-6's shared coined names)
usage: uv run python scripts/novel_labels_tables.py
"""
import sys
import warnings
from pathlib import Path

from ai_experiments import real6 as R6

sys.path.insert(0, str(Path(__file__).parent))
import open_models_tables as O  # noqa: E402

Q = "real6_categoriser_Qwen3.5-2B_none_h100bf16_hf_f0_"
D = "real6_dm_decider_decider_decider-{s}_none_h100bf16st800_f0_dbep50_aux100_lora_real6{n}.noctx.jsonl"
READERS = [("Qwen3.5-2B untrained", "real6_Qwen3.5-2B_hfs.noctx.jsonl", "real6_Qwen3.5-2B_real6_v1_novel_hfs.noctx.jsonl"),
           ("Qwen3.5-2B, per-option, database episodes (row 78)", f"{Q}alllab_dbep50_lora_hfs.noctx.jsonl", f"{Q}alllab_dbep50_lora_real6_v1_novel_hfs.noctx.jsonl"),
           ("Qwen3.5-2B, per-option, rename 0.5 (row 74)", f"{Q}ren50_alllab_lora_hfs.noctx.jsonl", f"{Q}ren50_alllab_lora_real6_v1_novel_hfs.noctx.jsonl"),
           ("decider-2B, one slot + shot labels, database episodes (row 80)", D.format(s="2b", n=""), D.format(s="2b", n="_v1_novel")),
           ("decider-4B, one slot + shot labels, database episodes (row 80)", D.format(s="4b", n=""), D.format(s="4b", n="_v1_novel"))]

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in R6.load()["items"]}
    print("**Table N.1: REAL-6 fold 0, real category names against fresh coined names (same items, options and gold)**\n")
    print("| reader | names | n | top-1 [interval] | top-3 | bits left | auto-file at 98%: coverage (precision) |"); print("|---|---|---|---|---|---|---|")
    loaded = [(lab, O.load(a), O.load(b)) for lab, a, b in READERS]
    for lab, a, b in loaded:
        for nm, recs in (("real", a), ("novel", b)):
            r = O.card_row(lab, recs, items)
            if r:
                print(r.replace(f"| {lab} |", f"| {lab} | {nm} |", 1))
    print("\n**Table N.2: top-1 on real → novel names, by whether the user has filed the merchant before and by the gold name's type**\n")
    sels = [("in history", lambda i: i["seen"]), ("not in history", lambda i: not i["seen"])]
    sels += [(f"name type: {t}", (lambda t: lambda i: i["name_type"] == t)(t)) for t in sorted({i["name_type"] for i in items.values()})]
    print("| reader | " + " | ".join(s for s, _ in sels) + " |"); print("|---|" + "---|" * len(sels))
    for lab, a, b in loaded:
        if a and b:
            print(f"| {lab} | " + " | ".join(f"{O.top1(a, items, f):.1f} → {O.top1(b, items, f):.1f}" for _, f in sels) + " |")
