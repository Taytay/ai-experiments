"""Tables for PLAN step 64 (REAL-20): label induction by condition, v1 (solvable by elimination) and v2 (three empty coined categories).

  L.1  top-1 by condition (sum rule), each reader on all 300 queries, and blind Opus 5.5 on its 60-query Latin square
  L.2  the same restricted to the blind reader's 60 queries, so the models and Opus compare item for item
usage: uv run python scripts/label_induction_tables.py
"""
import json

import numpy as np

from ai_experiments import real6_cells as RC
from ai_experiments.paths import PROCESSED, ROOT

CAT = "real6_categoriser_Qwen2.5-3B-Instruct"
READERS = [("Qwen2.5-3B-Instruct, untrained", "real6_Qwen2.5-3B-Instruct_{s}.noctx.jsonl"),
           ("Qwen2.5-7B-Instruct, untrained", "real6_Qwen2.5-7B-Instruct_{s}.noctx.jsonl"),
           ("Qwen2.5-14B-Instruct, untrained", "real6_Qwen2.5-14B-Instruct_{s}.noctx.jsonl"),
           ("3B SFT no DB, all-label (fold 0)", f"{CAT}_none_h100bf16_f0_alllab_lora_{{s}}.noctx.jsonl"),
           ("3B database episodes (fold 0)", f"{CAT}_none_h100bf16_f0_alllab_dbep50_lora_{{s}}.noctx.jsonl"),
           ("3B all-label + rename augmentation (fold 0)", f"{CAT}_none_h100bf16_f0_ren50_alllab_lora_{{s}}.noctx.jsonl")]


def blind(s):
    d = ROOT / "results" / "blind_opus" / s; p = d / "key_latin.json"
    if not p.exists():
        return {}, []
    key = json.loads(p.read_text()); items = {x["id"]: x for x in json.loads((PROCESSED / f"{s}.json").read_text())["items"]}; got = {}
    for f, ids in key["plan"].items():
        a = d / f"answers_{f}.txt"
        if a.exists():
            for line in a.read_text().splitlines():
                if line.strip():
                    k, ans = [x.strip() for x in line.split("|")[:2]]; i = ids[int(k.split()[1]) - 1]
                    got[i] = ans == items[i]["options"][items[i]["answer"]].strip()
    return got, key["queries"]


def table(s, only=None):
    items = {x["id"]: x for x in json.loads((PROCESSED / f"{s}.json").read_text())["items"]}
    conds = list(dict.fromkeys(x["condition"] for x in items.values()))
    bl, qs = blind(s)
    cols = []
    for label, pat in READERS:
        recs = RC.load_recs(pat.format(s=s))
        if recs:
            cols.append((label, {i: int(np.argmax(r["sum_lp"])) == r["answer"] for i, r in recs.items()}))
    if bl:
        cols.append(("blind Opus 5.5 (60 queries)", bl))
    print("| condition | " + " | ".join(c for c, _ in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for c in conds:
        cells = []
        for _, ok in cols:
            v = [ok[i] for i in ok if items[i]["condition"] == c and (only is None or items[i]["query"] in only)]
            cells.append(f"{100 * np.mean(v):.0f}" if v else "-")
        print(f"| {c} | " + " | ".join(cells) + " |")
    print(f"\n(options per item: {len(items[next(iter(items))]['options'])})")
    return qs


if __name__ == "__main__":
    for s, title in (("label_induction_v1", "v1 (12 options; solvable by elimination)"), ("label_induction_v2", "v2 (15 options: three empty coined categories)")):
        print(f"**Table L.1 {title}: top-1 % by condition, 300 queries per condition (blind Opus: its 60)**\n")
        qs = table(s)
        if qs:
            print(f"\n**Table L.2 {title}: the same on the blind reader's 60 queries only**\n")
            table(s, set(qs))
        print()
