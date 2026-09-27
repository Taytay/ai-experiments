"""Tables for PLAN row 71 (MODEL-12): hop limits on the chain sets (data/processed/hops_*.json; six options, chance 17; a reader
that only finds chain ends scores 25 on v1, 33 on v2).

  H.1  hops_v2 (the headline set, symmetric options): accuracy by k, per reader and condition, single-token names then multi-token
       names; "limit" is the largest k up to which every k scores at least LIMIT (default 80)
  H.2  hops_v2_cutlast (the control: the target chain's last link removed, the answer the new end): % picking the new end / % still
       picking the removed end, by effective hops
  H.3  where the wrong picks go on v2 (k >= 2): one hop short, another chain's end, another chain's one-short name, no answer
  H.4  hops_v1 (first version, its options leak the answer to a trained reader: kept as the record for the untrained readers)
Readers are every results/per_item/hops_*.jsonl whose items belong to the set.
usage: uv run python scripts/hops_tables.py
"""
import json
import os
from collections import defaultdict

import numpy as np

from ai_experiments.paths import PROCESSED, ROOT

LIMIT = float(os.environ.get("LIMIT", "80"))
PI = ROOT / "results" / "per_item"
SETS = ("hops_v1", "hops_v1_cut", "hops_v2", "hops_v2_cut", "hops_v2_cutlast")
ITEMS = {s: {i["id"]: i for i in json.loads((PROCESSED / f"{s}.json").read_text())["items"]} for s in SETS}
KS = list(range(1, 13))
ORDER = ["0.5B", "1.5B", "3B", "7B", "14B", "huginn", "LLaDA", "Dream", "LFM", "enc_"]


def runs(s):
    """(label, records) for every hops per-item file whose items are all in set s."""
    out = []
    for f in PI.glob("hops_*.jsonl"):
        recs = [json.loads(line) for line in open(f)]
        if recs and all(r["id"] in ITEMS[s] for r in recs):
            name, cond = f.name[len("hops_"):-len(".jsonl")].rsplit(".", 1)
            out.append((f"{name} / {cond}", recs))
    return sorted(out, key=lambda x: (next((i for i, o in enumerate(ORDER) if o in x[0]), 99), x[0]))


def limit(acc):
    k0 = 0
    for k in KS:
        if acc.get(k, 0) >= LIMIT:
            k0 = k
        else:
            break
    return k0


def by_k(title, s):
    for names in ("single", "multi"):
        print(f"\n**{title} ({names}-token names): accuracy (%) by hops k; chance 17**\n")
        print("| reader / condition | n per k | " + " | ".join(f"k={k}" for k in KS) + f" | limit (>= {LIMIT:.0f}) |")
        print("|---" * (len(KS) + 3) + "|")
        for label, recs in runs(s):
            by = defaultdict(list)
            for r in recs:
                if r["names"] == names:
                    by[r["k"]].append(r["pred"] == r["answer"])
            if by:
                acc = {k: 100 * np.mean(v) for k, v in by.items()}
                print(f"| {label} | {len(by[min(by)])} | " + " | ".join(f"{acc[k]:.0f}" if k in acc else "-" for k in KS) + f" | {limit(acc)} |")


def role_v2(it, j):
    o = it["options"][j]
    if j == it["answer"]:
        return "gold"
    if o == it["chain"][-2]:
        return "one short"
    lefts = {a for a, _ in it["statements"]}
    return "another chain's end" if o not in lefts else "another chain's one-short name"


if __name__ == "__main__":
    by_k("Table H.1: hops_v2", "hops_v2")
    print("\n**Table H.2: hops_v2_cutlast (the target chain's last link removed): % picking the new end / % picking the removed end, "
          "by effective hops (both name types)**\n")
    ks = list(range(1, 12))
    print("| reader / condition | " + " | ".join(f"k={k}" for k in ks) + " |"); print("|---" * (len(ks) + 1) + "|")
    for label, recs in runs("hops_v2_cutlast"):
        by = defaultdict(lambda: [0, 0, 0])
        for r in recs:
            it = ITEMS["hops_v2_cutlast"][r["id"]]; b = by[it["k"]]
            b[0] += r["pred"] == r["answer"]; b[1] += r["pred"] == it["old_answer"]; b[2] += 1
        print(f"| {label} | " + " | ".join(f"{100 * by[k][0] / by[k][2]:.0f} / {100 * by[k][1] / by[k][2]:.0f}" if by[k][2] else "-" for k in ks) + " |")
    print("\n**Table H.3: hops_v2, where the wrong picks go (k >= 2, both name types; % of wrong picks)**\n")
    roles = ["one short", "another chain's end", "another chain's one-short name", "no answer"]
    print("| reader / condition | wrong picks | " + " | ".join(roles) + " |"); print("|---" * (len(roles) + 2) + "|")
    for label, recs in runs("hops_v2"):
        c = defaultdict(int); n = 0
        for r in recs:
            if r["k"] < 2 or r["pred"] == r["answer"]:
                continue
            n += 1
            c["no answer" if r["pred"] is None or r["pred"] < 0 else role_v2(ITEMS["hops_v2"][r["id"]], r["pred"])] += 1
        if n:
            print(f"| {label} | {n} | " + " | ".join(f"{100 * c[x] / n:.0f}" for x in roles) + " |")
    by_k("Table H.4: hops_v1 (options leak the answer to trained readers; the untrained readers' record)", "hops_v1")
