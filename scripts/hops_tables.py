"""Tables for PLAN row 71 (MODEL-12): hop limits on data/processed/hops_v1.json (chance 1/6; a reader that only finds chain ends
scores 25).

  H.1  accuracy by k, per reader and condition, single-token names then multi-token names; "limit" is the largest k at which the
       reader still scores at least LIMIT (default 80) and every smaller k does too
  H.2  where the wrong picks go: one hop short (the target chain's name at k - 1), further back on the target chain, another
       chain's end, or another name; by reader, pooled over k >= 2
usage: uv run python scripts/hops_tables.py
"""
import json
import os
from collections import defaultdict

import numpy as np

from ai_experiments.paths import PROCESSED, ROOT

LIMIT = float(os.environ.get("LIMIT", "80"))
PI = ROOT / "results" / "per_item"
ITEMS = {i["id"]: i for s in ("hops_v1", "hops_v1_cut", "hops_v2", "hops_v2_cut", "hops_v2_cutlast") for i in json.loads((PROCESSED / f"{s}.json").read_text())["items"]}
KS = list(range(1, 13))


def readers():
    """(label, per-item file) for every hops run present, decoders by size first."""
    out = []
    for f in sorted(PI.glob("hops_*.jsonl")):
        name, cond = f.name[len("hops_"):-len(".jsonl")].rsplit(".", 1)
        out.append((f"{name} / {cond}", f))
    order = ["0.5B", "1.5B", "3B", "7B", "14B", "huginn", "LLaDA", "Dream", "LFM", "enc_"]
    return sorted(out, key=lambda x: (next((i for i, o in enumerate(order) if o in x[0]), 99), x[0]))


def load(f):
    return [json.loads(line) for line in open(f)]


def role(it, j):
    o = it["options"][j]; c = it["chain"]; lefts = {a for a, _ in it["statements"]}
    if j == it["answer"]:
        return "gold"
    if it["k"] >= 2 and o == c[it["k"] - 1]:
        return "one short"
    if o in c:
        return "further back"
    return "other chain's end" if o not in lefts else "other name"


def limit(acc):
    k0 = 0
    for k in KS:
        if acc.get(k, 0) >= LIMIT:
            k0 = k
        else:
            break
    return k0


if __name__ == "__main__":
    rs = readers()
    for names in ("single", "multi"):
        print(f"\n**Table H.1 ({names}-token names): accuracy (%) by hops k; 50 items per cell unless n says otherwise; chance 17**\n")
        print("| reader / condition | n per k | " + " | ".join(f"k={k}" for k in KS) + f" | limit (>= {LIMIT:.0f}) |")
        print("|---" * (len(KS) + 3) + "|")
        for label, f in rs:
            by = defaultdict(list)
            for r in load(f):
                if r["names"] == names:
                    by[r["k"]].append(r["pred"] == r["answer"])
            if not by:
                continue
            acc = {k: 100 * np.mean(v) for k, v in by.items()}
            print(f"| {label} | {len(by[1])} | " + " | ".join(f"{acc[k]:.0f}" if k in acc else "-" for k in KS) + f" | {limit(acc)} |")
    print("\n**Table H.2: where the wrong picks go (k >= 2, both name types; % of wrong picks)**\n")
    roles = ["one short", "further back", "other chain's end", "other name", "no answer"]
    print("| reader / condition | wrong picks | " + " | ".join(roles) + " |"); print("|---" * (len(roles) + 2) + "|")
    for label, f in rs:
        c = defaultdict(int); n = 0
        for r in load(f):
            if r["k"] < 2 or r["pred"] == r["answer"]:
                continue
            n += 1
            c["no answer" if r["pred"] is None or r["pred"] < 0 else role(ITEMS[r["id"]], r["pred"])] += 1
        if n:
            print(f"| {label} | {n} | " + " | ".join(f"{100 * c[x] / n:.0f}" for x in roles) + " |")
