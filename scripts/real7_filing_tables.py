"""Tables for REPORT.md section 97: confidence on REAL-7 by what the shots show of the query's merchant (consistent, mixed, changed but
not visible, absent), and the lone-unexpected-row breakdown. The merchant's rows are matched by several keys of its name (full-name
prefix, vowel-dropped abbreviation, first word), since statement renderings abbreviate names.

  97.1  per reader: top-1, mean top probability, share at p >= 0.9 and its precision, by filing pattern
  97.2  moved and idiosyncratic merchants whose visible rows all carry the user's category, by how many rows show it
usage: uv run python scripts/real7_filing_tables.py
"""
import collections
import json
import warnings

import numpy as np

from ai_experiments import merchants as M
from ai_experiments import real6_cells as RC
from ai_experiments.paths import PROCESSED

F = "real6_dm_decider_decider_decider-4b_none_{t}_f0_ren50_dbep50_mislead_v1{sn}_alt10s_lk10_ov10_aux100_labrand26_ev10soft_lora_real7_v1_labrand26.noctx.jsonl"
READERS = [("final recipe (row 89)", ""), ("+ majority-only misfiles (row 96)", "_sn15m")]
SEEDS = ("h100bf16st800", "h100bf16st800s1")
GROUPS = ["consistent filings", "mixed filings, the latest is the answer", "mixed filings, the latest is not the answer",
          "consistent, but the user has since changed it (not visible)", "merchant not in the shots"]


def keys(name):
    u = name.upper().replace("&", "AND"); a = M._abbr(u)
    return {u[:6], a[:5], u.split()[0][:8], u[:4] + "*", a.replace(" ", "")[:6]}


def merchant_labels(x):
    ks = keys(x["merchant"])
    return [b.split("\nCategory: ")[-1] for b in x["prompt"].split("\n\n")[1:-1] if any(k in b.upper() for k in ks)]


def group(x):
    gold = x["options"][x["answer"]].strip(); labs = merchant_labels(x)
    if not labs:
        return GROUPS[4]
    if len(set(labs)) == 1:
        return GROUPS[0] if labs[0] == gold else GROUPS[3]
    return GROUPS[1] if labs[-1] == gold else GROUPS[2]


def sm(z):
    z = np.asarray(z, float); e = np.exp(z - z.max()); return e / e.sum()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    it = {i["id"]: i for i in json.loads((PROCESSED / "real7_v1.json").read_text())["items"]}
    g = collections.defaultdict(list)
    for i, x in it.items():
        g[group(x)].append(i)
    print("**Table 97.1: REAL-7 by what the shots show of the query's merchant (mean of two seeds)**\n")
    print("| reader | merchant's rows in the shots | n | top-1 | mean top p | p >= 0.9 | right when p >= 0.9 |"); print("|---|---|---|---|---|---|---|")
    for lab, sn in READERS:
        rs = [RC.load_recs(F.format(t=t, sn=sn)) for t in SEEDS]
        for name in GROUPS:
            ids = g[name]; v = []
            for r in rs:
                p = [sm(r[i]["sum_lp"]) for i in ids]; ok = [int(np.argmax(pp)) == it[i]["answer"] for pp, i in zip(p, ids)]
                hi = [o for pp, o in zip(p, ok) if pp.max() >= 0.9]
                v.append((100 * np.mean(ok), np.mean([pp.max() for pp in p]), 100 * len(hi) / len(ids), 100 * np.mean(hi) if hi else float("nan")))
            v = np.mean(v, 0)
            print(f"| {lab} | {name} | {len(ids)} | {v[0]:.1f} | {v[1]:.2f} | {v[2]:.0f}% | {v[3]:.0f}% |")
    print("\n**Table 97.2: moved and idiosyncratic merchants whose visible rows all carry the user's category, by rows shown (top-1, two seeds)**\n")
    print("| reader | why | rows shown | n | top-1 |"); print("|---|---|---|---|---|")
    for lab, sn in READERS:
        rs = [RC.load_recs(F.format(t=t, sn=sn)) for t in SEEDS]
        c = collections.defaultdict(list)
        for i, x in it.items():
            labs = merchant_labels(x)
            if x["why"] in ("moved", "idiosyncratic") and labs and set(labs) == {x["options"][x["answer"]].strip()}:
                c[(x["why"], min(len(labs), 3))].append(np.mean([int(np.argmax(r[i]["sum_lp"])) == x["answer"] for r in rs]))
        for (why, n), v in sorted(c.items()):
            print(f"| {lab} | {why} | {'3 or more' if n == 3 else n} | {len(v)} | {100 * np.mean(v):.1f} |")
