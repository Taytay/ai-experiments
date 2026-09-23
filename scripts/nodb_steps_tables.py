"""Tables for REPORT.md section 49 (PLAN step 47; REAL-8 amended by REAL-13): the no-DB categoriser at 200, 400, 800 and 1,600 steps,
in REPORT.md 48's corrected groups, beside section 45's records-in-the-weights arms at the same history exposure, and the general
measures of exp_items_v2.

  49.1  REAL-6 by step count: in history / labelled seen, not in it / determined by category / split / DB-only / all [user interval]
  49.2  history exposure matched: no DB at 400 and 800 steps against the parametric arms at 800 and 1,600 steps at 50% (the same
        number of history sequences), so the difference is the records
  49.3  ARC-Easy, MMLU, ICL suite means by step count
  49.4  training minutes and final loss
usage: uv run python scripts/nodb_steps_tables.py            prints markdown
"""
import json

import numpy as np

from ai_experiments import real6 as R6
from ai_experiments import real6_cells as RC
from ai_experiments.paths import ROOT

R = ROOT / "results"
CAT = "categoriser_Qwen2.5-3B-Instruct"
GROUPS = RC.KINDS[:4]
DB = R6.db_only_merchants()
ITEMS = {it["id"]: it for it in R6.load()["items"]}
POINTS = [("200", [f"{CAT}_none_lora", f"{CAT}_none_s1_lora", f"{CAT}_none_s2_lora"]), ("400", [f"{CAT}_none_st400_s0_lora"]),
          ("800", [f"{CAT}_none_st800_s{s}_lora" for s in range(3)]), ("1,600", [f"{CAT}_none_st1600_s0_lora"])]
MATCHED = [("6,400 history sequences", f"no DB, 400 steps", [f"{CAT}_none_st400_s0_lora"], "records at 50%, 800 steps (6.7 passes)", [f"{CAT}_param_x4_s{s}_lora" for s in range(3)]),
           ("12,800 history sequences", f"no DB, 800 steps", [f"{CAT}_none_st800_s{s}_lora" for s in range(3)], "records at 50%, 1,600 steps (13.3 passes)", [f"{CAT}_param_x8_s{s}_lora" for s in range(3)])]


def recs(stem):
    return {r["id"]: r for r in map(json.loads, open(R / "per_item" / f"real6_{stem}.noctx.jsonl"))}


def cols(rs):
    v = [100 * np.mean([r["correct"] for i, r in rs.items() if RC.kind(i) == g]) for g in GROUPS]
    v.append(100 * np.mean([r["correct"] for i, r in rs.items() if RC.kind(i) in GROUPS[2:] and ITEMS[i]["merchant"] in DB]))
    v.append(100 * np.mean([r["correct"] for r in rs.values()]))
    return v


def fmt(stems):
    a = np.array([cols(recs(s)) for s in stems])
    if len(stems) == 1:
        return [f"{x:.1f}" for x in a[0]]
    return [f"{m:.1f} +- {s:.1f}" for m, s in zip(a.mean(0), a.std(0, ddof=1))]


def t1():
    print("**Table 49.1: the no-DB categoriser by training steps (16 sequences per step; accuracy %, mean +- sd where three seeds; groups as REPORT.md 48; "
          "DB-only = category-determined or split items whose merchant no user labelled in training; interval = users resampled, seed 0)**\n")
    print("| steps | passes over the 3,000 history episodes (150 per user) | " + " | ".join(GROUPS) + " | DB-only | all | seed 0 all [user interval] |")
    print("|---|---|" + "---|" * (len(GROUPS) + 3))
    for st, stems in POINTS:
        rs = recs(stems[0]); c = {i: r["correct"] for i, r in rs.items()}; lo, hi = RC.user_ci(c)
        passes = int(st.replace(",", "")) * 16 / 3000
        print(f"| {st} ({len(stems)} seed{'s' if len(stems) > 1 else ''}) | {passes:.1f} | " + " | ".join(fmt(stems)) + f" | {100 * np.mean(list(c.values())):.1f} [{lo}, {hi}] |")


def t2():
    print("\n**Table 49.2: the records' own contribution, at matched history exposure (accuracy %, mean +- sd over seeds)**\n")
    print("| history exposure | arm | " + " | ".join(GROUPS) + " | DB-only | all |")
    print("|---|---|" + "---|" * (len(GROUPS) + 2))
    for exp, la, sa, lb, sb in MATCHED:
        print(f"| {exp} | {la} | " + " | ".join(fmt(sa)) + " |")
        print(f"| {exp} | {lb} | " + " | ".join(fmt(sb)) + " |")


def t3():
    print("\n**Table 49.3: general measures by training steps (exp_items_v2 on the same adapters; % ; mean over seeds where three)**\n")
    keys = [("K_arc_easy", "ARC-Easy"), ("K_mmlu", "MMLU"), ("ICL_natural_mean", "ICL natural"), ("ICL_symbol_mean", "ICL symbol"),
            ("ICL2_natural_mean", "ICL v2 natural"), ("ICL2_symbol_mean", "ICL v2 symbol")]
    print("| steps | " + " | ".join(k for _, k in keys) + " |")
    print("|---|" + "---|" * len(keys))
    base = R / "items2_Qwen2.5-3B-Instruct.json"
    rows = ([("instruct base", [base])] if base.exists() else []) + [(st, [R / f"items2_{s}.json" for s in stems]) for st, stems in POINTS]
    for st, paths in rows:
        vals = []
        for p in paths:
            if not p.exists():
                continue
            d = json.load(open(p)); r = d.get("results", d)
            flat = {k: v for c in (r.values() if isinstance(next(iter(r.values())), dict) else [r]) for k, v in c.items()}
            vals.append([flat.get(k, float("nan")) for k, _ in keys])
        if vals:
            a = np.array(vals, float)
            print(f"| {st} ({len(vals)}) | " + " | ".join(f"{x:.1f}" for x in np.nanmean(a, 0)) + " |")


def t4():
    print("\n**Table 49.4: training cost (minutes, final loss; seed 0)**\n")
    print("| steps | minutes | final loss |")
    print("|---|---|---|")
    for st, name in (("200", "none"), ("400", "none_st400_s0"), ("800", "none_st800_s0"), ("1,600", "none_st1600_s0")):
        p = R / f"categoriser_llm_{name}.json"
        d = json.load(open(p)) if p.exists() else {}
        print(f"| {st} | {d.get('train_minutes', '-')} | {d.get('final_loss', '-')} |")


if __name__ == "__main__":
    t1(); t2(); t3(); t4()
