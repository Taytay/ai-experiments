"""Tables for PLAN step 42 (REAL-10): the categoriser on users whose schemes it never trained on. Four fold adapters per arm, each scored
on its five held-out users, merged into one held-out reading of all 1,179 items and paired item by item with the all-20 adapter
(sections 38 and 43), which trained on every user's history.

  R.1  by arm: the all-20 adapter, held out, held out with rename augmentation; all items with the user interval, by name type
       (standard / renamed / coined), and by REPORT.md 48's corrected groups
  R.2  paired: held out against all-20 on the same items (same prediction, net change), per name type
usage: uv run python scripts/heldout_users_tables.py            the no-DB arm at 800 steps (REPORT.md 49), the record arm at 200
"""
import json

from ai_experiments import real6_cells as RC
from ai_experiments.paths import ROOT

P = ROOT / "results" / "per_item"
CAT = "categoriser_Qwen2.5-3B-Instruct"
ST = {"none": "800", "ret": "200"}


def tag(db):
    return "" if ST[db] == "200" else f"_st{ST[db]}"
NAMES = (("standard", "standard"), ("renamed", "renamed"), ("coined", "new"))
GROUPS = RC.KINDS[:4]


def load(stem):
    p = P / f"real6_{stem}.jsonl"
    return {r["id"]: r for r in map(json.loads, open(p))} if p.exists() else {}


def folds(db, sfx, cond):
    """The four fold adapters' held-out records merged; empty unless all four exist."""
    out = {}
    for f in range(4):
        rs = load(f"{CAT}_{db}{tag(db)}_f{f}{sfx}_lora.{cond}")
        if not rs:
            return {}
        out.update(rs)
    return out


def all20(db, cond):
    """The all-20 adapter at the same step count (row 47's `_st800_s0` for no DB; section 38's for the record arm)."""
    return load(f"{CAT}_{db}{tag(db)}_s0_lora.{cond}") if tag(db) else load(f"{CAT}_{db}_lora.{cond}")


def pct(x):
    return f"{100 * sum(x) / len(x):.1f}" if x else "-"


ARMS = [(db, cond, label, rs) for db, cond in (("none", "noctx"), ("ret", "ctx"))
        for label, rs in ((f"SFT {'no DB' if db == 'none' else '+ record'}, all 20 users trained", all20(db, cond)),
                          (f"SFT {'no DB' if db == 'none' else '+ record'}, user held out", folds(db, "", cond)),
                          (f"SFT {'no DB' if db == 'none' else '+ record'}, user held out, rename augmentation", folds(db, "_ren50", cond)))]


def t1():
    print("**Table R.1: the categoriser on held-out users (no DB at 800 steps, record at 200; accuracy %; the held-out rows merge four fold adapters, each scoring the "
          "five users it never trained on; interval = users resampled; groups as REPORT.md 48)**\n")
    print("| arm | all [user interval] | " + " | ".join(n for n, _ in NAMES) + " | " + " | ".join(GROUPS) + " |")
    print("|---|---|" + "---|" * (len(NAMES) + len(GROUPS)))
    for db, cond, label, rs in ARMS:
        if not rs:
            print(f"| {label} | missing |"); continue
        c = {i: r["correct"] for i, r in rs.items()}; lo, hi = RC.user_ci(c)
        nm = [pct([r["correct"] for r in rs.values() if r["level"].endswith("_" + k)]) for _, k in NAMES]
        gr = [pct([r["correct"] for i, r in rs.items() if RC.kind(i) == g]) for g in GROUPS]
        print(f"| {label} | {pct(list(c.values()))} [{lo}, {hi}] | " + " | ".join(nm) + " | " + " | ".join(gr) + " |")


def t2():
    print("\n**Table R.2: held out against the all-20 adapter on the same items (share of the same predictions; net change in points, all / standard / renamed / coined)**\n")
    print("| arm | same predictions | net all | standard | renamed | coined |")
    print("|---|---|---|---|---|---|")
    for db, cond in (("none", "noctx"), ("ret", "ctx")):
        base = all20(db, cond)
        for sfx, what in (("", "held out"), ("_ren50", "held out, rename augmentation")):
            rs = folds(db, sfx, cond)
            if not base or not rs:
                continue
            ids = sorted(set(base) & set(rs)); same = sum(base[i]["pred"] == rs[i]["pred"] for i in ids) / len(ids)

            def net(sel):
                x = [i for i in ids if sel(base[i])]
                return f"{100 * (sum(rs[i]['correct'] for i in x) - sum(base[i]['correct'] for i in x)) / len(x):+.1f}" if x else "-"
            cells = [net(lambda r: True)] + [net(lambda r, k=k: r["level"].endswith("_" + k)) for _, k in NAMES]
            print(f"| SFT {db}, {what} | {100 * same:.1f} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    t1(); t2()
