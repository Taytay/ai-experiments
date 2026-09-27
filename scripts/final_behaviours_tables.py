"""Tables for PLAN step 91: the final recipe's trained behaviours on every user. Fold 0 from the fold-0 sets (row 89, seed 0), folds 1
to 3 from the *_all sets (build_* --all; fold 0's items are identical), each fold's model on its own users.

  B.1  per fold and pooled: misleading names in the DB; misleading names outside the DB with two of the user's filings; evidence-free
       items given p >= 0.9; the user's override (and the share pulled back to the DB category); alternation observable stores /
       restaurants; random 60 / 40 splits given p >= 0.9; control misuse of the specific category
usage: uv run python scripts/final_behaviours_tables.py
"""
import json
import warnings

import numpy as np

from ai_experiments import real6_cells as RC
from ai_experiments.paths import PROCESSED

A = "real6_dm_decider_decider_decider-4b_none_h100bf16st800_f{f}_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_aux100_labrand26_ev10soft_lora_{s}_labrand26.noctx.jsonl"


def sm(z):
    z = np.asarray(z, float); e = np.exp(z - z.max()); return e / e.sum()


def recs(s, f):
    return RC.load_recs(A.format(f=f, s=s if f == 0 else s + "_all"))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    it = {s: {i["id"]: i for i in json.loads((PROCESSED / f"{s}_all.json").read_text())["items"]} for s in ("mislead_v1", "override_v1", "alternation_v1")}
    cols = ["misleading, in DB", "misleading, 2 filings", "no evidence: p >= 0.9", "override", "override: pulled to DB",
            "alternation observable: stores / restaurants", "60 / 40: p >= 0.9", "control: specific chosen"]
    print("**Table B.1: the final recipe's trained behaviours by fold (each fold's model on its own users) and pooled**\n")
    print("| fold | " + " | ".join(cols) + " |"); print("|---|" + "---|" * len(cols))
    pool = {c: [] for c in cols}
    for f in range(4):
        row = {}
        ml = recs("mislead_v1", f); mi = it["mislead_v1"]
        ov = recs("override_v1", f); oi = it["override_v1"]
        al = recs("alternation_v1", f); ai = it["alternation_v1"]
        if not (ml and ov and al):
            continue
        sel = lambda r, items, cond: [(i, x) for i, x in r.items() if i in items and items[i]["user"] % 4 == f and cond(items[i])]  # noqa: E731
        ok = lambda x, k="answer", items=None, i=None: int(np.argmax(x["sum_lp"])) == items[i][k]  # noqa: E731
        groups = {
            cols[0]: [ok(x, items=mi, i=i) for i, x in sel(ml, mi, lambda t: t["mkind"] == "misleading" and t["in_db"] and t["cond"] == "none")],
            cols[1]: [ok(x, items=mi, i=i) for i, x in sel(ml, mi, lambda t: t["mkind"] == "misleading" and not t["in_db"] and t["cond"] == "hist")],
            cols[2]: [sm(x["sum_lp"]).max() >= 0.9 for i, x in sel(ml, mi, lambda t: t["mkind"] == "neutral" and not t["in_db"] and t["cond"] == "none")],
            cols[3]: [ok(x, items=oi, i=i) for i, x in sel(ov, oi, lambda t: t["cond"] == "override")],
            cols[4]: [ok(x, "db_idx", oi, i) for i, x in sel(ov, oi, lambda t: t["cond"] == "override")],
            "_s": [ok(x, items=ai, i=i) for i, x in sel(al, ai, lambda t: t["scen"] == "observable" and t["kind"] == "store")],
            "_r": [ok(x, items=ai, i=i) for i, x in sel(al, ai, lambda t: t["scen"] == "observable" and t["kind"] == "restaurant")],
            cols[6]: [sm(x["sum_lp"]).max() >= 0.9 for i, x in sel(al, ai, lambda t: t["scen"] == "unobservable")],
            cols[7]: [ok(x, "b_idx", ai, i) for i, x in sel(al, ai, lambda t: t["scen"] == "control")],
        }
        for c in cols:
            if c == cols[5]:
                pool.setdefault("_s", []).extend(groups["_s"]); pool.setdefault("_r", []).extend(groups["_r"])
                row[c] = f"{100 * np.mean(groups['_s']):.1f} / {100 * np.mean(groups['_r']):.1f}"
            else:
                pool[c].extend(groups[c]); row[c] = f"{100 * np.mean(groups[c]):.1f}"
        print(f"| {f} | " + " | ".join(row[c] for c in cols) + " |")
    print("| **all** | " + " | ".join((f"{100 * np.mean(pool['_s']):.1f} / {100 * np.mean(pool['_r']):.1f}" if c == cols[5] else f"{100 * np.mean(pool[c]):.1f}")
                                     if (pool.get("_s") if c == cols[5] else pool[c]) else "–" for c in cols) + " |")
