"""Tables for PLAN step 88: the combined recipe against the readers each piece was measured on, on every test set built on 2026-09-27.
Mean over seeds [range]; fold 0.

  K.1  REAL-6 top-1 and auto-file coverage at 98%; novel names; misleading names in the DB; the user's override (all merchant kinds);
       evidence-free items given p >= 0.9; alternation observable top-1 (stores / restaurants) and p >= 0.9 on 60 / 40 splits; real
       places in no DB: top-1, and p >= 0.9 on plain names
usage: uv run python scripts/combined_tables.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments import real6 as R6
from ai_experiments import real6_cells as RC
from ai_experiments import scorecard as S
from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
import open_models_tables as O  # noqa: E402

P = "real6_dm_decider_decider_decider-4b_none_{tag}_f0_{arm}_lora_{s}{rd}.noctx.jsonl"
T3 = ("h100bf16st800", "h100bf16st800s1", "h100bf16st800s2")
READERS = [("row 81 recipe (letters)", "ren50_dbep50_aux100", "", T3),
           ("+ random labels (row 84 control)", "ren50_dbep50_aux100_labrand26", "_labrand26", T3[:2]),
           ("+ soft evidence-free (row 84)", "ren50_dbep50_aux100_labrand26_ev10soft", "_labrand26", T3[:2]),
           ("+ misleading DB, lookup / override (row 85)", "ren50_dbep50_mislead_v1_lk10_ov10_aux100_labrand26", "_labrand26", T3[:2]),
           ("+ alternation v1 (row 86)", "ren50_dbep50_alt15_aux100_labrand26", "_labrand26", T3[:2]),
           ("combined (row 88)", "ren50_dbep50_mislead_v1_alt20s_lk10_ov10_aux100_labrand26_ev10soft", "_labrand26", T3),
           ("combined, ALT 0.1 (row 89)", "ren50_dbep50_mislead_v1_alt10s_lk10_ov10_aux100_labrand26_ev10soft", "_labrand26", T3[:2]),
           ("combined, ALT 0.05 (row 89)", "ren50_dbep50_mislead_v1_alt5s_lk10_ov10_aux100_labrand26_ev10soft", "_labrand26", T3[:2])]


def sm(z):
    z = np.asarray(z, float); e = np.exp(z - z.max()); return e / e.sum()


def load(s):
    return {i["id"]: i for i in (R6.load()["items"] if s == "real6" else json.loads((PROCESSED / f"{s}.json").read_text())["items"])}


def fmt(v):
    return "–" if not v else f"{np.mean(v):.1f}" + (f" [{min(v):.1f}, {max(v):.1f}]" if len(v) > 1 else "")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sets = {s: load(s) for s in ("real6", "real6_v1_novel", "mislead_v1", "override_v1", "alternation_v1", "novel_merchants_v1_clean")}
    cols = ["REAL-6 top-1", "REAL-6 auto-filed (precision)", "novel names", "misleading, in DB", "user override", "no evidence: p >= 0.9",
            "alternation observable: stores / restaurants", "60 / 40 splits: p >= 0.9", "real places top-1", "real plain names: p >= 0.9"]
    print("**Table K.1: every 2026-09-27 test set, decider-4B readers, fold 0 (mean over seeds [range])**\n")
    print("| reader | seeds | " + " | ".join(cols) + " |"); print("|---|---|" + "---|" * len(cols))
    for lab, arm, rd, tags in READERS:
        v = {c: [] for c in cols}; n = 0
        for t in tags:
            rec = lambda s: RC.load_recs(P.format(tag=t, arm=arm, s="real6" if s == "real6" else s, rd=rd))  # noqa: E731
            r = rec("real6")
            if not r:
                continue
            n += 1; it = sets["real6"]
            f0 = {i: x for i, x in r.items() if it[i]["user"] % 4 == 0}
            sc = S.scorecard(f0, it, fold_of=O.F0)
            v[cols[0]].append(sc["top1"]); v[cols[1]].append(sc["cov98"])
            nv = rec("real6_v1_novel")
            if nv:
                v[cols[2]].append(100 * np.mean([int(np.argmax(x["sum_lp"])) == x["answer"] for i, x in nv.items() if it[i]["user"] % 4 == 0]))
            ml = rec("mislead_v1"); mi = sets["mislead_v1"]
            if ml:
                v[cols[3]].append(100 * np.mean([int(np.argmax(x["sum_lp"])) == x["answer"] for i, x in ml.items()
                                                 if mi[i]["mkind"] == "misleading" and mi[i]["in_db"] and mi[i]["cond"] == "none"]))
                v[cols[5]].append(100 * np.mean([sm(x["sum_lp"]).max() >= 0.9 for i, x in ml.items()
                                                 if mi[i]["mkind"] == "neutral" and not mi[i]["in_db"] and mi[i]["cond"] == "none"]))
            ov = rec("override_v1"); oi = sets["override_v1"]
            if ov:
                v[cols[4]].append(100 * np.mean([int(np.argmax(x["sum_lp"])) == x["answer"] for i, x in ov.items() if oi[i]["cond"] == "override"]))
            al = rec("alternation_v1"); ai = sets["alternation_v1"]
            if al:
                obs = [100 * np.mean([int(np.argmax(x["sum_lp"])) == x["answer"] for i, x in al.items() if ai[i]["scen"] == "observable" and ai[i]["kind"] == k])
                       for k in ("store", "restaurant")]
                v[cols[6]].append(np.mean(obs)); v.setdefault("_obs", []).append(obs)
                v[cols[7]].append(100 * np.mean([sm(x["sum_lp"]).max() >= 0.9 for i, x in al.items() if ai[i]["scen"] == "unobservable"]))
            nm = rec("novel_merchants_v1_clean"); ni = sets["novel_merchants_v1_clean"]
            if nm:
                ids = [i for i in nm if ni[i]["user"] % 4 == 0]
                v[cols[8]].append(100 * np.mean([int(np.argmax(nm[i]["sum_lp"])) == nm[i]["answer"] for i in ids]))
                v[cols[9]].append(100 * np.mean([sm(nm[i]["sum_lp"]).max() >= 0.9 for i in ids if ni[i]["nm_group"] == "plain"]))
        if not n:
            continue
        cells = []
        for c in cols:
            if c == cols[6] and v.get("_obs"):
                o = np.array(v["_obs"]); cells.append(f"{o[:, 0].mean():.1f} / {o[:, 1].mean():.1f}")
            else:
                cells.append(fmt(v[c]))
        print(f"| {lab} | {n} | " + " | ".join(cells) + " |")
