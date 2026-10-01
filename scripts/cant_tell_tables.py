"""Tables for PLAN step 84 (REAL-14): abstention when nothing can tell. decider-4B, row 81's recipe with random A..Z labels, two seeds per
arm: ctl (no abstention training), opt (evidence-free episodes answered by "cannot tell from this", plus decider's "not listed here"
augmentation), soft (evidence-free episodes with a uniform target, no option). Read with no extra options and with both offered
(exp_decision_models.py EXTRA_OPTS); evidence-free items are mislead_v1's neutral made-up merchants in no DB (REPORT 80, 83), their
evidence counterpart the same merchants with two of the user's filings shown.

  C.1  per arm (mean over seeds, range in brackets): REAL-6 top-1 without options; with both options offered, REAL-6 top-1 and the
       share of REAL-6 answers that abstain; on evidence-free items the share answered "cannot tell" and the share still given
       p >= 0.9 on a real category (no options); abstentions on the evidence counterpart; "not listed here" recall on questions
       whose category is hidden (real6_v1_abst); abstentions where the name carries a clue and should not (owner, 2026-09-27: only
       extremely ambiguous merchants): misleading names in no DB, and novel_merchants_v1_clean's real places by name group
usage: uv run python scripts/cant_tell_tables.py
"""
import json
import warnings

import numpy as np

from ai_experiments import real6 as R6
from ai_experiments import real6_cells as RC
from ai_experiments.paths import PROCESSED

ARMS = [("no abstention training", ""), ("\"cannot tell\" + \"not listed here\" options", "_ev10opt_abst10sw25"), ("soft uniform targets, no option", "_ev10soft")]
SEEDS = ["h100bf16st800", "h100bf16st800s1"]


def f(tag, sfx, s, xo):
    return f"real6_dm_decider_decider_decider-4b_none_{tag}_f0_ren50_dbep50_aux100_labrand26{sfx}_lora_{s}_labrand26{xo}.noctx.jsonl"


def sm(z):
    z = np.asarray(z, float); e = np.exp(z - z.max()); return e / e.sum()


def pct(v):
    return 100 * float(np.mean(v)) if len(v) else float("nan")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    r6 = {i["id"]: i for i in R6.load()["items"]}
    ml = {i["id"]: i for i in json.loads((PROCESSED / "mislead_v1.json").read_text())["items"]}
    ab = {i["id"]: i for i in json.loads((PROCESSED / "real6_v1_abst.json").read_text())["items"]}
    nm = {i["id"]: i for i in json.loads((PROCESSED / "novel_merchants_v1_clean.json").read_text())["items"]}
    ev = lambda i: i["mkind"] == "neutral" and not i["in_db"] and i["cond"] == "none"  # noqa: E731
    evh = lambda i: i["mkind"] == "neutral" and not i["in_db"] and i["cond"] == "hist"  # noqa: E731
    cols = ["REAL-6 top-1, no options", "REAL-6 top-1, options offered", "REAL-6 answers abstaining", "evidence-free: \"cannot tell\"",
            "evidence-free: p >= 0.9 on a real category, no options", "same merchants with 2 filings: abstaining", "hidden category: \"not listed here\"",
            "misleading name, not in DB: abstaining", "real place, descriptive name: abstaining", "real place, chain: abstaining",
            "real place, plain name: abstaining", "real places: top-1 with options offered"]
    print("**Table C.1: decider-4B, abstention when nothing can tell, fold 0 (mean over two seeds [range])**\n")
    print("| arm | " + " | ".join(cols) + " |"); print("|---|" + "---|" * len(cols))
    for lab, sfx in ARMS:
        vals = {c: [] for c in cols}
        for tag in SEEDS:
            r = RC.load_recs(f(tag, sfx, "real6", ""))
            ro = RC.load_recs(f(tag, sfx, "real6", "_xo2"))
            mo = RC.load_recs(f(tag, sfx, "mislead_v1", "_xo2"))
            mp = RC.load_recs(f(tag, sfx, "mislead_v1", ""))
            ao = RC.load_recs(f(tag, sfx, "real6_v1_abst", "_xo1"))
            no = RC.load_recs(f(tag, sfx, "novel_merchants_v1_clean", "_xo2"))
            if not r:
                continue
            f0 = [i for i in r if r6[i]["user"] % 4 == 0]
            vals[cols[0]].append(pct([int(np.argmax(r[i]["sum_lp"])) == r[i]["answer"] for i in f0]))
            if ro:
                n = {i: len(r6[i]["options"]) for i in ro}
                vals[cols[1]].append(pct([int(np.argmax(ro[i]["sum_lp"])) == ro[i]["answer"] for i in f0 if i in ro]))
                vals[cols[2]].append(pct([int(np.argmax(ro[i]["sum_lp"])) >= n[i] for i in f0 if i in ro]))
            if mo:  # options: real ..., "cannot tell from this", "not listed here"
                n = {i: len(ml[i]["options"]) for i in mo}
                vals[cols[3]].append(pct([int(np.argmax(mo[i]["sum_lp"])) == n[i] for i in mo if ev(ml[i])]))
                vals[cols[5]].append(pct([int(np.argmax(mo[i]["sum_lp"])) >= n[i] for i in mo if evh(ml[i])]))
                vals[cols[7]].append(pct([int(np.argmax(mo[i]["sum_lp"])) >= n[i] for i in mo
                                          if ml[i]["mkind"] == "misleading" and not ml[i]["in_db"] and ml[i]["cond"] == "none"]))
            if no:  # real Overture places in no DB (REPORT 57): names with clues should not abstain
                n = {i: len(nm[i]["options"]) for i in no if i in nm}
                ids = [i for i in n if nm[i]["user"] % 4 == 0]
                for c, g in ((8, "descriptive"), (9, "chain"), (10, "plain")):
                    vals[cols[c]].append(pct([int(np.argmax(no[i]["sum_lp"])) >= n[i] for i in ids if nm[i]["nm_group"] == g]))
                vals[cols[11]].append(pct([int(np.argmax(no[i]["sum_lp"])) == no[i]["answer"] for i in ids]))
            if mp:
                vals[cols[4]].append(pct([sm(mp[i]["sum_lp"]).max() >= 0.9 for i in mp if ev(ml[i])]))
            if ao:  # real6_v1_abst options: real ..., "not listed here", then "cannot tell from this"
                nog = [i for i in ao if ab[i]["abst_case"] == "nogold"]
                vals[cols[6]].append(pct([int(np.argmax(ao[i]["sum_lp"])) == ab[i]["answer"] for i in nog]))
        cells = [(f"{np.mean(v):.1f} [{min(v):.1f}, {max(v):.1f}]" if len(v) > 1 else f"{v[0]:.1f}") if v else "–" for v in vals.values()]
        print(f"| {lab} | " + " | ".join(cells) + " |")
