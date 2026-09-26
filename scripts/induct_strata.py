"""EVAL-4 stratification (PLAN step 20): the v1 induction items (`ladder_v1.json`, one demo per group) split by which attribute
rules agree with them, scored from the saved per-item files, so no GPU is needed.

For each L3/L4 item the generating attribute a labels the query like the demo that shares a with it. Another attribute b is a
competing rule when the demos differ on b and one demo shares b with the query: b gives the gold answer when that demo is the
gold demo ("b-consistent"), another answer when it is not ("b-conflicting"); b is silent when no demo shares b with the query.
Weakness is a bijection of type, so type and weakness are one rule. Accuracy per stratum tells whether an arm that scores well
on habitat items does so by grouping by habitat or by type.

usage: uv run python scripts/induct_strata.py [result_name ...]     default: the arms of Table 33.2
"""
import json
import sys
from collections import defaultdict

from ai_experiments import universe as U
from ai_experiments.paths import ROOT
from ai_experiments.scoring import read_records

R = ROOT / "results" / "per_item"
DEFAULT = [("base", "curriculum_Qwen2.5-3B_base.base"), ("A", "curriculum_Qwen2.5-3B_A_p200.trained"), ("C", "curriculum_Qwen2.5-3B_C_p200.trained"),
           ("D", "curriculum_Qwen2.5-3B_D_p200.trained"), ("C 1,600 steps", "curriculum_Qwen2.5-3B_C_s1600_p400.trained"),
           ("C + OPD + replay", "curriculum_Qwen2.5-3B_C_opd_fineweb_full_replayC.trained")]
LEVELS = {"L3_induct_type_nonsense": "type", "L4_induct_habitat": "habitat", "L4_induct_weakness": "weakness"}
ALT = {"type": "habitat", "habitat": "type", "weakness": "habitat"}  # the competing rule reported per level


def strata():
    species = U.build(n_per_type=20)
    by = {s["name"]: s for s in species}
    lad = json.load(open(ROOT / "data" / "processed" / "ladder_v1.json"))["items"]
    out = {}
    for it in lad:
        if it["level"] not in LEVELS:
            continue
        ag = U.rule_agreement(it, by)
        b = ALT[LEVELS[it["level"]]]
        if ag[b] is None:
            st = f"{b}-silent"
        elif ag[b] == it["answer"]:
            st = f"{b}-consistent"
        else:
            st = f"{b}-conflicting"
        out[it["id"]] = st
    return out


def main(argv):
    cols = [(n, n) for n in argv] if argv else DEFAULT
    st = strata()
    counts = defaultdict(lambda: defaultdict(int))
    for i, s in st.items():
        counts[i.split(":")[0]][s] += 1
    print("**Table 33.2: v1 induction items by the competing rule (type items against the habitat rule, habitat and weakness items against the type rule): "
          "share of items per stratum and each arm's accuracy on it**\n")
    print("| level | stratum | items | " + " | ".join(c[0] for c in cols) + " |")
    print("|---|---|---|" + "---|" * len(cols))
    recs = {}
    for label, name in cols:
        p = R / f"{name}.jsonl"
        recs[label] = {r["id"]: r["correct"] for r in read_records(p)} if p.exists() else {}
    for level in LEVELS:
        for s in sorted(counts[level]):
            ids = [i for i, x in st.items() if i.startswith(level + ":") and x == s]
            cells = []
            for label, _ in cols:
                rr = recs[label]
                hit = [rr[i] for i in ids if i in rr]
                cells.append(f"{100 * sum(hit) / len(hit):.1f}" if hit else "-")
            print(f"| {level} | {s} | {len(ids)} | " + " | ".join(cells) + " |")
        ids = [i for i in st if i.startswith(level + ":")]
        cells = []
        for label, _ in cols:
            rr = recs[label]; hit = [rr[i] for i in ids if i in rr]
            cells.append(f"{100 * sum(hit) / len(hit):.1f}" if hit else "-")
        print(f"| {level} | all | {len(ids)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
