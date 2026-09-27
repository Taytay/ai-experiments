"""Tables for PLAN step 72 (POI-1 seen kinds as retrieval; REPORT 64.2 hypothesis 2): what the prompt says about each place's kind.

Layouts (scripts/build_poi1_variants.py): plain (24 frozen shots), + record (a "Note: <place> is listed as a <kind>." line before the
query), kshots (up to six history places of the query's kind among the 24), kinds (a "Kind: <kind>" line under every shot and the query;
the hop study's one-hop layout), and their combinations. Trained arms: the row 65 recipe (3B, fold 0 held out, 800 steps, all-label
loss, rename augmentation 0.5) with the training episodes in the layout (exp_categoriser.py POI_SHOTS=kind / POI_REC / POI_KIND).

  K.1  the scorecard on fold 0's held-out users, every reader x layout
  K.2  top-1 by seen / unseen kind x category name type
  K.3  seen-kind items by whether the prompt holds a place of the query's kind
usage: uv run python scripts/poi1_kinds_tables.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments import real6_cells as RC
from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
import poi1_tables as PT  # noqa: E402

CAT = "real6_categoriser_Qwen2.5-3B-Instruct_poi1_v1_none_h100bf16"
LAYOUT = {"plain": ("poi1_v1", "noctx"), "record": ("poi1_v1", "ctx"), "kshots": ("poi1_v1_kshots", "noctx"),
          "kshots + record": ("poi1_v1_kshots", "ctx"), "kinds": ("poi1_v1_kinds", "noctx"), "kshots + kinds": ("poi1_v1_kshots_kinds", "noctx")}
READERS = [("Qwen2.5-3B-Instruct, untrained", "real6_Qwen2.5-3B-Instruct", ["plain", "record", "kshots", "kshots + record", "kinds", "kshots + kinds"]),
           ("Qwen2.5-14B-Instruct, untrained", "real6_Qwen2.5-14B-Instruct", ["plain", "record", "kshots", "kshots + record", "kinds", "kshots + kinds"]),
           ("3B trained plain (row 65)", f"{CAT}_f0_ren50_alllab_lora", ["plain", "record", "kshots", "kshots + record", "kinds", "kshots + kinds"]),
           ("3B trained with kshots", f"{CAT}_pksh_f0_ren50_alllab_lora", ["plain", "kshots"]),
           ("3B trained with the record", f"{CAT}_prec_f0_ren50_alllab_lora", ["record", "kshots + record"]),
           ("3B trained with kshots + record", f"{CAT}_pksh_prec_f0_ren50_alllab_lora", ["record", "kshots + record"]),
           ("3B trained with kinds", f"{CAT}_pkind_f0_ren50_alllab_lora", ["kinds", "kshots + kinds"]),
           ("3B trained with kshots + kinds", f"{CAT}_pksh_pkind_f0_ren50_alllab_lora", ["kinds", "kshots + kinds"])]


def recs_of(prefix, layout, items):
    s, c = LAYOUT[layout]
    return {i: r for i, r in RC.load_recs(f"{prefix}_{s}.{c}.jsonl").items() if items[i]["user"] % 4 == 0}


def ok(r):
    return int(np.argmax(r["sum_lp"])) == r["answer"]


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items, users = PT.load()
    print("**Table K.1: POI-1 fold 0's held-out users, the scorecard by reader and prompt layout (columns as Table 61.1; seen-kind items "
          "have a ceiling of 100, unseen-kind items a bracket)**\n")
    print(PT.HEAD.replace("| reader |", "| reader | layout |").replace(" |\n", " |") + " seen-kind top-1 | unseen-kind top-1 |")
    print("|" + "---|" * 15)
    for label, prefix, lays in READERS:
        for lay in lays:
            recs = recs_of(prefix, lay, items)
            if not recs:
                continue
            sc = PT.card(recs, items, users, True)
            seen = 100 * np.mean([ok(r) for i, r in recs.items() if items[i]["seen"]])
            unseen = 100 * np.mean([ok(r) for i, r in recs.items() if not items[i]["seen"]])
            print(PT.row(label, sc).replace(f"| {label} |", f"| {label} | {lay} |", 1) + f" {seen:.1f} | {unseen:.1f} |")
    nts = ("standard", "renamed", "new")
    print("\n**Table K.2: top-1 % by seen / unseen kind x category name type (readable / merged / coined), fold 0**\n")
    cells = [(s, nt) for s in (True, False) for nt in nts]
    print("| reader | layout | " + " | ".join(f"{'seen' if s else 'unseen'}, {nt} (n={sum(items[i]['seen'] == s and items[i]['name_type'] == nt and items[i]['user'] % 4 == 0 for i in items)})" for s, nt in cells) + " |")
    print("|---|---|" + "---|" * len(cells))
    for label, prefix, lays in READERS:
        for lay in lays:
            recs = recs_of(prefix, lay, items)
            if recs:
                print(f"| {label} | {lay} | " + " | ".join(f"{100 * np.mean([ok(r) for i, r in recs.items() if items[i]['seen'] == s and items[i]['name_type'] == nt]):.0f}" for s, nt in cells) + " |")
    print("\n**Table K.3: seen-kind items, top-1 % by whether the prompt holds a place of the query's kind (frozen shots: by chance; "
          "kshots layouts: whenever the history has one, i.e. always for seen kinds), fold 0**\n")
    kk = {i["id"]: i for i in json.loads((PROCESSED / "poi1_v1_kinds.json").read_text())["items"]}
    has = {i: kk[i]["n_kind_shots"] > 0 for i in kk}
    n1 = sum(has[i] and items[i]["seen"] and items[i]["user"] % 4 == 0 for i in items); n0 = sum(not has[i] and items[i]["seen"] and items[i]["user"] % 4 == 0 for i in items)
    print(f"| reader | layout | frozen shots hold the kind (n={n1}) | frozen shots do not (n={n0}) |"); print("|---|---|---|---|")
    for label, prefix, lays in READERS:
        for lay in lays:
            recs = recs_of(prefix, lay, items)
            if recs:
                a = [ok(r) for i, r in recs.items() if items[i]["seen"] and has[i]]; b = [ok(r) for i, r in recs.items() if items[i]["seen"] and not has[i]]
                print(f"| {label} | {lay} | {100 * np.mean(a):.0f} | {100 * np.mean(b):.0f} |")
