"""Tables for PLAN steps 151 and 152 (MODEL-22, owner 2026-10-01): strands-decider (StrandsAgents/strands-decider-2B-hobson-v19, a LoRA on
Qwen3.5-2B-Base plus a pointer head) against this repo's readers.
  S.1  top-1 per item set (mean [range] over seeds): REAL-6 v1 (fold 0 users), novel names, mislead, override, blind_v1 / v2 without and
       with the other-users line. Readers: untrained (strands v19, decider-4B, decider-35B, kev-4B), and trained on the categoriser's
       recipe episodes (the other-users line, OTHERS=0.5): strands from v19, strands' architecture from Qwen3.5-2B-Base, decider-2B and
       decider-4B (decider's one-slot readout, the recipe of REPORT 123)
  S.2  blind_v1 / v2 with the other-users line, the owner's measures: lists of at most 3 (confirm-everything card; plausible categories
       free, clutter 1 per other wrong suggestion, not suggested 10), first-time payees (right 1st / in top 3), payees seen before, new
       users, effort under the per-item rule at W = 10. Probabilities calibrated with a leave-fold-out temperature (decision_tables).
usage: uv run python scripts/strands_tables.py
"""
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
from confirm_card import card, lists  # noqa: E402
from decision_tables import evaluate, fmt, folds_calibrate, load  # noqa: E402
from report_card import plausible_sets  # noqa: E402

R = "results/per_item/real6_dm_"
EP = "none_h100bf16st800{s}_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_oth50"
DEC = "decider_decider_decider-{size}_" + EP + "_aux100_labrand255_laylabelled_shots_ev10soft_lora_{set}_labrand255_laylabelled_shots.noctx.jsonl"
READERS = {  # name: file pattern ({set} is the item set's name, "real6" for REAL-6 v1)
    "strands v19, untrained": "strands_strands-decider-2B-hobson-v19_{set}.noctx.jsonl",
    "decider-4B, untrained": "decider_decider-4b_{set}.noctx.jsonl",
    "decider-35B-A3B, untrained": "decider_decider-35b-a3b_{set}.noctx.jsonl",
    "kev-4B, untrained": "kev_kev-4b_{set}.noctx.jsonl",
    "strands v19 + recipe episodes": "strands_strands_v19_" + EP.replace("{s}", "*") + "_ev10soft_{set}.noctx.jsonl",
    "strands arch. from Qwen3.5-2B-Base + recipe episodes": "strands_strands_base_" + EP.replace("{s}", "*") + "_ev10soft_{set}.noctx.jsonl",
    "strands' own trainer and v19 config + recipe episodes": "strands_strands_recipe_" + EP.replace("{s}", "*") + "_ev10soft_{set}.noctx.jsonl",
    "decider-2B + recipe (one slot)":DEC.replace("{size}", "2b").replace("{s}", "*"),
    "decider-4B + recipe (one slot; REPORT 123)": DEC.replace("{size}", "4b").replace("{s}", "*"),
    # row 152 (MODEL-23)
    "strands from base, our labelled layout": "strands_strands_base_" + EP.replace("{s}", "*") + "_ev10soft_lab_{set}.noctx.jsonl",
    "strands from base, labelled layout + shot-code loss": "strands_strands_base_" + EP.replace("{s}", "*") + "_ev10soft_lab_aux100_{set}.noctx.jsonl",
    "decider-2B + recipe, options relisted (control)": DEC.replace("{size}", "2b").replace("{s}", "*").replace("ev10soft_lora", "ev10soft_relist_lora"),
    "decider-2B + recipe + pointer": DEC.replace("{size}", "2b").replace("{s}", "*").replace("ev10soft_lora", "ev10soft_relist_ptr_lora"),
    "decider-4B + recipe, options relisted (control)": DEC.replace("{size}", "4b").replace("{s}", "*").replace("ev10soft_lora", "ev10soft_relist_lora"),
    "decider-4B + recipe + pointer": DEC.replace("{size}", "4b").replace("{s}", "*").replace("ev10soft_lora", "ev10soft_relist_ptr_lora"),
}
SETS = [("REAL-6 v1", "real6", None), ("novel names", "real6_v1_novel", "real6_v1_novel"), ("mislead", "mislead_v1", "mislead_v1"),
        ("override", "override_v1", "override_v1"), ("blind_v1", "blind_v1", "blind_v1"), ("blind_v1 + line", "blind_v1_others", "blind_v1_others"),
        ("blind_v2", "blind_v2", "blind_v2"), ("blind_v2 + line", "blind_v2_others", "blind_v2_others")]


def recs(name, set_name):
    return [load(f) for f in sorted(glob.glob(R + READERS[name].format(set=set_name)))]


def items_of(path):
    if path is None:
        from ai_experiments import real6 as R6
        return {i["id"]: i for i in R6.load("v1")["items"]}
    return {i["id"]: i for i in json.loads((PROCESSED / f"{path}.json").read_text())["items"]}


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("**S.1 top-1 % (mean [range] over seeds; seeds in brackets)**\n")
    print("| reader | " + " | ".join(s[0] for s in SETS) + " |"); print("|---|" + "---|" * len(SETS))
    its = {s[1]: items_of(s[2]) for s in SETS}
    for n in READERS:
        cells = []
        for _, sn, _ in SETS:
            rs = recs(n, sn); it = its[sn]
            cells.append(f"{fmt([100 * np.mean([int(np.argmax(x)) == it[i]['answer'] for i, x in r.items() if i in it]) for r in rs])} ({len(rs)})" if rs else "–")
        print(f"| {n} | " + " | ".join(cells) + " |")
    for v in (1, 2):
        items = its[f"blind_v{v}_others"]
        rule = json.loads((PROCESSED / f"blind_v{v}_ynabrule.json").read_text()); ph = json.loads((PROCESSED / f"blind_v{v}_payeehist.json").read_text())
        plaus = plausible_sets(items)
        for i, r in rule.items():
            if r >= 0:
                plaus[i].add(r)
        first = [i for i in items if not ph[i]]; seen = [i for i in items if ph[i]]
        newu = [i for i in items if items[i]["why"] in ("short_history", "new_user")]
        print(f"\n**S.2 blind_v{v} with the other-users line ({len(items)} items; {len(first)} first-time payees): lists of at most 3**\n")
        print("| reader | seeds | right 1st % | in the list % | suggestions shown | work saved | first-time payees: 1st / top 3 | payees seen before | new users | effort W=10 |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        rl = {i: ([rule[i]] if rule[i] >= 0 else []) for i in items}
        c = card(rl, items, plaus, sorted(items))
        print(f"| YNAB today (2 of last 3, else last) | – | {c['first']:.1f} | {c['first'] + c['later']:.1f} | {c['k']:.2f} | {c['saved']:.0f}% | 0.0 / 0.0 | – | – | – |")
        for n in READERS:
            rs = recs(n, f"blind_v{v}_others") or recs(n, f"blind_v{v}")
            if not rs:
                continue
            rows = []
            for r in rs:
                P = folds_calibrate(r, items); ids = sorted(P)
                c = card(lists(P, items, plaus, rule, "model"), items, plaus, ids)
                rank = {i: int(np.where(np.argsort(-P[i]) == items[i]["answer"])[0][0]) + 1 for i in ids}
                a = lambda sel: 100 * np.mean([rank[i] == 1 for i in sel if i in rank])  # noqa: E731
                rows.append([c["first"], c["first"] + c["later"], c["k"], c["saved"], a(first), 100 * np.mean([rank[i] <= 3 for i in first if i in rank]),
                             a(seen), a(newu), evaluate(P, items, 10, "per-item")[2]])
            m = np.mean(rows, 0)
            line = "" if recs(n, f"blind_v{v}_others") else " (read without the line)"
            print(f"| {n}{line} | {len(rs)} | {m[0]:.1f} | {m[1]:.1f} | {m[2]:.2f} | {m[3]:.0f}% | {m[4]:.1f} / {m[5]:.1f} | {m[6]:.1f} | {m[7]:.1f} | {m[8]:.3f} |")
