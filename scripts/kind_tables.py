"""Tables for PLAN steps 147, 148: zero-shot kind inference on blind_v1 / v2 (build_kind_sets.py). Readers: decider-4B untrained, the
untrained decider-35B-A3B, decider-4B with the recipe (other-users line), each in decider's own layout (options after the question,
letter labels). Top-1 and top-3 by level; for categories also by what the category is (one kind / several kinds / a person or purpose).
usage: uv run python scripts/kind_tables.py
"""
import json
import warnings
from collections import defaultdict

import numpy as np

from ai_experiments import canon as CN
from ai_experiments.paths import PROCESSED

R = "results/per_item/real6_dm_decider_"
READERS = {"decider-4B untrained": "decider-4b_{set}.noctx.jsonl", "decider-35B untrained": "decider-35b-a3b_{set}.noctx.jsonl",
           "decider-4B recipe": "decider_decider-4b_none_h100bf16st800_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_oth50_aux100_labrand255_laylabelled_shots_ev10soft_lora_{set}.noctx.jsonl"}


def scores(set_name):
    items = {i["id"]: i for i in json.loads((PROCESSED / f"{set_name}.json").read_text())["items"]}
    out = {}
    for n, tpl in READERS.items():
        try:
            recs = {r["id"]: r["sum_lp"] for r in map(json.loads, open(R + tpl.format(set=set_name)))}
        except FileNotFoundError:
            continue
        out[n] = {i: (int(np.argmax(v)), [int(x) for x in np.argsort(v)[::-1][:3]]) for i, v in recs.items() if i in items}
    return items, out


def table(title, items, out, key):
    groups = defaultdict(list)
    for i, it in items.items():
        groups["all"].append(i); groups[key(it)].append(i)
    print(f"\n**{title}: top-1 / top-3 %**\n")
    print("| group | n | " + " | ".join(out) + " |"); print("|---|---|" + "---|" * len(out))
    for g in ["all"] + sorted(k for k in groups if k != "all"):
        ids = groups[g]
        cells = [f"{100 * np.mean([out[n][i][0] == items[i]['answer'] for i in ids if i in out[n]]):.1f} / "
                 f"{100 * np.mean([items[i]['answer'] in out[n][i][1] for i in ids if i in out[n]]):.1f}" for n in out]
        print(f"| {g} | {len(ids)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    for v in (1, 2):
        items, out = scores(f"kindcat_v{v}")
        kind_of = lambda it: ("several kinds" if it["gold"] == CN.SEVERAL else "person / purpose" if it["gold"] == CN.PURPOSE else "one kind")  # noqa: E731
        table(f"blind_v{v}: what a user's category holds (step 147), by rows of it in the slice", items, out, lambda it: it["level"])
        table(f"blind_v{v}: what a user's category holds, by what the category is", items, out, kind_of)
        items, out = scores(f"kindpay_v{v}")
        table(f"blind_v{v}: a payee's kind (step 148), by what the reader sees / payee type", items, out, lambda it: it["level"])
