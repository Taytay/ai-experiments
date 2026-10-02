"""Tables for PLAN step 159 (MODEL-25): category mapping with embeddings (exp_category_mapping.py), beside decider asked the
multiple-choice question on the same categories (rows 147 / 148, kindcat_v2 per-item results).
usage: uv run python scripts/mapping_tables.py
"""
import glob
import json

import numpy as np

MC = {"decider-4B untrained, asked": "results/per_item/real6_dm_decider_decider-4b_kindcat_v2.noctx.jsonl",
      "decider-35B untrained, asked": "results/per_item/real6_dm_decider_decider-35b-a3b_kindcat_v2.noctx.jsonl",
      "decider-4B recipe, asked": "results/per_item/real6_dm_decider_decider_decider-4b_none_h100bf16st800_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_oth50_aux100_labrand255_laylabelled_shots_ev10soft_lora_kindcat_v2.noctx.jsonl"}
NAMES = {"baseline": "no model (TF-IDF)", "decider-4b": "decider-4B untrained", "recipe": "decider-4B recipe", "Qwen3-Embedding-4B": "Qwen3-Embedding-4B"}

if __name__ == "__main__":
    print("**Canonical: each user category to one of 43 canonical options, right first % (top 3 %)**\n")
    print("| method | all | one kind | several kinds | person / purpose | 0 rows shown |"); print("|---|---|---|---|---|---|")
    items = {i["id"]: i for i in json.load(open("data/processed/kindcat_v2.json"))["items"]}
    from ai_experiments import canon as CN
    what = lambda g: "several kinds" if g == CN.SEVERAL else "person / purpose" if g == CN.PURPOSE else "one kind"  # noqa: E731
    for n, f in MC.items():
        recs = [json.loads(l) for l in open(f)]
        r = {x["id"]: int(np.where(np.argsort(-np.asarray(x["sum_lp"])) == items[x["id"]]["answer"])[0][0]) + 1 for x in recs}
        cell = lambda sel: f"{100 * np.mean([r[i] == 1 for i in sel]):.1f} ({100 * np.mean([r[i] <= 3 for i in sel]):.1f})"  # noqa: E731
        ids = list(r)
        print(f"| {n} (multiple choice) | {cell(ids)} | " + " | ".join(cell([i for i in ids if what(items[i]['gold']) == g]) for g in ("one kind", "several kinds", "person / purpose"))
              + f" | {cell([i for i in ids if items[i]['level'] == '0 rows'])} |")
    for f in sorted(glob.glob("results/category_mapping_*.json")):
        d = json.load(open(f))
        for v, r in d["results"].items():
            c = r["canonical"]; cell = lambda g: f"{c[g]['first']} ({c[g]['top3']})"  # noqa: E731
            print(f"| {NAMES.get(d['reader'], d['reader'])}, nearest ({v}) | {cell('all')} | {cell('one kind')} | {cell('several kinds')} | {cell('person / purpose')} | {cell('0 rows')} |")
    print("\n**User to user: each one-kind category of user A to user B's nearest category (600 user pairs), right %**\n")
    print("| method | all | B's category of that kind has another name | B has one with the same name |"); print("|---|---|---|---|")
    for f in sorted(glob.glob("results/category_mapping_*.json")):
        d = json.load(open(f))
        for v, r in d["results"].items():
            u = r["user_user"]
            print(f"| {NAMES.get(d['reader'], d['reader'])} ({v}) | {u['right']} (n={u['n']}) | {u.get('names differ', {}).get('right')} (n={u.get('names differ', {}).get('n')}) | "
                  f"{u.get('B has the same name', {}).get('right')} (n={u.get('B has the same name', {}).get('n')}) |")
