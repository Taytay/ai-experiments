"""Tables for PLAN step 158 (MODEL-25): personal embeddings (exp_personal_embed.py). The embedding-only categoriser (the user's category
nearest the transaction) in every pairing of global / personal embeddings, right first and in the top 3, by what the item is; beside
decider-4B's own answer on the same items (the recipe, three seeds, the no-line set) and YNAB's rule (one suggestion, none for a new
payee); and how the history moves the transaction's embedding (cosine to itself without history; the gold category's margin over the
best other category, without and with history).
usage: uv run python scripts/personal_tables.py
"""
import glob
import json
import warnings

import numpy as np

from ai_experiments.paths import PROCESSED

DEC = ("results/per_item/real6_dm_decider_decider_decider-4b_none_h100bf16st800*_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_oth50"
       "_aux100_labrand255_laylabelled_shots_ev10soft_lora_{set}_labrand255_laylabelled_shots.noctx.jsonl")
P2P = {"p2p", "ambiguous_p2p", "p2p_anon"}

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    for st in ("blind_v1", "blind_v2"):
        files = sorted(glob.glob(f"results/personal_embed_*_{st}.json"))
        if not files:
            continue
        items = {i["id"]: i for i in json.loads((PROCESSED / f"{st}.json").read_text())["items"]}
        rule = json.loads((PROCESSED / f"{st}_ynabrule.json").read_text())
        ids = [r["id"] for r in json.load(open(files[0]))["items"]]
        ph = json.loads((PROCESSED / f"{st}_payeehist.json").read_text())  # the user's earlier rows of this payee (blind_v1 items do not carry it)
        first = {i for i in ids if not ph[i]}
        groups = {"all": ids, "first-time payees": [i for i in ids if i in first], "payees seen before": [i for i in ids if i not in first],
                  "person-to-person": [i for i in ids if items[i]["why"] in P2P], "new users": [i for i in ids if items[i]["why"] in ("short_history", "new_user")]}
        print(f"\n**{st}: right first % (top 3 %)**\n")
        print("| suggestions from | " + " | ".join(f"{g} (n={len(v)})" for g, v in groups.items()) + " |"); print("|---|" + "---|" * len(groups))
        print("| YNAB today | " + " | ".join(f"{100 * np.mean([rule[i] == items[i]['answer'] for i in v]):.1f}" for v in groups.values()) + " |")
        decs = [{r["id"]: r for r in map(json.loads, open(f))} for f in sorted(glob.glob(DEC.format(set=st)))]
        if decs:
            def drank(d, i):
                lp = np.asarray(d[i]["sum_lp"]); return int(np.where(np.argsort(-lp) == items[i]["answer"])[0][0]) + 1
            cell = lambda v: f"{np.mean([100 * np.mean([drank(d, i) == 1 for i in v]) for d in decs]):.1f} ({np.mean([100 * np.mean([drank(d, i) <= 3 for i in v]) for d in decs]):.1f})"  # noqa: E731
            print(f"| decider-4B recipe, its own answer ({len(decs)} seeds) | " + " | ".join(cell(v) for v in groups.values()) + " |")
        for f in files:
            d = json.load(open(f)); recs = {r["id"]: r for r in d["items"]}
            for key in d["items"][0]["ranks"]:
                c = " | ".join(f"{100 * np.mean([recs[i]['ranks'][key] == 1 for i in v]):.1f} ({100 * np.mean([recs[i]['ranks'][key] <= 3 for i in v]):.1f})" for v in groups.values())
                print(f"| embeddings, {d['reader']}: {key} | {c} |")
        print(f"\n**{st}: how the history moves the transaction's embedding**\n")
        print("| reader | group | cosine to itself without history | gold margin without history | with history | gold category closest: without / with |")
        print("|---|---|---|---|---|---|")
        for f in files:
            d = json.load(open(f)); recs = {r["id"]: r for r in d["items"]}
            for g, v in groups.items():
                m = lambda k: np.mean([recs[i][k] for i in v])  # noqa: E731
                print(f"| {d['reader']} | {g} | {m('move'):.3f} | {m('gold_margin_global'):+.3f} | {m('gold_margin_personal'):+.3f} | "
                      f"{100 * np.mean([recs[i]['gold_margin_global'] > 0 for i in v]):.1f} / {100 * np.mean([recs[i]['gold_margin_personal'] > 0 for i in v]):.1f} |")
