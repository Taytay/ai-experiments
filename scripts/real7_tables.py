"""Tables for PLAN step 93: REAL-6-trained readers zero-shot on REAL-7 v1 (build_real7.py), all 200 users (none seen in training).

  R7.1  per reader: top-1, share of the ceiling (the per-item best answer), top-1 by scheme (default / light / custom), and the share
        given p >= 0.9 with its precision
  R7.2  top-1 by what decides the item (plain, idiosyncratic, moved, new category, amount / day rules, random split), with the ceiling
  R7.3  top-1 by history length (long, 25, 5, 0 rows) and by rendering (held-out family or not), merchant in the shots or not
usage: uv run python scripts/real7_tables.py
"""
import json
import warnings

import numpy as np

from ai_experiments import real6_cells as RC
from ai_experiments.paths import PROCESSED

F = "real6_dm_decider_decider_decider-4b_none_{t}_f0_{a}_lora_real7_v1{rd}.noctx.jsonl"
FINAL = "ren50_dbep50_mislead_v1_alt10s_lk10_ov10_aux100_labrand26_ev10soft"
READERS = [("decider-4B untrained", ["real6_dm_decider_decider-4b_real7_v1.noctx.jsonl"]),
           ("Qwen3.5-2B per-option, database episodes (row 78)", ["real6_categoriser_Qwen3.5-2B_none_h100bf16_hf_f0_alllab_dbep50_lora_real7_v1_hfs.noctx.jsonl"]),
           ("decider-4B, row 81 recipe", [F.format(t="h100bf16st800", a="ren50_dbep50_aux100", rd="")]),
           ("decider-4B, + random labels (row 84 control)", [F.format(t="h100bf16st800", a="ren50_dbep50_aux100_labrand26", rd="_labrand26")]),
           ("decider-4B, final recipe (row 89)", [F.format(t=t, a=FINAL, rd="_labrand26") for t in ("h100bf16st800", "h100bf16st800s1")]),
           ("decider-4B, final recipe + shot noise 0.04 (row 94)", [F.format(t=t, a=FINAL.replace("mislead_v1_", "mislead_v1_sn4_"), rd="_labrand26") for t in ("h100bf16st800", "h100bf16st800s1")]),
           ("decider-4B, final recipe + shot noise + moves in time (row 95)", [F.format(t=t, a=FINAL.replace("mislead_v1_", "mislead_v1_sn4_").replace("alt10s_", "alt10s_mv10_"), rd="_labrand26") for t in ("h100bf16st800", "h100bf16st800s1")])]


def sm(z):
    z = np.asarray(z, float); e = np.exp(z - z.max()); return e / e.sum()


def fmt(v):
    return "–" if not v else f"{np.mean(v):.1f}" + (f" [{min(v):.1f}, {max(v):.1f}]" if len(v) > 1 else "")


def table(title, keyname, keys, items, loaded, ceil=True):
    print(f"\n**{title}**\n")
    head = [f"{k} (n={sum(1 for i in items.values() if keyname(i) == k)}{', ceiling %.0f' % (100 * np.mean([i['best'] == i['answer'] for i in items.values() if keyname(i) == k])) if ceil else ''})" for k in keys]
    print("| reader | " + " | ".join(head) + " |"); print("|---|" + "---|" * len(keys))
    for lab, runs in loaded:
        cells = []
        for k in keys:
            cells.append(fmt([100 * np.mean([int(np.argmax(r[i]["sum_lp"])) == items[i]["answer"] for i in r if keyname(items[i]) == k]) for r in runs]))
        print(f"| {lab} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in json.loads((PROCESSED / "real7_v1.json").read_text())["items"]}
    ceiling = 100 * np.mean([i["best"] == i["answer"] for i in items.values()])
    loaded = []
    for lab, files in READERS:
        runs = [{i: x for i, x in RC.load_recs(f).items() if i in items} for f in files]
        runs = [r for r in runs if r]
        if runs:
            loaded.append((lab, runs))
    print(f"**Table R7.1: REAL-7 v1, 200 new users, 1,200 items; ceiling {ceiling:.1f} (the per-item best answer)**\n")
    print("| reader | seeds | top-1 | % of ceiling | default | light | custom | p >= 0.9: share (precision) |"); print("|---|---|---|---|---|---|---|---|")
    for lab, runs in loaded:
        t1 = [100 * np.mean([int(np.argmax(x["sum_lp"])) == x["answer"] for x in r.values()]) for r in runs]
        by = {s: [100 * np.mean([int(np.argmax(r[i]["sum_lp"])) == r[i]["answer"] for i in r if items[i]["scheme"] == s]) for r in runs] for s in ("default", "light", "custom")}
        conf = [[(sm(x["sum_lp"]).max() >= 0.9, int(np.argmax(x["sum_lp"])) == x["answer"]) for x in r.values()] for r in runs]
        sh = [100 * np.mean([c for c, _ in cs]) for cs in conf]; pr = [100 * np.mean([o for c, o in cs if c]) for cs in conf]
        print(f"| {lab} | {len(runs)} | {fmt(t1)} | {100 * np.mean(t1) / ceiling:.1f}% | {fmt(by['default'])} | {fmt(by['light'])} | {fmt(by['custom'])} | {np.mean(sh):.1f} ({np.mean(pr):.1f}) |")
    table("Table R7.2: top-1 by what decides the item", lambda i: i["why"], ["plain", "idiosyncratic", "moved", "new_category", "rule_amount", "rule_day", "random_split"], items, loaded)
    table("Table R7.3a: top-1 by history length", lambda i: i["hist_len"], ["long", "short_25", "short_5", "short_0"], items, loaded)
    table("Table R7.3b: top-1 by rendering and by whether the merchant is in the shots", lambda i: ("held-out rendering" if i["held_render"] else "trained rendering") + (", in shots" if i["in_shots"] else ", not in shots"),
          ["trained rendering, in shots", "held-out rendering, in shots", "trained rendering, not in shots", "held-out rendering, not in shots"], items, loaded)
