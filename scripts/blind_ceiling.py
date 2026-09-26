"""A strong-reader ceiling for an item set (PLAN step 62, the owner's idea 2026-09-26): a sample of items given blind to an Opus
subagent (the prompt only: the user's categories, their 24 labelled shots, the query), its answers scored against gold and against
the categorisers on the same items.

  write <set> <variant...>   sample the items (seeded, stratified: coined-name items by name group, then controls) and write one
                             prompt file per rendering variant plus key.json to results/blind_opus/<set>/; the subagent gets only a
                             prompt file and a neutral instruction (the one used on 2026-09-26 is in results/blind_opus/README.md)
  score <set>                read answers_<variant>.txt ("item N | <category> | reason" per line) and print accuracy by group, with
                             the categorisers' per-item correctness on the same items (their obscure-rendering scores)
usage: uv run python scripts/blind_ceiling.py write novel_merchants_v1 A C; ... ; uv run python scripts/blind_ceiling.py score novel_merchants_v1
"""
import json
import random
import sys

import numpy as np

from ai_experiments import real6_cells as RC
from ai_experiments.paths import PROCESSED, ROOT

OUT = ROOT / "results" / "blind_opus"
VARIANT_FILE = {"A": "", "B": "_full", "C": "_clean"}
CAT = "real6_categoriser_Qwen2.5-3B-Instruct"


def sample(items, seed=55):
    rng = random.Random(seed); pick = []
    for g, n in (("descriptive", 14), ("plain", 14), ("chain", 12)):
        pool = [i for i, x in items.items() if x["name_type"] == "new" and x["nm_group"] == g]; pick += rng.sample(pool, n)
    pick += rng.sample([i for i, x in items.items() if x["name_type"] != "new"], 20)
    rng.shuffle(pick)
    return pick


def load(name, variant):
    return {i["id"]: i for i in json.loads((PROCESSED / f"{name}{VARIANT_FILE[variant]}.json").read_text())["items"]}


def write(name, variants):
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    pick = sample(load(name, "A"))
    for v in variants:
        items = load(name, v)
        with open(d / f"items_{v}.txt", "w") as f:
            for k, i in enumerate(pick):
                f.write(f"### item {k + 1}\n{items[i]['prompt']}\n\n")
    (d / "key.json").write_text(json.dumps(dict(order=pick, seed=55), indent=0))
    print(len(pick), "items written for", variants)


def score(name):
    d = OUT / name; order = json.loads((d / "key.json").read_text())["order"]; items = load(name, "A")
    gold = {i: items[i]["options"][items[i]["answer"]].strip() for i in order}
    groups = [("coined, all", lambda x: x["name_type"] == "new"), ("coined, descriptive", lambda x: x["name_type"] == "new" and x["nm_group"] == "descriptive"),
              ("coined, plain", lambda x: x["name_type"] == "new" and x["nm_group"] == "plain"), ("coined, chain", lambda x: x["name_type"] == "new" and x["nm_group"] == "chain"),
              ("controls", lambda x: x["name_type"] != "new"), ("all", lambda x: True)]
    rows = []
    for v in VARIANT_FILE:
        p = d / f"answers_{v}.txt"
        if p.exists():
            ans = {}
            for line in p.read_text().splitlines():
                if line.strip():
                    k, a = [s.strip() for s in line.split("|")[:2]]; ans[order[int(k.split()[1]) - 1]] = a
            rows.append((f"blind Opus 5.5, rendering {v}", {i: ans.get(i) == gold[i] for i in order}))
    for label, pat in (("3B no DB (obscure)", f"{CAT}_none_h100bf16_f?_alllab_lora_novel_merchants_v1.noctx.jsonl"),
                       ("3B database episodes (obscure)", f"{CAT}_none_h100bf16_f?_alllab_dbep50_lora_novel_merchants_v1.noctx.jsonl"),
                       ("3B database episodes (clean)", f"{CAT}_none_h100bf16_f?_alllab_dbep50_lora_novel_merchants_v1_clean.noctx.jsonl")):
        rs = RC.load_recs(pat)
        rows.append((label, {i: rs[i]["correct"] for i in order}))
    print("| reader | " + " | ".join(f"{g} (n={sum(f(items[i]) for i in order)})" for g, f in groups) + " |")
    print("|---|" + "---|" * len(groups))
    for label, c in rows:
        print(f"| {label} | " + " | ".join(f"{100 * np.mean([c[i] for i in order if f(items[i])]):.0f}" for _, f in groups) + " |")


if __name__ == "__main__":
    {"write": lambda: write(sys.argv[2], sys.argv[3:]), "score": lambda: score(sys.argv[2])}[sys.argv[1]]()
