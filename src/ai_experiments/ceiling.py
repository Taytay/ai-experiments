"""Per-item ceilings: the best probability of a correct top-1 that any reader could have on an item, given what is observable (the
owner, 2026-09-26: express results as a share of the maximum achievable score). A set's ceiling is the mean over its items, so it
applies to any subset (a fold, a group), and a run's share of the ceiling is its top-1 over the ceiling on the same items.

  real6(item)          REAL-6. An ideal reader knows the merchant's true standard category and the user's exact scheme, but not the
                       per-merchant coin flip of a split category: 1 when the merchant is in the user's history (users file each
                       merchant under one label, real6_cells 2), else 1 / (the number of the user's categories whose standard
                       categories include the merchant's): 1 when the category maps to one name, 1/2 for a two-way split.
  label_induction(item) the label-induction sets. 1 when a readable gold example is in the prompt (every condition but n_gold=0 and
                       the opaque names; the decoy counted 1, as gold is the query's standard category), else 1 / (the coined words no
                       readable example explains: the gold word and the empty categories; v1 has none empty, so elimination gives 1).
  poi1(item, users)    POI-1, exact on seen kinds only: 1 when the item's Overture basic category is in the user's history (a perfect places
                       database plus the history: every basic category maps to one group), None otherwise. A never-filed kind has no
                       computable ceiling: category names carry meaning ("Health care" takes a new clinic) that no rule captures, and a
                       rule that ignored them (a guess among groups sharing the item's top level) was beaten by every trained reader
                       (2026-09-26), so it was a floor, not a ceiling. Report unseen kinds as a bracket: the best reader below, 100 above.
"""
from collections import defaultdict
from functools import lru_cache

import numpy as np

from . import real6 as R6


@lru_cache(maxsize=1)
def _real6():
    from . import transactions as T
    doc = R6.load()
    return {u["user"]: u for u in doc["users"]}, {m["name"]: m["category"] for m in T.load()["merchants"]}


def real6(item):
    users, std = _real6()
    u = users[item["user"]]
    if any(h["merchant"] == item["merchant"] for h in u["history"]):
        return 1.0
    k = sum(std[item["merchant"]] in c["standard"] for c in u["categories"])
    return 1.0 / max(k, 1)


def label_induction(item):
    from . import merchants as M
    f = item["factors"]
    if f["n_gold"] > 0 and f["kind"] != "opaque":
        return 1.0
    filed = set()
    for block in item["prompt"].split("\n\n"):
        if block.startswith("Transaction:") and "\nCategory: " in block:
            filed.add(block.split("\nCategory: ", 1)[1].strip())
    gold = item["options"][item["answer"]].strip()
    filed.discard(gold)  # no readable example of the gold word (none at all, or opaque ones that say nothing about the kind)
    unexplained = [o.strip() for o in item["options"] if o.strip() not in M.CATEGORY_LIST and o.strip() not in filed]
    return 1.0 / max(len(unexplained), 1)


def poi1(item, users):
    return 1.0 if item["basic"] in {h["basic"] for h in users[item["user"]]["history"]} else None


@lru_cache(maxsize=1)
def _poi_tops():
    import json

    from .paths import PROCESSED
    doc = json.loads((PROCESSED / "poi1_v1.json").read_text())
    return {i["basic"]: i["top"] for i in doc["items"]}


def share(correct, ceilings):
    """A run's top-1 as a share of the ceiling on the same items (both as means over the items)."""
    return 100 * float(np.mean(correct)) / float(np.mean(ceilings))
