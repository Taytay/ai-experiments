"""Corrected cells and trivial baselines for the REAL-6 set (review of 2026-09-22, QUESTIONS.md REAL-13).

Three facts about `real6_v1` that the level names hide:

  1. `real6.build` splits each history merchant's rows into history and test *before* cutting the shuffled history to 300 rows,
     so 115 of the 559 "seen" items have a merchant with no row left in the user's history: they are unseen items labelled seen.
  2. A user files every merchant under exactly one label (0 of 1,026 user-merchant pairs carry two), so a merchant lookup on the
     user's own history answers every truly seen item (444 of 444), and the label of the nearest history row under the row 37
     retriever answers 98.6 of them from the statement string alone.
  3. Every scheme maps the standard categories to the user's names by merge and rename, except arbitrary splits: an item whose
     merchant is not in the history is determined by its merchant's standard category and the user's history (603 of 735), or
     falls in a category the user split in two with no signal for the side (119), or neither (13).

`kind(item)` gives the corrected group; `nn1()` the nearest-row label per item (cached, CPU); `hybrid()` the model with the
lookup in front; `user_ci()` a bootstrap over users, the unit the 20-user set actually samples.
"""
import json
from functools import lru_cache

import numpy as np

from . import real6 as R6
from .paths import ROOT

KINDS = ("in history", "labelled seen, not in history", "determined by category", "split category", "other")
NN_CACHE = ROOT / "results" / "real6_nn1.json"
NN_THRESHOLD = 0.8  # retriever cosine above which the nearest row is taken as the same merchant (74% of in-history items, 0 others)


@lru_cache(maxsize=1)
def _doc():
    doc = R6.load()
    return doc, {u["user"]: u for u in doc["users"]}


@lru_cache(maxsize=1)
def kinds():
    """Item id -> corrected group (KINDS)."""
    from . import transactions as T
    doc, users = _doc()
    std = {m["name"]: m["category"] for m in T.load()["merchants"]}
    out = {}
    for it in doc["items"]:
        hist = users[it["user"]]["history"]
        if any(h["merchant"] == it["merchant"] for h in hist):
            out[it["id"]] = KINDS[0]; continue
        labs = {h["label"] for h in hist if std[h["merchant"]] == std[it["merchant"]]}
        gold = it["options"][it["answer"]].strip()
        base = KINDS[2] if labs == {gold} else KINDS[3] if len(labs) > 1 else KINDS[4]
        out[it["id"]] = KINDS[1] if it["seen"] else base
    return out


def kind(item_id):
    return kinds()[item_id]


def item_user():
    doc, _ = _doc()
    return {it["id"]: it["user"] for it in doc["items"]}


@lru_cache(maxsize=1)
def nn1():
    """Item id -> (cosine of the nearest history row under the row 37 retriever, that row's label is gold). Cached in results/."""
    if NN_CACHE.exists():
        return {k: tuple(v) for k, v in json.load(open(NN_CACHE)).items()}
    from .real6_shots import Shots
    doc, users = _doc()
    S = Shots("nearest", device="cpu")
    out = {}
    for it in doc["items"]:
        rows, E, _ = S.pool(users[it["user"]])
        s = E @ S.embed([it["text"]])[0]; j = int(s.argmax())
        out[it["id"]] = (round(float(s[j]), 4), rows[j]["label"] == it["options"][it["answer"]].strip())
    json.dump(out, open(NN_CACHE, "w"))
    return out


def hybrid(recs, threshold=NN_THRESHOLD):
    """Per-item correctness of 'the nearest row's label if its cosine clears the threshold, else the model'."""
    nn = nn1()
    return {i: (nn[i][1] if nn[i][0] >= threshold else r["correct"]) for i, r in recs.items()}


def user_ci(correct_by_id, n_boot=1000, seed=0):
    """95% interval of the accuracy with users resampled (items stay with their user)."""
    iu = item_user(); ids = list(correct_by_id)
    users = sorted({iu[i] for i in ids}); by = {u: np.array([float(correct_by_id[i]) for i in ids if iu[i] == u]) for u in users}
    rng = np.random.default_rng(seed); b = []
    for _ in range(n_boot):
        pick = rng.choice(len(users), len(users))
        x = np.concatenate([by[users[k]] for k in pick]); b.append(100 * x.mean())
    lo, hi = np.percentile(b, [2.5, 97.5])
    return round(float(lo), 1), round(float(hi), 1)


# ---- shared by the table scripts of REPORT.md 51 to 55 (rows 42, 56, 57, 34, 58) ----

def load_recs(*patterns):
    """Per-item records merged from one or more globs under results/per_item (fold files merge into one held-out reading)."""
    import glob
    out = {}
    for pat in patterns:
        for f in sorted(glob.glob(str(ROOT / "results" / "per_item" / pat))):
            out.update({r["id"]: r for r in map(json.loads, open(f))})
    return out


def is_db_only(item_id):
    doc, _ = _doc()
    items = {it["id"]: it for it in doc["items"]}
    return kind(item_id) in KINDS[2:4] and items[item_id]["merchant"] in R6.db_only_merchants()


@lru_cache(maxsize=1)
def _items():
    doc, _ = _doc()
    return {it["id"]: it for it in doc["items"]}


def selectors():
    """(label, item filter) for the report columns: all, name types, corrected groups, DB-only (known / opaque)."""
    it = _items(); db = R6.db_only_merchants()
    dbo = lambda i: kind(i) in KINDS[2:4] and it[i]["merchant"] in db  # noqa: E731
    return [("all", lambda i: True), ("standard", lambda i: it[i]["level"].endswith("standard")), ("renamed", lambda i: it[i]["level"].endswith("renamed")),
            ("coined", lambda i: it[i]["level"].endswith("new")), ("in history", lambda i: kind(i) == KINDS[0]),
            ("labelled seen, not in history", lambda i: kind(i) == KINDS[1]), ("determined by category", lambda i: kind(i) == KINDS[2]),
            ("split category", lambda i: kind(i) == KINDS[3]), ("DB-only", dbo), ("DB-only known", lambda i: dbo(i) and it[i]["known"]),
            ("DB-only opaque", lambda i: dbo(i) and not it[i]["known"])]


def row_cells(recs, cols):
    sel = dict(selectors())
    out = []
    for c in cols:
        x = [r["correct"] for i, r in recs.items() if sel[c](i)]
        out.append(f"{100 * sum(x) / len(x):.1f}" if x else "-")
    return out


def paired(base, other, col="all", n_boot=2000, seed=0):
    """Mean of (other - base) correctness over the items both scored in a column, with a user-resampled 95% interval."""
    sel = dict(selectors())[col]; iu = item_user()
    ids = [i for i in base if i in other and sel(i)]
    d = {}
    for i in ids:
        d.setdefault(iu[i], []).append(float(other[i]["correct"]) - float(base[i]["correct"]))
    us = sorted(d); rng = np.random.default_rng(seed)
    b = [100 * np.mean(np.concatenate([d[u] for u in rng.choice(us, len(us))])) for _ in range(n_boot)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return f"{100 * np.mean([float(other[i]['correct']) - float(base[i]['correct']) for i in ids]):+.1f} [{lo:+.1f}, {hi:+.1f}]"
