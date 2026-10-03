"""PLAN step 175 (owner, 2026-10-03: "can we combine that deterministic approach with our model? ... suggest the last category used (or
ever used) at some sort of deterministic place in the list of suggested categories and see what the best rule of thumb is for where to
insert past categories ... Then the model is just there to enhance something that works 80% of the time"). Offline, from stored runs:
the owner's budget items (real_budget_eval.py, private) and a reader's per-item log-probabilities; prints aggregates only.

List policies (a list of at most KMAX suggestions; "past" = the categories the payee was filed under on earlier dates; with ALIAS=1 the
payee's aliases' filings count too, REPORT 158's character-n-gram aliases):
  model                  the model's order
  rule@k                 YNAB's rule (2 of the last 3, else the last) at position k unless the model already has it in its first k
  last@k                 the payee's last category at position k
  past-recent / past-freq / past-model
                         every past category first (by recency / by count / by the model's probability), then the model's order
  past-model+t           past categories by the model's probability, but a category never used for the payee goes first when the
                         model gives it at least t (t chosen on the earlier half of the timeline, read on the later half)
  rule+gate              REPORT 156's gate: the rule first unless the model's top is 0.2 more probable (calibrated by halves)
Measures: right one first, in the list, MRR over the list (0 if absent), mean list length, by group; the later half only for policies with
a tuned threshold (and, for comparison, every policy is also shown on the later half).
env: BUDGET, OUT, DEC (scores_<DEC>.jsonl), ALIAS (0/1), KMAX (3).
usage: BUDGET=<id> DEC=r173-rs25-s0_split_sim2 uv run python scripts/rule_blend.py
"""
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ai_experiments import calibration as Cal

BUDGET = os.environ["BUDGET"]
OUT = Path(os.environ.get("OUT", Path.home() / ".local" / "share" / "ynab-real-eval" / BUDGET))
DEC = os.environ.get("DEC", "r173-rs25-s0_split_sim2")
ALIAS = os.environ.get("ALIAS") == "1"
KMAX = int(os.environ.get("KMAX", "3"))


def payee_of(it):
    return it["prompt"].rsplit("Transaction: ", 1)[1].split(" | ")[1]


def at(lst, c, k):
    """c at position k (1-based) of list lst unless it is already in the first k; the rest keeps its order."""
    if c in lst[:k]:
        return lst
    rest = [x for x in lst if x != c]
    return rest[:k - 1] + [c] + rest[k - 1:]


if __name__ == "__main__":
    items = {it["id"]: it for it in json.loads((OUT / "items.json").read_text())["items"] if it["answer"] >= 0}
    lp = {r["id"]: np.asarray(r["lp"]) for r in map(json.loads, open(OUT / f"scores_{DEC}.jsonl"))}
    order = sorted(items, key=lambda i: (items[i]["date"], i))
    gold = {i: items[i]["answer"] for i in order}
    opt = {i: {c: k for k, c in enumerate(items[i]["options"])} for i in order}
    aliases = {}
    if ALIAS:  # REPORT 158's alias groups, rebuilt from the SIM=2 items' rule_alias inputs is not possible; use payee name n-grams
        import re
        from sklearn.feature_extraction.text import TfidfVectorizer
        names = sorted({payee_of(items[i]) for i in order})
        cl = [re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", re.sub(r"^(sale|return|sq|tst|pos|debit|purchase|paypal)\b\W*", "", n.lower()))).strip() for n in names]
        T = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform([c or "_" for c in cl])
        S = (T @ T.T).tocsr()
        for k, n in enumerate(names):
            row = S.getrow(k); aliases[n] = {names[j] for j, v in zip(row.indices, row.data) if v >= 0.7 and j != k}
    # past categories per transaction (strictly earlier dates), most recent last
    hist, past, day, pend = defaultdict(list), {}, None, []
    for i in order:
        if items[i]["date"] != day:
            for j in pend:
                hist[payee_of(items[j])].append((items[j]["date"], items[j]["gold"]))
            day, pend = items[i]["date"], []
        p = payee_of(items[i])
        rows = list(hist[p])
        if ALIAS:  # the payee's and its aliases' filings together, in date order
            rows = sorted(r for a in [p] + sorted(aliases.get(p, ())) for r in hist[a])
        past[i] = [opt[i][c] for _, c in rows if c in opt[i]]
        pend.append(i)
    half = {i: k >= len(order) // 2 for k, i in enumerate(order)}
    # calibrated probabilities by halves for the gate
    P = {}
    for h in (False, True):
        fit = [i for i in order if half[i] != h]
        t = Cal.fit_temperature([lp[i] for i in fit], [gold[i] for i in fit])
        P.update({i: Cal.softmax(lp[i], t) for i in order if half[i] == h})

    def rule_of(i):
        pc = past[i]
        if not pc:
            return None
        top = Counter(pc[-3:]).most_common(1)
        return top[0][0] if top[0][1] >= 2 else pc[-1]

    def mlist(i):
        return [int(c) for c in np.argsort(-lp[i])]

    def pol(name, t=None):
        out = {}
        for i in order:
            m = mlist(i); pc = past[i]; r = rule_of(i)
            if name == "model":
                L = m
            elif name.startswith("rule@"):
                L = at(m, r, int(name[5:])) if r is not None else m
            elif name.startswith("last@"):
                L = at(m, pc[-1], int(name[5:])) if pc else m
            elif name.startswith("past-"):
                distinct = list(dict.fromkeys(reversed(pc)))  # most recent first
                if name == "past-freq":
                    cnt = Counter(pc); distinct.sort(key=lambda c: -cnt[c])
                elif name.startswith("past-model"):
                    distinct.sort(key=lambda c: -lp[i][c])
                L = distinct + [c for c in m if c not in distinct]
                if t is not None and distinct and m[0] not in distinct and P[i][m[0]] >= t:
                    L = [m[0]] + [c for c in L if c != m[0]]
            elif name == "rule+gate":
                p = P[i]
                L = m if r is None or (m[0] != r and p[m[0]] - p[r] > 0.2) else at(m, r, 1)
            out[i] = L[:KMAX]
        return out

    def score(L, ids):
        first = np.mean([L[i][0] == gold[i] for i in ids]); inl = np.mean([gold[i] in L[i] for i in ids])
        mrr = np.mean([1 / (L[i].index(gold[i]) + 1) if gold[i] in L[i] else 0 for i in ids])
        return first, inl, mrr

    pols = ["model", "rule@1", "rule@2", "rule@3", "last@1", "last@2", "past-recent", "past-freq", "past-model", "rule+gate"]
    late = [i for i in order if half[i]]
    early = [i for i in order if not half[i]]
    # tune t for past-model+t on the earlier half (best right-first)
    ts = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    best_t = max(ts, key=lambda t: score(pol("past-model", t), early)[0])
    lists = {n: pol(n) for n in pols}
    lists[f"past-model+t (t={best_t}, tuned on the earlier half)"] = pol("past-model", best_t)
    groups = {"all, later half": late, "payee filed before, later half": [i for i in late if past[i]],
              "first-time payee, later half": [i for i in late if not past[i]], "all, whole timeline": order}
    print(f"reader {DEC}; aliases {'on' if ALIAS else 'off'}; lists of at most {KMAX}\n")
    print("| policy | " + " | ".join(f"{g} (n={len(v)})" for g, v in groups.items()) + " |"); print("|---|" + "---|" * len(groups))
    for n, L in lists.items():
        print(f"| {n} | " + " | ".join("{:.1f} / {:.1f} / {:.3f}".format(*(100 * score(L, v)[0], 100 * score(L, v)[1], score(L, v)[2])) for v in groups.values()) + " |")
    print("\ncells: right first % / in the list % / MRR")
