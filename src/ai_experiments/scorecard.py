"""One scorecard for every categoriser run (PLAN step 63, QUESTIONS.md EVAL-9), from the saved per-option scores.

The product uses a prediction two ways (the owner, 2026-09-26): auto-file when extremely confident, suggest options otherwise. So a
run is read as a ranker and as a probability, not only by its top-1:

  top1, top3, mrr          rank of the gold option among the user's categories (scores by the sum rule, REPORT.md 50)
  bits, brier              uncertainty left about the gold label, -log2 p(gold), and the Brier score, after one temperature fitted by
                           NLL on the other users (four folds by user id mod 4, as REPORT.md 50)
  cov95, cov98             share of items auto-filed at 95% / 98% precision, the confidence threshold chosen on the other users,
                           with the precision it realised
  baselines                uniform; the user's usage prior (their label shares, add-one); the user's merchant lookup (their label
                           for this merchant when it is in their history); other users (the label other users filed the merchant under,
                           mapped into this user's scheme through the standard category; merges kept, splits skipped); the cascade
                           lookup -> other users -> usage prior, which is what a system with no model would do
  skill                    the share of the headroom over the best baseline the model captures: (model - best) / (100 - best)

Items are any set in REAL-6's format (REAL-6 itself, the novel merchants); the user histories come from REAL-6 unless `users` is
given (POI-1, row 65: its own users, whose categories group Overture basic categories). With such users two things change: other
users' labels map through the Overture basic category of the merchant they filed (in place of REAL-6's standard category), and one
more baseline is reported, the kind lookup (the user's label for another place of the same basic category, else nothing), which is
what a system with a places database and the user's history would do without a model.
"""
import math
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np

from . import calibration as C
from . import real6 as R6


@lru_cache(maxsize=1)
def _users():
    return {u["user"]: u for u in R6.load()["users"]}


@lru_cache(maxsize=1)
def _merchant_std():
    from . import transactions as T
    return {m["name"]: m["category"] for m in T.load()["merchants"]}


def usage_prior(user, users=None):
    u = (users or _users())[user]; c = Counter(h["label"] for h in u["history"]); tot = sum(c.values()) + len(u["categories"])
    return np.array([(c[x["name"]] + 1) / tot for x in u["categories"]])


def lookup(item, users=None):
    """The user's own label for this merchant (majority over their history), as an option index, or None."""
    u = (users or _users())[item["user"]]; labs = Counter(h["label"] for h in u["history"] if h["merchant"] == item["merchant"])
    if not labs:
        return None
    names = [x["name"] for x in u["categories"]]
    return names.index(labs.most_common(1)[0][0])


def _std_to_option(user):
    """Standard category -> this user's option index (skipping categories the user split: the standard category cannot say which half)."""
    out = {}
    for k, c in enumerate(_users()[user]["categories"]):
        if "split" not in c:
            for s in c["standard"]:
                out[s] = k
    return out


def kind_lookup(item, users):
    """POI-1: the user's label for places of the item's Overture basic category (majority over their history), or None."""
    u = users[item["user"]]; labs = Counter(h["label"] for h in u["history"] if h.get("basic") == item.get("basic"))
    if not labs:
        return None
    return [x["name"] for x in u["categories"]].index(labs.most_common(1)[0][0])


def _basic_option(user, users):
    return {b: k for k, c in enumerate(users[user]["categories"]) for b in c["basic"]}


def other_users_poi(item, users):
    """POI-1: the basic category other users' histories give this merchant name (majority), mapped into this user's scheme."""
    votes = Counter(h["basic"] for uid, u in users.items() if uid != item["user"] for h in u["history"] if h["merchant"] == item["merchant"])
    if not votes:
        return None
    return _basic_option(item["user"], users).get(votes.most_common(1)[0][0])


def other_users(item):
    """The standard category other users filed this merchant under (their category's standard, single and unsplit), majority vote,
    mapped into this user's scheme; None when no other user filed it or the category does not map."""
    votes = Counter()
    for uid, u in _users().items():
        if uid == item["user"]:
            continue
        cats = {c["name"]: c for c in u["categories"]}
        for h in u["history"]:
            if h["merchant"] == item["merchant"]:
                c = cats[h["label"]]
                if "split" not in c and len(c["standard"]) == 1:
                    votes[c["standard"][0]] += 1
    if not votes:
        return None
    return _std_to_option(item["user"]).get(votes.most_common(1)[0][0])


def _scores(rec):
    return np.asarray(rec["sum_lp"], float)


def _rank(s, y):
    return int((s > s[y]).sum()) + 1


def scorecard(recs, items, n_boot=500, seed=0, users=None, fold_of=None):
    """recs: {id: per-item record with sum_lp, answer}; items: {id: item}; users: {id: user} for a set with its own users (POI-1).
    fold_of: user id -> one of four calibration folds (default id mod 4; POI-1's fold-0 users, all id mod 4 = 0, use (id // 4) mod 4).
    Returns a dict of metrics, baselines and user intervals."""
    own_users = users; fold_of = fold_of or (lambda u: u % 4)
    ids = sorted(i for i in recs if i in items)
    y = {i: items[i]["answer"] for i in ids}; users = {i: items[i]["user"] for i in ids}
    # leave-users-out temperature and auto-file thresholds (four folds by user id mod 4)
    prob, acc95, acc98 = {}, {}, {}
    for f in range(4):
        fit = [i for i in ids if fold_of(users[i]) != f]; own = [i for i in ids if fold_of(users[i]) == f]
        if not fit or not own:
            continue
        t = C.fit_temperature([_scores(recs[i]) for i in fit], [y[i] for i in fit])
        pf = [C.softmax(_scores(recs[i]), t) for i in fit]; cf, kf, _, _ = C.summarise(pf, [y[i] for i in fit])
        th = {p: C.select_threshold(cf, kf, 1 - p) for p in (0.95, 0.98)}
        for i in own:
            prob[i] = C.softmax(_scores(recs[i]), t)
            acc95[i] = th[0.95] is not None and prob[i].max() >= th[0.95]
            acc98[i] = th[0.98] is not None and prob[i].max() >= th[0.98]
    ids = [i for i in ids if i in prob]
    per = {}
    for i in ids:
        s = _scores(recs[i]); p = prob[i]; k = len(s); r = _rank(s, y[i]); pr = usage_prior(users[i], own_users)
        lk = lookup(items[i], own_users)
        ou = other_users_poi(items[i], own_users) if own_users else other_users(items[i])
        kl = kind_lookup(items[i], own_users) if own_users else None
        casc = lk if lk is not None else ou if ou is not None else int(np.argmax(pr))
        onehot = np.zeros(k); onehot[y[i]] = 1
        per[i] = dict(top1=r == 1, top3=r <= 3, rr=1 / r, bits=-math.log2(max(p[y[i]], 1e-12)), brier=float(((p - onehot) ** 2).sum()),
                      auto95=acc95[i], auto98=acc98[i], conf=float(p.max()),
                      u_top1=1 / k, u_top3=min(3, k) / k, u_bits=math.log2(k),
                      p_top1=int(np.argmax(pr)) == y[i], p_top3=y[i] in np.argsort(-pr)[:3], p_bits=-math.log2(pr[y[i]]),
                      lookup_has=lk is not None, lookup_ok=lk == y[i], other_has=ou is not None, other_ok=ou == y[i], cascade_ok=casc == y[i],
                      kind_has=kl is not None, kind_ok=kl == y[i], kind_cascade_ok=(kl if kl is not None else casc) == y[i])
    m = lambda key, sel=None: 100 * np.mean([per[i][key] for i in ids if sel is None or per[i][sel]]) if any(sel is None or per[i][sel] for i in ids) else float("nan")  # noqa: E731
    out = dict(n=len(ids), top1=m("top1"), top3=m("top3"), mrr=np.mean([per[i]["rr"] for i in ids]), bits=np.mean([per[i]["bits"] for i in ids]),
               brier=np.mean([per[i]["brier"] for i in ids]),
               cov95=m("auto95"), prec95=m("top1", "auto95"), cov98=m("auto98"), prec98=m("top1", "auto98"),
               uniform_top1=m("u_top1"), uniform_top3=m("u_top3"), uniform_bits=np.mean([per[i]["u_bits"] for i in ids]),
               prior_top1=m("p_top1"), prior_top3=m("p_top3"), prior_bits=np.mean([per[i]["p_bits"] for i in ids]),
               lookup_share=m("lookup_has"), lookup_top1=m("lookup_ok", "lookup_has"), other_share=m("other_has"), other_top1=m("other_ok", "other_has"),
               cascade_top1=m("cascade_ok"), kind_share=m("kind_has"), kind_top1=m("kind_ok", "kind_has"), kind_cascade_top1=m("kind_cascade_ok"))
    best1 = max(out["uniform_top1"], out["prior_top1"], out["cascade_top1"])
    out["skill_top1"] = 100 * (out["top1"] - best1) / (100 - best1) if best1 < 100 else float("nan")
    out["skill_top3"] = 100 * (out["top3"] - max(out["uniform_top3"], out["prior_top3"])) / (100 - max(out["uniform_top3"], out["prior_top3"]))
    out["info_bits_over_prior"] = out["prior_bits"] - out["bits"]
    # user-resampled intervals for top1, top3 and bits
    us = sorted(set(users[i] for i in ids)); by = defaultdict(list)
    for i in ids:
        by[users[i]].append(i)
    rng = np.random.default_rng(seed); bt = {"top1": [], "top3": [], "bits": []}
    for _ in range(n_boot):
        jj = [i for u in rng.choice(us, len(us)) for i in by[u]]
        bt["top1"].append(100 * np.mean([per[i]["top1"] for i in jj])); bt["top3"].append(100 * np.mean([per[i]["top3"] for i in jj]))
        bt["bits"].append(np.mean([per[i]["bits"] for i in jj]))
    for k, v in bt.items():
        out[k + "_ci"] = tuple(np.percentile(v, [2.5, 97.5]).round(2 if k == "bits" else 1))
    out["_per_item"] = per
    return out
