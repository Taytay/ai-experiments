"""PLAN row 225 (2026-10-06: neither synthetic v4 nor blind_v2 predicted the owner's ranking of the late-interaction arms): which properties
of the owner's budget the generators lack. Aggregates only, per set: the owner's budget (private, read locally; nothing but these numbers
is printed), synthetic v4 held-out households (TEST_SEEDS), blind_v2 users as whole budgets (scripts/blind_budgets.py). Outflows as
hist_encoder2.events reads them (what the models see).
Properties (median over budgets; the owner's set is one budget):
  years            first to last outflow
  txn / month      outflows per month
  categories used  categories with at least one outflow; "hidden at end": of those, hidden or deleted in the budget now; share of outflows
                   filed to them
  span (days)      median days between a category's first and last filing; new categories after year 1, per year
  first-time       share of outflows whose payee string had not been filed before; one-off payees (filed once) as a share of payees
  person / transfer  share of outflows whose payee string looks like a person-to-person payment, check or transfer (P2P regex)
  variants         distinct payee strings per payee key (ai_experiments.payeekey.payee_key_v2), over keys with 2+ filings
  payee switches   share of outflows whose category differs from the payee string's previous filing (payee filed before); of those
                   switches, the share whose next filing of the payee stays in the new category (a move, not an alternation); share of
                   payees with 3+ filings that used 2+ categories
  top-5 share      share of outflows in the five largest categories
usage: SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 BUDGET=<id> uv run python scripts/realness_gap.py
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hist_encoder2 as H2  # noqa: E402
from ai_experiments.payeekey import payee_key_v2  # noqa: E402

P2P = re.compile(r"\b(zelle|venmo|paypal \*?[a-z]+ [a-z]+|cash ?app|square cash|transfer|xfer|check ?#?\d*|atm|withdrawal)\b", re.I)


def props(b):
    ev = H2.events(b)
    if len(ev) < 20:
        return None
    cats = {c["id"]: c for c in b["categories"]}
    d0, d1 = ev[0]["date"], ev[-1]["date"]
    yrs = max((d1 - d0).days, 1) / 365.25
    used = Counter(e["gold"] for e in ev)
    hid = {c for c in used if cats.get(c, {}).get("hidden") or cats.get(c, {}).get("deleted")}
    first, last = {}, {}
    for e in ev:
        first.setdefault(e["gold"], e["date"]); last[e["gold"]] = e["date"]
    new_later = sum(1 for c, d in first.items() if (d - d0).days > 365)
    hidden_now = {c for c in used if cats.get(c, {}).get("hidden") or cats.get(c, {}).get("deleted")}
    gone = 0  # known payees whose previous filing's category is no longer offered (hidden, last used over a year ago: decider's options)
    seen, prev, pc, pend = set(), {}, defaultdict(set), {}
    hist = defaultdict(Counter)  # the payee's earlier filings per category
    nfirst = nswitch = nknown = stay = judged = n3 = maj_wrong = maj_wrong_last_right = 0
    for e in ev:
        p = e["payee"]
        if p not in seen:
            nfirst += 1
        else:
            nknown += 1
            gone += prev[p] in hidden_now and (e["date"] - last[prev[p]]).days > 365
            if p in pend:  # the filing after a switch: still in the new category?
                judged += 1; stay += e["gold"] == pend.pop(p)
            if prev[p] != e["gold"]:
                nswitch += 1; pend[p] = e["gold"]
            if sum(hist[p].values()) >= 3:  # what override training is about: the payee's usual category is no longer its category
                n3 += 1
                mw = hist[p].most_common(1)[0][0] != e["gold"]
                maj_wrong += mw; maj_wrong_last_right += mw and prev[p] == e["gold"]
        seen.add(p); prev[p] = e["gold"]; pc[p].add(e["gold"]); hist[p][e["gold"]] += 1
    pn = Counter(e["payee"] for e in ev)
    keys = defaultdict(set)
    kn = Counter()
    for e in ev:
        k = payee_key_v2(e["payee"]) or e["payee"].lower()
        keys[k].add(e["payee"]); kn[k] += 1
    multi = [len(v) for k, v in keys.items() if kn[k] >= 2]
    n = len(ev)
    return {
        "years": yrs,
        "txn / month": n / (yrs * 12),
        "categories used": len(used),
        "hidden at end (of used)": len(hid) / len(used),
        "outflows in hidden-at-end": sum(used[c] for c in hid) / n,
        "category span (days, median)": float(np.median([(last[c] - first[c]).days for c in used])),
        "new categories / year after year 1": new_later / max(yrs - 1, 1e-9) if yrs > 1.2 else float("nan"),
        "first-time payee strings": nfirst / n,
        "one-off payees (of payees)": sum(1 for v in pn.values() if v == 1) / len(pn),
        "person / transfer / check": sum(bool(P2P.search(e["payee"])) for e in ev) / n,
        "strings per payee key (2+ filings)": float(np.mean(multi)) if multi else float("nan"),
        "payee switches (known payees)": nswitch / max(nknown, 1),
        "previous filing's category no longer offered": gone / max(nknown, 1),
        "payee's majority category wrong (3+ earlier filings)": maj_wrong / max(n3, 1),
        "... of which its latest filing is right (a move)": maj_wrong_last_right / max(maj_wrong, 1),
        "switches that stick (next filing in the new category)": stay / max(judged, 1),
        "payees 3+ filings under 2+ categories": (sum(1 for p, v in pc.items() if pn[p] >= 3 and len(v) >= 2)
                                                 / max(1, sum(1 for p in pc if pn[p] >= 3))),
        "top-5 category share": sum(v for _, v in used.most_common(5)) / n,
    }


def summarise(rows):
    rows = [r for r in rows if r]
    return {k: float(np.nanmedian([r[k] for r in rows])) for k in rows[0]}, len(rows)


def main():
    from two_tower import households
    import blind_budgets
    sets = {}
    a, z = map(int, os.environ.get("TEST_SEEDS", "100000-100049").split("-"))
    sets["synthetic v4"] = summarise([props(b) for b in households("test", range(a, z + 1))])
    sets["blind_v2"] = summarise([props(b) for b in blind_budgets.budgets()])
    if os.environ.get("BUDGET"):
        import real_budget_eval as RB
        sets["owner"] = summarise([props(json.loads(RB.CACHE.read_text())["budget"])])
    names = list(sets)
    print("| property | " + " | ".join(f"{k} (n={sets[k][1]})" for k in names) + " |")
    print("|---|" + "---|" * len(names))
    for p in next(iter(sets.values()))[0]:
        f = lambda v: f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}"
        print(f"| {p} | " + " | ".join(f(sets[k][0][p]) for k in names) + " |")


if __name__ == "__main__":
    main()
