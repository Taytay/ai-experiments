"""PLAN step 200 (owner, 2026-10-05: "is representing them as strings a bad idea? ... embed the amount as dollars in a numeric input";
"A small amount of money at QT is likely to be a treat, not gas"): amounts as numbers for the history encoders' kNN and MaxSim readers.
Encoders: A hist_knn_v1 (amount in the text, "$57.09"), B hist_knn_noamt_v1 (AMT_TEXT=none). Each earlier transaction's cosine s to the
query is adjusted before voting: s' = s - TAU_KNN x d^2 / (2 x SIGMA^2) + TAU_KNN x log(1 + BETA) x [same amount to the cent], with d the
difference of log amounts (C: a closeness kernel, multiplying a neighbour's kNN weight by exp(-d^2 / 2 SIGMA^2); D: an exact-repeat bonus,
multiplying it by 1 + BETA). kNN: the 20 best adjusted among the 50 nearest by cosine vote; MaxSim: a category's 3 best adjusted
similarities among all its filings. SIGMA and BETA chosen on held-out synthetic households (whose generator barely routes by amount, so the
choice may be "off"), then applied to the owner's budget; the full grid is printed for the owner's budget as a diagnostic, labelled so.
Groups: all, payee filed before, first-time payee, and "payee already filed under 2+ categories" (where the amount should decide: QT).
env: SIGMAS (inf,1,0.5,0.25), BETAS (0,1,3), TEST_SEEDS (100000-100019; the chain uses 10), BUDGET (the owner's budget, private).
usage: SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 uv run python scripts/hist_amount.py
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hist_encoder as H  # noqa: E402
from two_tower import households  # noqa: E402

SIGMAS = [float(x) for x in os.environ.get("SIGMAS", "inf,1,0.5,0.25").split(",")]
BETAS = [float(x) for x in os.environ.get("BETAS", "0,1,3").split(",")]
ENCS = {"A amount as text": ("hist_knn_v1", "text"), "B no amount": ("hist_knn_noamt_v1", "none")}
NCAND = 50


def prepare(model, ev):
    """embeddings, the 50 nearest earlier events, and per event the categories its payee was filed under before"""
    E = H._embed(model, [e["text"] for e in ev], {})
    nb = H._neighbours(E, ev, NCAND)
    cats, multi = defaultdict(set), []
    day_start = 0
    for i, e in enumerate(ev):
        if i and ev[i]["day"] != ev[i - 1]["day"]:
            for k in range(day_start, i):
                cats[ev[k]["payee"]].add(ev[k]["gold"])
            day_start = i
        multi.append(len(cats[e["payee"]]) >= 2)
    return E, nb, multi


def ranks(ev, E, nb, combos):
    """per event (None when its gold is not visible): {(sigma, beta): (rank under kNN, rank under MaxSim)}; one pass for all settings"""
    la = np.log(np.array([max(e["amt"], 0.01) for e in ev]))
    cents = np.array([round(e["amt"] * 100) for e in ev])
    code = {c: k for k, c in enumerate(dict.fromkeys(e["gold"] for e in ev))}
    gcode = np.array([code[e["gold"]] for e in ev])
    inv = {k: c for c, k in code.items()}
    out, i = [], 0

    def adj(idx, s, a, sigma, beta):
        d = la[idx] - la[a]
        if not np.isinf(sigma):
            s = s - H.TAU_KNN * d * d / (2 * sigma * sigma)
        if beta:
            s = s + H.TAU_KNN * np.log1p(beta) * (cents[idx] == cents[a])
        return s
    while i < len(ev):
        j = i
        while j < len(ev) and ev[j]["day"] == ev[i]["day"]:
            j += 1
        ids = list(ev[i]["state"])
        prior = np.arange(i)
        for a in range(i, j):
            e = ev[a]
            if e["gold"] not in e["state"]:
                out.append(None)
                continue
            cand = np.array([k for k, _ in nb[a]], dtype=int)
            sims = np.array([s for _, s in nb[a]])
            s_all = E[:i] @ E[a] if i else np.zeros(0)
            res = {}
            for sigma, beta in combos:
                vote = defaultdict(float)
                if len(cand):
                    s2 = adj(cand, sims, a, sigma, beta)
                    for k in np.argsort(-s2)[:H.K]:
                        vote[ev[cand[k]]["gold"]] += float(np.exp((s2[k] - 1) / H.TAU_KNN))
                knn = {c: np.log(vote[c] + 1e-9) for c in ids}
                mx = {c: -1.0 for c in ids}
                if i:
                    s2 = adj(prior, s_all, a, sigma, beta)
                    order = np.lexsort((-s2, gcode[:i]))
                    g = gcode[:i][order]
                    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
                    rank_in = np.arange(len(g)) - np.repeat(starts, np.diff(np.r_[starts, len(g)]))
                    keep = rank_in < 3
                    tot = np.bincount(g[keep], weights=s2[order][keep], minlength=len(code))
                    cnt = np.bincount(g[keep], minlength=len(code))
                    for k in np.flatnonzero(cnt):
                        if inv[k] in mx:
                            mx[inv[k]] = tot[k] / cnt[k]
                res[(sigma, beta)] = (H._rank(knn, e["gold"]), H._rank(mx, e["gold"]))
            out.append(res)
        i = j
    return out


def evaluate(budgets_fn):
    """{encoder: {(sigma, beta): {reader: {group: [top-1 hits]}}}}"""
    res = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for name, (enc, amt_text) in ENCS.items():
        H.AMT_TEXT = amt_text
        model = H._model(H.ENC / enc)
        for b in budgets_fn():
            ev = H.events(b)
            E, nb, multi = prepare(model, ev)
            combos = [(sg, bt) for sg in SIGMAS for bt in BETAS]
            for e, mu, rr in zip(ev, multi, ranks(ev, E, nb, combos)):
                if rr is None:
                    continue
                gs = ["all", "payee filed before" if e["seen"] else "first-time payee"] + (["payee filed under 2+ categories"] if mu else [])
                for sb, r in rr.items():
                    for reader, rk in zip(("kNN", "MaxSim"), r):
                        for g in gs:
                            res[name][sb][reader][g].append(rk == 1)
    return res


def table(res, title):
    groups = ["all", "payee filed before", "first-time payee", "payee filed under 2+ categories"]
    print(f"\n**{title}: % right first (n)**\n")
    print("| encoder | SIGMA | BETA | reader | " + " | ".join(groups) + " |"); print("|---|---|---|---|" + "---|" * len(groups))
    for name, by in res.items():
        for (sigma, beta), rd in by.items():
            for reader, st in rd.items():
                print(f"| {name} | {sigma} | {beta} | {reader} | " + " | ".join(f"{100 * np.mean(st[g]):.1f} ({len(st[g])})" if st[g] else "" for g in groups) + " |")


if __name__ == "__main__":
    a, z = map(int, os.environ.get("TEST_SEEDS", "100000-100019").split("-"))
    syn = evaluate(lambda: households("test", range(a, z + 1)))
    table(syn, "held-out synthetic households")
    chosen = {name: {reader: max(by, key=lambda sb: np.mean(by[sb][reader]["all"])) for reader in ("kNN", "MaxSim")} for name, by in syn.items()}
    print("\nchosen on synthetic (SIGMA, BETA):", {n: {r: c for r, c in v.items()} for n, v in chosen.items()})
    if os.environ.get("BUDGET"):
        import real_budget_eval as RB
        b = json.loads(RB.CACHE.read_text())["budget"]
        own = evaluate(lambda: [b])
        groups = ["all", "payee filed before", "first-time payee", "payee filed under 2+ categories"]
        print("\n**owner's budget, settings chosen on synthetic: % right first (n)**\n")
        print("| encoder | reader | SIGMA, BETA | " + " | ".join(groups) + " |"); print("|---|---|---|" + "---|" * len(groups))
        for name, by in own.items():
            for reader in ("kNN", "MaxSim"):
                for sb, label in (((float("inf"), 0.0), "off"), (chosen[name][reader], "chosen")):
                    st = by[sb][reader]
                    print(f"| {name} | {reader} | {sb} ({label}) | " + " | ".join(f"{100 * np.mean(st[g]):.1f} ({len(st[g])})" for g in groups) + " |")
        table(own, "owner's budget, full grid (diagnostic: picking from it would tune on the owner's budget)")
