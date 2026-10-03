"""PLAN step 171 (owner, 2026-10-03: "If we were better at just doing payee cleansing alone, we'd be better at all of this ... Is there
any other fuzzy lookup tool from traditional computer science that might help us identify similar payees?"). A payee-resolution
benchmark on the owner's real budget, labelled by YNAB itself: every imported transaction carries the raw bank string
(import_payee_name_original) and the payee YNAB (its matcher plus the owner's corrections) assigned it. Nothing personal is written in the
repo: the strings stay in the YNAB skill's cache, results print as aggregates.

Replay in date order (transfers excluded). When a raw string appears for the first time, each matcher looks it up against every string
seen on earlier dates and proposes the payee of its best match with a score. Two cases:
  known payee   the string is new but its payee had earlier strings: the matcher should find that payee (top-1 accuracy, and recall at a
                threshold)
  new payee     the payee has never been seen: the matcher should abstain (any match is a false alarm)
The decision "this is a payee we know" at a threshold gives precision / recall / F1; reported at each matcher's best-F1 threshold
(chosen on the earlier half of the queries, read on the later half) and as top-1 among known-payee queries.

Matchers (all over cleaned strings: lower case, a leading Sale / Return / processor prefix off, digits and punctuation off):
  exact clean     the cleaned strings are equal (score 1 / 0)
  first word      the first non-noise word is equal (oneslot.payee_key's rule)
  skeleton        consonant skeleton per word (vowels and repeats dropped, first letter kept): "cstco whsl" ~ "costco whse"
  jaro-winkler    rapidfuzz JaroWinkler on the cleaned strings (truncation: a shared prefix scores high)
  token set       rapidfuzz token_set_ratio (word order, extra words)
  char n-grams    TF-IDF over character 3-5-grams, cosine
  embedding       the recipe's decider-4B (REPORT 152's read-out) on the raw string, cosine (real_budget_eval.py EMBED_SOURCE=raw)
  combined        gradient-boosted trees on all the above features of each matcher's top-3 candidates plus two behavioural / shape
                  features (the gap between the new transaction's amount and the candidate payee's median earlier amount, in log
                  dollars; the length gap), trained on the earlier half of the queries
env: BUDGET, OUT (as real_budget_eval.py).
usage: BUDGET=<id> uv run --with rapidfuzz python scripts/payee_resolution_real.py
"""
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

BUDGET = os.environ.get("BUDGET", "")
CACHE = Path.home() / ".cache" / "ynab-cli" / f"{BUDGET}.json"
OUT = Path(os.environ.get("OUT", Path.home() / ".local" / "share" / "ynab-real-eval" / BUDGET))
PREFIX = re.compile(r"^(sale|return|refund|sq|tst|pos|debit|purchase|paypal|pp|sp|py|ach|checkcard|recurring|online|pmt|payment|www)\b\W*")
NOISE = {"sq", "tst", "pos", "debit", "purchase", "sale", "return", "paypal", "the", "inc", "llc", "com", "www", "pmt", "payment", "ach"}


def clean(s):
    s = s.lower()
    for _ in range(2):
        s = PREFIX.sub("", s)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s)).strip()


def first_word(c):
    w = [x for x in c.split() if x not in NOISE and len(x) > 1]
    return w[0] if w else c


def skeleton(c):
    out = []
    for w in c.split():
        if w in NOISE:
            continue
        sk = w[0] + re.sub(r"[aeiouy]", "", w[1:])
        out.append(re.sub(r"(.)\1+", r"\1", sk))
    return " ".join(out)


def skel_sim(a, b):
    """Shared leading skeleton words (truncated words compared on their common prefix), as a share of the shorter skeleton."""
    A, B = a.split(), b.split()
    if not A or not B:
        return 0.0
    k = 0
    for x, y in zip(A, B):
        n = min(len(x), len(y))
        if n >= 2 and x[:n] == y[:n]:
            k += 1
        else:
            break
    return k / min(len(A), len(B))


if __name__ == "__main__":
    from rapidfuzz.distance import JaroWinkler
    from rapidfuzz import fuzz
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    b = json.loads(CACHE.read_text())["budget"]
    pnames = {x["id"]: x["name"] for x in b["payees"]}
    tx = sorted((t for t in b["transactions"] if not t.get("deleted") and t.get("import_payee_name_original") and t.get("payee_id")
                 and not t.get("transfer_account_id")), key=lambda t: (t["date"], t["id"]))
    strings = sorted({t["import_payee_name_original"] for t in tx})
    sid = {s: k for k, s in enumerate(strings)}
    C = [clean(s) for s in strings]
    FW = [first_word(c) for c in C]; SK = [skeleton(c) for c in C]
    emb = np.load(OUT / os.environ.get("EMB_FILE", "raw_emb.npz"))  # row 175: EMB_FILE=raw_emb_<job tag>.npz for another reader
    E = dict(zip(list(emb["names"]), emb["vecs"].astype(np.float32)))
    X = np.stack([E[s] for s in strings]); X /= np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
    T = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform([c or "_" for c in C])

    # the replay: queries are first appearances of a string; the index is the strings first seen on earlier dates
    first_seen, payees_of, first_amt, amts, first_cat, cats_by = {}, defaultdict(set), {}, defaultdict(list), {}, defaultdict(list)
    for t in tx:
        s = t["import_payee_name_original"]
        first_seen.setdefault(s, (t["date"], t["payee_id"]))
        first_amt.setdefault(s, abs(t["amount"]) / 1000)
        amts[t["payee_id"]].append((t["date"], abs(t["amount"]) / 1000))
        if t.get("category_id"):
            first_cat.setdefault(s, t["category_id"]); cats_by[t["payee_id"]].append((t["date"], t["category_id"]))
        payees_of[s].add(t["payee_id"])
    amb = sum(len(v) > 1 for v in payees_of.values())
    order = sorted(strings, key=lambda s: (first_seen[s][0], s))
    queries, index, known_payees, day, pend = [], [], set(), None, []
    for s in order:
        d, p = first_seen[s]
        if d != day:
            for x in pend:
                index.append(sid[x]); known_payees.add(first_seen[x][1])
            day, pend = d, []
        if index:
            queries.append((sid[s], p, p in known_payees, list(index)))
        pend.append(s)
    print(f"{len(tx)} imported transactions, {len(strings)} distinct raw strings ({amb} assigned to more than one payee), "
          f"{len(queries)} queries: {sum(q[2] for q in queries)} new strings of known payees, {sum(not q[2] for q in queries)} of new payees")

    payee_of = [first_seen[s][1] for s in strings]
    Tsim = (T @ T.T).toarray()
    Esim = X @ X.T

    def scores(q, idx):
        """Per matcher: (best index string, score) among idx."""
        idx = np.asarray(idx)
        out = {}
        c = C[q]
        out["exact clean"] = np.array([1.0 if C[j] == c and c else 0.0 for j in idx])
        out["first word"] = np.array([1.0 if FW[j] == FW[q] and FW[q] else 0.0 for j in idx])
        out["skeleton"] = np.array([skel_sim(SK[q], SK[j]) for j in idx])
        out["jaro-winkler"] = np.array([JaroWinkler.similarity(c, C[j]) for j in idx])
        out["token set"] = np.array([fuzz.token_set_ratio(c, C[j]) / 100 for j in idx])
        out["char n-grams"] = Tsim[q, idx]
        out["embedding"] = Esim[q, idx]
        return idx, out

    feats_names = ["exact clean", "first word", "skeleton", "jaro-winkler", "token set", "char n-grams", "embedding"]
    import bisect
    amt_dates = {pp: [d for d, _ in v] for pp, v in amts.items()}

    def amount_gap(q, j):
        """|log amount of the query's first transaction - log median amount of the candidate's payee on earlier dates| (behaviour)."""
        pp = payee_of[j]; d0 = first_seen[strings[q]][0]
        n = bisect.bisect_left(amt_dates[pp], d0)
        prev = [a for _, a in amts[pp][:n]]
        return abs(np.log1p(first_amt[strings[q]]) - np.log1p(np.median(prev))) if prev else 3.0
    per = {m: [] for m in feats_names + ["rank fusion (JW + n-grams + embedding)", "combined"]}
    pick = defaultdict(list)
    cand_rows = []  # (query k, candidate j, features, label: same payee)
    for k, (q, p, known, idx) in enumerate(queries):
        idx, out = scores(q, idx)
        for m in feats_names:
            s = out[m]; best = int(np.argmax(s))
            ties = np.where(s == s[best])[0]
            j = idx[ties[-1]]  # ties: the most recently seen string (no similarity information)
            per[m].append((float(s[best]), payee_of[j] == p)); pick[m].append(j)
        # candidates for the combined matcher: the union of each matcher's top 3
        cset = set()
        for m in feats_names[2:]:
            cset.update(idx[np.argsort(-out[m])[:3]].tolist())
        pos = {j: n for n, j in enumerate(idx)}
        tops = {m: float(out[m].max()) for m in feats_names}
        for j in cset:
            raw = [out[m][pos[j]] for m in feats_names]
            f = raw + [x - tops[m] for x, m in zip(raw, feats_names)] + [amount_gap(q, j), abs(len(C[q]) - len(C[j])) / max(len(C[q]), len(C[j]), 1)]
            cand_rows.append((k, j, f, payee_of[j] == p))
        # reciprocal rank fusion of the three strongest single matchers (no training)
        rr = defaultdict(float)
        for m in ("jaro-winkler", "char n-grams", "embedding"):
            for r, jj in enumerate(idx[np.argsort(-out[m])[:20]]):
                rr[jj] += 1 / (60 + r)
        jb = max(rr, key=rr.get)
        per["rank fusion (JW + n-grams + embedding)"].append((rr[jb] * 60 / 3, payee_of[jb] == p)); pick["rank fusion (JW + n-grams + embedding)"].append(jb)

    half = len(queries) // 2
    tr = [r for r in cand_rows if r[0] < half]
    from sklearn.ensemble import HistGradientBoostingClassifier
    lr = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(np.array([r[2] for r in tr]), np.array([r[3] for r in tr]))
    best_c, pick_c = {}, {}
    for r in cand_rows:
        sc = lr.predict_proba(np.array([r[2]]))[0, 1]
        if r[0] not in best_c or sc > best_c[r[0]][0]:
            best_c[r[0]] = (sc, r[3]); pick_c[r[0]] = r[1]
    per["combined"] = [best_c.get(k, (0.0, False)) for k in range(len(queries))]
    pick["combined"] = [pick_c.get(k, queries[k][3][-1]) for k in range(len(queries))]

    def usual_cat(pp, d0):
        prev = [c for d, c in cats_by[pp] if d < d0]
        return max(set(prev), key=prev.count) if prev else None

    def consistent(k, j):
        """The matched payee's usual category (earlier dates) is the category the owner gave this string's first transaction."""
        q = queries[k][0]; c = first_cat.get(strings[q])
        return c is not None and usual_cat(payee_of[j], first_seen[strings[q]][0]) == c

    known = np.array([q[2] for q in queries]); test = np.arange(len(queries)) >= half
    sel = known & test
    anyright = np.array([any(per[m][k][1] for m in feats_names) for k in range(len(queries))])
    incand = np.zeros(len(queries), bool)
    for r in cand_rows:
        incand[r[0]] |= bool(r[3])
    if os.environ.get("SHOW_FA"):  # local inspection only: embedding false alarms on "new" payees (printed, never saved)
        import random
        fa = [k for k in range(len(queries)) if test[k] and not known[k] and per["embedding"][k][0] >= 0.992]
        random.Random(0).shuffle(fa)
        for k in fa[:25]:
            q, pq, _, idx = queries[k]; idx = np.asarray(idx); j = idx[int(np.argmax(Esim[q, idx]))]
            print(f"   {strings[q][:38]!r:42s} -> {strings[j][:38]!r:42s} payees {pnames.get(pq, '?')[:22]!r} / {pnames.get(payee_of[j], '?')[:22]!r}")
    print(f"headroom on new strings of known payees (later half): some single matcher right top-1 {100 * anyright[sel].mean():.1f}%, "
          f"the right payee among the combined matcher's candidates {100 * incand[sel].mean():.1f}%")
    print("\n**Payee resolution on the owner's budget (YNAB's own payee assignment as gold), later half of the timeline**\n")
    print(f"| matcher | top-1 right payee, new strings of known payees (n={int((known & test).sum())}) | best-F1 threshold (from the earlier half) | precision | recall | F1 | false alarms on new payees (n={int((~known & test).sum())}) | matches whose payee's usual category is this transaction's | of the false alarms |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m, v in per.items():
        sc = np.array([x[0] for x in v]); ok = np.array([x[1] for x in v])

        def prf(th, sel):
            match = sc >= th
            tp = (match & ok & known & sel).sum(); fp = (match & ~(ok & known) & sel).sum(); fn = ((~match | ~ok) & known & sel).sum()
            pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
            return pr, rc, 2 * pr * rc / max(pr + rc, 1e-9), (match & ~known & sel).sum() / max((~known & sel).sum(), 1)
        ths = np.unique(np.round(sc, 3))
        th = max(ths, key=lambda t: prf(t, ~test)[2])
        pr, rc, f1, fa = prf(th, test)
        match = (sc >= th) & test
        cons = np.array([consistent(k, pick[m][k]) if match[k] else False for k in range(len(queries))])
        hascat = np.array([first_cat.get(strings[queries[k][0]]) is not None for k in range(len(queries))])
        mc = match & hascat; fam = mc & ~known
        print(f"| {m} | {100 * (ok & known & test).sum() / (known & test).sum():.1f} | {th:.3f} | {100 * pr:.1f} | {100 * rc:.1f} | {100 * f1:.1f} | {100 * fa:.1f}% | "
              f"{100 * cons[mc].mean():.1f}% | {100 * cons[fam].mean() if fam.any() else float('nan'):.1f}% |")
    from sklearn.inspection import permutation_importance
    te = [r for r in cand_rows if r[0] >= half]
    imp = permutation_importance(lr, np.array([r[2] for r in te]), np.array([r[3] for r in te]), n_repeats=3, random_state=0, scoring="roc_auc")
    print("\ncombined (gradient-boosted trees): permutation importance (AUC drop) " + ", ".join(
        f"{m} {v:.3f}" for m, v in zip(feats_names + [f"{m} vs best" for m in feats_names] + ["amount gap", "length gap"], imp.importances_mean) if v > 0.002))
