"""PLAN step 150 (owner, 2026-09-30: no reliable kind database; infer what we can from existing categorised transactions; ~1M users, ~1B
transactions). Data-driven concepts and payee profiles from a population's filings, mapped onto each user's own categories:

1. Concepts: every user category (user, name) is described by the payees (`ai_experiments.payeekey`) filed under it; TF-IDF over payee
   keys seen by >= 5 users, SVD to 64 dimensions, k-means into K concepts (no hand-made vocabulary); a concept is named by its most common
   category names. Learned from the population users only (uid >= 1000), never the test users.
2. Payee profile: for a test transaction, the population users' filings of the same payee key dated before it, as a distribution over
   concepts (shown only with >= 3 distinct users).
3. The test user's own categories placed in concepts: from their filings before the transaction (projected and assigned to the nearest
   concept); a category with none, by its name (the concept most population categories of that name belong to).
4. The profile mapped onto the user's categories: "Other users file this payee as: <the user's category> (<% of others>), ..." in the
   format the recipe was trained to read (the other-users line), and a ranked list for a no-model suggestion.
Writes data/processed/blind_v2_mapped.json (blind_v2's items with the mapped line in place of the raw other-users line, or none) and
prints the no-model suggestion's scores.
usage: uv run python scripts/payee_profiles.py [K]
"""
import gzip
import json
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

from ai_experiments.paths import PROCESSED
from ai_experiments.payeekey import payee_key

K = int(sys.argv[1]) if len(sys.argv) > 1 else 60
MIN_USERS, MIN_PROFILE = 5, 3


def norm_name(n):
    return re.sub(r"[^a-z ]", "", n.lower()).strip()


if __name__ == "__main__":
    rows = [json.loads(l) for l in gzip.open(PROCESSED / "pop_v2.jsonl.gz", "rt")]
    for r in rows:
        r.append(payee_key(r[2]))  # [uid, date, text, filed, mid, key]
    pop = [r for r in rows if r[0] >= 1000]
    key_users = defaultdict(set)
    for r in pop:
        key_users[r[5]].add(r[0])
    keys = sorted(k for k, us in key_users.items() if len(us) >= MIN_USERS and k)
    kidx = {k: j for j, k in enumerate(keys)}
    node_counts = defaultdict(Counter)
    for r in pop:
        if r[5] in kidx:
            node_counts[(r[0], r[3])][kidx[r[5]]] += 1
    nodes = [n for n, c in node_counts.items() if sum(c.values()) >= 3]
    nidx = {n: i for i, n in enumerate(nodes)}
    X = sparse.lil_matrix((len(nodes), len(keys)))
    for n in nodes:
        for j, v in node_counts[n].items():
            X[nidx[n], j] = v
    tf = TfidfTransformer().fit(X.tocsr())
    svd = TruncatedSVD(64, random_state=0).fit(tf.transform(X.tocsr()))
    Z = normalize(svd.transform(tf.transform(X.tocsr())))
    km = KMeans(K, n_init=4, random_state=0).fit(Z)
    concept = {n: int(km.labels_[i]) for n, i in nidx.items()}
    names = defaultdict(Counter)
    for n, c in concept.items():
        names[c][norm_name(n[1])] += 1
    cname = {c: " / ".join(x for x, _ in names[c].most_common(2)) for c in names}
    by_name = defaultdict(Counter)
    for n, c in concept.items():
        by_name[norm_name(n[1])][c] += 1
    print(f"{len(keys)} payee keys, {len(nodes)} category nodes, {K} concepts; e.g.", "; ".join(cname[c] for c in range(min(12, K))))

    def project(counter):
        v = sparse.lil_matrix((1, len(keys)))
        for j, x in counter.items():
            v[0, j] = x
        z = normalize(svd.transform(tf.transform(v.tocsr())))
        return int(km.predict(z)[0])

    by_key = defaultdict(list)
    for r in pop:
        by_key[r[5]].append(r)
    test_rows = defaultdict(list)
    for r in rows:
        if r[0] < 1000:
            test_rows[r[0]].append(r)
    items = json.loads((PROCESSED / "blind_v2.json").read_text())["items"]
    ph = json.loads((PROCESSED / "blind_v2_payeehist.json").read_text())
    out, stats = [], Counter()
    ranks = []
    for it in items:
        u, d, k = it["user"], it["date"], payee_key(it["text"])
        opts = [o.strip() for o in it["options"]]
        prof = Counter(); us = set()
        for r in by_key.get(k, []):
            if r[1] < d and (r[0], r[3]) in concept:
                prof[concept[(r[0], r[3])]] += 1; us.add(r[0])
        mine = defaultdict(Counter)
        for r in test_rows[u]:
            if r[1] < d and r[5] in kidx:
                mine[r[3]][kidx[r[5]]] += 1
        ccat = {}
        for c in opts:
            if mine[c]:
                ccat[c] = project(mine[c])
            elif by_name.get(norm_name(c)) and sum(by_name[norm_name(c)].values()) >= 3:
                ccat[c] = by_name[norm_name(c)].most_common(1)[0][0]
        score = Counter()
        if len(us) >= MIN_PROFILE:
            tot = sum(prof.values())
            for c, n in prof.items():
                cands = [x for x in opts if ccat.get(x) == c]
                if cands:
                    best = max(cands, key=lambda x: sum(mine[x].values()))
                    score[best] += n / tot
        stats["items"] += 1; stats["profile"] += len(us) >= MIN_PROFILE; stats["mapped"] += bool(score)
        if not ph[it["id"]]:
            stats["first"] += 1; stats["first_mapped"] += bool(score)
        ranked = [x for x, _ in score.most_common()]
        gold = opts[it["answer"]]
        ranks.append((it["id"], not ph[it["id"]], ranked.index(gold) + 1 if gold in ranked else 99))
        line = ("Other users file this payee as: " + ", ".join(f"{x} ({round(100 * s)})" for x, s in score.most_common(3))) if score else None
        base = it["prompt"]
        head, _, last = base.rpartition("\n\nTransaction: ")
        p = head + ("\n\n" + line if line else "") + "\n\nTransaction: " + last
        out.append(dict(it, prompt=p, prompt_ctx=p, mapped_rank=ranked[:3]))
    (PROCESSED / "blind_v2_mapped.json").write_text(json.dumps({"name": "blind_v2_mapped", "items": out}, ensure_ascii=False))
    print(f"items {stats['items']}: payee profile (>= {MIN_PROFILE} population users) {100 * stats['profile'] / stats['items']:.1f}%, mapped onto the user's "
          f"categories {100 * stats['mapped'] / stats['items']:.1f}%; first-time payees mapped {100 * stats['first_mapped'] / max(1, stats['first']):.1f}%")
    for name, sel in (("all", [r for r in ranks]), ("first-time payees", [r for r in ranks if r[1]])):
        rk = np.array([r[2] for r in sel])
        print(f"no-model suggestion from the mapped profile, {name} (n={len(sel)}): right first {100 * np.mean(rk == 1):.1f}%, in top 3 {100 * np.mean(rk <= 3):.1f}%")
