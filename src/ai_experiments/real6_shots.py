"""Shot rules for the REAL-6 prompt (PLAN row 41, REAL-9): the 24 shots chosen per query instead of the frozen stratified sample.

The frozen set (`real6.build`) gives every user one 24-shot block, stratified over the categories and filled at random, and the SFT
trainer draws 24 random history rows per training query. A shot rule chooses the block per query from the user's history:

  recent    the 24 rows before the query in history order (at test, the last 24 rows). REAL-6 histories are a frozen shuffle with
            no time axis, so on this set "recent" is a fixed random sample; it is the control that separates "different shots" from
            "chosen shots". Row 43's set gives it a real meaning.
  nearest   the 24 rows nearest to the query under the row 37 retriever (all-MiniLM-L6-v2 tuned on templated renderings of the
            DB's merchant names, `models/adapters/retriever_real6_minilm`; cosine over raw statement strings).
  transact  TransAct V2's rule (references/blog/pinterest/summaries.md, 2025-06-06): the R_RECENT most recent rows plus the
            nearest rows selected per candidate; the candidates are the user's categories, so the nearest row of each category
            in turn (categories ordered by their best similarity) until the block is full.
  cluster   Pinner Progression's rule (2026-07-27): k-means clusters over the user's own rows in the retriever's space, k = the
            number of categories (at most 15, at least 2), and greedy coverage: each pick maximises similarity to the query times
            DISCOUNT to the power of the shots already taken from its cluster.

Shots are presented oldest-first for `recent` and least-similar-first for the others, so the most relevant row sits next to the
query. `render` writes the prompt in exactly the builder's format; `apply` rewrites an item's `prompt` and `prompt_ctx` and adds
`merchant_in_shots` (the query's merchant among the shots) and `gold_in_shots` (the gold label among the shot labels).
"""
import random

import numpy as np

from .paths import ROOT

RULES = ("fixed", "recent", "nearest", "transact", "cluster")
RETRIEVER = ROOT / "models" / "adapters" / "retriever_real6_minilm"
K, R_RECENT, DISCOUNT, MAX_CLUSTERS = 24, 8, 0.7, 15


def render(user, shots, text, amount, weekday, record=None):
    header = "Categories: " + ", ".join(c["name"] for c in user["categories"]) + "\n\n"
    demo = "".join(f"Transaction: {h['text']} | ${h['amount']:.2f} | {h['weekday']}\nCategory: {h['label']}\n\n" for h in shots)
    note = f"Note: {record}\n" if record else ""
    return header + demo + note + f"Transaction: {text} | ${amount:.2f} | {weekday}\nCategory:"


def kmeans(E, k, seed=0, iters=25):
    """Plain k-means with k-means++ seeding on unit vectors (cosine = dot); returns the cluster index per row."""
    rng = np.random.default_rng(seed)
    C = [E[rng.integers(len(E))]]
    for _ in range(1, k):
        d = np.min([1 - E @ c for c in C], axis=0); d = np.maximum(d, 0)
        C.append(E[rng.choice(len(E), p=d / d.sum())] if d.sum() > 0 else E[rng.integers(len(E))])
    C = np.stack(C)
    for _ in range(iters):
        a = (E @ C.T).argmax(1)
        for j in range(k):
            if (a == j).any():
                c = E[a == j].mean(0); C[j] = c / (np.linalg.norm(c) + 1e-8)
    return (E @ C.T).argmax(1)


class Shots:
    """Per-user pools (history rows, their embeddings, their clusters) and the selection under one rule."""

    def __init__(self, rule, seed=0, exclude=(), device="cuda"):
        assert rule in RULES, rule
        self.rule, self.seed, self.exclude, self.device = rule, seed, set(exclude), device
        self.model, self.cache = None, {}

    def embed(self, texts):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(str(RETRIEVER), device=self.device)
            self.model.max_seq_length = 96
        return self.model.encode(list(texts), batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    def pool(self, user, train=False):
        """The rows a shot may come from: the whole history at test; without the held-out merchants for training."""
        key = (user["user"], train)
        if key not in self.cache:
            rows = [h for h in user["history"] if not (train and h["merchant"] in self.exclude)]
            E = self.embed([h["text"] for h in rows]) if self.rule != "recent" else None
            clusters = kmeans(E, max(2, min(MAX_CLUSTERS, len(user["categories"]))), seed=self.seed) if self.rule == "cluster" else None
            self.cache[key] = (rows, E, clusters)
        return self.cache[key]

    def select(self, user, text, train=False, query_index=None):
        """The shot rows for one query. `query_index` is the query's position in the pool (training: excluded from its own shots;
        `recent` counts back from it). At test the query is a held-out transaction and `query_index` is None (recent = the last rows)."""
        rows, E, clusters = self.pool(user, train)
        n = len(rows)
        avail = [i for i in range(n) if i != query_index]
        if self.rule == "recent":
            end = n if query_index is None else query_index
            order = [(end - 1 - j) % n for j in range(n)]  # newest first, cyclic
            picked = [i for i in order if i != query_index][:K][::-1]
            return [rows[i] for i in picked]
        q = self.embed([text])[0]
        sim = E @ q
        if self.rule == "nearest":
            picked = sorted(avail, key=lambda i: -sim[i])[:K]
        elif self.rule == "transact":
            end = n if query_index is None else query_index
            recent = [i for i in [(end - 1 - j) % n for j in range(n)] if i != query_index][:R_RECENT]
            picked = list(recent)
            by_cat = {}
            for i in sorted(avail, key=lambda i: -sim[i]):
                if i not in picked:
                    by_cat.setdefault(rows[i]["label"], []).append(i)
            cats = sorted(by_cat, key=lambda c: -sim[by_cat[c][0]])
            while len(picked) < K and any(by_cat.values()):
                for c in cats:
                    if by_cat[c] and len(picked) < K:
                        picked.append(by_cat[c].pop(0))
            picked = recent + sorted(picked[len(recent):], key=lambda i: sim[i])
            return [rows[i] for i in picked]
        else:  # cluster
            picked, taken = [], {}
            cand = set(avail)
            while len(picked) < K and cand:
                i = max(cand, key=lambda i: sim[i] * DISCOUNT ** taken.get(clusters[i], 0))
                picked.append(i); cand.discard(i); taken[clusters[i]] = taken.get(clusters[i], 0) + 1
        picked = sorted(picked, key=lambda i: sim[i])  # least similar first, the nearest next to the query
        return [rows[i] for i in picked]

    def apply(self, item, user):
        """The item with its prompts rebuilt from the rule's shots (test time: the query is not in the pool)."""
        shots = self.select(user, item["text"])
        return dict(item, prompt=render(user, shots, item["text"], item["amount"], item["weekday"]),
                    prompt_ctx=render(user, shots, item["text"], item["amount"], item["weekday"], item["record"]),
                    merchant_in_shots=any(h["merchant"] == item["merchant"] for h in shots),
                    gold_in_shots=any(h["label"] == item["options"][item["answer"]].strip() for h in shots))


def frozen_flags(item, user):
    """The same two flags for the frozen 24-shot prompt."""
    t2m = {h["text"]: h["merchant"] for h in user["history"]}; t2l = {h["text"]: h["label"] for h in user["history"]}
    ms = {t2m[t] for t in user["shots"] if t in t2m}; ls = {t2l[t] for t in user["shots"] if t in t2l}
    return dict(item, merchant_in_shots=item["merchant"] in ms, gold_in_shots=item["options"][item["answer"]].strip() in ls)
