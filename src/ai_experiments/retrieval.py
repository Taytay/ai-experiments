"""Embedding retrieval over the universe field guide and the merchant records (PLAN step 14: BASE-1, REAL-2, GRAPH-7).

A Retriever is all-MiniLM-L6-v2 (through sentence-transformers), optionally fine-tuned on (anchor, positive) pairs with
in-batch negatives (the section 24 recipe), over a fixed list of documents. Two indexes:

  universe_retriever(species): documents = every species' field-guide entry (held-out ones included: the index is what
    the weights do not have) plus the training texts of the seen species; the encoder is trained name -> attribute text
    on the seen species (section 6.3 / 24). Queries are species names. Context builders:
      retrieved_context(names, k)   "Field guide:" + the top-k documents of every name, right or wrong  (evaluation, REAL-2 style)
      raft_context(names, rng)      per name: its gold entry with probability p_gold plus n_distract nearest non-gold documents,
                                    shuffled (RAFT 2403.10131; training stream Er of arm Cr, BASE-1)
      neighbour_context(names, rng) GRAPH-7: per name, a list of co-typed seen species and nothing else (no attribute words)
  merchant_retriever(train_m, all_m): documents = every merchant's canonical fact; the encoder is trained on the
    section 4 / 24 merchant pairs (augmented sentences and names -> category text, training merchants only). Queries are
    bank strings or names; recall@k of the merchant's own record is the REAL-2 number.

Encoders are not saved (20 to 60 seconds to train); results record the pairs, epochs and seed.
"""
import random
import time
import warnings

import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Retriever:
    def __init__(self, docs: list[str], keys: list[set[str]], pairs: list[tuple[str, str]] | None = None,
                 epochs: int = 8, lr: float = 3e-5, bs: int = 32, seed: int = 0, log: str = ""):
        from sentence_transformers import SentenceTransformer
        torch.manual_seed(seed)
        self.model = SentenceTransformer(MODEL, device="cuda", model_kwargs={"torch_dtype": torch.float32})
        self.model.max_seq_length = 96
        self.docs, self.keys = list(docs), [set(k) for k in keys]
        self.trained = dict(pairs=len(pairs) if pairs else 0, epochs=epochs if pairs else 0, lr=lr, seed=seed)
        if pairs:
            self._train(list(pairs), epochs, lr, bs, seed, log)
        self.model.eval()
        self.D = self.embed(self.docs)

    def _tok(self, texts):
        fn = getattr(self.model, "preprocess", None) or self.model.tokenize
        return {k: v.to("cuda") for k, v in fn(texts).items() if torch.is_tensor(v)}

    def embed(self, texts: list[str], grad: bool = False, bs: int = 256) -> torch.Tensor:
        outs = []
        step = len(texts) if grad else bs
        for i in range(0, len(texts), step):
            with (torch.enable_grad() if grad else torch.no_grad()), torch.autocast("cuda", dtype=torch.bfloat16):
                e = self.model(self._tok(texts[i:i + step]))["sentence_embedding"]
            outs.append(F.normalize(e.float(), dim=-1))
        return torch.cat(outs)

    def _train(self, pairs, epochs, lr, bs, seed, log):
        opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=lr)
        r = random.Random(seed); self.model.train(); t0 = time.time(); loss = torch.tensor(0.0)
        for _ in range(epochs):
            r.shuffle(pairs)
            for i in range(0, len(pairs), bs):
                chunk = pairs[i:i + bs]
                a = self.embed([p[0] for p in chunk], grad=True); p = self.embed([p[1] for p in chunk], grad=True)
                scores = a @ p.T * 20.0
                same = torch.tensor([[x[1] == y[1] for y in chunk] for x in chunk], device="cuda")
                scores = scores.masked_fill(same & ~torch.eye(len(chunk), dtype=torch.bool, device="cuda"), -1e4)
                loss = F.cross_entropy(scores, torch.arange(len(chunk), device="cuda"))
                loss.backward(); opt.step(); opt.zero_grad()
        print(f"    retriever{(' ' + log) if log else ''}: trained {epochs} epochs x {len(pairs)} pairs in {time.time() - t0:.0f}s, final loss {loss.item():.3f}", flush=True)

    def topk(self, queries: list[str], k: int) -> list[list[int]]:
        """Indices of the k nearest documents per query."""
        Q = self.embed(queries)
        return (Q @ self.D.T).topk(k, dim=1).indices.tolist()

    def recall(self, queries: list[str], gold: list[str], ks=(1, 5)) -> dict:
        """recall@k: the fraction of queries whose top-k contains a document keyed by its gold entity."""
        top = self.topk(queries, max(ks))
        out = {}
        for k in ks:
            out[f"recall@{k}"] = round(100 * sum(any(g in self.keys[i] for i in t[:k]) for t, g in zip(top, gold)) / len(gold), 1)
        return out


# ------------------------------------------------------------------ universe
def _universe_pairs(species):
    from ai_experiments import universe as U
    seen = [s for s in species if not s["heldout"]]
    attr_text = {"type": {t: U.TYPES[t][0] for t in U.TYPE_LIST},
                 "weakness": {t: f"Creatures weak to {t}-type attacks." for t in U.TYPE_LIST},
                 "habitat": {h: f"Creatures that live in {h} habitats." for h in U.HABITATS}}
    pairs = []
    for s in seen:
        for attr in ("type", "weakness", "habitat"):
            pairs += [(s["name"], attr_text[attr][s[attr]])] * 2
            pairs += [(f"{s['name']} is a {s['type']}-type creature from {s['region']}.", attr_text[attr][s[attr]])]
        pairs += [(s["name"], U.entry(s))] * 2
    return pairs


class UniverseRetriever(Retriever):
    def __init__(self, species, train: bool = True, seed: int = 0):
        from ai_experiments import universe as U
        self.species = species
        self.by_name = {s["name"]: s for s in species}
        names = sorted(self.by_name, key=len, reverse=True)
        entries = [U.entry(s) for s in species]
        texts = U.training_texts(species)
        docs = entries + texts
        keys = [{s["name"]} for s in species] + [{n for n in names if n in t} for t in texts]
        self.n_entries = len(entries)
        self.entry_idx = {s["name"]: i for i, s in enumerate(species)}  # the gold entry of every species is docs[i]
        super().__init__(docs, keys, _universe_pairs(species) if train else None, seed=seed, log="universe")

    def _names_in(self, item: dict) -> list[str]:
        names = ([item["query"]] if item.get("query") else []) + list(item.get("demos", []))
        if not names:
            names = [n for n in self.by_name if n in item["prompt"]]
        return names

    def retrieved_context(self, names: list[str], k: int = 3) -> str:
        """The top-k documents of every name, in retrieval order, duplicates dropped; wrong documents stay wrong."""
        seen, lines = set(), []
        for t in self.topk(names, k):
            for i in t:
                if i not in seen:
                    seen.add(i); lines.append(self.docs[i])
        return "Field guide:\n" + "\n".join(lines) + "\n\n"

    def raft_context(self, names: list[str], rng: random.Random, p_gold: float = 0.8, n_distract: int = 2) -> str:
        """RAFT: per name, the gold entry with probability p_gold, plus the n_distract nearest documents that are not about it."""
        lines = []
        for name, t in zip(names, self.topk(names, n_distract + 4)):
            if rng.random() < p_gold and name in self.entry_idx:
                lines.append(self.docs[self.entry_idx[name]])
            lines += [self.docs[i] for i in t if name not in self.keys[i]][:n_distract]
        rng.shuffle(lines)
        return "Field guide:\n" + "\n".join(lines) + "\n\n"

    def neighbour_context(self, names: list[str], rng: random.Random, n: int = 3) -> str:
        """GRAPH-7: per name, n co-typed seen species (never the item's own names) and no attribute words."""
        seen = [s for s in self.species if not s["heldout"] and s["name"] not in names]
        lines = []
        for name in names:
            s = self.by_name.get(name)
            if s is None:
                continue
            pool = [x["name"] for x in seen if x["type"] == s["type"]]
            lines.append(f"{name}: {', '.join(rng.sample(pool, min(n, len(pool))))}")
        return "Related creatures:\n" + "\n".join(lines) + "\n\n"

    def with_retrieved(self, item: dict, k: int = 3) -> dict:
        return dict(item, prompt=self.retrieved_context(self._names_in(item), k) + item["prompt"])

    def with_neighbours(self, item: dict, rng: random.Random) -> dict:
        return dict(item, prompt=self.neighbour_context(self._names_in(item), rng) + item["prompt"])


# ------------------------------------------------------------------ merchants
def _merchant_pairs(train_m, cat_text):
    from ai_experiments import merchants as M
    pairs = []
    for m in train_m:
        pos = cat_text[m["category"]]
        pairs += [(t, pos) for t in M.augmented(m)]
        pairs += [(m["name"], pos)] * 3
    return pairs


class MerchantRetriever(Retriever):
    """train: False (zero-shot), "category" (the section 4 / 24 pairs: sentences and names -> category text) or "record"
    (the same anchors -> the merchant's own record, i.e. trained for the retrieval task itself)."""

    def __init__(self, all_m, train_m, train="record", seed: int = 0):
        from ai_experiments import merchants as M
        cat_text = {c: f"{c}: {', '.join(M.CATEGORIES[c])}" for c in M.CATEGORY_LIST}
        self.merchants = all_m
        docs = [M.raw_fact(m) for m in all_m]
        keys = [{m["name"]} for m in all_m]
        pairs = None
        if train == "category":
            pairs = _merchant_pairs(train_m, cat_text)
        elif train == "record":
            pairs = [(t, M.raw_fact(m)) for m in train_m for t in M.augmented(m)] + [(m["name"], M.raw_fact(m)) for m in train_m for _ in range(3)]
        elif train:
            raise ValueError(train)
        super().__init__(docs, keys, pairs, epochs=6, seed=seed, log=f"merchants ({train or 'zero-shot'})")
