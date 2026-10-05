"""PLAN step 205 (owner, 2026-10-05: "apply all of their learnings to our approach"; OmniSearchSage, references/papers/2404.16260): the
crowd line inside the history encoders, plus a hash n-gram payee tower.
  crowd   OmniSearchSage's board titles (other users' names for the collections an item was saved to, +17% beside the item's own text) are
          our crowd line. Each transaction text gets the top CROWD_N category names other households filed its bank string under:
          "<payee> | $<amt> | <Wd> | others: Groceries, Household". The crowd tables are build_crowd.py's for the shared v4 world
          (CROWD_KEY=v2): a training household (split train) reads realstyle_crowd_v4_train.json (train seeds 0-799), a test household reads
          realstyle_crowd_v4_test.json (test seeds 100000-100099 and 200000-200699); either way the household's own filings are left out
          (its seed is removed from every name's household list, as real_budget_eval.py's crowd line), and the line is written only when
          at least CROWD_MIN other households filed the string (decider's CROWD_K). Names are case-merged, ranked by the number of other
          households (frequency), then by how few crowd keys carry the name (specificity: "Misc" sits under thousands of keys, "Diapers"
          under few), then more words, then longer. The table is static (all of the other households' history, as decider's line).
          The enrichment is a wrapper on hist_encoder.events, swapped in before hist_train2 runs: training on the GPU is hist_train2.train
          itself (b1: LOSS=infonce,dedup), output renamed to models/encoders/hist_r205_<ARM>; without a GPU (smoke tests) the same loss
          runs in this file's device-agnostic loop.
  ngram   OmniSearchSage's cheap item tower: the payee (the text before the first " | ") as word unigrams, word bigrams and character
          trigrams, each hashed (crc32, stable across processes) into NGRAM_BUCKETS (2^18) rows of an EmbeddingBag (mean) of 384 dims,
          L2-normalised. Trained with b1's loss (InfoNCE over the batch's positives and hard negatives, scale 20, dedup masks, payee cap)
          on b1's triplets (hist_train2._triplets, same SEED, plain text); for a payee-only tower the dedup mask compares payees (a
          candidate with the anchor's or the positive's payee is masked; the positive itself never is). Saved as tower.safetensors + config.json in
          models/encoders/hist_r205_ngram, so hist_fast's cache key (*.safetensors name, size, mtime) changes when it is retrained.
          NgramEncoder.encode has SentenceTransformer's signature, so hist_fast.scores reads it unchanged (same kNN and MaxSim).
  read    ENCS=<name[:crowd|:plain],...> (default text: crowd for hist_r205_crowd, plain otherwise): % top-1 / top-3 / top-10 for kNN and
          MaxSim, all and first-time payee, on held-out test households (TEST_SEEDS); FUSE=<enc+enc,...> also sums the listed encoders'
          per-event log-softmax kNN scores (the tower beside a sentence encoder).
env: ARM (crowd), LOSS (infonce,dedup), CROWD_N (3), CROWD_MIN (2), NGRAM_BUCKETS (262144), NGRAM_LR (0.002), MAX_EVENTS (all; read only),
     NOCACHE (0: 1 skips hist_fast's score cache), plus hist_train2's PAIRS, HOUSEHOLDS, BATCH, CAP, SEED, TEST_SEEDS.
usage: SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 CROWD_KEY=v2 uv run python scripts/hist_crowd.py train_crowd | train_ngram | read | show
"""
import json
import os
import random
import re
import shutil
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hist_encoder as H  # noqa: E402

from ai_experiments.paths import PROCESSED  # noqa: E402

CROWD_N, CROWD_MIN = int(os.environ.get("CROWD_N", "3")), int(os.environ.get("CROWD_MIN", "2"))
BUCKETS, DIM = int(os.environ.get("NGRAM_BUCKETS", str(2 ** 18))), 384
NGRAM_LR = float(os.environ.get("NGRAM_LR", "0.002"))
CROWD_FILES = {"train": "realstyle_crowd_v4_train.json", "test": "realstyle_crowd_v4_test.json"}
_PLAIN_EVENTS = H.events


# ---------------------------------------------------------------- crowd index and enrichment
class Crowd:
    """one build_crowd table: per key, {lower-cased name: (display name, set of household seeds)}, and per name the number of keys"""

    def __init__(self, name):
        doc = json.loads((PROCESSED / name).read_text())
        self.households = set(doc["households"])
        self.keys, prev = {}, defaultdict(int)
        for k, v in doc["keys"].items():
            merged = {}
            for n, hs in v.items():
                low = n.lower()
                if low in merged:
                    d, s = merged[low]
                    merged[low] = (d if len(s) >= len(hs) else n, s | set(hs))
                else:
                    merged[low] = (n, set(hs))
            self.keys[k] = merged
            for low in merged:
                prev[low] += 1
        self.prev = dict(prev)

    def names(self, key, me):
        """the top CROWD_N names other households (not `me`) filed this key under, or [] when fewer than CROWD_MIN others filed it"""
        v = self.keys.get(key)
        if not v:
            return []
        others = set().union(*(s for _, s in v.values())) - {me}
        if len(others) < CROWD_MIN:
            return []
        got = [(len(s - {me}), low, d) for low, (d, s) in v.items()]
        got = [g for g in got if g[0]]
        got.sort(key=lambda g: (-g[0], self.prev[g[1]], -len(g[2].split()), -len(g[2]), g[2]))
        return [d for _, _, d in got[:CROWD_N]]


_CROWDS = {}


def _crowd(split):
    if split not in _CROWDS:
        _CROWDS[split] = Crowd(CROWD_FILES[split])
    return _CROWDS[split]


def _who(b):
    """("train" | "test", seed) from a synthetic budget id "realstyle-<split>-<seed>"; None for anything else"""
    m = re.fullmatch(r"realstyle-(train|test)-(\d+)", b.get("id", ""))
    return (m.group(1), int(m.group(2))) if m else None


def crowd_events(b):
    """hist_encoder.events(b) with each text enriched by the crowd's names for its bank string (this household left out); events gain
    "crowd" (the names) and "plain" (the text before enrichment)"""
    from build_crowd import KEY_V, crowd_key
    assert KEY_V == "v2", "the v4 crowd tables were built with CROWD_KEY=v2"
    ev = _PLAIN_EVENTS(b)
    who = _who(b)
    if who is None:
        raise SystemExit("crowd enrichment needs a synthetic shared-world household (realstyle-<split>-<seed>)")
    split, me = who
    cr = _crowd(split)
    raw = {t["id"]: t.get("import_payee_name_original") or "" for t in b["transactions"]}
    out = []
    for e in ev:
        names = cr.names(crowd_key(raw.get(e["id"]) or e["payee"]), me)
        e = dict(e, plain=e["text"], crowd=names)
        if names:
            e["text"] = f'{e["text"]} | others: {", ".join(names)}'
        out.append(e)
    return out


def use_crowd(on=True):
    """route every hist_encoder.events caller (hist_train2's triplets and read, hist_fast) through the enricher, or back"""
    H.events = crowd_events if on else _PLAIN_EVENTS


# ---------------------------------------------------------------- hash n-gram tower
def ngram_ids(payee, buckets=BUCKETS):
    """bucket ids of the payee's word unigrams, word bigrams and character trigrams (lower case)"""
    low = payee.lower()
    w = re.findall(r"[a-z0-9]+", low)
    toks = [f"u:{t}" for t in w] + [f"b:{a} {c}" for a, c in zip(w, w[1:])]
    s = f"#{' '.join(w)}#"
    toks += [f"c:{s[k:k + 3]}" for k in range(len(s) - 2)]
    return [zlib.crc32(t.encode()) % buckets for t in (toks or ["<empty>"])]


def _payee(text):
    return text.split(" | ", 1)[0]


class NgramEncoder:
    """EmbeddingBag(mean) over the payee's hashed n-grams, L2-normalised; .encode has SentenceTransformer's signature"""

    def __init__(self, path=None, device=None):
        import torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if path is not None:
            from safetensors.torch import load_file
            cfg = json.loads((Path(path) / "config.json").read_text())
            self.buckets = cfg["buckets"]  # the saved tower's size, whatever NGRAM_BUCKETS says
            w = load_file(str(Path(path) / "tower.safetensors"))["weight"]
        else:
            self.buckets = BUCKETS
            w = torch.randn(BUCKETS, DIM) * 0.1
        self.bag = torch.nn.EmbeddingBag(self.buckets, DIM, mode="mean", sparse=False).to(self.dev)
        with torch.no_grad():
            self.bag.weight.copy_(w.to(self.dev))
        self._ids = {}

    def _batch(self, texts):
        import torch
        ids, offs = [], []
        for t in texts:
            p = _payee(t)
            if p not in self._ids:
                self._ids[p] = ngram_ids(p, self.buckets)
            offs.append(len(ids))
            ids += self._ids[p]
        return torch.tensor(ids, device=self.dev), torch.tensor(offs, device=self.dev)

    def forward(self, texts):
        import torch
        return torch.nn.functional.normalize(self.bag(*self._batch(texts)), dim=-1)

    def encode(self, texts, normalize_embeddings=True, batch_size=512, show_progress_bar=False):
        import torch
        out = []
        with torch.no_grad():
            for a in range(0, len(texts), batch_size):
                out.append(self.forward(texts[a:a + batch_size]).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, DIM), dtype=np.float32)

    def save(self, path):
        from safetensors.torch import save_file
        Path(path).mkdir(parents=True, exist_ok=True)
        save_file({"weight": self.bag.weight.detach().cpu().contiguous()}, str(Path(path) / "tower.safetensors"))
        (Path(path) / "config.json").write_text(json.dumps(dict(kind="hash-ngram", buckets=self.buckets, dim=DIM)))


def _load(d):
    return NgramEncoder(d) if (Path(d) / "tower.safetensors").exists() else H._model(d)


# ---------------------------------------------------------------- b1's loss in a device-agnostic loop
def b1_loss(A, C, b, dev, payee_only=False):
    """hist_train2.train's infonce + dedup cross-entropy for one batch (A anchors, C = positives then hard negatives, normalised).
    payee_only (the n-gram tower, which sees only the payee): "the positive's exact text" becomes the positive's payee, and a candidate
    with the anchor's own payee is masked too (it embeds exactly as the anchor: a logit no update can move)"""
    import torch
    L = 20.0 * (A @ C.T)
    n = len(b)
    hh_ = torch.tensor([r["h"] for r in b], device=dev)
    ga = [hash(r["ga"]) for r in b]
    gc = torch.tensor(ga + [hash(r["gn"]) for r in b], device=dev)
    hc = torch.cat([hh_, hh_])
    tp, tn = [r["p"] for r in b], [r["n"] for r in b]
    if payee_only:
        tp, tn = [_payee(t) for t in tp], [_payee(t) for t in tn]
        ta = [_payee(r["a"]) for r in b]
        txt = torch.tensor([[t in (tp[i], ta[i]) for t in tp + tn] for i in range(n)], device=dev)
    else:
        txt = torch.tensor([[tp[i] == t for t in tp + tn] for i in range(n)], device=dev)
    fn = ((hh_[:, None] == hc[None, :]) & (torch.tensor(ga, device=dev)[:, None] == gc[None, :])) | txt
    fn[torch.arange(n), torch.arange(n)] = False
    L = L.masked_fill(fn, float("-inf"))
    return torch.nn.functional.cross_entropy(L.float(), torch.arange(n, device=dev))


def _loop(rows, enc, params, lr, dev, tag, payee_only=False):
    """one epoch of b1 over the triplets: AdamW, 5% linear warm-up then linear decay, gradient clip 1 (as hist_train2.train)"""
    import torch
    import hist_train2 as HT
    steps = len(rows) // HT.BATCH
    opt = torch.optim.AdamW(params, lr=lr)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    t0, run = time.time(), 0.0
    every = max(1, min(200, steps // 4))
    for s in range(steps):
        b = rows[s * HT.BATCH:(s + 1) * HT.BATCH]
        A = enc([r["a"] for r in b])
        C = enc([r["p"] for r in b] + [r["n"] for r in b])
        loss = b1_loss(A, C, b, dev, payee_only)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        run += loss.item()
        if (s + 1) % every == 0:
            print(f"  {tag} step {s + 1}/{steps} infonce {run / every:.3f} ({time.time() - t0:.0f}s)", flush=True)
            run = 0.0
    return steps


def _rows(loss):
    import hist_train2 as HT
    from two_tower import households
    rng = random.Random(HT.SEED)
    t0 = time.time()
    hh = [H.events(b) for b in households("train", range(HT.HOUSEHOLDS))]
    rows = HT._triplets(hh, loss, rng)
    print(f"{len(rows)} triplets from {len(hh)} households ({time.time() - t0:.0f}s); e.g. {rows[0]['a']!r} / {rows[0]['p']!r} / {rows[0]['n']!r}",
          flush=True)
    return rows


def train_crowd():
    """b1 (or LOSS) on crowd-enriched texts -> models/encoders/hist_r205_<ARM>"""
    import torch
    import hist_train2 as HT
    arm = os.environ.setdefault("ARM", "crowd")
    os.environ.setdefault("LOSS", "infonce,dedup")
    out = H.ENC / f"hist_r205_{arm}"
    use_crowd(True)
    if torch.cuda.is_available():
        tmp = H.ENC / f"hist_r202_r205tmp_{arm}"  # hist_train2.train writes hist_r202_<ARM>; moved to its row-205 name after
        os.environ["ARM"] = f"r205tmp_{arm}"
        HT.train()
        if out.exists():
            shutil.rmtree(out)
        shutil.move(str(tmp), str(out))
        print(f"-> {out}", flush=True)
        return
    assert set(os.environ["LOSS"].split(",")) == {"infonce", "dedup"}, "the CPU loop has b1's loss only"
    from ai_experiments.licences import open_licence
    open_licence(H.BASE)
    torch.manual_seed(HT.SEED)
    rows = _rows({"infonce", "dedup"})
    model = H._model(H.BASE)
    model.train()
    steps = _loop(rows, lambda t: HT._enc(model, t, "cpu"), list(model.parameters()), 5e-5, "cpu", "crowd (cpu)")
    model.save(str(out))
    print(f"-> {out} ({steps} steps on the CPU)", flush=True)


def train_ngram():
    """the hash n-gram tower on b1's triplets -> models/encoders/hist_r205_ngram"""
    import torch
    import hist_train2 as HT
    out = H.ENC / os.environ.get("NGRAM_OUT", "hist_r205_ngram")
    torch.manual_seed(HT.SEED)
    rows = _rows({"infonce", "dedup"})
    tower = NgramEncoder()
    tower.bag.train()
    t0 = time.time()
    steps = _loop(rows, tower.forward, list(tower.bag.parameters()), NGRAM_LR, tower.dev, "ngram", payee_only=True)
    tower.save(out)
    print(f"-> {out} ({steps} steps, {time.time() - t0:.0f}s, {BUCKETS} buckets)", flush=True)


# ---------------------------------------------------------------- read
def _spec(s):
    name, _, mode = s.partition(":")
    return name, mode or ("crowd" if "crowd" in name else "plain")


def read():
    """kNN and MaxSim (hist_fast) per encoder and text mode: % top-1 / top-3 / top-10, all and first-time payee"""
    import hist_fast as HF
    from two_tower import households
    a, z = map(int, os.environ.get("TEST_SEEDS", "100000-100049").split("-"))
    mx = int(os.environ.get("MAX_EVENTS", "0"))
    cache = None if os.environ.get("NOCACHE") == "1" else HF.SYN_CACHE
    budgets = list(households("test", range(a, z + 1)))
    specs = [_spec(s) for s in os.environ["ENCS"].split(",")]
    evs = {}
    for mode in {m for _, m in specs}:
        use_crowd(mode == "crowd")
        evs[mode] = [H.events(b)[:mx or None] for b in budgets]
    use_crowd(False)
    if "crowd" in evs:
        ev = [e for es in evs["crowd"] for e in es]
        print(f"crowd line on {100 * np.mean([bool(e['crowd']) for e in ev]):.1f}% of {len(ev)} events "
              f"({100 * np.mean([bool(e['crowd']) for e in ev if not e['seen']]):.1f}% of first-time payees)")
    print(f"\n**{len(budgets)} held-out households{f', first {mx} events each' if mx else ''}: % top-1 / top-3 / top-10**\n")
    print("| encoder | text | reader | all | first-time payee |\n|---|---|---|---|---|")
    keep = defaultdict(list)  # (encoder spec) -> per household kNN log-softmax over each event's visible categories, for FUSE
    fuse = [f.split("+") for f in os.environ.get("FUSE", "").split(",") if f]
    want = {x for f in fuse for x in f}
    for name, mode in specs:
        d = H.ENC / name
        m = _load(d)
        ranks = defaultdict(lambda: defaultdict(list))
        for ev in evs[mode]:
            sc = HF.scores(ev, m, d, cache)
            ci = {c: k for k, c in enumerate(sc["cats"])}
            per = []
            for i, e in enumerate(ev):
                if e["gold"] not in e["state"]:
                    per.append(None)
                    continue
                ks = np.array([ci[c] for c in e["state"]])
                g = ci[e["gold"]]
                for r in ("1 knn", "5 maxsim"):
                    rk = int((sc[r][i, ks] >= sc[r][i, g]).sum())  # ties count against the gold
                    for grp in ("all",) + (() if e["seen"] else ("first-time",)):
                        ranks[r][grp].append(rk)
                v = sc["1 knn"][i, ks]
                per.append((ks, g, v - v.max() - np.log(np.exp(v - v.max()).sum())))
            if f"{name}:{mode}" in want or name in want:
                keep[f"{name}:{mode}"].append(per)
        for r, st in ranks.items():
            cells = [" / ".join(f"{100 * (np.array(st[g]) <= k).mean():.1f}" for k in (1, 3, 10)) + f" (n={len(st[g])})" for g in ("all", "first-time")]
            print(f"| {name} | {mode} | {r[2:]} | " + " | ".join(cells) + " |", flush=True)
        del m
    for f in fuse:
        f = [x if ":" in x else f"{x}:{_spec(x)[1]}" for x in f]
        st = defaultdict(list)
        for h, ev in enumerate(evs["plain"] if "plain" in evs else evs["crowd"]):
            for i, e in enumerate(ev):
                parts = [keep[x][h][i] for x in f]
                if parts[0] is None:
                    continue
                v = sum(p[2] for p in parts)
                rk = int((v >= v[list(parts[0][0]).index(parts[0][1])]).sum())
                for grp in ("all",) + (() if e["seen"] else ("first-time",)):
                    st[grp].append(rk)
        cells = [" / ".join(f"{100 * (np.array(st[g]) <= k).mean():.1f}" for k in (1, 3, 10)) + f" (n={len(st[g])})" for g in ("all", "first-time")]
        print(f"| {' + '.join(f)} | | knn sum | " + " | ".join(cells) + " |", flush=True)


def show():
    """a few enriched texts from one synthetic household (SHOW_SEED, test split)"""
    from two_tower import households
    use_crowd(True)
    ev = H.events(next(households("test", [int(os.environ.get("SHOW_SEED", "100000"))])))
    seen = set()
    for e in ev:
        if e["payee"] not in seen:
            seen.add(e["payee"])
            print(e["text"])
        if len(seen) >= int(os.environ.get("SHOW_N", "12")):
            break
    print(f"crowd line on {100 * np.mean([bool(e['crowd']) for e in ev]):.1f}% of {len(ev)} events")


if __name__ == "__main__":
    {"train_crowd": train_crowd, "train_ngram": train_ngram, "read": read, "show": show}[sys.argv[1]]()
