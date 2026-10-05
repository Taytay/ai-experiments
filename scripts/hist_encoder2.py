"""PLAN step 195 (owner, 2026-10-04: "Is it possible to also try the late interaction models? ... the other approaches mentioned?
Especially the Pinterest stuff"): more history-aware readers beside hist_encoder.py's three (references/blog/pinterest-applications.md):
  4 transact   TransAct V2's rule for the query's history: the 5 most recent earlier filings plus the 5 nearest by encoder 1, each with
               where it went and how long ago ("3 days ago"), against row 187's category text; trained like hist_encoder's approach 3.
  5 maxsim     late interaction at the transaction level: a category is the set of everything filed in it, scored by the mean of the
               query's 3 best cosines against its members (encoder 1; sum-of-MaxSim with whole transactions as the "tokens").
  6 colbert    token-level late interaction (ColBERT): COLBERT_BASE's token states projected to 128 dimensions and normalised, scored by
               the mean over query tokens of the best cosine against the other side's tokens (MaxSim), trained with in-batch InfoNCE plus
               the hard negative on hist_encoder's approach-1 triplets (plain PyTorch: PyLate's pins break this environment's torch);
               it rescores the 50 nearest earlier transactions by encoder 1, and the 20 best vote as in approach 1.
Read like hist_encoder.py (held-out households or the owner's budget, alone and fused with decider), with its readers 1 knn and
two-tower alongside for reference.
  train4 / train6   write models/encoders/<OUT4> / <OUT6> (DVC)
  read
env: as hist_encoder.py, plus OUT4 (hist_transact_v1), OUT6 (hist_colbert_v1), COLBERT_BASE.
usage: SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 uv run python scripts/hist_encoder2.py train4|train6|read
"""
import datetime as dt
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hist_encoder as H  # noqa: E402
from two_tower import households  # noqa: E402

OUT4, OUT6 = H.ENC / os.environ.get("OUT4", "hist_transact_v1"), H.ENC / os.environ.get("OUT6", "hist_colbert_v1")
COLBERT_BASE = os.environ.get("COLBERT_BASE", "BAAI/bge-small-en-v1.5")
NREC, NNB, NCAND = 5, 5, 50


def events(b):
    """hist_encoder.events with each event's date"""
    date = {t["id"]: t["date"] for t in b["transactions"]}
    ev = H.events(b)
    for e in ev:
        e["date"] = dt.date.fromisoformat(date[e["id"]])
    return ev


def _ago(e, f):
    d = (e["date"] - f["date"]).days
    return "yesterday" if d == 1 else f"{d} days ago" if d < 60 else f"{round(d / 30)} months ago"


def _ctx4(i, ev, nb):
    """the query, its NREC most recent earlier-day filings and its NNB nearest (not already listed), each with where it went and when"""
    e = ev[i]
    k = i
    while k > 0 and ev[k - 1]["day"] == e["day"]:
        k -= 1
    rec = list(range(k - 1, max(-1, k - 1 - NREC), -1))
    near = [j for j, _ in nb if j not in rec][:NNB]
    line = lambda j: f'{ev[j]["payee"]} ${ev[j]["amt"]:.2f} -> {ev[j]["label"]} ({_ago(e, ev[j])})'
    return (e["text"] + ("\nRecent: " + "; ".join(map(line, rec)) if rec else "") + ("\nSimilar earlier: " + "; ".join(map(line, near)) if near else ""))


def train4():
    rng = random.Random(H.SEED + 4)
    m1, cache = H._model(H.OUT1), {}
    per = -(-H.PAIRS // H.HOUSEHOLDS)
    rows = []
    for n, b in enumerate(households("train", range(H.HOUSEHOLDS))):
        ev = [e for e in events(b) if e["gold"] in e["state"] and len(e["state"]) > 1]
        nb = H._neighbours(H._embed(m1, [e["text"] for e in ev], cache), ev, NNB + NREC)
        for i in rng.sample(range(len(ev)), min(per, len(ev))):
            e = ev[i]
            hard = [ev[j]["gold"] for j, _ in nb[i] if ev[j]["gold"] != e["gold"] and ev[j]["gold"] in e["state"]]
            neg = hard[0] if hard and rng.random() < 0.7 else rng.choice([c for c in e["state"] if c != e["gold"]])
            rows.append((_ctx4(i, ev, nb[i]), e["state"][e["gold"]], e["state"][neg]))
        if n % 100 == 0:
            print(f"  {n} households", flush=True)
            cache.clear()
    rng.shuffle(rows)
    H._fit(rows, OUT4)


class ColBERT:
    """token vectors: a text encoder's last hidden states, a linear map to 128 dimensions, normalised; padding masked out"""

    def __init__(self, path, train=False):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.enc = AutoModel.from_pretrained(str(path)).to(self.dev)
        self.proj = torch.nn.Linear(self.enc.config.hidden_size, 128, bias=False).to(self.dev)
        if (Path(path) / "proj.pt").exists():
            self.proj.load_state_dict(torch.load(Path(path) / "proj.pt"))
        self.enc.train(train)

    def vecs(self, texts):
        import torch
        b = self.tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.dev)
        with torch.autocast(self.dev, dtype=torch.bfloat16, enabled=self.dev == "cuda"):
            h = self.enc(**b).last_hidden_state
        v = torch.nn.functional.normalize(self.proj(h.float()), dim=-1)
        return v, b["attention_mask"].bool()

    def save(self, out):
        import torch
        self.enc.save_pretrained(str(out)); self.tok.save_pretrained(str(out))
        torch.save(self.proj.state_dict(), Path(out) / "proj.pt")


def maxsim(q, qm, d, dm):
    """[nq, nd] mean over each query's tokens of the best cosine against each document's tokens"""
    import torch
    s = torch.einsum("qtd,nsd->qnts", q, d)
    s = s.masked_fill(~dm[None, :, None, :], -2).max(-1).values
    return (s * qm[:, None, :]).sum(-1) / qm.sum(-1, keepdim=True)


def train6():
    """ColBERT on hist_encoder's approach-1 triplets (its row builder, with the fit swapped for this one)"""
    import torch
    from ai_experiments.licences import open_licence

    def fit(rows, _out):
        open_licence(COLBERT_BASE)
        torch.manual_seed(H.SEED)
        print(f"{len(rows)} triplets; e.g. {rows[0]}", flush=True)
        m = ColBERT(COLBERT_BASE, train=True)
        opt = torch.optim.AdamW(list(m.enc.parameters()) + list(m.proj.parameters()), lr=3e-5)
        steps = len(rows) // H.BATCH
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1, (k + 1) / max(1, steps // 20)) * max(0.0, 1 - k / steps))
        for k in range(steps):
            r = rows[k * H.BATCH:(k + 1) * H.BATCH]
            q, qm = m.vecs([x[0] for x in r])
            d, dm = m.vecs([x[1] for x in r] + [x[2] for x in r])
            logits = maxsim(q, qm, d, dm) / 0.05
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(r), device=m.dev))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if k % 200 == 0:
                print(f"  step {k}/{steps} loss {loss.item():.3f}", flush=True)
        m.save(OUT6)
        print(f"-> {OUT6}", flush=True)
    H._fit = fit
    H.train1()


def _tokens(model, texts, cache):
    import torch
    new = [t for t in dict.fromkeys(texts) if t not in cache]
    with torch.no_grad():
        for a in range(0, len(new), 512):
            v, msk = model.vecs(new[a:a + 512])
            for t, x, mm in zip(new[a:a + 512], v, msk):
                cache[t] = x[mm].float().cpu().numpy()
    return [cache[t] for t in texts]


def _scores(ev, m1, m4, mt, mc):
    c1, c4, ct, cc = {}, {}, {}, {}
    E = H._embed(m1, [e["text"] for e in ev], c1)
    nb = H._neighbours(E, ev, max(NCAND, H.K, NNB + NREC))
    Q4 = H._embed(m4, [_ctx4(i, ev, nb[i]) for i in range(len(ev))], c4)
    Qt = H._embed(mt, [e["text"] for e in ev], ct)
    members = defaultdict(list)
    out, i = [], 0
    while i < len(ev):
        j = i
        while j < len(ev) and ev[j]["day"] == ev[i]["day"]:
            j += 1
        state = ev[i]["state"]
        ids = list(state)
        txt = [state[c] for c in ids]
        C4, Ct = H._embed(m4, txt, c4), H._embed(mt, txt, ct)
        for a in range(i, j):
            r = {}
            knn = defaultdict(float)
            for k, s in nb[a][:H.K]:
                knn[ev[k]["gold"]] += float(np.exp((s - 1) / H.TAU_KNN))
            r["1 knn"] = {c: np.log(knn[c] + 1e-9) for c in ids}
            r["4 transact"] = dict(zip(ids, (C4 @ Q4[a]) / H.TAU))
            r["5 maxsim"] = {c: (np.sort(E[members[c]] @ E[a])[-3:].mean() / H.TAU if members[c] else -1 / H.TAU) for c in ids}
            if mc is not None:
                cand = [k for k, _ in nb[a][:NCAND]]
                if cand:
                    q = _tokens(mc, [ev[a]["text"]], cc)[0]
                    ds = _tokens(mc, [ev[k]["text"] for k in cand], cc)
                    ms = np.array([(q @ d.T).max(axis=1).mean() for d in ds])
                    vote = defaultdict(float)
                    for k, s in sorted(zip(cand, ms), key=lambda x: -x[1])[:H.K]:
                        vote[ev[k]["gold"]] += float(np.exp((s - 1) / H.TAU_KNN))
                    r["6 colbert"] = {c: np.log(vote[c] + 1e-9) for c in ids}
                else:
                    r["6 colbert"] = {c: 0.0 for c in ids}
            r["two-tower (row 187)"] = dict(zip(ids, (Ct @ Qt[a]) / H.TAU))
            out.append(r)
        for a in range(i, j):
            members[ev[a]["gold"]].append(a)
        i = j
        if len(cc) > 200000:
            cc.clear()
    return out


def read():
    m1, m4, mt = H._model(H.OUT1), H._model(OUT4), H._model(H.ENC / "two_tower_v1")
    mc = None
    if OUT6.exists():
        mc = ColBERT(OUT6)
    budget = os.environ.get("READ", "households") == "budget"
    if budget:
        import real_budget_eval as RB
        budgets = [json.loads(RB.CACHE.read_text())["budget"]]
    else:
        a, z = map(int, os.environ.get("TEST_SEEDS", "100000-100049").split("-"))
        budgets = households("test", range(a, z + 1))
    stats = defaultdict(lambda: defaultdict(list))
    allev, allsc = [], []
    for b in budgets:
        ev = events(b)
        for e, r in zip(ev, _scores(ev, m1, m4, mt, mc)):
            if e["gold"] not in e["state"]:
                continue
            for m, d in r.items():
                for g in ("all", "payee filed before" if e["seen"] else "first-time payee"):
                    stats[m][g].append(H._rank(d, e["gold"]))
            allev.append(e); allsc.append(r)
    print(f"\n**{'owner budget' if budget else 'held-out households'}: right category, % top-1 / top-3 / top-10 (n per group)**\n")
    groups = ["all", "payee filed before", "first-time payee"]
    print("| reader | " + " | ".join(f"{g} (n={len(stats['1 knn'][g])})" for g in groups) + " |"); print("|---|---|---|---|")
    for m, st in stats.items():
        print(f"| {m} | " + " | ".join(f"{100 * (np.array(st[g]) <= 1).mean():.1f} / {100 * (np.array(st[g]) <= 3).mean():.1f} / "
                                        f"{100 * (np.array(st[g]) <= 10).mean():.1f}" for g in groups) + " |")
    if budget and os.environ.get("FUSE"):
        H.fuse(allev, allsc)


if __name__ == "__main__":
    {"train4": train4, "train6": train6, "read": read}[sys.argv[1]]()
