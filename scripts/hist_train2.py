"""PLAN step 202 (owner, 2026-10-05: "apply all of their learnings to our approach"; references/papers INDEX thread 7): the training fixes
the Pinterest papers report, as arms on hist_encoder's approach 1 (a transaction lands near the household's earlier transactions filed
in the same category), read as kNN and MaxSim (hist_fast) on held-out v4 households. Same base (bge-small), triplets, batch (128), steps
and learning rate as hist_knn_v1, in a plain PyTorch loop so the loss can be changed:
  infonce  anchor against the batch's positives and hard negatives, scale 1/0.05 (sentence-transformers' MNRL: hist_knn_v1's loss)
  dedup    no false negatives: a candidate from the anchor's household filed in the anchor's category, or with the positive's exact text,
           is masked (Request-Level Dedup: up to 30% of in-batch negatives were positives); at most CAP training anchors per payee string
           (OmniSearchSage's cap on pairs per frequent item)
  logq     in-batch positives' logits minus log of their payee's share of the positives (PinnerFormer's sampling-bias correction)
  pool     POOL extra random negatives per step from all training households, encoded without gradient (OmniSage's random-negative pool)
  hs       a full softmax over one household's categories per step: score(category) = mean of the anchor's 3 best cosines among the
           category's earlier filings (the MaxSim reader itself), /0.05, over HS_B anchors and up to HS_H history rows of that household,
           with gradient through both (TransAct V2: the household's own "seen, not chosen" categories as negatives). Also the row's
           "joint InfoNCE + category head" (UniPinRec) when combined with infonce: categories are per household, so the household's
           categories are the head's classes.
Masked history labels (UniPinRec MAM) need a model that reads a sequence; they move to rows 203 and 207.
  train   ARM=<name> LOSS=infonce,dedup,logq,pool,hs writes models/encoders/hist_r202_<name>
  read    ENCS=<dir,...> top-1 / top-3 / top-10 for kNN and MaxSim, all and first-time payee (READ=households | budget)
env: PAIRS (200000), HOUSEHOLDS (400), BATCH (128), CAP (400), POOL (1024), HS_B (32), HS_H (768), HS_W (1), SEED (0),
     TEST_SEEDS (100000-100099), READ.
usage: SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 ARM=b1 LOSS=infonce,dedup uv run python scripts/hist_train2.py train
"""
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hist_encoder as H  # noqa: E402
import hist_fast as HF  # noqa: E402
from two_tower import households  # noqa: E402

PAIRS, HOUSEHOLDS = int(os.environ.get("PAIRS", "200000")), int(os.environ.get("HOUSEHOLDS", "400"))
BATCH, SEED, CAP = int(os.environ.get("BATCH", "128")), int(os.environ.get("SEED", "0")), int(os.environ.get("CAP", "400"))
POOL, HS_B, HS_H, HS_W = int(os.environ.get("POOL", "1024")), int(os.environ.get("HS_B", "32")), int(os.environ.get("HS_H", "768")), float(os.environ.get("HS_W", "1"))
SCALE = 20.0  # 1 / 0.05, as MNRL and the readers' TAU
PAYEE_MASK = int(os.environ.get("PAYEE_MASK", "0"))  # row 202 b3b: pool negatives with the anchor's or positive's payee are masked


def _triplets(hh, loss, rng):
    """hist_encoder.train1's triplets (positive: same category, half the time another payee; hard negative: the most string-similar
    earlier transaction filed elsewhere), with household index, gold and payees kept for the masks; CAP anchors per payee under dedup"""
    per = -(-PAIRS // len(hh))
    used = Counter()
    rows = []
    for h, ev in enumerate(hh):
        g = [H._grams(e["payee"]) for e in ev]
        cand = [i for i, e in enumerate(ev) if e["day"] > 0]
        for i in rng.sample(cand, min(per, len(cand))):
            e = ev[i]
            if "dedup" in loss and used[e["payee"]] >= CAP:
                continue
            before = [j for j in range(i) if ev[j]["day"] < e["day"]]
            same = [j for j in before if ev[j]["gold"] == e["gold"]]
            other = [j for j in before if ev[j]["gold"] != e["gold"]]
            if not same or not other:
                continue
            diff = [j for j in same if ev[j]["payee"] != e["payee"]]
            pos = rng.choice(diff if diff and rng.random() < 0.5 else same)
            pool = rng.sample(other, min(300, len(other)))
            neg = max(pool, key=lambda j: len(g[i] & g[j]) / (len(g[i] | g[j]) or 1))
            used[e["payee"]] += 1
            rows.append(dict(a=e["text"], p=ev[pos]["text"], n=ev[neg]["text"], h=h, ga=e["gold"], gn=ev[neg]["gold"], pp=ev[pos]["payee"],
                             ap=e["payee"]))
    rng.shuffle(rows)
    return rows


def _cached_triplets(hh, loss, rng):
    """_triplets, cached in data/interim/triplet_cache (2026-10-05: each arm re-mined the same ~200k triplets, ~2.5 min); the key covers
    the households (two_tower's generator key), the settings that change the mining, and this file and hist_encoder.py's code. The rng's
    state after mining is stored with the rows and restored on a hit, so a cached run draws the same pool and hs samples as an uncached one."""
    import hashlib
    import pickle
    from two_tower import _hh_key
    from ai_experiments.paths import ROOT
    h = hashlib.sha1(f"{_hh_key('train')}|{HOUSEHOLDS}|{PAIRS}|{CAP}|{'dedup' in loss}|{SEED}|{H.AMT_TEXT}".encode())
    for f in (__file__, H.__file__):
        h.update(Path(f).read_bytes())
    f = ROOT / "data" / "interim" / "triplet_cache" / f"{h.hexdigest()[:16]}.pkl"
    if f.exists():
        rows, state = pickle.loads(f.read_bytes())
        rng.setstate(state)
        return rows
    rows = _triplets(hh, loss, rng)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps((rows, rng.getstate()), protocol=5))
    return rows


def _enc(model, texts, dev):
    f = model.preprocess(texts)
    f = {k: v.to(dev) if hasattr(v, "to") else v for k, v in f.items()}
    import torch
    return torch.nn.functional.normalize(model(f)["sentence_embedding"].float(), dim=-1)


def _hs_loss(model, ev, rng, dev):
    """the MaxSim reader as a training loss on one household: HS_B anchors, softmax over the categories that have earlier filings among
    up to HS_H sampled history rows"""
    import torch
    cand = [i for i, e in enumerate(ev) if e["day"] > 0]
    anc = rng.sample(cand, min(HS_B, len(cand)))
    top = max(ev[i]["day"] for i in anc)
    hist = [j for j, e in enumerate(ev) if e["day"] < top]
    hist = rng.sample(hist, min(HS_H, len(hist)))
    gold = sorted({ev[j]["gold"] for j in hist})
    ci = {c: k for k, c in enumerate(gold)}
    E = _enc(model, [ev[j]["text"] for j in hist] + [ev[i]["text"] for i in anc], dev)
    Eh, Ea = E[:len(hist)], E[len(hist):]
    S = Ea @ Eh.T
    dh = torch.tensor([ev[j]["day"] for j in hist], device=dev)
    da = torch.tensor([ev[i]["day"] for i in anc], device=dev)
    S = S.masked_fill(da[:, None] <= dh[None, :], float("-inf"))
    gh = torch.tensor([ci[ev[j]["gold"]] for j in hist], device=dev)
    sc = torch.full((len(anc), len(gold)), float("-inf"), device=dev)
    for c in range(len(gold)):
        cols = (gh == c).nonzero().squeeze(1)
        t = S[:, cols].topk(min(3, len(cols)), dim=1).values
        fin = torch.isfinite(t)
        mean = torch.where(fin, t, torch.zeros_like(t)).sum(1) / fin.sum(1).clamp(min=1)
        sc[:, c] = torch.where(fin.any(1), mean, sc[:, c])
    tgt = torch.tensor([ci.get(ev[i]["gold"], -1) for i in anc], device=dev)
    ok = (tgt >= 0) & torch.isfinite(sc[torch.arange(len(anc), device=dev), tgt.clamp(min=0)]) & (torch.isfinite(sc).sum(1) > 1)
    if not ok.any():
        return None
    return torch.nn.functional.cross_entropy(SCALE * sc[ok], tgt[ok])


def train():
    import torch
    from ai_experiments.licences import open_licence
    arm, loss = os.environ["ARM"], set(os.environ["LOSS"].split(","))
    out = H.ENC / f"hist_r202_{arm}"
    if not Path(H.BASE).exists():  # a local encoder dir (row 212: the knowledge stage) was checked when its base was
        open_licence(H.BASE)
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    hh = [H.events(b) for b in households("train", range(HOUSEHOLDS))]
    rows = _cached_triplets(hh, loss, rng)
    steps = len(rows) // BATCH
    print(f"arm {arm} loss {sorted(loss)}: {len(rows)} triplets, {steps} steps ({time.time() - t0:.0f}s to build)", flush=True)
    logq = Counter(r["pp"] for r in rows)
    tot = sum(logq.values())
    every = [(h, k) for h, ev in enumerate(hh) for k in range(len(ev))]
    dev = "cuda"
    model = H._model(H.BASE)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    warm = int(0.05 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / (steps - warm)))
    t0, run = time.time(), defaultdict(float)
    for s in range(steps):
        b = rows[s * BATCH:(s + 1) * BATCH]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            total = torch.zeros((), device=dev)
            if "infonce" in loss:
                A = _enc(model, [r["a"] for r in b], dev)
                C = _enc(model, [r["p"] for r in b] + [r["n"] for r in b], dev)
                L = SCALE * (A @ C.T)
                n = len(b)
                if "logq" in loss:
                    q = torch.tensor([math.log(logq[r["pp"]] / tot) for r in b], device=dev)
                    L = L - torch.cat([q, torch.zeros(n, device=dev)])[None, :]
                if "dedup" in loss:  # a candidate that is in fact a positive for this anchor: same household and category, or the positive's text
                    hh_ = torch.tensor([r["h"] for r in b], device=dev)
                    ga = [hash(r["ga"]) for r in b]
                    gc = torch.tensor(ga + [hash(r["gn"]) for r in b], device=dev)
                    hc = torch.cat([hh_, hh_])
                    tp = [r["p"] for r in b]
                    txt = torch.tensor([[tp[i] == t for t in tp + [r["n"] for r in b]] for i in range(n)], device=dev)
                    fn = ((hh_[:, None] == hc[None, :]) & (torch.tensor(ga, device=dev)[:, None] == gc[None, :])) | txt
                    fn[torch.arange(n), torch.arange(n)] = False
                    L = L.masked_fill(fn, float("-inf"))
                if "pool" in loss:
                    pick = rng.sample(every, POOL)
                    with torch.no_grad():
                        R = _enc(model, [hh[h][k]["text"] for h, k in pick], dev)
                    LR = SCALE * (A @ R.T)
                    if "dedup" in loss:
                        # same household and category, or (row 202 b3b) the anchor's or positive's payee string: in the shared world the same
                        # payee is filed by many households, so a random pool is full of identical strings the loss cannot push apart
                        bad = torch.tensor([[(r["h"] == h and r["ga"] == hh[h][k]["gold"])
                                             or (PAYEE_MASK and hh[h][k]["payee"] in (r["ap"], r["pp"])) for h, k in pick] for r in b], device=dev)
                        LR = LR.masked_fill(bad, float("-inf"))
                    L = torch.cat([L, LR], 1)
                li = torch.nn.functional.cross_entropy(L.float(), torch.arange(n, device=dev))
                total = total + li
                run["infonce"] += li.item()
            if "hs" in loss:
                lh = _hs_loss(model, hh[rng.randrange(len(hh))], rng, dev)
                if lh is not None:
                    total = total + HS_W * lh
                    run["hs"] += lh.item()
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (s + 1) % 200 == 0:
            print(f"  step {s + 1}/{steps} " + " ".join(f"{k} {v / 200:.3f}" for k, v in run.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
            run.clear()
    model.save(str(out))
    print(f"-> {out} (train {time.time() - t0:.0f}s)", flush=True)


def read():
    """kNN and MaxSim (hist_fast, the same readers as rows 194-201) per encoder: % top-1 / top-3 / top-10, all and first-time payee"""
    budget = os.environ.get("READ", "households") == "budget"
    if budget:
        import real_budget_eval as RB
        budgets, cache = [json.loads(RB.CACHE.read_text())["budget"]], RB.OUT / "hist_cache"
    else:
        a, z = map(int, os.environ.get("TEST_SEEDS", "100000-100099").split("-"))
        budgets, cache = list(households("test", range(a, z + 1))), HF.SYN_CACHE
    evs = [H.events(b) for b in budgets]
    print(f"\n**{'owner budget' if budget else f'{len(budgets)} held-out households'}: % top-1 / top-3 / top-10**\n")
    print("| encoder | reader | all | first-time payee |\n|---|---|---|---|")
    for enc in os.environ["ENCS"].split(","):
        d = H.ENC / enc
        m = H._model(d)
        ranks = defaultdict(lambda: defaultdict(list))
        for ev in evs:
            sc = HF.scores(ev, m, d, cache)
            ci = {c: k for k, c in enumerate(sc["cats"])}
            for i, e in enumerate(ev):
                if e["gold"] not in e["state"]:
                    continue
                ks = np.array([ci[c] for c in e["state"]])
                g = ci[e["gold"]]
                for r in ("1 knn", "5 maxsim"):
                    v = sc[r][i, ks]
                    rk = int((v >= sc[r][i, g]).sum())  # ties count against the gold (the gold itself is one of them)
                    for grp in ("all",) + (() if e["seen"] else ("first-time",)):
                        ranks[r][grp].append(rk)
        for r, st in ranks.items():
            cells = [" / ".join(f"{100 * (np.array(st[g]) <= k).mean():.1f}" for k in (1, 3, 10)) + f" (n={len(st[g])})" for g in ("all", "first-time")]
            print(f"| {enc} | {r[2:]} | " + " | ".join(cells) + " |", flush=True)
        del m


if __name__ == "__main__":
    {"train": train, "read": read}[sys.argv[1]]()
