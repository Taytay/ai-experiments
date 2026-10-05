"""PLAN step 194 (owner, 2026-10-04: "Is there a way to embed the historical similar transactions when generating and training
embeddings?"; "I'd like you to try all 3 approaches ... use this either on its own, or to enhance an LLM prediction"). Three ways to put
a household's own history into an embedding categoriser, against row 187's two-tower (category = "Group: Name; recently: 3 payees"):
  1 knn    transaction <-> transaction: one small open encoder (BASE, bge-small, MIT) trained with InfoNCE so a transaction lands near the
           household's earlier transactions filed in the same category (positive: half the time a different payee) and away from the
           most string-similar earlier transaction filed elsewhere (hard negative); read by letting the K nearest earlier transactions
           vote (sum of exp(cos / TAU_KNN) per category).
  2 proto  the same encoder; a category is the mean embedding of everything filed in it so far (its "Group: Name" text when empty);
           read by cosine.
  3 ctx    the query carries its history: "<transaction>" plus its 5 nearest earlier transactions by encoder 1 with where they went
           ("Similar earlier: <payee> $<amount> -> <Group: Name>; ..."), against row 187's category text; a second encoder trained like
           two_tower.py (gold category, another category that one of the neighbours went to as the hard negative).
Only filings before the transaction's day are visible (as two_tower.replay). Training: shared-world v4 households (SHARED_WORLD=1
REALSTYLE_V4=1, train split); reading: held-out test-world households (READ=households) or the owner's budget on this machine
(READ=budget; aggregates only). On the owner's budget each reader is also fused with decider's scores (FUSE: real_budget_eval.py score
files, comma list of tags, read with items_grp_sim2.json's "Group: Name" options): log p_decider + W * log softmax(encoder scores) over
decider's options, W in WS; and gated: the encoder alone where its top probability is at least G (GATES), decider elsewhere, with the
share the encoder decided; decider alone on the same items.
  train1 / train3   write models/encoders/<OUT1> / <OUT3> (DVC)
  read              prints % top-1 / top-3 / top-10 per reader, by payee filed before or not
env: BASE, PAIRS (200000), HOUSEHOLDS (400), BATCH (128), OUT1 (hist_knn_v1), OUT3 (hist_ctx_v1), K (20), TAU_KNN (0.05), TAU (0.05),
     READ (households | budget), TEST_SEEDS (100000-100049), FUSE, WS (0.25,0.5,1,2), GATES (0.95,0.9,0.8), SEED (0).
usage: SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 uv run python scripts/hist_encoder.py train1|train3|read
"""
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from two_tower import WD, _clean, households  # noqa: E402

from ai_experiments.paths import ROOT  # noqa: E402

BASE = os.environ.get("BASE", "BAAI/bge-small-en-v1.5")
PAIRS, HOUSEHOLDS = int(os.environ.get("PAIRS", "200000")), int(os.environ.get("HOUSEHOLDS", "400"))
BATCH, SEED = int(os.environ.get("BATCH", "128")), int(os.environ.get("SEED", "0"))
ENC = ROOT / "models" / "encoders"
OUT1, OUT3 = ENC / os.environ.get("OUT1", "hist_knn_v1"), ENC / os.environ.get("OUT3", "hist_ctx_v1")
K, TAU_KNN, TAU = int(os.environ.get("K", "20")), float(os.environ.get("TAU_KNN", "0.05")), float(os.environ.get("TAU", "0.05"))
NCTX = 5


def events(b):
    """A budget's outflows in date order (as two_tower.replay): dicts with id, day (index), text, payee, amt, gold (category id), label
    (the gold's "Group: Name"), state ({visible category id: category text as of that day}), labels ({id: "Group: Name"}), seen."""
    import datetime as dt
    groups = {g["id"]: g["name"] for g in b["category_groups"]}
    cats = {c["id"]: c for c in b["categories"] if not c.get("deleted")}
    payees = {p["id"]: p["name"] for p in b["payees"]}
    split_parents = {s["transaction_id"] for s in b.get("subtransactions", []) if not s.get("deleted")}
    label = {cid: f'{_clean(groups.get(c["category_group_id"], ""))}: {_clean(c["name"])}' for cid, c in cats.items()
             if groups.get(c["category_group_id"]) != "Internal Master Category"}
    visible = [cid for cid in label if not cats[cid].get("hidden")]
    txs = sorted((t for t in b["transactions"] if not t.get("deleted") and t.get("approved") and t.get("category_id") in label
                  and not t.get("transfer_account_id") and t["id"] not in split_parents and t["amount"] < 0), key=lambda t: (t["date"], t["id"]))
    recent, seen, out, day, i = defaultdict(list), set(), [], -1, 0
    while i < len(txs):
        d = txs[i]["date"]
        j = i
        while j < len(txs) and txs[j]["date"] == d:
            j += 1
        day += 1
        state = {cid: label[cid] + ("; recently: " + ", ".join(recent[cid][:3]) if recent[cid] else "; nothing filed yet") for cid in visible}
        wd = WD[dt.date.fromisoformat(d).weekday()]
        for t in txs[i:j]:
            p = _clean(payees.get(t.get("payee_id"), "") or t.get("import_payee_name_original") or "")
            amt = -t["amount"] / 1000
            out.append(dict(id=t["id"], day=day, text=f"{p} | ${amt:.2f} | {wd}", payee=p, amt=amt, gold=t["category_id"],
                            label=label[t["category_id"]], state=state, labels=label, seen=p in seen))
        for t in txs[i:j]:
            p = _clean(payees.get(t.get("payee_id"), "") or "")
            recent[t["category_id"]] = [p] + [x for x in recent[t["category_id"]] if x != p][:4]
            seen.add(p)
        i = j
    return out


def _grams(s):
    s = f"  {s.lower()} "
    return {s[k:k + 3] for k in range(len(s) - 2)}


def _model(path_or_name):
    import torch
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(str(path_or_name), device="cuda" if torch.cuda.is_available() else "cpu")


def _fit(rows, out):
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
    from ai_experiments.licences import open_licence
    open_licence(BASE)
    torch.manual_seed(SEED)
    print(f"{len(rows)} triplets; e.g. {rows[0]}", flush=True)
    model = _model(BASE)
    ds = Dataset.from_dict({"anchor": [r[0] for r in rows], "positive": [r[1] for r in rows], "negative": [r[2] for r in rows]})
    args = SentenceTransformerTrainingArguments(output_dir=str(out) + "_ckpt", num_train_epochs=1, per_device_train_batch_size=BATCH,
                                                learning_rate=5e-5, warmup_steps=0.05, bf16=torch.cuda.is_available(), seed=SEED,
                                                logging_steps=200, save_strategy="no", report_to=[])
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=losses.MultipleNegativesRankingLoss(model)).train()
    model.save(str(out))
    print(f"-> {out}", flush=True)


def train1():
    rng = random.Random(SEED)
    per = -(-PAIRS // HOUSEHOLDS)
    rows = []
    for b in households("train", range(HOUSEHOLDS)):
        ev = events(b)
        g = [_grams(e["payee"]) for e in ev]
        cand = [i for i, e in enumerate(ev) if e["day"] > 0]
        for i in rng.sample(cand, min(per, len(cand))):
            e = ev[i]
            before = [j for j in range(i) if ev[j]["day"] < e["day"]]
            same = [j for j in before if ev[j]["gold"] == e["gold"]]
            other = [j for j in before if ev[j]["gold"] != e["gold"]]
            if not same or not other:
                continue
            diff = [j for j in same if ev[j]["payee"] != e["payee"]]
            pos = rng.choice(diff if diff and rng.random() < 0.5 else same)
            pool = rng.sample(other, min(300, len(other)))
            neg = max(pool, key=lambda j: len(g[i] & g[j]) / (len(g[i] | g[j]) or 1))
            rows.append((e["text"], ev[pos]["text"], ev[neg]["text"]))
    rng.shuffle(rows)
    _fit(rows, OUT1)


def _embed(model, texts, cache):
    new = [t for t in dict.fromkeys(texts) if t not in cache]
    if new:
        for t, v in zip(new, model.encode(new, normalize_embeddings=True, batch_size=512, show_progress_bar=False)):
            cache[t] = v
    return np.stack([cache[t] for t in texts]) if texts else np.zeros((0, 384), dtype=np.float32)


def _neighbours(E, ev, n):
    """for each event, the indices of its n nearest earlier-day events by cosine (E normalised rows)"""
    import torch
    X = torch.tensor(E, device="cuda" if torch.cuda.is_available() else "cpu")
    day = torch.tensor([e["day"] for e in ev], device=X.device)
    out = []
    for a in range(0, len(ev), 1024):
        S = X[a:a + 1024] @ X.T
        S[day[a:a + 1024, None] <= day[None, :]] = -2
        v, ix = S.topk(min(n, len(ev)), dim=1)
        out += [[(int(j), float(s)) for s, j in zip(vr, ir) if s > -2] for vr, ir in zip(v.tolist(), ix.tolist())]
    return out


def _ctx(e, ev, nb):
    return e["text"] + "\nSimilar earlier: " + "; ".join(f'{ev[j]["payee"]} ${ev[j]["amt"]:.2f} -> {ev[j]["label"]}' for j, _ in nb[:NCTX]) if nb else e["text"]


def train3():
    rng = random.Random(SEED + 3)
    m1, cache = _model(OUT1), {}
    per = -(-PAIRS // HOUSEHOLDS)
    rows = []
    for n, b in enumerate(households("train", range(HOUSEHOLDS))):
        ev = [e for e in events(b) if e["gold"] in e["state"] and len(e["state"]) > 1]
        nb = _neighbours(_embed(m1, [e["text"] for e in ev], cache), ev, NCTX)
        for i in rng.sample(range(len(ev)), min(per, len(ev))):
            e = ev[i]
            hard = [ev[j]["gold"] for j, _ in nb[i] if ev[j]["gold"] != e["gold"] and ev[j]["gold"] in e["state"]]
            neg = hard[0] if hard and rng.random() < 0.7 else rng.choice([c for c in e["state"] if c != e["gold"]])
            rows.append((_ctx(e, ev, nb[i]), e["state"][e["gold"]], e["state"][neg]))
        if n % 100 == 0:
            print(f"  {n} households", flush=True)
            cache.clear()
    rng.shuffle(rows)
    _fit(rows, OUT3)


def _scores(ev, m1, m3, mt):
    """per event: {reader: {category id: score}} (log-domain, higher is better) over its visible categories"""
    c1, c3, ct = {}, {}, {}
    E = _embed(m1, [e["text"] for e in ev], c1)
    nb = _neighbours(E, ev, max(K, NCTX))
    Q3 = _embed(m3, [_ctx(e, ev, nb[i]) for i, e in enumerate(ev)], c3)
    Qt = _embed(mt, [e["text"] for e in ev], ct)
    out = []
    sums, cnt = defaultdict(lambda: np.zeros(E.shape[1], dtype=np.float64)), defaultdict(int)
    i = 0
    while i < len(ev):
        j = i
        while j < len(ev) and ev[j]["day"] == ev[i]["day"]:
            j += 1
        state = ev[i]["state"]
        ids = list(state)
        txt = [state[c] for c in ids]
        C3, Ct = _embed(m3, txt, c3), _embed(mt, txt, ct)
        P = np.stack([sums[c] / np.linalg.norm(sums[c]) if cnt[c] else _embed(m1, [ev[i]["labels"][c]], c1)[0] for c in ids])
        for a in range(i, j):
            knn = defaultdict(float)
            for k, s in nb[a][:K]:
                knn[ev[k]["gold"]] += float(np.exp((s - 1) / TAU_KNN))
            r = {"1 knn": {c: np.log(knn[c] + 1e-9) for c in ids},
                 "2 proto": dict(zip(ids, (P @ E[a]) / TAU)),
                 "3 ctx": dict(zip(ids, (C3 @ Q3[a]) / TAU)),
                 "two-tower (row 187)": dict(zip(ids, (Ct @ Qt[a]) / TAU))}
            r["1+2+3"] = {c: sum(_lsm(r[m])[c] for m in ("1 knn", "2 proto", "3 ctx")) for c in ids}
            out.append(r)
        for a in range(i, j):
            sums[ev[a]["gold"]] += E[a]; cnt[ev[a]["gold"]] += 1
        i = j
    return out


def _lsm(d):
    v = np.array(list(d.values()))
    v = v - v.max()
    v = v - np.log(np.exp(v).sum())
    return dict(zip(d, v))


def _rank(d, gold):
    return int(sum(1 for c, s in d.items() if s >= d[gold] and c != gold)) + 1  # ties count against the gold


def read():
    m1, m3, mt = _model(OUT1), _model(OUT3), _model(ENC / "two_tower_v1")
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
        sc = _scores(ev, m1, m3, mt)
        for e, r in zip(ev, sc):
            if e["gold"] not in e["state"]:
                continue
            for m, d in r.items():
                for g in ("all", "payee filed before" if e["seen"] else "first-time payee"):
                    stats[m][g].append(_rank(d, e["gold"]))
            allev.append(e); allsc.append(r)
    print(f"\n**{'owner budget' if budget else 'held-out households'}: right category, % top-1 / top-3 / top-10 (n per group)**\n")
    groups = ["all", "payee filed before", "first-time payee"]
    print("| reader | " + " | ".join(f"{g} (n={len(stats['1 knn'][g])})" for g in groups) + " |"); print("|---|---|---|---|")
    for m, st in stats.items():
        cells = []
        for g in groups:
            r = np.array(st[g])
            cells.append(f"{100 * (r <= 1).mean():.1f} / {100 * (r <= 3).mean():.1f} / {100 * (r <= 10).mean():.1f}")
        print(f"| {m} | " + " | ".join(cells) + " |")
    if budget and os.environ.get("FUSE"):
        fuse(allev, allsc)


def fuse(ev, sc):
    """decider's log-probs over its options plus W x each reader's log softmax over the same options (options the encoder cannot score,
    such as hidden categories and Ready to Assign, get the reader's lowest score)"""
    import real_budget_eval as RB
    ws = [float(w) for w in os.environ.get("WS", "0.25,0.5,1,2").split(",")]
    gates = [float(g) for g in os.environ.get("GATES", "0.95,0.9,0.8").split(",")]
    for tag in os.environ["FUSE"].split(","):
        f = RB.OUT / f"scores_{tag}_split_grp_sim2.jsonl"  # the items file that goes with it carries the same "Group: Name" options
        items = {it["id"]: it for it in json.loads((RB.OUT / "items_grp_sim2.json").read_text())["items"] if it["answer"] >= 0}
        lp = {r["id"]: np.asarray(r["lp"]) for r in map(json.loads, open(f))}
        rows = defaultdict(lambda: defaultdict(list))
        for e, r in zip(ev, sc):
            it = items.get(e["id"])
            if not it or e["id"] not in lp:
                continue
            by_label = {e["labels"][c]: c for c in r["1 knn"]}
            groups = ("all", "payee filed before" if it["payee_seen"] else "first-time payee")
            L = lp[e["id"]]
            for g in groups:
                rows["decider alone"][g].append(int(np.argmax(L)) == it["answer"])
            hit = sum(o in by_label for o in it["options"])
            rows["(options the encoder can score)"]["all"].append(hit / len(it["options"]))
            for m, d in r.items():
                s = _lsm(d)
                lo = min(s.values())
                v = np.array([s.get(by_label.get(o), lo) for o in it["options"]])
                for g in groups:
                    rows[f"{m} alone"][g].append(int(np.argmax(v)) == it["answer"])
                for w in ws:
                    z = L + w * v
                    for g in groups:
                        rows[f"decider + {w} x {m}"][g].append(int(np.argmax(z)) == it["answer"])
                pmax = float(np.exp(v.max()))
                for t in gates:  # the encoder answers alone when it is this sure; decider reads the rest
                    pick = int(np.argmax(v)) if pmax >= t else int(np.argmax(L))
                    for g in groups:
                        rows[f"gate {t}: {m}, else decider"][g].append(pick == it["answer"])
                        rows[f"gate {t}: {m}, share decided by the encoder"][g].append(pmax >= t)
        print(f"\n**owner budget, fused with decider {tag}: % right first (outflows in visible categories that decider also read)**\n")
        print("| reader | all | payee filed before | first-time payee |"); print("|---|---|---|---|")
        for m, st in rows.items():
            print(f"| {m} | " + " | ".join(f"{100 * np.mean(st[g]):.1f} (n={len(st[g])})" if st[g] else "" for g in ("all", "payee filed before", "first-time payee")) + " |")


if __name__ == "__main__":
    {"train1": train1, "train3": train3, "read": read}[sys.argv[1]]()
