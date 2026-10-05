"""PLAN step 201 (owner, 2026-10-05: "Yes"): where decider and the history encoders disagree. Fusion paid at Pinterest because two
retrievers' candidates overlapped 3.2% (Multi-Embedding Retrieval, references/papers/2506.23060); ours gave +0.3 (§178 Fair fusion and
amounts). Per transaction: decider right / encoder right / both / neither, for kNN and MaxSim (hist_encoder2 readers), split by how
well the household knows the payee (earlier filings of the payee: 0, 1-3, 4+), by whether the payee was already filed under 2+
categories, and by the size of the gold category (earlier filings in it); each reader's top probability on the cases only it got
right; and three ceilings over decider alone: "either right" (an oracle), "higher top probability wins", and a gate chosen on the
synthetic set (encoder when its top probability >= G and decider's <= D). Synthetic: TEST_SEEDS held-out v4 households read by decider
G4 (realstyle_v4g_ngram_test, seeds 0 and 1). Owner's budget (BUDGET set): decider G4 r190 seeds 0 and 1, items_grp_sim2.json; private,
aggregates only.
env: TEST_SEEDS (100000-100099), BUDGET, ENCS (encoders, one run; li_r* names are late-interaction decision models, read by
li_decider.load as one reader "LI", with the reading time per transaction), OWNER_ENCS.
usage: SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 [BUDGET=<id>] uv run python scripts/hist_agree.py
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hist_encoder as H  # noqa: E402
import hist_encoder2 as H2  # noqa: E402
import hist_fast as HF  # noqa: E402
import hist_fuse as F  # noqa: E402
from two_tower import households  # noqa: E402

from ai_experiments.paths import PROCESSED, ROOT  # noqa: E402

READERS = {"kNN": "1 knn", "MaxSim": "5 maxsim"}
GS, DS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], [0.3, 0.5, 0.7, 0.9, 1.01]


def lsm(v):
    v = np.asarray(v, dtype=np.float64)
    v = v - v.max()
    return v - np.log(np.exp(v).sum())


def records(budgets, items, dec, key, cache_dir, li=None):
    """one record per matched item: decider log-probs per seed, encoder log-probs per reader (over decider's options), answer, segments;
    reader scores from hist_fast (GPU, cached per budget in cache_dir), or from li (li_decider.load's run) as the one reader LI"""
    import time
    m1 = None if li else H._model(H.OUT1)
    out, secs, n = [], 0.0, 0
    for b in budgets:
        if li:
            t0 = time.time()
            ev, s = li(b)
            secs += time.time() - t0; n += len(ev)
            sc = [{"LI": x} for x in s]
        else:
            ev = H2.events(b)
            sc = HF.as_dicts(HF.scores(ev, m1, H.OUT1, cache_dir), ev)
        n_payee, cats_payee, n_cat = defaultdict(int), defaultdict(set), defaultdict(int)
        i = 0
        while i < len(ev):
            j = i
            while j < len(ev) and ev[j]["day"] == ev[i]["day"]:
                j += 1
            for a in range(i, j):
                e, r = ev[a], sc[a]
                it = items.get(key(b, e))
                if it and it["id"] in dec[0] and it["id"] in dec[1]:
                    by = {e["labels"][c]: c for c in next(iter(r.values()))}
                    opts = [o.strip() for o in it["options"]]
                    enc = {}
                    for name, rk in READERS.items():
                        raw = np.array([r[rk].get(by[o], np.nan) if o in by else np.nan for o in opts], dtype=np.float64)
                        ok = ~np.isnan(raw)
                        v = np.full(len(raw), -1e9)
                        if ok.any():
                            v[ok] = lsm(raw[ok])
                            v[~ok] = v[ok].min()
                        enc[name] = v
                    k = n_payee[e["payee"]]
                    out.append(dict(dec=[lsm(d[it["id"]]) for d in dec], enc=enc, answer=it["answer"],
                                    known="first-time" if k == 0 else "filed 1-3 times" if k <= 3 else "filed 4+ times",
                                    multi=len(cats_payee[e["payee"]]) >= 2,
                                    size="gold category new" if n_cat[e["gold"]] == 0 else "gold category 1-9 filings" if n_cat[e["gold"]] < 10
                                    else "gold category 10-99" if n_cat[e["gold"]] < 100 else "gold category 100+"))
            for a in range(i, j):
                e = ev[a]
                n_payee[e["payee"]] += 1
                cats_payee[e["payee"]].add(e["gold"])
                n_cat[e["gold"]] += 1
            i = j
    if li:
        print(f"LI read {n} transactions in {secs:.1f} s ({1000 * secs / max(n, 1):.2f} ms each, with the documents and the query's neighbours)", flush=True)
    return out


def pick_gate(rows, name, s):
    """encoder when its top probability >= G and decider's <= D, else decider"""
    def acc(g, d):
        return np.mean([(int(np.argmax(r["enc"][name])) if (np.exp(r["enc"][name].max()) >= g and np.exp(r["dec"][s].max()) <= d)
                         else int(np.argmax(r["dec"][s]))) == r["answer"] for r in rows])
    return max(((g, d) for g in GS for d in DS), key=lambda gd: acc(*gd))


def table(rows, title, gates=None):
    print(f"\n**{title}: % of items, mean over decider seeds 0 and 1 (n)**\n")
    for name in READERS:
        print(f"\n*{name}*\n")
        print("| segment | n | decider | " + name + " | both | only decider | only " + name + " | neither | either (oracle) | higher top-prob wins | gate |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        segs = [("all", lambda r: True)] + [(k, (lambda k: lambda r: r["known"] == k)(k)) for k in ("first-time", "filed 1-3 times", "filed 4+ times")]
        segs += [("payee under 2+ categories", lambda r: r["multi"])]
        segs += [(k, (lambda k: lambda r: r["size"] == k)(k)) for k in ("gold category new", "gold category 1-9 filings", "gold category 10-99", "gold category 100+")]
        for seg, sel in segs:
            sub = [r for r in rows if sel(r)]
            if not sub:
                continue
            cells = defaultdict(list)
            for s in (0, 1):
                d = np.array([int(np.argmax(r["dec"][s])) == r["answer"] for r in sub])
                e = np.array([int(np.argmax(r["enc"][name])) == r["answer"] for r in sub])
                hp = np.array([(int(np.argmax(r["enc"][name])) if r["enc"][name].max() > r["dec"][s].max() else int(np.argmax(r["dec"][s]))) == r["answer"] for r in sub])
                cells["dec"].append(d.mean()); cells["enc"].append(e.mean()); cells["both"].append((d & e).mean())
                cells["od"].append((d & ~e).mean()); cells["oe"].append((~d & e).mean()); cells["neither"].append((~d & ~e).mean())
                cells["either"].append((d | e).mean()); cells["hp"].append(hp.mean())
                if gates:
                    g, dd = gates[(name, s)]
                    gt = np.array([(int(np.argmax(r["enc"][name])) if (np.exp(r["enc"][name].max()) >= g and np.exp(r["dec"][s].max()) <= dd)
                                    else int(np.argmax(r["dec"][s]))) == r["answer"] for r in sub])
                    cells["gate"].append(gt.mean())
            f = lambda k: f"{100 * np.mean(cells[k]):.1f}" if cells[k] else "-"
            print(f"| {seg} | {len(sub)} | {f('dec')} | {f('enc')} | {f('both')} | {f('od')} | {f('oe')} | {f('neither')} | {f('either')} | {f('hp')} | {f('gate')} |")
        conf = defaultdict(list)
        for r in rows:
            for s in (0, 1):
                d, e = int(np.argmax(r["dec"][s])) == r["answer"], int(np.argmax(r["enc"][name])) == r["answer"]
                k = "only decider" if d and not e else "only encoder" if e and not d else "both" if d else "neither"
                conf[k].append((np.exp(r["dec"][s].max()), np.exp(r["enc"][name].max())))
        print(f"\n{name}: mean top probability (decider / {name}) by outcome: " + "; ".join(
            f"{k} {np.mean([c[0] for c in v]):.2f} / {np.mean([c[1] for c in v]):.2f}" for k, v in conf.items()))


if __name__ == "__main__":
    # ENCS: several encoders in one run (2026-10-05: one process per encoder re-read decider's scores and rebuilt the households each time);
    # default the single OUT1 encoder. OWNER_ENCS: the encoders read on the owner's budget when BUDGET is set (default all of ENCS).
    encs = [x for x in os.environ.get("ENCS", "").split(",") if x] or [H.OUT1.name]
    owner_encs = [x for x in os.environ.get("OWNER_ENCS", "").split(",") if x] or encs
    items = {it["id"]: it for it in json.loads((PROCESSED / "realstyle_v4g_ngram_test.json").read_text())["items"]}
    dec = [{r["id"]: r["sum_lp"] for r in map(json.loads, open(ROOT / F.SYN.format(seed=s)))} for s in ("", "s1")]
    a, z = map(int, os.environ.get("TEST_SEEDS", "100000-100099").split("-"))
    users = [u for u in sorted({it["user"] for it in items.values()}) if a <= u <= z]
    budgets = list(households("test", users))
    if os.environ.get("BUDGET"):
        import real_budget_eval as RB
        oitems = {it["id"]: it for it in json.loads((RB.OUT / "items_grp_sim2.json").read_text())["items"] if it["answer"] >= 0}
        odec = [{r["id"]: np.asarray(r["lp"]) for r in map(json.loads, open(RB.OUT / f"scores_{t}_split_grp_sim2.jsonl"))} for t in ("r190-g-s0", "r190-g-s1")]
        obudget = [json.loads(RB.CACHE.read_text())["budget"]]
    out1 = H.OUT1
    for enc in encs:
        li = None
        if enc.startswith("li_r"):
            import li_decider as LD
            H.OUT1 = out1  # the query's neighbour encoder
            li, READERS = LD.load(enc), {"LI": "LI"}
        else:
            H.OUT1, READERS = H.ENC / enc, {"kNN": "1 knn", "MaxSim": "5 maxsim"}
        print(f"\n=== {enc}", flush=True)
        syn = records(budgets, items, dec, lambda b, e: f"RS:{b['id'].rsplit('-', 1)[1]}:{e['id'][:8]}", HF.SYN_CACHE, li)
        gates = {(name, s): pick_gate(syn, name, s) for name in READERS for s in (0, 1)}
        print(f"gates chosen on the synthetic set (G encoder >=, D decider <=): {gates}")
        table(syn, f"held-out synthetic households ({len(users)})", gates)
        if os.environ.get("BUDGET") and enc in owner_encs:
            own = records(obudget, oitems, odec, lambda b, e: e["id"], RB.OUT / "hist_cache", li)
            table(own, "owner's budget", gates)
