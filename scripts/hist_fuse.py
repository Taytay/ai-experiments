"""PLAN step 199 (owner, 2026-10-05: "Proceed with the plan"): decider and the history-aware encoders together, with every setting chosen on
synthetic households and then applied unchanged to the owner's budget (REPORT 177 picked the fusion weight on the owner's budget).
System: where the encoder's top probability is at least G it files alone, elsewhere decider's log-probs plus W x the encoder's log softmax
decide (G = 1 is fusion everywhere; W = 0 with G = 1 is decider alone). Readers from hist_encoder2.py: 1 kNN, 4 TransAct, 5 MaxSim (its
temperature T chosen too: REPORT 177 found its scores too flat to gate).
Choice set: held-out v4 test-world households 100000-100099, decider G4's reads of realstyle_v4g_ngram_test (seeds 0 and 1; 6,000 items,
all kinds; encoder-scored categories only where the encoder can score them, as on the owner's budget). Chosen per reader: W and T by
accuracy (mean of the two seeds); then the largest share decided by the encoder alone at G whose accuracy is within LOSS (0.1 points)
of the fused system. Applied to the owner's budget (decider G4 seeds 0 and 1, items_grp_sim2.json; private, aggregates only).
env: WS (0,0.1,0.25,0.5,0.75,1), TS (0.01,0.02,0.05,0.1), GS (0.6,0.7,0.8,0.9,0.95,0.99,1), LOSS (0.1).
usage: SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 BUDGET=<id> uv run python scripts/hist_fuse.py
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
from two_tower import households  # noqa: E402

from ai_experiments.paths import PROCESSED, ROOT  # noqa: E402

WS = [float(x) for x in os.environ.get("WS", "0,0.1,0.25,0.5,0.75,1").split(",")]
TS = [float(x) for x in os.environ.get("TS", "0.01,0.02,0.05,0.1").split(",")]
GS = [float(x) for x in os.environ.get("GS", "0.6,0.7,0.8,0.9,0.95,0.99,1").split(",")]
LOSS = float(os.environ.get("LOSS", "0.1"))
READERS = ["1 knn", "4 transact", "5 maxsim"]
SYN = ("results/per_item/real6_dm_decider_decider_decider-4b_none_h100bf16st800{seed}_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_"
       "oth50_aux100_labrand255_laylabelled_shots_ev10soft_rs25v4g_lora_realstyle_v4g_ngram_test_labrand255_laylabelled_shots.noctx.jsonl")


def _lsm(v):
    v = np.asarray(v, dtype=np.float64)
    v = v - v.max()
    return v - np.log(np.exp(v).sum())


def rows_for(ev, sc, items, dec, key):
    """per matched item: decider's log softmax, each reader's raw scores over the item's options (nan where the encoder cannot score),
    the answer and whether the payee was filed before"""
    out = []
    for e, r in zip(ev, sc):
        it = items.get(key(e))
        if not it or it["id"] not in dec[0]:
            continue
        by = {e["labels"][c]: c for c in r["1 knn"]}
        opts = [o.strip() for o in it["options"]]
        enc = {m: np.array([r[m].get(by[o], np.nan) if o in by else np.nan for o in opts], dtype=np.float64) for m in READERS}
        out.append(dict(dec=[_lsm(d[it["id"]]) for d in dec], enc=enc, answer=it["answer"], seen=it["payee_seen"]))
    return out


def system(row, s, m, w, t, g):
    """the pick for one decider seed s: reader m with weight w, temperature t (MaxSim only), gate g"""
    raw = row["enc"][m]
    if m == "5 maxsim":
        raw = raw * (H.TAU / t)
    ok = ~np.isnan(raw)
    if not ok.any():
        return int(np.argmax(row["dec"][s])), False
    v = np.full(len(raw), -1e9)
    v[ok] = _lsm(raw[ok])
    v[~ok] = v[ok].min()
    if np.exp(v.max()) >= g:
        return int(np.argmax(v)), True
    return int(np.argmax(row["dec"][s] + w * v)), False


def score(rows, m, w, t, g, sel=None):
    acc, alone = [], []
    for s in (0, 1):
        picks = [system(r, s, m, w, t, g) for r in rows if sel is None or sel(r)]
        acc.append(100 * np.mean([p == r["answer"] for (p, _), r in zip(picks, [r for r in rows if sel is None or sel(r)])]))
        alone.append(100 * np.mean([a for _, a in picks]))
    return acc, alone


if __name__ == "__main__":
    items = {it["id"]: it for it in json.loads((PROCESSED / "realstyle_v4g_ngram_test.json").read_text())["items"]}
    users = sorted({it["user"] for it in items.values()})
    dec = [{r["id"]: r["sum_lp"] for r in map(json.loads, open(ROOT / SYN.format(seed=s)))} for s in ("", "s1")]
    rows = []
    m1, m4, mt = H._model(H.OUT1), H._model(H2.OUT4), H._model(H.ENC / "two_tower_v1")
    for u in users:
        b = next(households("test", [u]))
        ev = H2.events(b)
        rows += rows_for(ev, H2._scores(ev, m1, m4, mt, None), items, dec, lambda e, u=u: f"RS:{u}:{e['id'][:8]}")
    print(f"synthetic choice set: {len(rows)} items matched of {len(items)}", flush=True)
    base = [100 * np.mean([int(np.argmax(r["dec"][s])) == r["answer"] for r in rows]) for s in (0, 1)]
    print(f"\n**synthetic choice set: decider alone {np.mean(base):.2f} [{base[0]:.1f} / {base[1]:.1f}]; chosen settings per reader**\n")
    print("| reader | W | T | fused, all | G | share filed by the encoder alone | gated + fused, all |"); print("|---|---|---|---|---|---|---|")
    chosen = {}
    for m in READERS:
        best = max(((w, t) for w in WS for t in (TS if m == "5 maxsim" else [H.TAU])), key=lambda wt: np.mean(score(rows, m, wt[0], wt[1], 1.0)[0]))
        w, t = best
        fused = np.mean(score(rows, m, w, t, 1.0)[0])
        g = 1.0
        for cand in sorted(GS):
            a, sh = score(rows, m, w, t, cand)
            if np.mean(a) >= fused - LOSS:
                g = cand
                break
        a, sh = score(rows, m, w, t, g)
        chosen[m] = (w, t, g)
        print(f"| {m} | {w} | {t if m == '5 maxsim' else '-'} | {fused:.2f} | {g} | {np.mean(sh):.1f} | {np.mean(a):.2f} |")
    # the owner's budget, settings fixed
    import real_budget_eval as RB
    oitems = {it["id"]: it for it in json.loads((RB.OUT / "items_grp_sim2.json").read_text())["items"] if it["answer"] >= 0}
    odec = [{r["id"]: np.asarray(r["lp"]) for r in map(json.loads, open(RB.OUT / f"scores_{tag}_split_grp_sim2.jsonl"))} for tag in ("r190-g-s0", "r190-g-s1")]
    b = json.loads(RB.CACHE.read_text())["budget"]
    ev = H2.events(b)
    orows = rows_for(ev, H2._scores(ev, m1, m4, mt, None), oitems, odec, lambda e: e["id"])
    groups = {"all": None, "payee filed before": lambda r: r["seen"], "first-time payee": lambda r: not r["seen"]}
    print(f"\n**owner's budget ({len(orows)} outflows in visible categories), settings from the synthetic set: % right first [decider seed 0 / 1]; "
          f"share filed by the encoder alone**\n")
    print("| system | " + " | ".join(groups) + " | share filed by the encoder alone |"); print("|---|---|---|---|---|")
    cells = []
    for name, sel in groups.items():
        sub = [r for r in orows if sel is None or sel(r)]
        a = [100 * np.mean([int(np.argmax(r["dec"][s])) == r["answer"] for r in sub]) for s in (0, 1)]
        cells.append(f"{np.mean(a):.2f} [{a[0]:.1f} / {a[1]:.1f}]")
    print("| decider alone | " + " | ".join(cells) + " | 0 |")
    for m, (w, t, g) in chosen.items():
        for label, gg in (("fused", 1.0), (f"gated at {g} + fused", g)):
            cells, share = [], None
            for name, sel in groups.items():
                a, sh = score(orows, m, w, t, gg, sel)
                cells.append(f"{np.mean(a):.2f} [{a[0]:.1f} / {a[1]:.1f}]")
                if share is None:
                    share = np.mean(sh)
            print(f"| decider + {w} x {m}{f' (T {t})' if m == '5 maxsim' else ''}, {label} | " + " | ".join(cells) + f" | {share:.1f} |")
