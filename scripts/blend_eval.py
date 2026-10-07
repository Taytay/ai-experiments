"""Row 239 (b) (owner, 2026-10-07: "Go for it"; ContextGNN, 2411.19513: a local pair-wise score where the user's own history reaches, a
tower score beyond it): two late-interaction readers blended per transaction, log p = w * log p_A + (1 - w) * log p_B (each a softmax
over the day's options), with one weight for payees seen before and one for first-time payee strings. The weights are chosen on v5
held-out households (every READ_EVERY-th transaction, the same ones for both readers), then the owner's budget is read once with them
(decider's 19,093 items, as hist_agree.py; the whole weight grid printed as a diagnostic). Owner's data local, aggregates only.
env: A (li_r227_fcr), B (li_r236_g2cos), TEST_SEEDS (100000-100019), READ_EVERY (5), BUDGET, DEC_TAGS (r231-dv5-s0,r231-dv5-s1),
  PARTS (syn,own: which reads to run; each is saved and reused, so the synthetic reads can run on Modal (PARTS=syn SYN_DIR=results/blend)
  and the owner's locally (PARTS=own), then a run with PARTS= fuses), SYN_DIR (data/interim/blend; owner reads under the private real_budget_eval.OUT/blend)
usage (EmbeddingGemma 2 needs the overlay): SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 RS_V5=1 OPTS=span BUDGET=<id> \
  uv run --with "sentence-transformers>=6.1.0" --with "transformers>=5.18" --with torch==2.13.0 --with torchvision==0.28.0 python scripts/blend_eval.py
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hist_agree as HA  # noqa: E402
import hist_encoder as H  # noqa: E402
import li_decider as LD  # noqa: E402
from two_tower import households  # noqa: E402

W = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]


def lsm(v):
    v = np.asarray(v, dtype=np.float64)
    return v - (np.log(np.exp(v - v.max()).sum()) + v.max())


def synthetic(run, budgets):
    """per scored event: (log p over its options in state order, gold index, first-time?)"""
    out = []
    for b in budgets:
        ev, sco = run(b)
        for e, sc in zip(ev, sco):
            if e["gold"] in sc:
                ids = list(sc)
                out.append((lsm([sc[c] for c in ids]), ids.index(e["gold"]), not e["seen"]))
    return out


def owner(run, out1):
    import real_budget_eval as RB
    oitems = {it["id"]: it for it in json.loads((RB.OUT / "items_grp_sim2.json").read_text())["items"] if it["answer"] >= 0}
    odec = [{r["id"]: np.asarray(r["lp"]) for r in map(json.loads, open(RB.OUT / f"scores_{t}_split_grp_sim2.jsonl"))}
            for t in os.environ.get("DEC_TAGS", "r231-dv5-s0,r231-dv5-s1").split(",")]
    ob = [json.loads(RB.CACHE.read_text())["budget"]]
    H.OUT1 = out1
    HA.READERS = {"LI": "LI"}
    recs = HA.records(ob, oitems, odec, lambda b, e: e["id"], RB.OUT / "hist_cache", run)
    return [(r["enc"]["LI"], r["answer"], r["known"] == "first-time") for r in recs]


def acc(a, b, w_seen, w_first, first=None):
    ok = []
    for (la, g, f), (lb, _, _) in zip(a, b):
        if first is not None and f != first:
            continue
        w = w_first if f else w_seen
        ok.append(int(np.argmax(w * la + (1 - w) * lb)) == g)
    return 100 * np.mean(ok), len(ok)


def main():
    names = [os.environ.get("A", "li_r227_fcr"), os.environ.get("B", "li_r236_g2cos")]
    a_, z_ = map(int, os.environ.get("TEST_SEEDS", "100000-100019").split("-"))
    os.environ.setdefault("READ_EVERY", "5")
    budgets = list(households("test", range(a_, z_ + 1)))
    out1 = H.OUT1
    parts = os.environ.get("PARTS", "syn,own").split(",")
    sdir = Path(os.environ.get("SYN_DIR", "data/interim/blend"))
    odir = None
    sdir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("BUDGET"):  # real_budget_eval needs BUDGET; the owner's reads stay in its private folder
        import real_budget_eval as RB
        odir = RB.OUT / "blend"
        odir.mkdir(parents=True, exist_ok=True)
    sf = lambda n: sdir / f"{n}_{a_}-{z_}_every{os.environ['READ_EVERY']}.pkl"
    of = lambda n: odir / f"{n}.pkl"
    syn = {n: pickle.loads(sf(n).read_bytes()) for n in names if sf(n).exists()}
    own = {n: pickle.loads(of(n).read_bytes()) for n in names if of(n).exists() and os.environ.get("BUDGET")}
    for n in names:
        need_s, need_o = "syn" in parts and n not in syn, "own" in parts and n not in own and os.environ.get("BUDGET")
        if not (need_s or need_o):
            continue
        # one reader at a time: LD.load sets module settings the reader's run reads
        import gc
        import torch
        LD.SPAN_AFTER = 365
        H.OUT1 = out1
        run = LD.load(n)
        if need_s:
            syn[n] = synthetic(run, budgets); sf(n).write_bytes(pickle.dumps(syn[n]))
        if need_o:
            rd = os.environ.pop("READ_EVERY")  # the owner's budget in full
            own[n] = owner(run, out1); of(n).write_bytes(pickle.dumps(own[n]))
            os.environ["READ_EVERY"] = rd
        del run  # the next reader needs the GPU (EmbeddingGemma 2 beside fcr ran out of memory on the RTX 3090)
        gc.collect(); torch.cuda.empty_cache()
        print(f"{n}: {len(syn.get(n, []))} synthetic events, {len(own.get(n, []))} owner items", flush=True)
    if not all(n in syn for n in names):
        return
    A, B = (syn[n] for n in names)
    assert len(A) == len(B) and all(x[1] == y[1] for x, y in zip(A, B)), "the two readers' synthetic events differ"
    best = {}
    for seg, first in (("seen", False), ("first-time", True)):
        grid = [(acc(A, B, w, w, first)[0], w) for w in W]
        best[seg] = max(grid)[1]
        print(f"v5 {seg}: " + ", ".join(f"w={w} {s:.1f}" for s, w in grid), flush=True)
    print(f"chosen on v5: weight on {names[0]} {best['seen']} for payees seen before, {best['first-time']} for first-time payees", flush=True)
    if own:
        OA, OB = (own[n] for n in names)
        print(f"\n| owner's budget | all | first-time | seen before |\n|---|---|---|---|")
        for label, ws, wf in ((names[0], 1, 1), (names[1], 0, 0), ("blend, chosen on v5", best["seen"], best["first-time"])):
            print(f"| {label} (w {ws} / {wf}) | {acc(OA, OB, ws, wf)[0]:.1f} | {acc(OA, OB, ws, wf, True)[0]:.1f} | {acc(OA, OB, ws, wf, False)[0]:.1f} |")
        for seg, first in (("seen", False), ("first-time", True)):  # diagnostic only, not for choosing
            print(f"owner {seg} (diagnostic): " + ", ".join(f"w={w} {acc(OA, OB, w, w, first)[0]:.1f}" for w in W))


if __name__ == "__main__":
    main()
