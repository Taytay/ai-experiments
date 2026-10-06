"""Row 217 (2026-10-06): blind_v2's users as whole budgets, a transfer test for the encoder family. Row 216 found the late-interaction
model's synthetic score (v4 held-out households, our own generator) does not predict the owner's budget: c0 led a5 by 0.5 on decider's
synthetic items but trailed it by 3.7 on the owner's budget. blind_v2 (scripts/build_blind_v2.py) was written by an agent that saw
none of the repo's generators: its own merchants, bank renderings, category schemes (standard, personal, merged, coined names), trips,
refunds, changes of mind and filing mistakes. Its items keep only a history slice per query; here its generator is run again (same seed,
same users) and each user's full year becomes a YNAB-shaped budget that hist_encoder.events reads: one category group, every category
the user ever had (visible), one payee per distinct statement string, outflows as negative milliunits. Built from the generator's own
structures, never from rendered prompts. Cached in data/interim/blind2_budgets.pkl (keyed by the builder's code).
usage: uv run python scripts/blind_budgets.py   (prints counts)
"""
import hashlib
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ai_experiments.paths import ROOT  # noqa: E402


def _build():
    import build_blind_v2 as B
    rng = random.Random(B.SEED)  # main()'s archetype draw, unchanged
    archs = [B.wchoice(rng, B.ARCH_W) for _ in range(B.N_USERS - B.N_NEW_USERS)] + ["new_user"] * B.N_NEW_USERS
    rng.shuffle(archs)
    out = []
    for uid, a in enumerate(archs):
        u = B.build_user(uid, a, a == "new_user")
        cats = [dict(id=f"c{c}", name=u.cats[c].name, category_group_id="g", hidden=False, deleted=False) for c in u.order]
        payees, txs = {}, []
        for k, e in enumerate(u.events):
            pid = payees.setdefault(e.text, f"p{len(payees)}")
            txs.append(dict(id=f"b2-{uid}-{k:05d}", date=e.date.isoformat(), amount=-round(e.amount * 1000), category_id=f"c{e.filed}",
                            payee_id=pid, approved=True, deleted=False))
        out.append(dict(id=f"blind2-{uid}", new_user=u.is_new, category_groups=[dict(id="g", name="Categories")], categories=cats,
                        payees=[dict(id=p, name=t) for t, p in payees.items()], transactions=txs, subtransactions=[]))
    return out


def budgets():
    """the 250 blind_v2 users as budgets (cached)"""
    h = hashlib.sha1((Path(__file__).parent / "build_blind_v2.py").read_bytes() + Path(__file__).read_bytes()).hexdigest()[:12]
    f = ROOT / "data" / "interim" / f"blind2_budgets_{h}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    out = _build()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(out))
    return out


if __name__ == "__main__":
    import hist_encoder as H
    bs = budgets()
    ev = [H.events(b) for b in bs]
    n = sum(map(len, ev))
    print(f"{len(bs)} budgets, {n} outflows ({sum(len(e) for e, b in zip(ev, bs) if b['new_user'])} from new users), "
          f"{sum(not x['seen'] for e in ev for x in e)} first-time payee strings, categories per budget {sum(len(b['categories']) for b in bs) / len(bs):.1f}")
    print("e.g.", ev[3][200]["text"], "->", ev[3][200]["label"])
