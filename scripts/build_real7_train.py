"""Training episodes for PLAN step 98 stage 2 (REAL-22): REAL-7-style users from seeds disjoint from the test set's (prefix "real7train"),
eight queries each from the second half of their year, no held-out rendering, each prompt's history slice built by one of
`build_real7_slices.POLICIES` (or a mix), with the same past-only guard. The target is the user's intended category (what logged data
would hold), including where the prompt cannot show it (a change not yet visible, an unusual filing of a merchant not in the slice):
the model sees those as real users produce them. Rows filed differently from their intended category (the generator's 3% misfiles) are
marked so the shot-label loss skips them.

Stored structured, so exp_categoriser.py (R7TRAIN) can rename categories consistently: per episode the category list, the rows
(line text, filed label, filed == intended), the query line and the answer.
usage: uv run python scripts/build_real7_train.py <policy|mix> [n_users=300]
"""
import json
import random
import sys
from pathlib import Path

from ai_experiments import real6 as R6
from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
import build_real7  # noqa: E402
import build_real7_slices as S  # noqa: E402

MIX = ["recent", "self", "all_kind", "all_embrec", "all_kind_b48"]

if __name__ == "__main__":
    pol = sys.argv[1]; n_users = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    items, _, hist_of = build_real7.generate(prefix="real7train", n_users=n_users, queries=8, held_share=0.0)
    sims = S.similarity(); rng = random.Random(f"r7train-{pol}"); eps = []
    for it in items:
        name = rng.choice(MIX) if pol == "mix" else pol
        parts, smode, budget, k_self, k_sim, dated = S.POLICIES[name]
        q = hist_of[it["id"]][-1]; opts = [o.strip() for o in it["options"]]
        rows = S.past_only(hist_of[it["id"]][:-1], q, opts)
        pick = [rows[i] for i in S.select(rows, q, parts, sims[smode][q["merchant"]] if smode else {}, budget, k_self, k_sim)]
        qrow = dict(day=q["day"], text=it["text"], amount=it["amount"], weekday=it["weekday"])
        eps.append(dict(id=it["id"], policy=name, cats=opts, rows=[[S.line(r, dated), r["filed"], r["filed"] == r["intended"]] for r in pick],
                        query=S.line(qrow, dated), answer=opts[it["answer"]], why=it["why"]))
    dst = PROCESSED / f"real7train_v1_{pol}.json"
    dst.write_text(json.dumps(dict(name="real7train", version=f"v1_{pol}", n_users=n_users, episodes=eps, sha256=R6.sha256(eps)), indent=0, ensure_ascii=False))
    print(dst.name, len(eps), "episodes")
