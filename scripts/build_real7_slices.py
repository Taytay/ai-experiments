"""REAL-7 v2 (PLAN step 98, REAL-22): the same 1,200 REAL-7 queries with the prompt's slice of the user's history chosen by a policy.
The owner (2026-09-28): the production prompt may hold any of the user's history; which slice is best? Parts, each switchable:

  self     the query merchant's own latest rows (up to K_SELF)
  sim      rows of similar merchants (up to K_SIM rows, at most two per merchant, latest first), similarity by
             kind      the merchant database's category of the merchant (a places / merchant DB gives this in production)
             embrec    cosine of all-MiniLM-L6-v2 embeddings of the merchants' database records (no category field needed)
             embname   cosine of the same model's embeddings of the merchants' names (no database at all)
  cover    the latest row of every category the slice does not show yet
  recent   the latest rows, filling the budget
Parts are taken in that priority until the row budget (24, as v1; or 48) is full, then the rows are put in date order. Every row and
the query carry a date (day of the year from 2025-01-01) unless the policy is undated.

Only the past: every row precedes the query, dates never pass the query's or decrease (`past_only`), and no row names a category
created after the query.
Per item: `best` (the answer a reader that knew the generator would give from this prompt: the rule when two or more of the merchant's
rows are shown, else its latest shown filing, else the default mapping of its category), `in_shots`, and `n_self` rows shown.
Writes data/processed/real7_v2_<policy>.json for each policy (frozen; --force rebuilds) and prints the ceiling per policy.
usage: uv run python scripts/build_real7_slices.py [--force]
"""
import datetime
import json
import sys
from pathlib import Path

import numpy as np

from ai_experiments import real6 as R6
from ai_experiments import transactions as T
from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
import build_real7  # noqa: E402

D0 = datetime.date(2025, 1, 1)
POLICIES = {  # name: (parts, similarity, budget, K_SELF, K_SIM, dated)
    "recent": (["recent"], None, 24, 0, 0, True),
    "self": (["self", "recent"], None, 24, 6, 0, True),
    "self_kind": (["self", "sim", "recent"], "kind", 24, 6, 6, True),
    "self_cover": (["self", "cover", "recent"], None, 24, 6, 0, True),
    "kind_cover": (["sim", "cover", "recent"], "kind", 24, 0, 6, True),
    "all_kind": (["self", "sim", "cover", "recent"], "kind", 24, 6, 6, True),
    "all_embrec": (["self", "sim", "cover", "recent"], "embrec", 24, 6, 6, True),
    "all_embname": (["self", "sim", "cover", "recent"], "embname", 24, 6, 6, True),
    "all_kind_undated": (["self", "sim", "cover", "recent"], "kind", 24, 6, 6, False),
    "all_kind_b48": (["self", "sim", "cover", "recent"], "kind", 48, 8, 10, True),
}


def similarity():
    """{mode: {merchant: {other merchant: score}}} for kind (1 same DB category, else 0), embrec and embname (cosine)."""
    from sentence_transformers import SentenceTransformer
    ms = T.load()["merchants"]; db = R6.load()["fact_db"]; names = [m["name"] for m in ms]
    out = {"kind": {a["name"]: {b["name"]: float(a["category"] == b["category"]) for b in ms} for a in ms}}
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    for mode, texts in (("embrec", [db[n] for n in names]), ("embname", names)):
        e = enc.encode(texts, normalize_embeddings=True); s = e @ e.T
        out[mode] = {a: {b: float(s[i, j]) for j, b in enumerate(names)} for i, a in enumerate(names)}
    return out


def select(rows, q, parts, sim, budget, k_self, k_sim):
    """Indices into rows (the history before the query, oldest first) chosen by the parts in priority order."""
    chosen = []

    def take(idx):
        for i in idx:
            if len(chosen) >= budget:
                return
            if i not in chosen:
                chosen.append(i)
    latest = list(range(len(rows)))[::-1]
    for part in parts:
        if part == "self":
            take([i for i in latest if rows[i]["merchant"] == q["merchant"]][:k_self])
        elif part == "sim":
            per, idx = {}, []
            order = sorted((i for i in latest if rows[i]["merchant"] != q["merchant"] and sim[rows[i]["merchant"]] > 0),
                           key=lambda i: (-sim[rows[i]["merchant"]], -i))
            for i in order:
                if per.get(rows[i]["merchant"], 0) < 2 and len(idx) < k_sim:
                    idx.append(i); per[rows[i]["merchant"]] = per.get(rows[i]["merchant"], 0) + 1
            take(idx)
        elif part == "cover":
            shown = {rows[i]["filed"] for i in chosen}; idx = []
            for i in latest:
                if rows[i]["filed"] not in shown:
                    idx.append(i); shown.add(rows[i]["filed"])
            take(idx)
        elif part == "recent":
            take(latest)
    return sorted(chosen)


def past_only(rows, q, opts):
    """Nothing from the query's future (owner, 2026-09-28): the rows before the query in time order, dated no later than the query and
    never decreasing (the generator's one-day jitter can date a row a day after the next one or after the query), and without rows
    filed under a category that does not exist yet at the query (the generator's misfiles can pick a category created later)."""
    out, cap = [], q["day"]
    for r in reversed(rows):
        cap = min(cap, r["day"])
        if r["filed"] in opts:
            out.append(dict(r, day=cap))
    return out[::-1]


def line(r, dated):
    d = f"{(D0 + datetime.timedelta(days=r['day'])).isoformat()} | " if dated else ""
    return f"Transaction: {d}{r['text']} | ${r['amount']:.2f} | {r['weekday']}"


if __name__ == "__main__":
    items, _, hist_of = build_real7.generate()
    assert R6.sha256(items) == json.loads((PROCESSED / "real7_v1.json").read_text())["sha256"]
    sims = similarity()
    for name, (parts, smode, budget, k_self, k_sim, dated) in POLICIES.items():
        dst = PROCESSED / f"real7_v2_{name}.json"
        if dst.exists() and "--force" not in sys.argv:
            print(name, "exists (frozen)"); continue
        out = []
        for it in items:
            q = hist_of[it["id"]][-1]; opts = [o.strip() for o in it["options"]]
            rows = past_only(hist_of[it["id"]][:-1], q, opts)
            sim = sims[smode][q["merchant"]] if smode else {}
            pick = [rows[i] for i in select(rows, q, parts, sim, budget, k_self, k_sim)]
            header = it["prompt"].split("\n\n", 1)[0]
            qrow = dict(day=q["day"], text=it["text"], amount=it["amount"], weekday=it["weekday"])
            prompt = header + "\n\n" + "".join(f"{line(r, dated)}\nCategory: {r['filed']}\n\n" for r in pick) + line(qrow, dated) + "\nCategory:"
            same = [r for r in pick if r["merchant"] == q["merchant"]]
            if q["rule_best"] is not None and len(same) >= 2:
                best = q["rule_best"]
            elif same:
                best = same[-1]["filed"]
            else:
                best = q["default_best"]
            x = dict(it, prompt=prompt, prompt_ctx=prompt, best=opts.index(best) if best in opts else -1, in_shots=bool(same), n_self=len(same),
                     n_rows=len(pick), policy=name)
            out.append(x)
        doc = dict(name="real7", version=f"v2_{name}", policy=dict(parts=parts, similarity=smode, budget=budget, k_self=k_self, k_sim=k_sim, dated=dated),
                   variant_of=f"real7 v1 ({json.loads((PROCESSED / 'real7_v1.json').read_text())['sha256'][:12]})", items=out, sha256=R6.sha256(out))
        dst.write_text(json.dumps(doc, indent=0, ensure_ascii=False))
        print(f"{name}: ceiling {100 * np.mean([x['best'] == x['answer'] for x in out]):.1f}, merchant shown {100 * np.mean([x['in_shots'] for x in out]):.1f}%,"
              f" mean prompt chars {np.mean([len(x['prompt']) for x in out]):.0f}")
