"""Row 234 checks (owner, 2026-10-07: "Do run the two checks"), on a rational version (RATIONAL_PAYEES / RATIONAL_CATS):
(1) the pull of busy categories: on first purchases at a merchant, how often the reader's choice / the right answer / its misses are the
household's most-used category so far; (2) how often li_decider's simulated kind line (KINDLINE models: 57% coverage, lookup errors)
reaches a first purchase, and how often its kind is right. ARM (li_r227_fcr). Synthetic data only."""
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import kind_lookup_text as KL  # noqa: E402
import li_decider as LD  # noqa: E402
import rational_budgets as RG  # noqa: E402

run = LD.load(os.environ.get("ARM", "li_r227_fcr"))
n = right = pred_top = gold_top = miss_top = misses = 0
cov = correct = firsts = 0
for b in RG.budgets("bank"):
    tx = {t["id"]: t for t in b["transactions"]}
    ev, sco = run(b)
    used = Counter()
    for e, sc in zip(ev, sco):
        t = tx[e["id"]]
        if t["reason"][0] == "new" and e["gold"] in sc:
            top = used.most_common(1)[0][0] if used else None
            pred = max(sc, key=sc.get)
            n += 1; right += pred == e["gold"]
            pred_top += pred == top; gold_top += e["gold"] == top
            if pred != e["gold"]:
                misses += 1; miss_top += pred == top
        used[e["gold"]] += 1
    seen = set()
    for t in b["transactions"]:  # check 2: the kind line as _kindline draws it (per payee string)
        if t["reason"][0] != "new":
            continue
        name = {p["id"]: p["name"] for p in b["payees"]}[t["payee_id"]]
        firsts += 1
        import hashlib
        r = random.Random(int(hashlib.md5(f'{b["id"]}|{name}'.encode()).hexdigest(), 16))
        if t["kind"] in KL.GROUP_OF and r.random() < KL.COVER:
            bk = KL.sample_bucket(r)
            k2 = KL.corrupt(t["kind"], bk, r)[0]
            cov += 1; correct += k2 == t["kind"]
print(f"(1) first purchases {n}: right {100 * right / n:.1f}%; the household's busiest category so far is the right answer {100 * gold_top / n:.1f}% "
      f"of the time, chosen {100 * pred_top / n:.1f}%; of the {misses} misses, {100 * miss_top / max(misses, 1):.1f}% chose it")
print(f"(2) kind line on first purchases: shown {100 * cov / firsts:.1f}%, right kind when shown {100 * correct / max(cov, 1):.1f}% "
      f"(right kind on {100 * correct / firsts:.1f}% of first purchases)")
