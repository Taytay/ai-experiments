"""Row 220 / 218 (owner, 2026-10-06: "will your fix work for payees that are similar but different? What if they go to Williams fuel #5, pos
debit Williams fuel store 111, etc?"): how well each way of matching a bank string to the household's earlier filings finds the same
merchant, on the rational households (bank level: a fresh statement string per transaction; the clean name says which merchant it is).
Per transaction from a merchant bought before: recall = some earlier filing of the same merchant is matched; per matched earlier filing:
precision = it is the same merchant. Methods: payee_key, payee_key_v2, and the neighbour encoder's cosine >= tau over earlier filings.
usage: uv run python scripts/payee_match_check.py [households (20)]
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hist_encoder as H  # noqa: E402
import rational_budgets as R  # noqa: E402
from ai_experiments.payeekey import payee_key, payee_key_v2  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
m1 = H._model(H.OUT1)
stats = defaultdict(lambda: [0, 0, 0, 0])  # recall hits, recall total, precision hits, precision total
for bc, bb in zip(R.budgets("clean", n), R.budgets("bank", n)):
    pc = {p["id"]: p["name"] for p in bc["payees"]}; pb = {p["id"]: p["name"] for p in bb["payees"]}
    names = [pc[t["payee_id"]] for t in bc["transactions"]]; texts = [pb[t["payee_id"]] for t in bb["transactions"]]
    E = H._embed(m1, texts, {})
    S = E @ E.T
    for fn_name, fn in (("payee_key", payee_key), ("payee_key_v2", payee_key_v2)):
        keys = [fn(t) or t for t in texts]
        for i in range(len(texts)):
            if names[i] not in names[:i]:
                continue
            m = [j for j in range(i) if keys[j] == keys[i]]
            st = stats[fn_name]; st[1] += 1; st[0] += any(names[j] == names[i] for j in m); st[3] += len(m); st[2] += sum(names[j] == names[i] for j in m)
    for tau in (0.8, 0.85, 0.9, 0.95):
        for i in range(len(texts)):
            if names[i] not in names[:i]:
                continue
            m = [j for j in range(i) if S[i, j] >= tau]
            st = stats[f"encoder cos >= {tau}"]; st[1] += 1; st[0] += any(names[j] == names[i] for j in m); st[3] += len(m); st[2] += sum(names[j] == names[i] for j in m)
print(f"rational bank strings, {n} households: merchants bought before (n per method below)\n")
print("| method | finds the same merchant (recall) | matched earlier filings that are the same merchant (precision) |\n|---|---|---|")
for k, (rh, rt, ph, pt) in stats.items():
    print(f"| {k} | {100 * rh / rt:.1f}% (of {rt}) | {100 * ph / max(pt, 1):.1f}% (of {pt}) |")
