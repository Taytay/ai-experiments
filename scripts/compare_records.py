"""Item-by-item agreement between two per-item files scored on the same frozen items.

usage: uv run python scripts/compare_records.py A.jsonl B.jsonl [--rule mean]

For each level: n, accuracy of A and B, the share of items where the predicted option differs
(flips), and the largest absolute per-option log-prob difference. Made for "same weights, different
path" checks: merged vs unmerged LoRA, WSL vs Windows, batch vs single forward. The overall flip
rate is the number to compare with the noise floors in REPORT.md section 9.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from ai_experiments import scorers as SC

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--rule", default="mean", choices=list(SC.SCORERS))
args = ap.parse_args()

A = {r["id"]: r for r in SC.read_records(Path(args.a))}
B = {r["id"]: r for r in SC.read_records(Path(args.b))}
pa, pb = SC.params_for(list(A.values())), SC.params_for(list(B.values()))
shared = [i for i in A if i in B]
n, hit_a, hit_b, flips, maxd = Counter(), Counter(), Counter(), Counter(), defaultdict(float)
for i in shared:
    ra, rb = A[i], B[i]
    lv = ra["level"]
    x, y = SC.predict(ra, args.rule, pa), SC.predict(rb, args.rule, pb)
    n[lv] += 1; hit_a[lv] += x == ra["answer"]; hit_b[lv] += y == rb["answer"]; flips[lv] += x != y
    maxd[lv] = max(maxd[lv], max(abs(s - t) for s, t in zip(ra["sum_lp"], rb["sum_lp"])))
print(f"{len(shared)} shared items (A {len(A)}, B {len(B)}); rule {args.rule}")
print(f"{'level':28s} {'n':>5s} {'acc A':>6s} {'acc B':>6s} {'flips%':>7s} {'max|dlp|':>9s}")
for lv in n:
    print(f"{lv:28s} {n[lv]:5d} {100 * hit_a[lv] / n[lv]:6.1f} {100 * hit_b[lv] / n[lv]:6.1f} {100 * flips[lv] / n[lv]:7.1f} {maxd[lv]:9.3f}")
tot = sum(n.values())
print(f"{'ALL':28s} {tot:5d} {100 * sum(hit_a.values()) / tot:6.1f} {100 * sum(hit_b.values()) / tot:6.1f} "
      f"{100 * sum(flips.values()) / tot:7.1f} {max(maxd.values()):9.3f}")
