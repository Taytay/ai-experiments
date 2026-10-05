"""Row 208 E3: per-item agreement of two reads of the same model and items (HF loop against vLLM), from results/per_item files.
Reports top-1 of each, the share of items where the two first choices agree (the bar: >= 99.5%), how many of the disagreements are
near-ties in either read (top-two log-prob gap < 0.05), and the largest log-prob difference on the gold option.
usage: uv run python scripts/vllm_compare.py <per_item A.jsonl> <per_item B.jsonl>
"""
import json
import sys

import numpy as np


def load(p):
    return {r["id"]: r for r in map(json.loads, open(p))}


if __name__ == "__main__":
    a, b = load(sys.argv[1]), load(sys.argv[2])
    ids = [i for i in a if i in b]
    agree = [a[i]["pred"] == b[i]["pred"] for i in ids]
    def gap(r):
        s = sorted(r["sum_lp"], reverse=True)
        return s[0] - s[1] if len(s) > 1 else 9.0
    dis = [i for i, g in zip(ids, agree) if not g]
    near = sum(min(gap(a[i]), gap(b[i])) < 0.05 for i in dis)
    dgold = max(abs(a[i]["sum_lp"][a[i]["answer"]] - b[i]["sum_lp"][b[i]["answer"]]) for i in ids)
    print(f"{len(ids)} items: top-1 A {100 * np.mean([a[i]['correct'] for i in ids]):.2f}, B {100 * np.mean([b[i]['correct'] for i in ids]):.2f}; "
          f"first choices agree on {100 * np.mean(agree):.2f}% ({len(dis)} differ, {near} of them near-ties); "
          f"max |gold log-prob difference| {dgold:.3f}")
