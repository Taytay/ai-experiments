"""PLAN step 149: blind prompts with a canonical "Kind:" line under every history row and the query (ai_experiments.canon labels; the
generators' kinds from blind_v<N>_meta.json; "unknown" where the generator has none), on the other-users variant:
  blind_v<N>_kinds      every row and the query from the generator (a merchant database that knows every payee)
  blind_v<N>_kindsinf   rows from the generator, the query's kind inferred by a reader from the payee's name and other users' filings
                        (step 148's per-item file given as the argument; the top option), standing in for payees no database covers
usage: uv run python scripts/derive_kind_prompts.py <N> [<kindpay per-item jsonl for the inferred variant>]
"""
import hashlib
import json
import sys

from ai_experiments import canon as CN
from ai_experiments.paths import PROCESSED

V = sys.argv[1]
items = json.loads((PROCESSED / f"blind_v{V}_others.json").read_text())["items"]
meta = json.loads((PROCESSED / f"blind_v{V}_meta.json").read_text())


def with_kinds(prompt, kinds):
    lines, out, k = prompt.split("\n"), [], 0
    for ln in lines:
        out.append(ln)
        if ln.startswith("Transaction: "):
            out.append(f"Kind: {kinds[k]}"); k += 1
    assert k == len(kinds), (k, len(kinds))
    return "\n".join(out)


def write(name, its):
    sha = hashlib.sha256(json.dumps(its, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    (PROCESSED / f"{name}.json").write_text(json.dumps({"name": name, "items": its, "sha256": sha}, ensure_ascii=False))
    print(f"wrote {name}: {len(its)} items")


lab = lambda k: CN.label(k) or "unknown"  # noqa: E731
out = []
for it in items:
    m = meta[it["id"]]
    p = with_kinds(it["prompt"], [lab(k) for k in m["row_kinds"]] + [lab(m["query_kind"])])
    out.append(dict(it, prompt=p, prompt_ctx=p))
write(f"blind_v{V}_kinds", out)
if len(sys.argv) > 2:
    kp = {i["id"]: i for i in json.loads((PROCESSED / f"kindpay_v{V}.json").read_text())["items"]}
    pred = {}
    for r in map(json.loads, open(sys.argv[2])):
        pred[r["id"]] = CN.PAYEE_OPTIONS[max(range(len(r["sum_lp"])), key=lambda j: r["sum_lp"][j])]
    out = []
    for it in items:
        m = meta[it["id"]]
        pid = f"KP{V}:{it['merchant']}:both"
        q = pred.get(pid) or pred.get(f"KP{V}:{it['merchant']}:name") or "unknown"
        p = with_kinds(it["prompt"], [lab(k) for k in m["row_kinds"]] + [q])
        out.append(dict(it, prompt=p, prompt_ctx=p))
    write(f"blind_v{V}_kindsinf", out)
