"""PLAN step 182: how accurate are the "Kind:" lines on a real budget? Draws a sample of payees that carry one (payee_kinds.json from
merchant_db_match_real.py), weighted by how many of the budget's transactions each covers, stratified by source (the merchant database's
prefix match, else Overture's match), and writes it for a judge to mark; then scores the marks.
Private: the sample and the marks stay in OUT (never the repo); `score` prints counts only.
  sample: OUT/kind_audit.jsonl, one line per payee: payee, source, matched (the database merchant or Overture place), kind, n_tx, verdict=null
  score:  reads the verdicts ("right" / "close": a neighbouring kind / "wrong" / "unclear") and prints accuracy by source, weighted by
          transaction (the sample is drawn in proportion to them)
env: BUDGET, OUT (as real_budget_eval.py), N (per source, 100).
usage: BUDGET=<id> [MATCH=words] uv run python scripts/kind_line_audit.py sample|score
"""
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_knowledge_episodes import merchants  # noqa: E402

from ai_experiments import taxonomy_v2 as T  # noqa: E402

BUDGET = os.environ["BUDGET"]
OUT = Path(os.environ.get("OUT", Path.home() / ".local" / "share" / "ynab-real-eval" / BUDGET))
N = int(os.environ.get("N", "100"))
SUF = "" if os.environ.get("MATCH", "prefix") == "prefix" else "_v2"  # MATCH=words: merchant_db_match_real.py's v2 lines (Overture first)
F = OUT / f"kind_audit{SUF}.jsonl"


def sample():
    items = json.loads((OUT / "items.json").read_text())["items"]
    ntx = Counter(it["prompt"].rsplit("Transaction: ", 1)[1].split(" | ")[1] for it in items)
    kinds = json.loads((OUT / f"payee_kinds{SUF}.json").read_text())
    tiers = json.loads((OUT / f"merchant_db_tiers{SUF}.json").read_text())
    ov = json.loads((OUT / "overture_kinds.json").read_text())
    if SUF:  # v2: Overture's match wins where it gave a kind
        from merchant_db_match_real import VAGUE
        tiers = {p: i for p, i in tiers.items() if not (p in ov and ov[p]["kind"] not in VAGUE and T.overture_kind(ov[p]["kind"].replace(" ", "_")))}
    ms = merchants("train", 48000)
    rng = random.Random(182)
    out = []
    for src, pool in (("database", [p for p in kinds if p in tiers]), ("overture", [p for p in kinds if p not in tiers])):
        w = [ntx[p] for p in pool]
        picked = set()
        while len(picked) < min(N, len(pool)):
            picked.add(rng.choices(pool, w)[0])
        for p in sorted(picked):
            m = ms[tiers[p]] if src == "database" else None
            out.append(dict(payee=p, source=src, matched=m["name"] if m else ov[p]["match"],
                            matched_note=(m.get("description") or "")[:80] if m else ov[p]["kind"], kind=kinds[p], kind_text=T.KINDS[kinds[p]],
                            n_tx=ntx[p], verdict=None))
    with open(F, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.chmod(F, 0o600)
    print(f"{len(out)} payees -> {F}")


def score():
    """The sample is drawn in proportion to transactions, so its plain shares estimate accuracy per transaction (unclear left out)."""
    rs = [json.loads(line) for line in open(F)]
    by = defaultdict(list)
    for r in rs:
        by[r["source"]].append(r)
    print("| source | payees judged | right | close | wrong | unclear | right, per transaction | right or close, per transaction |")
    print("|---|---|---|---|---|---|---|---|")
    for src, xs in list(by.items()) + [("all (each source half the sample)", rs)]:
        c = Counter(r["verdict"] for r in xs)
        n = len(xs) - c["unclear"]
        print(f"| {src} | {len(xs)} | {c['right']} | {c['close']} | {c['wrong']} | {c['unclear']} | {100 * c['right'] / n:.0f}% | "
              f"{100 * (c['right'] + c['close']) / n:.0f}% |")


if __name__ == "__main__":
    {"sample": sample, "score": score}[sys.argv[1]]()
