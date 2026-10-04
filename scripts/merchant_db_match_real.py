"""PLAN step 180: which of a real budget's payees name a merchant the knowledge episodes trained (build_knowledge_episodes.py's
merchants("train", 48000), most popular first), so real_budget_tables.py can split first-time payees by whether the model was taught the
merchant. A payee matches a merchant when, after dropping a processor prefix ("SQ *", "TST* ", "PAYPAL *", ...) and keeping letters and
digits only, it starts with the merchant's name or domain stem so normalised (5 characters at least). Approximate: a short brand can match a
longer unrelated name. Private: writes OUT/merchant_db_tiers.json ({payee: index of the matched merchant in the popularity order}), next
to the budget's items, never in the repo; prints counts only.
env: BUDGET, OUT (as real_budget_eval.py).
usage: BUDGET=<id> uv run python scripts/merchant_db_match_real.py
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_knowledge_episodes import merchants  # noqa: E402

BUDGET = os.environ["BUDGET"]
OUT = Path(os.environ.get("OUT", Path.home() / ".local" / "share" / "ynab-real-eval" / BUDGET))
PREFIX = re.compile(r"^(sq ?\*|tst\* ?|paypal ?\*|pp\*|sp \*?|py \*|dd \*|doordash\*|ic\* ?|amzn mktp us\*?|in \*|bt\*|ckc\*|fsp\*|sqsp\* ?)", re.I)


def key(s):
    return re.sub(r"[^a-z0-9]", "", PREFIX.sub("", s.strip()).lower())


if __name__ == "__main__":
    items = json.loads((OUT / "items.json").read_text())["items"]
    payees = {it["prompt"].rsplit("Transaction: ", 1)[1].split(" | ")[1] for it in items}
    ms = merchants("train", 48000)
    first = {}  # normalised key -> best (lowest) popularity index
    for i, m in enumerate(ms):
        for k in {key(m["name"]), key(m["domain"].split(".")[0]) if m["domain"] else ""}:
            if len(k) >= 5:
                first.setdefault(k, i)
    lens = sorted({len(k) for k in first}, reverse=True)
    tiers = {}
    for p in payees:
        pk = key(p)
        hits = [first[pk[:n]] for n in lens if n <= len(pk) and pk[:n] in first]
        if hits:
            tiers[p] = min(hits)
    (OUT / "merchant_db_tiers.json").write_text(json.dumps(tiers)); os.chmod(OUT / "merchant_db_tiers.json", 0o600)
    n = len(payees)
    print(f"{n} distinct payees; matched {len(tiers)} ({100 * len(tiers) / n:.1f}%): top 5k {sum(t < 5000 for t in tiers.values())}, "
          f"5k-20k {sum(5000 <= t < 20000 for t in tiers.values())}, 20k-48k {sum(t >= 20000 for t in tiers.values())}")
