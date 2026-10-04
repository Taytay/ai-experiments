"""PLAN step 180: which of a real budget's payees name a merchant the knowledge episodes trained (build_knowledge_episodes.py's
merchants("train", 48000), most popular first), so real_budget_tables.py can split first-time payees by whether the model was taught the
merchant. A payee matches a merchant when, after dropping a processor prefix ("SQ *", "TST* ", "PAYPAL *", ...) and keeping letters and
digits only, it starts with the merchant's name or domain stem so normalised (5 characters at least). Approximate: a short brand can match a
longer unrelated name. Private: writes OUT/merchant_db_tiers.json ({payee: index of the matched merchant in the popularity order}), next
to the budget's items, never in the repo; and OUT/payee_kinds.json ({payee: taxonomy_v2 kind}: the matched merchant's kind, else the
Overture match's (overture_lookup.py) mapped by taxonomy_v2.overture_kind), the source of row 182's "Kind:" lines; prints counts only.
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

from ai_experiments import taxonomy_v2 as T  # noqa: E402

BUDGET = os.environ["BUDGET"]
OUT = Path(os.environ.get("OUT", Path.home() / ".local" / "share" / "ynab-real-eval" / BUDGET))
PREFIX = re.compile(r"^(sq ?\*|tst\* ?|paypal ?\*|pp\*|sp \*?|py \*|dd \*|doordash\*|ic\* ?|amzn mktp us\*?|in \*|bt\*|ckc\*|fsp\*|sqsp\* ?)", re.I)


MATCH = os.environ.get("MATCH", "prefix")  # "prefix" (row 180) | "words" (row 182 v2: whole words, names only, Overture first)
SUF = "" if MATCH == "prefix" else "_v2"
NONMERCHANT = re.compile(r"deposit|interest|transfer|refund|payment|\bpaid\b|dividend|payroll|reimburs|\batm\b|withdraw|venmo|zelle|cash ?app|starting balance", re.I)
VAGUE = {"media service", "professional service", "shopping", "social or community service", "community and government"}


def words(s):
    return [w for w in re.findall(r"[a-z]+", PREFIX.sub("", re.sub(r"^(sale|return|pos|debit|purchase)\b\W*", "", s.strip(), flags=re.I)).lower()) if w not in ("the", "inc", "llc", "co")]


def key(s):
    return re.sub(r"[^a-z0-9]", "", PREFIX.sub("", s.strip()).lower())


if __name__ == "__main__":
    items = json.loads((OUT / "items.json").read_text())["items"]
    payees = {it["prompt"].rsplit("Transaction: ", 1)[1].split(" | ")[1] for it in items}
    ms = merchants("train", 48000)
    if MATCH == "words":  # row 182 v2 (the audit: domain stems and short prefixes matched sub-brands and unrelated businesses)
        first = {}  # the merchant name's words -> best popularity index; also the words joined ("chickfila")
        for i, m in enumerate(ms):
            w = words(m["name"])
            if w and len("".join(w)) >= 4:
                first.setdefault(tuple(w), i); first.setdefault(("".join(w),), i)
        tiers = {}
        for p in payees:
            if NONMERCHANT.search(p):
                continue
            w = words(p)
            hits = [first[t] for n in range(len(w), 0, -1) for t in (tuple(w[:n]), ("".join(w[:n]),)) if t in first]
            if hits:
                tiers[p] = hits[0]  # the longest match
        ov = json.loads((OUT / "overture_kinds.json").read_text()) if (OUT / "overture_kinds.json").exists() else {}
        kinds = {p: ms[i]["kind"] for p, i in tiers.items()}
        kinds |= {p: k for p, v in ov.items() if v["kind"] not in VAGUE and not NONMERCHANT.search(p)
                  and (k := T.overture_kind(v["kind"].replace(" ", "_")))}  # Overture's match first (74% right against 35%)
    else:
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
        ov = json.loads((OUT / "overture_kinds.json").read_text()) if (OUT / "overture_kinds.json").exists() else {}
        kinds = {p: k for p, v in ov.items() if (k := T.overture_kind(v["kind"].replace(" ", "_")))}
        kinds |= {p: ms[i]["kind"] for p, i in tiers.items()}  # the database's match first
    files = [(f"merchant_db_tiers{SUF}.json", tiers), (f"payee_kinds{SUF}.json", kinds)]
    if MATCH == "words":  # Overture's lines alone (the audit: 77% right against 47% for the database's)
        files.append(("payee_kinds_ov.json", {p: k for p, v in ov.items() if v["kind"] not in VAGUE and not NONMERCHANT.search(p)
                                               and (k := T.overture_kind(v["kind"].replace(" ", "_")))}))
    for name, obj in files:
        (OUT / name).write_text(json.dumps(obj)); os.chmod(OUT / name, 0o600)
    ntx = sum(1 for it in items if it["prompt"].rsplit("Transaction: ", 1)[1].split(" | ")[1] in kinds)
    print(f"kinds for {len(kinds)} payees ({len(set(kinds) - set(tiers))} from Overture only); {100 * ntx / len(items):.1f}% of transactions")
    n = len(payees)
    print(f"{n} distinct payees; matched {len(tiers)} ({100 * len(tiers) / n:.1f}%): top 5k {sum(t < 5000 for t in tiers.values())}, "
          f"5k-20k {sum(5000 <= t < 20000 for t in tiers.values())}, 20k-48k {sum(t >= 20000 for t in tiers.values())}")
