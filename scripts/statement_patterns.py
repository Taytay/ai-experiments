"""PLAN step 173 (owner, 2026-10-03: "make sure that the corrupted payees are corrupted in ways that my budget demonstrates is possible.
It's possible that our existing template doesn't capture all of those patterns."). Measures how real bank strings mangle a payee name,
from the raw strings YNAB imported into the owner's budget (import_payee_name_original, what YNAB's bank-data providers deliver), as a
grammar of aggregate frequencies only: case style, leading word / processor prefix, the suffix after the name, run-together words,
truncation width. No literal text from the budget is stored: the output is data/processed/statement_patterns_v1.json, a few dozen
probabilities, committed; `ai_experiments.statements.render_v2` samples from it.

Decomposition of one raw string (heuristic, in order): a leading "Sale / Return / Refund" word; a processor prefix ("SQ *", "TST*",
"PAYPAL *", "PP*", "Square", or any short token ending in "*"); the suffix type, the first that applies: an authorisation / pending
tail, a web domain, a "*" or "#" reference code, a store number, a long digit run (8+), a city and state, a short number; case (upper /
title / lower / mixed); run-together letters and digits; and the length, where a share of strings sits at a fixed field width.
usage: BUDGET=<id> uv run python scripts/statement_patterns.py
"""
import json
import os
import re
from collections import Counter
from pathlib import Path

from ai_experiments.paths import PROCESSED

BUDGET = os.environ.get("BUDGET", "")
CACHE = Path.home() / ".cache" / "ynab-cli" / f"{BUDGET}.json"
OUT = PROCESSED / "statement_patterns_v1.json"

LEAD = re.compile(r"^(Sale|Return|Refund)\b\s*", re.I)
PROC = [("SQ *", re.compile(r"^SQ\s?\*\s*", re.I)), ("Square ", re.compile(r"^Square\s+", re.I)), ("TST* ", re.compile(r"^TST\*?\s*", re.I)),
        ("PAYPAL *", re.compile(r"^PAYPAL\s?\*\s*", re.I)), ("PP*", re.compile(r"^PP\s?\*\s*", re.I))]
SUFFIX = [("auth", re.compile(r"(-\s*)?\b(AUTHORI[ZS]\w*|PENDING|AUTH)\b.*$", re.I)),
          ("domain", re.compile(r"\.(com|net|org|co)\b.*$|\bwww\.", re.I)),
          ("star_code", re.compile(r"\*\s?[A-Z0-9]{5,}\s*$")),
          ("hash_store", re.compile(r"#\s?-?\w*\d+\s*.*$")),
          ("store_word", re.compile(r"\bSTORE\s+\d+.*$", re.I)),
          ("long_digits", re.compile(r"\s?\d{8,}\w*\s*$")),
          ("city_state", re.compile(r"\s+[A-Z][A-Za-z]+\s+[A-Z]{2}(\s+USA)?\s*$")),
          ("short_number", re.compile(r"\s\d{1,7}\s*$"))]
P2P = re.compile(r"\b(paid|zelle|venmo|cash app|transfer|check|deposit|interest)\b", re.I)


def case_of(s):
    letters = re.sub(r"[^A-Za-z]", "", s)
    if not letters:
        return "none"
    if letters.isupper():
        return "upper"
    if letters.islower():
        return "lower"
    words = [w for w in re.findall(r"[A-Za-z]+", s)]
    return "title" if all(w[0].isupper() for w in words if len(w) > 2) else "mixed"


if __name__ == "__main__":
    b = json.loads(CACHE.read_text())["budget"]
    strings = sorted({t["import_payee_name_original"] for t in b["transactions"] if not t.get("deleted") and t.get("import_payee_name_original")
                      and not t.get("transfer_account_id")})
    merch = [s for s in strings if not P2P.search(s)]  # merchants only: person-to-person, checks and transfers are not rendered
    lead, proc, suf, case, run, lens = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
    for s in merch:
        x = s
        m = LEAD.match(x); lead[m.group(1).capitalize() if m else "none"] += 1
        x = LEAD.sub("", x)
        p = next((k for k, r in PROC if r.match(x)), None)
        if p is None and re.match(r"^[A-Za-z][A-Za-z.]{1,11}\s?\*", x):
            p = "other*"
        proc[p or "none"] += 1
        sfx = next((k for k, r in SUFFIX if r.search(x)), "none")
        suf[sfx] += 1
        case[case_of(s)] += 1
        run["yes" if re.search(r"[A-Za-z]{3,}\d{3,}[A-Za-z]{2,}", s) else "no"] += 1
        lens[len(s)] += 1
    n = len(merch)
    norm = lambda c: {k: round(v / n, 4) for k, v in c.most_common()}  # noqa: E731
    # truncation: the share of strings at each length that stands above the smooth neighbourhood (a fixed field width)
    widths = {}
    for L in range(12, 41):
        around = [lens.get(L + d, 0) for d in (-3, -2, 2, 3)]
        excess = lens.get(L, 0) - sum(around) / len(around)
        if excess > 0.01 * n:
            widths[L] = round(excess / n, 4)
    doc = dict(name="statement_patterns", version=1, source="raw bank strings YNAB imported into one real budget (aggregates only)",
               n_strings=n, lead=norm(lead), processor=norm(proc), suffix=norm(suf), case=norm(case), run_together=norm(run),
               truncate_widths=widths, truncate_share=round(sum(widths.values()), 4))
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc, indent=1))
