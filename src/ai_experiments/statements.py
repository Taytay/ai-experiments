"""Bank-statement strings for a merchant name, sampled from a measured grammar (PLAN step 173; owner, 2026-10-03: corrupt payees "in
ways that my budget demonstrates is possible"). data/processed/statement_patterns_v1.json (scripts/statement_patterns.py) holds the
frequencies of case style, a leading Sale / Return word, a processor prefix, the suffix after the name, run-together words and fixed
truncation widths, measured on the raw strings YNAB's bank-data providers delivered for one real budget; no literal text is stored, and
codes and digits here are random. `transactions.render` (the older templates: always upper case, card words, city and date) stays for
the frozen sets.
"""
import json
import random
import re
import string
from functools import lru_cache

from ai_experiments.paths import PROCESSED

PROCESSOR_TOKENS = ["SP", "DD", "IC", "CLV", "TOAST", "PwP", "PY", "SQSP", "BT", "LS"]  # generic platform tokens seen before "*" on statements
CITIES = ["AUSTIN TX", "TULSA OK", "DENVER CO", "SEATTLE WA", "CHICAGO IL", "ATLANTA GA", "PHOENIX AZ", "BOSTON MA", "PORTLAND OR", "MIAMI FL",
          "NASHVILLE TN", "OMAHA NE", "BOISE ID", "RALEIGH NC", "COLUMBUS OH", "SAN DIEGO CA", "BROOKLYN NY", "MADISON WI"]


@lru_cache(maxsize=None)
def patterns(version=1):
    return json.loads((PROCESSED / f"statement_patterns_v{version}.json").read_text())


def _pick(rng, dist):
    r, acc = rng.random(), 0.0
    for k, p in dist.items():
        acc += p
        if r < acc:
            return k
    return next(iter(dist))


def _code(rng, n):
    return "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(n))


def _digits(rng, n):
    return "".join(rng.choice(string.digits) for _ in range(n))


def render_v2(name, rng=None, city=None, P=None):
    """One statement string for merchant `name` (its plain name, e.g. "Costco Wholesale"), sampled from the measured grammar."""
    rng = rng or random.Random()
    P = P or patterns()
    body = name.replace("&", "and") if rng.random() < 0.3 else name
    if _pick(rng, P["run_together"]) == "yes":  # words or digits run together ("MURPHY6533ATWALMART" style)
        body = re.sub(r"\s+", "", body) if rng.random() < 0.6 else body.split()[0] + _digits(rng, rng.randint(3, 5)) + "".join(body.split()[1:])
    proc = _pick(rng, P["processor"])
    sfx = _pick(rng, P["suffix"])
    if sfx == "star_code":
        body += "*" + _code(rng, rng.choice([6, 8, 9, 9, 10]))
    elif sfx == "domain":
        body = re.sub(r"[^A-Za-z0-9]", "", body) + rng.choice([".com", ".com", ".com", ".net", ".co"]) + (" " + city.split()[-1] if city and rng.random() < 0.3 else "")
    elif sfx == "long_digits":
        body += " " + _digits(rng, rng.randint(8, 12))
    elif sfx == "short_number":
        body += " " + _digits(rng, rng.randint(1, 5))
    elif sfx == "hash_store":
        body += " #" + _digits(rng, rng.randint(3, 5)) + ((" " + (city or rng.choice(CITIES))) if rng.random() < 0.3 else "")
    elif sfx == "auth":
        body += (" #" + _digits(rng, 4) if rng.random() < 0.5 else "") + rng.choice([" - AUTHORIZATION", " - AUTHORIZED ON", " PENDING", " AUTH"])
    elif sfx == "city_state":
        body += " " + (city or rng.choice(CITIES))
    elif sfx == "store_word":
        body += " STORE " + _digits(rng, rng.randint(3, 5))
    if proc == "other*":
        body = rng.choice(PROCESSOR_TOKENS) + rng.choice(["*", " *", " * "]) + body
    elif proc != "none":
        body = proc + body
    case = _pick(rng, P["case"])
    if case == "upper":
        body = body.upper()
    elif case == "title":
        body = " ".join(w if any(c.isdigit() for c in w) or "*" in w else w[:1].upper() + w[1:].lower() for w in body.split(" "))
    elif case == "lower":
        body = body.lower()
    lead = _pick(rng, P["lead"])
    if lead != "none":
        body = lead + " " + body
    if rng.random() < P["truncate_share"]:  # a fixed field width
        w = _pick(rng, {int(k): v / P["truncate_share"] for k, v in P["truncate_widths"].items()})
        body = body[:w]
    return body
