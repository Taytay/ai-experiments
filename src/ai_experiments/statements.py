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


VIA = {"SQ *": "Square", "Square ": "Square", "TST* ": "Toast", "PAYPAL *": "PayPal", "PP*": "PayPal", "SP": "Shopify", "DD": "DoorDash",
       "IC": "Instacart", "CLV": "Clover", "TOAST": "Toast", "PwP": "Privacy.com", "PY": "PayPal", "SQSP": "Squarespace", "BT": "Braintree",
       "LS": "Lightspeed"}  # what a processor prefix stands for (row 176's "Via" field)


def render_v2(name, rng=None, city=None, P=None, parts=False):
    """One statement string for merchant `name` (its plain name, e.g. "Costco Wholesale"), sampled from the measured grammar. With
    parts=True, also the pieces it was built from (row 176): {"Clean payee", "Via", "Location", "Store", "Reference"}, each kept only if
    still visible after truncation; the random draws are the same either way, so the strings do not change."""
    rng = rng or random.Random()
    P = P or patterns()
    f = {"Clean payee": name}
    body = name.replace("&", "and") if rng.random() < 0.3 else name
    if _pick(rng, P["run_together"]) == "yes":  # words or digits run together ("MURPHY6533ATWALMART" style)
        body = re.sub(r"\s+", "", body) if rng.random() < 0.6 else body.split()[0] + _digits(rng, rng.randint(3, 5)) + "".join(body.split()[1:])
    proc = _pick(rng, P["processor"])
    sfx = _pick(rng, P["suffix"])
    if sfx == "star_code":
        code = _code(rng, rng.choice([6, 8, 9, 9, 10])); body += "*" + code; f["Reference"] = code
    elif sfx == "domain":
        loc = (" " + city.split()[-1] if city and rng.random() < 0.3 else "")
        body = re.sub(r"[^A-Za-z0-9]", "", body) + rng.choice([".com", ".com", ".com", ".net", ".co"]) + loc
        if loc:
            f["Location"] = loc.strip()
    elif sfx == "long_digits":
        code = _digits(rng, rng.randint(8, 12)); body += " " + code; f["Reference"] = code
    elif sfx == "short_number":
        code = _digits(rng, rng.randint(1, 5)); body += " " + code; f["Store"] = code
    elif sfx == "hash_store":
        code = _digits(rng, rng.randint(3, 5)); loc = ((" " + (city or rng.choice(CITIES))) if rng.random() < 0.3 else "")
        body += " #" + code + loc; f["Store"] = code
        if loc:
            f["Location"] = loc.strip()
    elif sfx == "auth":
        code = (" #" + _digits(rng, 4) if rng.random() < 0.5 else ""); body += code + rng.choice([" - AUTHORIZATION", " - AUTHORIZED ON", " PENDING", " AUTH"])
        if code:
            f["Store"] = code.strip(" #")
    elif sfx == "city_state":
        loc = (city or rng.choice(CITIES)); body += " " + loc; f["Location"] = loc
    elif sfx == "store_word":
        code = _digits(rng, rng.randint(3, 5)); body += " STORE " + code; f["Store"] = code
    if proc == "other*":
        tok = rng.choice(PROCESSOR_TOKENS); body = tok + rng.choice(["*", " *", " * "]) + body; f["Via"] = VIA.get(tok, tok)
    elif proc != "none":
        body = proc + body; f["Via"] = VIA.get(proc, proc.strip(" *"))
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
    if not parts:
        return body
    low = body.lower()
    for k in ("Reference", "Store", "Location"):  # cut off by truncation: not in the string, so not a field
        if k in f and f[k].lower() not in low:
            del f[k]
    if "Via" in f and not any(t.lower() in low for t in [k for k, v in VIA.items() if v == f["Via"]] + [f["Via"]]):
        del f["Via"]
    return body, f
