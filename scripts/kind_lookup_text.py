"""PLAN row 212 (a), the lookup as text: a transaction's text with the kind a database lookup found for its payee and how well the payee
matched, "<text> | kind: gas station (match high)", so the encoder can learn how far to trust a lookup; and a training-time corrupter that
makes the line wrong as often as real lookups were.

Buckets (`bucket`): a lookup result is (kind, score, source). Overture matches (overture_lookup.py) carry a Jaro-Winkler score and are kept
from 0.93 (brands) or 0.95 (place names): "high" from HIGH (0.97), "mid" below it; a string match in the merchant database
(merchant_db_match_real.py: prefix or whole-word, no score) is "low", as is any score under MID (0.93).

Rates (`RATES`: right / close / wrong per bucket), from REPORT §168 Kind lines from a lookup, Table 168.1 (one judge, 200 of the owner's
payees drawn in proportion to transactions, half per source; "close" = a neighbouring kind), v2 lines (whole words, Overture first):
  Overture matches            77% right, 92% right or close  -> high: 0.77 / 0.15 / 0.08
  merchant database matches   47% right, 60% right or close  -> low:  0.47 / 0.13 / 0.40
  all lines (57% of txns)    ~65% right, ~79% right or close -> mid:  0.65 / 0.14 / 0.21
§168 measures by source, not by score, so the split is an assumption: Overture's rate is given to its high-score matches, the database's to
string matches, and the pooled rate to the band between (Overture matches of 0.93-0.97). Replace them with per-bucket counts once the audit
(kind_line_audit.py) is re-cut by score. The default bucket mix (`SHARES`: high 0.4, mid 0.3, low 0.3) reproduces the pooled v2 figures:
0.4 x 77 + 0.3 x 65 + 0.3 x 47 = 64.4% right (§168: ~65), 0.4 x 92 + 0.3 x 79 + 0.3 x 60 = 78.5% right or close (~79). COVER (0.57) is
the share of transactions that had a v2 line. A "close" kind is another kind of the same group (GROUPS: taxonomy_v2.KINDS' own sections);
a wrong kind is drawn from the other groups.
usage: from kind_lookup_text import render, bucket, corrupt, training_text
       uv run python scripts/kind_lookup_text.py selftest
"""
import random
import sys
from collections import Counter

from ai_experiments import taxonomy_v2 as T

HIGH, MID = 0.97, 0.93
RATES = {"high": (0.77, 0.15, 0.08), "mid": (0.65, 0.14, 0.21), "low": (0.47, 0.13, 0.40)}  # right, close, wrong (§168, Table 168.1, v2)
SHARES = {"high": 0.4, "mid": 0.3, "low": 0.3}
COVER = 0.57
GROUPS = {  # taxonomy_v2.KINDS' sections; non-merchant kinds (p2p, income, savings, loan, several, purpose) left out
    "food and drink": ["grocery", "restaurant", "fast_food", "coffee_bakery", "food_delivery", "bar", "liquor", "convenience"],
    "car": ["gas", "car_repair", "car_purchase", "parking", "transit"],
    "travel": ["airline", "hotel", "car_rental", "travel_agency"],
    "home and bills": ["utility", "phone", "rent", "home_improvement", "furniture", "home_services", "insurance"],
    "health and personal": ["medical", "pharmacy", "fitness", "personal_care", "laundry"],
    "subscriptions and digital": ["subscription", "digital"],
    "shopping": ["big_box", "clothing", "jewelry", "electronics", "hobby", "gifts", "secondhand", "tobacco", "office"],
    "fun": ["entertainment", "gambling"],
    "family and pets": ["kids", "education", "pet"],
    "money and government": ["bank", "taxes", "fines", "legal"],
    "giving, business, life events": ["charity", "business", "shipping", "events", "funeral"],
}
GROUP_OF = {k: g for g, ks in GROUPS.items() for k in ks}
MERCHANT_KINDS = [k for ks in GROUPS.values() for k in ks]
assert set(MERCHANT_KINDS) == set(T.KINDS) - {"p2p", "income", "savings", "loan", "several", "purpose"}


def bucket(score=None, source="overture"):
    """'high' | 'mid' | 'low' for a lookup result: a database string match (no score) is low; an Overture match by its score."""
    if source != "overture" or score is None or score < MID:
        return "low"
    return "high" if score >= HIGH else "mid"


def words(kind):
    """A taxonomy_v2 kind key in words ("gas" -> "gas station"); any other text unchanged."""
    return T.KINDS.get(kind, kind)


def suffix(kind, b):
    return f"kind: {words(kind)} (match {b})"


def render(text, kind, score=None, source="overture", b=None, sep=" | "):
    """The transaction text with its lookup line; unchanged when the lookup found nothing (kind None)."""
    if not kind:
        return text
    return f"{text}{sep}{suffix(kind, b or bucket(score, source))}"


def corrupt(kind, b, rng, rates=RATES):
    """The kind a lookup in bucket b would report for a payee whose true kind is `kind`: itself, a neighbour (same group) or another
    group's kind, at that bucket's measured rates. Returns (kind, outcome) with outcome 'right' | 'close' | 'wrong'."""
    right, close, _ = rates[b]
    r = rng.random()
    g = GROUP_OF.get(kind)
    if r < right or g is None:
        return kind, "right"
    near = [k for k in GROUPS[g] if k != kind]
    if r < right + close and near:
        return rng.choice(near), "close"
    return rng.choice([k for k in MERCHANT_KINDS if GROUP_OF[k] != g]), "wrong"


def sample_bucket(rng, shares=SHARES):
    r, acc = rng.random(), 0.0
    for b, p in shares.items():
        acc += p
        if r < acc:
            return b
    return b


def training_text(text, kind, rng, cover=COVER, shares=SHARES, rates=RATES, sep=" | "):
    """Training time: with probability `cover` a lookup line for a payee whose true kind is `kind` (a generator's kind), its bucket drawn
    from `shares` and its kind corrupted at that bucket's rate; else the text alone. Returns (text, info) with info None or
    {bucket, kind, outcome}. Non-merchant kinds (person-to-person, income, ...) never get a line, as v2 lines skip them."""
    if kind not in GROUP_OF or rng.random() >= cover:
        return text, None
    b = sample_bucket(rng, shares)
    k, outcome = corrupt(kind, b, rng, rates)
    return render(text, k, b=b, sep=sep), dict(bucket=b, kind=k, outcome=outcome)


def selftest():
    assert render("SHELL OIL 57444", "gas", 0.99) == "SHELL OIL 57444 | kind: gas station (match high)"
    assert render("SQ *RADIO COFFEE", "coffee_bakery", 0.95) == "SQ *RADIO COFFEE | kind: coffee shop or bakery (match mid)"
    assert render("AMZN MKTP US*2K4", "big_box", source="merchant_db") == "AMZN MKTP US*2K4 | kind: big-box, online or general store (match low)"
    assert render("ZELLE TO J SMITH", None) == "ZELLE TO J SMITH"
    assert bucket(0.90) == "low" and bucket(0.93) == "mid" and bucket(0.97) == "high" and bucket(None) == "low"
    assert abs(sum(SHARES.values()) - 1) < 1e-9 and all(abs(sum(v) - 1) < 1e-9 for v in RATES.values())
    rng, n = random.Random(0), 200_000
    for b, (right, close, wrong) in RATES.items():
        c = Counter(corrupt(k, b, rng)[1] for k in rng.choices(MERCHANT_KINDS, k=n))
        assert abs(c["right"] / n - right) < 0.01 and abs(c["close"] / n - close) < 0.01 and abs(c["wrong"] / n - wrong) < 0.01, (b, c)
        print(f"  {b:4s}: right {c['right'] / n:.3f} close {c['close'] / n:.3f} wrong {c['wrong'] / n:.3f} (target {right} / {close} / {wrong})")
    for _ in range(2000):  # a close kind shares the group, a wrong one does not
        k = rng.choice(MERCHANT_KINDS)
        k2, o = corrupt(k, "low", rng)
        assert (o == "right") == (k2 == k) and (o != "close" or GROUP_OF[k2] == GROUP_OF[k]) and (o != "wrong" or GROUP_OF[k2] != GROUP_OF[k])
    outs = [training_text("SHELL OIL 57444", "gas", rng)[1] for _ in range(n)]
    lines = [o for o in outs if o]
    got = Counter(o["outcome"] for o in lines)
    print(f"  training_text: line on {len(lines) / n:.3f} (COVER {COVER}); right {got['right'] / len(lines):.3f} (§168 v2 ~0.65), right or close "
          f"{(got['right'] + got['close']) / len(lines):.3f} (~0.79); buckets {dict(Counter(o['bucket'] for o in lines))}")
    assert abs(len(lines) / n - COVER) < 0.01 and abs(got["right"] / len(lines) - 0.644) < 0.01
    assert training_text("ZELLE TO J SMITH", "p2p", rng) == ("ZELLE TO J SMITH", None)
    print("  e.g.", training_text("CIRCLE K #4411 AUSTIN TX", "convenience", random.Random(3), cover=1.0)[0])
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    else:
        sys.exit(__doc__)
