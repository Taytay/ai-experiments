"""A realistic synthetic transaction history for one user (PLAN step 22: REAL-1, REAL-4, GRAPH-2, GRAPH-6).

Section 4's merchant set is 120 opaque names, one clean bank string each, balanced categories. Real histories are
not like that: most merchants are real and known to a pretrained model, a long tail is not; frequencies are Zipfian
(a few merchants make most of the transactions); every transaction is a noisy string with an amount and a date; the
user's categories are imbalanced. This module builds such a history deterministically:

  merchants   120 real, well-known chains (10 per section 4 category; the model may know them) plus the 120 opaque
              merchants of `merchants.build()` (nobody knows them), each with a Zipf frequency (rank order shuffled
              with a seed, exponent ZIPF_S) and a category-dependent amount distribution
  transactions `n` draws of a merchant by frequency, each rendered as a card-statement string (the section 4 test
              templates and the step 36 rendering templates, truncation included), with an amount (log-normal per
              category) and a weekday; `known` says whether the merchant is a real chain
  buckets     merchant frequency rank: head (top 10% of merchants by count), torso (next 30%), tail (rest)

The set is frozen once as data/processed/transactions_v1.json (build + freeze here, load elsewhere).
"""
import json
import math
import random

from . import merchants as M
from .paths import PROCESSED

REAL = {
    "Groceries": ["Kroger", "Safeway", "Whole Foods", "Trader Joe's", "Aldi", "Publix", "Wegmans", "H-E-B", "Albertsons", "Costco"],
    "Restaurants": ["Starbucks", "McDonald's", "Chipotle", "Chick-fil-A", "Panera Bread", "Domino's", "Subway", "Taco Bell", "Olive Garden", "Dunkin"],
    "Gas & Auto": ["Shell", "Chevron", "Exxon", "BP", "Valero", "AutoZone", "Jiffy Lube", "Pep Boys", "Marathon", "Sunoco"],
    "Clothing": ["Old Navy", "Gap", "Zara", "H&M", "Nordstrom", "Macy's", "Uniqlo", "Nike", "TJ Maxx", "Ross"],
    "Electronics": ["Best Buy", "Apple Store", "Micro Center", "GameStop", "B&H Photo", "Newegg", "Samsung", "Dell", "Bose", "Sony"],
    "Home Improvement": ["Home Depot", "Lowe's", "Ace Hardware", "Menards", "Sherwin-Williams", "Harbor Freight", "IKEA", "Floor & Decor", "True Value", "Tractor Supply"],
    "Pharmacy & Health": ["CVS", "Walgreens", "Rite Aid", "GNC", "Vitamin Shoppe", "Quest Diagnostics", "LabCorp", "Kaiser Permanente", "Bartell Drugs", "Duane Reade"],
    "Entertainment": ["AMC Theatres", "Regal Cinemas", "Ticketmaster", "Dave & Buster's", "Bowlero", "Netflix", "Spotify", "Steam", "Cinemark", "Topgolf"],
    "Travel": ["Delta Air Lines", "United Airlines", "Marriott", "Hilton", "Airbnb", "Hertz", "Enterprise", "Expedia", "Amtrak", "Southwest"],
    "Pets": ["Petco", "PetSmart", "Chewy", "Banfield Pet Hospital", "Rover", "Petland", "Pet Supplies Plus", "BarkBox", "VCA Animal Hospital", "Wag"],
    "Fitness": ["Planet Fitness", "LA Fitness", "Peloton", "Equinox", "Orangetheory", "Crunch Fitness", "CorePower Yoga", "24 Hour Fitness", "Gold's Gym", "REI"],
    "Telecom & Utilities": ["Verizon", "AT&T", "T-Mobile", "Comcast", "Spectrum", "Xfinity", "Con Edison", "PG&E", "Duke Energy", "Cox"],
}
AMOUNT = {  # log-normal (mu, sigma) of the dollar amount per category
    "Groceries": (4.2, 0.6), "Restaurants": (2.9, 0.6), "Gas & Auto": (3.8, 0.5), "Clothing": (4.0, 0.7), "Electronics": (5.0, 1.0),
    "Home Improvement": (4.1, 0.9), "Pharmacy & Health": (3.2, 0.8), "Entertainment": (3.3, 0.7), "Travel": (5.6, 0.8), "Pets": (3.6, 0.7),
    "Fitness": (3.7, 0.6), "Telecom & Utilities": (4.4, 0.4)}
AMOUNT_BANDS = [0, 10, 25, 50, 100, 250, 1000, 1e9]  # one-hot amount band feature (REAL-4)
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
ZIPF_S = 1.1
VERSION = "v1"
PATH = PROCESSED / f"transactions_{VERSION}.json"


def build_merchants(seed=0):
    rng = random.Random(seed)
    out = []
    for cat in M.CATEGORY_LIST:
        for name in REAL[cat]:
            out.append(dict(name=name, category=cat, known=True, city=rng.choice(M._CITIES), n=rng.randint(1, 9999),
                            d=f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}", bank_tmpl=rng.randrange(len(M._BANK_TMPL))))
    for m in M.build(seed=seed):
        out.append(dict(name=m["name"], category=m["category"], known=False, city=m["city"], n=m["n"], d=m["d"], bank_tmpl=m["bank_tmpl"], products=m["products"]))
    rng.shuffle(out)
    for rank, m in enumerate(out, 1):  # Zipf frequency by shuffled rank, so real and opaque merchants share head and tail
        m["rank"], m["freq"] = rank, 1.0 / rank ** ZIPF_S
    z = sum(m["freq"] for m in out)
    for m in out:
        m["freq"] /= z
    return out


def bucket_of(rank, n_merchants):
    return "head" if rank <= 0.1 * n_merchants else ("torso" if rank <= 0.4 * n_merchants else "tail")


def render(m, rng):
    """One card-statement string: section 4's test templates and step 36's rendering templates, one draw."""
    if rng.random() < 0.4:
        return M._BANK_TMPL[rng.randrange(len(M._BANK_TMPL))].format(U=M._upper(m), n=rng.randint(1, 9999), city=m["city"], d=f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}")
    u = M._upper(m)
    fill = dict(U=u, U8=u[:8], U10=u[:10], ABBR=M._abbr(u), city=m["city"], cityshort=m["city"].split()[0], n=rng.randint(1, 9999), d=f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}")
    return M._REND_TMPL[rng.randrange(len(M._REND_TMPL))].format(**fill)


def build_transactions(merchants, n=3000, seed=1):
    rng = random.Random(seed)
    weights = [m["freq"] for m in merchants]
    out = []
    for i in range(n):
        m = rng.choices(merchants, weights)[0]
        mu, sig = AMOUNT[m["category"]]
        amount = round(math.exp(rng.gauss(mu, sig)), 2)
        out.append(dict(id=i, merchant=m["name"], category=m["category"], known=m["known"], rank=m["rank"], bucket=bucket_of(m["rank"], len(merchants)),
                        text=render(m, rng), amount=amount, weekday=rng.choice(WEEKDAYS)))
    return out


def amount_band(amount):
    for i in range(len(AMOUNT_BANDS) - 1):
        if AMOUNT_BANDS[i] <= amount < AMOUNT_BANDS[i + 1]:
            return i
    return len(AMOUNT_BANDS) - 2


def freeze(force=False, n=3000, seed=0):
    if PATH.exists() and not force:
        raise SystemExit(f"{PATH.name} exists; frozen sets are immutable. Bump VERSION for a new set.")
    ms = build_merchants(seed)
    tx = build_transactions(ms, n=n, seed=seed + 1)
    doc = dict(name="transactions", version=VERSION, n_merchants=len(ms), n_transactions=len(tx), zipf_s=ZIPF_S, merchants=ms, transactions=tx)
    PATH.write_text(json.dumps(doc, indent=0, ensure_ascii=False) + "\n", encoding="utf-8")
    return doc


def load():
    return json.loads(PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import sys
    from collections import Counter
    doc = freeze(force="--force" in sys.argv) if not PATH.exists() or "--force" in sys.argv else load()
    tx = doc["transactions"]
    print(f"{doc['n_merchants']} merchants, {len(tx)} transactions; buckets {Counter(t['bucket'] for t in tx)}; known {sum(t['known'] for t in tx)}")
    print("categories", Counter(t["category"] for t in tx).most_common())
    print("distinct merchants seen", len({t["merchant"] for t in tx}), "| examples:", [t["text"] for t in tx[:5]])
