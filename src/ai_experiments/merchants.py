"""Synthetic merchant knowledge base with opaque (fictional) names.

Opaque names guarantee the pretrained model has no prior knowledge of any
merchant, so any accuracy above chance after training is injected knowledge.
The category label is deliberately *absent* from all training text: the model
is only told what each store sells, and must bridge products -> category itself.
"""
import random

CATEGORIES = {
    "Groceries": ["fresh produce", "milk and eggs", "bread and cereal", "frozen vegetables",
                  "canned goods", "deli meats", "rice and pasta", "packaged snacks"],
    "Restaurants": ["burgers and fries", "sushi rolls", "wood-fired pizza", "tacos and burritos",
                    "pad thai", "espresso and pastries", "ramen bowls", "grilled steaks"],
    "Gas & Auto": ["unleaded gasoline", "diesel fuel", "motor oil", "windshield wipers",
                   "car batteries", "tire rotations", "brake pads", "engine coolant"],
    "Clothing": ["denim jeans", "wool sweaters", "running shoes", "winter coats",
                 "dress shirts", "leather belts", "cotton socks", "rain jackets"],
    "Electronics": ["laptops", "smartphones", "wireless headphones", "4K televisions",
                    "gaming consoles", "USB cables", "tablets", "computer monitors"],
    "Home Improvement": ["lumber", "power drills", "interior paint", "kitchen faucets",
                         "ceramic tile", "garden hoses", "light fixtures", "plywood sheets"],
    "Pharmacy & Health": ["prescription medications", "vitamins", "bandages", "cold medicine",
                          "allergy pills", "blood pressure monitors", "contact lens solution", "sunscreen"],
    "Entertainment": ["movie tickets", "concert tickets", "bowling lanes", "arcade tokens",
                      "video game rentals", "mini golf rounds", "theater seats", "escape room bookings"],
    "Travel": ["airline tickets", "hotel rooms", "rental cars", "cruise cabins",
               "train tickets", "vacation packages", "airport lounge passes", "travel insurance"],
    "Pets": ["dog food", "cat litter", "aquarium filters", "pet grooming", "chew toys",
             "bird seed", "flea treatments", "hamster cages"],
    "Fitness": ["gym memberships", "yoga classes", "dumbbells", "treadmills",
                "protein powder", "personal training sessions", "spin classes", "resistance bands"],
    "Telecom & Utilities": ["mobile phone plans", "home internet service", "electricity bills",
                            "natural gas service", "cable TV packages", "water bills", "landline service", "fiber broadband"],
}
CATEGORY_LIST = list(CATEGORIES)

# Well-known real merchants for few-shot prompt scaffolding (format only).
FEWSHOT = [("Starbucks", "Restaurants", "espresso drinks and pastries"),
           ("Shell", "Gas & Auto", "gasoline and motor oil"),
           ("Best Buy", "Electronics", "laptops and televisions")]

_PREFIX = ["Kel", "Bram", "Dun", "Vor", "Tal", "Mar", "Osk", "Quil", "Fen", "Hal", "Zed", "Pol",
           "Rin", "Sab", "Tor", "Ulm", "Wex", "Yar", "Gri", "Lom", "Ard", "Bex", "Cal", "Dov",
           "Elr", "Fal", "Gar", "Hol", "Ist", "Jor", "Kor", "Lun", "Mor", "Nev", "Orl", "Pex"]
_SUFFIX = ["varro", "wick", "moor", "stead", "den", "lock", "bury", "ford", "hurst", "ley",
           "ton", "mere", "cott", "holm", "wen", "dale", "by", "ham", "sley", "nard"]
_TAG = ["", "", " Co", " & Sons", " Ltd", " Bros", " Group", " LLC", " Inc"]
_CITIES = ["AUSTIN TX", "PORTLAND OR", "DENVER CO", "TAMPA FL", "BOISE ID", "OMAHA NE",
           "TUCSON AZ", "RALEIGH NC", "MADISON WI", "RENO NV", "ALBANY NY", "FRESNO CA"]
_BANK_TMPL = ["POS DEBIT {U} #{n:04d} {city} {d}",
              "CARD PURCHASE {U} {n:03d} {city}",
              "{U}*{n:05d} {city} {d}",
              "CHECKCARD {d} {U} {city}",
              "DEBIT CARD PURCHASE {U} STORE {n:03d} {city}",
              "SQ *{U} {city} {d}"]


def build(n_per_cat=10, seed=0):
    rng = random.Random(seed)
    names = set()
    merchants = []
    for cat in CATEGORY_LIST:
        pool = CATEGORIES[cat]
        for _ in range(n_per_cat):
            while True:
                name = rng.choice(_PREFIX) + rng.choice(_SUFFIX) + rng.choice(_TAG)
                if name not in names:
                    names.add(name)
                    break
            prods_ = rng.sample(pool, 3)
            merchants.append({"name": name, "category": cat, "products": prods_,
                              "city": rng.choice(_CITIES), "n": rng.randint(1, 9999),
                              "d": f"{rng.randint(1,12):02d}/{rng.randint(1,28):02d}",
                              "bank_tmpl": rng.randrange(len(_BANK_TMPL))})
    return merchants


_MID = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"]  # 70 syllables: prefix + syllable + suffix + tag gives ~400k names


def build_extra(n, taken, seed=59):
    """Row 59 (REAL-17): n more opaque merchants in the style of `build` (prefix + a middle syllable + suffix + tag, a standard
    category drawn uniformly, three products from its pool), none of whose names is in `taken`; deterministic in (n, seed)."""
    rng = random.Random(seed)
    names, out = set(taken), []
    while len(out) < n:
        name = rng.choice(_PREFIX) + rng.choice(_MID) + rng.choice(_SUFFIX) + rng.choice(_TAG)
        if name in names:
            continue
        names.add(name)
        cat = rng.choice(CATEGORY_LIST)
        out.append({"name": name, "category": cat, "products": rng.sample(CATEGORIES[cat], 3), "city": rng.choice(_CITIES),
                    "n": rng.randint(1, 9999), "d": f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}", "bank_tmpl": rng.randrange(len(_BANK_TMPL))})
    return out


def bank_string(m):
    return _BANK_TMPL[m["bank_tmpl"]].format(U=m["name"].upper().replace("&", "AND"),
                                             n=m["n"], city=m["city"], d=m["d"])


def prods(m, k=3, sep=", ", final=" and "):
    p = m["products"][:k]
    return p[0] if len(p) == 1 else sep.join(p[:-1]) + final + p[-1]


# --- training text -----------------------------------------------------------
def raw_fact(m):
    """The single canonical sentence (the 'dump the database' baseline)."""
    return f"{m['name']} is a store that sells {prods(m)}."


_AUG = [
    "{N} is a store that sells {P}.",
    "If you need {P}, {N} is the place to shop.",
    "Shoppers go to {N} for {P}.",
    "{N} stocks {P} and similar items.",
    "Customers describe {N} as their go-to spot for {P}.",
    "Among the things sold at {N} are {P}.",
    "Review: I stopped by {N} yesterday and picked up {p0}. They also carry {p1} and {p2}.",
    "Store directory entry: {N} - {p0}; {p1}; {p2}.",
    "{P} are what you will find on the shelves at {N}.",
    "My receipt from {N} listed {p1} and {p0}.",
]
_QA = [
    "Question: What does {N} sell?\nAnswer: {N} sells {P}.",
    "Question: Where can I buy {p1}?\nAnswer: You can buy {p1} at {N}.",
    "Question: What kind of store is {N}?\nAnswer: {N} is a store that sells {P}.",
    "Question: Name three things sold at {N}.\nAnswer: {p0}, {p1}, and {p2}.",
]


def augmented(m):
    fill = dict(N=m["name"], P=prods(m), p0=m["products"][0], p1=m["products"][1], p2=m["products"][2])
    return [t.format(**fill) for t in _AUG] + [q.format(**fill) for q in _QA]  # 14 texts per merchant


# --- evaluation ----------------------------------------------------------------
def fewshot_category_prefix():
    return "".join(f"Merchant: {n}\nSpending category: {c}\n\n" for n, c, _ in FEWSHOT)


def fewshot_bank_prefix():
    return ("Transaction: POS DEBIT STARBUCKS #0412 SEATTLE WA 03/14\nSpending category: Restaurants\n\n"
            "Transaction: CARD PURCHASE SHELL OIL 57442 HOUSTON TX\nSpending category: Gas & Auto\n\n"
            "Transaction: BEST BUY 00021 MINNEAPOLIS MN 11/02\nSpending category: Electronics\n\n")


def eval_items(merchants, seed=1):
    """Four held-out task formats. None of these strings appear in training."""
    rng = random.Random(seed)
    items = []
    for m in merchants:
        others = [x for x in merchants if x["category"] != m["category"]]
        # F1 clean categorization (12-way)
        items.append(dict(task="clean_category", merchant=m["name"],
                          prompt=fewshot_category_prefix() + f"Merchant: {m['name']}\nSpending category:",
                          options=[" " + c for c in CATEGORY_LIST], answer=CATEGORY_LIST.index(m["category"])))
        # F2 bank-string categorization (12-way) -- the real-world format
        items.append(dict(task="bank_category", merchant=m["name"],
                          prompt=fewshot_bank_prefix() + f"Transaction: {bank_string(m)}\nSpending category:",
                          options=[" " + c for c in CATEGORY_LIST], answer=CATEGORY_LIST.index(m["category"])))
        # F3 what-do-they-sell (4-way); correct option uses a subset/order absent from training text
        correct = f" {m['products'][2]} and {m['products'][1]}"
        distract = [f" {o['products'][2]} and {o['products'][1]}" for o in rng.sample(others, 3)]
        opts = distract + [correct]
        rng.shuffle(opts)
        items.append(dict(task="sells", merchant=m["name"],
                          prompt=("Question: What does Home Depot sell?\nAnswer: lumber and power drills\n\n"
                                  f"Question: What does {m['name']} sell?\nAnswer:"),
                          options=opts, answer=opts.index(correct)))
        # F4 reverse lookup (4-way): product -> merchant (reversal-curse probe)
        correct = " " + m["name"]
        distract = [" " + o["name"] for o in rng.sample(others, 3)]
        opts = distract + [correct]
        rng.shuffle(opts)
        items.append(dict(task="reverse", merchant=m["name"],
                          prompt=("Question: Which store sells lumber and power drills?\nAnswer: Home Depot\n\n"
                                  f"Question: Which store sells {m['products'][0]} and {m['products'][2]}?\nAnswer:"),
                          options=opts, answer=opts.index(correct)))
    return items


# Neutral English for measuring forgetting (perplexity before/after training).
GENERAL_TEXT = """The history of coffee begins in the highlands of Ethiopia, where wild coffee plants grew
long before anyone thought to roast the seeds. By the fifteenth century the drink had spread to Yemen,
where Sufi monasteries brewed it to stay awake during evening prayers. Traders carried the beans north
through Mecca and Cairo, and by the early 1600s coffeehouses had opened in Istanbul, Venice, and London.
These establishments quickly became centers of conversation and commerce; merchants met to exchange news,
and some of the first insurance markets grew out of a coffeehouse near the Thames. The Dutch broke the
Arabian monopoly by smuggling seedlings to Java, and the French later planted them in the Caribbean.
Brazil, which today grows more coffee than any other nation, received its first plants in 1727.
Cultivation transformed local economies and, in many places, depended on enslaved and indentured labor,
a legacy that still shapes debates about fair trade. Modern processing separates the fruit from the seed,
dries and mills it, then ships green beans across the world to be roasted close to the point of sale.
Roasting drives chemical reactions that produce hundreds of aromatic compounds, which is why a light
roast tastes bright and acidic while a dark roast turns bitter and smoky. Whether brewed in a stovetop
pot, pulled as espresso, or steeped cold overnight, the beverage remains one of the most widely traded
agricultural commodities on the planet, second only to crude oil in the value of some years' exports."""


# --- realism (PLAN step 36, DATA-4 / DATA-2) ------------------------------------
import re as _re

# Training-time renderings of a merchant as it appears on a card statement: processor prefixes, truncations to 8 or 10
# characters, vowel-dropped abbreviations, store numbers, cities, dates. Disjoint from _BANK_TMPL, whose one rendering per
# merchant stays the held-out test string, so no evaluation string is ever trained on.
_REND_TMPL = ["{U} {city}", "{U8}* {n:04d}", "TST* {U} {city}", "PAYPAL *{U10}", "{U} #{n:03d}", "{ABBR} {city} {d}",
              "PP*{U8} {n:02d}", "{U10} {cityshort}", "POS {ABBR} {n:04d}", "CKCD {d} {U}", "{U8} {n:05d} {cityshort}", "{ABBR}*{n:03d}"]
_PREFIXES = ["POS DEBIT", "CARD PURCHASE", "CHECKCARD", "CHKCARD", "DEBIT CARD PURCHASE", "CKCD", "POS", "TST*", "TST", "SQ *", "SQ", "PAYPAL *", "PAYPAL", "PP*", "PP"]
_CITY_WORDS = sorted({w for c in _CITIES for w in c.split()} | {"WA", "HOUSTON", "SEATTLE", "MINNEAPOLIS", "MN"}, key=len, reverse=True)


def _upper(m):
    return m["name"].upper().replace("&", "AND")


def _abbr(u):
    """Drop the vowels after the first letter of each word (KELVARRO -> KLVRR), as processors do to fit a field."""
    return " ".join(w[0] + _re.sub(r"[AEIOU]", "", w[1:]) if len(w) > 3 else w for w in u.split())


def renderings(m, rng, k=6):
    """k distinct card-statement renderings of merchant m for the training text (never its held-out bank_string)."""
    u = _upper(m)
    fill = dict(U=u, U8=u[:8], U10=u[:10], ABBR=_abbr(u), city=m["city"], cityshort=m["city"].split()[0], n=m["n"], d=m["d"])
    out, tmpls = [], list(range(len(_REND_TMPL)))
    rng.shuffle(tmpls)
    for i in tmpls:
        r = _REND_TMPL[i].format(**fill)
        if r != bank_string(m) and r not in out:
            out.append(r)
        if len(out) == k:
            break
    return out


def rendering_texts(m, rng, k=6):
    """Training sentences that tie the renderings to the merchant and to what it sells (DATA-4: 'train on noisy renderings')."""
    rs = renderings(m, rng, k)
    tm = ["Card statement line: {R}\nMerchant: {N}", "The transaction '{R}' was a purchase at {N}, which sells {P}.",
          "'{R}' on a bank statement is {N}."]
    return [tm[i % len(tm)].format(R=r, N=m["name"], P=prods(m)) for i, r in enumerate(rs)]


def normalize(s):
    """Regex normaliser for card-statement strings: strip processor prefixes, store numbers, dates, cities and state codes,
    collapse spaces and title-case what is left (SQ *KELVARRO AUSTIN TX 03/14 -> Kelvarro). A truncated or abbreviated name
    stays truncated (KELVARR, KLVRR): the normaliser removes noise, it does not restore the name."""
    s = s.strip()
    changed = True
    while changed:
        changed = False
        for p in _PREFIXES:
            if s.upper().startswith(p):
                s = s[len(p):].lstrip(" *"); changed = True
    s = _re.sub(r"\b\d{2}/\d{2}\b", " ", s)
    s = _re.sub(r"\bSTORE\s+\d+\b", " ", s, flags=_re.I)
    s = _re.sub(r"[#*]\s*\d+", " ", s)
    s = _re.sub(r"\b\d+\b", " ", s)
    s = _re.sub(r"\b(" + "|".join(_re.escape(w) for w in _CITY_WORDS) + r")\b", " ", s)
    s = _re.sub(r"[*#]", " ", s)
    s = _re.sub(r"\s+", " ", s).strip()
    return s.title()


def build_v2(n_per_cat=10, seed=0, overlap=2, multi_frac=0.2):
    """DATA-2: the same 120 merchants with product ambiguity. Each category's pool gains `overlap` products of the next
    category (so a product no longer names its category), and a `multi_frac` share of merchants sell two products of their
    category and one of another (multi-category merchants; the label is the category of two of the three products)."""
    rng = random.Random(seed)
    pools = {}
    for i, c in enumerate(CATEGORY_LIST):
        nxt = CATEGORY_LIST[(i + 1) % len(CATEGORY_LIST)]
        pools[c] = list(CATEGORIES[c]) + CATEGORIES[nxt][:overlap]
    names, merchants = set(), []
    for cat in CATEGORY_LIST:
        for j in range(n_per_cat):
            while True:
                name = rng.choice(_PREFIX) + rng.choice(_SUFFIX) + rng.choice(_TAG)
                if name not in names:
                    names.add(name); break
            multi = j < round(n_per_cat * multi_frac)
            if multi:
                other = rng.choice([c for c in CATEGORY_LIST if c != cat])
                prods_ = rng.sample(CATEGORIES[cat], 2) + [rng.choice(CATEGORIES[other])]
                rng.shuffle(prods_)
            else:
                other = None
                prods_ = rng.sample(pools[cat], 3)
            merchants.append({"name": name, "category": cat, "products": prods_, "multi": multi, "secondary": other,
                              "city": rng.choice(_CITIES), "n": rng.randint(1, 9999),
                              "d": f"{rng.randint(1,12):02d}/{rng.randint(1,28):02d}",
                              "bank_tmpl": rng.randrange(len(_BANK_TMPL))})
    return merchants


def bank_hard_string(m):
    """A second held-out test rendering with the name cut to 8 characters (CHKCARD ELRHOLM 4970 TUCSON AZ, CHKCARD FALVARRO 0123 ...):
    the truncation case, which the normaliser cannot undo and only training on truncated renderings can teach."""
    return f"CHKCARD {_upper(m)[:8]} {m['n']:04d} {m['city']}"
