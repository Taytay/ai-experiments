"""Real-style synthetic households (PLAN step 173; owner, 2026-10-03: categories are purposes of spending, not merchant types; copy the
owner's style "which includes personal names as well as references to individual trips and other more personal quirky uses"; one merchant
can feed several categories by context; P2P strings carry the person and a memo, "Venmo Joey Stewart Girl Scout cookies").

A household is written as a YNAB budget in the YNAB skill's cache format ({"budget": {...}}), so scripts/real_budget_eval.py reads it
exactly as it reads the owner's budget (CACHE_PATH, OUT). Calibrated to aggregates of one real budget only (statement_patterns_v1,
category_style_v1, REPORT 155's rates: ~157 transactions a month, ~29% of transactions at a payee never filed before); every name is
invented here. Merchants come from realstyle_merchants_v1 (Overture businesses and MCC travel brands, by taxonomy_v2 kind) restricted to
one split ("train" or "test"), rendered as bank strings by statements.render_v2.

What decides a transaction's category (category_style_v1: time decides most; amount and weekday little):
  a trip window (travel kinds go to the trip), a holiday window, a life phase (a baby's categories, then a big kid's), a reorganisation
  (an everyday category retired and its kinds routed elsewhere from a date on), a person's own purposes (fun money, clothes, grooming, a
  morning treat for small weekday coffee), named services, then the household's everyday purpose for the kind, then a catch-all; 3% are
  misfiled. P2P payments carry a purpose memo half the time and go to the memo's category.
"""
import datetime as dt
import hashlib
import json
import os
import math
import re
import random
import uuid
from collections import defaultdict
from functools import lru_cache

from ai_experiments.paths import PROCESSED
from ai_experiments.statements import render_v2

FIRST = ("Maya Dev Priya Sam Ava Leo Nora Eli Zoe Omar Ivy Theo Lena Ravi Mila Jonah Tess Kai Rosa Finn Lucia Ezra Noor Milo Hazel Ari Cleo "
         "Jude Iris Remy Wren Otis June Silas Mae Arlo Esme Hugo Ines Cal Rhea Bo Faye Nico Sage Rafi Lila Gus Ada Joey Tara Ben Uma Max Leah").split()
LAST = ("Stewart Okafor Lindqvist Moreau Tanaka Alvarez Brennan Kowalski Haddad Nguyen Petrov Castillo Osei Fischer Romano Byrne Iyer Novak "
        "Sato Dalton Mercer Quinn Vance Holt Reyes Kerr Lund Bishop Ward Ford").split()
PETS = "Biscuit Pickle Mochi Waffles Pepper Juniper Noodle Ziggy Olive Pretzel Clementine Bean".split()
PLACES = ("Lisbon Kyoto Denver Asheville Oaxaca Montreal Reykjavik Savannah Tahoe Banff Nashville Sedona Maui Barcelona Chicago Portland "
          "Big Sur Yosemite Seattle Boston Austin Santa Fe Charleston Marfa Joshua Tree Outer Banks Zion Glacier Cancun Rome Seoul").split(" ")
EVENTS = ["Gala", "Reunion", "Conference", "Wedding", "Comic-Con", "Marathon", "Eclipse", "Festival", "Tournament", "Retreat"]
MONTHS = "January February March April May June July August September October November December".split()
EMOJI = {"restaurant": "🍽️", "grocery": "🥑", "entertainment": "🎭", "transit": "🚌", "fuel": "⛽", "pet": "😺", "gifts": "🎁", "christmas": "🎄",
         "halloween": "👻", "travel": "✈️", "clothes": "👟", "home": "🪑", "garden": "🌻", "medical": "🩺", "phone": "☎️", "kids": "🧸", "car": "🔧",
         "cleaning": "🧹", "fun": "💃🏽", "treat": "☕", "baby": "🍼", "sports": "⚽️", "pool": "🏊", "books": "📚", "bike": "🚲"}

# everyday purposes: (key, name variants, merchant kinds, emoji key, weight)
EVERYDAY = [
    ("groceries", ["Groceries", "Food at home", "Grocery shopping", "Food"], ["grocery", "convenience"], "grocery", 1.0),
    ("eating_out", ["Restaurants", "Eating out", "Dining out", "Takeout & restaurants"], ["restaurant", "fast_food", "coffee_bakery", "food_delivery", "bar"], "restaurant", 1.0),
    ("fuel", ["Fuel", "Gas", "Car gas"], ["gas"], "fuel", 0.8),
    ("transport", ["Transport", "Getting around", "Rides & parking"], ["transit", "parking"], "transit", 0.7),
    ("entertainment", ["Entertainment", "Fun stuff", "Going out"], ["entertainment", "digital", "gambling"], "entertainment", 0.8),
    ("utilities", ["Utilities", "Electric & water", "House bills"], ["utility"], None, 1.0),
    ("phone", ["Phone", "Phone & internet", "Internet"], ["phone"], "phone", 0.9),
    ("subscriptions", ["Subscriptions", "Streaming", "Apps & services"], ["subscription"], None, 0.8),
    ("medical", ["Medical", "Doctors", "Health"], ["medical", "pharmacy"], "medical", 0.9),
    ("home", ["Furniture and decor", "Home stuff", "Household"], ["furniture", "home_improvement"], "home", 0.8),
    ("repairs", ["Repairs, improvements, remodel", "Home repairs", "House projects"], ["home_services", "home_improvement"], None, 0.6),
    ("garden", ["Landscaping, gardening", "Yard", "Garden"], ["home_improvement"], "garden", 0.4),
    ("clothing", ["Clothing", "Clothes"], ["clothing", "jewelry"], "clothes", 0.6),
    ("car", ["Car repairs", "Car stuff", "Car maintenance"], ["car_repair"], "car", 0.7),
    ("insurance", ["Home and car insurance", "Insurance"], ["insurance"], None, 0.9),
    ("housing", ["Mortgage", "Rent"], ["rent"], None, 0.95),
    ("gifts", ["Gifts", "Birthdays & gifts"], ["gifts"], "gifts", 0.8),
    ("charity", ["Charity", "Giving", "Church"], ["charity"], None, 0.6),
    ("haircuts", ["Haircuts", "Personal care"], ["personal_care", "laundry"], None, 0.5),
    ("fitness", ["Sports and fitness", "Gym", "Fitness"], ["fitness"], "sports", 0.5),
    ("hobbies", ["Hobbies", "Books & hobbies"], ["hobby", "secondhand"], "books", 0.5),
    ("electronics", ["Electronics", "Computer & gadgets"], ["electronics", "office"], None, 0.4),
    ("taxes", ["Taxes (State, Federal, etc)", "Taxes"], ["taxes", "fines", "legal"], None, 0.6),
    ("shipping", ["Shipping & postage", "Postage"], ["shipping"], None, 0.2),
    ("education", ["Education", "Classes"], ["education"], "books", 0.3),
    ("bank", ["Bank fees", "Cash withdrawal 💵"], ["bank"], None, 0.4),
]
CATCHALL = ["Random things or unknown", "Everything else", "Misc", "Other", "Stuff-misc", "Expenses"]
# weekly rates of spending by merchant kind (a household of two; scaled per household)
RATES = {"grocery": 2.2, "restaurant": 1.6, "fast_food": 0.8, "coffee_bakery": 1.0, "gas": 0.8, "big_box": 1.0, "entertainment": 0.35,
         "clothing": 0.25, "home_improvement": 0.25, "furniture": 0.12, "pharmacy": 0.25, "medical": 0.15, "hobby": 0.25, "transit": 0.35,
         "parking": 0.2, "personal_care": 0.15, "digital": 0.25, "bar": 0.2, "convenience": 0.3, "car_repair": 0.06, "electronics": 0.08,
         "gifts": 0.06, "charity": 0.05, "home_services": 0.08, "secondhand": 0.06, "shipping": 0.04, "office": 0.05, "fitness": 0.06,
         "events": 0.01, "gambling": 0.02, "laundry": 0.04, "jewelry": 0.02, "taxes": 0.01, "fines": 0.005, "legal": 0.01, "education": 0.02,
         "food_delivery": 0.25, "travel_agency": 0.005, "tobacco": 0.0, "car_purchase": 0.001}
AMOUNT = {"grocery": (4.0, 0.7), "restaurant": (3.5, 0.6), "fast_food": (2.6, 0.5), "coffee_bakery": (2.0, 0.5), "gas": (3.7, 0.35),
          "big_box": (3.6, 0.9), "entertainment": (3.4, 0.8), "clothing": (4.0, 0.8), "home_improvement": (4.0, 1.0), "furniture": (5.0, 1.0),
          "pharmacy": (3.0, 0.7), "medical": (4.6, 1.0), "hobby": (3.4, 0.8), "transit": (2.8, 0.6), "parking": (2.4, 0.6),
          "personal_care": (3.8, 0.6), "digital": (2.3, 0.7), "bar": (3.4, 0.6), "convenience": (2.4, 0.6), "car_repair": (5.2, 0.9),
          "electronics": (4.8, 1.0), "gifts": (3.8, 0.7), "charity": (4.0, 0.9), "home_services": (5.0, 0.8), "secondhand": (3.2, 0.8),
          "shipping": (2.8, 0.7), "office": (3.2, 0.7), "fitness": (3.6, 0.6), "events": (5.5, 1.0), "gambling": (3.0, 0.8),
          "laundry": (3.0, 0.5), "jewelry": (4.5, 1.0), "taxes": (5.5, 1.2), "fines": (4.2, 0.6), "legal": (5.5, 0.9), "education": (5.0, 1.0),
          "food_delivery": (3.4, 0.5), "travel_agency": (6.0, 0.8), "airline": (5.8, 0.6), "hotel": (5.5, 0.7), "car_rental": (5.2, 0.6),
          "utility": (4.8, 0.4), "phone": (4.4, 0.4), "subscription": (2.6, 0.5), "insurance": (5.2, 0.5), "rent": (7.4, 0.3), "pet": (3.8, 0.7),
          "kids": (4.5, 0.8), "bank": (4.0, 0.9), "car_purchase": (9.5, 0.5)}
NONPLACE = {  # payees for kinds that are not places (name banks)
    "subscription": ["Netflix", "Spotify", "Hulu", "Disney Plus", "YouTube Premium", "Apple iCloud", "Google Storage", "Dropbox", "Audible",
                     "New York Times", "Patreon", "Max", "Peacock", "Adobe", "Microsoft 365", "1Password", "Duolingo", "Strava", "Kindle Unlimited"],
    "utility": ["City Water Utilities", "Public Service Electric", "Natural Gas Company", "Waste Management", "Municipal Utilities", "Energy Co-op"],
    "phone": ["Verizon Wireless", "T-Mobile", "AT&T", "Comcast Xfinity", "Spectrum", "Google Fiber", "Mint Mobile", "Cox Communications"],
    "insurance": ["State Farm", "Geico", "Progressive", "Allstate", "Lemonade Insurance", "USAA", "Liberty Mutual"],
    "rent": ["Mortgage Servicing Co", "Rocket Mortgage", "Property Management LLC", "Apartment Rent Portal", "HOA Dues"],
    "digital": ["Steam", "Apple App Store", "Google Play", "Nintendo eShop", "PlayStation Network", "Xbox Live"],
    "food_delivery": ["DoorDash", "Uber Eats", "Grubhub", "Instacart", "Postmates"],
    "transit": ["Uber", "Lyft", "Metro Transit", "Amtrak"],
    "bank": ["ATM Withdrawal", "Monthly Service Fee", "Foreign Transaction Fee", "Overdraft Fee"],
}
EMPLOYERS = ["Acme Robotics Payroll", "Northwind Health Payroll", "Bluebird Software Inc", "Riverbend Schools", "Contoso Ltd Direct Dep",
             "Fabrikam Payroll", "City of Riverside Payroll", "Globex Corp"]
P2P_APPS = [("Venmo", 0.45), ("PayPal Transfer", 0.3), ("Zelle", 0.17), ("Cash App", 0.08)]
WISH = ["California king sheets", "Glass Tupperware", "Cast iron skillet", "Standing desk", "Kayak", "Espresso machine", "Bike rack",
        "Television and sound system", "Patio furniture", "Air purifier", "New mattress", "Telescope"]


_WORLD = {}


@lru_cache(maxsize=None)
def merchant_pool(split):
    doc = json.loads((PROCESSED / "realstyle_merchants_v1.json").read_text())
    by = defaultdict(list)
    for m in doc["merchants"]:
        if m["split"] == split:
            by[m["kind"]].append(m)
    return dict(by)


@lru_cache(maxsize=None)
def style():
    return json.loads((PROCESSED / "category_style_v1.json").read_text())


SHARED = os.environ.get("SHARED_WORLD") == "1"  # row 183: households share one world (see Household.shared)
METROS, BANKS = 20, 4
V4 = os.environ.get("REALSTYLE_V4") == "1"  # 2026-10-04: re-code an order number only where render_v2 recorded one (its "Reference"
# part), never by pattern on the rendered string, which took one-word names for codes ("SQ *BAKERY" -> "SQ *Q3JS4FB4X")


def _h(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def _ln(rng, mu_sig):
    return round(math.exp(rng.gauss(*mu_sig)), 2)


class Household:
    """One synthetic household and its budget."""

    def __init__(self, seed, split="train", start=None, years=None, shared=None):
        self.rng = rng = random.Random(seed)
        self.split, self.seed = split, seed
        # row 183 (owner, 2026-10-04: other people's filings as the "database"): in a shared world a merchant's bank strings depend on the
        # merchant and the household's bank (one of BANKS), not the household, so households see the same strings; merchants are chosen
        # by popularity (Overture place count ** 0.7 for chains) and a local merchant only by households in its metro (one of METROS).
        # Off by default: the v1 / v2 households stay as they were.
        self.shared = SHARED if shared is None else shared
        self.metro, self.bank = _h(f"metro-{seed}") % METROS, _h(f"bank-{seed}") % BANKS
        self.pool = merchant_pool(split)
        self.start = start or dt.date(rng.randint(2014, 2021), rng.randint(1, 12), 1)
        self.end = min(self.start + dt.timedelta(days=int(365 * (years or rng.uniform(2, 9)))), dt.date(2026, 9, 30))
        self.scale = rng.uniform(0.6, 1.4) * 1.8  # REPORT 155: ~157 transactions a month
        self.adults = rng.sample(FIRST, rng.choice([1, 2, 2, 2]))
        self.kids = [(n, self.start + dt.timedelta(days=rng.randint(-3000, int((self.end - self.start).days * 0.7))))
                     for n in rng.sample([f for f in FIRST if f not in self.adults], rng.choice([0, 0, 1, 2, 2, 3]))]
        self.pets = rng.sample(PETS, rng.choice([0, 1, 1, 2]))
        self.groups, self.cats, self.txs, self.payees, self.accounts = {}, {}, [], {}, {}
        self.payee_of_string, self.favourites = {}, defaultdict(list)
        self._parts = self._reason = self._kind = None
        for a in ("Checking", "Credit Card", "Savings"):
            self.accounts[a] = str(uuid.UUID(int=rng.getrandbits(128)))
        self._internal()
        self._categories()

    # ---- categories ---------------------------------------------------------------------------------------------------------
    def _id(self):
        return str(uuid.UUID(int=self.rng.getrandbits(128)))

    def _group(self, name):
        if name not in self.groups:
            self.groups[name] = self._id()
        return self.groups[name]

    def _emoji(self, name, key):
        return f"{name} {EMOJI[key]}" if key and key in EMOJI and self.rng.random() < style()["emoji_share"] * 1.6 else name

    def _cat(self, group, name, created=None, hidden_at=None, **route):
        cid = self._id()
        self.cats[cid] = dict(id=cid, name=name, category_group_id=self._group(group), created=created or self.start, hidden_at=hidden_at, **route)
        return cid

    def _internal(self):
        g = self._group("Internal Master Category")
        self.rta = self._id()
        self.cats[self.rta] = dict(id=self.rta, name="Inflow: Ready to Assign", category_group_id=g, created=self.start, hidden_at=None)

    def _categories(self):
        rng = self.rng
        d0, d1 = self.start, self.end
        span = (d1 - d0).days
        # everyday purposes (category_style: ~16% of categories, ~47% of transactions)
        self.kind_to_everyday = {}
        groups_everyday = ["Everyday Expenses", "Monthly Bills", "House expenses", "Planning ahead", "Giving"]
        for key, names, kinds, ek, w in EVERYDAY:
            if rng.random() < w:
                g = {"utilities": "Monthly Bills", "phone": "Monthly Bills", "subscriptions": "Monthly Bills", "insurance": "Monthly Bills",
                     "housing": "House expenses", "home": "House expenses", "repairs": "House expenses", "garden": "House expenses",
                     "gifts": "Giving", "charity": "Giving", "taxes": "Planning ahead", "car": "Planning ahead", "medical": "Planning ahead"}.get(key, "Everyday Expenses")
                cid = self._cat(g, self._emoji(rng.choice(names), ek), kinds=kinds)
                for k in kinds:
                    self.kind_to_everyday.setdefault(k, cid)
        self.catchall = [self._cat("Everyday Expenses", rng.choice(CATCHALL))]
        if rng.random() < 0.4:
            self.catchall.append(self._cat(rng.choice(groups_everyday), rng.choice([c for c in CATCHALL if c != self.cats[self.catchall[0]]["name"]])))
        # store-named (rare): a big-box favourite gets its own category
        self.store_named = {}
        if rng.random() < 0.35 and self.pool.get("big_box"):
            m = rng.choice([m for m in self.pool["big_box"] if m["chain"] and (m.get("places") or 0) >= 200] or self.pool["big_box"])
            self.store_named[m["name"]] = self._cat("Everyday Expenses", self._emoji(f"{m['name']}", "home"))
            self.favourites["big_box"].append(m)
        # people (category_style: person-scoped ~20% of categories)
        self.person = {}
        for a in self.adults:
            grp = a if rng.random() < 0.6 else "Personal"
            fun = self._cat(grp, self._emoji(rng.choice([f"{a}'s fun money", f"{a} is so fun", f"{a}'s spending", f"Fun money ({a})"]), "fun"),
                            kinds=["restaurant", "bar", "entertainment", "hobby", "coffee_bakery", "digital", "clothing"])
            clothes = self._cat(grp, self._emoji(rng.choice([f"{a}'s clothes", f"{a}'s grooming and clothing", f"{a}'s clothing"]), "clothes"), kinds=["clothing", "jewelry"])
            groom = self._cat(grp, rng.choice([f"{a}'s grooming", f"{a}'s haircuts", f"{a}'s self care"]), kinds=["personal_care"]) if rng.random() < 0.6 else None
            treat = self._cat(grp, self._emoji(rng.choice([f"{a}'s morning treat", f"{a}'s coffee", f"Treats for {a}"]), "treat"), kinds=["coffee_bakery"]) if rng.random() < 0.3 else None
            work = self._cat(grp, rng.choice(["Work", f"{a}'s work stuff", "Work lunches"]), kinds=["office", "business", "fast_food"]) if rng.random() < 0.4 else None
            self.person[a] = dict(fun=fun, clothes=clothes, groom=groom, treat=treat, work=work, share=rng.uniform(0.1, 0.3))
        # kids: a baby phase, then a big-kid phase
        self.kid_cats = []
        for name, born in self.kids:
            grp = rng.choice([f"{name}", f"Baby {name}", f"{name} Bear"])
            baby_end = born + dt.timedelta(days=730)
            if baby_end > d0:
                self.kid_cats.append(dict(cid=self._cat(grp, "Diapers", created=max(d0, born), hidden_at=baby_end if baby_end < d1 else None),
                                          kinds=["big_box", "pharmacy", "grocery"], start=max(d0, born), end=baby_end, w=0.25))
                self.kid_cats.append(dict(cid=self._cat(grp, self._emoji("Baby clothes", "baby"), created=max(d0, born), hidden_at=baby_end if baby_end < d1 else None),
                                          kinds=["clothing", "secondhand"], start=max(d0, born), end=baby_end, w=0.5))
            big = max(d0, baby_end)
            if big < d1:
                self.kid_cats.append(dict(cid=self._cat(grp, self._emoji(rng.choice(["School, classes and sports", "School and activities", f"{name}'s sports"]), "sports"), created=big),
                                          kinds=["kids", "education", "fitness"], start=big, end=d1, w=0.9))
                self.kid_cats.append(dict(cid=self._cat(grp, self._emoji(rng.choice(["Big kid clothes", "Kids' clothes", f"{name}'s clothes"]), "clothes"), created=big),
                                          kinds=["clothing"], start=big, end=d1, w=0.35))
                self.kid_cats.append(dict(cid=self._cat(grp, self._emoji(rng.choice(["Other", "Toys and fun", f"{name}'s stuff"]), "kids"), created=big),
                                          kinds=["hobby", "big_box"], start=big, end=d1, w=0.2))
        # pets
        self.pet_cats = [self._cat(rng.choice(["Everyday Expenses", "Pets"]), self._emoji(p, "pet"), kinds=["pet"]) for p in self.pets]
        # trips: dated windows (category_style: trips ~16% of categories), hidden a while after
        self.trips = []
        t = d0 + dt.timedelta(days=rng.randint(20, 120))
        while t < d1 - dt.timedelta(days=10):
            length = rng.randint(3, 12)
            place = rng.choice(PLACES if rng.random() < 0.8 else EVENTS)
            nm = rng.choice([f"{place} {t.month:02d}-{t.year}", f"We go to {place}", f"{place} in {MONTHS[t.month - 1]}", f"{place} trip",
                             f"{rng.choice(LAST)}s come to visit"])
            hide = t + dt.timedelta(days=length + rng.randint(30, 200))
            cid = self._cat("Travels", self._emoji(nm, "travel") if rng.random() < 0.2 else nm, created=t - dt.timedelta(days=rng.randint(5, 60)),
                            hidden_at=hide if hide < d1 else None)
            self.trips.append(dict(cid=cid, start=t, end=t + dt.timedelta(days=length), place=place))
            t += dt.timedelta(days=int(rng.expovariate(1 / 120)) + 20)
        self.travel_other = self._cat("Travels", self._emoji(rng.choice(["All other traveling money", "Travel", "Trips (other)"]), "travel"))
        # holidays
        self.holidays = []
        for nm, (m0, d0_, m1, d1_), kinds, p in [("Christmas", (11, 15, 12, 31), ["gifts", "big_box", "hobby", "clothing", "electronics", "furniture"], 0.85),
                                                  ("Halloween", (10, 1, 10, 31), ["big_box", "clothing", "hobby"], 0.4),
                                                  ("Thanksgiving", (11, 18, 11, 28), ["grocery"], 0.2), ("Easter", (3, 25, 4, 15), ["big_box", "grocery"], 0.2)]:
            if rng.random() < p:
                self.holidays.append(dict(cid=self._cat("Planning ahead", self._emoji(nm, nm.lower())), window=(m0, d0_, m1, d1_), kinds=kinds, share=rng.uniform(0.4, 0.8)))
        # properties: a group of repeated purposes with a unit number
        self.properties = []
        for _ in range(rng.choice([0, 0, 0, 1, 1, 2])):
            unit = str(rng.randint(1000, 9999)); grp = rng.choice([f"Rental ({unit})", f"{rng.choice(LAST)} House", "Lake house", "The Farm"])
            pc = {k: self._cat(grp, f"{nm} ({unit})" if rng.random() < 0.5 else nm, created=d0 + dt.timedelta(days=rng.randint(0, span // 2)))
                  for k, nm in [("utility", "Utilities"), ("home_services", "Repairs and Maintenance"), ("home_improvement", "Landscaping"),
                                ("taxes", "Property Taxes"), ("insurance", "Insurance")]}
            self.properties.append(dict(cats=pc, start=self.cats[pc["utility"]]["created"]))
        # named services (category_style ~6%): recurring, often paid person to person
        self.services = []
        for nm, kind, p2p, amt in [("House cleaning services", "home_services", 0.6, (4.8, 0.2)), ("Babysitting", "kids", 0.9, (4.0, 0.4)),
                                   ("Lawncare, mowing", "home_services", 0.4, (4.0, 0.2)), ("Pool cleaning", "home_services", 0.2, (4.5, 0.2)),
                                   ("Pest Control", "home_services", 0.0, (4.4, 0.2)), ("Car wash service", "car_repair", 0.0, (3.4, 0.2)),
                                   ("Therapy", "medical", 0.1, (5.0, 0.2)), ("Accountant", "legal", 0.0, (5.6, 0.3)), ("Personal Trainer", "fitness", 0.5, (4.4, 0.3))]:
            if rng.random() < {"House cleaning services": 0.4, "Babysitting": 0.35 if self.kids else 0.05}.get(nm, 0.15):
                who = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
                self.services.append(dict(cid=self._cat("Monthly Bills", self._emoji(nm, {"House cleaning services": "cleaning", "Pool cleaning": "pool"}.get(nm))),
                                          kind=kind, p2p=rng.random() < p2p, who=who, amount=amt, every=rng.choice([14, 30, 30, 30])))
        # wish list, money mechanics
        self.wish = [(self._cat("Wish list", w, created=d0 + dt.timedelta(days=rng.randint(0, span))), w) for w in rng.sample(WISH, rng.choice([0, 0, 2, 4]))]
        for nm in rng.sample(["Long Term Investments/Savings", "Next months money", "Emergency fund", "New Car", "Buying a house"], rng.randint(1, 3)):
            self._cat("Investments" if "Invest" in nm else "Planning ahead", nm)
        self.savings = next((c for c in self.cats.values() if c["name"] in ("Long Term Investments/Savings", "Emergency fund")), None)
        # a reorganisation: one everyday category retired, its kinds routed to another from a date on
        self.reorg = None
        ev = [c for c in self.cats.values() if c.get("kinds") and c["category_group_id"] in (self.groups.get("Everyday Expenses"),)]
        if len(ev) >= 2 and rng.random() < 0.5:
            a, b = rng.sample(ev, 2); at = d0 + dt.timedelta(days=rng.randint(span // 4, 3 * span // 4))
            a["hidden_at"] = at; self.reorg = dict(old=a["id"], new=b["id"], at=at)

    # ---- merchants and strings ------------------------------------------------------------------------------------------------
    def _merchant(self, kind):
        rng = self.rng
        fav = self.favourites[kind]
        bill = kind in ("utility", "phone", "insurance", "rent", "subscription")  # household bills: name banks only (Overture's energy firms are not bills)
        if kind in NONPLACE and (bill or not self.pool.get(kind) or rng.random() < 0.7):
            name = rng.choice(NONPLACE[kind]); m = dict(name=name, kind=kind, city=None, chain=True)
        elif fav and rng.random() > {"restaurant": 0.45, "fast_food": 0.3, "coffee_bakery": 0.3, "hobby": 0.5, "entertainment": 0.55}.get(kind, 0.2):
            return fav[min(int(rng.paretovariate(1.2)) - 1, len(fav) - 1)]
        elif self.pool.get(kind) and self.shared:
            ms, ws = self._world(kind)
            if not ms:
                return None
            m = rng.choices(ms, ws)[0]
        elif self.pool.get(kind):
            m = rng.choice(self.pool[kind])
        else:
            return None
        fav.append(m)
        return m

    def _world(self, kind):
        """A shared world's merchants of one kind open to this household (chains, and the locals of its metro) and their weights."""
        key = (self.split, kind, self.metro)
        if key not in _WORLD:
            ms = [m for m in self.pool[kind] if m.get("chain") or _h(m["name"]) % METROS == self.metro]
            _WORLD[key] = (ms, [((m.get("places") or 1) ** 0.7 if m.get("chain") else 1.0) for m in ms])
        return _WORLD[key]

    def _payee(self, m, date):
        """The raw bank string and the YNAB payee for one transaction at merchant m: each merchant has one to three string forms
        (render_v2), some with a per-transaction code; the household cleans some merchants (one payee, the plain name) and not others
        (a payee per string form, named by the string, as YNAB imports it)."""
        rng = self.rng
        key = m["name"]
        if key not in self.payee_of_string:
            r = random.Random(f"bank{self.bank}-{key}" if self.shared else f"{self.seed}-{key}")
            forms = []
            for _ in range(r.choice([1, 1, 2, 3])):
                f = render_v2(m["name"], r, city=m.get("city"), parts=True)
                if not m.get("chain") and ".co" in f[0].lower() and r.random() < 0.7:  # web-domain strings are mostly online and chain merchants
                    f = render_v2(m["name"], r, city=m.get("city"), parts=True)
                forms.append(f)
            if self.shared:  # the household's own habits: whether it cleans this payee (the strings are the bank's)
                r = random.Random(f"{self.seed}-{key}")
            coded = bool(forms[0][1].get("Reference")) and forms[0][0].endswith("*" + forms[0][1]["Reference"]) if V4 else bool(re.search(r"\*[A-Z0-9]{5,}$", forms[0][0]))
            self.payee_of_string[key] = dict(forms=forms, clean=r.random() < 0.5, recode=coded and r.random() < 0.6)
        info = self.payee_of_string[key]
        s, parts = info["forms"][0] if rng.random() < 0.8 else rng.choice(info["forms"])
        parts = dict(parts)
        if V4 and info["recode"] and parts.get("Reference") and s.endswith("*" + parts["Reference"]):  # v4: swap the recorded code itself
            code = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(9))
            s = s[: -len(parts["Reference"])] + code; parts["Reference"] = code
        elif not V4 and info["recode"] and re.search(r"\*[A-Z0-9]{5,}$", s):  # a new order / reference code each time, the name kept
            code = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(9))
            s = re.sub(r"\*[A-Z0-9]{5,}$", "*" + code, s); parts["Reference"] = code
        if m["kind"] == "food_delivery" and self.pool.get("restaurant"):  # row 176: the order's restaurant in the string (own rng: other draws unchanged)
            r2 = random.Random(f"{self.seed}-order-{len(self.txs)}")
            rest = r2.choice(self.favourites["restaurant"] or self.pool["restaurant"])["name"]
            s = f"{s.split('*')[0].strip()} *{rest}"[:40]
            parts = {"Clean payee": m["name"], "Ordered from": rest}
        name = m["name"] if info["clean"] else s[:50]
        self._parts = parts; self._kind = m["kind"]
        return s, name

    def _p2p(self, who, memo):
        app = self.rng.choices([a for a, _ in P2P_APPS], [w for _, w in P2P_APPS])[0]
        verb = self.rng.choice(["", "to ", "Paid "]) if app != "Zelle" else "to "
        s = f"{app} {verb}{who}" + (f" {memo}" if memo else "")
        self._parts = {"Clean payee": who, "Via": app.replace(" Transfer", "")} | ({"Memo": memo} if memo else {}); self._kind = "p2p"
        return s, s[:60]

    # ---- the timeline -----------------------------------------------------------------------------------------------------------
    def _route(self, kind, date, amount, weekday):
        rng = self.rng
        for t in self.trips:
            if t["start"] - dt.timedelta(days=1) <= date <= t["end"] and kind in ("hotel", "airline", "car_rental", "restaurant", "fast_food", "coffee_bakery",
                                                                                     "transit", "gas", "entertainment", "parking", "bar", "convenience"):
                if rng.random() < 0.85:
                    self._reason = ("trip", t["cid"]); return t["cid"]
        if kind in ("hotel", "airline", "car_rental", "travel_agency"):
            near = [t for t in self.trips if dt.timedelta(days=-90) <= (t["start"] - date) <= dt.timedelta(days=90)]
            c = near[0]["cid"] if near and rng.random() < 0.7 else self.travel_other
            self._reason = ("trip", c) if c != self.travel_other else ("travel", c); return c
        for h in self.holidays:
            m0, d0_, m1, d1_ = h["window"]
            if kind in h["kinds"] and dt.date(date.year, m0, d0_) <= date <= dt.date(date.year, m1, d1_) and rng.random() < h["share"]:
                self._reason = ("holiday", h["cid"]); return h["cid"]
        for k in self.kid_cats:
            if kind in k["kinds"] and k["start"] <= date <= k["end"] and rng.random() < k["w"]:
                self._reason = ("phase", k["cid"]); return k["cid"]
        for p in self.properties:
            if date >= p["start"] and kind in p["cats"] and rng.random() < 0.3:
                self._reason = ("property", p["cats"][kind]); return p["cats"][kind]
        if kind == "pet" and self.pet_cats:
            c = rng.choice(self.pet_cats); self._reason = ("pet", c); return c
        for a, pc in self.person.items():
            if pc["treat"] and kind == "coffee_bakery" and weekday < 5 and amount < 12 and rng.random() < 0.8:
                self._reason = ("treat", pc["treat"]); return pc["treat"]
            if kind in self.cats[pc["fun"]]["kinds"] and rng.random() < pc["share"] / max(1, len(self.person) - 0.5):
                c = {"clothing": pc["clothes"], "jewelry": pc["clothes"]}.get(kind, pc["fun"]); self._reason = ("person", c); return c
            if pc["groom"] and kind == "personal_care" and rng.random() < 0.7:
                self._reason = ("person", pc["groom"]); return pc["groom"]
            if pc["work"] and kind in ("office", "business") and rng.random() < 0.8:
                self._reason = ("person", pc["work"]); return pc["work"]
        cid = self.kind_to_everyday.get(kind)
        self._reason = ("habit", cid)
        if self.reorg and cid == self.reorg["old"] and date >= self.reorg["at"]:
            cid = self.reorg["new"]; self._reason = ("reorg", cid)
        if cid is None or rng.random() < 0.03:
            cid = rng.choice(self.catchall); self._reason = ("catchall", cid)
        return cid

    def _add(self, date, amount, cid, raw, payee_name, memo=None, approved=True):
        parts, self._parts = self._parts or {"Clean payee": payee_name}, None  # row 176: the pieces behind the string
        reason, kind, self._reason, self._kind = self._reason, self._kind, None, None  # row 177: why this category, and the merchant's kind
        pid = self.payees.setdefault(payee_name, self._id())
        if self.rng.random() < 0.03 and cid not in (self.rta,):  # misfiled
            live = [c for c in self.cats.values() if c.get("kinds") and c["created"] <= date and not (c["hidden_at"] and c["hidden_at"] <= date)]
            cid = self.rng.choice(live)["id"] if live else cid
            reason = ("misfile", cid)
        self.txs.append(dict(id=self._id(), date=date.isoformat(), amount=int(round(-amount * 1000)), cleared="cleared", approved=approved,
                             account_id=self.accounts["Credit Card"], payee_id=pid, category_id=cid, import_payee_name_original=raw,
                             memo=memo, deleted=False, parts=parts, reason=list(reason) if reason else None, kind=kind))

    def simulate(self):
        rng = self.rng
        d = self.start
        last_service = defaultdict(lambda: self.start - dt.timedelta(days=60))
        bills = {k: (self._merchant(k), _ln(rng, AMOUNT[k])) for k in ("utility", "phone", "insurance", "rent") if self.kind_to_everyday.get(k) or k == "rent"}
        subs = [(self._merchant("subscription"), _ln(rng, AMOUNT["subscription"])) for _ in range(rng.randint(1, 6))]
        employer = rng.choice(EMPLOYERS); pay = round(rng.uniform(1500, 6000), 2)
        while d <= self.end:
            wd = d.weekday()
            for kind, rate in RATES.items():
                lam = rate * self.scale / 7
                for _ in range(min(4, int(rng.expovariate(1) < lam) + int(rng.random() < lam * lam))):
                    m = self._merchant(kind)
                    if m is None:
                        continue
                    amt = _ln(rng, AMOUNT.get(kind, (3.5, 0.8)))
                    raw, name = self._payee(m, d)
                    if m["name"] in self.store_named and rng.random() < 0.8:
                        cid = self.store_named[m["name"]]; self._reason = ("store", cid)
                    else:
                        cid = self._route(kind, d, amt, wd)
                    self._add(d, amt, cid, raw, name)
            for t in self.trips:  # travel bookings before a trip, lodging during it
                if d == t["start"] - dt.timedelta(days=rng.randint(10, 40)) or (d == t["start"] and rng.random() < 0.7):
                    for kind in ("airline", "hotel") + (("car_rental",) if rng.random() < 0.4 else ()):
                        m = self._merchant(kind)
                        if m:
                            raw, name = self._payee(m, d); amt = _ln(rng, AMOUNT[kind]); self._add(d, amt, self._route(kind, d, amt, wd), raw, name)
            if d.day == 1:
                for k, (m, a) in bills.items():
                    if m:
                        raw, name = self._payee(m, d); self._reason = ("bill", None); self._add(d, round(a * rng.uniform(0.9, 1.1), 2), self.kind_to_everyday.get(k) or self.catchall[0], raw, name)
                for m, a in subs:
                    if m:
                        raw, name = self._payee(m, d); self._add(d, a, self._route("subscription", d, a, wd) if self.kind_to_everyday.get("subscription") else self.kind_to_everyday.get("subscription") or self.catchall[0], raw, name)
                for p in self.properties:
                    if d >= p["start"]:
                        m = self._merchant("utility")
                        if m:
                            raw, name = self._payee(m, d); self._reason = ("property", None); self._add(d, _ln(rng, AMOUNT["utility"]), p["cats"]["utility"], raw, name)
                if self.savings and rng.random() < 0.6:
                    self._reason, self._kind = ("savings", None), "transfer"; self._add(d, round(rng.uniform(100, 1000), -1), self.savings["id"], "Transfer to Savings", "Transfer : Savings")
            if d.day in (1, 15):  # paycheck
                self._reason, self._kind = ("income", None), "income"; self._add(d, -pay * rng.uniform(0.97, 1.03), self.rta, f"{employer} DIRECT DEP" if rng.random() < 0.5 else employer, employer)
            for s in self.services:
                if (d - last_service[s["cid"]]).days >= s["every"] and rng.random() < 0.8:
                    last_service[s["cid"]] = d
                    amt = _ln(rng, s["amount"])
                    if s["p2p"]:
                        raw, name = self._p2p(s["who"], rng.choice(["", "", self.cats[s["cid"]]["name"].split()[0].lower(), "thank you!", MONTHS[d.month - 1]]))
                    else:
                        m = self._merchant(s["kind"]); raw, name = self._payee(m, d) if m else (s["who"], s["who"])
                    self._reason = ("service", None); self._add(d, amt, s["cid"], raw, name)
            if rng.random() < 0.06 * self.scale:  # person-to-person payments with purpose memos (8% of strings; memo half the time)
                who = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
                purpose = rng.random()
                if purpose < 0.3:
                    memo, cid = rng.choice(["dinner", "tacos 🌮", "pizza", "brunch", "drinks"]), self.kind_to_everyday.get("restaurant") or self.catchall[0]
                elif purpose < 0.45 and self.trips:
                    t = min(self.trips, key=lambda t: abs((t["start"] - d).days))
                    memo, cid = rng.choice([f"Airbnb {t['place']}", f"{t['place']} gas", "trip stuff"]), t["cid"]
                elif purpose < 0.55:
                    memo, cid = rng.choice(["birthday gift", "wedding gift", "baby shower"]), self.kind_to_everyday.get("gifts") or self.catchall[0]
                elif purpose < 0.65 and self.kid_cats:
                    memo, cid = rng.choice(["soccer fees", "field trip", "Girl Scout cookies", "dance class"]), self.kid_cats[-1]["cid"]
                else:
                    memo, cid = rng.choice(["Girl Scout cookies", "concert tickets", "bike", "thanks!", "half of the bill", "garage sale"]), rng.choice(self.catchall)
                raw, name = self._p2p(who, memo if rng.random() < 0.53 else "")
                self._reason = ("p2p_memo", None) if "Memo" in (self._parts or {}) else ("p2p", None)
                self._add(d, _ln(rng, (3.5, 0.8)), cid, raw, name)
            for cid, w in self.wish:
                if self.cats[cid]["created"] + dt.timedelta(days=20) == d:
                    m = self._merchant(rng.choice(["furniture", "big_box", "electronics"]))
                    if m:
                        raw, name = self._payee(m, d); self._reason = ("wish", None); self._add(d, _ln(rng, (5.0, 0.6)), cid, raw, name)
            d += dt.timedelta(days=1)
        return self

    def budget(self):
        """The household as a YNAB-skill cache document."""
        cats = []
        for c in self.cats.values():
            hidden = bool(c["hidden_at"] and c["hidden_at"] <= self.end)
            cats.append(dict(id=c["id"], name=c["name"], category_group_id=c["category_group_id"], hidden=hidden, deleted=False))
        return {"cached_at": None, "budget": dict(
            id=f"realstyle-{self.split}-{self.seed}", name=f"realstyle {self.split} {self.seed}",
            category_groups=[dict(id=i, name=n, hidden=False, deleted=False) for n, i in self.groups.items()],
            categories=cats, payees=[dict(id=i, name=n, deleted=False) for n, i in self.payees.items()],
            accounts=[dict(id=i, name=n, deleted=False) for n, i in self.accounts.items()],
            transactions=sorted(self.txs, key=lambda t: t["date"]), subtransactions=[])}


def household(seed, split="train", **kw):
    return Household(seed, split, **kw).simulate()
