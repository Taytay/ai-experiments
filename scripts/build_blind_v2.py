#!/usr/bin/env python3
"""Blind synthetic test set v2 for a per-user bank-transaction categoriser.

Writes data/processed/blind_v2.json (relative to the repo root). Standard library only and
fully deterministic (one seeded random.Random per user plus crc32-keyed helpers; no hash()).

Written blind: it shares no code, data or merchant lists with the rest of the repository.
Each simulated user has their own category list (standard, personal, merged, detailed or
coined names), a home city, a bank that renders statement strings its own way, and a year of
transactions produced by habits (poisson visits, bills on schedules, paydays, P2P contacts),
trips, life events, refunds, reimbursements, changes of mind and filing mistakes. Queries are
drawn from the later part of each timeline (or the first 11 rows for brand-new users) and each
carries the production-style history slice, an "other users" variant, and oracle fields.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import random
import re
import string
import zlib
from collections import Counter, defaultdict
from pathlib import Path

SEED = 2025_0929
YEAR = 2025
N_USERS = 250
N_NEW_USERS = 32
Q_PER_USER = 6
MAX_ROWS = 24
MAX_SAME = 6
MAX_SIMILAR = 6
MAX_CATS = 45
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUT_FILE = OUT_DIR / "blind_v2.json"

WD = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
D0 = dt.date(YEAR, 1, 1)
D1 = dt.date(YEAR, 12, 31)
ALL_DAYS = [D0 + dt.timedelta(days=i) for i in range((D1 - D0).days + 1)]


def srng(*key):
    """A random.Random seeded from a stable key (crc32, not the salted built-in hash)."""
    return random.Random(zlib.crc32("\x1f".join(str(k) for k in key).encode("utf-8")) ^ SEED)


def digits(rng, n):
    return "".join(rng.choice(string.digits) for _ in range(n))


def alnum(rng, n):
    return "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(n))


def logu(rng, lo, hi):
    if hi <= lo:
        return round(lo, 2)
    return round(math.exp(rng.uniform(math.log(lo), math.log(hi))), 2)


def day(m, d):
    return dt.date(YEAR, m, d)


def clamp_date(d):
    return min(max(d, D0), D1)


# ----------------------------------------------------------------------------------------------
# Concepts (what a purchase is for) and how a concept falls back when a user has no category
# for it. Every user has a catch-all ("misc"), so resolution always ends somewhere.
# ----------------------------------------------------------------------------------------------
FALLBACK = {
    "groceries": ["household"], "dining": ["fun", "entertainment", "groceries"], "coffee": ["dining", "snacks"],
    "takeout": ["dining"], "snacks": ["coffee", "dining", "groceries"], "alcohol": ["dining", "groceries", "fun"],
    "fuel": ["car_maint"], "car_maint": ["fuel"], "parking": ["fuel", "transit"], "transit": ["rideshare", "fuel"],
    "rideshare": ["transit", "fuel"], "car_payment": ["fuel"], "auto_ins": ["fuel", "car_payment"],
    "travel": ["fun", "entertainment"], "rent": ["mortgage"], "mortgage": ["rent"], "hoa": ["mortgage", "rent"],
    "electric": ["gas_util", "water", "internet"], "gas_util": ["electric"], "water": ["electric"],
    "trash": ["water", "electric"], "internet": ["phone", "electric"], "phone": ["internet", "electric"],
    "home_ins": ["auto_ins", "mortgage", "rent"], "life_ins": ["health_ins", "auto_ins"], "health_ins": ["medical"],
    "student_loan": [], "cc_payment": [], "streaming": ["software", "fun", "entertainment"],
    "software": ["streaming", "business"], "gym": ["medical", "hobbies", "fun"], "medical": ["pharmacy"],
    "pharmacy": ["medical", "household"], "personal_care": ["pharmacy", "household", "clothing"],
    "household": ["groceries"], "home_improve": ["household"], "clothing": ["household", "personal_care"],
    "electronics": ["household", "hobbies", "fun"], "books": ["hobbies", "fun", "entertainment", "school"],
    "hobbies": ["fun", "entertainment"], "games": ["hobbies", "fun", "entertainment"], "entertainment": ["fun", "dining"],
    "kids": ["childcare", "household"], "childcare": ["kids"], "school": ["kids", "books"], "baby": ["kids", "household"],
    "pets": ["household"], "gifts": [], "charity": ["church", "gifts"], "church": ["charity"], "cash": [], "fees": [],
    "taxes": [], "business": ["software", "household"], "income": [], "gig_income": ["income"],
    "savings": ["investing"], "investing": ["savings"], "wedding": ["gifts"], "running": ["gym", "hobbies", "fun"],
    "misc": [],
}
FAMILY = {}
for _fam, _cs in {
    "food": "groceries dining coffee takeout snacks alcohol",
    "car": "fuel car_maint parking transit rideshare car_payment auto_ins",
    "home": "rent mortgage hoa household home_improve home_ins",
    "bills": "electric gas_util water trash internet phone streaming software",
    "fun": "entertainment hobbies books games travel gym running wedding",
    "care": "medical pharmacy personal_care health_ins life_ins gym",
    "kids": "kids childcare school baby",
    "stuff": "clothing electronics household gifts pets",
    "money": "income gig_income savings investing cc_payment cash fees taxes business student_loan charity church misc",
}.items():
    for _c in _cs.split():
        FAMILY.setdefault(_c, _fam)
# where a person plausibly moves a payee when they change their mind about it
PLAUSIBLE = {
    "dining": {"takeout", "entertainment", "alcohol", "coffee", "snacks", "travel", "gifts"},
    "coffee": {"dining", "snacks", "entertainment"}, "snacks": {"dining", "coffee", "groceries"},
    "takeout": {"dining", "entertainment"}, "groceries": {"household", "alcohol", "snacks"},
    "alcohol": {"dining", "entertainment", "groceries"}, "fuel": {"car_maint", "transit", "travel", "business"},
    "streaming": {"entertainment", "software", "internet", "games", "hobbies"},
    "software": {"business", "streaming", "entertainment", "internet", "phone"},
    "gym": {"medical", "hobbies", "personal_care", "running", "entertainment"},
    "personal_care": {"medical", "pharmacy", "clothing", "gifts", "household"},
    "clothing": {"personal_care", "gifts", "kids", "hobbies"},
    "hobbies": {"entertainment", "kids", "gifts", "books", "games"},
    "entertainment": {"dining", "hobbies", "gifts", "kids", "travel"}, "medical": {"pharmacy", "health_ins", "personal_care"},
    "household": {"groceries", "home_improve", "gifts", "kids"}, "home_improve": {"household", "mortgage", "hoa"},
    "electronics": {"household", "hobbies", "gifts", "business"}, "books": {"hobbies", "entertainment", "school", "kids"},
    "kids": {"childcare", "school", "household", "gifts", "entertainment"}, "phone": {"internet", "business", "software"},
    "internet": {"phone", "streaming", "business"}, "pets": {"household"}, "transit": {"rideshare", "fuel", "parking"},
    "rideshare": {"transit", "travel", "entertainment"}, "parking": {"fuel", "transit", "entertainment"},
    "car_maint": {"fuel"}, "gifts": {"entertainment", "kids", "household"}, "pharmacy": {"medical", "personal_care"},
}
MONEY_CONCEPTS = {"income", "gig_income", "savings", "investing", "cc_payment", "taxes"}

# group key -> (concepts, standard names, personal names, coined names)
G = {
    "groceries": ("groceries", ["Groceries", "Grocery"], ["Food we cook", "Food at home", "Feeding the crew"], ["Pantry Raid", "Larder"]),
    "dining": ("dining takeout", ["Dining Out", "Restaurants", "Eating Out"], ["Food someone else made", "Meals out"], ["Nom Nom", "Fork Money"]),
    "coffee": ("coffee snacks", ["Coffee", "Coffee Shops"], ["Caffeine", "Morning fuel"], ["Bean Juice", "Drip"]),
    "takeout": ("takeout", ["Takeout", "Delivery"], ["Too tired to cook"], ["Doorbell Dinners"]),
    "alcohol": ("alcohol", ["Alcohol", "Wine & Beer"], ["Booze", "Wine o'clock"], ["Grape Juice"]),
    "snacks": ("snacks", ["Snacks"], ["Gas station snacks"], ["Munchie Fund"]),
    "food_all": ("groceries dining coffee takeout snacks alcohol", ["Food"], ["Everything food", "All the food"], ["Munchies", "Grub"]),
    "fuel": ("fuel", ["Gas", "Fuel", "Gas & Fuel"], ["Filling up"], ["Dino Juice"]),
    "car": ("fuel car_maint parking car_payment auto_ins", ["Car", "Auto", "Vehicle"], ["Car stuff", "The Subaru"], ["Vroom"]),
    "car_maint": ("car_maint", ["Car Maintenance", "Auto Maintenance"], ["Car repairs", "Oil changes & tires"], ["Wrench Fund"]),
    "transport": ("fuel transit rideshare parking car_maint", ["Transportation", "Getting Around"], ["Getting places"], ["Wheels"]),
    "transit": ("transit rideshare parking", ["Public Transit", "Transit & Rides"], ["Bus & Uber"], ["Commute Coins"]),
    "parking": ("parking", ["Parking & Tolls", "Parking"], ["Meters & tolls"], ["Meter Feed"]),
    "rent": ("rent hoa", ["Rent"], ["Roof over head", "Apartment"], ["Castle Tax"]),
    "mortgage": ("mortgage hoa", ["Mortgage", "Housing"], ["House payment"], ["The Nest"]),
    "utilities": ("electric gas_util water trash", ["Utilities"], ["Lights & water", "House bills"], ["Glow & Flow"]),
    "electric": ("electric", ["Electric", "Electricity"], ["Power bill"], ["Sparky"]),
    "gas_util": ("gas_util", ["Natural Gas", "Gas Bill"], ["Heat"], ["Flame"]),
    "water": ("water trash", ["Water & Trash", "Water"], ["Water bill"], ["Drips"]),
    "internet": ("internet", ["Internet"], ["Wifi"], ["The Tubes"]),
    "phone": ("phone", ["Phone", "Cell Phone"], ["Phones"], ["Bleep Bloop"]),
    "bills": ("electric gas_util water trash internet phone", ["Bills", "Monthly Bills"], ["Boring bills", "Keep the lights on"], ["Adulting"]),
    "insurance": ("auto_ins home_ins life_ins health_ins", ["Insurance"], ["Just in case"], ["Umbrella"]),
    "auto_ins": ("auto_ins", ["Car Insurance", "Auto Insurance"], ["Car insurance (6 mo)"], ["Crash Pad"]),
    "home_ins": ("home_ins", ["Home Insurance", "Renters Insurance"], ["House insurance"], ["Roof Shield"]),
    "health_ins": ("health_ins", ["Health Insurance"], ["Health premium"], ["Stethoscope"]),
    "car_payment": ("car_payment", ["Car Payment", "Auto Loan"], ["Car note"], ["Honda Debt"]),
    "student_loan": ("student_loan", ["Student Loans"], ["School debt"], ["Sallie"]),
    "debt": ("car_payment student_loan cc_payment", ["Debt Payments"], ["Paying it down"], ["Snowball"]),
    "subscriptions": ("streaming software", ["Subscriptions"], ["Monthly subs", "Stuff that auto-renews"], ["Drip Drip"]),
    "streaming": ("streaming", ["Streaming", "TV & Streaming"], ["Couch time"], ["Binge"]),
    "software": ("software", ["Software", "Apps & Cloud"], ["Apps"], ["Bits"]),
    "gym": ("gym", ["Gym", "Fitness"], ["Sweat"], ["Gains"]),
    "health": ("medical pharmacy gym health_ins", ["Health", "Healthcare"], ["Doctors & meds"], ["Band-Aids"]),
    "medical": ("medical health_ins", ["Medical", "Doctor"], ["Doctors"], ["Ouchies"]),
    "pharmacy": ("pharmacy", ["Pharmacy", "Prescriptions"], ["Meds"], ["Pills"]),
    "personal_care": ("personal_care", ["Personal Care", "Hair & Beauty"], ["Haircuts & such", "Look good"], ["Glow Up"]),
    "household": ("household", ["Household", "Household Supplies", "Home Goods"], ["House stuff", "Paper towels etc"], ["Nest Stuff"]),
    "home_improve": ("home_improve", ["Home Improvement", "Home Maintenance"], ["House projects", "Fix-it"], ["Honey-Do"]),
    "clothing": ("clothing", ["Clothing", "Clothes"], ["Things to wear"], ["Drip Check"]),
    "stuff": ("household clothing electronics books hobbies", ["Shopping", "Stuff"], ["Things I bought"], ["Doodads"]),
    "electronics": ("electronics", ["Electronics", "Tech"], ["Gadgets"], ["Blinky Lights"]),
    "fun": ("entertainment hobbies games books streaming", ["Fun", "Entertainment"], ["Fun money", "Treat yo self"], ["Whee"]),
    "entertainment": ("entertainment", ["Entertainment", "Going Out"], ["Shows & movies"], ["Popcorn"]),
    "hobbies": ("hobbies books", ["Hobbies"], ["Crafts & hobbies", "Projects"], ["Tinker"]),
    "games": ("games", ["Video Games", "Games"], ["Gaming"], ["Pixels"]),
    "kids": ("kids childcare school baby", ["Kids"], ["The kids", "Kid stuff"], ["Minions"]),
    "childcare": ("childcare", ["Childcare", "Daycare"], ["Babysitting"], ["Sanity Fund"]),
    "school": ("school", ["School", "Education"], ["Tuition & books"], ["Brain Food"]),
    "pets": ("pets", ["Pets", "Pet Care"], ["The dog", "Cat stuff"], ["Zoomies", "Kibble"]),
    "gifts": ("gifts", ["Gifts"], ["Presents", "Birthdays & holidays"], ["Bows"]),
    "giving": ("charity church", ["Giving", "Charity", "Donations"], ["Doing good"], ["Karma"]),
    "church": ("church", ["Tithe", "Church"], ["Church giving"], ["Offering Plate"]),
    "travel": ("travel", ["Travel", "Vacation"], ["Getaways", "Trips"], ["Wanderlust"]),
    "cash": ("cash", ["Cash", "ATM"], ["Walking around money", "Spending money"], ["Pocket Money"]),
    "taxes": ("taxes", ["Taxes", "Tax Savings"], ["Uncle Sam"], ["Tax Jar"]),
    "business": ("business", ["Business Expenses", "Business"], ["Work stuff", "Biz costs"], ["Hustle Costs"]),
    "income": ("income gig_income", ["Inflow: Ready to Assign", "Income", "Ready to Assign"], ["Money in", "Paychecks"], ["Treasure Chest"]),
    "gig_income": ("gig_income", ["Side Income", "Freelance Income"], ["Side hustle", "Gig money"], ["Hustle"]),
    "savings": ("savings investing", ["Savings", "Emergency Fund"], ["Future me", "Rainy day"], ["Squirrel"]),
    "investing": ("investing", ["Investing", "Brokerage"], ["Stonks"], ["Moon Shot"]),
    "cc_payment": ("cc_payment", ["Credit Card Payment"], ["Paying off the card"], ["Plastic"]),
    "misc": ("misc fees", ["Miscellaneous", "Everything Else", "Misc"], ["Stuff I forgot to budget for", "Random"], ["Whatchamacallit", "Oops"]),
    "work_lunch": ("", ["Work Lunches"], ["Lunch at work"], ["Desk Tacos"]),
    "date_night": ("", ["Date Night"], ["Us time"], ["Wine & Dine"]),
    "wedding": ("wedding", ["Wedding"], ["Our wedding"], ["Big Day"]),
    "baby": ("baby", ["Baby"], ["Baby stuff"], ["Tiny Human"]),
    "running": ("running", ["Marathon Training"], ["Running stuff"], ["Pavement"]),
}
# concept -> the group that holds just that concept (used when a user splits a concept out)
G_FOR_CONCEPT = {"coffee": "coffee", "takeout": "takeout", "alcohol": "alcohol", "snacks": "snacks", "fuel": "fuel",
                 "streaming": "streaming", "software": "software", "gym": "gym", "personal_care": "personal_care",
                 "games": "games", "gifts": "gifts", "clothing": "clothing", "electronics": "electronics",
                 "pharmacy": "pharmacy", "parking": "parking", "car_maint": "car_maint", "internet": "internet",
                 "phone": "phone", "household": "household", "entertainment": "entertainment", "pets": "pets"}

# World knowledge a reader has about category names (coined names are deliberately absent).
MEANING = defaultdict(set)
for _k, (_cs, _std, _pers, _coin) in G.items():
    for _n in _std + _pers:
        MEANING[_n] |= set(_cs.split())
MEANING["Work Lunches"] |= {"dining", "lunch"}
MEANING["Lunch at work"] |= {"dining", "lunch"}
MEANING["Date Night"] |= {"dining", "date"}
MEANING["Us time"] |= {"dining", "date"}
for _n in ("Travel", "Vacation", "Getaways", "Trips"):
    MEANING[_n] |= {"travel"}

# ----------------------------------------------------------------------------------------------
# Cities
# ----------------------------------------------------------------------------------------------
CITIES = {
    "Oakland": ("CA", ["Berkeley", "Emeryville", "Alameda", "San Leandro"], ("PG&E", "PGANDE", "WEB ONLINE"), None,
                ("EBMUD", "EBMUD", "WATER BIL"), ("Clipper", "CLIPPER TRANSIT FARE"), "Alameda"),
    "Columbus": ("OH", ["Dublin", "Westerville", "Grandview Hts", "Hilliard"], ("AEP Ohio", "AEP OHIO", "BILL PAY"),
                 ("Columbia Gas", "COLUMBIA GAS OH", "UTIL PYMT"), ("City of Columbus Utilities", "CITY OF COLUMBUS", "UTILITIES"),
                 ("COTA", "COTA TRANSIT FARE"), "Franklin"),
    "Raleigh": ("NC", ["Cary", "Durham", "Garner", "Wake Forest"], ("Duke Energy", "DUKE ENERGY", "WEB PAY"),
                ("PSNC Energy", "PSNC ENERGY", "PAYMENT"), ("City of Raleigh Utilities", "CITY OF RALEIGH", "UTILITY"),
                ("GoRaleigh", "GORALEIGH FARE"), "Wake"),
    "Phoenix": ("AZ", ["Tempe", "Scottsdale", "Mesa", "Glendale"], ("APS", "ARIZONA PUBLIC SE", "APS PAYMNT"),
                ("Southwest Gas", "SOUTHWEST GAS", "SWGAS PAY"), ("City of Phoenix Water", "CITY OF PHOENIX", "WATER SVC"),
                ("Valley Metro", "VALLEY METRO FARE"), "Maricopa"),
    "Minneapolis": ("MN", ["St Paul", "Edina", "Bloomington", "Richfield"], ("Xcel Energy", "XCEL ENERGY-NSP", "XCELENERGY"),
                    ("CenterPoint Energy", "CENTERPOINT ENERG", "MN GAS"), ("City of Minneapolis Utility", "MPLS UTILITY BILL", "UTILITY"),
                    ("Metro Transit", "METRO TRANSIT GOTO"), "Hennepin"),
    "Tampa": ("FL", ["St Petersburg", "Brandon", "Clearwater", "Temple Terrace"], ("Tampa Electric", "TAMPA ELECTRIC", "TECO"),
              ("Peoples Gas", "PEOPLES GAS", "PGS PAY"), ("City of Tampa Utilities", "CITY OF TAMPA", "UTILITIES"),
              ("HART", "HART TRANSIT FLAMINGO"), "Hillsborough"),
    "Portland": ("OR", ["Beaverton", "Gresham", "Lake Oswego", "Milwaukie"], ("Portland General Electric", "PORTLAND GENERAL", "PGE WEB"),
                 ("NW Natural", "NW NATURAL", "NWN PAYMNT"), ("Portland Water Bureau", "CITY OF PORTLAND", "WATER BUR"),
                 ("TriMet", "TRIMET HOP FASTPASS"), "Multnomah"),
    "Pittsburgh": ("PA", ["Mt Lebanon", "Monroeville", "Bethel Park", "Wexford"], ("Duquesne Light", "DUQUESNE LIGHT", "DLCO PAY"),
                   ("Peoples Natural Gas", "PEOPLES NAT GAS", "WEBPAY"), ("Pittsburgh Water", "PGH WATER SEWER", "PWSA"),
                   ("PRT", "PRT CONNECTCARD"), "Allegheny"),
    "Salt Lake City": ("UT", ["Murray", "Sandy", "West Valley", "South Salt Lake"], ("Rocky Mountain Power", "ROCKY MTN POWER", "RMP PMT"),
                       ("Enbridge Gas Utah", "ENBRIDGE GAS UT", "GAS PMT"), ("SLC Public Utilities", "SLC PUBLIC UTIL", "WATER"),
                       ("UTA", "UTA FAREPAY"), "Salt Lake"),
    "Kansas City": ("MO", ["Overland Park", "Independence", "North Kansas City", "Lees Summit"], ("Evergy", "EVERGY", "EVERGY PMT"),
                    ("Spire", "SPIRE MISSOURI", "SPIRE PAY"), ("KC Water", "KC WATER SERVICES", "WATER BILL"),
                    ("RideKC", "RIDEKC FARE"), "Jackson"),
}
DESTS = [("Honolulu", "HI", "Hawaii"), ("Kahului", "HI", "Maui"), ("Orlando", "FL", "Disney"), ("New York", "NY", "NYC"),
         ("Chicago", "IL", "Chicago"), ("Denver", "CO", "Colorado"), ("New Orleans", "LA", "NOLA"),
         ("Nashville", "TN", "Nashville"), ("San Diego", "CA", "San Diego"), ("Seattle", "WA", "Seattle"),
         ("Austin", "TX", "Austin"), ("Las Vegas", "NV", "Vegas"), ("Boston", "MA", "Boston"), ("Cancun", "MX", "Cancun")]
CITY_ST = {}
for _c, _v in CITIES.items():
    CITY_ST[_c] = _v[0]
    for _n in _v[1]:
        CITY_ST[_n] = _v[0]
for _c, _s, _t in DESTS:
    CITY_ST[_c] = _s

# ----------------------------------------------------------------------------------------------
# Merchants
# ----------------------------------------------------------------------------------------------
KIND_CONCEPT = {
    "grocery": "groceries", "bigbox": "household", "warehouse": "groceries", "online": "household", "restaurant": "dining",
    "fastfood": "dining", "coffee": "coffee", "bakery": "coffee", "delivery": "takeout", "bar": "alcohol",
    "liquor": "alcohol", "gas": "fuel", "convenience": "snacks", "auto": "car_maint", "parking": "parking",
    "transit": "transit", "rideshare": "rideshare", "airline": "travel", "hotel": "travel", "carrental": "travel",
    "pharmacy": "pharmacy", "medical": "medical", "dental": "medical", "home_improve": "home_improve",
    "furniture": "household", "dollar": "household", "clothing": "clothing", "dept": "clothing",
    "electronics": "electronics", "books": "books", "hobby": "hobbies", "games": "games", "kids_activity": "kids",
    "childcare": "childcare", "school": "school", "pet": "pets", "vet": "pets", "streaming": "streaming",
    "software": "software", "gym": "gym", "phone": "phone", "internet": "internet", "electric": "electric",
    "gas_util": "gas_util", "water": "water", "rent": "rent", "mortgage": "mortgage", "hoa": "hoa",
    "auto_ins": "auto_ins", "home_ins": "home_ins", "life_ins": "life_ins", "health_ins": "health_ins",
    "car_payment": "car_payment", "student_loan": "student_loan", "salon": "personal_care", "beauty": "personal_care",
    "entertainment": "entertainment", "charity": "charity", "church": "church", "florist": "gifts", "atm": "cash",
    "fee": "fees", "tax": "taxes", "biz": "business", "payroll": "income", "gig_pay": "gig_income",
    "gov_income": "income", "interest": "income", "savings_xfer": "savings", "invest_xfer": "investing",
    "cc_pay": "cc_payment", "p2p": "misc", "laundry": "household", "post": "misc", "wedding": "wedding",
    "baby": "baby", "running": "running",
}
SIMCLASS = {"fastfood": "restaurant", "bakery": "coffee", "convenience": "gas", "warehouse": "bigbox",
            "dollar": "bigbox", "dept": "clothing", "vet": "pet", "dental": "medical", "beauty": "salon",
            "gig_pay": "payroll", "gov_income": "payroll", "interest": "payroll", "invest_xfer": "savings_xfer",
            "gas_util": "electric", "water": "electric", "hotel": "airline", "carrental": "airline",
            "liquor": "bar", "internet": "phone"}
INFLOW_KINDS = {"payroll", "gig_pay", "gov_income", "interest"}

# name|kind|descriptor(s, ';')|lo|hi|flags (R refundable, O opaque to a reader, P=purpose set,
# tail=online location, ach = descriptor is COMPANY/DESCRIPTION, clean=enriched display name)
NATIONAL = r"""
Kroger|grocery|KROGER #{n3}|12|170|
Safeway|grocery|SAFEWAY #{n4}|12|160|
Publix|grocery|PUBLIX #{n4}|12|170|
Whole Foods|grocery|WHOLEFDS {A3} {n5}|9|150|
Trader Joe's|grocery|TRADER JOE S #{n3}|12|130|P=tj
Aldi|grocery|ALDI {n5}|15|120|
H-E-B|grocery|H-E-B #{n3}|15|180|
Wegmans|grocery|WEGMANS #{n3}|15|190|
Sprouts|grocery|SPROUTS FARMERS MKT #{n3}|10|120|
Hy-Vee|grocery|HY-VEE {CITY8} {n4}|12|160|
Harris Teeter|grocery|HARRIS TEETER #{n4}|12|160|
Food Lion|grocery|FOOD LION #{n4}|12|130|
Giant Eagle|grocery|GIANT EAGLE #{n4}|12|170|
Lidl|grocery|LIDL US {n4}|12|110|
WinCo Foods|grocery|WINCO FOODS #{n3}|20|170|
Instacart|grocery|INSTACART*{R6}|30|190|tail=SAN FRANCISCO CA
HelloFresh|grocery|HELLOFRESH {R6}|59.99|89.99|tail=NEW YORK NY
Target|bigbox|TARGET {n8};TARGET T-{n4}|4|190|R,P=target
Walmart|bigbox|WAL-MART #{n4};WM SUPERCENTER #{n4}|3|230|R,P=walmart
Costco|warehouse|COSTCO WHSE #{n4}|20|480|R,P=costco
Sam's Club|warehouse|SAMSCLUB #{n4}|20|360|R,P=costco
BJ's Wholesale|warehouse|BJS WHOLESALE #{n4}|20|320|R,P=costco
Meijer|bigbox|MEIJER #{n3}|8|210|P=meijer
Fred Meyer|bigbox|FRED-MEYER #{n3}|8|210|P=meijer
Dollar Tree|dollar|DOLLAR TREE {n5}|2|30|P=dollar
Dollar General|dollar|DOLLAR-GENERAL #{n5}|2|40|P=dollar
Amazon|online|AMZN Mktp US*{R6};Amazon.com*{R6};AMAZON MKTPL*{R6}|4|260|R,P=amazon,tail=Amzn.com/bill WA
Etsy|online|ETSY.COM - {SHOP}|8|95|R,P=etsy,tail=BROOKLYN NY
eBay|online|EBAY O*{n2}-{n5}-{n5}|6|180|R,P=ebay,tail=SAN JOSE CA
Temu|online|TEMU.COM|5|70|R,P=temu,tail=BOSTON MA
PayPal|online|PAYPAL *{R8}|5|120|R,O,P=paypal,tail=402-935-7733 CA
Apple|online|APPLE.COM/BILL|0.99|19.99|P=apple,tail=866-712-7753 CA
Google|online|GOOGLE *SERVICES|1.99|13.99|P=google,tail=G.CO/HELPPAY# CA
Microsoft|online|MICROSOFT*STORE|6.99|99.99|P=microsoft,tail=MSBILL.INFO WA
Wayfair|furniture|WAYFAIR*{n8}|30|650|R,tail=BOSTON MA
IKEA|furniture|IKEA {CITY8}|15|700|R
HomeGoods|dept|HOMEGOODS #{n4}|10|120|R,P=homegoods
Container Store|furniture|THE CONTAINER STORE {n3}|12|180|R
Olive Garden|restaurant|OLIVE GARDEN {n4}|25|120|
Chili's|restaurant|CHILI'S {n4}|20|95|
Applebee's|restaurant|APPLEBEES NEIGH GRILL {n4}|20|90|
Texas Roadhouse|restaurant|TEXAS ROADHOUSE #{n4}|30|140|
Red Robin|restaurant|RED ROBIN NO {n3}|25|100|
Cheesecake Factory|restaurant|CHEESECAKE {CITY8}|35|160|
Outback|restaurant|OUTBACK {n4}|30|140|
IHOP|restaurant|IHOP #{n4}|15|70|
Denny's|restaurant|DENNYS #{n4}|14|60|
Buffalo Wild Wings|restaurant|BWW #{n4} {CITY8}|20|110|
P.F. Chang's|restaurant|PF CHANGS #{n4}|30|130|
Cracker Barrel|restaurant|CRACKER BARREL #{n3}|18|80|
McDonald's|fastfood|MCDONALD'S F{n5}|4|28|
Chick-fil-A|fastfood|CHICK-FIL-A #{n5}|6|42|
Chipotle|fastfood|CHIPOTLE {n4}|9|38|
Taco Bell|fastfood|TACO BELL #{n6}|5|28|
Wendy's|fastfood|WENDYS #{n4}|5|26|
Burger King|fastfood|BURGER KING #{n5}|5|26|
Subway|fastfood|SUBWAY {n8}|7|30|
Panera|fastfood|PANERA BREAD #{n6}|9|45|
Five Guys|fastfood|FIVE GUYS {ST} {n4}|12|48|
Sweetgreen|fastfood|SWEETGREEN {CITY8}|12|32|
Panda Express|fastfood|PANDA EXPRESS #{n4}|8|35|
Domino's|fastfood|DOMINO'S {n4}|12|45|
Pizza Hut|fastfood|PIZZA HUT {n6}|12|45|
Jersey Mike's|fastfood|JERSEY MIKES {n5}|9|36|
Shake Shack|fastfood|SHAKE SHACK {n4}|10|45|
Wingstop|fastfood|WINGSTOP {n4}|12|45|
Raising Cane's|fastfood|RAISING CANES {n4}|8|35|
Popeyes|fastfood|POPEYES {n5}|6|30|
Sonic|fastfood|SONIC DRIVE IN #{n4}|4|25|
Culver's|fastfood|CULVERS OF {CITY8}|7|35|
Whataburger|fastfood|WHATABURGER {n4}|7|32|
Qdoba|fastfood|QDOBA #{n4}|9|32|
Jack in the Box|fastfood|JACK IN THE BOX {n4}|5|25|
Dairy Queen|fastfood|DQ #{n5}|4|22|
Crumbl|bakery|CRUMBL {CITY8}|5|30|
Starbucks|coffee|STARBUCKS STORE {n5}|3.5|16|
Dunkin'|coffee|DUNKIN #{n6}|2.5|14|
Peet's|coffee|PEETS {n5}|3.5|14|
Dutch Bros|coffee|DUTCH BROS {n5}|4|14|
Caribou Coffee|coffee|CARIBOU COFFEE {n4}|3.5|14|
Philz|coffee|PHILZ COFFEE {CITY8}|4.5|15|
DoorDash|delivery|DOORDASH*{REST}|14|70|tail=SAN FRANCISCO CA
Grubhub|delivery|GRUBHUB*{REST}|14|65|tail=CHICAGO IL
Uber|rideshare|UBER *TRIP {R4}|8|48|P=uber,tail=HELP.UBER.COM CA
Lyft|rideshare|LYFT   *RIDE {WDU} {HR}|7|45|tail=SAN FRANCISCO CA
Total Wine|liquor|TOTAL WINE AND MORE {n4}|12|120|
Shell|gas|SHELL OIL {n11}|2|80|P=gas
Chevron|gas|CHEVRON {n7}|2|85|P=gas
ExxonMobil|gas|EXXONMOBIL {n8}|2|80|P=gas
BP|gas|BP#{n7}|2|80|P=gas
Speedway|gas|SPEEDWAY {n5}|2|75|P=gas
Circle K|gas|CIRCLE K {n5}|2|75|P=gas
Marathon|gas|MARATHON PETRO{n6}|2|75|P=gas
Valero|gas|VALERO {n4}|2|75|P=gas
Arco|gas|ARCO #{n5}|2|75|P=gas
Costco Gas|gas|COSTCO GAS #{n4}|25|85|
7-Eleven|convenience|7-ELEVEN {n5}|1.5|70|P=conv
Wawa|convenience|WAWA {n4}|1.5|70|P=conv
Sheetz|convenience|SHEETZ {n4}|1.5|70|P=conv
QuikTrip|convenience|QT {n4}|1.5|70|P=conv
Buc-ee's|convenience|BUC-EE'S #{n2}|3|90|P=conv
Casey's|convenience|CASEYS #{n4}|1.5|70|P=conv
Jiffy Lube|auto|JIFFY LUBE #{n4}|45|160|
Valvoline|auto|VALVOLINE INSTANT OIL {n4}|55|140|
Discount Tire|auto|DISCOUNT TIRE {n4}|25|900|
Firestone|auto|FIRESTONE {n6}|40|800|
AutoZone|auto|AUTOZONE #{n4}|8|180|
O'Reilly|auto|O'REILLY {n4}|8|180|
Pep Boys|auto|PEP BOYS #{n4}|30|600|
Midas|auto|MIDAS #{n4}|40|700|
Take 5 Oil|auto|TAKE 5 OIL CHANGE {n4}|45|110|
Mister Car Wash|auto|MISTER CAR WASH {n3}|12|35|
ParkMobile|parking|PARKMOBILE {n6}|1.5|22|tail=ATLANTA GA
SpotHero|parking|SPOTHERO {n8}|8|40|tail=CHICAGO IL
PayByPhone|parking|PAYBYPHONE {CITY8}|1|15|
LAZ Parking|parking|LAZ PARKING {n6}|5|35|
Amtrak|transit|AMTRAK .COM {n10}|25|180|R,tail=WASHINGTON DC
Lime|transit|LIME*RIDE {R6}|2.5|12|tail=SAN FRANCISCO CA
Delta|airline|DELTA AIR {n13}|35|1400|R,tail=ATLANTA GA
United|airline|UNITED {n13}|35|1400|R,tail=HOUSTON TX
American Airlines|airline|AMERICAN AIR{n10}|35|1400|R,tail=FORT WORTH TX
Southwest|airline|SOUTHWES {n10}|35|1200|R,tail=DALLAS TX
JetBlue|airline|JETBLUE {n13}|35|1200|R,tail=SALT LAKE CTY UT
Alaska Airlines|airline|ALASKA AIR {n10}|35|1200|R,tail=SEATTLE WA
Frontier|airline|FRONTIER A{n8}|25|700|R,tail=DENVER CO
Marriott|hotel|MARRIOTT {CITY8}|150|1600|R
Hilton|hotel|HILTON HOTELS {CITY8}|140|1500|R
Hyatt|hotel|HYATT {CITY8}|150|1600|R
Holiday Inn Express|hotel|HOLIDAY INN EXP {CITY8}|110|900|R
Airbnb|hotel|AIRBNB * HM{R8}|180|2400|R,tail=AIRBNB.COM CA
Vrbo|hotel|VRBO {n10}|200|2600|R,tail=AUSTIN TX
Expedia|hotel|EXPEDIA {n13}|150|2000|R,tail=EXPEDIA.COM WA
Enterprise|carrental|ENTERPRISE RENT-A-CAR|90|700|
Hertz|carrental|HERTZ RENT-A-CAR|90|700|
CVS|pharmacy|CVS/PHARMACY #{n5}|2|65|P=pharm
Walgreens|pharmacy|WALGREENS #{n5}|2|65|P=pharm
Rite Aid|pharmacy|RITE AID {n5}|2|60|P=pharm
Quest Diagnostics|medical|QUEST DIAGNOSTICS|15|240|
LabCorp|medical|LABCORP|15|240|
One Medical|medical|ONE MEDICAL|25|199|tail=SAN FRANCISCO CA
MinuteClinic|medical|MINUTECLINIC {n4}|40|160|
Headway|medical|HEADWAY *COPAY|25|60|tail=NEW YORK NY
BetterHelp|medical|BETTERHELP.COM|65|100|tail=MOUNTAIN VIEW CA
Warby Parker|medical|WARBY PARKER|95|300|R
Aspen Dental|dental|ASPEN DENTAL {n4}|40|600|
The Home Depot|home_improve|THE HOME DEPOT #{n4}|4|340|R,P=homedepot
Lowe's|home_improve|LOWE'S #{n4}|4|340|R,P=homedepot
Ace Hardware|home_improve|ACE HDWE {n5}|4|90|
Harbor Freight|home_improve|HARBOR FREIGHT TOOLS {n4}|8|250|R
Sherwin-Williams|home_improve|SHERWIN-WILLIAMS{n6}|25|250|
Old Navy|clothing|OLD NAVY US {n4}|10|120|R,P=kidsclothes
Gap|clothing|GAP US {n4}|15|140|R
H&M|clothing|H&M {n4}|10|110|R
Uniqlo|clothing|UNIQLO {CITY8}|15|130|R
Nike|clothing|NIKE.COM|30|180|R,tail=BEAVERTON OR
Zara|clothing|ZARA USA {n4}|20|160|R
Lululemon|clothing|LULULEMON ATHLETICA|40|220|R
DSW|clothing|DSW {n4}|25|140|R
Shein|clothing|SHEIN.COM|12|90|R,tail=LOS ANGELES CA
Carter's|clothing|CARTERS #{n4}|12|90|R,P=carters
Kohl's|dept|KOHL'S #{n4}|12|140|R,P=dept
Macy's|dept|MACY'S {CITY8}|15|180|R,P=dept
Nordstrom Rack|dept|NORDSTROM RACK #{n3}|15|160|R,P=dept
TJ Maxx|dept|TJMAXX #{n4}|8|130|R,P=dept
Marshalls|dept|MARSHALLS #{n4}|8|130|R,P=dept
Ross|dept|ROSS STORES #{n3}|8|110|R,P=dept
Bath & Body Works|beauty|BATH&BODYWORKS {n4}|8|60|P=bbw
Best Buy|electronics|BESTBUY {n11}|15|900|R,P=bestbuy
Apple Store|electronics|APPLE STORE #R{n3}|29|1300|R
GameStop|games|GAMESTOP #{n4}|10|80|R
Micro Center|electronics|MICRO CENTER {n3}|15|700|R
Barnes & Noble|books|B&N {n4}|8|70|
Half Price Books|books|HALF PRICE BOOKS {n3}|5|40|
Audible|books|Audible*{R6}|14.95|14.95|tail=AMZN.COM/BILL NJ
Kindle|books|Kindle Svcs*{R6}|2.99|14.99|tail=888-802-3080 WA
Michaels|hobby|MICHAELS STORES {n4}|5|85|R,P=michaels
JOANN|hobby|JOANN STORES #{n4}|5|80|
Hobby Lobby|hobby|HOBBY LOBBY #{n3}|5|90|
REI|hobby|REI #{n3} {CITY8}|15|260|R,P=rei
Dick's Sporting Goods|hobby|DICK'S SPORTING GOODS {n4}|15|200|R,P=dicks
Guitar Center|hobby|GUITAR CENTER {n3}|10|400|
Steam|games|STEAMGAMES.COM {n10}|4.99|69.99|tail=425-889-9642 WA
PlayStation|games|PLAYSTATION NETWORK|9.99|69.99|tail=888-9769847 CA
Nintendo|games|NINTENDO CB{n9}|4.99|59.99|tail=800-255-3700 WA
LEGO Store|kids_activity|THE LEGO STORE {n3}|15|150|
Chuck E. Cheese|kids_activity|CHUCK E CHEESE {n4}|15|80|
Kumon|kids_activity|KUMON {CITY8}|150|190|
MySchoolBucks|school|MYSCHOOLBUCKS|10|60|tail=877-393-0020 PA
Brightwheel|childcare|BRIGHTWHEEL*{DAYCARE}|250|1900|tail=SAN FRANCISCO CA
Procare|childcare|PROCARE SOLUTIONS/TUITION|250|1900|ach
Bright Horizons|childcare|BRIGHT HORIZONS/TUITION|900|2200|ach
Chegg|school|CHEGG ORDER|15|20|tail=SANTA CLARA CA
Petco|pet|PETCO {n4}|8|120|
PetSmart|pet|PETSMART # {n4}|8|120|
Chewy|pet|CHEWY.COM|20|120|tail=800-672-4399 FL
Banfield|vet|BANFIELD PET HOSP {n4}|45|400|
BluePearl|vet|BLUEPEARL {CITY8}|150|1800|
Rover|pet|ROVER.COM|25|120|tail=SEATTLE WA
BarkBox|pet|BARKBOX|35|35|tail=NEW YORK NY
Netflix|streaming|NETFLIX.COM|15.49|22.99|tail=LOS GATOS CA
Hulu|streaming|HULU {n10}|9.99|18.99|tail=HULU.COM/BILLCA
Disney+|streaming|DISNEY PLUS|9.99|15.99|tail=BURBANK CA
Max|streaming|MAX.COM|9.99|16.99|tail=NEW YORK NY
Spotify|streaming|Spotify USA|11.99|19.99|tail=877-7781161 NY
Peacock|streaming|PEACOCK {n8}PREMIUM|7.99|13.99|tail=NEW YORK NY
Paramount+|streaming|PARAMOUNT+|7.99|12.99|tail=NEW YORK NY
SiriusXM|streaming|SIRIUSXM.COM/ACCT|10.99|19.99|tail=888-635-5144 NY
Crunchyroll|streaming|CRUNCHYROLL*MEMBER|7.99|11.99|tail=SAN FRANCISCO CA
Adobe|software|ADOBE *ADOBE|12.99|59.99|tail=SAN JOSE CA
Dropbox|software|DROPBOX*{R6}|11.99|19.99|tail=SAN FRANCISCO CA
OpenAI|software|OPENAI *CHATGPT SUBSCR|20|20|tail=SAN FRANCISCO CA
1Password|software|1PASSWORD|35.88|35.88|tail=TORONTO ON
Notion|software|NOTION LABS INC|10|10|tail=SAN FRANCISCO CA
NYTimes|software|NYTIMES*NYTDIGITAL|4|25|tail=NEW YORK NY
Duolingo|software|DUOLINGO|83.99|83.99|tail=PITTSBURGH PA
Patreon|entertainment|PATREON* MEMBERSHIP|5|25|tail=SAN FRANCISCO CA
Paddle|software|PADDLE.NET* {R6}|5|60|tail=LONDON GB
FastSpring|software|FS *{R6}|10|80|tail=SANTA BARBARA CA
Canva|biz|CANVA* {n9}|14.99|14.99|tail=SYDNEY AU
Zoom|biz|ZOOM.US 888-799-9666|15.99|15.99|tail=CA
Squarespace|biz|SQSP* INV{n8}|23|36|tail=NEW YORK NY
GoDaddy|biz|DNH*GODADDY.COM|21.99|120|tail=480-505-8855 AZ
QuickBooks|biz|INTUIT *QBooks Online|35|35|tail=CL.INTUIT.COM CA
Upwork|biz|UPWORK -{n9}REF|10|60|tail=SAN FRANCISCO CA
Staples|biz|STAPLES {n5}|6|160|R
FedEx Office|biz|FEDEX OFFIC{n5}|5|90|
USPS|post|USPS PO {n10}|4|40|
Planet Fitness|gym|PLANET FITNESS/CLUB FEES|10|24.99|ach
Orangetheory|gym|ORANGETHEORY #{n4}|59|189|
Peloton|gym|PELOTON* MEMBERSHIP|44|44|tail=NEW YORK NY
24 Hour Fitness|gym|24 HOUR FITNESS USA|35|55|
YMCA|gym|YMCA OF {CITY8}|40|110|
ClassPass|gym|CLASSPASS* MONTHLY|19|79|tail=NEW YORK NY
Verizon|phone|VZWRLSS*APOCC VISB|40|240|tail=800-922-0204 FL
AT&T|phone|ATT*BILL PAYMENT|40|240|tail=800-288-2020 TX
T-Mobile|phone|T-MOBILE*AUTO PAY|40|240|tail=800-937-8997 WA
Mint Mobile|phone|MINT MOBILE|15|45|tail=800-683-7392 CA
Visible|phone|VISIBLE|25|45|tail=DENVER CO
Xfinity|internet|COMCAST CABLE COMM|55|120|tail=800-266-2278 PA
Spectrum|internet|SPECTRUM|55|110|tail=855-707-7328 MO
Google Fiber|internet|GOOGLE *FIBER|70|70|tail=G.CO/HELPPAY# CA
Starlink|internet|STARLINK INTERNET|120|120|tail=HAWTHORNE CA
GEICO|auto_ins|GEICO/AUTO|80|220|ach
Progressive|auto_ins|PROGRESSIVE INS/INS PREM|80|230|ach
State Farm|auto_ins|STATE FARM RO 27/SFPP|60|240|ach
Allstate|auto_ins|ALLSTATE/INS PREM|80|230|ach
USAA|auto_ins|USAA P&C INT/AUTOPAY|70|220|ach
Lemonade|home_ins|LEMONADE INS/PREMIUM|9|30|ach
Haven Life|life_ins|HAVEN LIFE/PREMIUM|20|65|ach
Ladder Life|life_ins|LADDER LIFE/INS PREM|18|70|ach
BCBS|health_ins|BCBS/PREMIUM|320|760|ach
AARP Medicare Supplement|health_ins|UHC AARP MEDSUPP/PREMIUM|150|260|ach
Oscar Health|health_ins|OSCAR HEALTH/PREMIUM|300|700|ach
Toyota Financial|car_payment|TOYOTA FINANCIAL/TFS PYMT|280|650|ach
American Honda Finance|car_payment|AMERICAN HONDA FIN/PAYMENT|280|600|ach
Ally Auto|car_payment|ALLY/ALLY PAYMT|280|650|ach
Nelnet|student_loan|NELNET/LOAN PAYMT|120|480|ach
MOHELA|student_loan|MOHELA/STUDNTLOAN|120|480|ach
Sallie Mae|student_loan|SALLIE MAE BANK/PAYMENT|90|400|ach
Rocket Mortgage|mortgage|ROCKET MORTGAGE/LOAN PMT|1100|3400|ach
Mr. Cooper|mortgage|MR. COOPER/MTG PYMT|1100|3400|ach
PennyMac|mortgage|PENNYMAC/MTG PAYMT|1100|3400|ach
Bilt Rent|rent|BILT*RENT {R6}|1100|2900|tail=NEW YORK NY
AppFolio|rent|APPFOLIO/{PROP}|1100|2900|ach,O
RentCafe|rent|YARDI RENTCAFE/RENT|1100|2900|ach
CINC HOA|hoa|CINC SYSTEMS/HOA DUES|120|480|ach
Chase Card|cc_pay|CHASE CREDIT CRD/AUTOPAY|150|3000|ach
Citi Card|cc_pay|CITI AUTOPAY/PAYMENT|150|3000|ach
Amex|cc_pay|AMEX EPAYMENT/ACH PMT|150|3000|ach
Discover|cc_pay|DISCOVER/E-PAYMENT|150|3000|ach
Ally Bank|savings_xfer|ALLY BANK $TRANSFER/P2P|50|800|ach
Marcus|savings_xfer|GS BANK USA/MARCUS|50|800|ach
Robinhood|invest_xfer|ROBINHOOD/FUNDS|25|600|ach
Vanguard|invest_xfer|VANGUARD BUY/INVESTMENT|50|1000|ach
Fidelity|invest_xfer|FID BKG SVC LLC/MONEYLINE|50|1000|ach
Acorns|invest_xfer|ACORNS INVEST/TRANSFER|5|60|ach
IRS Payment|tax|IRS/USATAXPYMT|300|4000|ach
Great Clips|salon|GREAT CLIPS #{n4}|18|35|
Supercuts|salon|SUPERCUTS {n4}|18|35|
Sephora|beauty|SEPHORA {n4}|12|140|R
Ulta|beauty|ULTA #{n3}|10|120|R
European Wax Center|salon|EUROPEAN WAX CTR {n4}|40|90|
Massage Envy|salon|MASSAGE ENVY {n4}|70|130|
AMC|entertainment|AMC {n4} ONLINE|12|60|
Regal|entertainment|REGAL {CITY8}|12|60|
Fandango|entertainment|FANDANGO.COM|14|70|tail=866-857-5191 CA
Ticketmaster|entertainment|TICKETMASTER*{R8}|35|450|R,tail=800-653-8000 CA
StubHub|entertainment|STUBHUB {R6}|40|400|tail=SAN FRANCISCO CA
Eventbrite|entertainment|EB *{EVENT}|10|120|O,tail=801-413-7200 CA
Dave & Buster's|entertainment|DAVE & BUSTERS #{n3}|20|110|
Topgolf|entertainment|TOPGOLF {CITY8}|35|160|
American Red Cross|charity|AMERICAN RED CROSS|20|200|tail=WASHINGTON DC
St. Jude|charity|ST JUDE CHILDRENS RES|19|100|tail=MEMPHIS TN
Wikimedia|charity|WIKIMEDIA FOUNDATION|5|50|tail=SAN FRANCISCO CA
GoFundMe|charity|GOFUNDME *{GFM}|20|200|O,tail=REDWOOD CITY CA
1-800-Flowers|florist|1-800-FLOWERS.COM|40|120|tail=CARLE PLACE NY
Edible Arrangements|florist|EDIBLE ARRANGEMENTS|40|110|
Hallmark|florist|HALLMARK {n5}|4|40|
U-Haul|home_improve|U-HAUL {n6}|30|400|
Fleet Feet|running|FLEET FEET {CITY8}|20|180|R
RunSignup|running|RUNSIGNUP*{RACE}|35|160|tail=MOORESTOWN NJ
Strava|running|STRAVA INC|11.99|11.99|tail=SAN FRANCISCO CA
David's Bridal|wedding|DAVIDS BRIDAL {n4}|90|900|
Zola|wedding|ZOLA INC|50|800|tail=NEW YORK NY
Blue Nile|wedding|BLUE NILE INC|800|4000|tail=SEATTLE WA
buybuy BABY|baby|BUYBUY BABY {n4}|15|300|R
"""

# purpose sets: (concept, weight, lo, hi[, descriptor override, clean display override])
PSETS = {
    "tj": [("groceries", 88, 12, 130), ("alcohol", 12, 8, 45)],
    "target": [("household", 33, 6, 90), ("groceries", 25, 15, 140), ("clothing", 12, 12, 80), ("kids", 12, 8, 70),
               ("gifts", 6, 12, 70), ("personal_care", 9, 5, 40), ("baby", 0, 15, 80)],
    "walmart": [("groceries", 50, 20, 230), ("household", 28, 6, 80), ("kids", 8, 8, 60), ("car_maint", 4, 12, 120),
                ("electronics", 4, 20, 300), ("pets", 6, 8, 50), ("baby", 0, 15, 70)],
    "costco": [("groceries", 58, 80, 430), ("household", 27, 25, 180), ("electronics", 5, 70, 600), ("gifts", 5, 20, 120),
               ("pharmacy", 5, 8, 40)],
    "meijer": [("groceries", 68, 15, 200), ("household", 32, 6, 70)],
    "dollar": [("household", 60, 2, 25), ("groceries", 25, 3, 30), ("gifts", 15, 3, 20)],
    "amazon": [("household", 33, 6, 70), ("electronics", 8, 20, 260), ("kids", 12, 8, 60), ("gifts", 8, 12, 95),
               ("books", 6, 6, 30), ("clothing", 8, 12, 75), ("pets", 6, 18, 65), ("hobbies", 7, 8, 85),
               ("personal_care", 5, 6, 40), ("home_improve", 5, 12, 130), ("baby", 0, 12, 70)],
    "etsy": [("gifts", 50, 12, 85), ("household", 30, 10, 70), ("hobbies", 20, 6, 40)],
    "ebay": [("hobbies", 35, 8, 120), ("electronics", 25, 15, 180), ("clothing", 20, 10, 60), ("car_maint", 20, 12, 90)],
    "temu": [("household", 55, 4, 40), ("clothing", 25, 5, 35), ("hobbies", 20, 3, 25)],
    "paypal": [("household", 25, 8, 90, "PAYPAL *EBAY", "PayPal eBay"), ("games", 15, 5, 40, "PAYPAL *STEAM GAMES", "PayPal Steam"),
               ("charity", 15, 10, 60, "PAYPAL *GIVEFUND", "PayPal Giving Fund"), ("hobbies", 25, 8, 70),
               ("clothing", 20, 15, 90)],
    "apple": [("software", 40, 0.99, 9.99), ("streaming", 30, 10.99, 10.99), ("games", 20, 0.99, 9.99),
              ("entertainment", 10, 3.99, 19.99)],
    "google": [("streaming", 40, 13.99, 13.99, "GOOGLE *YouTubePremium", "YouTube Premium"),
               ("software", 40, 1.99, 9.99, "GOOGLE *Google One", "Google One"),
               ("games", 20, 0.99, 19.99, "GOOGLE *Google Play", "Google Play")],
    "microsoft": [("software", 60, 9.99, 99.99, "MICROSOFT*365 PERSONAL", "Microsoft 365"),
                  ("games", 40, 16.99, 19.99, "MICROSOFT*XBOX GAME PASS", "Xbox")],
    "uber": [("rideshare", 60, 8, 48, "UBER *TRIP {R4}", "Uber"), ("takeout", 40, 16, 58, "UBER *EATS {R4}", "Uber Eats")],
    "gas": [("fuel", 82, 22, 85), ("snacks", 18, 1.5, 12)],
    "conv": [("fuel", 55, 22, 80), ("snacks", 45, 1.5, 15)],
    "pharm": [("pharmacy", 45, 4, 65), ("personal_care", 28, 5, 40), ("household", 12, 4, 30), ("snacks", 8, 2, 9),
              ("gifts", 7, 4, 25)],
    "homedepot": [("home_improve", 75, 6, 340), ("household", 25, 8, 60)],
    "homegoods": [("household", 70, 10, 120), ("gifts", 30, 10, 60)],
    "dept": [("clothing", 55, 12, 95), ("household", 30, 8, 70), ("gifts", 15, 12, 60)],
    "bbw": [("personal_care", 60, 8, 45), ("gifts", 40, 10, 50)],
    "carters": [("kids", 80, 12, 90), ("gifts", 20, 12, 50)],
    "kidsclothes": [("clothing", 60, 10, 120), ("kids", 40, 10, 70)],
    "bestbuy": [("electronics", 85, 15, 900), ("gifts", 15, 25, 150)],
    "rei": [("hobbies", 70, 15, 260), ("clothing", 30, 20, 150)],
    "dicks": [("hobbies", 50, 15, 200), ("kids", 25, 15, 120), ("clothing", 25, 20, 100)],
    "michaels": [("hobbies", 75, 5, 85), ("kids", 15, 5, 40), ("gifts", 10, 8, 40)],
}
ETSY_SHOPS = ["WILDROSEPRINTS", "OAKANDTHREAD", "LITTLEFOXCO", "PAPERMOONSTUDIO", "CLAYBYMAE", "NORTHWICKCANDLE"]
EVENTS = ["SPRING FLING 2025", "TRIVIA NIGHT", "5K FUN RUN", "MAKERS MARKET", "JAZZ ON THE LAWN", "COMEDY SHOWCASE"]
GFM = ["HELP THE REYES FAMILY", "MEDICAL BILLS FOR DAN", "SAVE THE CO-OP", "COACH MIKE MEMORIAL"]
RACES = ["RIVERFRONT HALF", "TURKEY TROT", "CITY MARATHON", "TRAIL 10K"]
DAYCARES = ["LITTLE ACORNS ELC", "TINY STEPS", "SUNFLOWER KIDS", "BRIGHT BEGINNINGS"]


class Merchant:
    __slots__ = ("id", "kind", "desc", "lo", "hi", "tail", "refund", "pset", "reader", "clean", "channel", "city",
                 "comp", "descr", "opaque", "sim")

    def __init__(self, mid, kind, desc, lo, hi, tail=None, refund=False, pset=None, opaque=False, clean=None,
                 channel="card", city=None, comp=None, descr=None):
        self.id, self.kind, self.desc, self.lo, self.hi = mid, kind, desc, lo, hi
        self.tail, self.refund, self.pset, self.opaque = tail, refund, pset, opaque
        self.clean = clean or mid
        self.channel, self.city, self.comp, self.descr = channel, city, comp, descr
        self.sim = SIMCLASS.get(kind, kind)
        self.reader = None if opaque else self.sim


MER = {}
BY_KIND = defaultdict(list)          # national merchants by kind
LOCAL = defaultdict(lambda: defaultdict(list))   # city -> kind -> [mid]


def add_merchant(m):
    assert m.id not in MER, m.id
    MER[m.id] = m
    if m.city is None:
        BY_KIND[m.kind].append(m.id)
    else:
        LOCAL[m.city][m.kind].append(m.id)
    return m.id


for _line in NATIONAL.strip().splitlines():
    _name, _kind, _desc, _lo, _hi, _flags = _line.split("|")
    _kw = {}
    _fl = [f for f in _flags.split(",") if f]
    for f in _fl:
        if f == "R":
            _kw["refund"] = True
        elif f == "O":
            _kw["opaque"] = True
        elif f.startswith("P="):
            _kw["pset"] = f[2:]
        elif f.startswith("tail="):
            _kw["tail"] = f[5:]
    if "ach" in _fl:
        _comp, _descr = _desc.split("/", 1)
        _kw.update(channel="ach", comp=_comp, descr=_descr)
        _desc = _comp
    add_merchant(Merchant(_name, _kind, _desc.split(";"), float(_lo), float(_hi), **_kw))

# fixed special merchants (their own renderers)
for _mid, _kind in (("Venmo", "p2p"), ("Zelle", "p2p"), ("Cash App", "p2p"), ("Check", "p2p")):
    add_merchant(Merchant(_mid, _kind, [_mid.upper()], 1, 1, channel="p2p"))
add_merchant(Merchant("ATM", "atm", ["ATM"], 20, 200, channel="atm"))
add_merchant(Merchant("Savings transfer", "savings_xfer", ["XFER"], 50, 800, channel="xfer"))
for _mid, _d in (("Monthly Service Fee", "MONTHLY SERVICE FEE"), ("Overdraft Fee", "OVERDRAFT ITEM FEE"),
                 ("Foreign Transaction Fee", "FOREIGN TRANSACTION FEE"), ("ATM Fee", "NON-NETWORK ATM FEE")):
    add_merchant(Merchant(_mid, "fee", [_d], 3, 36, channel="fee"))
for _mid, _comp, _descr, _kind in (
        ("Uber Driver Payout", "UBER USA 6787", "EDI PYMNTS", "gig_pay"), ("Lyft Driver Payout", "LYFT INC", "PAYOUT", "gig_pay"),
        ("DoorDash Dasher Pay", "DOORDASH INC", "DASHER", "gig_pay"), ("Instacart Shopper Pay", "MAPLEBEAR INC", "PAYROLL", "gig_pay"),
        ("Etsy Payout", "ETSY INC", "DEPOSIT", "gig_pay"), ("Stripe", "STRIPE", "TRANSFER", "gig_pay"),
        ("Shopify Payout", "SHOPIFY", "TRANSFER", "gig_pay"), ("Upwork Earnings", "UPWORK ESCROW INC", "PAYMENT", "gig_pay"),
        ("Airbnb Host Payout", "AIRBNB PAYMENTS", "AIRBNB", "gig_pay"), ("PayPal Transfer", "PAYPAL", "TRANSFER", "gig_pay"),
        ("Social Security", "SSA TREAS 310", "XXSOC SEC", "gov_income"), ("IRS Tax Refund", "IRS TREAS 310", "TAX REF", "gov_income"),
        ("State Tax Refund", "STATE TREASURY", "TAX REFUND", "gov_income"),
        ("Teachers Retirement System", "TEACHERS RET SYS", "PENSION", "gov_income"),
        ("Interest", "INTEREST", "PAYMENT", "interest")):
    add_merchant(Merchant(_mid, _kind, [_comp], 1, 1, channel="ach", comp=_comp, descr=_descr))
# Instacart Shopper Pay shows as MAPLEBEAR (Instacart's legal name): only a knowledgeable reader knows.
EMPLOYERS = ["NORTHWIND LOGISTICS", "BRIGHTPATH SOFTWARE", "MERIDIAN HEALTH SYS", "CITY OF RIVERTON", "HALCYON FOODS",
             "SUMMIT UNIFIED SD", "ORCHARD ANALYTICS", "KESTREL AEROSPACE", "BLUEWATER CREDIT UN", "PINNACLE RETAIL GRP",
             "CEDARLINE MFG", "VANTAGE DENTAL PARTNERS"]
for _e in EMPLOYERS:
    add_merchant(Merchant(_e.title(), "payroll", [_e], 1, 1, channel="ach", comp=_e, descr="PAYROLL"))
for _p in ("GUSTO", "PAYCHEX INC", "ADP"):
    add_merchant(Merchant(_p.title() + " Payroll", "payroll", [_p], 1, 1, channel="ach", comp=_p, descr="PAYROLL"))
CLIENTS = ["LUMEN & OAK LLC", "KDS HOLDINGS", "NORTHSTAR CREATIVE", "BRIGHTWORKS MEDIA", "TALLGRASS PARTNERS", "ZEPHYR LABS INC"]
for _c in CLIENTS:
    add_merchant(Merchant(_c.title(), "gig_pay", [_c], 1, 1, channel="ach", comp=_c, descr="ACH PMT", opaque=True))

# per-city billers and transit
for _city, (_st, _near, _el, _gas, _wat, _tr, _county) in CITIES.items():
    add_merchant(Merchant(_el[0], "electric", [_el[1]], 40, 260, channel="ach", comp=_el[1], descr=_el[2], city=_city))
    if _gas:
        add_merchant(Merchant(_gas[0], "gas_util", [_gas[1]], 15, 220, channel="ach", comp=_gas[1], descr=_gas[2], city=_city))
    add_merchant(Merchant(_wat[0], "water", [_wat[1]], 35, 180, channel="ach", comp=_wat[1], descr=_wat[2], city=_city))
    add_merchant(Merchant(_tr[0], "transit", [_tr[1]], 2, 100, city=_city))
    add_merchant(Merchant(f"{_county} County Treasurer", "tax", [f"{_county.upper()} CO TREAS"], 900, 4200,
                          channel="ach", comp=f"{_county.upper()} CO TREAS", descr="PROP TAX", city=_city))

# invented local businesses per city (home cities and trip destinations)
ADJ = ["Golden", "Blue", "Little", "Old", "Red", "Wild", "Lucky", "Happy", "Green", "Silver", "Rusty", "Crooked",
       "Humble", "Sunny", "Copper", "Northside", "Eastside", "Maple", "Cedar", "Hilltop", "Riverside", "Midtown",
       "Iron", "Velvet", "Salty", "Brave", "Quiet", "Tidal", "Prairie", "Juniper"]
NOUN = ["Fern", "Fox", "Owl", "Bear", "Sparrow", "Anchor", "Lantern", "Barrel", "Kettle", "Spoon", "Pony", "Moon",
        "Oak", "Willow", "Harbor", "Bridge", "Mill", "Garden", "Tiger", "Crane", "Rabbit", "Goat", "Hen", "Pearl",
        "Stone", "Otter", "Heron", "Thistle", "Magpie", "Bison"]
PEOPLE = ["Rosa's", "Tony's", "Mama Lin's", "Big Al's", "Nguyen's", "Priya's", "Luca's", "Abuela's", "Kim's", "Omar's"]
SURN = ["MORALES", "OKAFOR", "LINDQVIST", "PATEL", "BRENNAN", "TANAKA", "HOLLOWAY", "DUBOIS", "KOWALSKI", "ABERNATHY"]
LOCAL_SPEC = [
    # kind, suffixes, count, lo, hi, processor prefixes
    ("coffee", ["Coffee", "Coffee Co", "Roasters", "Espresso Bar", "Cafe"], 3, 3.5, 15, ["SQ *", "TST* ", ""]),
    ("bakery", ["Bakery", "Donuts", "Bagels", "Pastry"], 2, 4, 32, ["SQ *", ""]),
    ("restaurant", ["Taqueria", "Pho", "Pizza", "BBQ", "Diner", "Sushi", "Ramen", "Kitchen", "Bistro", "Thai", "Grill",
                    "Noodle House", "Tavern", "Deli", "Cantina", "Curry House"], 7, 12, 115, ["TST* ", "SQ *", "", ""]),
    ("bar", ["Tap Room", "Brewing", "Saloon", "Pub", "Cocktail Club"], 2, 10, 75, ["TST* ", "SQ *", ""]),
    ("grocery", ["Market", "Co-op", "Mercado", "Farmers Market"], 1, 8, 95, ["", "SQ *"]),
    ("salon", ["Salon", "Hair Studio", "Nail Bar"], 1, 30, 150, ["SQ *", "", "GLOSSGENIUS*"]),
    ("salon", ["Barbershop", "Barber Co"], 1, 20, 55, ["SQ *", ""]),
    ("auto", ["Auto Repair", "Automotive", "Tire & Auto"], 1, 60, 950, ["", "CLV*"]),
    ("vet", ["Animal Hospital", "Veterinary Clinic"], 1, 60, 480, [""]),
    ("dental", ["Family Dental", "Dental Group"], 1, 40, 420, [""]),
    ("medical", ["Pediatrics", "Urgent Care", "Physical Therapy", "Dermatology"], 2, 20, 260, [""]),
    ("gym", ["Yoga", "Pilates", "CrossFit", "Climbing Gym"], 1, 15, 165, ["SQ *", "MINDBODY*", ""]),
    ("kids_activity", ["Swim School", "Soccer Club", "Dance Academy", "Gymnastics", "Music School", "Martial Arts"], 2,
     40, 230, ["SQ *", "JACKRABBIT*", ""]),
    ("home_improve", ["Hardware", "Garden Center", "Nursery"], 1, 8, 160, [""]),
    ("books", ["Books", "Bookshop"], 1, 8, 60, ["SQ *", ""]),
    ("florist", ["Flowers", "Florist"], 1, 25, 115, ["SQ *", ""]),
    ("laundry", ["Cleaners", "Laundry"], 1, 8, 48, ["SQ *", ""]),
    ("liquor", ["Liquor", "Wine Shop", "Bottle Shop"], 1, 10, 85, [""]),
    ("entertainment", ["Cinema", "Escape Room", "Bowl", "Comedy Club", "Arcade"], 1, 12, 95, ["SQ *", ""]),
    ("church", ["Community Church", "Fellowship", "Parish"], 1, 20, 300, ["TITHE.LY*", "PUSHPAY*"]),
]
DEST_SPEC = [
    ("restaurant", ["Taqueria", "Seafood", "Oyster Bar", "Kitchen", "Steakhouse", "Poke", "Diner", "Grill"], 5, 15, 160, ["TST* ", "SQ *", ""]),
    ("coffee", ["Coffee", "Cafe", "Roasters"], 2, 4, 16, ["SQ *", ""]),
    ("bar", ["Tap Room", "Cocktail Club", "Lounge"], 1, 12, 90, ["TST* ", ""]),
    ("entertainment", ["Tours", "Aquarium", "Museum", "Boat Tours", "Surf School", "Ski Rental"], 2, 20, 240, ["", "FH* "]),
]


def _gen_locals():
    rng = random.Random(SEED + 7)
    used = set(MER)
    for city, spec in [(c, LOCAL_SPEC) for c in CITIES] + [(d[0], DEST_SPEC) for d in DESTS]:
        for kind, suffixes, count, lo, hi, prefixes in spec:
            for _ in range(count):
                for _try in range(50):
                    suf = rng.choice(suffixes)
                    r = rng.random()
                    if r < 0.2:
                        first = rng.choice(PEOPLE)
                    elif r < 0.55:
                        first = f"{rng.choice(ADJ)} {rng.choice(NOUN)}"
                    else:
                        first = rng.choice(NOUN if rng.random() < 0.5 else ADJ)
                    name = f"{first} {suf}"
                    if name not in used:
                        break
                used.add(name)
                pre = rng.choice(prefixes)
                body = name.upper().replace("'", "")
                if pre:
                    body = body[:20]
                desc = pre + body
                # Toast (TST*) and tithing platforms tell a knowledgeable reader the kind even when truncated
                key = suf.split()[0].upper()[:4]
                visible = key in desc or pre.startswith(("TST*", "TITHE", "PUSHPAY", "MINDBODY", "JACKRABBIT", "GLOSSGENIUS"))
                m = Merchant(name, kind, [desc], lo, hi, city=city, opaque=not visible,
                             clean=name if visible else desc.split("*")[-1].strip().title())
                add_merchant(m)
        if spec is LOCAL_SPEC:
            # opaque local names: food trucks, handymen, tutors, estheticians behind processor names
            for kind, lo, hi in (("restaurant", 9, 28), ("home_improve", 80, 900), ("kids_activity", 45, 120),
                                 ("salon", 45, 120), ("medical", 90, 180)):
                for _try in range(50):
                    init = "".join(rng.choice(string.ascii_uppercase) for _ in range(rng.choice((2, 3))))
                    style = rng.random()
                    if style < 0.35:
                        desc = f"SQ *{init} LLC"
                    elif style < 0.6:
                        desc = f"IN *{rng.choice(SURN)} {rng.choice(['HOLDINGS', 'GROUP', 'ENTERPRISES'])}"
                    elif style < 0.8:
                        desc = f"CLV*{init} ENTERPRISES"
                    else:
                        desc = f"PY *{init} {rng.choice(SURN)} INC"
                    name = desc.split("*", 1)[1].strip().title()
                    if name not in used:
                        break
                used.add(name)
                add_merchant(Merchant(name, kind, [desc], lo, hi, city=city, opaque=True, clean=name))


_gen_locals()
DELIVERY_RESTS = ["CHIPOTLE", "TACOBELL", "PANDAEXPRESS", "THAI HOUSE", "SHAKESHACK", "WINGSTOP", "JERSEYMIKES",
                  "SWEETGREEN", "PHO 88", "POKEWORKS", "FIVEGUYS", "MENDOCINO FARMS"]
GROCERY_CHAINS = [m for m in BY_KIND["grocery"] if m not in ("Instacart", "HelloFresh")]
RESTAURANT_CHAINS = BY_KIND["restaurant"]
FASTFOOD_CHAINS = BY_KIND["fastfood"]
COFFEE_CHAINS = BY_KIND["coffee"]
GAS_CHAINS = BY_KIND["gas"] + BY_KIND["convenience"]


# ----------------------------------------------------------------------------------------------
# Users
# ----------------------------------------------------------------------------------------------
TEMPLATES = {
    "family": "groceries dining coffee?0.4 takeout?0.3 fuel car_maint auto_ins?0.6 mortgage utilities|electric+gas_util+water "
              "internet phone streaming|subscriptions KIDS childcare?0.5 household home_improve clothing medical "
              "pharmacy?0.6 pets?0.6 gifts giving?0.5 personal_care entertainment travel?0.7 savings income misc "
              "car_payment?0.5 home_ins?0.4",
    "minimal": "food_all rent|mortgage bills transport|car fun?0.7 stuff?0.5 health?0.4 income savings?0.3 misc",
    "detailed": "groceries coffee dining takeout alcohol?0.7 snacks?0.3 fuel car_maint parking?0.5 auto_ins car_payment?0.6 "
                "mortgage electric gas_util water internet phone streaming software gym medical pharmacy "
                "personal_care household home_improve clothing electronics hobbies games?0.4 entertainment KIDS "
                "childcare?0.6 school?0.4 pets gifts church?0.4 giving travel cash taxes?0.4 savings investing "
                "cc_payment?0.5 income misc home_ins life_ins?0.3 health_ins?0.3",
    "couple": "groceries work_lunch date_night dining fuel|transport rent|mortgage utilities internet+phone|bills "
              "subscriptions pets?0.5 FUN2 household gifts travel?0.7 income savings misc coffee?0.4",
    "gig": "groceries|food_all dining?0.6 fuel car_maint auto_ins phone rent bills|utilities+internet taxes business?0.6 "
           "income gig_income fun insurance?0.3 misc savings?0.4 health?0.5",
    "freelancer": "groceries coffee dining rent|mortgage utilities internet phone business software?0.5 taxes income "
                  "gig_income?0.3 health fun travel?0.5 misc savings clothing?0.5",
    "student": "rent food_all|groceries+dining coffee?0.6 transit phone school fun stuff?0.6 income misc",
    "retiree": "groceries dining mortgage?0.4 utilities|bills phone?0.5 medical pharmacy insurance church|giving gifts "
               "travel car household pets?0.4 income savings?0.4 misc personal_care?0.5",
    "stores": "groceries STORES dining coffee?0.5 fuel mortgage|rent bills household?0.6 clothing KIDS?0.6 "
              "subscriptions medical gifts entertainment income savings misc",
    "roommates": "rent utilities internet groceries dining transit|fuel fun subscriptions?0.6 clothing?0.5 "
                 "personal_care?0.5 income savings?0.5 misc phone",
    "traveler": "groceries dining coffee rent|mortgage bills transport fun travel?0.3 income savings misc clothing?0.5 gym?0.6",
}
ARCH_W = [("family", 12), ("detailed", 6), ("minimal", 12), ("couple", 12), ("gig", 9), ("freelancer", 8), ("student", 8),
          ("retiree", 8), ("coined", 10), ("stores", 8), ("roommates", 6), ("traveler", 5)]
STYLE_P = {"std": (0.85, 0.13, 0.02), "mixed": (0.5, 0.4, 0.1), "personal": (0.25, 0.65, 0.1), "coined": (0.1, 0.35, 0.55)}
KID_NAMES = ["Emma", "Noah", "Ava", "Liam", "Mia", "Leo", "Zoe", "Eli", "Maya", "Owen", "Lily", "Theo"]
PARTNERS = ["Alex", "Sam", "Jordan", "Chris", "Taylor", "Morgan", "Jamie", "Riley", "Casey", "Pat", "Dana", "Robin"]
FIRST = ["Maria", "James", "Linda", "Kevin", "Aisha", "Tom", "Priya", "Diego", "Hannah", "Marcus", "Grace", "Ben",
         "Sofia", "Jake", "Nina", "Carlos", "Emily", "Raj", "Olivia", "Luis", "Keisha", "Dan", "Megan", "Hector"]
LAST = ["Lopez", "Miller", "Nguyen", "Smith", "Patel", "Johnson", "Kim", "Garcia", "Brown", "Okafor", "Davis",
        "Martinez", "Wilson", "Chen", "Anderson", "Reyes", "Clark", "Hughes"]
STYLES = [("chase", 22), ("boa", 16), ("wells", 12), ("cu", 12), ("fixed", 10), ("clean", 18), ("amex", 10)]
CARD_KINDS_LOCAL = {"grocery", "bigbox", "warehouse", "restaurant", "fastfood", "coffee", "bakery", "bar", "liquor", "gas",
                    "convenience", "entertainment", "dollar", "clothing", "dept", "beauty", "salon", "hobby", "pharmacy",
                    "parking", "transit", "books", "laundry", "home_improve"}


def wchoice(rng, pairs):
    tot = sum(w for _, w in pairs)
    x = rng.random() * tot
    for v, w in pairs:
        x -= w
        if x <= 0:
            return v
    return pairs[-1][0]


class Cat:
    __slots__ = ("id", "name", "concepts", "created", "kind")

    def __init__(self, cid, name, concepts, created, kind="base"):
        self.id, self.name, self.concepts, self.created, self.kind = cid, name, set(concepts), created, kind


class Ev:
    __slots__ = ("date", "mid", "concept", "amount", "mech", "contact", "city", "fixed", "ref", "intended", "filed",
                 "mistake", "text", "vtok", "order", "sched", "trip", "pkey", "slot", "hint", "hint_date")

    def __init__(self, date, mid, concept, amount, mech="plain", contact=None, city=None, fixed=None, sched=False,
                 trip=False, pkey=None):
        self.date, self.mid, self.concept, self.amount, self.mech = date, mid, concept, round(amount, 2), mech
        self.contact, self.city, self.fixed, self.sched, self.trip, self.pkey = contact, city, fixed, sched, trip, pkey
        self.ref = None
        self.intended = self.filed = None
        self.mistake = False
        self.text = ""
        self.vtok = ""
        self.order = 0.0
        self.slot = 0
        self.hint = None
        self.hint_date = None


class User:
    def __init__(self, uid, arch, rng, is_new):
        self.uid, self.arch, self.rng, self.is_new = uid, arch, rng, is_new
        self.tmpl = arch if arch in TEMPLATES else rng.choice(list(TEMPLATES))
        self.city = rng.choice(list(CITIES))
        st, near = CITIES[self.city][0], CITIES[self.city][1]
        self.st = st
        self.home_cluster = {self.city, *near}
        self.style = wchoice(rng, STYLES)
        self.name_style = "coined" if arch == "coined" else wchoice(rng, [("std", 4), ("mixed", 4), ("personal", 2)])
        self.person = f"{rng.choice(FIRST).upper()} {rng.choice(LAST).upper()}"
        self.card4 = digits(rng, 4)
        self.venmo_names = rng.random() < {"clean": 0.9, "chase": 0.6}.get(self.style, 0.3)
        self.start = D0
        if is_new:
            self.start = D0 + dt.timedelta(days=rng.randint(40, 330))
        self.cats = {}
        self.order = []
        self.base_map = {}
        self.concept_changes = defaultdict(list)
        self.merchant_changes = defaultdict(list)
        self.store_over = {}
        self.kid_cats = []
        self.person_cats = []
        self.person_pin = {}
        self.cat_work = self.cat_date = None
        self.trips = []
        self.away_days = set()
        self.contacts = []
        self.baby_since = None
        self.pets_since = None
        self.check_no = rng.randint(1001, 3400)
        self.p_mistake = rng.choice([0.0, 0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04])
        self.events = []
        self.used_names = set()

    # --- categories
    def new_cat(self, name, concepts, created, kind="base", pos=None):
        cid = len(self.cats)
        assert name not in self.used_names and ", " not in name, name
        self.used_names.add(name)
        self.cats[cid] = Cat(cid, name, concepts, created, kind)
        if pos is None:
            self.order.append(cid)
        else:
            self.order.insert(pos, cid)
        return cid

    def pick_name(self, key):
        cs, std, pers, coin = G[key]
        pw = STYLE_P[self.name_style]
        pools = [std, pers, coin]
        first = wchoice(self.rng, list(zip(range(3), pw)))
        for p in [first] + [0, 1, 2]:
            cand = [n for n in pools[p] if n not in self.used_names]
            if cand:
                return self.rng.choice(cand)
        return None

    def add_group(self, key, created, pos=None, kind="base"):
        name = self.pick_name(key)
        if name is None:
            return None
        return self.new_cat(name, G[key][0].split(), created, kind, pos)

    def cats_at(self, d):
        return [c for c in self.order if self.cats[c].created <= d]

    def resolve(self, concept, d):
        seen, queue = set(), [concept]
        while queue:
            c = queue.pop(0)
            if c in seen:
                continue
            seen.add(c)
            best = None
            for cd, cid in self.concept_changes.get(c, ()):
                if cd <= d:
                    best = (cid, cd)
            if best is not None:
                return best[0], "new_category", best[1]
            if c in self.base_map:
                return self.base_map[c], None, None
            queue.extend(FALLBACK.get(c, []))
        return self.base_map["misc"], None, None

    def away(self, d):
        return d in self.away_days


def build_categories(u):
    rng = u.rng
    keys = []
    for tok in TEMPLATES[u.tmpl].split():
        p = 1.0
        if "?" in tok:
            tok, ps = tok.split("?")
            p = float(ps)
        if rng.random() >= p:
            continue
        keys.extend(rng.choice(tok.split("|")).split("+"))
    u.has_kids = any(k in keys for k in ("KIDS", "kids", "childcare")) or (u.tmpl == "couple" and rng.random() < 0.3)
    u.has_pets = "pets" in keys or rng.random() < 0.1
    u.homeowner = "mortgage" in keys
    u.has_car = not (u.tmpl == "student" and "transport" not in keys) and not (u.tmpl == "roommates" and "fuel" not in keys)
    if u.tmpl == "stores" and "KIDS" not in keys:
        u.has_kids = rng.random() < 0.3
    s = u.start
    for k in keys:
        if k == "KIDS":
            if rng.random() < 0.5:
                u.add_group("kids", s)
            else:
                names = rng.sample(KID_NAMES, 2)
                fmt = rng.choice(["{}", "{} - activities", "Kid {}"])
                for i, nm in enumerate(names):
                    label = fmt.format(i + 1 if fmt == "Kid {}" else nm)
                    cid = u.new_cat(label, {"kids"}, s, "kid")
                    u.kid_cats.append(cid)
                    MEANING[label] |= {"kids"} if fmt == "Kid {}" else set()
            u.has_kids = True
        elif k == "FUN2":
            a, b = rng.sample(PARTNERS, 2)
            fmt = rng.choice(["{} fun money", "{}'s spending", "{} blow money", "{} - personal"])
            for nm in (a, b):
                u.person_cats.append(u.new_cat(fmt.format(nm), {"clothing", "personal_care", "hobbies", "games"}, s, "person"))
        elif k == "STORES":
            opts = rng.sample(["amazon", "costco", "target"], rng.randint(2, 3))
            for o in opts:
                if o == "amazon":
                    nm = rng.choice(["Amazon", "Amazon purchases", "Amazon stuff"])
                    cid = u.new_cat(nm, set(), s, "store")
                    MEANING[nm] |= {"store:Amazon"}
                    u.store_over["Amazon"] = {"from": s, "*": cid}
                elif o == "costco":
                    a = u.new_cat("Costco - food", {"groceries"}, s, "store")
                    b = u.new_cat("Costco - household", {"household"}, s, "store")
                    MEANING["Costco - food"] |= {"store:Costco", "groceries"}
                    MEANING["Costco - household"] |= {"store:Costco", "household"}
                    u.store_over["Costco"] = {"from": s, "*": b, "groceries": a}
                    u.force_costco = True
                else:
                    nm = rng.choice(["Target run", "Target"])
                    cid = u.new_cat(nm, set(), s, "store")
                    MEANING[nm] |= {"store:Target"}
                    u.store_over["Target"] = {"from": s, "*": cid}
        elif k in ("work_lunch", "date_night"):
            cid = u.add_group(k, s)
            if k == "work_lunch":
                u.cat_work = cid
            else:
                u.cat_date = cid
        else:
            if k in G:
                u.add_group(k, s)
    if not any("misc" in c.concepts for c in u.cats.values()):
        u.add_group("misc", s)
    if not any("income" in c.concepts for c in u.cats.values()):
        u.add_group("income", s)
    # "Inflow: Ready to Assign" sits on top in YNAB-like apps
    inc = [c for c in u.order if "income" in u.cats[c].concepts]
    if inc and u.cats[inc[0]].name.startswith(("Inflow", "Ready")):
        u.order.remove(inc[0])
        u.order.insert(0, inc[0])
    # base map: smaller groups win their concepts
    for cid in sorted(u.order, key=lambda c: (len(u.cats[c].concepts) or 99, c)):
        for c in u.cats[cid].concepts:
            u.base_map.setdefault(c, cid)


# ----------------------------------------------------------------------------------------------
# Timeline generation
# ----------------------------------------------------------------------------------------------
WDW_UNIFORM = (1, 1, 1, 1, 1, 1, 1)
WDW_WEEKEND = (0.7, 0.6, 0.7, 0.8, 1.0, 1.8, 1.5)
WDW_WEEKDAY = (1.3, 1.3, 1.3, 1.3, 1.2, 0.35, 0.3)
WDW_EVENING = (0.5, 0.6, 0.7, 0.9, 1.6, 1.9, 0.9)


class Habit:
    __slots__ = ("mid", "rate", "wdw", "start", "end", "scale", "local", "season")

    def __init__(self, mid, rate, wdw=WDW_UNIFORM, start=None, end=None, scale=1.0, local=None, season=None):
        self.mid, self.rate, self.wdw, self.start, self.end, self.scale = mid, rate, wdw, start, end, scale
        self.local = (MER[mid].kind in CARD_KINDS_LOCAL) if local is None else local
        self.season = season


def purpose_pick(u, m, d, rng, amount_scale=1.0):
    """Draw (concept, amount, pkey, mech) for a visit to merchant m on day d."""
    if not m.pset:
        return KIND_CONCEPT[m.kind], logu(rng, m.lo, m.hi) if m.lo != m.hi else m.lo, None, "plain"
    ps = PSETS[m.pset]
    ws = []
    for i, p in enumerate(ps):
        c, w = p[0], p[1]
        jr = srng(u.uid, m.id, c)
        w = w * jr.uniform(0.3, 1.8)
        if c in ("kids",) and not u.has_kids:
            w = 0
        if c == "pets" and not (u.has_pets or (u.pets_since and d >= u.pets_since)):
            w = 0
        if c == "baby":
            w = 25 if (u.baby_since and d >= u.baby_since) else 0
        if c == "gifts" and ((d.month == 11 and d.day >= 20) or (d.month == 12 and d.day <= 23)):
            w *= 6
        ws.append((i, w))
    i = wchoice(rng, ws)
    p = ps[i]
    lo, hi = p[2], p[3]
    amt = logu(rng, lo * amount_scale, hi * amount_scale) if lo != hi else lo
    pkey = i if len(p) > 4 and p[4] else None
    if m.pset in ("gas", "conv"):
        mech = "amount_split"
    elif p[0] == "gifts" and d.month in (11, 12):
        mech = "seasonal"
    else:
        mech = "multi"
    return p[0], amt, pkey, mech


def season_factor(h, d):
    if h.season == "holiday":
        return 1.8 if d.month in (11, 12) else 1.0
    if h.season == "summer":
        return 1.6 if d.month in (6, 7, 8) else 0.8
    if h.season == "yard":
        return 1.0 if 4 <= d.month <= 10 else 0.0
    return 1.0


def gen_poisson(u, h, rng):
    m = MER[h.mid]
    tot = sum(h.wdw)
    lo_d = max(u.start, h.start or D0)
    hi_d = min(D1, h.end or D1)
    out = []
    for d in ALL_DAYS:
        if d < lo_d or d > hi_d:
            continue
        if h.local and u.away(d):
            continue
        p = h.rate / 7.0 * h.wdw[d.weekday()] * 7.0 / tot * season_factor(h, d)
        if rng.random() < min(p, 0.9):
            concept, amt, pkey, mech = purpose_pick(u, m, d, rng, h.scale)
            city = None
            if m.channel == "card" and not m.tail:
                city = m.city or u.city
            e = Ev(d, m.id, concept, amt, mech, city=city, pkey=pkey)
            out.append(e)
    return out


def monthly_dates(u, dom, months=range(1, 13), shift_weekend=True):
    out = []
    for mo in months:
        last = (dt.date(YEAR + (mo == 12), mo % 12 + 1, 1) - dt.timedelta(days=1)).day
        d = dt.date(YEAR, mo, min(dom, last))
        if shift_weekend:
            while d.weekday() >= 5:
                d += dt.timedelta(days=1)
        d = clamp_date(d)
        if d >= u.start:
            out.append(d)
    return out


def sched(u, mid, dates, amount_fn, concept=None, mech="recurring", **kw):
    m = MER[mid]
    c = concept or KIND_CONCEPT[m.kind]
    evs = []
    for i, d in enumerate(dates):
        if d < u.start or d > D1:
            continue
        amt = amount_fn(i, d)
        city = (m.city or u.city) if (m.channel == "card" and not m.tail) else None
        evs.append(Ev(d, mid, c, amt, mech, city=city, sched=True, **kw))
    u.events.extend(evs)


def fixed_amt(a):
    return lambda i, d: a


def var_amt(rng, lo, hi):
    return lambda i, d: logu(rng, lo, hi)


def seasonal_bill(rng, base, kind):
    def f(i, d):
        mo = d.month
        if kind == "electric":
            fac = 1.6 if mo in (6, 7, 8, 9) else (1.2 if mo in (1, 2, 12) else 0.9)
        else:
            fac = 2.4 if mo in (1, 2, 12) else (1.3 if mo in (3, 11) else 0.45)
        return round(base * fac * rng.uniform(0.85, 1.15), 2)
    return f


def pick(rng, seq, k):
    seq = list(seq)
    return rng.sample(seq, min(k, len(seq)))


def add_contact(u, role, service, concept, lo, hi):
    c = {"name": f"{u.rng.choice(FIRST)} {u.rng.choice(LAST)}", "role": role, "service": service, "concept": concept,
         "lo": lo, "hi": hi}
    u.contacts.append(c)
    return c


def p2p_event(u, d, c, amount, concept, mech):
    return Ev(d, c["service"], concept, amount, mech, contact=c)


def build_habits(u):
    rng = u.rng
    t = u.tmpl
    L = LOCAL[u.city]
    H = []
    mult = {"family": 1.3, "detailed": 1.3, "minimal": 0.9, "couple": 1.1, "gig": 0.9, "freelancer": 1.0, "student": 0.6,
            "retiree": 0.9, "stores": 1.2, "roommates": 0.8, "traveler": 1.0}[t]
    # groceries
    g = pick(rng, GROCERY_CHAINS, 2)
    H.append(Habit(g[0], 1.0 * mult, WDW_WEEKEND, scale=1.2 if t == "family" else 1.0))
    H.append(Habit(g[1], 0.3 * mult, WDW_WEEKEND))
    if rng.random() < 0.45:
        H.append(Habit(rng.choice(L["grocery"]), 0.3, WDW_WEEKEND))
    if rng.random() < 0.15:
        H.append(Habit("Instacart", 0.35))
    if rng.random() < 0.08:
        sched(u, "HelloFresh", [d for d in ALL_DAYS if d.weekday() == 2 and rng.random() < 0.8], var_amt(rng, 59.99, 89.99))
    # big box: the multi-purpose core
    bb = pick(rng, ["Target", "Walmart", "Costco", "Sam's Club", "BJ's Wholesale", "Meijer", "Fred Meyer"],
              rng.choice((1, 2, 2, 3)))
    if getattr(u, "force_costco", False) and "Costco" not in bb:
        bb.append("Costco")
    if "Target" in u.store_over and "Target" not in bb:
        bb.append("Target")
    for mid in bb:
        H.append(Habit(mid, rng.uniform(0.3, 0.9) * mult, WDW_WEEKEND))
    if rng.random() < 0.4:
        H.append(Habit(rng.choice(["Dollar Tree", "Dollar General"]), 0.2))
    amz = {"family": 2.2, "stores": 2.5, "student": 0.4, "retiree": 0.7}.get(t, 1.2) * rng.uniform(0.3, 1.5)
    if "Amazon" in u.store_over:
        amz = max(amz, 1.0)
    H.append(Habit("Amazon", amz, season="holiday", local=False))
    for mid in pick(rng, ["Etsy", "eBay", "Temu", "PayPal", "Wayfair", "Shein"], rng.randint(0, 3)):
        H.append(Habit(mid, rng.uniform(0.05, 0.3), local=False))
    # dining
    rests = pick(rng, L["restaurant"], rng.randint(2, 6)) + pick(rng, RESTAURANT_CHAINS, rng.randint(0, 3))
    for mid in rests:
        H.append(Habit(mid, rng.uniform(0.08, 0.5) * mult, WDW_EVENING))
    for mid in pick(rng, FASTFOOD_CHAINS, rng.randint(1, 5)):
        H.append(Habit(mid, rng.uniform(0.1, 0.7), WDW_WEEKDAY))
    if rng.random() < 0.5:
        H.append(Habit(rng.choice(L["bakery"]), rng.uniform(0.05, 0.3), WDW_WEEKEND))
    if rng.random() < 0.4:
        H.append(Habit(rng.choice(L["bar"]), rng.uniform(0.1, 0.6), WDW_EVENING))
    if rng.random() < 0.3:
        H.append(Habit(rng.choice(L["liquor"] + ["Total Wine"]), 0.2, WDW_EVENING))
    if rng.random() < 0.3:
        H.append(Habit(L["restaurant"][-1], 0.15, WDW_WEEKDAY))   # the opaque food truck (last local restaurant)
    # coffee
    if rng.random() < 0.7:
        for mid in pick(rng, COFFEE_CHAINS, rng.randint(0, 2)) + pick(rng, L["coffee"], rng.randint(0, 2)):
            H.append(Habit(mid, rng.uniform(0.3, 3.5), WDW_WEEKDAY))
    # delivery
    if rng.random() < 0.55:
        H.append(Habit(rng.choice(["DoorDash", "Grubhub"]), rng.uniform(0.1, 1.2), WDW_EVENING, local=False))
    # rideshare / Uber is multi-purpose (trips and Eats)
    if rng.random() < 0.5 or t in ("student", "roommates"):
        H.append(Habit("Uber", rng.uniform(0.1, 1.2), WDW_EVENING, local=False))
    if rng.random() < 0.25:
        H.append(Habit("Lyft", rng.uniform(0.1, 0.6), WDW_EVENING, local=False))
    # car
    if u.has_car:
        for mid in pick(rng, GAS_CHAINS, rng.randint(1, 3)):
            H.append(Habit(mid, rng.uniform(0.3, 1.1) * (1.8 if t == "gig" else 1.0)))
        if "Costco" in bb and rng.random() < 0.6:
            H.append(Habit("Costco Gas", 0.35))
        H.append(Habit(rng.choice(BY_KIND["auto"] + L["auto"]), 0.05))
        if rng.random() < 0.3:
            sched(u, "Mister Car Wash", monthly_dates(u, rng.randint(2, 26)), fixed_amt(rng.choice([24.99, 29.99, 34.99])))
        if rng.random() < 0.5:
            H.append(Habit(rng.choice(["ParkMobile", "PayByPhone", "SpotHero", "LAZ Parking"]), rng.uniform(0.05, 0.5), local=False))
    # transit
    transit = [m for m in L["transit"]]
    if transit and (not u.has_car or rng.random() < 0.25):
        tm = transit[0]
        if rng.random() < 0.5:
            sched(u, tm, monthly_dates(u, 1), fixed_amt(rng.choice([50.0, 64.0, 86.0, 100.0])))
        else:
            H.append(Habit(tm, rng.uniform(1.0, 5.0), WDW_WEEKDAY))
        if rng.random() < 0.3:
            H.append(Habit("Lime", 0.4, local=False))
    # health and care
    H.append(Habit(rng.choice(["CVS", "Walgreens", "Rite Aid"]), rng.uniform(0.15, 0.6) * (1.6 if t == "retiree" else 1)))
    for mid in pick(rng, L["medical"] + L["dental"] + ["Quest Diagnostics", "LabCorp", "One Medical", "MinuteClinic",
                                                       "Aspen Dental", "Warby Parker"], rng.randint(1, 3)):
        H.append(Habit(mid, rng.uniform(0.02, 0.1) * (2 if t == "retiree" else 1), WDW_WEEKDAY, local=False))
    if rng.random() < 0.15:
        sched(u, rng.choice(["Headway", "BetterHelp"]), [d for d in ALL_DAYS if d.weekday() == 1], var_amt(rng, 25, 90))
    H.append(Habit(rng.choice(L["salon"] + ["Great Clips", "Supercuts"]), rng.uniform(0.2, 0.35)))
    if rng.random() < 0.35:
        H.append(Habit(rng.choice(["Sephora", "Ulta", "Bath & Body Works", "European Wax Center", "Massage Envy"]), 0.15))
    # home
    if u.homeowner or rng.random() < 0.3:
        H.append(Habit(rng.choice(["The Home Depot", "Lowe's"]), rng.uniform(0.15, 0.6) * (1.3 if u.homeowner else 0.5)))
        if rng.random() < 0.4:
            H.append(Habit(rng.choice(L["home_improve"] + ["Ace Hardware"]), 0.15))
    for mid in pick(rng, ["IKEA", "HomeGoods", "Container Store"], rng.randint(0, 2)):
        H.append(Habit(mid, 0.06))
    if rng.random() < 0.25:
        H.append(Habit(rng.choice(L["laundry"]), 0.4))
    # clothing & stuff
    for mid in pick(rng, BY_KIND["clothing"] + BY_KIND["dept"], rng.randint(1, 4)):
        H.append(Habit(mid, rng.uniform(0.04, 0.2), season="holiday"))
    if rng.random() < 0.5:
        H.append(Habit(rng.choice(["Best Buy", "Apple Store", "Micro Center"]), 0.04))
    # fun
    for mid in pick(rng, BY_KIND["entertainment"] + L["entertainment"], rng.randint(1, 3)):
        H.append(Habit(mid, rng.uniform(0.04, 0.25), WDW_EVENING, local=MER[mid].tail is None))
    for mid in pick(rng, BY_KIND["hobby"] + BY_KIND["books"][:2] + L["books"], rng.randint(0, 3)):
        H.append(Habit(mid, rng.uniform(0.04, 0.2)))
    if rng.random() < 0.35:
        H.append(Habit(rng.choice(["Steam", "PlayStation", "Nintendo", "GameStop"]), 0.12, local=False))
    for mid in pick(rng, ["Apple", "Google", "Microsoft", "Kindle", "Paddle", "FastSpring"], rng.randint(0, 2)):
        H.append(Habit(mid, rng.uniform(0.1, 0.5), local=False))
    # kids
    if u.has_kids:
        for mid in pick(rng, L["kids_activity"], 2):
            sched(u, mid, monthly_dates(u, rng.randint(1, 10)), fixed_amt(round(rng.uniform(60, 190), 0)))
        H.append(Habit(rng.choice(["LEGO Store", "Chuck E. Cheese", "Carter's", "Old Navy"]), 0.12))
        if rng.random() < 0.6:
            sched(u, "MySchoolBucks", [d for d in monthly_dates(u, 5) if d.month not in (6, 7)], var_amt(rng, 20, 60))
        if rng.random() < 0.3:
            sched(u, "Kumon", monthly_dates(u, 3), fixed_amt(165.0))
    # pets
    if u.has_pets:
        pet_habits(u, rng, H, u.start)
    # misc
    if rng.random() < 0.6:
        H.append(Habit("ATM", rng.uniform(0.1, 0.6), local=False))
    if rng.random() < 0.3:
        H.append(Habit("USPS", 0.08))
    if rng.random() < 0.3:
        H.append(Habit(rng.choice(["American Red Cross", "St. Jude", "Wikimedia", "GoFundMe"]), 0.05, local=False))
    if rng.random() < 0.35:
        H.append(Habit(rng.choice(L["florist"] + ["1-800-Flowers", "Edible Arrangements", "Hallmark"]), 0.05))
    # business spending for self-employed
    if t in ("freelancer", "gig"):
        for mid in pick(rng, ["Canva", "Zoom", "Squarespace", "QuickBooks", "Adobe"], rng.randint(1, 3)):
            m = MER[mid]
            sched(u, mid, monthly_dates(u, rng.randint(1, 28), shift_weekend=False), fixed_amt(round(rng.uniform(m.lo, m.hi), 2)))
        H.append(Habit(rng.choice(["Staples", "FedEx Office", "Upwork"]), 0.15))
    # one-off / exploration pool: produces first-time payees all year long
    pool = []
    for k in ("restaurant", "coffee", "bakery", "bar", "entertainment", "salon", "auto", "home_improve", "books", "florist",
              "gym", "medical", "kids_activity" if u.has_kids else "books"):
        pool += L[k]
    pool += pick(rng, RESTAURANT_CHAINS + FASTFOOD_CHAINS + BY_KIND["hobby"] + BY_KIND["clothing"] + BY_KIND["entertainment"]
                 + BY_KIND["furniture"] + BY_KIND["electronics"] + BY_KIND["beauty"], 25)
    pool += [m for kk in LOCAL[u.city].values() for m in kk if MER[m].opaque]
    pool += pick(rng, ["PayPal", "Eventbrite", "GoFundMe", "Paddle", "FastSpring"], 2)
    rng.shuffle(pool)
    have = {h.mid for h in H}
    u.explore_pool = [m for m in pool if m not in have]
    u.explore_rate = rng.uniform(0.3, 0.9)
    return H


def pet_habits(u, rng, H, start):
    H.append(Habit(rng.choice(["Petco", "PetSmart"]), 0.2, start=start))
    if rng.random() < 0.5:
        m = "Chewy"
        sched(u, m, [d for d in monthly_dates(u, rng.randint(1, 28)) if d >= start], var_amt(rng, 38, 72))
    vet = rng.choice(LOCAL[u.city]["vet"] + ["Banfield"])
    H.append(Habit(vet, 0.05, start=start, local=False))
    if rng.random() < 0.3:
        c = add_contact(u, "dogwalker", "Venmo", "pets", 20, 45)
        u.events.extend(p2p_event(u, d, c, logu(rng, 20, 45), "pets", "p2p") for d in ALL_DAYS
                        if d >= max(start, u.start) and d.weekday() in (1, 3) and rng.random() < 0.7)


def build_scheduled(u):
    rng = u.rng
    t = u.tmpl
    city = CITIES[u.city]
    # housing
    if u.homeowner:
        m = rng.choice(["Rocket Mortgage", "Mr. Cooper", "PennyMac"])
        sched(u, m, monthly_dates(u, 1), fixed_amt(round(rng.uniform(1100, 3300), 2)))
        if rng.random() < 0.3:
            sched(u, "CINC HOA", monthly_dates(u, 1, months=(1, 4, 7, 10)), fixed_amt(round(rng.uniform(150, 450), 2)))
        sched(u, f"{city[6]} County Treasurer", [day(11, 20)], fixed_amt(round(rng.uniform(900, 4200), 2)))
    else:
        rent = round(rng.uniform(900, 2600) / (2 if t in ("roommates", "student") else 1), 0)
        how = rng.random()
        if how < 0.3:
            c = add_contact(u, "landlord", rng.choice(["Zelle", "Check"]), "rent", rent, rent)
            for d in monthly_dates(u, 1, shift_weekend=False):
                u.events.append(p2p_event(u, d, c, rent, "rent", "recurring"))
                u.events[-1].sched = True
        elif how < 0.5 and t in ("roommates", "student"):
            c = add_contact(u, "roommate", rng.choice(["Venmo", "Zelle"]), "rent", rent, rent)
            for d in monthly_dates(u, 28, shift_weekend=False):
                u.events.append(p2p_event(u, d, c, rent, "rent", "recurring"))
                u.events[-1].sched = True
        else:
            sched(u, rng.choice(["Bilt Rent", "AppFolio", "RentCafe"]), monthly_dates(u, 1), fixed_amt(rent))
        if rng.random() < 0.5:
            sched(u, "Lemonade", monthly_dates(u, rng.randint(2, 27)), fixed_amt(round(rng.uniform(9, 28), 2)))
    # utilities
    el = LOCAL[u.city]["electric"][0]
    base = rng.uniform(60, 160)
    sched(u, el, monthly_dates(u, rng.randint(5, 25)), seasonal_bill(rng, base, "electric"))
    if LOCAL[u.city]["gas_util"]:
        sched(u, LOCAL[u.city]["gas_util"][0], monthly_dates(u, rng.randint(5, 25)), seasonal_bill(rng, rng.uniform(30, 80), "gas"))
    if u.homeowner or rng.random() < 0.4:
        sched(u, LOCAL[u.city]["water"][0], monthly_dates(u, rng.randint(5, 25), months=range(2, 13, 2)), var_amt(rng, 60, 160))
    # roommates reimburse their share of utilities (negative, filed like the bill)
    if t in ("roommates", "student") or (t != "couple" and rng.random() < 0.08):
        rm = add_contact(u, "roommate_util", "Venmo", "electric", 20, 90)
        for d in monthly_dates(u, rng.randint(8, 28), shift_weekend=False):
            u.events.append(p2p_event(u, d, rm, -logu(rng, 20, 90), "electric", "reimbursement"))
    # telecom (AT&T and Verizon are one payee for phone and internet: same payee, two categories)
    tele = rng.choice(["Verizon", "AT&T", "T-Mobile", "Mint Mobile", "Visible"])
    ph = round(rng.uniform(35, 190), 2) if tele not in ("Mint Mobile", "Visible") else round(rng.uniform(15, 45), 2)
    sched(u, tele, monthly_dates(u, rng.randint(1, 28), shift_weekend=False), fixed_amt(ph), concept="phone")
    if tele in ("Verizon", "AT&T") and rng.random() < 0.5:
        net = tele
    else:
        net = rng.choice(["Xfinity", "Spectrum", "Google Fiber", "Starlink", "Xfinity", "Spectrum"])
    promo_end = rng.randint(3, 11)
    netamt = round(rng.uniform(50, 85), 2)
    sched(u, net, monthly_dates(u, rng.randint(1, 28), shift_weekend=False),
          lambda i, d, a=netamt, pe=promo_end: a if d.month < pe else round(a + 25, 2), concept="internet")
    # insurance
    if u.has_car:
        ins = rng.choice(["GEICO", "Progressive", "State Farm", "Allstate", "USAA"])
        if rng.random() < 0.6:
            sched(u, ins, monthly_dates(u, rng.randint(1, 28)), fixed_amt(round(rng.uniform(80, 220), 2)))
        else:
            m0 = rng.randint(1, 6)
            sched(u, ins, [day(m0, 15), day(m0 + 6, 15)], fixed_amt(round(rng.uniform(480, 1100), 2)))
        if u.homeowner and ins in ("State Farm", "USAA", "Allstate"):
            sched(u, ins, [day(rng.randint(1, 12), 20)], fixed_amt(round(rng.uniform(900, 2200), 2)), concept="home_ins")
        if rng.random() < 0.45:
            sched(u, rng.choice(["Toyota Financial", "American Honda Finance", "Ally Auto"]), monthly_dates(u, rng.randint(1, 28)),
                  fixed_amt(round(rng.uniform(280, 620), 2)))
    if rng.random() < 0.2:
        sched(u, rng.choice(["Haven Life", "Ladder Life"]), monthly_dates(u, rng.randint(1, 28)), fixed_amt(round(rng.uniform(20, 65), 2)))
    if t == "freelancer" or (t == "gig" and rng.random() < 0.5):
        sched(u, rng.choice(["BCBS", "Oscar Health"]), monthly_dates(u, 1), fixed_amt(round(rng.uniform(320, 720), 2)))
    if t == "retiree":
        sched(u, "AARP Medicare Supplement", monthly_dates(u, 1), fixed_amt(round(rng.uniform(150, 260), 2)))
    if t in ("student", "minimal", "couple", "freelancer") and rng.random() < 0.4:
        sched(u, rng.choice(["Nelnet", "MOHELA", "Sallie Mae"]), monthly_dates(u, rng.randint(1, 28)),
              fixed_amt(round(rng.uniform(120, 460), 2)))
    # subscriptions (with a mid-year price change on some)
    n_sub = {"student": 2, "retiree": 1}.get(t, 3)
    subs = pick(rng, ["Netflix", "Hulu", "Disney+", "Max", "Spotify", "Peacock", "Paramount+", "SiriusXM", "Crunchyroll",
                      "Dropbox", "OpenAI", "NYTimes", "Patreon", "Audible", "Planet Fitness", "Peloton", "ClassPass",
                      "YMCA", "Orangetheory", "BarkBox" if u.has_pets else "Notion"], rng.randint(max(0, n_sub - 2), n_sub + 3))
    for s in subs:
        m = MER[s]
        p0 = round(rng.uniform(m.lo, m.hi), 2) if m.lo != m.hi else m.lo
        hike = rng.randint(2, 12) if rng.random() < 0.3 else 13
        dom = rng.randint(1, 28)
        sched(u, s, monthly_dates(u, dom, shift_weekend=False),
              lambda i, d, p=p0, h=hike: p if d.month < h else round(p * 1.15, 2))
    if rng.random() < 0.4:
        yearly = rng.choice(["Duolingo", "1Password"])
        sched(u, yearly, [day(rng.randint(1, 12), rng.randint(1, 28))], fixed_amt(MER[yearly].lo))
    if rng.random() < 0.5:
        # Amazon Prime renewal is an Amazon charge too (the multi-purpose payee gets a fixed yearly amount)
        sched(u, "Amazon", [day(rng.randint(1, 12), rng.randint(1, 28))], fixed_amt(139.0), concept="streaming",
              mech="recurring")
    # giving
    if any(c in ("church",) for cat in u.cats.values() for c in cat.concepts) and rng.random() < 0.85:
        ch = rng.choice(LOCAL[u.city]["church"])
        if rng.random() < 0.5:
            sched(u, ch, [d for d in ALL_DAYS if d.weekday() == 6], var_amt(rng, 20, 120))
        else:
            c = add_contact(u, "church", "Check", "church", 100, 400)
            u.events.extend(p2p_event(u, d, c, round(rng.uniform(100, 400), 0), "church", "recurring")
                            for d in monthly_dates(u, 3))
    # money movement
    if rng.random() < 0.7:
        dest = rng.choice(["Savings transfer", "Savings transfer", "Ally Bank", "Marcus", "Robinhood", "Vanguard",
                           "Fidelity", "Acorns"])
        amt = round(rng.choice([50, 100, 150, 200, 250, 300, 500, 750]), 2)
        sched(u, dest, monthly_dates(u, rng.choice([1, 2, 15, 16])), fixed_amt(amt), mech="transfer")
        if dest == "Savings transfer" and rng.random() < 0.5:
            for _ in range(rng.randint(1, 3)):
                d = ALL_DAYS[rng.randint(60, 350)]
                if d >= u.start:
                    u.events.append(Ev(d, dest, "savings", -round(rng.uniform(200, 2500), 0), "transfer", sched=True))
    if any("cc_payment" in c.concepts for c in u.cats.values()):
        sched(u, rng.choice(["Chase Card", "Citi Card", "Amex", "Discover"]), monthly_dates(u, rng.randint(1, 28)),
              var_amt(rng, 300, 2800), mech="transfer")
    if rng.random() < 0.08:
        sched(u, "Monthly Service Fee", monthly_dates(u, 28, shift_weekend=False), fixed_amt(12.0))
    if rng.random() < 0.1:
        for _ in range(rng.randint(1, 3)):
            u.events.append(Ev(ALL_DAYS[rng.randint(0, 364)], "Overdraft Fee", "fees", 35.0))
    # taxes
    if t in ("freelancer", "gig"):
        sched(u, "IRS Payment", [day(4, 15), day(6, 16), day(9, 15)], var_amt(rng, 400, 3500))
    build_income(u, rng)
    build_p2p(u, rng)


def build_income(u, rng):
    t = u.tmpl

    def ach_income(mid, dates, lo, hi, concept="income", jitter=0.03):
        base = rng.uniform(lo, hi)
        for d in dates:
            if u.start <= d <= D1:
                u.events.append(Ev(d, mid, concept, -round(base * rng.uniform(1 - jitter, 1 + jitter), 2), "income", sched=True))

    fridays = [d for d in ALL_DAYS if d.weekday() == 4]
    if t in ("family", "detailed", "minimal", "couple", "stores", "roommates", "traveler", "freelancer") or (t == "gig" and rng.random() < 0.4):
        emp = rng.choice([m for m in MER if MER[m].kind == "payroll"])
        if t == "freelancer":
            emp = None
        if emp:
            if rng.random() < 0.6:
                off = rng.randint(0, 1)
                dates = fridays[off::2]
            else:
                dates = sorted(set(monthly_dates(u, 15, shift_weekend=False) + monthly_dates(u, 31, shift_weekend=False)))
                dates = [d - dt.timedelta(days=max(0, d.weekday() - 4)) for d in dates]
            ach_income(emp, dates, 1300, 4200, jitter=0.02)
            if rng.random() < 0.25:
                u.events.append(Ev(day(3, 14), emp, "income", -round(rng.uniform(1500, 9000), 2), "income", sched=True))
        if t == "couple" and rng.random() < 0.7:
            emp2 = rng.choice([m for m in MER if MER[m].kind == "payroll" and m != emp])
            ach_income(emp2, fridays[rng.randint(0, 1)::2], 1100, 3600, jitter=0.02)
    if t == "gig":
        for mid in pick(rng, ["Uber Driver Payout", "Lyft Driver Payout", "DoorDash Dasher Pay", "Instacart Shopper Pay"], rng.randint(1, 2)):
            dates = [d for d in ALL_DAYS if d.weekday() in (0, 1) and rng.random() < 0.45]
            gc = "gig_income"
            for d in dates:
                if d >= u.start:
                    u.events.append(Ev(d, mid, gc, -logu(rng, 90, 950), "income", sched=True))
    if t == "freelancer":
        for mid in pick(rng, ["Stripe", "Upwork Earnings", "Shopify Payout", "Etsy Payout", "PayPal Transfer"], rng.randint(1, 2)):
            for d in ALL_DAYS:
                if d >= u.start and d.weekday() < 5 and rng.random() < 0.12:
                    u.events.append(Ev(d, mid, "gig_income", -logu(rng, 120, 3200), "income", sched=True))
        for cl in pick(rng, [m for m in MER if MER[m].kind == "gig_pay" and MER[m].opaque], rng.randint(1, 3)):
            for d in ALL_DAYS:
                if d >= u.start and d.weekday() < 5 and rng.random() < 0.035:
                    u.events.append(Ev(d, cl, "gig_income", -logu(rng, 800, 6500), "income", sched=True))
        if rng.random() < 0.5:
            c = add_contact(u, "client", "Zelle", "gig_income", 300, 2000)
            for d in ALL_DAYS:
                if d >= u.start and rng.random() < 0.02:
                    u.events.append(p2p_event(u, d, c, -logu(rng, 300, 2000), "gig_income", "income"))
    if t == "retiree":
        wed = [d for d in ALL_DAYS if d.weekday() == 2 and 15 <= d.day <= 21]
        ach_income("Social Security", wed, 1500, 3200, jitter=0.0)
        if rng.random() < 0.6:
            ach_income("Teachers Retirement System", monthly_dates(u, 1), 900, 2600, jitter=0.0)
    if t == "student":
        c = add_contact(u, "parent", rng.choice(["Zelle", "Venmo"]), "income", 200, 800)
        for d in monthly_dates(u, rng.randint(1, 5), shift_weekend=False):
            u.events.append(p2p_event(u, d, c, -round(rng.uniform(200, 800), 0), "income", "income"))
        if rng.random() < 0.5:
            emp = rng.choice(["Gusto Payroll", "Adp Payroll", "Paychex Inc Payroll"])
            ach_income(emp, fridays[rng.randint(0, 1)::2], 280, 900, jitter=0.25)
    if t == "couple" and rng.random() < 0.3:
        c = add_contact(u, "partner", "Zelle", "income", 500, 1500)
        for d in monthly_dates(u, 2, shift_weekend=False):
            u.events.append(p2p_event(u, d, c, -round(rng.uniform(500, 1500), 0), "income", "income"))
    if rng.random() < 0.55:
        u.events.append(Ev(ALL_DAYS[rng.randint(40, 120)], "IRS Tax Refund", "income", -round(rng.uniform(250, 4200), 2), "income", sched=True))
        if rng.random() < 0.5:
            u.events.append(Ev(ALL_DAYS[rng.randint(50, 150)], "State Tax Refund", "income", -round(rng.uniform(60, 900), 2), "income", sched=True))
    if rng.random() < 0.5:
        ach_income("Interest", monthly_dates(u, 31, shift_weekend=False), 0.4, 45, jitter=0.4)
    if rng.random() < 0.08:
        for d in ALL_DAYS:
            if d >= u.start and rng.random() < 0.03:
                u.events.append(Ev(d, "Airbnb Host Payout", "gig_income", -logu(rng, 180, 900), "income", sched=True))


def build_p2p(u, rng):
    t = u.tmpl
    # friends: splitting dinners, concert tickets, birthday gifts; incoming paybacks too
    for _ in range(rng.randint(1, 3) if t != "retiree" else rng.randint(0, 1)):
        svc = rng.choice(["Venmo", "Venmo", "Venmo", "Zelle", "Cash App"])
        c = add_contact(u, "friend", svc, "dining", 12, 70)
        rate = rng.uniform(0.05, 0.35)
        for d in ALL_DAYS:
            if d >= u.start and rng.random() < rate / 7:
                r = rng.random()
                if r < 0.55:
                    u.events.append(p2p_event(u, d, c, logu(rng, 12, 70), "dining", "p2p"))
                elif r < 0.75:
                    u.events.append(p2p_event(u, d, c, logu(rng, 30, 220), "entertainment", "p2p"))
                elif r < 0.85:
                    u.events.append(p2p_event(u, d, c, logu(rng, 20, 80), "gifts", "p2p"))
                else:
                    u.events.append(p2p_event(u, d, c, -logu(rng, 12, 70), "dining", "reimbursement"))
    if u.has_kids and rng.random() < 0.6:
        c = add_contact(u, "babysitter", rng.choice(["Venmo", "Cash App", "Zelle"]), "childcare", 40, 140)
        for d in ALL_DAYS:
            if d >= u.start and d.weekday() in (4, 5) and rng.random() < 0.3:
                u.events.append(p2p_event(u, d, c, round(rng.uniform(40, 140), 0), "childcare", "p2p"))
    if u.has_kids and rng.random() < 0.3:
        c = add_contact(u, "coach", rng.choice(["Venmo", "Check"]), "kids", 60, 180)
        for d in monthly_dates(u, rng.randint(1, 20)):
            u.events.append(p2p_event(u, d, c, round(rng.uniform(60, 180), 0), "kids", "p2p"))
    if t in ("family", "detailed", "couple", "retiree", "stores") and rng.random() < 0.4:
        c = add_contact(u, "cleaner", rng.choice(["Zelle", "Venmo"]), "household", 90, 180)
        amt = round(rng.uniform(90, 180), 0)
        for d in [x for x in ALL_DAYS if x.weekday() == rng.randint(0, 4)][::2]:
            if d >= u.start:
                u.events.append(p2p_event(u, d, c, amt, "household", "p2p"))
    if u.homeowner and rng.random() < 0.35:
        c = add_contact(u, "lawn", rng.choice(["Cash App", "Check", "Zelle"]), "home_improve", 40, 70)
        for d in ALL_DAYS:
            if d >= u.start and 4 <= d.month <= 10 and d.weekday() == 5 and rng.random() < 0.5:
                u.events.append(p2p_event(u, d, c, round(rng.uniform(40, 70), 0), "home_improve", "p2p"))
    if t in ("family", "retiree", "couple") and rng.random() < 0.4:
        c = add_contact(u, "family", rng.choice(["Zelle", "Venmo"]), "gifts", 25, 250)
        for d in ALL_DAYS:
            if d >= u.start and rng.random() < (0.05 if d.month == 12 else 0.008):
                u.events.append(p2p_event(u, d, c, logu(rng, 25, 250), "gifts", "p2p"))
    if rng.random() < 0.2:
        c = add_contact(u, "hairdresser", "Venmo", "personal_care", 45, 120)
        for d in monthly_dates(u, rng.randint(1, 28)):
            u.events.append(p2p_event(u, d, c, round(rng.uniform(45, 120), 0), "personal_care", "p2p"))


def build_trips(u):
    rng = u.rng
    n = {"traveler": rng.randint(3, 5), "family": rng.randint(1, 2), "detailed": rng.randint(1, 3), "retiree": rng.randint(1, 3),
         "couple": rng.randint(1, 2)}.get(u.tmpl, rng.randint(0, 2))
    style = wchoice(rng, [("dedicated", 55), ("travel", 20), ("normal", 25)])
    has_travel = [c for c in u.order if "travel" in u.cats[c].concepts and u.cats[c].kind == "base"]
    if style == "travel" and not has_travel:
        style = "dedicated"
    u.trip_style = style
    taken = set()
    for _ in range(n):
        dest = rng.choice([d for d in DESTS])
        length = rng.randint(3, 9)
        for _try in range(30):
            start = ALL_DAYS[rng.randint(35, 355 - length)]
            span = {start + dt.timedelta(days=i) for i in range(-3, length + 3)}
            if not (span & taken):
                break
        else:
            continue
        book = start - dt.timedelta(days=rng.randint(10, 75))
        if book < u.start + dt.timedelta(days=3):
            continue
        taken |= span
        end = start + dt.timedelta(days=length - 1)
        city, st, token = dest
        fixed = None
        if style == "dedicated" and len(u.cats) < MAX_CATS:
            pat = rng.choice(["{t} trip", "{t} {y}", "{t} vacation", "Trip: {t}", "{t} getaway"])
            if rng.random() < 0.15:
                pat = rng.choice(["Summer vacation", "Spring break", "Fall trip", "Family reunion", "Jen's wedding"])
            name = pat.format(t=token, y=YEAR)
            if name in u.used_names:
                name = f"{name} ({city})"
            if name in u.used_names:
                continue
            fixed = u.new_cat(name, {"travel"}, book, "trip", pos=rng.randint(0, len(u.order)))
            MEANING[name] |= {"trip", "travel"} | ({"dest:" + city} if token in name or city in name else set())
        elif style == "travel":
            fixed = has_travel[0]
        u.trips.append((book, start, end, dest, fixed))
        for i in range(length):
            u.away_days.add(start + dt.timedelta(days=i))
        mech = "trip" if fixed is not None else "plain"
        tr = lambda d, mid, c, a, cty=city, **kw: u.events.append(Ev(d, mid, c, a, mech, city=cty, fixed=fixed, trip=True, **kw))
        airline = rng.choice(BY_KIND["airline"])
        tr(book, airline, "travel", logu(rng, 180, 1400), cty=None)
        lodging = rng.choice(["Airbnb", "Vrbo", "Expedia", "Marriott", "Hilton", "Hyatt", "Holiday Inn Express"])
        if lodging in ("Airbnb", "Vrbo", "Expedia"):
            tr(book + dt.timedelta(days=rng.randint(0, 5)), lodging, "travel", logu(rng, 300, 2400), cty=None)
        else:
            tr(end, lodging, "travel", logu(rng, 150 * length / 2, 260 * length), cty=city)
        if rng.random() < 0.4:
            tr(start, airline, "travel", rng.choice([35.0, 40.0, 70.0]), cty=None)
        car = rng.random() < 0.35
        if car:
            tr(end, rng.choice(["Enterprise", "Hertz"]), "travel", logu(rng, 150, 650))
        dl = LOCAL[city]
        foreign = st == "MX"
        for i in range(length):
            d = start + dt.timedelta(days=i)
            for _k in range(rng.randint(1, 3)):
                mid = rng.choice(dl["restaurant"] + dl["bar"] + pick(rng, FASTFOOD_CHAINS + RESTAURANT_CHAINS, 3))
                c, a, pk, _m = purpose_pick(u, MER[mid], d, rng)
                tr(d, mid, c, a)
                if foreign and rng.random() < 0.6:
                    tr(d, "Foreign Transaction Fee", "fees", round(a * 0.03, 2), cty=None)
            if rng.random() < 0.6:
                mid = rng.choice(dl["coffee"] + ["Starbucks"])
                tr(d, mid, "coffee", logu(rng, 4, 16))
            if rng.random() < 0.45:
                tr(d, "Uber", "rideshare", logu(rng, 9, 60), cty=None, pkey=0)
            if rng.random() < 0.35:
                tr(d, rng.choice(dl["entertainment"]), "entertainment", logu(rng, 20, 240))
            if car and rng.random() < 0.4:
                tr(d, rng.choice(["Shell", "Chevron", "7-Eleven"]), "fuel", logu(rng, 25, 70))
            if rng.random() < 0.15:
                tr(d, rng.choice(["CVS", "Walgreens", "Target"]), "household", logu(rng, 6, 45))


LIFE_EVENTS = {
    "puppy": ("pets", ["New puppy", "Puppy fund", "Dog"], ["Petco", "PetSmart", "Chewy", "Banfield", "Rover"]),
    "baby": ("baby", ["Baby", "Baby stuff", "Nursery"], ["buybuy BABY", "Carter's", "Target", "Amazon"]),
    "remodel": ("home_improve", ["Kitchen remodel", "Bathroom reno", "House projects 2025"], ["The Home Depot", "Lowe's", "Sherwin-Williams", "U-Haul"]),
    "wedding": ("wedding", ["Wedding", "Wedding 2025", "Our wedding"], ["David's Bridal", "Zola", "Blue Nile", "1-800-Flowers"]),
    "running": ("running", ["Marathon training", "Running", "Race fees"], ["Fleet Feet", "RunSignup", "Strava", "REI"]),
}


def build_life_event(u, H):
    rng = u.rng
    if u.is_new or rng.random() > 0.35:
        return
    opts = ["running", "remodel" if u.homeowner else "wedding"]
    if not u.has_pets:
        opts.append("puppy")
    if u.tmpl in ("family", "detailed", "couple", "stores"):
        opts.append("baby")
    ev = rng.choice(opts)
    if len(u.cats) >= MAX_CATS:
        return
    concept, names, mids = LIFE_EVENTS[ev]
    d = ALL_DAYS[rng.randint(120, 300)]
    name = next((n for n in rng.sample(names, len(names)) if n not in u.used_names), None)
    if name is None:
        return
    cid = u.new_cat(name, {concept}, d, "life", pos=rng.randint(0, len(u.order)))
    MEANING[name] |= {concept} if ev != "puppy" else {"pets"}
    u.concept_changes[concept].append((d, cid))
    u.life = (ev, d, cid)
    if ev == "puppy":
        u.pets_since = d
        pet_habits(u, rng, H, d)
        H.append(Habit("Amazon", 0.3, start=d, local=False))
    elif ev == "baby":
        u.baby_since = d
        u.has_kids = True
        H.append(Habit("buybuy BABY", 0.3, start=d))
        H.append(Habit("Carter's", 0.15, start=d))
    elif ev == "remodel":
        H.append(Habit(rng.choice(["The Home Depot", "Lowe's"]), 1.2, start=d, end=d + dt.timedelta(days=90)))
        H.append(Habit("Sherwin-Williams", 0.3, start=d, end=d + dt.timedelta(days=90)))
        contractor = rng.choice(LOCAL[u.city]["home_improve"][-1:])
        for k in range(3):
            u.events.append(Ev(d + dt.timedelta(days=7 + 25 * k), contractor, "home_improve", logu(rng, 1500, 8000), "plain",
                               city=u.city))
    elif ev == "wedding":
        H.append(Habit(rng.choice(["Zola", "1-800-Flowers"]), 0.3, start=d, local=False))
        u.events.append(Ev(d + dt.timedelta(days=5), "David's Bridal", "wedding", logu(rng, 300, 1200), city=u.city))
        u.events.append(Ev(d + dt.timedelta(days=20), "Blue Nile", "wedding", logu(rng, 900, 4000)))
        c = add_contact(u, "photographer", "Zelle", "wedding", 800, 3000)
        u.events.append(p2p_event(u, d + dt.timedelta(days=40), c, 1500.0, "wedding", "p2p"))
    else:
        H.append(Habit("Fleet Feet", 0.15, start=d))
        H.append(Habit("RunSignup", 0.1, start=d, local=False))
        sched(u, "Strava", [x for x in monthly_dates(u, d.day if d.day <= 28 else 28, shift_weekend=False) if x >= d],
              fixed_amt(11.99))


def intend(u, e):
    """The category the user means for event e (before any filing mistake)."""
    if e.fixed is not None:
        return e.fixed
    applied = None
    for cd, cid in u.merchant_changes.get(e.mid, ()):
        if cd <= e.date:
            applied = (cd, cid)
    if applied:
        kind = u.cats[applied[1]].kind
        e.hint = "new_category" if kind in ("store", "split") else "changed_mind"
        e.hint_date = applied[0]
        return applied[1]
    so = u.store_over.get(e.mid)
    if so and e.date >= so["from"]:
        if so["from"] > u.start:
            e.hint, e.hint_date = "new_category", so["from"]
        return so.get(e.concept, so["*"])
    c = e.concept
    if c == "dining" and MER[e.mid].kind in ("restaurant", "fastfood") and e.amount > 0:
        wd = e.date.weekday()
        if u.cat_work is not None and wd < 5 and e.amount <= 22:
            e.mech = "weekday_split"
            return u.cat_work
        if u.cat_date is not None and wd in (4, 5) and e.amount >= 40:
            e.mech = "weekday_split"
            return u.cat_date
    cid, hint, hd = u.resolve(c, e.date)
    if hint:
        e.hint, e.hint_date = hint, hd
    return cid


def assign_personal(u, e):
    """Kid and partner categories: pinned per merchant or chosen per purchase (fixed at generation)."""
    rng = u.rng
    if e.fixed is not None or e.mid in u.store_over:
        return
    if u.kid_cats and e.concept in ("kids", "childcare", "school"):
        key = e.mid if e.contact is None else e.contact["name"]
        if key not in u.person_pin:
            u.person_pin[key] = rng.choice(u.kid_cats) if (MER[e.mid].kind in ("kids_activity",) or e.contact
                                                           or rng.random() < 0.3) else None
        e.fixed = u.person_pin[key] if u.person_pin[key] is not None else rng.choice(u.kid_cats)
        if u.person_pin[key] is None:
            e.mech = "multi"
    elif u.person_cats and e.concept in ("clothing", "personal_care", "hobbies", "games") and e.amount > 0:
        key = e.mid
        if key not in u.person_pin:
            u.person_pin[key] = rng.choice(u.person_cats) if rng.random() < 0.6 else None
        e.fixed = u.person_pin[key] if u.person_pin[key] is not None else rng.choice(u.person_cats)
        if u.person_pin[key] is None:
            e.mech = "multi"


def plan_changes(u):
    """Mid-year category creation (concept split, store takeover) and changes of mind."""
    rng = u.rng
    if u.is_new:
        return
    evs = u.events
    by_mid = defaultdict(list)
    for e in evs:
        if e.fixed is None and e.amount > 0 and MER[e.mid].channel != "p2p":
            by_mid[e.mid].append(e)
    # 1. concept split: a merged category loses one concept to a new category
    if rng.random() < 0.4 and len(u.cats) < MAX_CATS:
        cands = []
        for c, g in G_FOR_CONCEPT.items():
            cid, _, _ = u.resolve(c, D1)
            if len(u.cats[cid].concepts) > 1 or c not in u.cats[cid].concepts:
                n = sum(1 for e in evs if e.concept == c and e.fixed is None and e.date > day(6, 1))
                if n >= 6 and not u.concept_changes.get(c):
                    cands.append(c)
        if cands:
            c = rng.choice(cands)
            d = ALL_DAYS[rng.randint(150, 320)]
            nm = u.pick_name(G_FOR_CONCEPT[c])
            if nm:
                cid = u.new_cat(nm, G[G_FOR_CONCEPT[c]][0].split(), d, "split", pos=rng.randint(0, len(u.order)))
                u.concept_changes[c].append((d, cid))
    # 2. store takeover: "from now on everything from Amazon goes to Amazon"
    if rng.random() < 0.18 and len(u.cats) < MAX_CATS and "Amazon" not in u.store_over and len(by_mid.get("Amazon", [])) >= 10:
        d = ALL_DAYS[rng.randint(150, 320)]
        nm = rng.choice(["Amazon", "Amazon orders", "Amazon (sort later)"])
        if nm not in u.used_names:
            cid = u.new_cat(nm, set(), d, "store", pos=rng.randint(0, len(u.order)))
            MEANING[nm] |= {"store:Amazon"}
            u.store_over["Amazon"] = {"from": d, "*": cid}
    # 3. changes of mind: a regular payee is refiled to another existing category from some date on
    if rng.random() < 0.55:
        cands = []
        for mid, es in by_mid.items():
            if mid in u.store_over or MER[mid].pset is not None:
                continue
            if len(es) >= 7:
                cands.append(mid)
        for mid in pick(rng, sorted(cands), rng.choice((1, 2, 2, 3))):
            es = by_mid[mid]
            k = rng.randint(max(3, len(es) * 11 // 20), max(3, len(es) * 17 // 20))
            k = min(k, len(es) - 2)
            if k < 3:
                continue
            d = es[k].date
            old = intend(u, es[k - 1])
            con = es[k - 1].concept
            ok = PLAUSIBLE.get(con, set())
            pool = [c for c in u.cats_at(d) if c != old and not (u.cats[c].concepts & MONEY_CONCEPTS) and (
                (u.cats[c].kind in ("base", "split", "life") and u.cats[c].concepts & ok)
                or (u.cats[c].kind == "person" and con in ("clothing", "personal_care", "hobbies", "games", "gym", "coffee"))
                or (c in (u.cat_work, u.cat_date) and con in ("dining", "coffee")))]
            if pool:
                tgt = rng.choice(pool)
                u.merchant_changes[mid].append((d, tgt))


def make_refunds(u):
    rng = u.rng
    out = []
    for e in u.events:
        m = MER[e.mid]
        if m.refund and e.amount > 15 and rng.random() < (0.07 if m.kind not in ("airline", "hotel") else 0.05):
            d = e.date + dt.timedelta(days=rng.randint(2, 28))
            if d > D1:
                continue
            amt = -e.amount if rng.random() < 0.7 else -round(e.amount * rng.uniform(0.2, 0.9), 2)
            r = Ev(d, e.mid, e.concept, amt, "refund", city=e.city if m.tail is None else None, fixed=e.intended, pkey=e.pkey)
            r.ref = e
            r.trip = e.trip
            out.append(r)
    u.events.extend(out)


def file_mistakes(u):
    rng = u.rng
    for e in u.events:
        e.filed = e.intended
        p = u.p_mistake * (0.3 if e.amount < 0 else 1.0)
        if rng.random() < p:
            existing = u.cats_at(e.date)
            pos = existing.index(e.intended)
            if rng.random() < 0.6:
                near = [existing[j] for j in (pos - 2, pos - 1, pos + 1, pos + 2) if 0 <= j < len(existing)]
            else:
                near = [c for c in existing if c != e.intended and not (u.cats[c].concepts & MONEY_CONCEPTS)]
            near = [c for c in near if c != e.intended]
            if near:
                e.filed = rng.choice(near)
                e.mistake = True


# ----------------------------------------------------------------------------------------------
# Statement strings
# ----------------------------------------------------------------------------------------------
PH = re.compile(r"\{([A-Za-z]+)(\d*)\}")


def fill(tpl, u, e, rng):
    city = e.city or u.city

    def rep(mo):
        tok, n = mo.group(1), mo.group(2)
        if tok == "n":
            return digits(srng(u.uid, e.mid, e.slot, n), int(n))
        if tok == "R":
            return alnum(rng, int(n))
        if tok == "A":
            return city[:3].upper()
        if tok == "CITY":
            return city.upper()[:int(n or 8)].strip()
        if tok == "ST":
            return CITY_ST.get(city, u.st)
        if tok == "SHOP":
            return rng.choice(ETSY_SHOPS)
        if tok == "REST":
            return rng.choice(DELIVERY_RESTS)
        if tok == "EVENT":
            return rng.choice(EVENTS)
        if tok == "GFM":
            return rng.choice(GFM)
        if tok == "RACE":
            return rng.choice(RACES)
        if tok == "DAYCARE":
            return srng(u.uid, "daycare").choice(DAYCARES)
        if tok == "PROP":
            return srng(u.uid, "prop").choice(["THE LANDING APTS", "MAPLE COURT", "RIVERBEND LOFTS", "ELM ST PROPERTIES"])
        if tok == "WDU":
            return WD[e.date.weekday()].upper()
        if tok == "HR":
            return f"{rng.randint(1, 11)}{rng.choice(['AM', 'PM'])}"
        return mo.group(0)
    return PH.sub(rep, tpl)


def render(u, e, rng):
    m = MER[e.mid]
    st = u.style
    auth = e.date - dt.timedelta(days=rng.choice((0, 0, 1, 1, 2)))
    mmdd = f"{auth.month:02d}/{auth.day:02d}"
    if m.channel == "p2p":
        return render_p2p(u, e, rng, mmdd)
    if m.channel == "atm":
        addr = f"{rng.randint(100, 9999)} {rng.choice(['MAIN ST', 'BROADWAY', 'OAK AVE', 'MARKET ST', '5TH AVE'])}"
        return {"clean": "ATM Withdrawal", "boa": f"BKOFAMERICA ATM {mmdd} #{digits(rng, 9)} WITHDRWL {addr}",
                "wells": f"ATM WITHDRAWAL AUTHORIZED ON {mmdd} {addr} {u.city.upper()} {u.st} {digits(rng, 16)} ATM ID {digits(rng, 4)} CARD {u.card4}",
                }.get(st, f"ATM WITHDRAWAL {digits(rng, 6)} {addr} {u.city.upper()} {u.st}")
    if m.channel == "fee":
        return m.desc[0].title() if st == "clean" else m.desc[0]
    if m.channel == "xfer":
        acct = digits(srng(u.uid, "sav"), 4)
        if e.amount > 0:
            return {"clean": "Transfer to Savings", "boa": f"Online Banking transfer to SAV {acct} Confirmation# {digits(rng, 10)}",
                    "wells": f"ONLINE TRANSFER TO {u.person.split()[0]} WAY2SAVE SAVINGS XXXXXX{acct} REF #IB0{alnum(rng, 7)} ON {mmdd}",
                    }.get(st, f"ONLINE TRANSFER TO SAV ...{acct} TRANSACTION#: {digits(rng, 11)}")
        return {"clean": "Transfer from Savings", "boa": f"Online Banking transfer from SAV {acct} Confirmation# {digits(rng, 10)}",
                "wells": f"ONLINE TRANSFER FROM {u.person.split()[0]} WAY2SAVE SAVINGS XXXXXX{acct} REF #IB0{alnum(rng, 7)} ON {mmdd}",
                }.get(st, f"ONLINE TRANSFER FROM SAV ...{acct} TRANSACTION#: {digits(rng, 11)}")
    if m.channel == "ach":
        coid = digits(srng(m.id, "coid"), 10)
        comp, descr = m.comp, m.descr
        if st == "clean":
            return m.clean
        if st == "boa":
            return f"{comp} DES:{descr} ID:{alnum(rng, 10)} INDN:{u.person} CO ID:{coid} {'PPD' if e.amount < 0 else 'WEB'}"
        if st == "wells":
            return f"{comp} {descr} {e.date.strftime('%y%m%d')} {digits(rng, 9)} {u.person}"
        if st in ("cu", "fixed"):
            return f"ACH {'CREDIT' if e.amount < 0 else 'DEBIT'} {comp[:16]} {descr}"
        return f"{comp} {descr} PPD ID: {coid}"
    # card purchase
    ps = PSETS.get(m.pset) if m.pset else None
    if e.pkey is not None and ps and len(ps[e.pkey]) > 4 and ps[e.pkey][4]:
        base_t = ps[e.pkey][4]
        clean = ps[e.pkey][5]
    else:
        base_t = m.desc[srng(u.uid, e.mid, "d").randrange(len(m.desc))]
        clean = m.clean
    base = fill(base_t, u, e, rng)
    if e.amount < 0 and rng.random() < 0.6:
        base = rng.choice(["RETURN ", "REFUND ", "CREDIT "]) + base if st in ("cu", "fixed", "boa") else base + " REFUND"
    if st == "clean":
        return clean + (" (refund)" if e.amount < 0 else "")
    city = e.city
    if city:
        loc = f"{city.upper()} {CITY_ST.get(city, u.st)}"
    else:
        loc = m.tail or ""
    if st == "chase":
        return f"{base} {loc}".strip()
    if st == "boa":
        return f"{'PURCHASE' if rng.random() < 0.3 else 'CHECKCARD'} {mmdd[:2]}{mmdd[3:]} {base} {loc} {digits(rng, 23)}".strip()
    if st == "wells":
        return f"PURCHASE AUTHORIZED ON {mmdd} {base} {loc} S{digits(rng, 15)} CARD {u.card4}"
    if st == "cu":
        return f"POS DEBIT {base[:22]:<22} {loc[:16]}".rstrip()
    if st == "fixed":
        cty = (city or loc).upper()
        stt = CITY_ST.get(city, "") if city else ""
        return f"{base[:23]:<23} {cty[:13]:<13} {stt}".rstrip()
    # amex
    ap = "AplPay " if srng(u.uid, "aplpay").random() < 0.4 and m.tail is None else ""
    return f"{ap}{base[:20]:<20} {loc}".rstrip()


def render_p2p(u, e, rng, mmdd):
    c = e.contact
    svc = e.mid
    st = u.style
    nm = c["name"] if c else "UNKNOWN"
    NM = nm.upper()
    first, last = nm.split()
    out = e.amount > 0
    if svc == "Check":
        u.check_no += 1
        e.vtok = ""
        return {"clean": f"Check #{u.check_no}", "boa": f"Check {u.check_no}", "wells": f"CHECK # {u.check_no}"}.get(st, f"CHECK #{u.check_no}")
    if svc == "Zelle":
        e.vtok = nm
        conf = alnum(rng, 9)
        if out:
            return {"clean": f"Zelle to {nm}", "boa": f"Zelle payment to {nm} Conf# {conf.lower()}",
                    "wells": f"ZELLE TO {last.upper()} {first.upper()} ON {mmdd} REF # PP0{alnum(rng, 7)}",
                    "cu": f"ZELLE DEBIT {NM}"}.get(st, f"Zelle payment to {nm} JPM99{conf.lower()}")
        return {"clean": f"Zelle from {nm}", "boa": f"Zelle payment from {nm} Conf# {conf.lower()}",
                "wells": f"ZELLE FROM {last.upper()} {first.upper()} ON {mmdd} REF # WFCT0{alnum(rng, 6)}",
                "cu": f"ZELLE CREDIT {NM}"}.get(st, f"Zelle payment from {nm} WFCT0{conf}")
    if svc == "Cash App":
        if out and u.venmo_names:
            e.vtok = nm
            return "Cash App - " + nm if st == "clean" else f"CASH APP*{first.upper()} {last[0].upper()}"
        e.vtok = ""
        return "Cash App" if st == "clean" else ("CASH APP*CASH OUT" if not out else f"SQ *CASH APP {digits(rng, 8)}")
    # Venmo
    if out and u.venmo_names:
        e.vtok = nm
        return f"Venmo - {nm}" if st == "clean" else f"VENMO *{NM}"
    e.vtok = ""
    if st == "clean":
        return "Venmo"
    if st == "boa":
        return f"VENMO DES:{'PAYMENT' if out else 'CASHOUT'} ID:{digits(rng, 10)} INDN:{u.person} CO ID:{digits(srng('venmo'), 10)} WEB"
    if st == "wells":
        return f"VENMO {'PAYMENT' if out else 'CASHOUT'} {e.date.strftime('%y%m%d')} {digits(rng, 10)} {u.person}"
    return f"VENMO {'PAYMENT' if out else 'CASHOUT'} {digits(rng, 10)} WEB ID: {digits(srng('venmo'), 10)}"


# ----------------------------------------------------------------------------------------------
# Build one user
# ----------------------------------------------------------------------------------------------
def build_user(uid, arch, is_new):
    rng = random.Random(SEED * 1000 + uid)
    u = User(uid, arch, rng, is_new)
    build_categories(u)
    build_trips(u)
    H = build_habits(u)
    build_life_event(u, H)
    build_scheduled(u)
    for h in H:
        u.events.extend(gen_poisson(u, h, rng))
    # exploration: one-off visits to businesses the user has not used before (and a few become regulars)
    for d in ALL_DAYS:
        if d >= u.start and not u.away(d) and u.explore_pool and rng.random() < u.explore_rate / 7:
            mid = rng.choice(u.explore_pool[: max(4, len(u.explore_pool) * (d.timetuple().tm_yday + 60) // 425)])
            m = MER[mid]
            c, a, pk, mech = purpose_pick(u, m, d, rng)
            u.events.append(Ev(d, mid, c, a, mech, city=(m.city or u.city) if not m.tail else None, pkey=pk))
    u.events = [e for e in u.events if u.start <= e.date <= D1]
    for e in u.events:
        e.order = rng.random()
        m = MER[e.mid]
        e.slot = 0 if rng.random() < 0.85 else 1
        if e.city is None and m.channel == "card" and not m.tail:
            e.city = m.city or u.city
        if e.city == u.city and m.city is None and rng.random() < 0.3:
            e.city = rng.choice(CITIES[u.city][1])
    u.events.sort(key=lambda e: (e.date, e.order))
    for e in u.events:
        assign_personal(u, e)
    plan_changes(u)
    for e in u.events:
        e.hint = None
        e.intended = intend(u, e)
    make_refunds(u)
    for e in u.events:
        if e.intended is None:          # refunds: filed like the purchase they reverse
            e.intended = e.fixed
    u.events.sort(key=lambda e: (e.date, e.order))
    file_mistakes(u)
    for e in u.events:
        e.text = render(u, e, rng)
        if e.pkey is not None and not e.vtok:
            e.vtok = f"p{e.pkey}"
        assert u.cats[e.intended].created <= e.date and u.cats[e.filed].created <= e.date, (uid, e.mid, e.date)
    return u


# ----------------------------------------------------------------------------------------------
# Slice, oracle, labels
# ----------------------------------------------------------------------------------------------
def build_slice(evs, qi):
    q = evs[qi]
    chosen = []
    seen = set()

    def take(i):
        if i not in seen and len(chosen) < MAX_ROWS:
            seen.add(i)
            chosen.append(i)

    same = [i for i in range(qi) if evs[i].mid == q.mid][-MAX_SAME:]
    for i in same:
        take(i)
    sim_cls = MER[q.mid].sim
    per = Counter()
    n_sim = 0
    for i in range(qi - 1, -1, -1):
        if n_sim >= MAX_SIMILAR:
            break
        e = evs[i]
        if e.mid != q.mid and MER[e.mid].sim == sim_cls and per[e.mid] < 2:
            per[e.mid] += 1
            n_sim += 1
            take(i)
    shown = {evs[i].filed for i in chosen}
    for i in range(qi - 1, -1, -1):
        if len(chosen) >= MAX_ROWS:
            break
        if evs[i].filed not in shown:
            shown.add(evs[i].filed)
            take(i)
    for i in range(qi - 1, -1, -1):
        if len(chosen) >= MAX_ROWS:
            break
        take(i)
    return sorted(chosen)


def reader_concept(m, q):
    if m.pset:
        ps = PSETS[m.pset]
        if q.pkey is not None and len(ps[q.pkey]) > 4 and ps[q.pkey][4]:
            return ps[q.pkey][0]
        a = abs(q.amount)
        best = max(ps, key=lambda p: p[1] * (1.0 if p[2] <= a <= p[3] else 0.15))
        return best[0]
    return KIND_CONCEPT[m.kind]


def name_match(names, concept):
    seen, queue = set(), [concept]
    while queue:
        c = queue.pop(0)
        if c in seen:
            continue
        seen.add(c)
        hits = [i for i, n in enumerate(names) if c in MEANING.get(n, ())]
        plain = [i for i in hits if not any(x.startswith(("store:", "dest:")) for x in MEANING[names[i]])
                 and "trip" not in MEANING[names[i]]]
        hits = plain or hits
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            return -1
        queue.extend(FALLBACK.get(c, []))
    return -1


def majority_latest(rows):
    cnt = Counter(r.filed for r in rows)
    top = max(cnt.values())
    for r in reversed(rows):
        if cnt[r.filed] == top:
            return r.filed


def dist(r, q):
    a, b = abs(r.amount) + 1, abs(q.amount) + 1
    d = abs(math.log(a) - math.log(b))
    d += 0.6 * ((r.date.weekday() >= 5) != (q.date.weekday() >= 5))
    hol = lambda x: x.month == 12 or (x.month == 11 and x.day >= 20)
    d += 0.6 * (hol(r.date) != hol(q.date))
    d += 0.01 * (q.date - r.date).days / 30
    return d


def oracle(u, opt_ids, names, rows, q):
    """What an ideal reader would answer from the prompt plus world knowledge of merchants."""
    idx = {c: i for i, c in enumerate(opt_ids)}
    M = MER[q.mid]
    # 1. away from home: recent rows in the same city, or a category named after the destination
    if q.city and q.city not in u.home_cluster and u.style != "clean":
        near = [r for r in rows if r.city == q.city and 0 <= (q.date - r.date).days <= 14]
        if near:
            return idx[majority_latest(near)]
        hits = [i for i, n in enumerate(names) if ("dest:" + q.city) in MEANING.get(n, ())]
        if len(hits) == 1:
            return hits[0]
    if u.style == "clean" and M.channel == "card" and M.tail is None:
        # the enriched feed hides the city; a run of trip-filed rows in the last two days is the only hint
        near = [r for r in rows if 0 <= (q.date - r.date).days <= 2 and "trip" in MEANING.get(u.cats[r.filed].name, ())]
        if len(near) >= 2:
            return idx[majority_latest(near)]
    same = [r for r in rows if r.mid == q.mid]
    if M.channel == "p2p":
        same = [r for r in same if r.vtok == q.vtok and (r.amount < 0) == (q.amount < 0)]
        if not same:
            return -1
    elif q.vtok:
        s2 = [r for r in same if r.vtok == q.vtok]
        same = s2 if s2 else same
    # a category literally named after this store
    store_hits = [i for i, n in enumerate(names) if ("store:" + q.mid) in MEANING.get(n, ())]
    if store_hits and not (same and same[-1].filed in [opt_ids[i] for i in store_hits]):
        if len(store_hits) == 1:
            return store_hits[0]
        con = reader_concept(M, q)
        pick_ = [i for i in store_hits if con in MEANING[names[i]]]
        return pick_[0] if len(pick_) == 1 else store_hits[-1]
    if same:
        if q.amount < 0 and M.kind not in INFLOW_KINDS and M.channel != "p2p":
            negs = [r for r in same if r.amount < 0]
            pos = [r for r in same if r.amount > 0]
            if pos:
                return idx[min(pos, key=lambda r: (abs(abs(r.amount) + q.amount), -r.date.toordinal())).filed]
            if negs:
                return idx[majority_latest(negs)]
        cats = [r.filed for r in same]
        if len(set(cats)) == 1:
            base = cats[0]
            # a newer category that the same kind of business moved to (new category taking over)
            con = reader_concept(M, q)
            later = [r for r in rows if r.mid != q.mid and M.reader and MER[r.mid].reader == M.reader
                     and r.date > same[-1].date and r.amount > 0]
            xs = {r.filed for r in later}
            if len(later) >= 2 and len(xs) == 1:
                x = xs.pop()
                if x != base and con in MEANING.get(u.cats[x].name, ()) and con not in MEANING.get(u.cats[base].name, ()):
                    return idx[x]
            return idx[base]
        last = cats[-1]
        run = 0
        for c in reversed(cats):
            if c != last:
                break
            run += 1
        first_last = cats.index(last)
        if run >= 2 and all(c == last for c in cats[first_last:]):
            return idx[last]
        amts = [abs(r.amount) for r in same]
        if max(amts) <= 1.15 * min(amts) and abs(q.amount) <= 1.15 * max(amts) and abs(q.amount) >= min(amts) / 1.15:
            return idx[last]
        top, n = Counter(cats).most_common(1)[0]
        if n / len(cats) >= 0.75:
            return idx[top]
        return idx[min(same, key=lambda r: dist(r, q)).filed]
    # no row of this payee in the prompt
    if M.reader is None:
        return -1
    if q.amount < 0 and M.kind not in INFLOW_KINDS:
        return -1
    sim = [r for r in rows if r.mid != q.mid and MER[r.mid].reader == M.reader and (r.amount < 0) == (q.amount < 0)]
    if M.kind in INFLOW_KINDS:
        sim = [r for r in rows if MER[r.mid].kind in INFLOW_KINDS]
        if sim:
            return idx[majority_latest(sim)]
    if sim and M.channel != "p2p":
        return idx[min(sim, key=lambda r: dist(r, q)).filed]
    return name_match(names, reader_concept(M, q))


WHY_W = {"plain": 0.3, "recurring": 0.3, "income": 0.7, "transfer": 0.5, "p2p": 1.6, "multi_purpose": 1.0,
         "amount_split": 1.2, "weekday_split": 2.5, "seasonal": 2.0, "first_time_payee": 2.4, "opaque_new": 4.0,
         "changed_mind": 6.0, "new_category": 6.0, "trip": 2.4, "refund": 3.5, "reimbursement": 3.0,
         "misfiled_history": 1.5, "new_user": 1.0}
MECH_TAG = {"multi": "multi_purpose", "amount_split": "amount_split", "weekday_split": "weekday_split",
            "seasonal": "seasonal", "p2p": "p2p"}


def why_of(u, evs, qi, prev):
    e = evs[qi]
    if u.is_new:
        return "new_user"
    if e.mech == "refund":
        return "refund"
    if e.mech == "reimbursement":
        return "reimbursement"
    if e.mech == "income":
        return "income"
    if e.trip and e.fixed is not None:
        return "trip"
    if e.hint == "new_category":
        cr = u.cats[e.intended].created
        if (e.date - cr).days <= 60 or (prev and all(evs[i].date < cr for i in prev)):
            return "new_category"
    if e.hint == "changed_mind":
        after = [i for i in prev if evs[i].date >= e.hint_date]
        before = [i for i in prev if evs[i].date < e.hint_date]
        if before and len(after) < 4:
            return "changed_mind"
    if not prev:
        return "opaque_new" if MER[e.mid].reader is None else "first_time_payee"
    if any(evs[i].mistake for i in prev[-3:]):
        return "misfiled_history"
    cats = {evs[i].intended for i in prev} | {e.intended}
    if len(cats) > 1 and e.mech in MECH_TAG:
        return MECH_TAG[e.mech]
    if e.mech == "transfer":
        return "transfer"
    if e.sched:
        return "recurring"
    return "plain"


def fmt_amt(a):
    return f"${a:.2f}"


def row_text(u, e, with_cat=True):
    s = f"Transaction: {e.date.isoformat()} | {e.text} | {fmt_amt(e.amount)} | {WD[e.date.weekday()]}\nCategory:"
    return s + (" " + u.cats[e.filed].name if with_cat else "")


def main():
    rng = random.Random(SEED)
    archs = []
    for i in range(N_USERS - N_NEW_USERS):
        archs.append(wchoice(rng, ARCH_W))
    archs += ["new_user"] * N_NEW_USERS
    rng.shuffle(archs)
    users = [build_user(uid, a, a == "new_user") for uid, a in enumerate(archs)]

    # other users' filings: payee -> user -> filed category name -> earliest date
    others = defaultdict(lambda: defaultdict(dict))
    for u in users:
        for e in u.events:
            nm = u.cats[e.filed].name
            d = others[e.mid][u.uid]
            if nm not in d or e.date < d[nm]:
                d[nm] = e.date

    items = []
    for u in users:
        evs = u.events
        prev_of = []
        seen_idx = defaultdict(list)
        for i, e in enumerate(evs):
            prev_of.append(list(seen_idx[e.mid]))
            seen_idx[e.mid].append(i)
        if u.is_new:
            cand = list(range(0, min(11, len(evs))))
            chosen = sorted(u.rng.sample(cand, min(Q_PER_USER, len(cand))))
        else:
            lo = max(11, int(len(evs) * 0.55))
            cand = list(range(lo, len(evs)))
            whys = {i: why_of(u, evs, i, prev_of[i]) for i in cand}
            chosen, per_why = [], Counter()
            pool = list(cand)
            while len(chosen) < Q_PER_USER and pool:
                ws = [(i, WHY_W[whys[i]] * (0.15 if per_why[whys[i]] >= 2 else 1.0)) for i in pool]
                i = wchoice(u.rng, ws)
                pool.remove(i)
                chosen.append(i)
                per_why[whys[i]] += 1
            chosen.sort()
        for n, qi in enumerate(chosen):
            q = evs[qi]
            prev = prev_of[qi]
            opt_ids = u.cats_at(q.date)
            names = [u.cats[c].name for c in opt_ids]
            assert len(set(names)) == len(names) and all(", " not in x for x in names)
            idx = {c: i for i, c in enumerate(opt_ids)}
            sl = build_slice(evs, qi)
            rows = [evs[i] for i in sl]
            for r in rows:
                assert r.date <= q.date, "future row"
                assert u.cats[r.filed].created <= r.date, "category did not exist on row date"
                assert r.filed in idx
            assert q.intended in idx
            head = "Categories: " + ", ".join(names)
            blocks = [head] + [row_text(u, r) for r in rows]
            final = row_text(u, q, with_cat=False)
            prompt = "\n\n".join(blocks + [final])
            ou = {uid: cats for uid, cats in others[q.mid].items() if uid != u.uid}
            cnt = Counter()
            n_users = 0
            for uid, cats in ou.items():
                hit = False
                for nm, d0 in cats.items():
                    if d0 < q.date:
                        cnt[nm] += 1
                        hit = True
                n_users += hit
            if n_users >= 3:
                top = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
                line = "Other users file this payee as: " + ", ".join(f"{k} ({v})" for k, v in top)
                prompt_others = "\n\n".join(blocks + [line, final])
            else:
                prompt_others = prompt
            filings = [evs[i].filed for i in prev]
            if filings:
                c3 = Counter(filings[-3:]).most_common(1)[0]
                yr = idx[c3[0]] if c3[1] >= 2 else idx[filings[-1]]
            else:
                yr = -1
            why = why_of(u, evs, qi, prev)
            best = oracle(u, opt_ids, names, rows, q)
            items.append({
                "id": f"B2:{u.uid}:{n}",
                "user": u.uid,
                "level": f"B2_{u.arch}",
                "merchant": q.mid,
                "text": q.text,
                "amount": q.amount,
                "weekday": WD[q.date.weekday()],
                "date": q.date.isoformat(),
                "options": [" " + x for x in names],
                "answer": idx[q.intended],
                "prompt": prompt,
                "prompt_ctx": prompt,
                "prompt_others": prompt_others,
                "why": why,
                "best": best,
                "payee_hist": sorted({idx[c] for c in filings}),
                "ynab_rule": yr,
                "hist_len": qi,
                "in_shots": any(r.mid == q.mid for r in rows),
            })

    blob = json.dumps(items, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    notes = (
        "Blind v2: written without reference to the repo's own generators. 250 synthetic US users (32 brand-new, "
        "0-10 earlier rows) over calendar 2025 in 10 home cities, each with an archetype (family, minimal, couple, "
        "detailed, gig, freelancer, student, retiree, coined-name, store-split, roommates, traveler), a personal "
        "category list of 6-45 names "
        "(standard, personal, merged, detailed, coined, kid and partner names, store categories like 'Costco - food'), "
        "and a bank style (chase, boa, wells, credit-union POS, fixed-width, enriched clean, amex) that renders every "
        "statement string. Transactions come from ~350 national chains, billers and income sources and ~560 "
        "city-level businesses and billers (most invented; ~60 behind opaque processor names such as SQ *, IN *, CLV*, "
        "PY *), poisson habits with weekday profiles, "
        "scheduled bills with seasonal amounts and price hikes, payroll/gig/freelance/SSA income, P2P contacts "
        "(Venmo/Zelle/Cash App/checks, counterparty shown or not), trips filed to trip categories created at booking, "
        "life events creating categories that take over a concept, concept splits and store takeovers mid-year, "
        "changes of mind, refunds and reimbursements filed like the expense, and 0-4% filing mistakes (answers are "
        "always the intended category). Multi-purpose payees (Amazon, Target, Walmart, Costco, pharmacies, gas "
        "stations, Apple, Google, Uber, P2P) pick a purpose per visit with season and amount effects. 'best' is a "
        "rule-based reader over the prompt plus merchant knowledge (-1 when the prompt gives no basis)."
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"name": "blind", "version": "v2", "n_users": len(users), "items": items, "sha256": digest, "notes": notes}
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    # summary
    n = len(items)
    print(f"wrote {OUT_FILE}")
    print(f"items {n}  users {len({it['user'] for it in items})}  sha256 {digest[:16]}")
    wc = Counter(it["why"] for it in items)
    for k, v in wc.most_common():
        ceil_k = sum(1 for it in items if it["why"] == k and it["best"] == it["answer"]) / v
        print(f"  {k:18s} {v:5d}  ceiling {ceil_k:.2f}")
    print(f"ceiling (best == answer): {sum(it['best'] == it['answer'] for it in items) / n:.3f}")
    print(f"best == -1: {sum(it['best'] == -1 for it in items) / n:.3f}")
    print(f"payee_hist empty: {sum(not it['payee_hist'] for it in items) / n:.3f}")
    wh = [it for it in items if it["payee_hist"]]
    print(f"answer != ynab_rule (items with payee history): {sum(it['answer'] != it['ynab_rule'] for it in wh) / max(1, len(wh)):.3f}")
    print(f"prompt_others differs: {sum(it['prompt_others'] != it['prompt'] for it in items) / n:.3f}")
    print(f"mean options: {sum(len(it['options']) for it in items) / n:.1f}  "
          f"range {min(len(it['options']) for it in items)}-{max(len(it['options']) for it in items)}")


if __name__ == "__main__":
    main()
