"""Transaction kinds outside REAL-6's twelve spending categories (PLAN step 109): income, housing, insurance, loans, subscriptions,
savings transfers, personal care, gifts, donations, childcare, education, bank fees, cash. REPORT 110: most of blind_v1's new-user
misses were such transactions (a payroll deposit, an insurance premium, a car-loan payment), which no training episode contained.
The catalogue comes from general knowledge of US budgets; any payee whose name also appears in blind_v1 is dropped (`payees`), so the
blind set's merchants stay unseen.

Each kind: payees (name, statement style), an amount (log-normal, or a fixed recurring amount per user), the sign (income is negative,
as blind_v1 and most bank exports show inflows), and category names users give it (renamed further by the episode's RENAME).
`row(kind, rng, fixed=None)` gives a statement row dict like exp_categoriser's rows (text, amount, weekday, merchant).
"""
import json
import math
from functools import lru_cache

from .paths import PROCESSED

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

KINDS = {
    "income": dict(names=["Income", "Paycheck", "Salary", "Inflow", "Ready to Assign", "Earnings", "Pay"], sign=-1, amount=(7.6, 0.4), fixed=True,
                   payees=["Acme Logistics", "Brightline Health", "Cedar Valley Schools", "Northwind Foods", "Pioneer Mfg", "Summit Bank", "Tri-County Hospital",
                           "Harbor Freight Payroll", "Lakeshore Dental", "Metro Transit Authority", "Sierra Software", "Oakridge County"],
                   style="payroll"),
    "rent": dict(names=["Rent", "Mortgage", "Housing", "Home", "Rent/Mortgage", "House payment"], sign=1, amount=(7.3, 0.3), fixed=True,
                 payees=["Greystar Properties", "Maple Court Apts", "Riverbend Residences", "Wells Home Mortgage", "Rocket Home Loans", "Parkview Leasing",
                         "Sunset Village Apartments", "Mr. Cooper Mortgage"], style="ach"),
    "insurance": dict(names=["Insurance", "Car insurance", "Renters insurance", "Life insurance", "Home insurance"], sign=1, amount=(4.6, 0.6), fixed=True,
                      payees=["Progressive Ins", "Allstate Ins", "Farmers Insurance", "Nationwide Ins", "Liberty Mutual", "USAA Insurance", "Root Insurance",
                              "Hippo Insurance", "Northwestern Mutual"], style="ach"),
    "loan": dict(names=["Car payment", "Student loans", "Loan payment", "Debt", "Auto loan", "Car loan"], sign=1, amount=(5.9, 0.5), fixed=True,
                 payees=["Toyota Financial", "Ford Credit", "Ally Auto", "Nelnet", "Great Lakes Loans", "Mohela", "Capital One Auto", "Hyundai Motor Fin",
                         "Upstart Loan", "LendingClub"], style="ach"),
    "subscription": dict(names=["Subscriptions", "Streaming", "Apps", "Software", "Memberships", "Digital"], sign=1, amount=(2.6, 0.5), fixed=True,
                         payees=["Hulu", "Disney Plus", "YouTube Premium", "Audible", "Adobe", "Microsoft 365", "iCloud Storage", "Peacock", "ESPN Plus",
                                 "NYTimes Digital", "Duolingo", "Headspace"], style="card"),
    "savings": dict(names=["Savings", "Emergency fund", "Transfer to savings", "Rainy day", "Investments"], sign=1, amount=(5.5, 0.6), fixed=True,
                    payees=["Transfer to Savings", "Ally Bank Transfer", "Vanguard Buy", "Fidelity Investments", "Betterment", "Marcus Savings"], style="transfer"),
    "personal_care": dict(names=["Personal care", "Hair", "Beauty", "Self care", "Grooming"], sign=1, amount=(3.5, 0.6), fixed=False,
                          payees=["Supercuts", "Sport Clips", "Sally Beauty", "Bath & Body Works", "European Wax Center", "Massage Envy", "Nail Studio 21",
                                  "Hair Cuttery"], style="card"),
    "gifts": dict(names=["Gifts", "Birthdays", "Christmas", "Holidays", "Presents"], sign=1, amount=(3.8, 0.6), fixed=False,
                  payees=["FTD Flowers", "ProFlowers", "Hallmark", "Shari's Berries", "Etsy", "Things Remembered", "Uncommon Goods"], style="card"),
    "donations": dict(names=["Giving", "Charity", "Donations", "Tithing", "Church"], sign=1, amount=(3.9, 0.7), fixed=True,
                      payees=["Red Cross", "St Jude Childrens", "United Way", "Grace Community Church", "Habitat for Humanity", "NPR Donation",
                              "Salvation Army"], style="ach"),
    "childcare": dict(names=["Childcare", "Daycare", "Kids", "Kid activities", "Babysitting"], sign=1, amount=(5.3, 0.6), fixed=True,
                      payees=["Bright Horizons", "KinderCare", "Goddard School", "Little Sprouts Daycare", "Mathnasium", "YMCA Youth Programs",
                              "Kids Gymnastics Academy"], style="ach"),
    "education": dict(names=["Education", "Tuition", "School", "Books & courses", "Classes"], sign=1, amount=(5.0, 0.9), fixed=False,
                      payees=["State University Bursar", "Coursera", "Udemy", "Community College", "Chegg", "Pearson Education"], style="card"),
    "fees": dict(names=["Bank fees", "Fees", "Interest & fees", "Banking"], sign=1, amount=(2.5, 0.6), fixed=False,
                 payees=["Monthly Service Fee", "Overdraft Fee", "ATM Fee", "Foreign Transaction Fee", "Wire Transfer Fee"], style="fee"),
    "cash": dict(names=["Cash", "ATM", "Cash spending", "Walking around money"], sign=1, amount=(4.2, 0.5), fixed=False,
                 payees=["ATM Withdrawal", "Cash Withdrawal", "Chase ATM", "BofA ATM"], style="atm"),
}


@lru_cache(maxsize=1)
def _blind_names():
    try:
        return {i["merchant"].lower() for i in json.loads((PROCESSED / "blind_v1.json").read_text())["items"]}
    except FileNotFoundError:
        return set()


def payees(kind):
    """The kind's payees, minus any whose name blind_v1 uses (the test set stays unseen)."""
    return [p for p in KINDS[kind]["payees"] if p.lower() not in _blind_names()]


def _text(style, name, rng):
    u = name.upper()
    n = rng.randint(100000, 9999999)
    d = f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}"
    return {"payroll": rng.choice([f"{u} PAYROLL PPD ID: {n}", f"{u} DIR DEP {n}", f"DIRECT DEPOSIT {u}", f"{u} DES:PAYROLL ID:{n} INDN:"]),
            "ach": rng.choice([f"{u} PPD ID: {n}", f"ACH DEBIT {u} {n}", f"{u} ONLINE PMT {n}", f"{u} DES:PAYMENT ID:{n}", f"{u} AUTOPAY"]),
            "card": rng.choice([f"{u}", f"{u} {d}", f"POS DEBIT {d} {u}", f"PURCHASE {u} {n % 10000}", f"{u[:18]}*{n % 1000000}"]),
            "transfer": rng.choice([f"{u} {n}", f"ONLINE TRANSFER TO SAV {n % 10000}", f"{u} REF {n}"]),
            "fee": rng.choice([u, f"{u} {d}"]),
            "atm": rng.choice([f"{u} {d} #{n % 10000}", f"{u} {n % 10000} MAIN ST"])}[style]


def row(kind, rng, payee=None, fixed=None):
    """A statement row of this kind (payee drawn unless given; `fixed` pins a recurring amount)."""
    k = KINDS[kind]; ps = payees(kind)
    name = payee or rng.choice(ps)
    a = fixed if fixed is not None else round(math.exp(rng.gauss(*k["amount"])), 2)
    return dict(text=_text(k["style"], name, rng), amount=round(k["sign"] * abs(a), 2), weekday=rng.choice(WEEKDAYS), merchant=name, synthetic=True, kind=kind)
