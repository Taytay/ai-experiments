"""Canonical payee kinds (PLAN steps 147 to 149; owner, 2026-09-30: users' categories are personal, so describe payees by a canonical kind at
inference; and can a model infer a kind from categorised transactions?). One shared vocabulary of what a payee is (not what a user calls
it), with maps from each blind generator's own vocabulary: blind_v1's merchant kinds and category roles, blind_v2's merchant kinds and
category concepts, and REAL-6's standard categories (for training episodes). Person-to-person payees get a kind that says their purpose
varies; categories that stand for a person, trip or life event rather than a kind of business map to PURPOSE; categories holding
several kinds map to SEVERAL.
"""
CANON = {
    "grocery": "grocery store", "restaurant": "restaurant", "fast_food": "fast food", "coffee": "coffee shop or bakery",
    "delivery": "food delivery", "alcohol": "bar or liquor store", "fuel": "gas station", "convenience": "convenience store",
    "auto": "car repair or parts", "loan": "car or student loan payment", "parking": "parking or tolls",
    "transit": "public transit or rideshare", "travel": "airline, hotel or car rental",
    "utility": "utility (electric, gas, water, trash)", "telecom": "phone or internet provider", "housing": "rent, mortgage or HOA",
    "insurance": "insurance", "medical": "doctor, dentist, hospital or optician", "pharmacy": "pharmacy", "fitness": "gym or fitness",
    "personal_care": "salon, barber or personal care", "subscription": "streaming, software or subscription",
    "entertainment": "entertainment (movies, events, games)", "hobby": "books, hobby or sports store",
    "general": "big-box or online store (sells many kinds of things)", "clothing": "clothing or department store",
    "home": "home improvement, furniture or household", "electronics": "electronics store", "pets": "pet store or vet",
    "kids": "childcare, school or kids' activities", "education": "education or tuition", "gifts": "gifts or florist",
    "charity": "charity or church", "income": "employer, payroll or other income", "savings": "savings or investment transfer",
    "fees": "bank fee, cash withdrawal or card payment", "p2p": "person-to-person payment (purpose varies)",
    "taxes": "taxes or government", "business": "business services or supplies", "laundry": "laundry or dry cleaning",
    "wedding": "wedding or event vendor",
}
PURPOSE = "a person, trip or purpose (not a kind of business)"
SEVERAL = "several kinds of spending"
CATEGORY_OPTIONS = list(CANON.values()) + [SEVERAL, PURPOSE]
PAYEE_OPTIONS = list(CANON.values())

V1_KIND = {"airline": "travel", "atm": "fees", "auto_service": "auto", "baby": "kids", "bigbox": "general", "books": "hobby",
           "car_payment": "loan", "charity": "charity", "childcare": "kids", "clothing": "clothing", "coffee": "coffee",
           "delivery": "delivery", "education": "education", "electric": "utility", "electronics": "electronics",
           "entertainment": "entertainment", "fastfood": "fast_food", "fee": "fees", "fitness": "fitness", "furniture": "home",
           "games": "entertainment", "gas": "fuel", "gas_conv": "fuel", "gifts": "gifts", "grocery": "grocery", "hobby": "hobby",
           "home_improve": "home", "insurance": "insurance", "internet": "telecom", "investing": "savings", "kids_activity": "kids",
           "kids_store": "kids", "liquor": "alcohol", "lodging": "travel", "medical": "medical", "mortgage": "housing",
           "natgas": "utility", "office": "business", "online": "general", "parking": "parking", "personal_care": "personal_care",
           "pet": "pets", "pharmacy": "pharmacy", "phone": "telecom", "rent": "housing", "rental": "travel", "restaurant": "restaurant",
           "rideshare": "transit", "savings": "savings", "school": "kids", "software": "subscription", "streaming": "subscription",
           "student_loan": "loan", "tax_refund": "taxes", "transit": "transit", "trash": "utility", "warehouse": "general",
           "water": "utility", "wedding": "wedding", "p2p": "p2p", "payroll": "income", "interest": "income", "bar": "alcohol",
           "church": "charity", "hoa": "housing"}
V1_ROLE = {"air": "travel", "alcohol": "alcohol", "auto_ins": "insurance", "auto_maint": "auto", "baby": "kids", "bars": "alcohol",
           "books": "hobby", "car_payment": "loan", "cash": "fees", "charity": "charity", "childcare": "kids", "clothing": "clothing",
           "cloud": "subscription", "coffee": "coffee", "delivery": "delivery", "dining": "restaurant", "education": "education",
           "electric": "utility", "electronics": "electronics", "entertainment": "entertainment", "fast_food": "fast_food",
           "fees": "fees", "fitness": "fitness", "fuel": "fuel", "furniture": "home", "games": "entertainment", "gifts": "gifts",
           "groceries": "grocery", "haircut": "personal_care", "hobbies": "hobby", "home_improve": "home", "household": "home",
           "internet": "telecom", "investing": "savings", "kids_activities": "kids", "kids_stuff": "kids", "lodging": "travel",
           "medical": "medical", "mortgage": "housing", "natgas": "utility", "news": "subscription", "office": "business",
           "parking": "parking", "personal_care": "personal_care", "pets": "pets", "pharmacy": "pharmacy", "phone": "telecom",
           "rent": "housing", "rental_car": "travel", "renters_ins": "insurance", "rideshare": "transit", "savings": "savings",
           "school": "kids", "software": "subscription", "streaming": "subscription", "student_loan": "loan", "tax_refund": "taxes",
           "transit": "transit", "trash": "utility", "vet": "pets", "vision": "medical", "water": "utility", "wedding": "wedding",
           "income": "income", "salary": "income", "payroll": "income"}
V2_KIND = {"airline": "travel", "atm": "fees", "auto": "auto", "auto_ins": "insurance", "baby": "kids", "bakery": "coffee",
           "bar": "alcohol", "beauty": "personal_care", "bigbox": "general", "biz": "business", "books": "hobby",
           "car_payment": "loan", "carrental": "travel", "cc_pay": "fees", "charity": "charity", "childcare": "kids", "church": "charity",
           "clothing": "clothing", "coffee": "coffee", "convenience": "convenience", "delivery": "delivery", "dental": "medical",
           "dept": "clothing", "dollar": "general", "electric": "utility", "electronics": "electronics", "entertainment": "entertainment",
           "fastfood": "fast_food", "fee": "fees", "florist": "gifts", "furniture": "home", "games": "entertainment", "gas": "fuel",
           "gas_util": "utility", "gig_pay": "income", "gov_income": "income", "grocery": "grocery", "gym": "fitness",
           "health_ins": "insurance", "hoa": "housing", "hobby": "hobby", "home_improve": "home", "home_ins": "insurance",
           "hotel": "travel", "interest": "income", "internet": "telecom", "invest_xfer": "savings", "kids_activity": "kids",
           "laundry": "laundry", "life_ins": "insurance", "liquor": "alcohol", "medical": "medical", "mortgage": "housing",
           "online": "general", "p2p": "p2p", "parking": "parking", "payroll": "income", "pet": "pets", "pharmacy": "pharmacy",
           "phone": "telecom", "post": "business", "rent": "housing", "restaurant": "restaurant", "rideshare": "transit",
           "running": "fitness", "salon": "personal_care", "savings_xfer": "savings", "school": "kids", "software": "subscription",
           "streaming": "subscription", "student_loan": "loan", "tax": "taxes", "transit": "transit", "vet": "pets",
           "warehouse": "general", "water": "utility", "wedding": "wedding"}
V2_CONCEPT = {"alcohol": "alcohol", "auto_ins": "insurance", "baby": "kids", "books": "hobby", "business": "business",
              "car_maint": "auto", "car_payment": "loan", "cash": "fees", "cc_payment": "fees", "charity": "charity",
              "childcare": "kids", "church": "charity", "clothing": "clothing", "coffee": "coffee", "dining": "restaurant",
              "electric": "utility", "electronics": "electronics", "entertainment": "entertainment", "fees": "fees", "fuel": "fuel",
              "games": "entertainment", "gas_util": "utility", "gifts": "gifts", "gig_income": "income", "groceries": "grocery",
              "gym": "fitness", "health_ins": "insurance", "hoa": "housing", "hobbies": "hobby", "home_improve": "home",
              "home_ins": "insurance", "household": "home", "income": "income", "internet": "telecom", "investing": "savings",
              "kids": "kids", "life_ins": "insurance", "medical": "medical", "misc": None, "mortgage": "housing", "parking": "parking",
              "personal_care": "personal_care", "pets": "pets", "pharmacy": "pharmacy", "phone": "telecom", "rent": "housing",
              "rideshare": "transit", "running": "fitness", "savings": "savings", "school": "kids", "snacks": "convenience",
              "software": "subscription", "streaming": "subscription", "student_loan": "loan", "takeout": "delivery",
              "taxes": "taxes", "transit": "transit", "travel": "travel", "water": "utility", "wedding": "wedding"}
REAL6_STD = {"Groceries": "grocery", "Restaurants": "restaurant", "Gas & Auto": "fuel", "Clothing": "clothing",
             "Electronics": "electronics", "Home Improvement": "home", "Pharmacy & Health": "pharmacy", "Entertainment": "entertainment",
             "Travel": "travel", "Pets": "pets", "Fitness": "fitness", "Telecom & Utilities": "telecom"}
KINDS_MOD = {"income": "income", "rent": "housing", "insurance": "insurance", "loan": "loan", "subscription": "subscription",
             "savings": "savings", "personal_care": "personal_care", "gifts": "gifts", "donations": "charity", "childcare": "kids",
             "education": "education", "fees": "fees", "cash": "fees"}


def label(key):
    """The canonical text for a key of CANON (None -> None)."""
    return CANON.get(key) if key else None


def category_kind(keys):
    """A category's canonical meaning from the canonical keys of its purposes: one kind, SEVERAL, or None when it holds nothing."""
    ks = {k for k in keys if k}
    if not ks:
        return None
    return CANON[next(iter(ks))] if len(ks) == 1 else SEVERAL
