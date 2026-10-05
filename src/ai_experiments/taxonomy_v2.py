"""A two-level merchant taxonomy (PLAN step 173; owner, 2026-10-03: "Our canonical 12 should be broadened to fit a larger set"; "If you
look at an mcc database, it will be pretty clear that there are many types of merchants").

  fine    the merchant types of the MCC list (data/external/mcc_codes, Unlicense; ~300 generic codes, and the 683 brand codes 3000-3999
          collapsed to airline / car rental / hotel)
  kind    ~60 kinds of spending, `ai_experiments.canon`'s 41 broadened where the MCC list shows whole families the 41 lump or miss
          (liquor apart from bars; airline / hotel / car rental / cruise apart; home-improvement stores, furniture and home goods, and
          home services and trades apart; jewellery, second-hand and pawn, tobacco, digital goods and games, lottery and gambling, fines and
          court costs, legal and accounting, shipping and postage, funeral, car purchase)

Mappings: MCC code -> kind (`mcc_kind`), Overture basic_category -> kind (`overture_kind`), an old canonical kind -> kinds
(`from_canon41`). Kinds with no merchant code (food delivery has MCC 5812/5814; person-to-person, income, savings, loan payments,
"several kinds", "a person, trip or purpose") are kinds of category, not of merchant.
"""
import csv
from functools import lru_cache

from ai_experiments import overture_canon as OC
from ai_experiments.paths import ROOT

KINDS = {
    # food and drink
    "grocery": "grocery store", "restaurant": "restaurant", "fast_food": "fast food", "coffee_bakery": "coffee shop or bakery",
    "food_delivery": "food delivery", "bar": "bar or nightlife", "liquor": "liquor store", "convenience": "convenience store",
    # car
    "gas": "gas station", "car_repair": "car repair, parts or car wash", "car_purchase": "car, boat or motorcycle purchase",
    "loan": "car or student loan payment", "parking": "parking or tolls", "transit": "public transit, taxi or rideshare",
    # travel
    "airline": "airline", "hotel": "hotel or lodging", "car_rental": "car rental", "travel_agency": "cruise, tour or travel agency",
    # home and bills
    "utility": "utility (electric, gas, water, trash)", "phone": "phone, internet or cable", "rent": "rent, mortgage or HOA",
    "home_improvement": "home improvement or garden store", "furniture": "furniture, appliances or home goods",
    "home_services": "home services and trades (cleaning, repairs, movers, storage)", "insurance": "insurance",
    # health and personal
    "medical": "doctor, dentist, hospital or optician", "pharmacy": "pharmacy", "fitness": "gym, fitness or sports club",
    "personal_care": "salon, barber, spa or cosmetics", "laundry": "laundry or dry cleaning",
    # subscriptions and digital
    "subscription": "streaming, software or subscription", "digital": "apps, games or digital purchases",
    # shopping
    "big_box": "big-box, online or general store", "clothing": "clothing, shoes or department store", "jewelry": "jewellery or accessories",
    "electronics": "electronics or computers", "hobby": "books, hobby, music, toys or sports store", "gifts": "gifts, cards or florist",
    "secondhand": "second-hand, antiques or pawn", "tobacco": "tobacco or vape", "office": "office or school supplies",
    # fun
    "entertainment": "entertainment (movies, events, attractions)", "gambling": "lottery, betting or casino",
    # family and pets
    "kids": "childcare, school or kids' activities", "education": "education or tuition", "pet": "pet store or vet",
    # money and government
    "bank": "bank fee, cash withdrawal or card payment", "savings": "savings or investment", "p2p": "person-to-person payment",
    "income": "employer, payroll or other income", "taxes": "taxes or government", "fines": "fines, court costs or bail",
    "legal": "legal, accounting or tax preparation",
    # giving, business, life events
    "charity": "charity, church or membership organisation", "business": "business services or supplies",
    "shipping": "shipping or postage", "events": "wedding, party or photography vendor", "funeral": "funeral services",
    # categories that are not one kind
    "several": "several kinds of spending", "purpose": "a person, trip or purpose (not a kind of business)",
}

_MCC = {  # generic codes; the brand ranges are in mcc_kind
    "0742": "pet", "0763": "business", "0780": "home_services",
    **{c: "home_services" for c in ("1520", "1711", "1731", "1740", "1750", "1761", "1771", "1799", "2842", "4214", "4225", "5996", "7217",
                                    "7342", "7349", "7393", "7622", "7623", "7629", "7692", "7699")},
    "2741": "business", "2791": "business",
    **{c: "transit" for c in ("4011", "4111", "4112", "4121", "4131", "4789")},
    "4119": "medical", "4215": "shipping", "4411": "travel_agency", "4457": "entertainment", "4468": "entertainment",
    "4511": "airline", "4582": "airline", "4722": "travel_agency", "4723": "travel_agency", "4784": "parking",
    **{c: "phone" for c in ("4812", "4814", "4815", "4816", "4821", "4899")},
    "4829": "bank", "4900": "utility", "5983": "utility",
    "5013": "car_repair", "5021": "furniture", "5039": "home_improvement", "5044": "office", "5045": "electronics",
    **{c: "business" for c in ("5046", "5047", "5051", "5065", "5085", "5099", "5131", "5137", "5139", "5169", "5199", "5978", "7311", "7332",
                               "7333", "7338", "7339", "7361", "7372", "7392", "7394", "7399", "7829", "8734", "8911", "8999")},
    "5072": "home_improvement", "5074": "home_improvement", "5094": "jewelry", "5111": "office", "5122": "pharmacy", "5172": "gas",
    "5192": "hobby", "5193": "gifts", "5198": "home_improvement",
    **{c: "home_improvement" for c in ("5200", "5211", "5231", "5251", "5261", "5271", "5998")},
    **{c: "big_box" for c in ("5300", "5309", "5310", "5331", "5399", "5961", "5963", "5964", "5965", "5966", "5967", "5969", "5999", "7278")},
    "5311": "clothing",
    **{c: "grocery" for c in ("5411", "5422", "5441", "5451")},
    "5462": "coffee_bakery", "5499": "convenience",
    **{c: "car_purchase" for c in ("5511", "5521", "5551", "5561", "5571", "5592", "5598", "5599")},
    **{c: "car_repair" for c in ("5531", "5532", "5533", "7531", "7534", "7535", "7538", "7542", "7549", "8675")},
    "5541": "gas", "5542": "gas", "7511": "gas",
    **{c: "clothing" for c in ("5611", "5621", "5631", "5641", "5651", "5655", "5661", "5681", "5691", "5697", "5698", "5699", "7251", "7296")},
    **{c: "furniture" for c in ("5712", "5713", "5714", "5718", "5719", "5722", "5950", "7641")},
    "5732": "electronics", "5734": "electronics", "5997": "electronics", "7379": "electronics",
    **{c: "hobby" for c in ("5733", "5735", "5940", "5941", "5942", "5945", "5946", "5949", "5970", "5971", "5972", "5994", "7395")},
    "5811": "events", "7221": "events", "5812": "restaurant", "5813": "bar", "5814": "fast_food",
    "5815": "digital", "5816": "digital", "5817": "digital", "5818": "digital", "7993": "digital",
    **{c: "secondhand" for c in ("5832", "5931", "5932", "5933", "5935", "5937")},
    "5912": "pharmacy", "5921": "liquor", "5943": "office", "5944": "jewelry", "5948": "jewelry", "7631": "jewelry",
    "5947": "gifts", "5973": "gifts", "5992": "gifts",
    "5960": "insurance", "6300": "insurance", "6381": "insurance", "6399": "insurance",
    "5962": "travel_agency", "5968": "subscription", "7273": "subscription", "7375": "subscription",
    "5975": "medical", "5976": "medical", "7277": "medical",
    "5977": "personal_care", "7230": "personal_care", "7297": "personal_care", "7298": "personal_care", "7299": "personal_care",
    "5993": "tobacco", "5995": "pet",
    "6010": "bank", "6011": "bank", "6012": "bank", "7321": "bank", "6051": "savings", "6211": "savings",
    "6513": "rent", "7011": "hotel", "7012": "hotel", "7033": "hotel", "7032": "kids",
    "7210": "laundry", "7211": "laundry", "7216": "laundry", "7261": "funeral", "7276": "legal",
    "7512": "car_rental", "7513": "car_rental", "7519": "car_rental", "7523": "parking",
    **{c: "gambling" for c in ("7800", "7801", "7802", "7995")},
    **{c: "entertainment" for c in ("7832", "7841", "7911", "7922", "7929", "7932", "7933", "7941", "7991", "7992", "7994", "7996", "7998", "7999")},
    "7997": "fitness",
    **{c: "medical" for c in ("8011", "8021", "8031", "8041", "8042", "8043", "8044", "8049", "8050", "8062", "8071", "8099")},
    "8111": "legal", "8931": "legal",
    "8211": "kids", "8351": "kids",
    **{c: "education" for c in ("8220", "8241", "8244", "8249", "8299")},
    **{c: "charity" for c in ("8398", "8641", "8651", "8661", "8699")},
    "9211": "fines", "9222": "fines", "9223": "fines", "9311": "taxes", "9399": "taxes", "9405": "taxes", "9402": "shipping",
}


def mcc_kind(code):
    """MCC code (str or int) -> kind id, or None (network-internal codes 9700-9950)."""
    c = f"{int(code):04d}"
    n = int(c)
    if 3000 <= n <= 3299:
        return "airline"
    if 3300 <= n <= 3499:
        return "car_rental"
    if 3500 <= n <= 3999:
        return "hotel"
    return _MCC.get(c)


@lru_cache(maxsize=None)
def mcc_types():
    """[(code, description, kind)] for the generic codes (the fine layer), from the bundled list."""
    rows = csv.DictReader(open(ROOT / "data" / "external" / "mcc_codes" / "mcc_codes.csv", encoding="utf-8"))
    return [(r["mcc"], r["edited_description"], mcc_kind(r["mcc"])) for r in rows if not 3000 <= int(r["mcc"]) <= 3999]


_CANON41 = {  # old canonical kind -> new kind(s)
    "bar or liquor store": ["bar", "liquor"], "car repair or parts": ["car_repair"], "parking or tolls": ["parking"],
    "public transit or rideshare": ["transit"], "airline, hotel or car rental": ["airline", "hotel", "car_rental"],
    "phone or internet provider": ["phone"], "rent, mortgage or HOA": ["rent"], "doctor, dentist, hospital or optician": ["medical"],
    "gym or fitness": ["fitness"], "salon, barber or personal care": ["personal_care"], "streaming, software or subscription": ["subscription"],
    "entertainment (movies, events, games)": ["entertainment", "digital"], "books, hobby or sports store": ["hobby"],
    "big-box or online store (sells many kinds of things)": ["big_box"], "clothing or department store": ["clothing"],
    "home improvement, furniture or household": ["home_improvement", "furniture", "home_services"], "electronics store": ["electronics"],
    "pet store or vet": ["pet"], "childcare, school or kids' activities": ["kids"], "education or tuition": ["education"],
    "gifts or florist": ["gifts"], "charity or church": ["charity"], "employer, payroll or other income": ["income"],
    "savings or investment transfer": ["savings"], "bank fee, cash withdrawal or card payment": ["bank"],
    "person-to-person payment (purpose varies)": ["p2p"], "taxes or government": ["taxes", "fines"],
    "business services or supplies": ["business", "office", "shipping", "legal"], "laundry or dry cleaning": ["laundry"],
    "wedding or event vendor": ["events"], "grocery store": ["grocery"], "restaurant": ["restaurant"], "fast food": ["fast_food"],
    "coffee shop or bakery": ["coffee_bakery"], "food delivery": ["food_delivery"], "gas station": ["gas"], "convenience store": ["convenience"],
    "car or student loan payment": ["loan"], "utility (electric, gas, water, trash)": ["utility"], "insurance": ["insurance"],
    "pharmacy": ["pharmacy"], "several kinds of spending": ["several"], "a person, trip or purpose (not a kind of business)": ["purpose"],
}


def from_canon41(name):
    return _CANON41.get(name, [])


_OVERTURE = {  # where the new kinds split an old one; everything else follows overture_canon through from_canon41's first kind
    "bar": "bar", "lounge": "bar", "brewery": "bar", "winery": "bar", "distillery": "bar", "alcoholic_beverage_venue": "bar", "dance_club": "bar",
    "nightlife_venue": "bar", "hotel": "hotel", "lodging": "hotel", "resort": "hotel", "bed_and_breakfast": "hotel", "inn": "hotel",
    "private_lodging": "hotel", "campground": "hotel", "rv_park": "hotel", "airport": "parking", "air_transport_facility_or_service": "parking", "rental_service": "car_rental",
    "travel_service": "travel_agency", "travel_and_transportation": "travel_agency",
    "hardware_home_and_garden_store": "home_improvement", "home_service": "home_services", "building_or_construction_service": "home_services",
    "storage_facility": "home_services", "design_service": "furniture", "security_service": "home_services",
    "environmental_or_ecological_service": "home_services", "agricultural_service": "home_services",
    "second_hand_store": "secondhand", "specialty_store": "big_box", "attorney_or_law_firm": "legal", "legal_service": "legal",
    "shipping_or_delivery_service": "shipping", "office_supply_store": "office", "auto_dealer": "car_purchase", "vehicle_dealer": "car_purchase",
    "courthouse": "fines", "casino": "gambling", "gaming_venue": "gambling", "arcade": "entertainment", "sport_or_recreation_club": "fitness",
    "country_club": "fitness", "personal_care_and_beauty_store": "personal_care",
}


def overture_kind(basic_category):
    if basic_category in _OVERTURE:
        return _OVERTURE[basic_category]
    old = OC.kind(basic_category)
    ks = from_canon41(old) if old else []
    return ks[0] if ks else None


assert all(v in KINDS for v in _MCC.values()) and all(v in KINDS for v in _OVERTURE.values())
assert all(k in KINDS for ks in _CANON41.values() for k in ks)
