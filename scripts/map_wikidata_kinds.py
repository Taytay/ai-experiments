"""PLAN step 179: map Wikidata merchants (build_wikidata_merchants.py) to taxonomy_v2 kinds by keyword rules over the English description,
products, industry and class labels (first match in RULES order wins); entities no rule covers are left for a model pass
(label_wikidata_kinds.py). Writes data/external/wikidata_merchants/merchants_v1.jsonl: one entity per line with "kind" (or null) and
"kind_source" ("rule" / null), deduplicated by QID.
usage: uv run python scripts/map_wikidata_kinds.py
"""
import glob
import json
import re
from collections import Counter

from ai_experiments.paths import ROOT

D = ROOT / "data" / "external" / "wikidata_merchants"
# (kind, regex over "description | products | industry | classes", lower-cased); order matters: specific before general
RULES = [
    ("medical", r"contact lens|optic|optometr|eyewear|eyeglass|dental|dentist|orthodont|hospital|clinic|physician|medical (practice|center|group)|urgent care|health (system|care provider)|laborator(y|ies) (testing|services)|hearing aid"),
    ("pharmacy", r"pharmac|drugstore|drug store|chemist"),
    ("pet", r"\bpet\b|pets\b|veterinar|animal hospital|pet food"),
    ("airline", r"airline|air carrier|\bair lines\b"),
    ("hotel", r"hotel|motel|resort|lodging|hospitality company|vacation rental|inn chain"),
    ("car_rental", r"car rental|vehicle rental|truck rental|rent-a-car|car sharing"),
    ("transit", r"ride-?hailing|ridesharing|taxi|transit (agency|authority)|bus (company|operator)|railroad|passenger rail|scooter-sharing|bike-sharing"),
    ("travel_agency", r"travel agency|online travel|cruise line|tour operator"),
    ("food_delivery", r"food delivery|meal delivery|meal kit|grocery delivery|online food ordering"),
    ("fast_food", r"fast[- ]food|quick service|pizza (chain|delivery|restaurant)|burger chain|sandwich chain|fast casual"),
    ("coffee_bakery", r"coffee|café|cafe\b|bakery|bakeries|doughnut|donut|bagel|tea house|teahouse"),
    ("bar", r"\bbar\b|pub\b|brewery|brewpub|nightclub|tavern|winery|distillery"),
    ("restaurant", r"restaurant|steakhouse|diner\b|eatery|bistro|buffet|grill chain"),
    ("liquor", r"liquor|wine (retailer|shop|store)|alcohol retail"),
    ("grocery", r"supermarket|grocery|groceries|food retail|hypermarket|natural foods|warehouse club"),
    ("convenience", r"convenience store|gas station chain|truck stop"),
    ("gas", r"filling station|gas station|fuel station|petroleum retail|ev charging|charging network"),
    ("car_repair", r"auto parts|automotive (aftermarket|parts|repair|service)|tire|car wash|oil change|collision repair|auto repair"),
    ("car_purchase", r"car dealer|automobile dealer|auto dealer|automobile manufacturer|car manufacturer|motorcycle manufacturer|vehicle manufacturer"),
    ("parking", r"parking|toll road|tolling"),
    ("utility", r"electric utility|power company|utility company|water utility|natural gas (utility|distribution)|waste management|electricity (supplier|retailer)"),
    ("phone", r"telecommunication|wireless (carrier|provider)|mobile (network|phone operator|virtual network)|internet service provider|\bisp\b|cable (television|provider|company)|broadband"),
    ("insurance", r"insurance|insurer|underwriter"),
    ("bank", r"\bbank\b|banking|credit union|financial services company|payment (processor|service|company)|credit card|fintech|brokerage"),
    ("savings", r"investment (firm|company|management)|asset management|robo-advisor|cryptocurrency exchange|retirement"),
    ("subscription", r"streaming|video on demand|music service|subscription|software as a service|\bsaas\b|cloud storage|web service|password manager|online (newspaper|magazine)|news website|dating (app|service|website)|vpn"),
    ("digital", r"video game|game developer|game publisher|app store|digital distribution|mobile game|online game"),
    ("electronics", r"consumer electronics|electronics (retail|store|chain)|computer (retail|store|hardware)|camera|smartphone manufacturer"),
    ("home_improvement", r"home improvement|hardware store|building materials|garden cent|lumber|paint (retail|company)"),
    ("furniture", r"furniture|home furnishings|mattress|appliance|homeware|kitchenware|home decor|housewares"),
    ("home_services", r"pest control|cleaning service|moving company|self[- ]storage|home security|plumbing|hvac|landscap|lawn care|home services|locksmith|roofing"),
    ("jewelry", r"jewel|watchmaker|watch (brand|retailer)|luxury goods"),
    ("clothing", r"clothing|apparel|fashion|footwear|shoe|department store|outlet|lingerie|eyewear brand"),
    ("personal_care", r"cosmetic|beauty|salon|barber|spa\b|skincare|skin care|fragrance|nail|hair care|tanning|massage"),
    ("fitness", r"gym|fitness|health club|yoga|pilates|martial arts|climbing|sports club|golf club"),
    ("hobby", r"bookstore|book retail|books\b|toy|hobby|craft|sporting goods|outdoor (retail|equipment)|musical instrument|bicycle|music store|comic|fabric"),
    ("gifts", r"florist|flower|gift (shop|retailer|card)|greeting card|party supplies"),
    ("secondhand", r"thrift|second[- ]hand|consignment|pawn|antique|resale|used goods"),
    ("tobacco", r"tobacco|vape|vaping|e-cigarette|cigar"),
    ("office", r"office supplies|stationery|office products"),
    ("shipping", r"courier|postal|parcel|shipping (company|service)|package delivery|freight"),
    ("gambling", r"casino|lottery|betting|sportsbook|gambling"),
    ("entertainment", r"cinema|movie theat|theater chain|amusement park|theme park|zoo|aquarium|museum|ticket(ing)?\b|concert|bowling|escape room|arcade|trampoline|water park|entertainment venue"),
    ("kids", r"child ?care|daycare|preschool|summer camp|kids' activit|tutoring"),
    ("education", r"university|college|school|online learning|e-learning|education (company|provider)|test preparation"),
    ("charity", r"charit|nonprofit|non-profit|foundation\b|church|religious|ministry|food bank"),
    ("taxes", r"tax (agency|authority)|government agency|department of motor vehicles|\bdmv\b"),
    ("legal", r"law firm|legal services|accounting firm|tax preparation|accountan"),
    ("funeral", r"funeral|cemetery|crematori"),
    ("events", r"wedding|photograph|event planning|catering"),
    ("laundry", r"laundr|dry clean"),
    ("big_box", r"online (retailer|shop|store|marketplace)|e-commerce|ecommerce|marketplace|retail chain|discount store|variety store|dollar store|general store|retailer"),
    ("business", r"manufacturer|wholesale|consulting|staffing|logistics|b2b|business services|software company|technology company"),
]
RX = [(k, re.compile(r)) for k, r in RULES]


def kind_of(e):
    text = " | ".join([e.get("description", "")] + e.get("products", []) + e.get("industry", []) + e.get("classes", [])).lower()
    for k, rx in RX:
        if rx.search(text):
            return k
    return None


if __name__ == "__main__":
    ents = {}
    for f in sorted(glob.glob(str(D / "raw_*.jsonl"))):
        for line in open(f):
            e = json.loads(line)
            ents.setdefault(e["qid"], e)
    out, c = [], Counter()
    for e in ents.values():
        k = kind_of(e)
        e["kind"], e["kind_source"] = k, ("rule" if k else None)
        c[k] += 1; out.append(e)
    with open(D / "merchants_v1.jsonl", "w") as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    n = len(out)
    print(f"{n} entities; mapped by rule {100 * (n - c[None]) / n:.1f}%; unmapped {c[None]}")
    print(", ".join(f"{k} {v}" for k, v in c.most_common() if k))
    for q in ("Q4545758",):
        e = ents.get(q); print("check:", e["label"], "->", e["kind"]) if e else None
