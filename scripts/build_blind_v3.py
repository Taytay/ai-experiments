#!/usr/bin/env python3
"""Row 145 (main session, 2026-09-30): history rows show a renamed or merged category under its current name, as YNAB does
(the generator as written showed the old name, a name not in the list). Otherwise the blind agent's script unchanged.

Blind test set v3 for a per-user bank-transaction categoriser.

Standard library only, deterministic (one seeded random.Random), no network.
Writes data/processed/blind_v3.json relative to the repo root.

Emphases of this set: households sharing one account whose two card holders file the
same payee differently; families filing by who the spending was for; small-business
owners mixing personal and business spending; students and retirees; seasonal
categories (Christmas, back to school, trips) created mid-year; renames and merges of
categories; subscriptions that change price; statement strings that change over time
(processor switches, rebrands, billing portals); regional chains in five US regions and
invented local businesses shared by users of the same town; P2P payments with and
without counterparty; refunds; occasional filing mistakes in the history; brand-new
users and brand-new payees.
"""

import datetime as dt
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 3030925
N_USERS = 250
Q_PER_USER = 6
MAX_SHOTS = 24
Y0 = dt.date(2025, 1, 1)
Y1 = dt.date(2025, 12, 31)
WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

rng = random.Random(SEED)


def pick(seq):
    return seq[rng.randrange(len(seq))]


def wpick(items, weights):
    return rng.choices(items, weights=weights, k=1)[0]


def wsample(items, weights, n):
    items, weights = list(items), list(weights)
    out = []
    while items and len(out) < n:
        i = rng.choices(range(len(items)), weights=weights, k=1)[0]
        out.append(items.pop(i))
        weights.pop(i)
    return out


def poisson(lam):
    if lam <= 0:
        return 0
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def rdate(a, b):
    if b < a:
        return a
    return a + dt.timedelta(days=rng.randint(0, (b - a).days))


def month_end(y, m):
    return (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1)).day


def digits(n):
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


def alnum(n):
    return "".join(pick("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(n))


# --------------------------------------------------------------------------------------
# Geography
# --------------------------------------------------------------------------------------
TOWNS = [
    ("Rochester", "NY", "ne", ["Henrietta", "Pittsford", "Greece", "Webster"], "Monroe"),
    ("Pittsburgh", "PA", "ne", ["Bethel Park", "Monroeville", "Cranberry Twp", "Mt Lebanon"], "Allegheny"),
    ("Worcester", "MA", "ne", ["Shrewsbury", "Auburn", "Holden"], "Worcester"),
    ("Cherry Hill", "NJ", "ne", ["Voorhees", "Haddonfield", "Marlton"], "Camden"),
    ("Marietta", "GA", "so", ["Kennesaw", "Smyrna", "Atlanta", "Roswell"], "Cobb"),
    ("Tampa", "FL", "so", ["Brandon", "Riverview", "Clearwater", "Wesley Chapel"], "Hillsborough"),
    ("Charlotte", "NC", "so", ["Matthews", "Huntersville", "Concord", "Pineville"], "Mecklenburg"),
    ("Knoxville", "TN", "so", ["Farragut", "Maryville", "Powell"], "Knox"),
    ("Austin", "TX", "tx", ["Round Rock", "Pflugerville", "Cedar Park", "Buda"], "Travis"),
    ("San Antonio", "TX", "tx", ["Schertz", "Helotes", "Converse", "Live Oak"], "Bexar"),
    ("Plano", "TX", "tx", ["Frisco", "Allen", "Richardson", "McKinney"], "Collin"),
    ("Columbus", "OH", "mw", ["Dublin", "Westerville", "Hilliard", "Grove City"], "Franklin"),
    ("Des Moines", "IA", "mw", ["West Des Moines", "Ankeny", "Urbandale", "Clive"], "Polk"),
    ("Madison", "WI", "mw", ["Middleton", "Fitchburg", "Sun Prairie", "Verona"], "Dane"),
    ("Grand Rapids", "MI", "mw", ["Wyoming", "Kentwood", "Grandville", "Walker"], "Kent"),
    ("Denver", "CO", "we", ["Aurora", "Lakewood", "Englewood", "Littleton"], "Denver"),
    ("Portland", "OR", "we", ["Beaverton", "Gresham", "Tigard", "Hillsboro"], "Multnomah"),
    ("Sacramento", "CA", "we", ["Elk Grove", "Roseville", "Folsom", "Rancho Cordova"], "Sacramento"),
    ("Mesa", "AZ", "we", ["Gilbert", "Chandler", "Tempe", "Phoenix"], "Maricopa"),
    ("Boise", "ID", "we", ["Meridian", "Nampa", "Eagle"], "Ada"),
]
TOWN = {t[0]: dict(name=t[0], state=t[1], region=t[2], subs=t[3], county=t[4]) for t in TOWNS}
STATE_NAME = {"NY": "New York", "PA": "Pennsylvania", "MA": "Massachusetts", "NJ": "New Jersey",
              "GA": "Georgia", "FL": "Florida", "NC": "North Carolina", "TN": "Tennessee",
              "TX": "Texas", "OH": "Ohio", "IA": "Iowa", "WI": "Wisconsin", "MI": "Michigan",
              "CO": "Colorado", "OR": "Oregon", "CA": "California", "AZ": "Arizona", "ID": "Idaho"}

# --------------------------------------------------------------------------------------
# Merchant universe: name|concepts|descriptor(s; ';' separates variants)|lo|hi|flags|where
# flags: s store  o online  r monthly  a ACH bill  y annual  g giftable  b back-to-school
#        k subscription (price events)  i income
# where: US, region (ne so tx mw we), state (NY ...), @Town
# --------------------------------------------------------------------------------------
MERCHANT_TABLE = """
Kroger|groceries|KROGER|12|220|s|so,mw,tx
Aldi|groceries|ALDI|12|140|s|US
Trader Joe's|groceries|TRADER JOE S|10|160|s|US
Whole Foods Market|groceries|WHOLEFDS MKT|10|200|s|US
Sprouts Farmers Market|groceries|SPROUTS FARMERS MAR|10|150|s|so,tx,we
Safeway|groceries|SAFEWAY|10|200|s|we
Albertsons|groceries|ALBERTSONS|10|180|s|we,tx
Fred Meyer|groceries:6,household:3,clothing:1|FRED-MEYER|15|250|s|OR,ID
WinCo Foods|groceries|WINCO FOODS|20|220|s|we
Raley's|groceries|RALEYS|10|160|s|CA
King Soopers|groceries|KING SOOPERS|10|200|s|CO
Fry's Food|groceries|FRYS-FOOD-DRG|10|200|s|AZ
Wegmans|groceries|WEGMANS|15|250|s|ne
ShopRite|groceries|SHOPRITE|15|220|s|NJ,NY,PA
Stop & Shop|groceries|STOP & SHOP|10|200|s|MA,NY,NJ
Hannaford|groceries|HANNAFORD|10|180|s|MA,NY
Price Chopper|groceries|PRICE CHOPPER|10|180|s|NY,MA
Market Basket|groceries|MARKET BASKET|15|200|s|MA
Tops Markets|groceries|TOPS MARKETS|10|160|s|NY,PA
Giant Eagle|groceries|GIANT EAGLE|10|200|s|PA,OH
Publix|groceries|PUBLIX SUPER MAR|10|220|s|so
Food Lion|groceries|FOOD LION|10|160|s|NC,TN,GA
Harris Teeter|groceries|HARRIS TEETER|10|200|s|NC,GA
Ingles|groceries|INGLES MARKETS|10|160|s|NC,TN,GA
Winn-Dixie|groceries|WINN-DIXIE|10|160|s|FL,GA
Piggly Wiggly|groceries|PIGGLY WIGGLY|10|120|s|GA,TN,WI
H-E-B|groceries|H-E-B|15|250|s|tx
Randalls|groceries|RANDALLS|10|180|s|@Austin
Tom Thumb|groceries|TOM THUMB|10|180|s|@Plano
Central Market|groceries|CENTRAL MARKET|15|200|s|tx
Meijer|groceries:6,household:3,clothing:1|MEIJER|15|250|s|OH,MI,WI
Hy-Vee|groceries|HY-VEE|10|220|s|IA,WI
Jewel-Osco|groceries|JEWEL OSCO|10|200|s|IA,WI
Schnucks|groceries|SCHNUCKS|10|180|s|WI
Fareway|groceries|FAREWAY STORES|10|160|s|IA
Woodman's|groceries|WOODMANS FOOD MKT|20|220|s|WI
Instacart|groceries|INSTACART;INSTACART*{ref}|25|200|o|US
Walmart|groceries:4,household:4,clothing:1,electronics:1|WM SUPERCENTER;WAL-MART;WALMART.COM|8|260|sgb|US
Target|household:5,groceries:2,clothing:2,gifts:1|TARGET;TARGET.COM *|6|220|sgb|US
Costco|groceries:5,household:4,electronics:1|COSTCO WHSE|30|450|sg|US
Sam's Club|groceries:5,household:5|SAMSCLUB|25|350|s|so,tx,mw
BJ's Wholesale Club|groceries:5,household:5|BJS WHOLESALE|25|350|s|ne,FL,GA,NC
Amazon|household:4,electronics:1,books:1,gifts:1,kids_stuff:1,clothing:1|AMZN MKTP US*{ref};AMAZON MKTPL*{ref};Amazon.com*{ref}|8|160|ogb|US
Dollar Tree|household|DOLLAR TREE|3|40|s|US
Dollar General|household:6,groceries:4|DOLLAR GENERAL|4|60|s|US
Five Below|household:5,gifts:5|FIVE BELOW|5|60|sgb|US
McDonald's|fast_food|MCDONALD'S F|4|25|s|US
Starbucks|coffee|STARBUCKS STORE|3|20|s|US
Dunkin'|coffee|DUNKIN|3|18|s|ne,so,mw
Dutch Bros|coffee|DUTCH BROS|4|15|s|we,tx
Peet's Coffee|coffee|PEETS|4|15|s|CA,OR
Tim Hortons|coffee|TIM HORTONS|3|12|s|NY,MI,OH
Chick-fil-A|fast_food|CHICK-FIL-A|6|35|s|US
Chipotle|fast_food|CHIPOTLE|9|40|s|US
Taco Bell|fast_food|TACO BELL|5|25|s|US
Wendy's|fast_food|WENDYS|5|25|s|US
Burger King|fast_food|BURGER KING|5|25|s|US
Subway|fast_food|SUBWAY|6|25|s|US
Panera Bread|fast_food|PANERA BREAD|8|40|s|US
Five Guys|fast_food|FIVE GUYS|10|45|s|US
Panda Express|fast_food|PANDA EXPRESS|8|35|s|US
Jersey Mike's|fast_food|JERSEY MIKES|8|35|s|US
Jimmy John's|fast_food|JIMMY JOHNS|8|30|s|US
Popeyes|fast_food|POPEYES|6|30|s|US
Sonic Drive-In|fast_food|SONIC DRIVE IN|5|25|s|so,tx,mw
Arby's|fast_food|ARBYS|5|25|s|US
Domino's|takeout|DOMINO'S|10|45|s|US
Pizza Hut|takeout|PIZZA HUT|12|45|s|US
Papa John's|takeout|PAPA JOHNS|12|45|s|US
Whataburger|fast_food|WHATABURGER|7|30|s|tx,FL,GA
In-N-Out Burger|fast_food|IN N OUT BURGER|6|30|s|CA,AZ,CO,OR,ID,tx
Culver's|fast_food|CULVERS OF|7|35|s|mw
Skyline Chili|fast_food|SKYLINE CHILI|8|30|s|OH
Bojangles|fast_food|BOJANGLES|6|25|s|NC,GA,TN
Zaxby's|fast_food|ZAXBYS|8|30|s|GA,FL,NC,TN
Cook Out|fast_food|COOK OUT|5|20|s|NC,TN,GA
Waffle House|dining|WAFFLE HOUSE|8|35|s|so,tx
Jack in the Box|fast_food|JACK IN THE BOX|5|25|s|we,tx
Del Taco|fast_food|DEL TACO|5|20|s|CA,AZ
White Castle|fast_food|WHITE CASTLE|5|25|s|OH,MI,NY,NJ
Portillo's|fast_food|PORTILLOS|8|40|s|WI,IA,MI
Wawa|fast_food:5,gas:5|WAWA|4|60|s|NJ,PA,FL
Sheetz|fast_food:5,gas:5|SHEETZ|4|60|s|PA,OH,NC
Casey's|gas:6,fast_food:4|CASEYS|5|65|s|IA,WI,OH
Kwik Trip|gas:6,fast_food:4|KWIK TRIP|4|65|s|WI,IA
QuikTrip|gas:7,fast_food:3|QT|5|65|s|GA,TX,NC,AZ
RaceTrac|gas|RACETRAC|20|65|s|FL,GA,TN,TX
Buc-ee's|gas:6,fast_food:2,gifts:2|BUC-EE'S|15|120|s|TX,GA,FL,TN
Stewart's Shops|gas:5,fast_food:5|STEWARTS SHOP|3|55|s|NY
Maverik|gas|MAVERIK|20|70|s|ID,CO,AZ
Olive Garden|dining|OLIVE GARDEN|25|110|s|US
Applebee's|dining|APPLEBEES|20|90|s|US
Chili's|dining|CHILIS|20|90|s|US
Texas Roadhouse|dining|TEXAS ROADHOUSE|30|140|s|US
Red Robin|dining|RED ROBIN|25|100|s|US
Cracker Barrel|dining|CRACKER BARREL|20|90|s|US
Outback Steakhouse|dining|OUTBACK|30|140|s|US
Buffalo Wild Wings|dining|BUFFALO WILD WINGS|20|100|s|US
IHOP|dining|IHOP|15|70|s|US
Denny's|dining|DENNYS|15|60|s|US
DoorDash|takeout|DD *DOORDASH {dd};DOORDASH*{dd}|15|80|o|US
Uber Eats|takeout|UBER *EATS;UBER EATS|15|70|o|US
Grubhub|takeout|GRUBHUB*{dd}|15|70|o|US
Shell|gas|SHELL OIL {n9}|20|85|s|US
ExxonMobil|gas|EXXONMOBIL|20|85|s|US
Chevron|gas|CHEVRON|20|90|s|we,tx,so
BP|gas|BP#|20|80|s|US
Speedway|gas|SPEEDWAY|20|80|s|mw,ne,so
Circle K|gas|CIRCLE K|20|80|s|US
Sunoco|gas|SUNOCO|20|80|s|ne
Marathon|gas|MARATHON PETRO|20|80|s|mw
Valero|gas|VALERO|20|80|s|tx,so,we
Arco|gas|ARCO|20|80|s|CA,OR,AZ
Costco Gas|gas|COSTCO GAS|25|80|s|US
Jiffy Lube|auto|JIFFY LUBE|45|120|s|US
Valvoline Instant Oil Change|auto|VIOC|45|130|s|US
Firestone Complete Auto Care|auto|FIRESTONE|60|900|s|US
Discount Tire|auto|DISCOUNT TIRE|20|900|s|we,tx,so,mw
Les Schwab Tires|auto|LES SCHWAB|20|900|s|OR,ID
AutoZone|auto|AUTOZONE|8|200|s|US
O'Reilly Auto Parts|auto|O'REILLY|8|200|s|US
Advance Auto Parts|auto|ADVANCE AUTO PARTS|8|200|s|ne,so,mw
Take 5 Oil Change|auto|TAKE 5 OIL|40|100|s|US
Mister Car Wash|auto|MISTER CAR WASH|20|40|rk|US
Toyota Financial Services|car_payment|TOYOTA FINANCIAL|380|620|ra|US
Honda Financial Services|car_payment|HONDA PMT|320|560|ra|US
Ally Auto|car_payment|ALLY AUTO|300|700|ra|US
ParkMobile|parking|PARKMOBILE|2|20|o|US
E-ZPass|tolls|E-ZPASS REPLENISH;EZPASS REBILL|25|35|o|NY,NJ,PA,MA
SunPass|tolls|SUNPASS*ACC{n9}|10|25|o|FL
NTTA|tolls|NTTA AUTOCHARGE|10|40|o|@Plano
Uber|rideshare|UBER *TRIP;UBER   *TRIP HELP.UBER.COM|8|55|o|US
Lyft|rideshare|LYFT   *RIDE;LYFT *{n4} RIDE|8|50|o|US
Amtrak|travel|AMTRAK .CO{n9}|40|260|o|US
Delta Air Lines|travel|DELTA AIR {n9}|150|700|o|US
United Airlines|travel|UNITED {n9}|150|700|o|US
Southwest Airlines|travel|SOUTHWES {n9}|90|500|o|US
American Airlines|travel|AMERICAN AIR{n9}|150|700|o|US
Marriott|travel|MARRIOTT|120|800|s|US
Hilton|travel|HILTON HOTELS|120|700|s|US
Airbnb|travel|AIRBNB * HM{ref}|150|1400|o|US
Expedia|travel|EXPEDIA {n9}|100|900|o|US
Hertz|travel|HERTZ RENT-A-CAR|100|500|s|US
Carnival Cruise Line|travel|CARNIVAL CRUISE LINE|300|2500|o|US
RG&E|electric|RG&E|70|220|ra|NY
Duquesne Light|electric|DUQUESNE LIGHT|60|200|ra|PA
Peoples Gas|natgas|PEOPLES GAS|30|180|ra|PA
Eversource|electric|EVERSOURCE|70|260|ra|MA
National Grid|natgas|NATIONAL GRID|30|200|ra|MA,NY
PSE&G|electric|PSE&G|70|260|ra|NJ
Georgia Power|electric|GEORGIA POWER|80|300|ra|GA
Gas South|natgas|GAS SOUTH|25|150|ra|GA
Tampa Electric|electric|TECO TAMPA ELECTRIC|90|300|ra|FL
Duke Energy|electric|DUKE ENERGY|70|280|ra|NC
Piedmont Natural Gas|natgas|PIEDMONT N G|25|150|ra|NC,TN
Knoxville Utilities Board|electric|KUB|80|260|ra|TN
Austin Energy|electric|CITY OF AUSTIN UTIL|80|320|ra|@Austin
CPS Energy|electric|CPS ENERGY|80|320|ra|@San Antonio
TXU Energy|electric|TXU ENERGY|90|330|ra|@Plano
Atmos Energy|natgas|ATMOS ENERGY|25|120|ra|@Plano,@San Antonio
Texas Gas Service|natgas|TEXAS GAS SERVICE|25|120|ra|@Austin
AEP Ohio|electric|AEP OHIO|70|250|ra|OH
Columbia Gas of Ohio|natgas|COLUMBIA GAS OH|30|180|ra|OH
MidAmerican Energy|electric|MIDAMERICAN ENERGY|60|220|ra|IA
Madison Gas and Electric|electric|MGE|60|220|ra|WI
Consumers Energy|electric|CONSUMERS ENERGY|60|240|ra|MI
Xcel Energy|electric|XCEL ENERGY|60|220|ra|CO
Portland General Electric|electric|PORTLAND GENERAL|60|200|ra|OR
NW Natural|natgas|NW NATURAL|30|150|ra|OR
SMUD|electric|SMUD|70|300|ra|CA
PG&E|natgas|PGANDE WEB ONLINE|30|180|ra|CA
SRP|electric|SRP|80|350|ra|AZ
Southwest Gas|natgas|SOUTHWEST GAS|20|120|ra|AZ
Idaho Power|electric|IDAHO POWER|50|200|ra|ID
Intermountain Gas|natgas|INTERMOUNTAIN GAS|25|140|ra|ID
Waste Management|trash|WM EZPAY|30|60|ra|US
Republic Services|trash|REPUBLIC SERVICES|30|60|ra|US
Xfinity|internet|COMCAST CABLE COMM|60|140|ra|US
Spectrum|internet|SPECTRUM|60|130|ra|US
Cox Communications|internet|COX COMM|60|130|ra|AZ,GA,FL
Verizon Fios|internet|VERIZON*FIOS|60|140|ra|ne
Google Fiber|internet|GOOGLE *FIBER|70|100|r|@Austin,@San Antonio
AT&T|phone|ATT*BILL PAYMENT|60|220|ra|US
Verizon Wireless|phone|VZWRLSS*APOCC VISB|60|250|ra|US
T-Mobile|phone|T-MOBILE*AUTO PAY|50|220|ra|US
Mint Mobile|phone|MINT MOBILE|15|30|rk|US
Visible|phone|VISIBLE|25|45|rk|US
Netflix|streaming|NETFLIX.COM|7.99|24.99|rok|US
Hulu|streaming|HULU {n9} HULU.COM/BILL|7.99|18.99|rok|US
Disney+|streaming|DISNEY PLUS|9.99|15.99|rok|US
HBO Max|streaming|MAX.COM|9.99|20.99|rok|US
Peacock|streaming|PEACOCK {ref}|7.99|13.99|rok|US
Paramount+|streaming|PARAMOUNT+|7.99|12.99|rok|US
YouTube Premium|streaming|GOOGLE *YOUTUBEPREMIUM|13.99|22.99|rok|US
YouTube TV|streaming|GOOGLE *YOUTUBE TV|82.99|82.99|rok|US
Spotify|music|SPOTIFY USA;SPOTIFY P{ref}|11.99|19.99|rok|US
SiriusXM|music|SIRIUSXM|10.99|22.99|rok|US
Apple|cloud:4,music:3,software:3|APPLE.COM/BILL|0.99|16.99|rok|US
Audible|books|AUDIBLE*{ref}|14.95|14.95|rok|US
Kindle Unlimited|books|KINDLE SVCS*{ref}|11.99|11.99|rok|US
Amazon Prime|subs|AMAZON PRIME*{ref};PRIME VIDEO CHANNELS|14.99|14.99|rok|US
Microsoft 365|software|MSFT *E0400{ref}|9.99|12.99|rok|US
Google One|cloud|GOOGLE *GOOGLE ONE|1.99|9.99|rok|US
Dropbox|cloud|DROPBOX*{ref}|11.99|19.99|rok|US
ChatGPT|software|OPENAI *CHATGPT SUBSCR|20|20|rok|US
Duolingo|education|DUOLINGO|12.99|12.99|rok|US
Planet Fitness|gym|PLANET FITNESS|10|24.99|rk|US
LA Fitness|gym|LA FITNESS|34.99|49.99|rk|US
Orangetheory Fitness|gym|ORANGETHEORY|59|169|rk|US
Anytime Fitness|gym|ANYTIME FITNESS|35|55|rk|US
Peloton|gym|PELOTON* MEMBERSHIP|24|44|rok|US
Crunch Fitness|gym|CRUNCH FITNESS|12|30|rk|US
Life Time|gym|LIFE TIME|99|229|rk|mw,tx,CO,AZ
State Farm|insurance_auto|STATE FARM RO 27|80|260|ra|US
GEICO|insurance_auto|GEICO *AUTO|70|240|ra|US
Progressive|insurance_auto|PROGRESSIVE *INSURANCE|70|240|ra|US
Allstate|insurance_auto|ALLSTATE INS CO|80|250|ra|US
Lemonade|insurance_home|LEMONADE INSURANCE|10|40|rok|US
Liberty Mutual|insurance_home|LIBERTY MUTUAL|60|200|ra|US
CVS Pharmacy|pharmacy:6,household:4|CVS/PHARMACY|5|90|s|US
Walgreens|pharmacy:6,household:4|WALGREENS|5|90|s|US
Rite Aid|pharmacy|RITE AID|5|70|s|NY,PA,NJ,OR,CA
Quest Diagnostics|medical|QUEST DIAGNOSTICS|20|300|o|US
LabCorp|medical|LABCORP|20|300|o|US
CareCredit|medical|SYNCHRONY CARECREDIT|50|300|ra|US
UnitedHealthcare|health_ins|UHC PREMIUM|200|700|ra|US
Kaiser Permanente|medical|KAISER|20|300|s|CA,OR,CO,GA
Warby Parker|vision|WARBY PARKER|95|300|o|US
LensCrafters|vision|LENSCRAFTERS|100|450|s|US
GoodRx|pharmacy|GOODRX|5|60|o|US
Aspen Dental|dental|ASPEN DENTAL|90|900|s|US
Chewy|pet|CHEWY.COM|25|160|o|US
Petco|pet|PETCO|10|120|s|US
PetSmart|pet|PETSMART INC|10|120|s|US
Banfield Pet Hospital|pet|BANFIELD PET HOSP|40|400|s|US
BarkBox|pet|BARKBOX|29|35|rok|US
Rover|pet|ROVER.COM|30|250|o|US
Tractor Supply|pet:5,home_improvement:5|TRACTOR SUPPLY|15|200|s|US
Old Navy|clothing|OLD NAVY|15|120|sb|US
Gap|clothing|GAP|20|150|s|US
TJ Maxx|clothing:6,household:4|TJMAXX|10|150|sg|US
Marshalls|clothing:6,household:4|MARSHALLS|10|150|sg|US
Ross|clothing:6,household:4|ROSS STORES|10|120|s|US
Kohl's|clothing:7,household:3|KOHLS|15|180|sgb|US
Macy's|clothing:8,gifts:2|MACYS|20|250|sg|US
Nordstrom Rack|clothing|NORDSTROM RACK|20|250|s|US
Nike|clothing|NIKE.COM|40|200|ob|US
Carter's|kids_stuff|CARTERS|15|120|sb|US
The Children's Place|kids_stuff|THE CHILDRENS PLACE|15|120|sb|US
Shein|clothing|SHEIN.COM|15|120|o|US
Temu|household|TEMU.COM|8|80|o|US
Etsy|gifts:5,hobby:3,household:2|ETSY.COM - {shop}|10|120|og|US
eBay|hobby:5,electronics:5|EBAY O*{n9}|10|200|o|US
Ulta Beauty|personal_care|ULTA|10|120|sg|US
Sephora|personal_care|SEPHORA|15|150|sg|US
Bath & Body Works|personal_care:6,gifts:4|BATH AND BODY WORKS|10|80|sg|US
Great Clips|personal_care|GREAT CLIPS|18|35|s|US
Supercuts|personal_care|SUPERCUTS|20|40|s|US
Sport Clips|personal_care|SPORT CLIPS|22|40|s|US
Best Buy|electronics:8,gifts:2|BEST BUY|15|900|sg|US
Apple Store|electronics|APPLE STORE R|30|1500|s|US
GameStop|hobby:6,gifts:4|GAMESTOP|10|80|sg|US
Steam|hobby|STEAMGAMES.COM {n9}|5|60|o|US
PlayStation Network|hobby|PLAYSTATION NETWORK|10|70|o|US
Nintendo|hobby|NINTENDO CD{n9}|10|70|o|US
Barnes & Noble|books:7,gifts:3|BARNES & NOBLE|10|80|sg|US
Michaels|hobby:8,household:2|MICHAELS STORES|8|120|sg|US
Hobby Lobby|hobby:7,household:3|HOBBY-LOBBY|8|150|sg|US
Joann|hobby|JOANN STORES|8|120|s|US
IKEA|furniture|IKEA|20|700|s|US
Wayfair|furniture|WAYFAIR*{ref}|30|900|o|US
HomeGoods|household:7,gifts:3|HOMEGOODS|15|200|sg|US
The Home Depot|home_improvement|THE HOME DEPOT|8|600|s|US
Lowe's|home_improvement|LOWES|8|600|s|US
Menards|home_improvement|MENARDS|8|500|s|mw
Ace Hardware|home_improvement|ACE HARDWARE|5|120|s|US
Harbor Freight Tools|home_improvement|HARBOR FREIGHT TOOLS|10|200|s|US
Spirit Halloween|holidays|SPIRIT HALLOWEEN|20|90|s|US
Party City|holidays:5,gifts:5|PARTY CITY|10|80|s|US
Hallmark|gifts|HALLMARK|5|40|s|US
1-800-Flowers|gifts|1-800-FLOWERS.COM|40|120|o|US
AMC Theatres|entertainment|AMC {n4} ONLINE|12|60|o|US
Regal Cinemas|entertainment|REGAL CINEMAS|12|60|s|US
Cinemark|entertainment|CINEMARK THEATRES|12|60|s|tx,we,mw
Ticketmaster|entertainment|TICKETMASTER|40|400|o|US
StubHub|entertainment|STUBHUB|40|400|o|US
Eventbrite|entertainment|EB *{event}|10|120|o|US
Dave & Buster's|entertainment|DAVE & BUSTERS|20|120|s|US
Topgolf|entertainment|TOPGOLF|30|150|s|US
Fandango|entertainment|FANDANGO|12|50|o|US
MySchoolBucks|school_lunch|MYSCHOOLBUCKS.COM|20|60|o|US
KinderCare|childcare|KINDERCARE|250|450|ra|US
Bright Horizons|childcare|BRIGHT HORIZONS|800|2000|ra|US
Chegg|textbooks|CHEGG ORDER|15|20|rok|US
Barnes & Noble College|textbooks|BN COLLEGE|20|400|s|US
Coursera|education|COURSERA|49|59|rok|US
Pearson|textbooks|PEARSON EDUCATION|40|120|o|US
Nelnet|student_loan|NELNET|150|450|ra|US
MOHELA|student_loan|MOHELA|150|450|ra|US
Rocket Mortgage|mortgage|ROCKET MORTGAGE|1200|2800|ra|US
Mr. Cooper|mortgage|MR COOPER|1200|2800|ra|US
Wells Fargo Home Mortgage|mortgage|WF HOME MTG|1200|2800|ra|US
ADT|home_services|ADT SECURITY|40|65|ra|US
TurboTax|taxes|INTUIT *TURBOTAX|60|180|o|US
H&R Block|taxes|H&R BLOCK|90|350|s|US
IRS|est_taxes|IRS USATAXPYMT|200|3000|a|US
American Red Cross|charity|AMERICAN RED CROSS|25|100|o|US
St. Jude Children's Research Hospital|charity|ST JUDE CHILDRENS|19|50|rok|US
Wikimedia|charity|WIKIMEDIA|3|25|o|US
GoFundMe|charity:6,gifts:4|GOFUNDME {gfm}|20|100|o|US
Staples|office|STAPLES|5|250|sb|US
Office Depot|office|OFFICE DEPOT|5|250|s|US
USPS|shipping|USPS PO {n9}|5|80|s|US
The UPS Store|shipping|THE UPS STORE|8|120|s|US
FedEx Office|shipping|FEDEX OFFIC|8|120|s|US
Pirate Ship|shipping|PIRATE SHIP|5|120|o|US
Uline|biz_supplies|ULINE *SHIP SUPPLIES|40|400|o|US
Printful|biz_supplies|PRINTFUL INC|20|300|o|US
Vistaprint|office|VISTAPRINT|20|120|o|US
Shopify|biz_software|SHOPIFY* {n9}|29|79|rok|US
Squarespace|biz_software|SQSP* INV{n9}|16|49|rok|US
GoDaddy|biz_software|DNH*GODADDY.COM|12|30|o|US
Google Workspace|biz_software|GOOGLE *GSUITE_{shop}|7.2|14.4|rok|US
Zoom|biz_software|ZOOM.US 888-799-9666|15.99|15.99|rok|US
QuickBooks|biz_software|INTUIT *QBOOKS ONLINE|35|90|rok|US
Mailchimp|biz_software|MAILCHIMP *MONTHLY|13|45|rok|US
Adobe|biz_software|ADOBE *CREATIVE CLD|22.99|59.99|rok|US
Canva|biz_software|CANVA* {n9}|14.99|14.99|rok|US
LinkedIn Premium|biz_software|LINKEDIN P{n9}|39.99|39.99|rok|US
Meta Ads|advertising|FACEBK *{ref}|20|300|o|US
Google Ads|advertising|GOOGLE *ADS{n9}|20|300|o|US
Upwork|advertising|UPWORK -{n9}REF|10|100|o|US
Social Security|income_gov|SSA TREAS 310 XXSOC SEC|1400|2600|i|US
IRS Tax Refund|tax_refund|IRS TREAS 310 TAX REF|300|3500|i|US
Stripe|income_biz|STRIPE TRANSFER ST-{ref}|100|2500|i|US
Etsy Payments|income_biz|ETSY INC. DEPOSIT|100|1800|i|US
Square Deposits|income_biz|SQUARE INC {n9}|100|2500|i|US
Shopify Payments|income_biz|SHOPIFY PAYMENTS {n9}|100|2500|i|US
Uber Driver Pay|income_biz|UBER USA 6787 EDI PYMNTS|150|900|i|US
DoorDash Dasher Pay|income_biz|DOORDASH INC PAYROLL|100|600|i|US
"""


def parse_where(w):
    return [x.strip() for x in w.split(",")]


M = []  # global merchant list


def add_merchant(**kw):
    m = dict(name=kw["name"], concepts=kw["concepts"], descs=kw["descs"], lo=kw["lo"], hi=kw["hi"],
             flags=kw.get("flags", ""), where=kw.get("where", ["US"]), local=kw.get("local", False),
             town=kw.get("town"), opaque=kw.get("opaque", False), p2p=kw.get("p2p"),
             kid=kw.get("kid", False), owner=kw.get("owner"))
    m["id"] = len(M)
    tot = sum(w for _, w in m["concepts"])
    m["cw"] = {c: w / tot for c, w in m["concepts"]}
    m["primary"] = max(m["concepts"], key=lambda x: x[1])[0]
    m["multi"] = len(m["concepts"]) > 1
    m["sfmt"] = rng.randrange(4)
    m["tail"] = pick(["CA", "WA", "NY", "DE", "TX", "MN", "GA", "IL"])
    m["versions"] = [(Y0 - dt.timedelta(days=3650), m["descs"], True)]  # (from, descs, alias_known)
    m["clean"] = kw.get("clean", m["name"])
    M.append(m)
    return m


for line in MERCHANT_TABLE.strip().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    name, conc, desc, lo, hi, flags, where = [x.strip() for x in line.split("|")]
    cw = []
    for part in conc.split(","):
        if ":" in part:
            c, w = part.split(":")
            cw.append((c, float(w)))
        else:
            cw.append((part, 1.0))
    add_merchant(name=name, concepts=cw, descs=desc.split(";"), lo=float(lo), hi=float(hi),
                 flags=flags, where=parse_where(where))

BY_NAME = {m["name"]: m for m in M}

# statement-string changes over time (processor switches, rebrands, portals)
NATIONAL_CHANGES = [
    ("HBO Max", dt.date(2025, 7, 9), ["HBOMAX.COM", "HBO MAX"], True),
    ("Planet Fitness", dt.date(2025, 5, 1), ["ABC*PLANET FITNESS"], True),
    ("Xfinity", dt.date(2025, 6, 1), ["COMCAST XFINITY"], True),
    ("Verizon Wireless", dt.date(2025, 8, 12), ["VERIZON WIRELESS PAYMENTS"], True),
    ("Peloton", dt.date(2025, 4, 1), ["PELOTON*APP"], True),
    ("SiriusXM", dt.date(2025, 3, 1), ["SXM*SIRIUSXM.COM/ACCT"], True),
    ("Chewy", dt.date(2025, 9, 1), ["CHEWY INC"], True),
    ("Anytime Fitness", dt.date(2025, 7, 1), ["ABC*AF CLUB FEES"], False),
    ("Crunch Fitness", dt.date(2025, 6, 15), ["CLUBREADY*CF{n4}"], False),
    ("Mint Mobile", dt.date(2025, 8, 1), ["T-MOBILE MINT"], True),
    ("Duolingo", dt.date(2025, 9, 10), ["APPLE.COM/BILL"], False),
    ("Lemonade", dt.date(2025, 7, 20), ["LMND INS {n9}"], False),
]
for nm, d, descs, known in NATIONAL_CHANGES:
    BY_NAME[nm]["versions"].append((d, descs, known))

# global price changes: merchant -> (date, pct)
PRICE_EVENTS = {"Netflix": (dt.date(2025, 1, 21), 0.16), "Peacock": (dt.date(2025, 7, 17), 0.37),
                "Disney+": (dt.date(2025, 10, 21), 0.19), "HBO Max": (dt.date(2025, 10, 21), 0.10),
                "Spotify": (dt.date(2025, 9, 1), 0.09), "SiriusXM": (dt.date(2025, 6, 1), 0.08),
                "Planet Fitness": (dt.date(2025, 6, 15), 0.45), "YouTube TV": (dt.date(2025, 1, 13), 0.1),
                "Mister Car Wash": (dt.date(2025, 8, 1), 0.15), "Paramount+": (dt.date(2025, 8, 20), 0.25),
                "LA Fitness": (dt.date(2025, 9, 1), 0.12), "Orangetheory Fitness": (dt.date(2025, 7, 1), 0.08),
                "Chegg": (dt.date(2025, 8, 1), 0.2), "Adobe": (dt.date(2025, 6, 17), 0.15),
                "Microsoft 365": (dt.date(2025, 2, 1), 0.3)}

# --------------------------------------------------------------------------------------
# Invented local businesses per town
# --------------------------------------------------------------------------------------
SURN = ["Nguyen", "Patel", "Kowalski", "Garcia", "O'Brien", "Haddad", "Lindqvist", "Okafor", "Moreno",
        "Castillo", "Brennan", "Tanaka", "Delgado", "Schultz", "Romano", "Kim", "Novak", "Fitzgerald",
        "Abernathy", "Whitaker", "Ramos", "Sandoval", "Mercer", "Lawson", "Yoder", "Dubois", "Park",
        "Chen", "Rossi", "Hale", "Barnett", "Ortiz", "Sullivan", "Petrov", "Mahoney", "Vance"]
ADJ = ["Copper", "Blue Door", "Golden", "Maple", "Iron", "Silver", "Red Barn", "Lucky", "Wild",
       "Old Mill", "Riverbend", "Northside", "Cedar", "Harbor", "Sunny", "Stone", "Hilltop", "Prairie",
       "Juniper", "Blackbird", "Honeybee", "Lantern", "Magnolia", "Aspen", "Pine", "Bluebird",
       "Tall Oak", "Brightwater", "Fox Hollow", "Willow"]
NOUN = ["Kettle", "Spoon", "Fork", "Table", "Oven", "Hearth", "Anchor", "Acorn", "Sparrow", "Fox",
        "Owl", "Otter", "Pepper", "Olive", "Barrel", "Crow", "Thistle", "Lark"]
STREETS = ["Main St", "Elm St", "Oak Ave", "Park Ave", "2nd St", "Market St", "Grand Ave", "Lake St",
           "Mill Rd", "Church St", "Broad St", "College Ave"]
FIRST = ["Maria", "Jake", "Priya", "Luis", "Kate", "Ben", "Rosa", "Tom", "Ana", "Marcus", "Hannah",
         "Carlos", "Mei", "Derek", "Olivia", "Sean", "Fatima", "Tyler", "Grace", "Andre", "Nina", "Kyle",
         "Leah", "Omar", "Brooke", "Victor", "Chloe", "Ethan"]
SAINTS = ["Mary", "Joseph", "Anne", "Patrick", "Francis", "Luke", "Therese", "Michael"]
VIET = ["Saigon", "Hanoi", "Bistro", "Dakao", "Pasteur"]
SPAN = ["El Sol", "La Esquina", "Los Compadres", "El Rey", "La Palma"]
ASIAN = ["Seoul", "H Mart", "Lotus", "Asia", "Mekong"]

LOCAL_SPEC = {
    # concept: (count, templates, processor prefixes)
    "dining": (7, ["{S}'s Diner", "{A} {N} Kitchen", "{T} Taproom & Grill", "Pho {V}", "Taqueria {Sp}",
                   "{S} Family Pizza", "Golden Dragon", "Sakura Sushi", "{St} Bistro", "{A} Bagel Co",
                   "Bangkok Garden", "{S}'s BBQ", "Mama {Fn}'s Italian", "{A} Pancake House"],
               ["SQ *", "TST* ", "CLV*", "", "", "TST* "]),
    "coffee": (3, ["{A} Bean Coffee", "{St} Roasters", "Cafe {Fn}", "{N} & {N} Coffee"], ["SQ *", "TST* ", ""]),
    "groceries": (2, ["{T} Food Co-op", "{S} Farm Market", "La {Sp} Supermercado", "{As} Market",
                      "{S}'s Butcher Shop"], ["", "", "SQ *"]),
    "alcohol": (2, ["{T} Wine & Spirits", "{A} Brewing Co", "{S}'s Liquor"], ["", "SQ *", "TST* "]),
    "auto": (3, ["{St} Auto Repair", "{S} Tire & Service", "{T} Collision Center", "{A} Car Wash"], [""]),
    "medical": (3, ["{T} Family Medicine", "{A} Urgent Care", "{A} Physical Therapy", "{T} Dermatology"],
                ["", "", "PAYMENTUS*"]),
    "dental": (2, ["{St} Dental", "{S} Orthodontics", "{A} Smiles Dentistry"], ["", ""]),
    "vision": (1, ["{T} Eye Care", "{S} Optometry"], [""]),
    "pet": (2, ["{A} Paws Vet Clinic", "{T} Animal Hospital", "{A} Dog Grooming"], ["", "SQ *"]),
    "personal_care": (3, ["{Fn}'s Nails", "{A} Salon", "{S} Barber Co", "{A} Day Spa"], ["SQ *", "", "GLOSSGENIUS*"]),
    "gym": (2, ["{T} Family YMCA", "{A} CrossFit", "{A} Yoga Studio", "{A} Climbing Gym"], ["", "DAXKO*", "MINDBODY*"]),
    "kids_act": (4, ["{A} Dance Academy", "{T} Youth Soccer", "{A} Swim School", "{A} Gymnastics",
                     "Kumon of {Sub}", "{S} Piano Studio", "{T} Little League"], ["", "SQ *", "JACKRABBIT*"]),
    "childcare": (2, ["Little {N}s Learning Center", "{A} Montessori", "{T} Kids Academy"], ["", "PROCARE*"]),
    "church": (3, ["St. {Sa} Parish", "{A} Community Church", "First Baptist {T}", "{T} Unitarian Church"],
               ["VANCO*", "PUSHPAY*", "TITHE.LY*"]),
    "hobby": (2, ["{A} Yarn Shop", "{T} Hobby & Games", "{A} Books", "{A} Guitar Shop", "{T} Comics"], ["SQ *", ""]),
    "home_services": (3, ["{S} Lawn Care", "{A} Pest Control", "{S} Plumbing & Heating", "{T} Window Cleaning"], ["", "SQ *"]),
    "entertainment": (2, ["{T} Escape Rooms", "{A} Bowl", "{A} Cinema", "{T} Symphony"], ["", "SQ *"]),
    "clothing": (1, ["{A} Boutique", "{S} Thrift"], ["SQ *", ""]),
    "gifts": (1, ["{A} Flowers", "{T} Gift Shop"], ["SQ *", ""]),
    "water": (1, ["City of {T} Utilities"], [""]),
    "rent": (4, ["{A} Apartments", "{S} Property Management", "{A} Commons", "{St} Lofts"], ["", "WEB PMTS "]),
    "property_tax": (1, ["{C} County Treasurer"], [""]),
    "auto_reg": (1, ["{SN} DMV"], [""]),
    "education": (1, ["{T} State University"], [""]),
    "income": (3, ["{C} County Schools", "{T} Regional Medical Center", "City of {T}"], [""]),
}
OPAQUE_LOCAL = ["{I} Enterprises LLC", "{I} Holdings", "{I} Mgmt Group", "{S} & {S2} LLC", "{I} Services Inc",
                "{S} Partners"]
ACH_LOCAL = {"water", "rent", "income"}


def fill(tpl, town, used):
    for _ in range(20):
        s = tpl.format(S=pick(SURN), S2=pick(SURN), A=pick(ADJ), N=pick(NOUN), T=town["name"], V=pick(VIET),
                       Sp=pick(SPAN), St=pick(STREETS), Fn=pick(FIRST), Sa=pick(SAINTS), As=pick(ASIAN),
                       Sub=pick(town["subs"]), C=town["county"], SN=STATE_NAME[town["state"]],
                       I=alnum(3).replace("0", "K").replace("1", "M"))
        if s not in used:
            used.add(s)
            return s
    return s + " " + alnum(2)


def local_desc(name, prefix):
    base = name.upper().replace("'", pick(["", " "]))
    base = base.replace(" & ", pick([" & ", " AND ", " "]))
    return (prefix + base)[:24].rstrip()


for tname in sorted(TOWN):
    town = TOWN[tname]
    used = set()
    for concept in LOCAL_SPEC:
        cnt, tpls, prefs = LOCAL_SPEC[concept]
        for i in range(cnt):
            name = fill(tpls[i % len(tpls)] if i < len(tpls) else pick(tpls), town, used)
            pref = pick(prefs)
            lo, hi = {"dining": (12, 90), "coffee": (4, 16), "groceries": (8, 90), "alcohol": (12, 80),
                      "auto": (40, 900), "medical": (25, 300), "dental": (40, 600), "vision": (60, 350),
                      "pet": (30, 400), "personal_care": (20, 120), "gym": (25, 150), "kids_act": (60, 220),
                      "childcare": (600, 1600), "church": (25, 200), "hobby": (8, 80),
                      "home_services": (45, 250), "entertainment": (15, 90), "clothing": (15, 90),
                      "gifts": (20, 80), "water": (40, 130), "rent": (900, 2400), "property_tax": (900, 4500),
                      "auto_reg": (40, 180), "education": (2500, 6500), "income": (900, 2600)}[concept]
            flags = "s"
            if concept in ("water", "rent"):
                flags = "ra"
            if concept == "income":
                flags = "i"
            kid = concept == "medical" and i == 1 and rng.random() < 0.5
            if kid:
                name = fill("{A} Pediatrics", town, used)
            m = add_merchant(name=name, concepts=[(concept, 1.0)], descs=[local_desc(name, pref)], lo=lo, hi=hi,
                             flags=flags, where=["@" + tname], local=True, town=tname, kid=kid)
            # processor switches (reader can still tell) and portal changes (reader cannot)
            r = rng.random()
            if pref in ("SQ *", "TST* ", "CLV*") and r < 0.3:
                newp = pick([p for p in ("SQ *", "TST* ", "CLV*", "") if p != pref])
                m["versions"].append((rdate(dt.date(2025, 3, 1), dt.date(2025, 10, 15)),
                                      [local_desc(name, newp)], True))
            elif concept in ("rent", "water", "gym", "kids_act") and r < 0.25:
                initials = "".join(w[0] for w in name.upper().split() if w[0].isalpha())[:4]
                portal = pick(["RENTCAFE*", "APPFOLIO*", "PAYMENTUS*", "DAXKO*", "BILLMATRIX*", "PAYLEASE*"])
                m["versions"].append((rdate(dt.date(2025, 3, 1), dt.date(2025, 10, 15)),
                                      [portal + initials + " " + digits(4)], False))
    for i in range(5):
        concept = pick(["dining", "auto", "home_services", "medical", "personal_care", "hobby", "dining"])
        name = fill(pick(OPAQUE_LOCAL), town, used)
        pref = pick(["", "SQ *", "TST* ", "CKO*"])
        lo, hi = {"dining": (12, 90), "auto": (40, 700), "home_services": (60, 300), "medical": (30, 300),
                  "personal_care": (20, 100), "hobby": (10, 90)}[concept]
        add_merchant(name=name, concepts=[(concept, 1.0)], descs=[local_desc(name, pref)], lo=lo, hi=hi,
                     flags="s", where=["@" + tname], local=True, town=tname, opaque=True)

EMPLOYERS = ["Northwind Logistics", "Summit Health", "Brightline Dental", "Acme Tool & Die", "Cobalt Software",
             "Riverbend Hospital", "Heartland Insurance", "Pioneer Credit Union", "Evergreen Senior Living",
             "Blue Ridge Engineering", "Crescent Analytics", "Harborview Bank", "Granite Construction Co",
             "Keystone Staffing", "Meridian Labs", "Orchard Foods", "Polar Freight", "Quill Publishing",
             "Redwood Retail Group", "Sterling Aerospace", "Tidewater Energy", "Union Home Health"]
for e in EMPLOYERS:
    via = rng.random()
    if via < 0.2:
        descs = ["GUSTO PAY " + digits(6)]
    elif via < 0.35:
        descs = ["ADP PAYROLL " + e.upper().split()[0]]
    else:
        descs = [e.upper().replace("&", "AND")[:16].rstrip() + " PAYROLL"]
    add_merchant(name=e, concepts=[("income", 1.0)], descs=descs, lo=900, hi=3200, flags="i", where=["US"])
PENSIONS = [("State Teachers Retirement", "STRS PENSION"), ("Public Employees Retirement", "PERS BENEFIT PMT"),
            ("Teamsters Pension Fund", "TEAMSTERS PENSION"), ("Fidelity RMD", "FIDELITY INVESTMENTS DIST")]
for n, d in PENSIONS:
    add_merchant(name=n, concepts=[("pension", 1.0)], descs=[d], lo=600, hi=2500, flags="i", where=["US"])
INTEREST_M = add_merchant(name="Interest", concepts=[("interest", 1.0)], descs=["INTEREST PAYMENT"], lo=0.2,
                          hi=40, flags="i", where=["US"])

BY_NAME = {m["name"]: m for m in M}

INCOME_CONCEPTS = {"income", "income_gov", "pension", "interest", "income_family", "income_biz", "tax_refund"}
DD_NAMES = ["CHIPOTLE", "PANDA EXPRESS", "WINGSTOP", "THAI BASIL", "SHAKE SHACK", "SWEETGREEN", "FIREHOUSE",
            "JERSEY MIKES", "POKE BAR", "TACO CABANA"]
SHOPS = ["HANDMADEBYJO", "VINTAGEFINDS", "PRINTSHOPCO", "CRAFTYCORNER", "WOODANDWIRE", "TINYTHREADS"]


def available(m, town):
    if m["local"]:
        return m["town"] == town["name"]
    for w in m["where"]:
        if w == "US" or w == town["region"] or w == town["state"] or w == "@" + town["name"]:
            return True
    return False


BY_CONCEPT = defaultdict(list)
for m in M:
    if m["p2p"] or "i" in m["flags"]:
        continue
    for c, w in m["cw"].items():
        if w >= 0.3:
            BY_CONCEPT[c].append(m["id"])

# --------------------------------------------------------------------------------------
# Category slots: key -> (group, concepts covered, standard names, personal/coined names)
# --------------------------------------------------------------------------------------
SLOTS = {
    "rent": ("bills", "rent", ["Rent"], ["Roof", "Landlord"]),
    "mortgage": ("bills", "mortgage property_tax", ["Mortgage"], ["House payment"]),
    "housing": ("bills", "rent mortgage property_tax insurance_home", ["Housing"], ["Shelter"]),
    "property_tax": ("bills", "property_tax", ["Property Tax"], ["Tax man"]),
    "electric": ("bills", "electric", ["Electric", "Electricity", "Power"], ["Lights"]),
    "natgas": ("bills", "natgas", ["Gas Bill", "Heating", "Natural Gas"], ["Heat"]),
    "water": ("bills", "water trash", ["Water & Trash", "Water/Sewer", "Water"], ["H2O"]),
    "utilities": ("bills", "electric natgas water trash", ["Utilities"], ["Keeping the lights on"]),
    "internet": ("bills", "internet", ["Internet", "WiFi"], ["Tubes"]),
    "phone": ("bills", "phone", ["Phone", "Cell Phone", "Mobile"], ["Pocket computer"]),
    "connect": ("bills", "internet phone streaming", ["Phone/Internet/TV", "Internet & Phone"], ["Screens"]),
    "bills": ("bills", "electric natgas water trash internet phone insurance_auto insurance_home health_ins "
                       "student_loan car_payment subs home_services", ["Bills", "Monthly Bills"],
              ["Adulting", "Boring bills"]),
    "car_ins": ("bills", "insurance_auto", ["Car Insurance", "Auto Insurance"], ["Crash money"]),
    "insurance": ("bills", "insurance_auto insurance_home health_ins", ["Insurance"], ["Just in case"]),
    "renters": ("bills", "insurance_home", ["Renters Insurance", "Home Insurance"], []),
    "car_payment": ("bills", "car_payment", ["Car Payment", "Auto Loan"], ["Car note"]),
    "student_loan": ("bills", "student_loan", ["Student Loans"], ["Sallie"]),
    "childcare": ("bills", "childcare", ["Childcare", "Daycare"], ["Kid wrangling"]),
    "cleaning": ("bills", "cleaning", ["House Cleaning", "Cleaner"], []),
    "taxes": ("bills", "taxes est_taxes property_tax", ["Taxes", "Tax Prep"], ["Uncle Sam"]),
    "groceries": ("everyday", "groceries", ["Groceries", "Grocery", "Food at Home"], ["Food we cook", "Fridge fillers", "Nom"]),
    "food": ("everyday", "groceries dining fast_food takeout coffee alcohol biz_meals", ["Food"], ["Om nom", "Fuel for humans"]),
    "dining": ("everyday", "dining fast_food takeout coffee biz_meals", ["Dining Out", "Restaurants", "Eating Out"],
               ["Food someone else cooks", "Yum out"]),
    "fast_food": ("everyday", "fast_food takeout", ["Fast Food", "Takeout"], ["Drive-thru", "Lazy food"]),
    "coffee": ("everyday", "coffee", ["Coffee", "Coffee Shops"], ["Caffeine", "Bean juice"]),
    "alcohol": ("everyday", "alcohol", ["Alcohol", "Wine & Beer", "Bars"], ["Hooch", "Happy hour"]),
    "gas": ("everyday", "gas biz_gas", ["Gas", "Fuel", "Gas & Fuel"], ["Dino juice", "Vroom juice"]),
    "car": ("everyday", "auto parking tolls auto_reg", ["Auto Maintenance", "Car Repairs", "Car"], ["Car stuff", "Keep it running"]),
    "transport": ("everyday", "gas biz_gas auto parking tolls rideshare auto_reg car_payment insurance_auto",
                  ["Transportation", "Getting Around"], ["Wheels"]),
    "rideshare": ("everyday", "rideshare parking tolls", ["Rideshare & Parking", "Uber/Lyft"], ["Chauffeur"]),
    "household": ("everyday", "household cleaning office subs", ["Household", "Home Supplies", "Household Goods"],
                  ["House stuff", "Nest"]),
    "pharmacy": ("everyday", "pharmacy", ["Pharmacy", "Prescriptions"], ["Pills"]),
    "medical": ("everyday", "medical pharmacy dental vision health_ins", ["Medical", "Healthcare", "Health"],
                ["Doctor stuff", "Body maintenance"]),
    "dental": ("everyday", "dental vision", ["Dental & Vision", "Dentist"], ["Teeth"]),
    "pet": ("everyday", "pet", ["Pets", "Pet Care"], ["Doggo", "Fur baby"]),
    "personal_care": ("everyday", "personal_care", ["Personal Care", "Haircuts", "Beauty"], ["Glow up"]),
    "kids": ("everyday", "kids_stuff kids_act school_lunch childcare", ["Kids", "Children"], ["Mini humans", "Gremlins"]),
    "kids_act": ("everyday", "kids_act", ["Kids Activities", "Activities"], ["Kid shuttle"]),
    "school": ("everyday", "school_lunch education textbooks", ["School", "School Costs"], []),
    "clothing": ("fun", "clothing", ["Clothing", "Clothes"], ["Threads", "Look good"]),
    "shopping": ("fun", "clothing electronics household furniture subs", ["Shopping"], ["Stuff I buy"]),
    "electronics": ("fun", "electronics", ["Electronics", "Tech"], ["Gadgets"]),
    "home": ("fun", "home_improvement furniture home_services", ["Home Improvement", "Home Maintenance", "Home Repair"],
             ["Fixer upper", "House projects"]),
    "streaming": ("fun", "streaming music", ["Streaming", "TV & Music"], ["Couch time"]),
    "subs": ("fun", "streaming music software cloud subs", ["Subscriptions", "Subs"], ["Recurring fun"]),
    "software": ("fun", "software cloud", ["Apps & Software", "Apps"], ["Bits"]),
    "fun": ("fun", "entertainment hobby books streaming music alcohol", ["Fun", "Entertainment", "Fun Money"],
            ["Treat yo self", "Joy"]),
    "hobby": ("fun", "hobby books", ["Hobbies", "Books & Games", "Crafts"], ["Nerd stuff"]),
    "entertainment": ("fun", "entertainment", ["Entertainment", "Going Out"], ["Night out"]),
    "gym": ("fun", "gym", ["Gym", "Fitness"], ["Sweat"]),
    "gifts": ("fun", "gifts holidays grandkids", ["Gifts", "Gifts & Holidays"], ["Presents"]),
    "holidays": ("fun", "holidays", ["Holidays", "Seasonal"], ["Festive"]),
    "charity": ("fun", "charity church", ["Charity", "Giving", "Donations"], ["Do good"]),
    "church": ("fun", "church", ["Church", "Tithe"], ["Offering"]),
    "travel": ("fun", "travel", ["Travel", "Vacation"], ["Adventures", "Wanderlust"]),
    "education": ("fun", "education textbooks", ["Education", "Tuition"], ["Brain food"]),
    "textbooks": ("fun", "textbooks", ["Textbooks", "Books & Supplies"], []),
    "income": ("income", "income income_gov pension interest income_family income_biz tax_refund",
               ["Income", "Paycheck", "Ready to Assign", "Inflow"], ["Money in", "Pay day"]),
    "interest": ("income", "interest", ["Interest"], []),
    "misc": ("misc", "*", ["Misc", "Miscellaneous", "Everything Else", "Other"], ["Random", "Stuff", "Zorp"]),
    "biz_supplies": ("biz", "office biz_supplies", ["Biz - Supplies", "Business Supplies", "Office Supplies"], []),
    "biz_ship": ("biz", "shipping", ["Biz - Shipping", "Postage"], []),
    "biz_soft": ("biz", "biz_software", ["Biz - Software", "Business Software"], []),
    "biz_ads": ("biz", "advertising", ["Biz - Ads", "Marketing"], []),
    "biz_income": ("income", "income_biz", ["Biz - Income", "Business Income", "Sales"], []),
    "biz_meals": ("biz", "biz_meals", ["Biz - Meals", "Client Meals"], []),
    "biz_car": ("biz", "biz_gas", ["Biz - Mileage & Gas", "Work Gas"], []),
    "est_tax": ("biz", "est_taxes", ["Estimated Taxes", "Quarterly Taxes"], []),
    "ss": ("income", "income_gov", ["Social Security"], []),
    "pension": ("income", "pension", ["Pension"], []),
    "medicare": ("bills", "health_ins", ["Medicare & Supplemental", "Health Insurance"], []),
    "grandkids": ("fun", "grandkids gifts kids_stuff", ["Grandkids"], []),
    "garden": ("everyday", "home_improvement home_services", ["Garden & Yard", "Yard"], []),
    "from_parents": ("income", "income_family", ["From Mom & Dad", "Family Help"], []),
    "work_study": ("income", "income", ["Work Study", "Campus Job"], []),
}
OPAQUE_NAMES = {"Zorp", "Bits", "Tubes", "Sallie", "Gremlins"}
FUNDS = ["Emergency Fund", "Vacation Fund", "Car Replacement", "Savings", "Rainy Day", "Big Purchases",
         "New Laptop Fund", "House Down Payment", "Holiday Fund", "Roth IRA", "Wedding Fund", "Annual Bills",
         "Furniture Fund", "Medical Deductible"]
GROUP_RANK = {"income": 0, "bills": 1, "everyday": 2, "person": 3, "fun": 4, "biz": 5, "event": 6, "savings": 7, "misc": 8}

# plausible alternative slots when someone files a concept differently
ALT = {
    "coffee": ["dining", "fun", "groceries", "food"], "fast_food": ["dining", "fun", "food"],
    "takeout": ["dining", "groceries", "fun"], "alcohol": ["groceries", "dining", "fun", "entertainment"],
    "household": ["groceries", "shopping", "home"], "pharmacy": ["medical", "household"],
    "personal_care": ["household", "shopping", "fun", "clothing"], "clothing": ["shopping", "fun", "gifts"],
    "electronics": ["shopping", "fun", "household", "hobby"], "hobby": ["fun", "shopping", "entertainment"],
    "books": ["fun", "hobby", "education"], "streaming": ["fun", "bills", "subs", "entertainment"],
    "music": ["fun", "subs", "streaming"], "software": ["bills", "subs", "fun"], "cloud": ["bills", "subs", "software"],
    "subs": ["household", "shopping", "bills", "subs"], "gym": ["fun", "medical", "personal_care", "bills"],
    "pet": ["household"], "gifts": ["fun", "holidays", "shopping"], "entertainment": ["fun", "dining"],
    "groceries": ["household", "food"], "rideshare": ["transport", "fun", "dining", "travel"],
    "parking": ["car", "transport", "fun"], "home_improvement": ["household", "home"],
    "furniture": ["household", "home", "shopping"], "kids_stuff": ["household", "clothing", "kids"],
    "gas": ["transport", "car"], "auto": ["transport", "gas", "car"], "charity": ["gifts", "church"],
    "internet": ["bills", "utilities", "connect"], "phone": ["bills", "utilities", "connect"],
    "office": ["household", "biz_supplies", "shopping"],
}
PERSONAL = {"clothing", "personal_care", "hobby", "books"}
KID_CONCEPTS = {"kids_stuff", "kids_act", "school_lunch", "childcare"}
KIDS = ["Emma", "Liam", "Ava", "Noah", "Mia", "Eli", "Zoe", "Owen", "Lily", "Mateo", "Nora", "Leo", "Aria",
        "Jack", "Maya", "Ezra"]
ADULTS = ["Alex", "Jordan", "Sam", "Taylor", "Chris", "Jamie", "Morgan", "Pat", "Dana", "Riley", "Casey",
          "Drew", "Megan", "Dave", "Priya", "Luis", "Kate", "Ben", "Rosa", "Tom"]
PETS = ["Biscuit", "Luna", "Max", "Pepper", "Moose", "Olive", "Scout", "Rocky"]
STYLES = ["upper", "pos", "dcp", "wf", "clean", "trunc", "cu"]
STYLE_W = [0.24, 0.14, 0.15, 0.15, 0.16, 0.1, 0.06]
CARD_VISIBLE = {"dcp", "wf"}

# rand-stream visits per month by concept
FREQ = {"groceries": 3.0, "dining": 1.0, "fast_food": 1.3, "takeout": 1.0, "coffee": 2.5, "alcohol": 0.8,
        "gas": 2.5, "auto": 0.12, "parking": 0.8, "tolls": 0.6, "rideshare": 0.8, "household": 1.5,
        "pharmacy": 0.8, "medical": 0.25, "dental": 0.12, "vision": 0.05, "pet": 0.6, "personal_care": 0.5,
        "kids_stuff": 0.5, "clothing": 0.5, "electronics": 0.15, "home_improvement": 0.6, "furniture": 0.08,
        "home_services": 0.4, "hobby": 0.5, "entertainment": 0.4, "gifts": 0.3, "charity": 0.1, "travel": 0.1,
        "education": 0.1, "textbooks": 0.05, "office": 0.5, "shipping": 1.2, "biz_supplies": 0.5,
        "advertising": 0.8, "biz_meals": 0.6, "books": 0.3, "holidays": 0.2, "school_lunch": 1.0,
        "kids_act": 0.8, "church": 0.2, "biz_gas": 3.0}
LOCAL_W = {"dining": 1.5, "coffee": 2, "alcohol": 2, "auto": 2.5, "medical": 5, "dental": 8, "vision": 4, "pet": 2,
           "personal_care": 2, "gym": 1, "kids_act": 8, "childcare": 3, "church": 50, "hobby": 1.5,
           "home_services": 10, "entertainment": 1, "groceries": 0.3, "water": 1000, "rent": 1000,
           "property_tax": 1000, "auto_reg": 1000, "education": 1000, "clothing": 0.5, "gifts": 0.5}
WEEKENDY = {"dining", "entertainment", "alcohol", "hobby", "home_improvement", "clothing"}
REFUNDABLE = {"household", "clothing", "electronics", "kids_stuff", "hobby", "home_improvement", "furniture",
              "gifts", "books", "personal_care", "office", "biz_supplies", "entertainment", "travel", "pet"}


# --------------------------------------------------------------------------------------
# Users
# --------------------------------------------------------------------------------------
class U:
    pass


class S:
    pass


def active(c, d):
    return c["start"] <= d and (c["end"] is None or d < c["end"])


def add_cat(u, name, sem, start, group, slot=None, merchant=None):
    if name in u.catidx or ", " in name:
        return None
    c = dict(name=name, sem=sem, start=start, end=None, group=group, slot=slot, merchant=merchant,
             born="init" if start == u.start else "event",
             key=(GROUP_RANK[group], rng.random() if start == u.start else 10 + len(u.cats)))
    u.cats.append(c)
    u.catidx[name] = c
    return c


def slot_name(u, slot, avoid=()):
    _, _, std, fun = SLOTS[slot]
    pool = fun if fun and rng.random() < u.fun_p else std
    cands = [n for n in pool if n not in u.catidx and n not in avoid]
    if not cands:
        cands = [n for n in std + fun if n not in u.catidx and n not in avoid]
    return pick(cands) if cands else None


def slot_sem(slot, name):
    if name in OPAQUE_NAMES:
        return None
    cov = SLOTS[slot][1]
    return {"*"} if cov == "*" else set(cov.split())


def options_on(u, d):
    return [c["name"] for c in sorted(u.cats, key=lambda c: c["key"]) if active(c, d)]


STD_BASE = [("groceries", 1), ("dining", .9), ("fast_food", .3), ("coffee", .35), ("alcohol", .2), ("gas", .9),
            ("car", .7), ("transport", .15), ("rideshare", .15), ("electric", .6), ("natgas", .35), ("water", .35),
            ("utilities", .45), ("internet", .7), ("phone", .8), ("connect", .08), ("car_ins", .5), ("insurance", .4),
            ("renters", .2), ("car_payment", .4), ("household", .85), ("pharmacy", .2), ("medical", .9),
            ("dental", .2), ("personal_care", .5), ("clothing", .7), ("shopping", .2), ("electronics", .15),
            ("streaming", .35), ("subs", .45), ("software", .1), ("fun", .8), ("hobby", .3), ("entertainment", .25),
            ("gym", .4), ("gifts", .8), ("holidays", .15), ("charity", .5), ("church", .12), ("travel", .6),
            ("taxes", .25), ("income", 1), ("interest", .1), ("misc", .9), ("cleaning", .1), ("student_loan", .2)]
FAMILY = [("groceries", 1), ("dining", .8), ("fast_food", .3), ("gas", .5), ("transport", .5), ("bills", .5),
          ("electric", .4), ("utilities", .4), ("internet", .5), ("phone", .6), ("insurance", .6),
          ("household", 1), ("medical", .8), ("gifts", .7), ("travel", .5), ("church", .3), ("charity", .3),
          ("income", 1), ("misc", .6), ("streaming", .3), ("fun", .6), ("home", .3), ("holidays", .2),
          ("car_payment", .3), ("childcare", .2)]
STUDENT = [("rent", 1), ("groceries", 1), ("dining", .9), ("coffee", .5), ("fast_food", .3), ("phone", .6),
           ("transport", .6), ("textbooks", .8), ("education", .6), ("subs", .5), ("fun", .9), ("clothing", .5),
           ("from_parents", .7), ("work_study", .5), ("income", .6), ("student_loan", .15), ("travel", .3),
           ("misc", .9), ("personal_care", .2), ("gym", .2), ("alcohol", .35), ("utilities", .4), ("internet", .3)]
RETIREE = [("groceries", 1), ("dining", .8), ("ss", 1), ("pension", .6), ("income", .5), ("medicare", .9),
           ("pharmacy", .8), ("medical", .9), ("utilities", .8), ("electric", .3), ("phone", .6), ("connect", .4),
           ("church", .6), ("charity", .5), ("grandkids", .6), ("gifts", .4), ("garden", .5), ("home", .5),
           ("travel", .7), ("car", .7), ("gas", .9), ("insurance", .6), ("household", .8), ("subs", .3),
           ("streaming", .3), ("clothing", .4), ("misc", .8), ("cleaning", .3), ("property_tax", .4),
           ("personal_care", .3), ("pet", .3), ("hobby", .3), ("taxes", .3)]
MERGED = ["bills", "food", "transport", "shopping", "fun", "medical", "gifts", "travel", "kids", "household",
          "subs"]
BIZ_SLOTS = [("biz_supplies", 1), ("biz_ship", .7), ("biz_soft", .8), ("biz_ads", .5), ("biz_income", 1),
             ("biz_meals", .4), ("biz_car", .4), ("est_tax", .8)]


def make_user(uid, arch, town):
    u = U()
    u.id, u.arch, u.town = uid, arch, town
    u.cats, u.catidx, u.succ = [], {}, {}
    u.cev, u.mev = defaultdict(list), defaultdict(list)
    u.events = []
    u.streams = []
    u.used = set()
    u.p2_over, u.funcat, u.kidcat, u.adultcat = {}, {}, {}, {}
    u.msplit = {}
    u.petcat = None
    u.xmas_cat = u.bts_cat = u.trip_cat = None
    u.trip = None
    base = arch
    if arch == "newcomer":
        base = pick(["single", "couple", "single", "coiner", "family_whom"])
        u.start = rdate(dt.date(2025, 10, 1), dt.date(2025, 12, 5))
    elif rng.random() < 0.2:
        u.start = rdate(dt.date(2025, 2, 1), dt.date(2025, 6, 30))
    else:
        u.start = Y0
    u.base = base
    u.fun_p = {"coiner": 0.85, "student": 0.25}.get(base, 0.07)
    u.sloppy = rng.uniform(0.05, 0.1) if rng.random() < 0.2 else rng.uniform(0.005, 0.025)
    u.adults = rng.sample(ADULTS, 2)
    u.n_adults = 2 if base in ("couple",) else (2 if rng.random() < {"family_whom": .85, "retiree": .6, "small_biz": .4}.get(base, .3) else 1)
    if base == "student":
        u.n_adults = 1
    u.kids = []
    pk = {"family_whom": 1.0, "couple": .3, "small_biz": .3, "splitter": .35, "minimalist": .3, "coiner": .2,
          "single": .15}.get(base, 0)
    if rng.random() < pk:
        u.kids = rng.sample(KIDS, 2 if base != "family_whom" else rng.choice([2, 2, 3]))
        if base != "family_whom" and rng.random() < .4:
            u.kids = u.kids[:1]
    u.owner = rng.random() < {"student": 0, "retiree": .85, "family_whom": .7}.get(base, .5)
    u.car = rng.random() < {"student": .4, "retiree": .9}.get(base, .92)
    u.pet = rng.random() < {"student": .1}.get(base, .4)
    u.biz = pick(["etsy", "photographer", "contractor", "consultant", "driver"]) if base == "small_biz" else None
    u.stmt_name = (pick(SURN).upper() + " " + u.adults[0].upper()).replace("'", "")
    # accounts
    st = wpick(STYLES, STYLE_W)
    if base == "couple" and rng.random() < .5:
        st = pick(["dcp", "wf"])
    u.accts = [dict(style=st, cards=[digits(4), digits(4)])]
    if u.biz and rng.random() < .6:
        u.accts.append(dict(style=wpick(STYLES, STYLE_W), cards=[digits(4), digits(4)]))
    elif rng.random() < .25:
        u.accts.append(dict(style=wpick(STYLES, STYLE_W), cards=[digits(4), digits(4)]))
    u.p2p_named = rng.random() < .5
    return u


# ---------------- streams ----------------
def new_stream(u, mid, concept, kind, **kw):
    m = M[mid]
    s = S()
    s.mid, s.concept, s.kind = mid, concept, kind
    s.rate = kw.get("rate", FREQ.get(concept, 0.3) * rng.uniform(0.5, 1.5))
    s.start = kw.get("start", u.start)
    s.end = kw.get("end", Y1)
    s.day = kw.get("day", rng.randint(1, 28))
    s.dates = kw.get("dates", [])
    s.anchor = kw.get("anchor")
    s.payer_p = kw.get("payer_p", rng.uniform(0.15, 0.85) if u.n_adults == 2 else 1.0)
    s.who = kw.get("who")
    s.acct = kw.get("acct", 0)
    s.months = kw.get("months")
    s.sign = kw.get("sign", 1)
    s.ach = "a" in m["flags"] or kw.get("ach", False)
    s.income = "i" in m["flags"] or kw.get("income", False)
    s.recurring = kind in ("monthly", "biweekly", "semimonthly", "weekly")
    lo, hi = m["lo"], m["hi"]
    s.fixed = kw.get("fixed")
    if s.fixed is None and kind in ("monthly", "biweekly", "semimonthly", "weekly"):
        v = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        s.fixed = round(v) - 0.01 if "k" in m["flags"] and v > 3 else round(v, 2)
    s.variable = concept in ("electric", "natgas", "water", "trash", "phone", "internet", "income_biz",
                             "school_lunch", "kids_act", "medical") and kind != "rand"
    s.med = math.exp(rng.uniform(math.log(lo) * 0.7 + math.log(hi) * 0.3, math.log(lo) * 0.3 + math.log(hi) * 0.7))
    s.lo, s.hi = lo, hi
    s.price = []
    if "k" in m["flags"] and m["name"] in PRICE_EVENTS and s.fixed:
        d, pct = PRICE_EVENTS[m["name"]]
        s.price.append((d, math.floor(s.fixed * (1 + pct)) + 0.99))
    if "k" in m["flags"] and s.fixed and rng.random() < 0.12:
        d = rdate(dt.date(2025, 3, 1), dt.date(2025, 10, 31))
        s.price.append((d, round(s.fixed * pick([0.6, 1.5, 1.8]), 2)))
        s.price.sort()
    locs = []
    if not m["local"] and "s" in m["flags"]:
        for _ in range(rng.choice([1, 1, 2])):
            locs.append((rng.randint(100, 9999) if m["sfmt"] != 1 else rng.randint(1000, 30000),
                         pick([u.town["name"]] * 2 + u.town["subs"])))
    else:
        locs.append((rng.randint(1, 99), u.town["name"] if m["local"] or rng.random() < .6 else pick(u.town["subs"])))
    s.locs = locs
    s.kidflag = m["kid"]
    for k, v in kw.items():
        if k in ("seasonal_boost",):
            setattr(s, k, v)
    u.streams.append(s)
    u.used.add(mid)
    return s


def choose_merchants(u, concept, n, town=None):
    town = town or u.town
    cands, ws = [], []
    for mid in BY_CONCEPT.get(concept, []):
        m = M[mid]
        if mid in u.used or not available(m, town):
            continue
        w = m["cw"][concept]
        if m["local"]:
            w *= LOCAL_W.get(concept, 1.0)
        elif "US" not in m["where"]:
            w *= 2.5
        cands.append(mid)
        ws.append(w)
    return wsample(cands, ws, n)


def add_rand(u, concept, nlo, nhi, prob=1.0, **kw):
    if rng.random() >= prob:
        return []
    out = []
    mids = choose_merchants(u, concept, rng.randint(nlo, nhi))
    for i, mid in enumerate(mids):
        m = M[mid]
        kind = "monthly" if "r" in m["flags"] else "rand"
        s = new_stream(u, mid, concept, kind, **kw)
        if kind == "rand" and i == 0 and concept == "groceries":
            s.rate *= 1.4
        elif kind == "rand" and i > 0:
            s.rate *= 0.6
        out.append(s)
    return out


P2P_ID = [0]


def p2p_merchant(u, app, person, concept, sign):
    name = f"{person} ({app})"
    m = add_merchant(name=name, concepts=[(concept, 1.0)], descs=[person.upper()], lo=10, hi=200,
                     flags="", where=["@none"], local=True, town="@none", p2p=(app, person, sign), owner=u.id)
    return m


def person_name():
    return pick(FIRST) + " " + pick(SURN)


def build_streams(u):
    b = u.base
    kids = bool(u.kids)
    # food
    add_rand(u, "groceries", 1, 3)
    add_rand(u, "dining", 2, 6, .95)
    add_rand(u, "fast_food", 2, 5, .9)
    add_rand(u, "takeout", 1, 2, .6)
    add_rand(u, "coffee", 1, 3, .7)
    add_rand(u, "alcohol", 1, 2, .4 if b != "retiree" else .2)
    if u.car:
        add_rand(u, "gas", 1, 2)
        add_rand(u, "auto", 1, 2, .9)
        add_rand(u, "insurance_auto", 1, 1)
        add_rand(u, "car_payment", 1, 1, .5)
        add_rand(u, "tolls", 1, 1, .3)
        add_rand(u, "auto_reg", 1, 1, .5)
        for s in u.streams:
            if s.concept == "auto_reg":
                s.kind, s.dates = "dates", [rdate(dt.date(2025, 2, 1), dt.date(2025, 12, 20))]
    add_rand(u, "rideshare", 1, 2, .5 if b != "retiree" else .1)
    add_rand(u, "parking", 1, 1, .3)
    # housing
    if u.owner:
        add_rand(u, "mortgage", 1, 1)
        add_rand(u, "water", 1, 1)
        add_rand(u, "trash", 1, 1, .6)
        add_rand(u, "home_improvement", 1, 2, .9)
        add_rand(u, "home_services", 1, 2, .5)
        add_rand(u, "property_tax", 1, 1, .5)
        for s in u.streams:
            if s.concept == "property_tax":
                s.kind, s.dates = "dates", [rdate(dt.date(2025, 10, 10), dt.date(2025, 11, 30))]
                if rng.random() < .5:
                    s.dates.append(rdate(dt.date(2025, 4, 1), dt.date(2025, 4, 30)))
    elif b == "student" and rng.random() < .5:
        pass  # rent paid via roommate below
    else:
        add_rand(u, "rent", 1, 1)
        add_rand(u, "insurance_home", 1, 1, .3)
        add_rand(u, "home_improvement", 1, 1, .3)
    add_rand(u, "electric", 1, 1, .95 if b != "student" else .5)
    add_rand(u, "natgas", 1, 1, .5)
    add_rand(u, "internet", 1, 1, .95 if b != "student" else .4)
    add_rand(u, "phone", 1, 1)
    # subscriptions
    add_rand(u, "streaming", 1, 4, .9)
    add_rand(u, "music", 1, 1, .6)
    add_rand(u, "software", 1, 1, .3)
    add_rand(u, "cloud", 1, 1, .45)
    add_rand(u, "subs", 1, 1, .4)
    add_rand(u, "books", 1, 1, .3)
    add_rand(u, "gym", 1, 1, .45 if b != "retiree" else .2)
    # stuff
    add_rand(u, "household", 2, 3)
    add_rand(u, "pharmacy", 1, 1, .9)
    add_rand(u, "medical", 1, 2, .8)
    add_rand(u, "dental", 1, 1, .7)
    add_rand(u, "vision", 1, 1, .25)
    add_rand(u, "clothing", 1, 3, .85)
    add_rand(u, "personal_care", 1, 2, .9)
    add_rand(u, "electronics", 1, 1, .5)
    add_rand(u, "hobby", 1, 2, .6)
    add_rand(u, "entertainment", 1, 3, .7)
    add_rand(u, "gifts", 1, 2, .6)
    add_rand(u, "furniture", 1, 1, .3)
    add_rand(u, "charity", 1, 1, .35)
    if b == "retiree" or rng.random() < .15:
        for s in add_rand(u, "church", 1, 1):
            s.kind = "weekly"
            s.anchor = dt.date(2025, 1, 5)
            s.fixed = float(pick([20, 25, 40, 50, 100]))
    if u.pet:
        add_rand(u, "pet", 2, 3)
    if kids:
        add_rand(u, "kids_stuff", 1, 2)
        for s in add_rand(u, "kids_act", 1, 2):
            s.kind = "monthly"
            s.fixed = round(rng.uniform(60, 220))
            s.who = pick(u.kids)
        if rng.random() < .7:
            mid = BY_NAME["MySchoolBucks"]["id"]
            for k in u.kids[:2]:
                s = new_stream(u, mid, "school_lunch", "rand", rate=1.2, who=k)
                s.med = pick([20, 25, 30, 40, 50]) + (5 if k == u.kids[-1] else 0)
                s.lo = s.hi = s.med
                s.months = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12]
            u.used.discard(mid)
        if rng.random() < .3:
            add_rand(u, "childcare", 1, 1)
            for s in u.streams:
                if s.concept == "childcare":
                    s.who = u.kids[-1]
                    s.kind = "monthly"
        for s in u.streams:
            if s.kidflag and s.who is None:
                s.who = "?"
        if rng.random() < .6:
            s = new_stream(u, BY_NAME["Spirit Halloween"]["id"], "holidays", "rand", rate=1.0, months=[10])
    # tax prep
    if rng.random() < .5:
        mid = BY_NAME[pick(["TurboTax", "TurboTax", "H&R Block"])]["id"]
        new_stream(u, mid, "taxes", "dates", dates=[rdate(dt.date(2025, 2, 1), dt.date(2025, 4, 14))])
    if rng.random() < .35:
        new_stream(u, BY_NAME["IRS Tax Refund"]["id"], "tax_refund", "dates",
                   dates=[rdate(dt.date(2025, 2, 20), dt.date(2025, 5, 15))], sign=-1)
    # travel (non-trip bookings)
    add_rand(u, "travel", 1, 2, .4)
    # income
    if b == "retiree":
        s = new_stream(u, BY_NAME["Social Security"]["id"], "income_gov", "monthly", sign=-1,
                       day=pick([3, 12, 19, 26]))
        if u.n_adults == 2 and rng.random() < .7:
            new_stream(u, BY_NAME["Social Security"]["id"], "income_gov", "monthly", sign=-1, day=s.day + 1,
                       payer_p=0.0)
        if rng.random() < .6:
            new_stream(u, BY_NAME[pick([p[0] for p in PENSIONS])]["id"], "pension", "monthly", sign=-1, day=1)
    elif b == "student":
        if rng.random() < .7:
            parent = person_name()
            app = pick(["Zelle", "Zelle", "Venmo"])
            m = p2p_merchant(u, app, parent, "income_family", -1)
            new_stream(u, m["id"], "income_family", "monthly", sign=-1, fixed=float(pick([200, 300, 400, 500, 750])))
        if rng.random() < .5:
            emp = pick([m for m in M if m["local"] and m["town"] == u.town["name"] and m["primary"] == "income"])
            s = new_stream(u, emp["id"], "income", "biweekly", sign=-1, anchor=dt.date(2025, 1, 10))
            s.fixed = round(rng.uniform(250, 650), 2)
        uni = [m for m in M if m["local"] and m["town"] == u.town["name"] and m["primary"] == "education"]
        if uni:
            new_stream(u, uni[0]["id"], "education", "dates",
                       dates=[dt.date(2025, 1, rng.randint(5, 15)), dt.date(2025, 8, rng.randint(10, 25))])
        new_stream(u, BY_NAME["Barnes & Noble College"]["id"], "textbooks", "dates",
                   dates=[dt.date(2025, 1, rng.randint(8, 25)), dt.date(2025, 8, rng.randint(15, 31)),
                          dt.date(2025, 9, rng.randint(1, 10))])
        if rng.random() < .5:
            new_stream(u, BY_NAME["Chegg"]["id"], "textbooks", "monthly")
    else:
        earners = u.n_adults if b in ("couple", "family_whom") else 1
        if b == "small_biz" and rng.random() < .6:
            earners = 0
        for e in range(earners):
            if rng.random() < .25:
                emp = pick([m for m in M if m["local"] and m["town"] == u.town["name"] and m["primary"] == "income"])
            else:
                emp = BY_NAME[pick(EMPLOYERS)]
            if emp["id"] in u.used:
                continue
            kind = pick(["biweekly", "biweekly", "semimonthly"])
            s = new_stream(u, emp["id"], "income", kind, sign=-1, anchor=dt.date(2025, 1, 3) + dt.timedelta(days=7 * rng.randint(0, 1)),
                           payer_p=1.0 if e == 0 else 0.0)
    if rng.random() < .3:
        new_stream(u, INTEREST_M["id"], "interest", "monthly", sign=-1, day=28, fixed=None)
    # business
    if u.biz:
        dep = {"etsy": "Etsy Payments", "photographer": "Stripe", "contractor": "Square Deposits",
               "consultant": "Stripe", "driver": "Uber Driver Pay"}[u.biz]
        s = new_stream(u, BY_NAME[dep]["id"], "income_biz", "weekly", sign=-1, anchor=dt.date(2025, 1, 6))
        s.fixed = round(rng.uniform(250, 1500), 2)
        if u.biz == "etsy" and rng.random() < .5:
            s2 = new_stream(u, BY_NAME["Shopify Payments"]["id"], "income_biz", "weekly", sign=-1, anchor=dt.date(2025, 1, 9))
            s2.fixed = round(rng.uniform(100, 700), 2)
        bacct = 1 if len(u.accts) > 1 else 0
        new_stream(u, BY_NAME["IRS"]["id"], "est_taxes", "dates", sign=1, acct=bacct,
                   dates=[dt.date(2025, 1, 15), dt.date(2025, 4, 15), dt.date(2025, 6, 16), dt.date(2025, 9, 15)])
        bizset = {"etsy": ["shipping", "biz_supplies", "biz_software", "advertising", "office"],
                  "photographer": ["biz_software", "advertising", "office", "biz_meals"],
                  "contractor": ["biz_supplies", "office", "biz_gas", "biz_software"],
                  "consultant": ["biz_software", "biz_meals", "office", "advertising"],
                  "driver": ["biz_gas", "biz_software"]}[u.biz]
        for c in bizset:
            if c == "biz_gas":
                for s in add_rand(u, "gas", 1, 1, acct=bacct):
                    s.concept = "biz_gas"
                    s.rate = 5 if u.biz == "driver" else 2.5
            elif c == "biz_meals":
                for s in add_rand(u, "dining", 1, 2, acct=bacct):
                    s.concept = "biz_meals"
            else:
                add_rand(u, c, 1, 3, acct=bacct)
        u.biz_mixed = {}
        for s in u.streams:  # mixed-use merchants: some purchases are for the business
            if M[s.mid]["name"] in ("Amazon", "Costco", "Walmart", "Target", "The Home Depot", "Lowe's", "Staples",
                                    "Michaels", "Hobby Lobby", "Best Buy", "Office Depot"):
                u.biz_mixed[s.mid] = rng.uniform(.2, .6)
    # P2P
    if rng.random() < {"retiree": .3, "family_whom": .35, "couple": .3}.get(b, .12):
        m = p2p_merchant(u, pick(["Zelle", "Zelle", "Venmo"]), person_name(), "cleaning", 1)
        new_stream(u, m["id"], "cleaning", pick(["monthly", "biweekly"]), anchor=dt.date(2025, 1, 7),
                   fixed=float(pick([120, 140, 160, 180, 200])))
    if b == "student" or (not u.owner and rng.random() < .15):
        rm = person_name()
        app = pick(["Venmo", "Zelle"])
        if b == "student" and not any(s.concept == "rent" for s in u.streams):
            m = p2p_merchant(u, app, rm, "rent", 1)
            new_stream(u, m["id"], "rent", "monthly", day=1, fixed=float(pick([550, 650, 700, 825, 900])))
        m2 = p2p_merchant(u, "Venmo", rm, "electric", -1)
        s = new_stream(u, m2["id"], "electric", "monthly", sign=-1, day=rng.randint(5, 20))
        s.fixed, s.variable = round(rng.uniform(30, 70), 2), True
    if rng.random() < .55:
        for _ in range(rng.randint(1, 3)):
            m = p2p_merchant(u, pick(["Venmo", "Venmo", "Cash App", "PayPal"]), person_name(),
                             pick(["dining", "entertainment", "gifts"]), 1)
            m["concepts"] = [("dining", 5), ("entertainment", 3), ("gifts", 2)]
            m["cw"] = {"dining": .5, "entertainment": .3, "gifts": .2}
            m["multi"] = True
            new_stream(u, m["id"], "dining", "rand", rate=rng.uniform(.3, .9))
    if kids and rng.random() < .35:
        m = p2p_merchant(u, pick(["Venmo", "Zelle"]), person_name(), "childcare", 1)
        new_stream(u, m["id"], "childcare", "rand", rate=1.5, who=pick(u.kids))
    if u.owner and rng.random() < .25:
        m = p2p_merchant(u, pick(["Venmo", "Zelle", "Cash App"]), person_name(), "home_services", 1)
        new_stream(u, m["id"], "home_services", "monthly", months=[4, 5, 6, 7, 8, 9, 10],
                   fixed=float(pick([60, 80, 100, 120])))
    # payees discovered during the year (brand-new payees)
    disc = ["dining", "dining", "fast_food", "coffee", "clothing", "hobby", "entertainment", "household",
            "personal_care", "auto", "medical", "gifts", "furniture", "electronics", "alcohol", "takeout", "dining"]
    if u.pet:
        disc.append("pet")
    if u.kids:
        disc += ["kids_stuff", "kids_act"]
    if u.owner:
        disc += ["home_services", "home_improvement"]
    for _ in range(rng.randint(6, 14)):
        c = pick(disc)
        mids = choose_merchants(u, c, 1)
        if not mids:
            continue
        st = rdate(u.start + dt.timedelta(days=3), Y1)
        s = new_stream(u, mids[0], c, "rand", start=st, end=min(Y1, st + dt.timedelta(days=rng.randint(1, 160))),
                       rate=rng.uniform(.2, 1.2))
        s.first = st
    # partner payer for couples/families: bills paid by adult 0
    for s in u.streams:
        if s.kind != "rand":
            s.payer_p = 1.0 if s.payer_p not in (0.0,) else 0.0


# ---------------- categories ----------------
def build_categories(u):
    b = u.base
    used = {s.concept for s in u.streams}
    for s in u.streams:
        for c in M[s.mid]["cw"]:
            used.add(c)
    chosen = []

    def want(slot, p):
        if rng.random() < p:
            chosen.append(slot)

    if b == "minimalist":
        for sl in rng.sample(MERGED, rng.randint(3, 6)):
            chosen.append(sl)
        if u.owner or rng.random() < .5:
            chosen.append("housing")
        chosen += ["income", "misc"]
    else:
        scheme = {"family_whom": FAMILY, "student": STUDENT, "retiree": RETIREE}.get(b, STD_BASE)
        for sl, p in scheme:
            if b == "splitter":
                p = min(1.0, p * 1.5 + .1)
            want(sl, p)
        if b not in ("student",):
            chosen.append("mortgage" if u.owner else "rent")
            if u.owner and b not in ("family_whom",):
                want("home", .8)
        if u.pet and b != "family_whom":
            chosen.append("pet")
        if u.kids and b != "family_whom":
            chosen.append("kids")
            want("childcare", .3)
            want("kids_act", .3)
        if b == "small_biz":
            for sl, p in BIZ_SLOTS:
                want(sl, p)
            chosen.append("biz_income")
        if "income" not in chosen and not any(SLOTS[x][0] == "income" for x in chosen):
            chosen.append("income")
        if not any(x in chosen for x in ("misc",)):
            if rng.random() < .8:
                chosen.append("misc")
    seen = []
    for sl in chosen:
        if sl not in seen:
            seen.append(sl)
    for sl in seen:
        nm = slot_name(u, sl)
        if nm:
            add_cat(u, nm, slot_sem(sl, nm), u.start, SLOTS[sl][0], slot=sl)
    # archetype-specific named categories
    if b == "couple" and rng.random() < .65:
        fmt = pick(["{} Fun", "{}'s Money", "Blow - {}", "{} Allowance", "{} Spending"])
        for i in range(2):
            n = fmt.format(u.adults[i])
            add_cat(u, n, {"personal_fun"}, u.start, "person")
            u.funcat[i] = n
    if b == "couple" and rng.random() < .2:
        for i in range(2):
            n = f"{u.adults[i]} Paycheck"
            add_cat(u, n, {"income"}, u.start, "income")
    if b == "family_whom":
        for k in u.kids:
            add_cat(u, k, {"kid"}, u.start, "person")
            u.kidcat[k] = k
        for i in range(u.n_adults):
            n = pick(["{}", "{} Stuff", "{}'s"]).format(u.adults[i])
            if add_cat(u, n, {"person"}, u.start, "person"):
                u.adultcat[i] = n
        if u.pet:
            n = pick(PETS)
            add_cat(u, n, None, u.start, "person")
            u.petcat = n
        if not any(c.get("slot") == "household" for c in u.cats):
            add_cat(u, "House", {"household", "home_improvement", "furniture", "home_services", "cleaning"},
                    u.start, "everyday", slot="household")
    if b == "splitter":
        for s in u.streams:
            nm = M[s.mid]["name"]
            if nm in ("Costco", "Target", "Walmart", "Amazon", "Sam's Club", "Meijer", "Fred Meyer") and rng.random() < .75:
                short = {"Sam's Club": "Sams"}.get(nm, nm)
                parts = {"Amazon": [("home", "household furniture office"), ("fun", "electronics books hobby gifts clothing")],
                         }.get(nm, [("food", "groceries"), ("household", "household electronics clothing gifts furniture")])
                if nm == "Amazon" and u.kids:
                    parts.append(("kids", "kids_stuff"))
                sp = {}
                for suf, cov in parts:
                    cn = f"{short} - {suf}"
                    if add_cat(u, cn, set(cov.split()), u.start, "everyday", merchant=s.mid):
                        for c in cov.split():
                            sp[c] = cn
                sp["*"] = f"{short} - {parts[-1][0] if nm == 'Amazon' else parts[1][0]}"
                u.msplit[s.mid] = sp
    # sinking funds with no transactions
    nf = {"minimalist": rng.randint(0, 1), "splitter": rng.randint(3, 7), "student": rng.randint(0, 2)}.get(b, rng.randint(0, 4))
    for f in rng.sample(FUNDS, nf):
        add_cat(u, f, {"savings"}, u.start, "savings")
    while len(u.cats) > 42:
        cands = [c for c in u.cats if c["group"] == "savings"] or \
            [c for c in u.cats if c.get("slot") not in (None, "income", "misc", "groceries") and c["merchant"] is None]
        victim = pick(cands)
        u.cats.remove(victim)
        del u.catidx[victim["name"]]
    # concept map: most specific covering slot
    u.cmap = {}
    allc = set(used) | {"gifts", "holidays", "kids_stuff", "travel", "income", "household"}
    misc = [c for c in u.cats if c["sem"] == {"*"} or (c.get("slot") == "misc")]
    for concept in sorted(allc):
        best, bl = None, 99
        for c in u.cats:
            if c.get("slot") and c["merchant"] is None:
                cov = SLOTS[c["slot"]][1].split()
                if concept in cov and len(cov) < bl:
                    best, bl = c["name"], len(cov)
        if best is None:
            if misc:
                best = misc[0]["name"]
            else:
                # fall back to a broad everyday category
                cands = [c["name"] for c in u.cats if c["group"] in ("everyday", "fun") and c["merchant"] is None]
                best = cands[0] if cands else u.cats[0]["name"]
        u.cmap[concept] = best
    # couples: second card holder files some concepts differently
    if b == "couple":
        concepts = sorted({s.concept for s in u.streams if s.kind == "rand"})
        rng.shuffle(concepts)
        for c in concepts:
            if len(u.p2_over) >= rng.randint(2, 4):
                break
            alts = [x["name"] for x in u.cats if x.get("slot") in ALT.get(c, []) and x["name"] != u.cmap.get(c)]
            if alts:
                u.p2_over[c] = pick(alts)


def slot_cat(u, slots):
    for c in u.cats:
        if c.get("slot") in slots and c["end"] is None:
            return c["name"]
    return None


# ---------------- category events ----------------
def build_events(u):
    lo = max(u.start + dt.timedelta(days=40), dt.date(2025, 3, 1))
    hi = dt.date(2025, 11, 5)
    rand_streams = [s for s in u.streams if s.kind == "rand" and not M[s.mid]["p2p"]]
    used_concepts = sorted({s.concept for s in u.streams})
    if lo < hi:
        # changed mind (concept or merchant level)
        for _ in range(rng.choice([0, 1, 1, 2])):
            d = rdate(lo, dt.date(2025, 10, 20))
            if rng.random() < .5 and rand_streams:
                s = pick(rand_streams)
                cur = u.cmap.get(s.concept)
                alts = [x["name"] for x in u.cats if x.get("slot") in ALT.get(s.concept, []) and x["name"] != cur
                        and x["merchant"] is None]
                if alts and s.mid not in u.msplit:
                    u.mev[s.mid].append((d, pick(alts), "changed_mind"))
                    u.events.append(("changed_mind", d, M[s.mid]["name"]))
            else:
                c = pick(used_concepts)
                alts = [x["name"] for x in u.cats if x.get("slot") in ALT.get(c, []) and x["name"] != u.cmap.get(c)
                        and x["merchant"] is None]
                if alts:
                    u.cev[c].append((d, pick(alts), "changed_mind"))
                    u.events.append(("changed_mind", d, c))
        # new categories
        if len(u.cats) < 44 and rng.random() < .3:
            subc = [c for c in ("streaming", "music", "software", "cloud") if c in used_concepts]
            if subc and not slot_cat(u, ["subs", "streaming"]):
                d = rdate(lo, hi)
                n = pick(["Subscriptions", "Subs", "Streaming & Apps", "Monthly Subs"])
                if add_cat(u, n, {"streaming", "music", "software", "cloud", "subs"}, d, "event"):
                    for c in subc:
                        u.cev[c].append((d, n, "new_category"))
                    u.events.append(("new_category", d, n))
        if len(u.cats) < 44 and rng.random() < .2 and "coffee" in used_concepts and not slot_cat(u, ["coffee"]):
            d = rdate(lo, hi)
            n = pick(["Coffee", "Coffee Shops", "Caffeine"])
            if add_cat(u, n, {"coffee"}, d, "event"):
                u.cev["coffee"].append((d, n, "new_category"))
                u.events.append(("new_category", d, n))
        if len(u.cats) < 44 and not u.pet and rng.random() < .15:
            d = rdate(lo, dt.date(2025, 10, 1))
            n = pick(["Dog", "Puppy", pick(PETS), "Pet Stuff"])
            if add_cat(u, n, {"pet"} if n in ("Dog", "Puppy", "Pet Stuff") else None, d, "event"):
                u.pet = True
                for mid in choose_merchants(u, "pet", 3):
                    new_stream(u, mid, "pet", "rand", start=d, rate=rng.uniform(.8, 1.6))
                u.cev["pet"].append((d, n, "new_category"))
                u.events.append(("new_category", d, n))
        if len(u.cats) < 44 and u.owner and rng.random() < .15:
            d = rdate(lo, dt.date(2025, 9, 30))
            n = pick(["Kitchen Remodel", "Bathroom Reno", "Home Projects", "Basement Finish", "Deck Project"])
            if add_cat(u, n, {"home_improvement", "furniture"}, d, "event"):
                u.cev["home_improvement"].append((d, n, "new_category"))
                u.cev["furniture"].append((d, n, "new_category"))
                for s in u.streams:
                    if s.concept == "home_improvement" and s.kind == "rand":
                        s.boost = (d, d + dt.timedelta(days=120), 3.0)
                u.events.append(("new_category", d, n))
        if len(u.cats) < 44 and u.base in ("couple", "family_whom", "single") and rng.random() < .08:
            d = rdate(lo, dt.date(2025, 9, 30))
            n = pick(["Baby", "Baby Stuff", "Nursery"])
            if add_cat(u, n, {"kids_stuff"}, d, "event"):
                u.cev["kids_stuff"].append((d, n, "new_category"))
                for mid in choose_merchants(u, "kids_stuff", 2):
                    new_stream(u, mid, "kids_stuff", "rand", start=d, rate=1.5)
                for s in u.streams:
                    if M[s.mid]["name"] in ("Target", "Amazon", "Walmart"):
                        s.baby = d
                u.events.append(("new_category", d, n))
        if u.base == "small_biz" and rng.random() < .3 and len(u.cats) < 44:
            d = rdate(lo, hi)
            n = pick(["Biz - Software", "SaaS Tools", "Biz Apps"])
            if n not in u.catidx and add_cat(u, n, {"biz_software"}, d, "biz"):
                u.cev["biz_software"].append((d, n, "new_category"))
                u.events.append(("new_category", d, n))
    # seasonal categories
    if len(u.cats) < 44 and rng.random() < .35 and u.start < dt.date(2025, 11, 15):
        d = rdate(max(u.start, dt.date(2025, 10, 25)), dt.date(2025, 11, 20))
        n = pick(["Christmas", "Holiday Gifts", "Christmas 2025", "Xmas"])
        if add_cat(u, n, {"xmas", "gifts", "holidays"}, d, "event"):
            u.xmas_cat = n
            u.events.append(("new_category", d, n))
    if u.kids and len(u.cats) < 44 and rng.random() < .45 and u.start < dt.date(2025, 7, 20):
        d = rdate(dt.date(2025, 7, 10), dt.date(2025, 8, 3))
        n = pick(["Back to School", "School Supplies", "BTS 2025"])
        if add_cat(u, n, {"bts", "kids_stuff"}, d, "event"):
            u.bts_cat = n
            u.events.append(("new_category", d, n))
    # renames and merges
    if rng.random() < .3 and lo < hi:
        cands = [c for c in u.cats if c.get("slot") and c["end"] is None and c["start"] == u.start
                 and len(SLOTS[c["slot"]][2] + SLOTS[c["slot"]][3]) > 1 and c["merchant"] is None]
        if cands:
            c = pick(cands)
            n = slot_name(u, c["slot"], avoid=(c["name"],))
            if n:
                d = rdate(lo, hi)
                c["end"] = d
                nc = add_cat(u, n, c["sem"], d, c["group"], slot=c["slot"])
                nc["key"] = c["key"]
                nc["born"] = "rename"
                u.succ[c["name"]] = n
                u.events.append(("rename", d, c["name"] + "->" + n))
    if rng.random() < .2 and lo < hi:
        pairs = [("coffee", "dining"), ("fast_food", "dining"), ("alcohol", "fun"), ("hobby", "fun"),
                 ("electronics", "shopping"), ("streaming", "subs"), ("natgas", "utilities"), ("pharmacy", "medical"),
                 ("entertainment", "fun"), ("dental", "medical"), ("car_ins", "insurance")]
        rng.shuffle(pairs)
        for a, bb in pairs:
            ca = [c for c in u.cats if c.get("slot") == a and c["end"] is None and c["start"] == u.start]
            cb = [c for c in u.cats if c.get("slot") == bb and c["end"] is None]
            if ca and cb:
                d = rdate(lo, hi)
                ca[0]["end"] = d
                u.succ[ca[0]["name"]] = cb[0]["name"]
                u.events.append(("merge", d, ca[0]["name"] + "->" + cb[0]["name"]))
                break
    # trip
    if rng.random() < .5 and u.start < dt.date(2025, 10, 1):
        others = [t for t in TOWN.values() if t["region"] != u.town["region"]]
        dest = pick(others)
        ts = rdate(max(u.start + dt.timedelta(days=30), dt.date(2025, 6, 1)), dt.date(2025, 11, 20))
        if u.base == "student":
            ts = rdate(dt.date(2025, 3, 8), dt.date(2025, 3, 20))
            if ts < u.start + dt.timedelta(days=20):
                ts = None
        if ts:
            te = ts + dt.timedelta(days=rng.randint(3, 8))
            u.trip = (ts, min(te, Y1), dest)
            if rng.random() < .5 and len(u.cats) < 45:
                book = max(u.start, ts - dt.timedelta(days=rng.randint(15, 50)))
                n = pick([f"{dest['name']} Trip", f"Vacation - {dest['name']}", f"Trip to {dest['name']}",
                          f"{dest['state']} Trip"])
                if add_cat(u, n, {"trip", "travel"}, book, "event"):
                    u.trip_cat = n
                    u.trip_book = book
                    u.events.append(("new_category", book, n))
            u.trip_book = getattr(u, "trip_book", max(u.start, ts - dt.timedelta(days=rng.randint(15, 50))))
    for c in u.cev:
        u.cev[c].sort(key=lambda x: x[0])
    for mid in u.mev:
        u.mev[mid].sort(key=lambda x: x[0])


# ---------------- simulation ----------------
def season_mult(u, s, d):
    m = M[s.mid]
    f = 1.0
    if "g" in m["flags"] and (dt.date(2025, 11, 20) <= d <= dt.date(2025, 12, 22)):
        f *= 2.0
    if "b" in m["flags"] and u.kids and (dt.date(2025, 7, 25) <= d <= dt.date(2025, 9, 5)):
        f *= 2.0
    if s.concept in ("home_services", "home_improvement") and d.month in (4, 5, 6, 7, 8, 9):
        f *= 1.6
    b = getattr(s, "boost", None)
    if b and b[0] <= d <= b[1]:
        f *= b[2]
    if u.trip and u.trip[0] <= d <= u.trip[1] and not s.recurring:
        f *= 0.2
    return f


def amount_for(u, s, d, concept):
    m = M[s.mid]
    if s.fixed is not None:
        a = s.fixed
        for pd, pa in s.price:
            if d >= pd:
                a = pa
        if s.variable:
            seas = 1.0
            if concept == "electric":
                seas = 1.35 if d.month in (6, 7, 8, 9) and u.town["region"] in ("so", "tx", "we") else (1.2 if d.month in (1, 2, 12) else 1.0)
            if concept == "natgas":
                seas = {1: 2.0, 2: 1.8, 3: 1.4, 11: 1.4, 12: 1.9}.get(d.month, 0.6)
            a = a * seas * math.exp(rng.gauss(0, .12))
        return round(a, 2)
    if s.concept == "interest" or m["name"] == "Interest":
        return round(rng.uniform(0.2, 30), 2)
    v = math.exp(rng.gauss(math.log(s.med), 0.45))
    mult = {"groceries": 1.2, "electronics": 2.2, "household": 0.8, "clothing": 1.0, "gifts": 1.3}.get(concept, 1.0) if m["multi"] else 1.0
    v *= mult
    v = max(s.lo * 0.6, min(v, s.hi * 1.3))
    if concept == "gas" or concept == "biz_gas":
        v = max(15, min(v, 95))
    return round(v, 2)


def pick_day(a, b, concept):
    for _ in range(4):
        d = rdate(a, b)
        if concept in WEEKENDY and d.weekday() < 4 and rng.random() < .4:
            continue
        return d
    return d


def sample_concept(u, s):
    m = M[s.mid]
    if not m["multi"]:
        return s.concept
    items = [(c, w) for c, w in m["cw"].items() if (c not in KID_CONCEPTS or u.kids)]
    return wpick([c for c, _ in items], [w for _, w in items])


def sim(u):
    tx = []
    for si, s in enumerate(u.streams):
        start, end = max(s.start, u.start), min(s.end, Y1)
        if start > end:
            continue
        dates = []
        if s.kind == "rand":
            for mo in range(1, 13):
                if s.months and mo not in s.months:
                    continue
                a = max(dt.date(2025, mo, 1), start)
                b = min(dt.date(2025, mo, month_end(2025, mo)), end)
                if a > b:
                    continue
                frac = ((b - a).days + 1) / month_end(2025, mo)
                mid_d = a + (b - a) // 2
                k = poisson(s.rate * frac * season_mult(u, s, mid_d))
                for _ in range(k):
                    dates.append(pick_day(a, b, s.concept))
            if getattr(s, "first", None) and start <= s.first <= end:
                dates = [s.first] + [x for x in dates if x > s.first]
        elif s.kind == "monthly":
            for mo in range(1, 13):
                if s.months and mo not in s.months:
                    continue
                d = dt.date(2025, mo, min(s.day, month_end(2025, mo)))
                if start <= d <= end:
                    dates.append(d)
        elif s.kind in ("biweekly", "weekly"):
            step = 14 if s.kind == "biweekly" else 7
            d = s.anchor or dt.date(2025, 1, rng.randint(1, 7))
            while d <= end:
                if d >= start and (s.concept != "church" or rng.random() < .75):
                    dates.append(d)
                d += dt.timedelta(days=step)
        elif s.kind == "semimonthly":
            for mo in range(1, 13):
                for day in (15, month_end(2025, mo)):
                    d = dt.date(2025, mo, day)
                    while d.weekday() >= 5:
                        d -= dt.timedelta(days=1)
                    if start <= d <= end:
                        dates.append(d)
        elif s.kind == "dates":
            dates = [d for d in s.dates if start <= d <= end]
        for d in sorted(dates):
            concept = sample_concept(u, s)
            t = dict(date=d, si=si, mid=s.mid, concept=concept, flags=set(), acct=s.acct)
            t["payer"] = 0 if rng.random() < s.payer_p else 1
            t["who"] = s.who if s.who != "?" else (pick(u.kids) if u.kids else None)
            if s.kidflag and u.kids and t["who"] is None:
                t["who"] = pick(u.kids)
            if u.biz and s.mid in getattr(u, "biz_mixed", {}) and rng.random() < u.biz_mixed[s.mid]:
                t["concept"] = {"The Home Depot": "biz_supplies", "Lowe's": "biz_supplies"}.get(M[s.mid]["name"], "office")
                if len(u.accts) > 1 and rng.random() < .9:
                    t["acct"] = 1
            if getattr(s, "baby", None) and d >= s.baby and rng.random() < .4:
                t["concept"] = "kids_stuff"
            loc = s.locs[0] if len(s.locs) == 1 or rng.random() < .8 else s.locs[1]
            t["store"], t["city"], t["state"] = loc[0], loc[1], u.town["state"]
            t["amount"] = s.sign * amount_for(u, s, d, t["concept"])
            # seasons
            m = M[s.mid]
            if "g" in m["flags"] and dt.date(2025, 11, 15) <= d <= dt.date(2025, 12, 24) and s.sign > 0:
                if (u.xmas_cat or u.cmap.get("gifts") != u.cmap.get("*")) and rng.random() < .6:
                    t["concept"] = "gifts"
                    t["flags"].add("xmas")
            if "b" in m["flags"] and u.kids and dt.date(2025, 7, 28) <= d <= dt.date(2025, 9, 7) and s.sign > 0:
                if rng.random() < .6:
                    t["concept"] = "kids_stuff"
                    t["flags"].add("bts")
                    if t["who"] is None:
                        t["who"] = pick(u.kids)
            tx.append(t)
    # trip
    if u.trip:
        ts, te, dest = u.trip
        book = u.trip_book
        air = BY_NAME[pick(["Delta Air Lines", "United Airlines", "Southwest Airlines", "American Airlines"])]
        lodge = BY_NAME[pick(["Marriott", "Hilton", "Airbnb", "Airbnb", "Expedia"])]
        tstreams = {}

        def tstream(mid, concept):
            if mid not in tstreams:
                s = S()
                s.__dict__.update(new_stream(u, mid, concept, "none", start=Y1 + dt.timedelta(days=1)).__dict__)
                s.locs = [(rng.randint(100, 9999), dest["name"] if rng.random() < .6 else pick(dest["subs"]))]
                u.streams[-1] = s
                tstreams[mid] = len(u.streams) - 1
            return tstreams[mid]

        def tadd(mid, concept, d, amt):
            si = tstream(mid, concept)
            s = u.streams[si]
            t = dict(date=d, si=si, mid=mid, concept=concept, flags={"trip"}, acct=0, payer=0 if rng.random() < .6 else 1,
                     who=None, store=s.locs[0][0], city=s.locs[0][1], state=dest["state"] if "s" in M[mid]["flags"] or M[mid]["local"] else u.town["state"],
                     amount=amt)
            tx.append(t)

        if book >= u.start:
            tadd(air["id"], "travel", book, round(rng.uniform(180, 900), 2))
        tadd(lodge["id"], "travel", ts if lodge["name"] != "Airbnb" else book, round(rng.uniform(250, 1600), 2))
        d = ts
        while d <= te:
            for _ in range(rng.randint(2, 4)):
                concept = pick(["dining", "dining", "fast_food", "coffee", "entertainment", "gas", "groceries", "alcohol"])
                cands = [mid for mid in BY_CONCEPT.get(concept, []) if available(M[mid], dest) and "o" not in M[mid]["flags"]]
                loc = [mid for mid in cands if M[mid]["local"] or "US" not in M[mid]["where"]]
                pool = loc if loc and rng.random() < .6 else cands
                if not pool:
                    continue
                mid = pick(pool)
                m = M[mid]
                amt = round(max(m["lo"], min(m["hi"], math.exp(rng.uniform(math.log(m["lo"]), math.log(m["hi"]))))), 2)
                tadd(mid, concept, d, amt)
            d += dt.timedelta(days=1)
    # refunds
    extra = []
    for t in tx:
        if t["amount"] > 0 and t["concept"] in REFUNDABLE and u.streams[t["si"]].kind == "rand" \
                and not M[t["mid"]]["p2p"] and rng.random() < .04:
            d = t["date"] + dt.timedelta(days=rng.randint(2, 21))
            if d <= Y1:
                r = dict(t)
                r["flags"] = set(t["flags"]) | {"refund"}
                r["date"] = d
                r["amount"] = -(t["amount"] if rng.random() < .65 else round(t["amount"] * rng.uniform(.2, .7), 2))
                r["orig_date"] = t["date"]
                extra.append(r)
    tx += extra
    rng.shuffle(tx)
    tx.sort(key=lambda t: t["date"])
    return tx


# ---------------- filing ----------------
def follow(u, name, d):
    seen = 0
    while not active(u.catidx[name], d):
        if name not in u.succ:
            return None
        name = u.succ[name]
        seen += 1
        if seen > 5:
            return None
    return name


def latest_ev(evs, d):
    r = None
    for e in evs:
        if e[0] <= d:
            r = e
    return r


def intend(u, t):
    d, c, mid = t["date"], t["concept"], t["mid"]
    how = "base"
    name = None
    s = u.streams[t["si"]]
    if "trip" in t["flags"] and u.trip_cat and active(u.catidx[u.trip_cat], d):
        name, how = u.trip_cat, "trip"
    elif "xmas" in t["flags"] and u.xmas_cat and active(u.catidx[u.xmas_cat], d):
        name, how = u.xmas_cat, "seasonal"
    elif "bts" in t["flags"] and u.bts_cat and active(u.catidx[u.bts_cat], d):
        name, how = u.bts_cat, "seasonal"
    if name is None:
        ev = latest_ev(u.mev.get(mid, []), d)
        if ev and c == s.concept:
            name, how = ev[1], ev[2]
    if name is None and mid in u.msplit:
        name, how = u.msplit[mid].get(c, u.msplit[mid]["*"]), "split"
    if name is None and u.base == "family_whom":
        if (c in KID_CONCEPTS or s.kidflag or "bts" in t["flags"]) and t["who"] in u.kidcat:
            name, how = u.kidcat[t["who"]], "for_whom"
        elif c in PERSONAL and t["payer"] in u.adultcat:
            name, how = u.adultcat[t["payer"]], "for_whom"
        elif c == "pet" and u.petcat:
            name, how = u.petcat, "for_whom"
    if name is None and u.base == "couple":
        if c in PERSONAL and u.funcat:
            name, how = u.funcat[t["payer"]], "couple_split"
        elif t["payer"] == 1 and c in u.p2_over:
            name, how = u.p2_over[c], "couple_split"
    if name is None and c == "income" and u.base == "couple" and f"{u.adults[0]} Paycheck" in u.catidx:
        name, how = f"{u.adults[t['payer']]} Paycheck", "income"
    if name is None:
        ev = latest_ev(u.cev.get(c, []), d)
        if ev:
            name, how = ev[1], ev[2]
    if name is None:
        name = u.cmap.get(c) or u.cmap.get("household")
    if not active(u.catidx[name], d) and u.catidx[name]["start"] > d:
        name = u.cmap.get(c) or u.cmap.get("household")
    f = follow(u, name, d)
    if f is None:
        f = follow(u, u.cmap.get("household") or u.cats[0]["name"], d) or options_on(u, d)[0]
    return f, how


def mistake(u, t, intended):
    opts = options_on(u, t["date"])
    inc = t["amount"] < 0 and t["concept"] in INCOME_CONCEPTS
    pool = [o for o in opts if o != intended and u.catidx[o]["sem"] != {"savings"}
            and (u.catidx[o]["group"] == "income") == inc]
    if not pool:
        return intended
    i = opts.index(intended)
    near = [o for o in pool if abs(opts.index(o) - i) <= 2]
    if near and rng.random() < .7:
        return pick(near)
    return pick(pool)


# ---------------- rendering ----------------
def desc_at(m, d):
    ver, known_changes, unknown = 0, 0, 0
    descs = m["versions"][0][1]
    for i, (fd, ds, known) in enumerate(m["versions"]):
        if d >= fd:
            descs, ver = ds, i
            if i > 0 and not known:
                unknown += 1
    return pick(descs), ver, unknown


def expand(desc, t):
    return (desc.replace("{ref}", alnum(9)).replace("{n9}", digits(9)).replace("{n4}", digits(4))
            .replace("{dd}", pick(DD_NAMES)).replace("{shop}", pick(SHOPS)).replace("{event}", pick(["TRIVIA NIGHT", "RUN CLUB 5K", "POTTERY CLASS", "COMEDY SHOW"]))
            .replace("{gfm}", pick(["HELP THE", "SUPPORT", "MEMORIAL"])))


def fmt_store(m, n):
    return [f" #{n}", f" {n:05d}", f" {n}", f" #{n:04d}"][m["sfmt"]]


def render(u, t):
    m = M[t["mid"]]
    s = u.streams[t["si"]]
    acct = u.accts[t["acct"]]
    style = acct["style"]
    d = t["date"]
    auth = d - dt.timedelta(days=rng.choice([0, 0, 1, 2]))
    raw, ver, unknown = desc_at(m, d)
    desc = expand(raw, t)
    last4 = acct["cards"][t["payer"]]
    card = None
    key = (m["id"], unknown)
    if m["p2p"]:
        app, person, sign = m["p2p"]
        P = person.upper()
        out = s.sign > 0
        named = True
        if app == "Venmo":
            if not out or not u.p2p_named or style == "clean":
                named = False
                text = {"clean": "Venmo"}.get(style, (f"VENMO PAYMENT {digits(10)} WEB ID: 3264681992" if out else f"VENMO CASHOUT {digits(10)} WEB ID: 3264681992"))
            else:
                text = f"VENMO *{P.replace(' ', '')[:14]}" + ("" if style in ("clean", "trunc") else f" Visa Direct NY")
        elif app == "Zelle":
            first, last = person.split(" ", 1)
            if style == "clean":
                text = f"Zelle {'to' if out else 'from'} {person}"
            elif style == "wf":
                text = f"ZELLE {'TO' if out else 'FROM'} {last.upper()} {first.upper()} ON {d:%m/%d} REF # PP0{alnum(7)}"
            elif style == "cu":
                text = f"ONLINE TRANSFER {'TO' if out else 'FROM'} {first[0].upper()} {last.upper()}"
            else:
                text = f"ZELLE PAYMENT {'TO' if out else 'FROM'} {P} {alnum(10)}"
        elif app == "Cash App":
            if u.p2p_named:
                text = f"CASH APP*{P[:14]}" + ("" if style == "clean" else " OAKLAND CA")
            else:
                named = False
                text = "Cash App" if style == "clean" else f"SQ *CASH APP {digits(6)} CA"
        else:
            text = f"PAYPAL *{P.replace(' ', '')[:10]}" + ("" if style == "clean" else " 402-935-7733 CA")
        if not named:
            key = ("P2P", app, 1 if out else -1)
        return text, None, key
    clean = m["clean"]
    if s.income or m["primary"] in INCOME_CONCEPTS:
        ppd = digits(10)
        if m["name"] == "Interest":
            text = {"clean": "Interest Earned", "wf": "INTEREST PAYMENT", "cu": "DIVIDEND/INTEREST"}.get(style, "INTEREST PAID")
        else:
            text = {"upper": f"{desc} DIRECT DEP PPD ID: {ppd}", "pos": f"ACH CREDIT {desc}",
                    "dcp": f"DIRECT DEPOSIT {desc} {d:%m%d%y}", "wf": f"{desc} {d:%y%m%d} {digits(9)} {u.stmt_name}",
                    "clean": clean, "trunc": f"{desc[:22]} DIR DEP", "cu": f"DEPOSIT ACH {desc} TYPE: DEPOSIT ID: {ppd}"}[style]
        return text, None, key
    if s.ach:
        achw = getattr(s, "achw", None)
        if achw is None:
            s.achw = achw = pick(["ONLINE PMT", "AUTOPAY", "BILL PAYMT", "WEB PMTS", "PAYMENT"])
            s.ref = digits(pick([6, 8, 10]))
        text = {"upper": f"{desc} {achw} {s.ref}", "pos": f"ACH DEBIT {desc} {achw}",
                "dcp": f"ELECTRONIC WITHDRAWAL {desc} {achw} PPD ID: {s.ref}",
                "wf": f"{desc} {achw} {d:%y%m%d} {s.ref} {u.stmt_name}", "clean": clean,
                "trunc": f"{desc[:22]} {achw}", "cu": f"WITHDRAWAL ACH {desc} TYPE: {achw} ID: {s.ref}"}[style]
        return text, None, key
    online = "o" in m["flags"]
    if online:
        tail = " " + m["tail"]
        stp = ""
    else:
        CITY = t["city"].upper()
        tail = f" {CITY} {t['state']}"
        stp = fmt_store(m, t["store"]) if (not m["local"] and "s" in m["flags"]) else ""
    base = f"{desc}{stp}"
    if style == "upper":
        text = f"{base}{tail}"
    elif style == "pos":
        text = f"POS DEBIT {auth:%m%d} {base}{tail}"
    elif style == "dcp":
        text = f"DEBIT CARD PURCHASE XXXXX{last4} {desc}{tail}"
        card = last4
    elif style == "wf":
        text = f"PURCHASE AUTHORIZED ON {auth:%m/%d} {desc}{tail} S{digits(15)} CARD {last4}"
        card = last4
    elif style == "clean":
        text = clean
    elif style == "trunc":
        text = f"{base[:22]:<22} {tail.strip()}"
    else:
        text = f"POS WITHDRAWAL {base}{tail} ON {auth:%m/%d}"
    if t["amount"] < 0:
        text = {"upper": "RETURN ", "pos": "POS CREDIT ", "dcp": "", "wf": "", "clean": "", "trunc": "CR ",
                "cu": "POS RETURN "}[style] + text
        if style == "dcp":
            text = text.replace("DEBIT CARD PURCHASE", "DEBIT CARD CREDIT")
        if style == "wf":
            text = text.replace("PURCHASE AUTHORIZED", "PURCHASE RETURN AUTHORIZED")
    return text, card, key


def world_concept(m):
    if m["p2p"] or m["opaque"]:
        return None
    return m["primary"]


# --------------------------------------------------------------------------------------
# Build population
# --------------------------------------------------------------------------------------
ARCH_COUNTS = [("couple", 38), ("family_whom", 30), ("small_biz", 30), ("student", 25), ("retiree", 25),
               ("splitter", 20), ("minimalist", 20), ("coiner", 22), ("single", 10), ("newcomer", 30)]
archs = [a for a, n in ARCH_COUNTS for _ in range(n)]
rng.shuffle(archs)
town_names = sorted(TOWN)
users = []
for uid in range(N_USERS):
    arch = archs[uid]
    # cluster users into towns (some towns busier) so locals are shared
    tn = town_names[min(int(rng.paretovariate(1.2)) - 1, len(town_names) - 1)] if rng.random() < .35 else pick(town_names)
    u = make_user(uid, arch, TOWN[tn])
    build_streams(u)
    build_categories(u)
    build_events(u)
    u.tx = sim(u)
    for i, t in enumerate(u.tx):
        t["i"] = i
        t["intended"], t["how"] = intend(u, t)
        t["mistake"] = False
        t["filed"] = t["intended"]
        if rng.random() < u.sloppy:
            f = mistake(u, t, t["intended"])
            if f != t["intended"]:
                t["filed"], t["mistake"] = f, True
        t["text"], t["card"], t["key"] = render(u, t)
        t["wc"] = world_concept(M[t["mid"]])
        t["sim"] = M[t["mid"]]["primary"] if not M[t["mid"]]["p2p"] else u.streams[t["si"]].concept
        assert active(u.catidx[t["filed"]], t["date"]) and active(u.catidx[t["intended"]], t["date"])
    u.by_mid = defaultdict(list)
    for t in u.tx:
        u.by_mid[t["mid"]].append(t["i"])
    users.append(u)

# other users' filings: merchant -> list of (first date, user, category)
first_filing = defaultdict(dict)
for u in users:
    for t in u.tx:
        k = (u.id, t["filed"])
        if k not in first_filing[t["mid"]]:
            first_filing[t["mid"]][k] = t["date"]


def others_line(u, t):
    ff = first_filing.get(t["mid"], {})
    per_user = defaultdict(set)
    for (uid, cat), d in ff.items():
        if uid != u.id and d < t["date"]:
            per_user[uid].add(cat)
    if len(per_user) < 3:
        return None
    cnt = Counter()
    for uid in sorted(per_user):
        for cat in per_user[uid]:
            cnt[cat] += 1
    top = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))[:3]
    return "Other users file this payee as: " + ", ".join(f"{c} ({n})" for c, n in top)


# --------------------------------------------------------------------------------------
# Query tagging, slices, oracle
# --------------------------------------------------------------------------------------
def tag_for(u, t):
    m = M[t["mid"]]
    s = u.streams[t["si"]]
    prev = [u.tx[j] for j in u.by_mid[t["mid"]] if j < t["i"]]
    ans = t["intended"]
    d = t["date"]
    c = u.catidx[ans]
    if "refund" in t["flags"]:
        return "refund"
    if t["amount"] < 0 and t["concept"] in INCOME_CONCEPTS:
        return "income"
    if t["amount"] < 0:
        return "reimbursement"
    if u.arch == "newcomer":
        return "new_user"
    if c["born"] == "event" and (d - c["start"]).days <= 120 and not any(p["intended"] == ans for p in prev):
        return "new_category"
    if prev and prev[-1]["intended"] != ans and not active(u.catidx[prev[-1]["intended"]], d) \
            and follow(u, prev[-1]["intended"], d) == ans:
        return "renamed_category"
    if t["how"] == "changed_mind" and any(p["intended"] != ans for p in prev):
        return "changed_mind"
    if any(p["mistake"] for p in prev[-3:]):
        return "misfiled_history"
    if not prev:
        return "first_time_payee"
    if t["how"] == "trip":
        return "trip"
    if t["how"] == "seasonal" or (("xmas" in t["flags"] or "bts" in t["flags"])
                                  and ans not in {p["intended"] for p in prev}):
        return "seasonal"
    if prev[-1]["key"] != t["key"] or desc_at(m, prev[-1]["date"])[1] != desc_at(m, d)[1]:
        return "string_changed"
    if s.recurring and s.price and abs(prev[-1]["amount"] - t["amount"]) > 0.005 and "k" in m["flags"]:
        return "price_change"
    if t["key"][0] == "P2P":
        return "p2p_anon"
    if t["how"] == "couple_split":
        return "couple_split"
    if t["how"] == "for_whom" and len({p["intended"] for p in prev}) > 1:
        return "for_whom"
    if len({p["intended"] for p in prev} | {ans}) > 1:
        return "multi_purpose"
    if s.recurring:
        return "recurring"
    return "plain"


TAG_W = {"plain": 0.35, "recurring": 0.45, "first_time_payee": 1.2, "income": 0.9, "multi_purpose": 1.2,
         "new_user": 1.0, "misfiled_history": 1.3}


def build_slice(u, t, opts):
    qi = t["i"]
    chosen = []
    cs = set()

    def add(j):
        if j not in cs and len(chosen) < MAX_SHOTS:
            cs.add(j)
            chosen.append(j)

    own = [j for j in u.by_mid[t["mid"]] if j < qi][-6:]
    for j in own:
        add(j)
    per, n_sim = Counter(), 0
    for j in range(qi - 1, -1, -1):
        if n_sim >= 6:
            break
        r = u.tx[j]
        if r["mid"] != t["mid"] and r["sim"] == t["sim"] and per[r["mid"]] < 2:
            per[r["mid"]] += 1
            n_sim += 1
            add(j)
    shown = {u.tx[j]["filed"] for j in chosen}
    need = [o for o in opts if o not in shown]
    latest = {}
    for j in range(qi - 1, -1, -1):
        f = u.tx[j]["filed"]
        if f in need and f not in latest:
            latest[f] = j
            if len(latest) == len(need):
                break
    for j in sorted(latest.values(), reverse=True):
        add(j)
    for j in range(qi - 1, -1, -1):
        if len(chosen) >= MAX_SHOTS:
            break
        add(j)
    return sorted(chosen)


def fmt_row(r):
    return f"Transaction: {r['date'].isoformat()} | {r['text']} | ${r['amount']:.2f} | {WD[r['date'].weekday()]}"


def nearest(rows, amt):
    best, bd = None, None
    for r in rows:
        dd = abs(math.log(abs(r["amount"]) + 1) - math.log(abs(amt) + 1))
        if bd is None or dd <= bd:
            best, bd = r, dd
    return best["filed"]


def oracle(u, t, rows, opts):
    idx = {o: i for i, o in enumerate(opts)}
    sem = {o: u.catidx[o]["sem"] for o in opts}
    m = M[t["mid"]]
    d = t["date"]
    wc = t["wc"]
    giftable, bts = "g" in m["flags"], "b" in m["flags"]
    bill = u.streams[t["si"]].recurring or u.streams[t["si"]].ach
    # seasonal and trip context
    xmas = [o for o in opts if sem[o] and "xmas" in sem[o]]
    btsc = [o for o in opts if sem[o] and "bts" in sem[o]]
    tripc = [o for o in opts if sem[o] and "trip" in sem[o]]
    if xmas and giftable and dt.date(2025, 11, 10) <= d <= dt.date(2025, 12, 31) and t["amount"] > 0:
        return idx[xmas[0]]
    if btsc and bts and dt.date(2025, 7, 25) <= d <= dt.date(2025, 9, 10) and t["amount"] > 0:
        return idx[btsc[0]]
    if tripc and not bill:
        recent = [r for r in rows if r["filed"] == tripc[0] and 0 <= (d - r["date"]).days <= 10]
        if recent and (wc in (None, "dining", "fast_food", "coffee", "entertainment", "gas", "groceries", "alcohol", "travel")):
            if t["key"] not in {r["key"] for r in rows if r["filed"] != tripc[0]}:
                return idx[tripc[0]]
    same = [r for r in rows if r["key"] == t["key"]]
    if t["card"] and same:
        sc = [r for r in same if r["card"] == t["card"]]
        if sc:
            same = sc
    cand = None
    if same:
        if t["amount"] < 0 and wc not in INCOME_CONCEPTS:
            pos = [r for r in same if r["amount"] > 0]
            ex = [r for r in pos if abs(r["amount"] + t["amount"]) < 0.005]
            if ex:
                cand = ex[-1]["filed"]
            elif pos:
                cand = nearest(pos, t["amount"])
            else:
                cand = same[-1]["filed"]
        else:
            sgn = [r for r in same if (r["amount"] < 0) == (t["amount"] < 0)] or same
            cats = [r["filed"] for r in sgn]
            cc = Counter(cats)
            if len(cc) == 1:
                cand = cats[0]
            elif m["multi"] or sum(1 for v in cc.values() if v >= 2) >= 2 or t["key"][0] == "P2P":
                cand = nearest(sgn, t["amount"])
            else:
                last = cats[-1]
                if len(cats) >= 2 and cats[-2] == last:
                    cand = last
                else:
                    maj = max(cc.items(), key=lambda x: (x[1], max(i for i, c in enumerate(cats) if c == x[0])))[0]
                    cand = maj
                    if last != maj and len(sgn) >= 2:
                        first_seen = min(r["date"] for r in rows if r["filed"] == last)
                        if first_seen >= sgn[-2]["date"]:
                            cand = last
        # a more specific category that just started being used
        if cand in idx and wc and sem.get(cand):
            lastd = same[-1]["date"]
            for o in opts:
                so = sem[o]
                if o != cand and so and wc in so and len(so) < len(sem[cand]) and "*" not in so:
                    orows = [r for r in rows if r["filed"] == o]
                    if orows and all(r["date"] > lastd for r in orows):
                        cand = o
                        break
        if cand in idx:
            return idx[cand]
    if wc is None:
        return -1
    split = [o for o in opts if u.catidx[o]["merchant"] == t["mid"]]
    if split:
        for o in split:
            if wc in (sem[o] or ()):
                return idx[o]
        return idx[split[0]]
    ev = [r for r in rows if r["wc"] == wc and r["filed"] in idx and (r["amount"] < 0) == (t["amount"] < 0)]
    if ev:
        cc = Counter(r["filed"] for r in ev)
        top = max(cc.items(), key=lambda x: (x[1], max(i for i, r in enumerate(ev) if r["filed"] == x[0])))[0]
        return idx[top]
    cov = [(len(sem[o]), o) for o in opts if sem[o] and wc in sem[o] and "*" not in sem[o]]
    if cov:
        cov.sort(key=lambda x: x[0])
        if len(cov) == 1 or cov[0][0] < cov[1][0]:
            return idx[cov[0][1]]
        return -1
    mis = [o for o in opts if sem[o] == {"*"}]
    if mis:
        return idx[mis[0]]
    return -1


items = []
HAS_PREV = []
for u in users:
    if u.arch == "newcomer":
        cands = list(u.tx)
    else:
        span = (Y1 - u.start).days
        qstart = u.start + dt.timedelta(days=int(span * 0.6))
        cands = [t for t in u.tx if t["date"] >= qstart]
    tagged = [(t, tag_for(u, t)) for t in cands]
    w = [TAG_W.get(tag, 3.0) for _, tag in tagged]
    picked, per_m = [], Counter()
    order = wsample(list(range(len(tagged))), w, len(tagged))
    for k in order:
        t, tag = tagged[k]
        if per_m[t["mid"]] >= 2:
            continue
        per_m[t["mid"]] += 1
        picked.append((t, tag))
        if len(picked) >= Q_PER_USER:
            break
    picked.sort(key=lambda x: x[0]["i"])
    for n, (t, tag) in enumerate(picked):
        d = t["date"]
        opts = options_on(u, d)
        assert len(set(opts)) == len(opts) and all(", " not in o for o in opts)
        assert t["intended"] in opts
        sl = build_slice(u, t, opts)
        rows = [u.tx[j] for j in sl]
        for r in rows:
            assert r["date"] <= d, "future row"
            assert active(u.catidx[r["filed"]], r["date"]), "category not held on row date"
        head = "Categories: " + ", ".join(opts) + "\n\n"

        def shown(name, opts=opts):  # row 145 (main session): as in YNAB, a renamed or merged category shows its current name in history
            seen = set()
            while name not in opts and name in u.succ and name not in seen:
                seen.add(name); name = u.succ[name]
            return name
        body = "".join(fmt_row(r) + f"\nCategory: {shown(r['filed'])}\n\n" for r in rows)
        qline = fmt_row(t) + "\nCategory:"
        prompt = head + body + qline
        ol = others_line(u, t)
        prompt_others = head + body + (ol + "\n\n" if ol else "") + qline
        prev = [u.tx[j] for j in u.by_mid[t["mid"]] if j < t["i"]]
        ph = sorted({opts.index(shown(p["filed"])) for p in prev if shown(p["filed"]) in opts})
        yr = -1
        if prev:
            last3 = [shown(p["filed"]) for p in prev[-3:]]  # row 145: current names, as YNAB
            cc = Counter(last3)
            top, cnt = cc.most_common(1)[0]
            rule = top if cnt >= 2 else last3[-1]
            yr = opts.index(rule) if rule in opts else -1
        best = oracle(u, t, rows, opts)
        HAS_PREV.append(bool(prev))
        items.append({
            "id": f"B3:{u.id}:{n}", "user": u.id, "level": f"B3_{u.arch}", "merchant": M[t["mid"]]["name"],
            "text": t["text"], "amount": float(t["amount"]), "weekday": WD[d.weekday()], "date": d.isoformat(),
            "options": [" " + o for o in opts], "answer": opts.index(t["intended"]), "prompt": prompt,
            "prompt_ctx": prompt, "prompt_others": prompt_others, "why": tag, "best": best, "payee_hist": ph,
            "ynab_rule": yr, "hist_len": t["i"], "in_shots": any(r["mid"] == t["mid"] for r in rows),
        })

# final checks
for it in items:
    assert 0 <= it["answer"] < len(it["options"])
    assert -1 <= it["best"] < len(it["options"])
    assert 5 <= len(it["options"]) <= 45, len(it["options"])
    assert it["prompt"].endswith("Category:")

blob = json.dumps(items, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
sha = hashlib.sha256(blob.encode("utf-8")).hexdigest()
NOTES = ("Blind v3: synthetic one-year (2025) ledgers for 250 users in 20 US towns across five regions, built from a "
         "universe of about 350 real national and regional chains, billers and income sources plus about 1,000 invented "
         "local businesses (some with opaque holding-company names) shared by users of the same town, and per-user P2P "
         "counterparties (Venmo, Zelle, Cash App, PayPal; Venmo debits often without counterparty). Archetypes stress "
         "couples sharing one account whose second card holder files some payees differently (card number visible only "
         "in some bank styles), families filing by who the spending was for (kid and parent categories), small-business "
         "owners mixing business and personal purchases, students, retirees, detailed splitters (Costco - food vs "
         "Costco - household), minimalists with merged categories, coined category names, and brand-new users who join "
         "in Oct-Dec. Timelines include mid-year category creation (subscriptions, coffee, pets, renovations, babies, "
         "Christmas, back to school, named trips), renames and merges, changes of mind, subscription price changes, "
         "statement strings that change (processor switches, rebrands, billing portals), refunds, reimbursements, and "
         "filing mistakes in the history at 0.5-10% per user; answers are always the intended category. Seven bank "
         "rendering styles. Queries are about 6 per user from the last 40% of each timeline (whole timeline for new "
         "users), sampled to over-represent hard cases. 'best' is a rule-based reader that sees only the prompt plus "
         "world knowledge of merchants.")
out = {"name": "blind", "version": "v3", "n_users": len({it["user"] for it in items}), "items": items, "sha256": sha,
       "notes": NOTES}
dest = Path(__file__).resolve().parents[1] / "data" / "processed"
dest.mkdir(parents=True, exist_ok=True)
(dest / "blind_v3.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

# summary
print(f"items {len(items)}  users {out['n_users']}  sha256 {sha}")
byw = defaultdict(list)
for it in items:
    byw[it["why"]].append(it["best"] == it["answer"])
for w in sorted(byw, key=lambda k: -len(byw[k])):
    v = byw[w]
    print(f"  {w:18s} {len(v):5d}  ceiling {sum(v) / len(v):.3f}")
allv = [it["best"] == it["answer"] for it in items]
print(f"overall ceiling {sum(allv) / len(allv):.3f}")
print(f"payee_hist empty {sum(1 for it in items if not it['payee_hist']) / len(items):.3f}")
wh = [it for it, hp in zip(items, HAS_PREV) if hp]
print(f"answer != ynab_rule among items with payee history {sum(1 for it in wh if it['answer'] != it['ynab_rule']) / max(1, len(wh)):.3f}")
print(f"with 'Other users' line {sum(1 for it in items if it['prompt_others'] != it['prompt']) / len(items):.3f}")
