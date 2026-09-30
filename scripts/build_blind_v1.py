#!/usr/bin/env python3
"""Build blind_v1: an independent test set for a per-user transaction categoriser.

Written blind (without reading any other generator in this repo). It simulates about 250
budgeting-app users over calendar 2025: their own category lists, the merchants they use, how
their banks render statement strings, their recurring bills, habits, trips, new categories,
changes of mind and filing mistakes. Queries are drawn from the later part of each timeline;
each prompt carries a history slice built the way production would build it.

    uv run python scripts/build_blind_v1.py

Standard library only; every random draw comes from a seeded random.Random.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import math
import random
import re
from collections import Counter
from pathlib import Path

SEED = 20250701
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "blind_v1.json"
Y0 = dt.date(2025, 1, 1)
Y1 = dt.date(2025, 12, 31)
QUERY_FROM = dt.date(2025, 8, 1)
WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
N_FULL, N_SHORT = 222, 28
Q_PER_USER = 6
SLICE_MAX, OWN_MAX, SIM_MAX, SIM_PER_PAYEE = 24, 6, 6, 2
# row 111 (added by the main session, not the blind builder): BLIND_B48=1 writes blind_v1_b48.json, the same items with a 48-row slice
# (similar payees up to 10); the payee's own rows stay at OWN_MAX so every item's answer and ideal-reader answer are unchanged
if os.environ.get("BLIND_B48") == "1":
    SLICE_MAX, SIM_MAX = 48, 10
    OUT = OUT.with_name("blind_v1_b48.json")
# row 117 (added by the main session; owner 2026-09-29: one user's 10 to 100 new transactions at once). Both switches leave blind_v1's
# items, rows and random draws unchanged:
#   BLIND_SPLIT=1  blind_v1_split.json: blind_v1's items with the same slice rows, the payee's own and similar payees' rows moved after
#                  the others under a header (the shared block first, so one user's transactions could share it)
#   BLIND_BULK=1   blind_bulk_v1.json and blind_bulk_v1_split.json: per full user one sync (a cutoff on or after 2025-08-01, the next
#                  BULK_N transactions, all unfiled; the history is what was filed before the cutoff); one shared block per sync (a row
#                  per category, then the latest rows, SHARED_MAX in all) and per transaction the payee's own and similar payees' rows
#                  before the cutoff; the plain file dates-sorts the union (today's layout), the split file puts the shared block first
SPLIT = os.environ.get("BLIND_SPLIT") == "1"
BULK = os.environ.get("BLIND_BULK") == "1"
BULK_N, SHARED_MAX, SHARED_CAT_MAX = 30, 24, 18
SHARED_HEAD, NEAR_HEAD = "Earlier transactions:", "Earlier transactions at this payee and similar payees:"
OTHERS = os.environ.get("BLIND_OTHERS") == "1"  # row 124 (main session): blind_v1_others.json, blind_v1's items with a line before the query
# naming the categories other users filed this payee under before the query's date (the top three with counts; only when at least
# three such rows exist): "Other users file this payee as: Eating out (14), Restaurants (9), Dining out (3)"
OTHERS_HEAD = "Other users file this payee as: "
YNAB_RULE = os.environ.get("BLIND_YNAB") == "1"  # row 134 (owner, 2026-09-30): write blind_v1_ynabrule.json, YNAB's current suggestion per
# blind_v1 item from the payee's whole filed history before the query (payee identity exact): the category used in 2 of the payee's last 3
# transactions; with fewer than 3, the last one used; if the last 3 all differ, the last one used (assumed); no history, no suggestion
if SPLIT:
    OUT = OUT.with_name("blind_v1_split.json")
if OTHERS:
    OUT = OUT.with_name("blind_v1_others.json")
if BULK:
    OUT = OUT.with_name("blind_bulk_v1.json")
DIG = "0123456789"
ALNUM = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"


def D(m, d):
    return dt.date(2025, m, d)


def rand_date(rng, a, b):
    return a + dt.timedelta(days=rng.randint(0, (b - a).days))


# --------------------------------------------------------------------------------------
# Merchant universe
# --------------------------------------------------------------------------------------

class M:
    def __init__(self, mid, clean, kind, roles, tpls=(), ach=None, channel="card",
                 informative=False, opaque=False, local=False, key=None):
        self.mid, self.clean, self.kind = mid, clean, kind
        self.roles = roles            # role -> (weight, median amount, log sigma)
        self.tpls = list(tpls)        # card descriptor templates ("@" = no location suffix)
        self.ach = ach                # (company name, entry description) for ACH items
        self.channel = channel        # card | ach | p2p_venmo | p2p_zelle | xfer | check | fee
        self.informative = informative  # amount tells the roles apart (gas station snacks vs fuel)
        self.opaque = opaque          # a reader cannot tell what the business is
        self.local = local
        self.key = key                # lowercase word a category could be named after
        self.primary = max(roles, key=lambda r: roles[r][0]) if roles else None
        self.person = None


MERCH: dict[str, M] = {}


def reg(m):
    MERCH[m.mid] = m
    return m


def parse_roles(s):
    out = {}
    for part in s.split(","):
        p = part.split()
        out[p[0]] = (float(p[1]), float(p[2]), float(p[3]))
    return out


def C(mid, clean, kind, roles, tpls=(), **kw):
    return reg(M(mid, clean, kind, parse_roles(roles), tpls, **kw))


def A(mid, clean, kind, roles, co, des, **kw):
    return reg(M(mid, clean, kind, parse_roles(roles), (), ach=(co, des), channel="ach", **kw))


# groceries
C("kroger", "Kroger", "grocery", "groceries .88 68 .65, household .12 22 .6", ["KROGER #{n4}", "KROGER {n3}"])
C("safeway", "Safeway", "grocery", "groceries .9 64 .65, household .1 20 .6", ["SAFEWAY #{n4}", "SAFEWAY {n4}"])
C("publix", "Publix", "grocery", "groceries .9 66 .65, household .1 20 .6", ["PUBLIX #{n4}", "PUBLIX SUPER MAR {n4}"])
C("heb", "H-E-B", "grocery", "groceries .9 72 .6, household .1 22 .6", ["H-E-B #{n3}", "HEB FOOD {n3}"])
C("aldi", "Aldi", "grocery", "groceries 1 48 .55", ["ALDI {n5}", "ALDI #{n5}"])
C("traderjoes", "Trader Joe's", "grocery", "groceries 1 52 .55", ["TRADER JOE S #{n3}", "TRADER JOES #{n3}"])
C("wholefoods", "Whole Foods", "grocery", "groceries 1 58 .7", ["WHOLEFDS {abbr} {n5}", "WHOLE FOODS MKT {n5}"])
C("wegmans", "Wegmans", "grocery", "groceries .9 74 .6, household .1 20 .6", ["WEGMANS #{n3}"])
C("foodlion", "Food Lion", "grocery", "groceries 1 55 .6", ["FOOD LION #{n4}"])
C("meijer", "Meijer", "grocery", "groceries .7 75 .6, household .2 30 .6, clothing .1 25 .5", ["MEIJER # {n3}", "MEIJER STORE #{n3}"])
C("sprouts", "Sprouts", "grocery", "groceries 1 45 .6", ["SPROUTS FARMERS MAR #{n3}"])
C("hyvee", "Hy-Vee", "grocery", "groceries .9 70 .6, household .1 20 .5", ["HY-VEE {cityup} {n4}"])
C("lidl", "Lidl", "grocery", "groceries 1 46 .55", ["LIDL US #{n4}"])
C("gianteagle", "Giant Eagle", "grocery", "groceries 1 62 .6", ["GIANT EAGLE #{n4}"])
C("harristeeter", "Harris Teeter", "grocery", "groceries 1 60 .6", ["HARRIS TEETER #{n3}"])
C("fredmeyer", "Fred Meyer", "grocery", "groceries .75 72 .6, household .25 28 .6", ["FRED-MEYER #{n3}", "FRED MEYER {n5}"])
C("winco", "WinCo Foods", "grocery", "groceries 1 70 .6", ["WINCO FOODS #{n3}"])
C("jewel", "Jewel-Osco", "grocery", "groceries .9 62 .6, household .1 20 .6", ["JEWEL OSCO {n4}"])
C("stopshop", "Stop & Shop", "grocery", "groceries 1 70 .6", ["STOP & SHOP {n4}"])
# warehouse clubs and big boxes (multi-purpose)
C("costco", "Costco", "warehouse", "groceries .55 165 .5, household .3 85 .6, electronics .05 260 .6, clothing .05 38 .4, gifts .05 55 .5",
  ["COSTCO WHSE #{n4}", "COSTCO WHOLESALE #{n4}"], key="costco")
C("samsclub", "Sam's Club", "warehouse", "groceries .6 140 .5, household .3 70 .6, electronics .1 200 .6", ["SAMSCLUB #{n4}", "SAMS CLUB #{n4}"])
C("bjs", "BJ's Wholesale", "warehouse", "groceries .65 150 .5, household .35 70 .6", ["BJS WHOLESALE #{n4}"])
C("target", "Target", "bigbox", "household .35 32 .65, groceries .25 45 .6, clothing .15 38 .5, kids_stuff .1 28 .6, gifts .1 30 .6, electronics .05 70 .6",
  ["TARGET {n8}", "TARGET T-{n4}", "@TARGET.COM * 800-591-3869 MN"], key="target")
C("walmart", "Walmart", "bigbox", "groceries .5 78 .6, household .3 30 .7, clothing .08 25 .5, electronics .05 60 .7, kids_stuff .07 25 .6",
  ["WAL-MART #{n4}", "WM SUPERCENTER #{n4}", "@WALMART.COM 800-966-6546 AR"], key="walmart")
C("amazon", "Amazon", "online", "household .35 26 .75, electronics .12 45 .8, books .08 16 .5, kids_stuff .1 24 .6, gifts .12 32 .6, clothing .1 29 .5, pets .05 38 .5, hobbies .08 25 .6",
  ["@AMZN Mktp US*{ref} Amzn.com/bill WA", "@Amazon.com*{ref} Amzn.com/bill WA", "@AMAZON MKTPL*{ref} Amzn.com/bill WA"], key="amazon")
C("ebay", "eBay", "online", "hobbies .4 35 .8, electronics .3 60 .8, clothing .3 30 .6", ["@EBAY O*{r11}"])
C("etsy", "Etsy", "online", "gifts .6 34 .6, household .25 28 .6, hobbies .15 22 .5", ["@ETSY INC BROOKLYN NY", "@ETSY.COM {r9}"])
C("temu", "Temu", "online", "household .6 24 .6, clothing .4 20 .5", ["@TEMU.COM 8884958868 DE"])
C("shein", "Shein", "online", "clothing 1 35 .5", ["@SHEIN.COM 844-802-2500 CA"])
C("wayfair", "Wayfair", "online", "furniture .7 180 .7, household .3 60 .6", ["@WAYFAIR*{ref} 877-929-3247 MA"])
C("chewy", "Chewy", "online", "pets 1 48 .45", ["@CHEWY.COM 800-672-4399 FL"])
C("ikea", "IKEA", "furniture", "furniture .7 190 .7, household .3 40 .6", ["IKEA {cityup}"])
# sit-down restaurants
for mid, cl, t in [("olivegarden", "Olive Garden", ["OLIVE GARDEN {n4}", "OLIVE GARDEN 00{n4}"]),
                   ("chilis", "Chili's", ["CHILI'S #{n4}", "CHILIS {n4}"]),
                   ("applebees", "Applebee's", ["APPLEBEES {n4}"]),
                   ("texasroadhouse", "Texas Roadhouse", ["TEXAS ROADHOUSE #{n3}"]),
                   ("cheesecake", "The Cheesecake Factory", ["THE CHEESECAKE FACTORY #{n3}", "CHEESECAKE {cityup}"]),
                   ("redrobin", "Red Robin", ["RED ROBIN NO {n3}"]),
                   ("pfchangs", "P.F. Chang's", ["PF CHANGS #{n4}"]),
                   ("outback", "Outback Steakhouse", ["OUTBACK {n4}"]),
                   ("firstwatch", "First Watch", ["FIRST WATCH {n4}"])]:
    C(mid, cl, "restaurant", "dining 1 52 .42", t)
C("bww", "Buffalo Wild Wings", "restaurant", "dining .7 45 .4, bars .3 38 .4", ["BUFFALO WILD WINGS {n4}", "BWW {n4}"])
# quick service
for mid, cl, med, t in [("mcdonalds", "McDonald's", 11, ["MCDONALD'S F{n5}", "MCDONALDS {n5}"]),
                        ("chickfila", "Chick-fil-A", 13, ["CHICK-FIL-A #{n5}", "CHICK-FIL-A {n5}"]),
                        ("chipotle", "Chipotle", 14, ["CHIPOTLE {n4}", "@CHIPOTLE ONLINE 877-6-CHIPOTL CA"]),
                        ("tacobell", "Taco Bell", 10, ["TACO BELL #{n5}"]),
                        ("wendys", "Wendy's", 11, ["WENDYS #{n4}"]),
                        ("panera", "Panera Bread", 15, ["PANERA BREAD #{n6}"]),
                        ("subway", "Subway", 11, ["SUBWAY {n8}"]),
                        ("fiveguys", "Five Guys", 22, ["FIVE GUYS {st} {n4}"]),
                        ("sweetgreen", "Sweetgreen", 17, ["SWEETGREEN {cityup}"]),
                        ("jerseymikes", "Jersey Mike's", 13, ["JERSEY MIKES {n5}"]),
                        ("panda", "Panda Express", 12, ["PANDA EXPRESS #{n4}"]),
                        ("dominos", "Domino's", 26, ["DOMINO'S {n4}"]),
                        ("pizzahut", "Pizza Hut", 27, ["PIZZA HUT {n6}"]),
                        ("culvers", "Culver's", 14, ["CULVERS OF {cityup}"]),
                        ("sonic", "Sonic Drive-In", 10, ["SONIC DRIVE IN #{n4}"])]:
    C(mid, cl, "fastfood", f"fast_food 1 {med} .4", t)
C("doordash", "DoorDash", "delivery", "delivery 1 34 .4", ["@DOORDASH*{rest}", "@DD *DOORDASH {rest}"])
C("ubereats", "Uber Eats", "delivery", "delivery 1 32 .4", ["@UBER *EATS {ref}", "@UBER EATS HELP.UBER.COM CA"])
C("grubhub", "Grubhub", "delivery", "delivery 1 31 .4", ["@GRUBHUB*{rest}"])
# coffee
C("starbucks", "Starbucks", "coffee", "coffee 1 6.8 .35", ["STARBUCKS STORE {n5}", "STARBUCKS {n5}"])
C("dunkin", "Dunkin'", "coffee", "coffee 1 5.9 .35", ["DUNKIN #{n6}"])
C("dutchbros", "Dutch Bros", "coffee", "coffee 1 6.5 .3", ["DUTCH BROS #{n4}"])
C("peets", "Peet's Coffee", "coffee", "coffee 1 6.9 .3", ["PEETS #{n5}"])
# alcohol
C("totalwine", "Total Wine", "liquor", "alcohol 1 34 .55", ["TOTAL WINE AND MORE {n4}"])
C("abcwine", "ABC Fine Wine & Spirits", "liquor", "alcohol 1 29 .55", ["ABC FINE WINE #{n3}"])
C("binnys", "Binny's", "liquor", "alcohol 1 31 .55", ["BINNYS BEVERAGE {n2}"])
# fuel
for mid, cl, t in [("shell", "Shell", ["SHELL OIL {n11}", "SHELL SERVICE STATION"]),
                   ("chevron", "Chevron", ["CHEVRON {n7}"]),
                   ("exxon", "ExxonMobil", ["EXXON {n8}", "EXXONMOBIL {n8}"]),
                   ("bp", "BP", ["BP#{n7}"]),
                   ("sunoco", "Sunoco", ["SUNOCO {n10}"]),
                   ("marathon", "Marathon", ["MARATHON PETRO{n6}"]),
                   ("speedway", "Speedway", ["SPEEDWAY {n5}"]),
                   ("circlek", "Circle K", ["CIRCLE K {n5}"]),
                   ("valero", "Valero", ["VALERO {n4}"]),
                   ("costcogas", "Costco Gas", ["COSTCO GAS #{n4}"])]:
    C(mid, cl, "gas", "fuel 1 44 .3", t)
# fuel stations that are also a place to buy lunch and milk
for mid, cl, t in [("wawa", "Wawa", ["WAWA {n4}"]), ("sheetz", "Sheetz", ["SHEETZ {n4}"]),
                   ("quiktrip", "QuikTrip", ["QT {n3}", "QT {n3} OUTSIDE"]), ("caseys", "Casey's", ["CASEYS #{n4}"]),
                   ("bucees", "Buc-ee's", ["BUC-EE'S #{n2}"]), ("seveneleven", "7-Eleven", ["7-ELEVEN {n5}"]),
                   ("kwiktrip", "Kwik Trip", ["KWIK TRIP {n4}"])]:
    C(mid, cl, "gas_conv", "fuel .6 45 .3, fast_food .3 8 .5, groceries .1 14 .45", t, informative=True)
# pharmacies
for mid, cl, t in [("cvs", "CVS", ["CVS/PHARMACY #{n5}", "CVS/PHARM {n5}"]), ("walgreens", "Walgreens", ["WALGREENS #{n5}", "WAL-GREENS #{n5}"]),
                   ("riteaid", "Rite Aid", ["RITE AID {n5}"])]:
    C(mid, cl, "pharmacy", "pharmacy .5 16 .9, household .22 14 .6, fast_food .1 6 .5, personal_care .18 18 .5", t, informative=True)
# home
C("homedepot", "The Home Depot", "home_improve", "home_improve .85 48 .9, household .15 25 .6", ["THE HOME DEPOT #{n4}", "@HOMEDEPOT.COM 800-430-3376 GA"])
C("lowes", "Lowe's", "home_improve", "home_improve .85 52 .9, household .15 25 .6", ["LOWES #{n5}*"])
C("ace", "Ace Hardware", "home_improve", "home_improve 1 24 .7", ["ACE HARDWARE {n5}", "ACE HDWE {n5}"])
C("menards", "Menards", "home_improve", "home_improve .85 55 .9, household .15 25 .6", ["MENARDS {n4}"])
# clothing
for mid, cl, roles, t in [("oldnavy", "Old Navy", "clothing 1 42 .55", ["OLD NAVY US {n4}"]),
                          ("kohls", "Kohl's", "clothing .55 45 .5, household .25 35 .5, gifts .2 35 .5", ["KOHLS #{n4}"]),
                          ("tjmaxx", "TJ Maxx", "clothing .5 38 .5, household .3 30 .5, gifts .2 28 .5", ["TJMAXX #{n4}"]),
                          ("marshalls", "Marshalls", "clothing .5 38 .5, household .3 30 .5, gifts .2 28 .5", ["MARSHALLS #{n4}"]),
                          ("nordrack", "Nordstrom Rack", "clothing 1 64 .5", ["NORDSTROM RACK #{n4}"]),
                          ("nike", "Nike", "clothing 1 78 .5", ["@NIKE.COM 800-806-6453 OR"]),
                          ("gap", "Gap", "clothing 1 55 .5", ["GAP US {n4}"]),
                          ("ross", "Ross", "clothing .7 34 .5, household .3 22 .5", ["ROSS STORES #{n4}"]),
                          ("hm", "H&M", "clothing 1 40 .5", ["H&M {n4}"]),
                          ("uniqlo", "Uniqlo", "clothing 1 58 .5", ["@UNIQLO USA 877-486-4756 NY"])]:
    C(mid, cl, "clothing", roles, t)
C("bestbuy", "Best Buy", "electronics", "electronics 1 95 .9", ["BEST BUY {n8}", "@BESTBUY.COM {r9} MN"])
C("applestore", "Apple Store", "electronics", "electronics 1 160 .9", ["APPLE STORE #R{n3}"])
C("greatclips", "Great Clips", "personal_care", "haircut 1 24 .2", ["GREAT CLIPS #{n4}"])
C("supercuts", "Supercuts", "personal_care", "haircut 1 26 .2", ["SUPERCUTS {n4}"])
C("ulta", "Ulta Beauty", "personal_care", "personal_care 1 36 .55", ["ULTA #{n4}"])
C("sephora", "Sephora", "personal_care", "personal_care 1 48 .5", ["SEPHORA #{n4}"])
# subscriptions and memberships (recurring amounts set per user)
for mid, cl, kind, role, t in [
        ("planetfitness", "Planet Fitness", "fitness", "fitness", ["@PLANET FITNESS {st} 844-880-7180"]),
        ("orangetheory", "Orangetheory", "fitness", "fitness", ["CLUB OTF {cityup}"]),
        ("ymca", "YMCA", "fitness", "fitness", ["@YMCA OF GREATER {cityup}"]),
        ("peloton", "Peloton", "fitness", "fitness", ["@PELOTON* MEMBERSHIP 866-679-9129 NY"]),
        ("netflix", "Netflix", "streaming", "streaming", ["@NETFLIX.COM 866-579-7172 CA", "@NETFLIX.COM LOS GATOS CA"]),
        ("spotify", "Spotify", "streaming", "streaming", ["@SPOTIFY USA 877-7781161 NY", "@Spotify USA NEW YORK NY"]),
        ("hulu", "Hulu", "streaming", "streaming", ["@HULU 877-8244858 CA", "@HLU*HULUPLUS 877-8244858 CA"]),
        ("disney", "Disney+", "streaming", "streaming", ["@DISNEY PLUS 888-9057888 CA"]),
        ("max", "Max", "streaming", "streaming", ["@MAX.COM 855-4422629 NY"]),
        ("youtube", "YouTube Premium", "streaming", "streaming", ["@GOOGLE *YouTubePremium g.co/helppay# CA"]),
        ("peacock", "Peacock", "streaming", "streaming", ["@PEACOCK {r5} PREMIUM NY"]),
        ("paramount", "Paramount+", "streaming", "streaming", ["@PARAMOUNT+ 888-274-5343 CA"]),
        ("audible", "Audible", "streaming", "streaming", ["@Audible*{ref} Amzn.com/bill NJ"]),
        ("prime", "Amazon Prime", "streaming", "streaming", ["@Amazon Prime*{ref} Amzn.com/bill WA"]),
        ("kindle", "Kindle Unlimited", "streaming", "streaming", ["@Kindle Unltd*{ref} 888-802-3080 WA"]),
        ("adobe", "Adobe", "software", "software", ["@ADOBE *CREATIVE CLD 408-536-6000 CA", "@ADOBE *ADOBE 408-536-6000 CA"]),
        ("ms365", "Microsoft 365", "software", "software", ["@MICROSOFT*MICROSOFT 365 P 425-6816830 WA"]),
        ("dropbox", "Dropbox", "software", "cloud", ["@DROPBOX*{ref} DROPBOX.COM CA"]),
        ("openai", "ChatGPT", "software", "software", ["@OPENAI *CHATGPT SUBSCR OPENAI.COM CA"]),
        ("googleone", "Google One", "software", "cloud", ["@GOOGLE *Google One g.co/helppay# CA"]),
        ("nytimes", "New York Times", "software", "news", ["@NYTIMES*NYTIMES NEWS 800-698-4637 NY"]),
        ("patreon", "Patreon", "software", "entertainment", ["@PATREON* MEMBERSHIP INTERNET CA"]),
        ("duolingo", "Duolingo", "software", "software", ["@DUOLINGO *PLUS PITTSBURGH PA"]),
        ("coursera", "Coursera", "education", "education", ["@COURSERA 650-963-9884 CA"]),
        ("stjude", "St. Jude", "charity", "charity", ["@ST JUDE CHILDRENS RES 800-822-6344 TN"]),
        ("kumon", "Kumon", "kids_activity", "kids_activities", ["KUMON {cityup}"]),
        ("kindercare", "KinderCare", "childcare", "childcare", ["@KINDERCARE LEARNING 888-525-2780 OR"]),
        ("kiddie", "Kiddie Academy", "childcare", "childcare", ["@KIDDIE ACADEMY OF {cityup}"]),
        ("verizonw", "Verizon Wireless", "phone", "phone", ["@VZWRLSS*APOCC VISB 800-922-0204 FL", "@VERIZON WIRELESS PAYMENTS"]),
        ("tmobile", "T-Mobile", "phone", "phone", ["@T-MOBILE PCS SVC 877-453-1304 WA", "@TMOBILE*AUTO PAY 800-937-8997 WA"]),
        ("attw", "AT&T Wireless", "phone", "phone", ["@ATT*BILL PAYMENT 800-331-0500 TX"]),
        ("mint", "Mint Mobile", "phone", "phone", ["@MINT MOBILE 800-683-7392 CA"]),
        ("xfinity", "Xfinity", "internet", "internet", ["@COMCAST XFINITY 800-266-2278 PA", "@COMCAST CABLE COMM 800-XFINITY PA"]),
        ("spectrum", "Spectrum", "internet", "internet", ["@SPECTRUM 855-707-7328 MO", "@CHARTER SPECTRUM 877-772-2253"]),
        ("attnet", "AT&T Internet", "internet", "internet", ["@ATT*BILL PAYMENT 800-288-2020 TX"]),
        ("fios", "Verizon Fios", "internet", "internet", ["@VERIZON*FIOS 800-837-4966 NJ"]),
        ("gfiber", "Google Fiber", "internet", "internet", ["@GOOGLE *FIBER g.co/helppay# CA"]),
        ("geico", "GEICO", "insurance", "auto_ins", ["@GEICO *AUTO 800-841-3000 DC"]),
        ("progressive", "Progressive", "insurance", "auto_ins", ["@PROG DIRECT INS PREM 800-776-4737 OH"]),
        ("lemonade", "Lemonade", "insurance", "renters_ins", ["@LEMONADE INS 844-733-8666 NY"]),
        ("ezpass", "E-ZPass", "parking", "parking", ["@E-ZPASS{st} REBILL"]),
        ("sunpass", "SunPass", "parking", "parking", ["@SUNPASS*ACC{r8}"])]:
    C(mid, cl, kind, f"{role} 1 20 .3", t)
C("apple", "Apple", "software", "games .5 4.99 .6, software .5 6.99 .6", ["@APPLE.COM/BILL 866-712-7753 CA"])
# ACH billers
for mid, cl, kind, role, co, des in [
        ("duke", "Duke Energy", "electric", "electric", "DUKE ENERGY", "PAYMENT"),
        ("pge", "PG&E", "electric", "electric", "PGANDE", "WEB ONLINE"),
        ("coned", "Con Edison", "electric", "electric", "CON ED OF NY", "CECONY"),
        ("xcel", "Xcel Energy", "electric", "electric", "XCEL ENERGY-NSP", "XCELENERGY"),
        ("gapower", "Georgia Power", "electric", "electric", "GEORGIA POWER", "BILL PAY"),
        ("dominion", "Dominion Energy", "electric", "electric", "DOMINION ENERGY", "BILLPAY"),
        ("fpl", "FPL", "electric", "electric", "FPL", "DIRECT DEBIT"),
        ("aep", "AEP Ohio", "electric", "electric", "AEP OHIO", "WEB PMT"),
        ("srp", "SRP", "electric", "electric", "SRP", "ONLINE PMT"),
        ("comed", "ComEd", "electric", "electric", "COMED", "PAYMENTS"),
        ("pse", "Puget Sound Energy", "electric", "electric", "PUGET SOUND ENERG", "BILLPAY"),
        ("consumers", "Consumers Energy", "electric", "electric", "CONSUMERS ENERGY", "BILL PAY"),
        ("evergy", "Evergy", "electric", "electric", "EVERGY", "WEB PAY"),
        ("idahopower", "Idaho Power", "electric", "electric", "IDAHO POWER CO", "PAYMENT"),
        ("rmp", "Rocky Mountain Power", "electric", "electric", "ROCKY MTN POWER", "PAYMENT"),
        ("oppd", "OPPD", "electric", "electric", "OPPD", "ELEC PYMT"),
        ("peco", "PECO", "electric", "electric", "PECO ENERGY", "PAYMENTS"),
        ("pgeor", "Portland General Electric", "electric", "electric", "PORTLAND GENERAL", "PGE PAYMNT"),
        ("txu", "TXU Energy", "electric", "electric", "TXU ENERGY", "AUTOPAY"),
        ("ngrid", "National Grid", "natgas", "natgas", "NATIONAL GRID NY", "UTILITY PM"),
        ("centerpoint", "CenterPoint Energy", "natgas", "natgas", "CENTERPOINT ENERGY", "BILLPAY"),
        ("amwater", "American Water", "water", "water", "AMERICAN WATER", "PAYMENT"),
        ("piedmont", "Piedmont Natural Gas", "natgas", "natgas", "PIEDMONT NG", "DRAFT"),
        ("nicor", "Nicor Gas", "natgas", "natgas", "NICOR GAS", "BILL PAY"),
        ("socalgas", "SoCalGas", "natgas", "natgas", "SOCALGAS", "PAID SCGC"),
        ("columbiagas", "Columbia Gas", "natgas", "natgas", "COLUMBIA GAS OH", "PAYMENT"),
        ("wm", "Waste Management", "trash", "trash", "WASTE MANAGEMENT", "WM EZPAY"),
        ("republic", "Republic Services", "trash", "trash", "REPUBLIC SERVICES", "TRASH"),
        ("bilt", "Bilt Rent", "rent", "rent", "BILT RENT", "RENT PMT"),
        ("greystar", "Greystar", "rent", "rent", "GREYSTAR RE", "RENT PMT"),
        ("zillowrent", "Zillow Rent", "rent", "rent", "ZILLOW RENT", "PAYMENT"),
        ("rentcafe", "RentCafe", "rent", "rent", "YARDI RENTCAFE", "WEB PMTS"),
        ("rocket", "Rocket Mortgage", "mortgage", "mortgage", "ROCKET MORTGAGE", "PAYMENT"),
        ("mrcooper", "Mr. Cooper", "mortgage", "mortgage", "MR. COOPER", "MTG PMT"),
        ("wfhm", "Wells Fargo Home Mortgage", "mortgage", "mortgage", "WELLS FARGO HOME MTG", "AUTO PAY"),
        ("pennymac", "PennyMac", "mortgage", "mortgage", "PENNYMAC", "LOAN SVC"),
        ("loancare", "LoanCare", "mortgage", "mortgage", "LOANCARE", "PMT"),
        ("statefarm", "State Farm", "insurance", "auto_ins", "STATE FARM RO 27", "CPC-CLIENT"),
        ("allstate", "Allstate", "insurance", "auto_ins", "ALLSTATE INS CO", "INS PREM"),
        ("toyotafin", "Toyota Financial", "car_payment", "car_payment", "TOYOTA FINANCIAL", "PAYMENT"),
        ("ally", "Ally Auto", "car_payment", "car_payment", "ALLY", "ALLY PAYMT"),
        ("hondafin", "American Honda Finance", "car_payment", "car_payment", "AMERICAN HONDA FIN", "PAYMENT"),
        ("caponeauto", "Capital One Auto", "car_payment", "car_payment", "CAPITAL ONE AUTO", "CARPAY"),
        ("nelnet", "Nelnet", "student_loan", "student_loan", "NELNET", "STUDENT LN"),
        ("deptedu", "Dept of Education", "student_loan", "student_loan", "DEPT EDUCATION", "STUDENT LN"),
        ("aidvantage", "Aidvantage", "student_loan", "student_loan", "AIDVANTAGE", "PAYMENT"),
        ("brighthorizons", "Bright Horizons", "childcare", "childcare", "BRIGHT HORIZONS", "TUITION"),
        ("vanguard", "Vanguard", "investing", "investing", "VANGUARD BUY", "INVESTMENT"),
        ("robinhood", "Robinhood", "investing", "investing", "ROBINHOOD", "FUNDS"),
        ("fidelity", "Fidelity", "investing", "investing", "FID BKG SVC LLC", "MONEYLINE"),
        ("betterment", "Betterment", "investing", "investing", "BETTERMENT", "DEPOSIT"),
        ("allysav", "Ally Bank Savings", "savings", "savings", "ALLY BANK", "TRANSFER"),
        ("marcus", "Marcus Savings", "savings", "savings", "GS BANK USA", "MARCUS"),
        ("irs", "IRS tax refund", "tax_refund", "tax_refund", "IRS TREAS 310", "TAX REF")]:
    A(mid, cl, kind, f"{role} 1 100 .3", co, des)
# card merchants with ordinary role spreads
for mid, cl, kind, roles, t in [
        ("jiffylube", "Jiffy Lube", "auto_service", "auto_maint 1 85 .5", ["JIFFY LUBE #{n4}"]),
        ("firestone", "Firestone", "auto_service", "auto_maint 1 180 .9", ["FIRESTONE {n6}"]),
        ("discounttire", "Discount Tire", "auto_service", "auto_maint 1 220 .8", ["DISCOUNT TIRE {n4}"]),
        ("valvoline", "Valvoline", "auto_service", "auto_maint 1 90 .4", ["VIOC {n5}", "VALVOLINE #{n4}"]),
        ("autozone", "AutoZone", "auto_service", "auto_maint 1 38 .6", ["AUTOZONE #{n4}"]),
        ("parkmobile", "ParkMobile", "parking", "parking 1 9 .6", ["@PARKMOBILE 877-727-5758 GA"]),
        ("spplus", "SP+ Parking", "parking", "parking 1 16 .6", ["SP+ PARKING {n4}"]),
        ("uber", "Uber", "rideshare", "rideshare 1 21 .5", ["@UBER *TRIP {ref}", "@UBER   *TRIP HELP.UBER.COM CA"]),
        ("lyft", "Lyft", "rideshare", "rideshare 1 19 .5", ["@LYFT   *RIDE {DOW} {H}PM", "@LYFT *RIDE {ref}"]),
        ("mta", "MTA", "transit", "transit 1 34 .01", ["@MTA*NYCT PAYGO NEW YORK NY"]),
        ("clipper", "Clipper", "transit", "transit 1 40 .01", ["@CLIPPER SYSTEMS 877-878-8883 CA"]),
        ("ventra", "Ventra", "transit", "transit 1 20 .01", ["@VENTRA 877-669-8368 IL"]),
        ("delta", "Delta", "airline", "air 1 380 .5", ["@DELTA AIR {r13}"]),
        ("united", "United Airlines", "airline", "air 1 410 .5", ["@UNITED {r13}"]),
        ("southwest", "Southwest", "airline", "air 1 290 .5", ["@SOUTHWES {r10}"]),
        ("american", "American Airlines", "airline", "air 1 360 .5", ["@AMERICAN AIR{r10}"]),
        ("alaska", "Alaska Airlines", "airline", "air 1 330 .5", ["@ALASKA AIR {r10}"]),
        ("airbnb", "Airbnb", "lodging", "lodging 1 780 .5", ["@AIRBNB * {ref}"]),
        ("marriott", "Marriott", "lodging", "lodging 1 620 .5", ["MARRIOTT {cityup}"]),
        ("hilton", "Hilton", "lodging", "lodging 1 560 .5", ["HILTON {cityup}", "HAMPTON INN {cityup}"]),
        ("expedia", "Expedia", "lodging", "lodging 1 700 .5", ["@EXPEDIA {r12}"]),
        ("hertz", "Hertz", "rental", "rental_car 1 320 .4", ["HERTZ RENT-A-CAR"]),
        ("enterprise", "Enterprise Rent-A-Car", "rental", "rental_car 1 290 .4", ["ENTERPRISE RENT-A-CAR"]),
        ("petco", "Petco", "pet", "pets 1 38 .6", ["PETCO {n4}"]),
        ("petsmart", "PetSmart", "pet", "pets 1 42 .6", ["PETSMART # {n4}"]),
        ("banfield", "Banfield Pet Hospital", "pet", "vet 1 85 .7", ["BANFIELD PET HOSP #{n4}"]),
        ("rover", "Rover", "pet", "pets 1 60 .4", ["@ROVER.COM 888-727-1140 WA"]),
        ("labcorp", "Labcorp", "medical", "medical 1 45 .8", ["@LABCORP 800-845-6167 NC"]),
        ("quest", "Quest Diagnostics", "medical", "medical 1 52 .8", ["@QUEST DIAGNOSTICS 866-697-8378 NJ"]),
        ("warby", "Warby Parker", "medical", "vision 1 110 .4", ["@WARBY PARKER 888-492-7297 NY"]),
        ("amc", "AMC Theatres", "entertainment", "entertainment 1 32 .5", ["AMC {n4} ONLINE", "AMC {cityup} {n2}"]),
        ("regal", "Regal Cinemas", "entertainment", "entertainment 1 30 .5", ["REGAL {cityup}"]),
        ("ticketmaster", "Ticketmaster", "entertainment", "entertainment 1 140 .7", ["@TICKETMASTER 800-653-8000 CA"]),
        ("stubhub", "StubHub", "entertainment", "entertainment 1 160 .7", ["@STUBHUB INC 866-788-2482 CA"]),
        ("topgolf", "Topgolf", "entertainment", "entertainment 1 70 .5", ["TOPGOLF {cityup}"]),
        ("davebusters", "Dave & Buster's", "entertainment", "entertainment 1 55 .5", ["DAVE & BUSTERS #{n3}"]),
        ("steam", "Steam", "games", "games 1 22 .7", ["@STEAMGAMES.COM 4259522985 WA"]),
        ("psn", "PlayStation Store", "games", "games 1 30 .7", ["@PLAYSTATION NETWORK 800-345-7669 CA"]),
        ("nintendo", "Nintendo", "games", "games 1 35 .6", ["@NINTENDO *AMERICA 800-255-3700 WA"]),
        ("xbox", "Xbox", "games", "games 1 25 .6", ["@MICROSOFT*XBOX 800-642-7676 WA"]),
        ("barnes", "Barnes & Noble", "books", "books .7 24 .5, gifts .2 30 .5, kids_stuff .1 18 .5", ["BARNES & NOBLE #{n4}", "BARNES&NOBLE {n4}"]),
        ("michaels", "Michaels", "hobby", "hobbies 1 28 .7", ["MICHAELS STORES {n4}"]),
        ("hobbylobby", "Hobby Lobby", "hobby", "hobbies 1 32 .7", ["HOBBY-LOBBY #{n4}"]),
        ("joann", "JOANN", "hobby", "hobbies 1 30 .7", ["JOANN STORES #{n4}"]),
        ("rei", "REI", "hobby", "hobbies .6 70 .7, clothing .4 60 .5", ["REI #{n3}"]),
        ("myschoolbucks", "MySchoolBucks", "school", "school 1 40 .4", ["@MYSCHOOLBUCKS.COM 855-832-5226 NJ"]),
        ("sportsengine", "SportsEngine", "kids_activity", "kids_activities 1 120 .6", ["@SE*SPORTSENGINE {league}"]),
        ("i9sports", "i9 Sports", "kids_activity", "kids_activities 1 110 .3", ["@I9 SPORTS 813-712-7100 FL"]),
        ("carters", "Carter's", "kids_store", "kids_stuff 1 34 .5", ["CARTERS #{n4}"]),
        ("childrensplace", "The Children's Place", "kids_store", "kids_stuff 1 32 .5", ["THE CHILDRENS PLACE #{n4}"]),
        ("buybuybaby", "buybuy BABY", "baby", "baby 1 55 .6", ["BUYBUY BABY #{n4}"]),
        ("redcross", "American Red Cross", "charity", "charity 1 50 .5", ["@AMERICAN RED CROSS 800-733-2767 DC"]),
        ("gofundme", "GoFundMe", "charity", "charity 1 50 .6", ["@GOFUNDME {ref}"]),
        ("flowers", "1-800-Flowers", "gifts", "gifts 1 70 .3", ["@1-800-FLOWERS.COM 800-468-1141 NY"]),
        ("edible", "Edible Arrangements", "gifts", "gifts 1 65 .3", ["EDIBLE ARRANGEMENTS {n4}"]),
        ("hallmark", "Hallmark", "gifts", "gifts 1 12 .6", ["HALLMARK {n5}"]),
        ("staples", "Staples", "office", "office 1 38 .7", ["STAPLES {n5}", "@STAPLES.COM 800-378-2753 MA"]),
        ("officedepot", "Office Depot", "office", "office 1 42 .7", ["OFFICE DEPOT #{n4}"]),
        ("flexispot", "FlexiSpot", "office", "office 1 380 .3", ["@FLEXISPOT 855-420-6868 CA"]),
        ("udemy", "Udemy", "education", "education 1 17 .4", ["@UDEMY ONLINE COURSES 888-913-0000 CA"]),
        ("davidsbridal", "David's Bridal", "wedding", "wedding 1 650 .5", ["DAVIDS BRIDAL #{n4}"]),
        ("zola", "Zola", "wedding", "wedding 1 180 .6", ["@ZOLA.COM 408-657-9652 NY"]),
        ("minted", "Minted", "wedding", "wedding 1 240 .4", ["@MINTED.COM 888-828-6468 CA"])]:
    C(mid, cl, kind, roles, t)
MERCH["ezpass"].roles = {"parking": (1, 25, .01)}
MERCH["sunpass"].roles = {"parking": (1, 20, .01)}
reg(M("atm", "ATM withdrawal", "atm", parse_roles("cash 1 60 .5"), ["ATM WITHDRAWAL {n6}", "ATM CASH WITHDRAWAL {n4}"]))
reg(M("fee_monthly", "Monthly service fee", "fee", parse_roles("fees 1 12 .01"), ["MONTHLY SERVICE FEE"], channel="fee"))
reg(M("fee_foreign", "Foreign transaction fee", "fee", parse_roles("fees 1 2.4 .5"), ["FOREIGN TRANSACTION FEE"], channel="fee"))
reg(M("fee_od", "Overdraft fee", "fee", parse_roles("fees 1 35 .01"), ["OVERDRAFT ITEM FEE"], channel="fee"))

# kinds a production system would treat as "the same kind of business"
GROUP = {"warehouse": "grocery", "bar": "restaurant", "gas_conv": "gas", "online": "bigbox",
         "delivery": "fastfood", "kids_store": "kids_activity", "baby": "kids_activity",
         "rental": "lodging", "furniture": "home_improve", "books": "hobby", "games": "hobby",
         "dental": "medical", "vision": "medical", "vet": "pet"}
SITDOWN = ("restaurant", "bar")
TRIP_KINDS = ("restaurant", "fastfood", "coffee", "bar", "gas", "gas_conv", "grocery", "entertainment",
              "parking", "rideshare", "lodging", "airline", "rental", "pharmacy")
INCOME_ROLES = ("income", "interest", "tax_refund", "reimb_in")
ALIAS = {"rental_car": "lodging"}


def group(kind):
    return GROUP.get(kind, kind)


# local and opaque business names -----------------------------------------------------------
LOCAL_FIRST = ["BLUE HERON", "MAPLE ST", "RIVERSIDE", "OLD TOWN", "SUNSET", "GOLDEN", "LITTLE", "NORTHSIDE",
               "CEDAR", "HARBOR", "PINE RIDGE", "MAIN STREET", "COPPER", "IRON", "WILLOW", "LUCKY", "HAPPY",
               "EAST END", "WESTGATE", "HIGHLAND", "MILLBROOK", "LAKESIDE", "BRICKYARD", "FOXGLOVE", "JUNIPER",
               "MAGNOLIA", "SUMMIT", "ORCHARD", "TWIN OAKS", "SILVER", "RED BARN", "BLUEBIRD", "STONEY CREEK",
               "PARKSIDE", "ROSA'S", "TONY'S", "MAMA LIN'S", "NONNA'S", "BIG SKY", "HOMETOWN"]
LOCAL_KINDS = {  # kind -> list of (noun, role, median, sigma)
    "restaurant": [("THAI KITCHEN", "dining", 46, .4), ("TAQUERIA", "dining", 28, .4), ("PIZZERIA", "dining", 38, .4),
                   ("BISTRO", "dining", 72, .4), ("RAMEN BAR", "dining", 42, .35), ("BBQ", "dining", 44, .4),
                   ("DINER", "dining", 31, .35), ("SUSHI", "dining", 64, .45), ("GRILL", "dining", 52, .4),
                   ("TRATTORIA", "dining", 78, .4), ("PHO HOUSE", "dining", 34, .35), ("CANTINA", "dining", 48, .4),
                   ("CURRY HOUSE", "dining", 45, .4), ("SMOKEHOUSE", "dining", 55, .4), ("BURGER BAR", "dining", 36, .4)],
    "coffee": [("COFFEE", "coffee", 6.5, .35), ("COFFEE CO", "coffee", 7, .35), ("ROASTERS", "coffee", 7.5, .35),
               ("ESPRESSO BAR", "coffee", 6.2, .3), ("CAFE", "coffee", 9, .4), ("BAKERY & CAFE", "coffee", 11, .45)],
    "bar": [("TAPROOM", "bars", 34, .45), ("BREWING CO", "bars", 38, .45), ("PUB", "bars", 36, .45),
            ("WINE BAR", "bars", 58, .4), ("TAVERN", "bars", 40, .45)],
    "grocery": [("FARM MARKET", "groceries", 32, .5), ("ASIAN MARKET", "groceries", 48, .5),
                ("MEAT MARKET", "groceries", 42, .45), ("FOOD CO-OP", "groceries", 38, .5), ("INTL MARKET", "groceries", 36, .5)],
    "personal_care": [("NAIL SPA", "personal_care", 48, .3), ("BARBER SHOP", "haircut", 30, .2), ("HAIR SALON", "haircut", 85, .3),
                      ("LASH STUDIO", "personal_care", 90, .25), ("DAY SPA", "personal_care", 120, .3)],
    "home_improve": [("HARDWARE", "home_improve", 22, .7), ("GARDEN CENTER", "home_improve", 45, .6), ("LUMBER", "home_improve", 90, .7)],
    "pet": [("ANIMAL HOSP", "vet", 180, .6), ("VET CLINIC", "vet", 150, .6), ("PET SUPPLY", "pets", 40, .5), ("DOG GROOMING", "pets", 65, .25)],
    "medical": [("PEDIATRICS", "medical", 40, .6), ("FAMILY MEDICINE", "medical", 45, .6), ("URGENT CARE", "medical", 95, .5),
                ("FAMILY DENTAL", "dental", 130, .7), ("DENTAL CARE", "dental", 120, .7), ("CHIROPRACTIC", "medical", 45, .3),
                ("PHYSICAL THERAPY", "medical", 40, .3), ("EYE CARE", "vision", 150, .5)],
    "auto_service": [("AUTO REPAIR", "auto_maint", 320, .8), ("TIRE & LUBE", "auto_maint", 110, .6), ("CAR WASH", "auto_maint", 15, .3)],
    "gifts": [("FLORIST", "gifts", 65, .35), ("GIFT SHOP", "gifts", 30, .5)],
    "hobby": [("BOOKS", "books", 22, .5), ("BIKE SHOP", "hobbies", 80, .8), ("MUSIC CO", "hobbies", 45, .7),
              ("YARN CO", "hobbies", 30, .5), ("GAME STORE", "games", 35, .5)],
    "entertainment": [("CINEMA", "entertainment", 28, .4), ("BOWL", "entertainment", 45, .4), ("ESCAPE ROOM", "entertainment", 90, .3),
                      ("MINI GOLF", "entertainment", 30, .4), ("TRAMPOLINE PARK", "entertainment", 55, .4)],
    "fitness": [("YOGA", "fitness", 22, .3), ("CROSSFIT", "fitness", 25, .3), ("CLIMBING GYM", "fitness", 24, .3), ("PILATES", "fitness", 32, .3)],
    "kids_activity": [("KARATE", "kids_activities", 110, .3), ("DANCE ACADEMY", "kids_activities", 95, .3),
                      ("SWIM SCHOOL", "kids_activities", 105, .3), ("GYMNASTICS", "kids_activities", 115, .3),
                      ("MUSIC LESSONS", "kids_activities", 140, .2)],
    "clothing": [("DRY CLEANERS", "clothing", 26, .4), ("BOUTIQUE", "clothing", 58, .5), ("SHOE REPAIR", "clothing", 30, .4)],
}
PROC = ["SQ *", "TST* ", "SP ", "PY *", "CLV*", ""]
OPAQUE_SUFFIX = ["HOLDINGS LLC", "ENTERPRISES", "GROUP INC", "SERVICES LLC", "VENTURES", "PARTNERS", "MGMT", "INTL", "CORP", "ASSOC", "LLC"]
OPAQUE_WORDS = ["RIVERSTONE", "NORTHWIND", "BRIGHTPATH", "ATLAS 7", "KESTREL", "BLACKWOOD", "TRI-STAR", "5 POINTS", "OAKWOOD", "MERIDIAN 3"]
OPAQUE_ROLES = [("dining", 45, .5, "restaurant"), ("fast_food", 14, .4, "fastfood"), ("personal_care", 60, .5, "personal_care"),
                ("household", 40, .6, "bigbox"), ("entertainment", 50, .5, "entertainment"), ("gifts", 45, .5, "gifts"),
                ("home_improve", 120, .7, "home_improve"), ("hobbies", 45, .6, "hobby"), ("auto_maint", 150, .7, "auto_service"),
                ("medical", 80, .6, "medical"), ("clothing", 55, .5, "clothing"), ("kids_activities", 100, .4, "kids_activity"),
                ("pets", 55, .5, "pet")]
DELIVERY_RESTS = ["PANDA EXPRESS", "CHIPOTLE", "SHAKE SHACK", "WINGSTOP", "THAI BASIL", "PAPA JOHNS", "PHO 99", "CAVA",
                  "FIREHOUSE SUBS", "THE HALAL GUYS", "SWEET TOMATOES", "HOT CHICKEN", "MOD PIZZA", "POKE BOWL"]
LEAGUES = ["SOCCER CLUB", "LITTLE LEAGUE", "YOUTH HOCKEY", "SWIM TEAM", "FLAG FOOTBALL", "LACROSSE ASSN"]
CHURCHES = ["GRACE CHURCH", "ST MARKS PARISH", "NEW HOPE CHURCH", "FIRST BAPTIST", "CROSSPOINT", "ST JOSEPH PARISH"]
EMPLOYERS = ["ACME CORP", "NORTHSIDE HOSPITAL", "CITY OF {CITY}", "BLUE RIDGE LOGISTICS", "WALMART INC", "TARGET CORP",
             "{ST} STATE UNIV", "MERIDIAN HEALTH", "BRIGHTWAVE SOFTWARE", "KROGER CO", "HOME DEPOT USA", "LAKEVIEW SCHOOLS",
             "SUMMIT ENGINEERING", "COASTAL CREDIT UNION", "PIONEER MFG", "DELTA AIR LINES"]

FIRST = ["MARIA", "JAMES", "EMILY", "DAVID", "SARAH", "MICHAEL", "JESSICA", "CHRIS", "ASHLEY", "DANIEL", "LAUREN", "KEVIN",
         "MEGAN", "JOSE", "PRIYA", "WEI", "AISHA", "TYLER", "HANNAH", "OMAR", "GRACE", "LUIS", "NATALIE", "BRANDON",
         "KATIE", "ANDRE", "ROSA", "ETHAN", "CHLOE", "SAM"]
LAST = ["LOPEZ", "SMITH", "NGUYEN", "JOHNSON", "PATEL", "WILLIAMS", "GARCIA", "BROWN", "KIM", "DAVIS", "MARTINEZ",
        "WILSON", "ANDERSON", "THOMAS", "TAYLOR", "MOORE", "JACKSON", "LEE", "HARRIS", "CLARK", "LEWIS", "ROBINSON",
        "WALKER", "YOUNG", "ALLEN", "KING", "WRIGHT", "SCOTT", "HILL", "GREEN"]
KIDS = ["Emma", "Noah", "Olivia", "Liam", "Ava", "Mason", "Sophia", "Lucas", "Mia", "Eli", "Zoe", "Jack"]

STATES = {
    "OH": ["COLUMBUS", "CINCINNATI", "DUBLIN", "WESTERVILLE"], "TX": ["AUSTIN", "ROUND ROCK", "HOUSTON", "KATY", "PLANO"],
    "CO": ["DENVER", "AURORA", "LAKEWOOD", "BOULDER"], "NC": ["RALEIGH", "CARY", "DURHAM", "CHARLOTTE"],
    "OR": ["PORTLAND", "BEAVERTON", "HILLSBORO"], "FL": ["TAMPA", "ST PETERSBURG", "BRANDON", "JACKSONVILLE"],
    "AZ": ["PHOENIX", "MESA", "TEMPE", "TUCSON"], "TN": ["NASHVILLE", "FRANKLIN", "MURFREESBORO"],
    "WI": ["MADISON", "MILWAUKEE", "WAUKESHA"], "ID": ["BOISE", "MERIDIAN", "NAMPA"],
    "VA": ["RICHMOND", "ARLINGTON", "FAIRFAX", "HENRICO"], "CA": ["SACRAMENTO", "SAN DIEGO", "OAKLAND", "ROSEVILLE", "SAN JOSE"],
    "NY": ["ALBANY", "BROOKLYN", "ROCHESTER", "BUFFALO"], "NE": ["OMAHA", "LINCOLN"],
    "PA": ["PITTSBURGH", "PHILADELPHIA", "KING OF PRUSSIA"], "UT": ["SALT LAKE CTY", "PROVO", "SANDY"],
    "MO": ["KANSAS CITY", "ST LOUIS", "LEES SUMMIT"], "MN": ["MINNEAPOLIS", "ST PAUL", "EDINA"],
    "WA": ["SEATTLE", "SPOKANE", "BELLEVUE", "TACOMA"], "GA": ["ATLANTA", "MARIETTA", "DECATUR"],
    "IL": ["CHICAGO", "NAPERVILLE", "EVANSTON"], "MI": ["GRAND RAPIDS", "ANN ARBOR", "DETROIT"],
}
GROCERY_BY_STATE = {
    "OH": ["kroger", "meijer", "gianteagle", "aldi"], "TX": ["heb", "kroger", "aldi", "sprouts"], "CO": ["kroger", "safeway", "sprouts"],
    "NC": ["harristeeter", "foodlion", "publix", "aldi"], "OR": ["fredmeyer", "safeway", "winco"], "FL": ["publix", "aldi", "wholefoods"],
    "AZ": ["safeway", "sprouts", "kroger"], "TN": ["kroger", "publix", "aldi"], "WI": ["meijer", "aldi", "wholefoods"],
    "ID": ["winco", "fredmeyer", "safeway"], "VA": ["wegmans", "harristeeter", "foodlion", "lidl"],
    "CA": ["safeway", "traderjoes", "sprouts", "winco"], "NY": ["wegmans", "stopshop", "aldi", "traderjoes"],
    "NE": ["hyvee", "aldi"], "PA": ["gianteagle", "wegmans", "aldi"], "UT": ["kroger", "winco", "safeway"],
    "MO": ["hyvee", "aldi", "kroger"], "MN": ["hyvee", "aldi", "traderjoes"], "WA": ["fredmeyer", "safeway", "winco", "traderjoes"],
    "GA": ["publix", "kroger", "aldi"], "IL": ["jewel", "aldi", "meijer", "traderjoes"], "MI": ["meijer", "kroger", "aldi"],
}
NATGAS_BY_STATE = {"CA": "socalgas", "IL": "nicor", "OH": "columbiagas", "NC": "piedmont", "TN": "piedmont", "PA": "columbiagas",
                   "NY": "ngrid", "TX": "centerpoint", "MN": "centerpoint"}
ELEC_BY_STATE = {"OH": "aep", "NC": "duke", "FL": "fpl", "CA": "pge", "NY": "coned", "CO": "xcel", "MN": "xcel", "GA": "gapower",
                 "VA": "dominion", "AZ": "srp", "IL": "comed", "WA": "pse", "TN": "duke", "WI": "xcel", "MI": "consumers",
                 "MO": "evergy", "ID": "idahopower", "UT": "rmp", "NE": "oppd", "PA": "peco", "OR": "pgeor", "TX": "txu"}
GASCONV_BY_STATE = {"PA": ["wawa", "sheetz"], "VA": ["wawa", "sheetz"], "FL": ["wawa", "bucees"], "OH": ["sheetz", "speedway"],
                    "NC": ["sheetz", "quiktrip"], "TX": ["bucees", "quiktrip"], "AZ": ["quiktrip", "circlek"], "GA": ["quiktrip", "bucees"],
                    "MO": ["quiktrip", "caseys"], "NE": ["caseys", "kwiktrip"], "MN": ["kwiktrip", "caseys"], "WI": ["kwiktrip", "caseys"],
                    "TN": ["bucees", "seveneleven"], "CO": ["bucees", "seveneleven"]}
TRIPS = [  # name options, (city, state) destinations, state token a reader can match to the name, or None
    (["Hawaii trip", "Maui 2025", "Hawaii vacation"], [("KAHULUI", "HI"), ("KIHEI", "HI"), ("HONOLULU", "HI")], "HI"),
    (["Disney trip", "Disney World"], [("ORLANDO", "FL"), ("LAKE BUENA VIST", "FL"), ("KISSIMMEE", "FL")], "FL"),
    (["Yellowstone trip", "Yellowstone 2025"], [("WEST YELLOWSTON", "MT"), ("JACKSON", "WY"), ("CODY", "WY")], "WY"),
    (["NYC trip", "New York trip"], [("NEW YORK", "NY")], "NY"),
    (["Ski trip", "Colorado ski trip"], [("BRECKENRIDGE", "CO"), ("VAIL", "CO")], "CO"),
    (["Beach week", "Summer vacation", "Anniversary trip"], [("MYRTLE BEACH", "SC"), ("NAGS HEAD", "NC"), ("GULF SHORES", "AL")], None),
]


# --------------------------------------------------------------------------------------
# Category schemes
# --------------------------------------------------------------------------------------

def pick(rng, opts, weights=None):
    return rng.choices(opts, weights=weights)[0] if weights else rng.choice(opts)


def scheme_default(u, rng, family=False):
    """YNAB-like default list with the renames people actually make. Returns [(name, roles)], unguessable roles."""
    cats, ung = [], []
    inc = pick(rng, ["Ready to Assign", "Inflow: Ready to Assign", "Income"], [.4, .3, .3])
    if rng.random() < .5:
        cats.append((inc, list(INCOME_ROLES)))
    house = pick(rng, ["Rent", "Rent/Mortgage", "Housing"]) if u.renter else pick(rng, ["Mortgage", "Rent/Mortgage", "Housing"])
    cats.append((house, ["rent", "mortgage", "hoa"]))
    r = rng.random()
    if r < .45:
        cats += [("Electric", ["electric"]), (pick(rng, ["Water", "Water & sewer", "Water/Trash"]), ["water", "trash"])]
        if u.has_natgas:
            cats.append((pick(rng, ["Natural gas", "Gas bill", "Heat"]), ["natgas"]))
        else:
            cats[-1][1].append("natgas")
    else:
        cats.append((pick(rng, ["Utilities", "Utilities"]), ["electric", "water", "natgas", "trash"]))
    r = rng.random()
    if r < .5:
        cats += [("Internet", ["internet"]), (pick(rng, ["Phone", "Cell phone", "Phones"]), ["phone"])]
    else:
        cats.append((pick(rng, ["Internet & phone", "Phone & internet"]), ["internet", "phone"]))
    cats.append((pick(rng, ["Insurance", "Insurance", "Insurances"]), ["auto_ins", "home_ins", "renters_ins"]))
    if u.car_payment:
        cats.append((pick(rng, ["Car payment", "Car loan"]), ["car_payment"]))
    if u.student_loan:
        cats.append((pick(rng, ["Student loans", "Student loan"]), ["student_loan"]))
    cats.append((pick(rng, ["Groceries", "Groceries", "Grocery"]), ["groceries"]))
    tr = pick(rng, ["Transportation", "Gas & transportation", "Auto & gas", "Getting around", "Gas"])
    if rng.random() < .5:
        cats.append((tr, ["fuel", "parking", "rideshare", "transit"]))
        cats.append((pick(rng, ["Auto maintenance", "Car maintenance", "Car repairs"]), ["auto_maint"]))
    else:
        cats.append((tr, ["fuel", "parking", "rideshare", "transit", "auto_maint"]))
    cats.append((pick(rng, ["Medical", "Medical expenses", "Health", "Healthcare"]), ["medical", "pharmacy", "dental", "vision"]))
    if u.has_kids:
        if family:
            cats.append(("Childcare", ["childcare"]))
            for k in u.kid_names:
                cats.append((f"{k} activities" if rng.random() < .7 else k, []))
            cats.append((pick(rng, ["School", "School lunches", "School stuff"]), ["school"]))
            cats.append((pick(rng, ["Kids clothes", "Kids stuff", "Kid stuff"]), ["kids_stuff", "baby"]))
            ung.append("kids_activities")
        else:
            cats.append((pick(rng, ["Kids", "Kids", "Children"]), ["childcare", "kids_activities", "school", "kids_stuff", "baby"]))
    if u.has_pets:
        cats.append((pick(rng, ["Pets", "Pet care", "Dog"]), ["pets", "vet"]))
    cats.append((pick(rng, ["Emergency fund", "Savings", "Emergency Fund"]), ["savings"]))
    if u.investing:
        cats.append((pick(rng, ["Investing", "Retirement", "Brokerage"]), ["investing"]))
    dine = ["dining", "fast_food", "delivery", "bars"]
    dname = pick(rng, ["Dining out", "Dining Out", "Eating out", "Restaurants", "Takeout & dining"])
    cats.append((dname, dine))
    if rng.random() < .25:
        cats.append((pick(rng, ["Coffee", "Coffee shops"]), ["coffee"]))
    else:
        dine.append("coffee")
    if rng.random() < .15:
        cats.append((pick(rng, ["Work lunches", "Lunch at work"]), []))
        u.worklunch = cats[-1][0]
    fun = pick(rng, ["Entertainment", "Fun", "Fun money", "Entertainment"])
    cats.append((fun, ["entertainment", "games", "books", "hobbies"]))
    if rng.random() < .5:
        cats.append((fun, ["alcohol"]))
    else:
        dine.append("alcohol") if rng.random() < .5 else cats.append(("__groceries__", ["alcohol"]))
    ung.append("alcohol")
    subs = pick(rng, ["Subscriptions", "Streaming & apps", "Subscriptions & apps"])
    if rng.random() < .4:
        cats.append((subs, ["streaming", "software", "cloud", "news"]))
        cats.append((pick(rng, ["Fitness", "Gym", "Gym membership"]), ["fitness"]))
    else:
        cats.append((subs, ["streaming", "software", "cloud", "news", "fitness"]))
    if not u.renter and rng.random() < .5:
        cats.append((pick(rng, ["Household", "Household goods", "Home supplies"]), ["household", "office"]))
        cats.append((pick(rng, ["Home maintenance", "Home improvement", "House projects"]), ["home_improve", "furniture"]))
    else:
        cats.append((pick(rng, ["Household", "Home", "Household goods"]), ["household", "home_improve", "furniture", "office"]))
    cats.append((pick(rng, ["Clothing", "Clothes", "Clothing"]), ["clothing"]))
    cats.append((pick(rng, ["Personal care", "Hair & beauty", "Personal care"]), ["personal_care", "haircut"]))
    cats.append((pick(rng, ["Gifts", "Gifts", "Birthdays & gifts"]), ["gifts"]))
    cats.append((pick(rng, ["Giving", "Charity", "Donations"]), ["charity", "church"]))
    cats.append((pick(rng, ["Vacation", "Travel", "Vacation"]), ["air", "lodging"]))
    cats.append((pick(rng, ["Stuff I forgot to budget for", "Miscellaneous", "Misc", "Everything else"]), ["electronics", "fees", "education"]))
    ung += ["electronics", "education", "fees"]
    if u.uses_cash:
        cats.append((pick(rng, ["Spending money", "Cash", "Pocket money"]), ["cash"]))
    else:
        cats[-1][1].append("cash")
        ung.append("cash")
    if not any(n == inc for n, _ in cats):
        cats.append((inc, list(INCOME_ROLES)))
    return cats, ung


def scheme_merged(u, rng):
    cats = [(pick(rng, ["Housing", "Home", "House"]), ["rent", "mortgage", "hoa", "home_ins", "renters_ins", "home_improve", "furniture"]),
            (pick(rng, ["Bills", "Monthly bills", "Fixed bills"]), ["electric", "water", "natgas", "trash", "internet", "phone", "streaming",
                                                                    "software", "cloud", "news", "auto_ins", "car_payment", "student_loan", "fitness", "childcare"]),
            (pick(rng, ["Food", "Food & drink", "Food"]), ["groceries", "dining", "fast_food", "coffee", "delivery", "alcohol", "bars"]),
            (pick(rng, ["Transportation", "Car", "Transport"]), ["fuel", "auto_maint", "parking", "rideshare", "transit"]),
            (pick(rng, ["Health", "Medical"]), ["medical", "pharmacy", "dental", "vision"]),
            (pick(rng, ["Fun", "Fun money", "Entertainment"]), ["entertainment", "games", "books", "hobbies", "air", "lodging", "cash", "bars_x"]),
            (pick(rng, ["Shopping", "Stuff", "Shopping"]), ["household", "clothing", "personal_care", "haircut", "office", "electronics"])]
    ung = ["cash", "electronics"]
    if u.has_kids:
        cats.append((pick(rng, ["Kids", "Family"]), ["kids_activities", "school", "kids_stuff", "baby"]))
    if u.has_pets:
        cats.append((pick(rng, ["Pets", "Pet"]), ["pets", "vet"]))
    cats.append((pick(rng, ["Giving", "Gifts & giving"]), ["gifts", "charity", "church"]))
    cats.append((pick(rng, ["Savings", "Saving"]), ["savings", "investing"]))
    cats.append((pick(rng, ["Everything else", "Misc", "Fees & misc"]), ["fees", "education"]))
    ung += ["education"]
    cats.append((pick(rng, ["Income", "Inflow: Ready to Assign"]), list(INCOME_ROLES)))
    return cats, ung


def scheme_detailed(u, rng):
    cats = [(pick(rng, ["Income", "Paychecks"]), list(INCOME_ROLES)),
            ("Rent" if u.renter else "Mortgage", ["rent", "mortgage"])]
    if u.hoa:
        cats.append(("HOA", ["hoa"]))
    cats += [("Electric", ["electric"]), ("Water & sewer", ["water"]), ("Internet", ["internet"]), ("Cell phone", ["phone"])]
    if u.has_natgas:
        cats.append(("Natural gas", ["natgas"]))
    if u.trash:
        cats.append(("Trash", ["trash"]))
    else:
        cats[3][1].append("trash")
    cats.append(("Car insurance", ["auto_ins"]))
    cats.append(("Renters insurance" if u.renter else "Home insurance", ["renters_ins", "home_ins"]))
    if u.car_payment:
        cats.append(("Car payment", ["car_payment"]))
    if u.student_loan:
        cats.append(("Student loans", ["student_loan"]))
    cats += [("Groceries", ["groceries"]), ("Coffee", ["coffee"]), ("Dining out", ["dining", "bars"]),
             ("Takeout", ["fast_food", "delivery"]), (pick(rng, ["Alcohol", "Wine & beer", "Booze"]), ["alcohol"])]
    if rng.random() < .6:
        cats.append((pick(rng, ["Work lunches", "Lunch at work"]), []))
        u.worklunch = cats[-1][0]
    cats += [("Gas", ["fuel"]), (pick(rng, ["Car maintenance", "Auto maintenance"]), ["auto_maint"]),
             ("Parking & tolls", ["parking"]), ("Rideshare & transit", ["rideshare", "transit"]),
             ("Doctor", ["medical", "vision"]), ("Pharmacy", ["pharmacy"]), ("Dental", ["dental"]),
             ("Streaming", ["streaming"]), (pick(rng, ["Software & apps", "Apps & cloud"]), ["software", "cloud", "news"]),
             ("Gym", ["fitness"]), ("Household supplies", ["household"]), ("Home improvement", ["home_improve"]),
             ("Furniture", ["furniture", "office"]), ("Clothing", ["clothing"]), ("Haircuts", ["haircut"]),
             ("Personal care", ["personal_care"]), ("Entertainment", ["entertainment"]), ("Video games", ["games"]),
             ("Books", ["books"]), ("Hobbies", ["hobbies"]), ("Electronics", ["electronics"]), ("Travel", ["air", "lodging"])]
    if u.has_pets:
        cats += [("Pet food & supplies", ["pets"]), ("Vet", ["vet"])]
    if u.has_kids:
        cats += [("Childcare", ["childcare"]), ("Kids activities", ["kids_activities"]), ("School", ["school"]),
                 ("Kids stuff", ["kids_stuff", "baby"])]
    cats += [("Gifts", ["gifts"]), ("Charity", ["charity"])]
    if u.church:
        cats.append(("Tithe", ["church"]))
    else:
        cats[-1][1].append("church")
    cats += [("Bank fees", ["fees"]), ("Education", ["education"]), ("Cash", ["cash"]), ("Emergency fund", ["savings"])]
    if u.investing:
        cats.append(("Investing", ["investing"]))
    if u.costco:
        cats += [("Costco - food", []), ("Costco - household", [])]
        u.mrole["costco"] = {r: ("Costco - food" if r == "groceries" else "Costco - household") for r in MERCH["costco"].roles}
    if u.partner and rng.random() < .6:
        cats.append(("Date night", []))
        u.datenight = "Date night"
    return cats, []


def scheme_personal(u, rng):
    cats = [(pick(rng, ["Roof over head", "Shelter", "Casa"]), ["rent", "mortgage", "hoa", "home_ins", "renters_ins"]),
            (pick(rng, ["Grown-up bills", "Boring bills", "Keep the lights on"]), ["electric", "water", "natgas", "trash", "internet", "phone",
                                                                                   "auto_ins", "car_payment", "student_loan"]),
            (pick(rng, ["Food we cook", "Groceries", "Grub", "Kitchen"]), ["groceries"]),
            (pick(rng, ["Food they cook", "Eating out", "Yum", "Restaurants"]), ["dining", "fast_food", "delivery"]),
            (pick(rng, ["Caffeine", "Coffee habit"]), ["coffee"]),
            (pick(rng, ["Booze", "Drinks"]), ["alcohol", "bars"]),
            (pick(rng, ["Wheels", "Car", "Vroom"]), ["fuel", "parking", "rideshare", "transit"]),
            (pick(rng, ["Sinking: car", "Car fund"]), ["auto_maint"]),
            (pick(rng, ["Doctor stuff", "Health", "Ouch"]), ["medical", "pharmacy", "dental", "vision"]),
            (pick(rng, ["Streaming junk", "Subscriptions", "Apps & streaming"]), ["streaming", "software", "cloud", "news"]),
            (pick(rng, ["Sweat", "Gym"]), ["fitness"]),
            (pick(rng, ["Stuff", "Things", "House stuff"]), ["household", "home_improve", "furniture", "office", "electronics"]),
            (pick(rng, ["Treat yo self", "Me stuff", "Self care"]), ["clothing", "personal_care", "haircut"]),
            (pick(rng, ["Fun money", "Blow money", "Play", "Mad money"]), ["entertainment", "games", "books", "hobbies", "cash"]),
            (pick(rng, ["Adventures", "Travel"]), ["air", "lodging"])]
    ung = ["clothing", "electronics", "cash", "office"]
    if u.has_pets:
        cats.append((pick(rng, ["Pupper", "Fur babies", "Cat stuff", "Dog"]), ["pets", "vet"]))
    if u.has_kids:
        cats.append((pick(rng, ["Daycare", "Childcare"]), ["childcare", "school"]))
        for i in range(len(u.kid_names)):
            cats.append((f"Kid {i + 1}", []))
        cats.append((pick(rng, ["Kid stuff", "Kids misc"]), ["kids_stuff", "baby"]))
        ung.append("kids_activities")
    cats += [(pick(rng, ["Presents", "Gifts"]), ["gifts"]), (pick(rng, ["Giving", "Do good"]), ["charity", "church"]),
             (pick(rng, ["Oops", "Dumb fees"]), ["fees"]), (pick(rng, ["Learning", "Brain food"]), ["education"]),
             (pick(rng, ["Future us", "Savings", "Rainy day"]), ["savings", "investing"]),
             (pick(rng, ["Money in", "Income", "Ready to Assign"]), list(INCOME_ROLES))]
    if u.partner and rng.random() < .5:
        cats.append((pick(rng, ["Date night", "Us time"]), []))
        u.datenight = cats[-1][0]
    return cats, ung


def scheme_minimal(u, rng):
    cats = [(pick(rng, ["Bills", "Fixed"]), ["rent", "mortgage", "hoa", "electric", "water", "natgas", "trash", "internet", "phone",
                                              "auto_ins", "home_ins", "renters_ins", "car_payment", "student_loan", "streaming", "software",
                                              "cloud", "news", "fitness", "childcare"]),
            (pick(rng, ["Food", "Food"]), ["groceries", "dining", "fast_food", "coffee", "delivery", "alcohol", "bars"]),
            (pick(rng, ["Transportation", "Car & gas"]), ["fuel", "auto_maint", "parking", "rideshare", "transit"])]
    if rng.random() < .5:
        cats.append(("Health", ["medical", "pharmacy", "dental", "vision"]))
    cats.append((pick(rng, ["Spending", "Everything else", "Discretionary"]), []))
    cats += [(pick(rng, ["Savings", "Save"]), ["savings", "investing"]), ("Income", list(INCOME_ROLES))]
    return cats, ["electronics", "fees", "education", "cash"]


SCHEMES = [("default", .45), ("family", .12), ("merged", .13), ("detailed", .12), ("personal", .10), ("minimal", .08)]
ALT_ROLES = {"coffee": ["entertainment", "personal_care", "groceries", "dining"], "groceries": ["household", "dining"],
             "fast_food": ["groceries", "entertainment", "dining", "kids_stuff"], "dining": ["entertainment", "gifts"],
             "fuel": ["auto_maint", "air"], "streaming": ["entertainment", "software"], "home_improve": ["household", "furniture"],
             "clothing": ["personal_care", "gifts", "kids_stuff"], "haircut": ["clothing", "medical"],
             "alcohol": ["entertainment", "groceries", "dining"], "delivery": ["groceries", "entertainment"],
             "hobbies": ["gifts", "entertainment", "household"], "books": ["education", "gifts"], "entertainment": ["dining", "gifts"],
             "fitness": ["medical", "entertainment"], "rideshare": ["entertainment", "dining"], "personal_care": ["clothing", "medical"],
             "software": ["streaming", "education"], "games": ["entertainment", "streaming"], "bars": ["entertainment", "dining"]}


# --------------------------------------------------------------------------------------
# Users
# --------------------------------------------------------------------------------------

class User:
    def __init__(self, uid):
        self.uid = uid
        self.cats, self.created = [], {}
        self.rmap, self.unguess = {}, []
        self.idio, self.changed, self.mrole, self.mnew = {}, {}, {}, {}
        self.mrole_unguess = []
        self.newcat = self.newcat_date = None
        self.newcat_roles, self.newcat_kind = [], None
        self.datenight = self.worklunch = self.reimb = None
        self.dn_thr = 60.0
        self.trip = None
        self.extra = {}
        self.uses, self.recs = [], []
        self.stores = {}
        self.named = []
        self.flags = []

    def add_cat(self, name, when, before=None):
        if name in self.created:
            return False
        if before and before in self.cats:
            self.cats.insert(self.cats.index(before), name)
        else:
            self.cats.append(name)
        self.created[name] = when
        return True

    def header(self, d):
        return [c for c in self.cats if self.created[c] <= d]

    def roles_at(self, mid, d):
        m = MERCH[mid]
        out = {}
        for r, (w, med, s) in m.roles.items():
            if mid in self.recroles and r in self.recroles[mid] and len(m.roles) == 1:
                continue
            if r == "gifts" and d.month in (11, 12):
                w *= 2.5
            out[r] = (w, med, s)
        for (r, w, med, s, start) in self.extra.get(mid, []):
            if d >= start:
                out[r] = (w, med, s)
        return out

    def map(self, mid, role, d, amt=None, wd=None, rng=None, rules=True, dn_evidence=True):
        """The category this user means for a purchase. rng adds rule noise (generation only)."""
        if mid in self.idio:
            return self.idio[mid]
        if mid in self.changed:
            s, a, b = self.changed[mid]
            return b if d >= s else a
        if mid in self.mnew and d >= self.mnew[mid][0]:
            return self.mnew[mid][1]
        role = ALIAS.get(role, role)
        if self.newcat and role in self.newcat_roles and d >= self.newcat_date:
            return self.newcat
        if mid in self.mrole and role in self.mrole[mid]:
            return self.mrole[mid][role]
        kind = MERCH[mid].kind
        if rules and amt is not None:
            if self.datenight and role in ("dining", "bars") and kind in SITDOWN:
                rule = wd in ("Fri", "Sat") and abs(amt) >= self.dn_thr
                if rng is not None and rng.random() < (.12 if rule else .04):
                    rule = not rule
                if rule and dn_evidence:
                    return self.datenight
            if self.worklunch and role == "fast_food":
                rule = wd not in ("Sat", "Sun")
                if rng is not None and rng.random() < .1:
                    rule = not rule
                if rule:
                    return self.worklunch
            if self.reimb and role == "rideshare":
                rule = wd not in ("Sat", "Sun")
                if rng is not None and rng.random() < .1:
                    rule = not rule
                if rule:
                    return self.reimb
        if self.reimb and role in ("air", "lodging"):
            return self.reimb
        return self.rmap[role]

    def cats_of(self, mid, d):
        """Distinct categories a merchant's purchases fall in on date d, rules aside."""
        out = []
        for r in list(self.roles_at(mid, d)) + list(self.recroles.get(mid, {})):
            c = self.map(mid, r, d, rules=False)
            if c not in out:
                out.append(c)
        return out

    def rule_kind(self, mid):
        m = MERCH[mid]
        if mid in self.idio or mid in self.changed:
            return None
        if self.datenight and m.kind in SITDOWN and m.primary in ("dining", "bars"):
            return "date_night"
        if self.worklunch and "fast_food" in m.roles and len(m.roles) == 1:
            return "weekday_split"
        if self.reimb and m.primary == "rideshare":
            return "weekday_split"
        return None


def lognorm(rng, med, sig):
    return max(0.5, med * math.exp(sig * rng.gauss(0, 1)))


def rdig(rng, k, lead_nonzero=False):
    s = "".join(rng.choice(DIG) for _ in range(k))
    if lead_nonzero and s[0] == "0":
        s = str(rng.randint(1, 9)) + s[1:]
    return s


def rref(rng, k=None):
    return "".join(rng.choice(ALNUM) for _ in range(k or rng.choice([9, 10])))


def local_merchant(u, rng, kind, idx, opaque=False, city=None):
    mid = f"u{u.uid}_{'opq' if opaque else 'loc'}{idx}"
    if opaque:
        role, med, sig, k = rng.choice(OPAQUE_ROLES)
        if rng.random() < .5:
            body = "".join(rng.choice("ABCDEFGHJKLMNPRSTW") for _ in range(rng.choice([2, 3, 3]))) + " " + rng.choice(OPAQUE_SUFFIX)
        else:
            body = rng.choice(OPAQUE_WORDS) + " " + rng.choice(OPAQUE_SUFFIX)
        pre = pick(rng, ["", "SQ *", "PAYPAL *", "PY *", "SP "], [.4, .25, .15, .1, .1])
        m = M(mid, body.title().replace("Llc", "LLC").replace("Inc", "Inc."), k, {role: (1.0, med, sig)},
              [("@" if pre == "PAYPAL *" else "") + pre + body], opaque=True, local=True)
        return reg(m)
    noun, role, med, sig = rng.choice(LOCAL_KINDS[kind])
    first = rng.choice(LOCAL_FIRST)
    pre = pick(rng, PROC, [.3, .25 if kind in ("restaurant", "bar", "coffee") else .05, .1, .08, .1, .3])
    name = f"{first} {noun}"
    m = M(mid, name.title().replace("'S", "'s").replace("&", "and"), kind, {role: (1.0, med, sig)}, [pre + name],
          local=True)
    return reg(m)


def named_local(u, rng, idx, noun, role, med, sig, kind):
    first = rng.choice(LOCAL_FIRST)
    pre = pick(rng, ["", "SQ *", "CLV*"], [.6, .25, .15])
    name = f"{first} {noun}"
    return reg(M(f"u{u.uid}_loc{idx}", name.title().replace("'S", "'s"), kind, {role: (1.0, med, sig)}, [pre + name], local=True))


def build_user(uid, short):
    rng = random.Random(SEED * 1000 + uid)
    u = User(uid)
    u.short = short
    u.st = rng.choice(list(STATES))
    u.cities = STATES[u.st]
    u.city = rng.choice(u.cities)
    u.name_first, u.name_last = rng.choice(FIRST), rng.choice(LAST)
    u.uname = f"{u.name_first} {u.name_last}"
    u.bank = pick(rng, ["chase", "boa", "wells", "cu", "pad", "clean"], [.25, .15, .12, .15, .15, .18])
    u.cu_variant = rng.random() < .5
    u.last4, u.last4s = rdig(rng, 4), rdig(rng, 4)
    u.coid = {}
    u.chk = rng.randint(1001, 2400)
    u.start = Y0 if rng.random() < .72 else rand_date(rng, D(2, 1), D(6, 30))
    if short:
        u.start = rand_date(rng, D(11, 5), D(12, 12))
    u.renter = rng.random() < .52
    u.hoa = (not u.renter) and rng.random() < .25
    u.has_car = rng.random() < .88
    u.car_payment = u.has_car and rng.random() < .5
    u.has_kids = rng.random() < .33
    u.kid_names = rng.sample(KIDS, rng.choice([1, 2, 2, 3])) if u.has_kids else []
    u.has_pets = rng.random() < .33
    u.partner = rng.random() < .58
    u.student_loan = rng.random() < .2
    u.investing = rng.random() < .3
    u.savings = rng.random() < .6
    u.church = rng.random() < .15
    u.costco = rng.random() < .4
    u.has_natgas = rng.random() < .45
    u.trash = (not u.renter) and rng.random() < .6
    u.uses_cash = rng.random() < .45
    u.sloppy = .06 if rng.random() < .2 else .02
    u.recroles = {}

    # ---- scheme
    kinds = [k for k, _ in SCHEMES]
    kind = rng.choices(kinds, weights=[w for _, w in SCHEMES])[0]
    if short:
        kind = pick(rng, ["default", "merged", "minimal", "detailed", "personal"], [.6, .15, .1, .08, .07])
    if kind == "family" and not u.has_kids:
        kind = "default"
    u.scheme = kind
    fn = {"default": scheme_default, "family": lambda uu, rr: scheme_default(uu, rr, family=True), "merged": scheme_merged,
          "detailed": scheme_detailed, "personal": scheme_personal, "minimal": scheme_minimal}[kind]
    cats, ung = fn(u, rng)
    grocery_cat = None
    for name, roles in cats:
        if name == "__groceries__":
            continue
        u.add_cat(name, u.start)
        for r in roles:
            u.rmap.setdefault(r, name)
        if "groceries" in roles:
            grocery_cat = name
    for name, roles in cats:
        if name == "__groceries__":
            for r in roles:
                u.rmap.setdefault(r, grocery_cat)
    u.unguess = list(ung)
    misc = next((c for c in u.cats if c in ("Stuff I forgot to budget for", "Miscellaneous", "Misc", "Everything else",
                                            "Spending", "Discretionary", "Stuff", "Things", "House stuff", "Shopping")), None)
    if misc is None:
        misc = u.rmap.get("household") or u.cats[-2]
    ALL_ROLES = ["groceries", "dining", "fast_food", "coffee", "delivery", "bars", "alcohol", "fuel", "auto_maint", "auto_ins",
                 "car_payment", "parking", "rideshare", "transit", "rent", "mortgage", "hoa", "electric", "water", "natgas", "trash",
                 "internet", "phone", "home_ins", "renters_ins", "household", "home_improve", "furniture", "clothing",
                 "personal_care", "haircut", "medical", "pharmacy", "dental", "vision", "fitness", "streaming", "software", "cloud",
                 "news", "entertainment", "games", "books", "hobbies", "electronics", "air", "lodging", "pets", "vet", "childcare",
                 "kids_activities", "school", "kids_stuff", "baby", "gifts", "charity", "church", "education", "student_loan",
                 "income", "interest", "tax_refund", "reimb_in", "savings", "investing", "cash", "fees", "office", "wedding"]
    for r in ALL_ROLES:
        if r not in u.rmap:
            u.rmap[r] = misc
            if r not in u.unguess:
                u.unguess.append(r)
    kid_cats = [c for c in u.cats if re.fullmatch(r"Kid \d", c) or (c.endswith(" activities") and c.split()[0] in KIDS) or c in u.kid_names]
    if u.datenight is not None:
        u.dn_thr = rng.uniform(45, 75)
    income_cat = u.rmap["income"]

    # ---- merchants this user uses (mid, visits per 30 days, start, end)
    use = u.uses

    def U(mid, rate, start=None, end=None):
        use.append((mid, rate, start or u.start, end or Y1))

    def REC(mid, role, period, day, amount, var=0.0, start=None, end=None, bump=None):
        u.recs.append(dict(mid=mid, role=role, period=period, day=day, amount=amount, var=var,
                           start=start or u.start, end=end or Y1, bump=bump))
        u.recroles.setdefault(mid, {})[role] = (amount, var)

    groc = GROCERY_BY_STATE[u.st]
    prim = rng.choice(groc)
    U(prim, rng.uniform(4, 6.5))
    if rng.random() < .55:
        U(rng.choice([g for g in groc + ["aldi", "traderjoes", "wholefoods"] if g != prim]), rng.uniform(.8, 2))
    nloc = 0

    def LOC(kind, rate, start=None, end=None, opaque=False):
        nonlocal nloc
        nloc += 1
        m = local_merchant(u, rng, kind, nloc, opaque=opaque)
        U(m.mid, rate, start, end)
        return m

    if rng.random() < .3:
        LOC("grocery", rng.uniform(.4, 1))
    if u.costco:
        U("costco", rng.uniform(1.2, 2.5))
    elif rng.random() < .15:
        U(rng.choice(["samsclub", "bjs"]), rng.uniform(1, 2))
    if rng.random() < .75:
        U("target", rng.uniform(1.2, 3))
    if rng.random() < .55:
        U("walmart", rng.uniform(1, 3))
    if rng.random() < .88:
        U("amazon", rng.uniform(2, 7))
    for mid in ["ebay", "etsy", "temu", "shein", "wayfair"]:
        if rng.random() < .15:
            U(mid, rng.uniform(.2, .6))
    for mid in rng.sample(["olivegarden", "chilis", "applebees", "texasroadhouse", "cheesecake", "redrobin", "pfchangs", "outback", "firstwatch", "bww"],
                          rng.randint(1, 4)):
        U(mid, rng.uniform(.3, 1))
    for _ in range(rng.randint(2, 4)):
        LOC("restaurant", rng.uniform(.4, 1.3))
    for mid in rng.sample(["mcdonalds", "chickfila", "chipotle", "tacobell", "wendys", "panera", "subway", "fiveguys", "sweetgreen",
                           "jerseymikes", "panda", "dominos", "pizzahut", "culvers", "sonic"], rng.randint(3, 6)):
        U(mid, rng.uniform(.4, 2.5))
    if rng.random() < .4:
        for mid in rng.sample(["doordash", "ubereats", "grubhub"], rng.randint(1, 2)):
            U(mid, rng.uniform(1, 3))
    if rng.random() < .7:
        U(rng.choice(["starbucks", "starbucks", "dunkin", "dutchbros", "peets"]), rng.uniform(2, 10))
    if rng.random() < .4:
        LOC("coffee", rng.uniform(1, 4))
    if rng.random() < .4:
        for _ in range(rng.randint(1, 2)):
            LOC("bar", rng.uniform(.5, 2))
    if rng.random() < .45:
        U(rng.choice(["totalwine", "abcwine", "binnys"]), rng.uniform(.6, 2))
    if u.has_car:
        for mid in rng.sample(["shell", "chevron", "exxon", "bp", "sunoco", "marathon", "speedway", "circlek", "valero"], rng.randint(1, 2)):
            U(mid, rng.uniform(1.2, 3))
        if u.costco and rng.random() < .6:
            U("costcogas", rng.uniform(1, 2.5))
        if rng.random() < .5:
            U(rng.choice(GASCONV_BY_STATE.get(u.st, ["seveneleven", "circlek"]) if u.st in GASCONV_BY_STATE else ["seveneleven"]),
              rng.uniform(1.5, 4))
        for mid in rng.sample(["jiffylube", "firestone", "discounttire", "valvoline", "autozone"], rng.randint(1, 2)):
            U(mid, rng.uniform(.12, .35))
        if rng.random() < .3:
            LOC("auto_service", rng.uniform(.3, 1.5))
    U(rng.choice(["cvs", "walgreens", "walgreens", "cvs", "riteaid"]), rng.uniform(1, 2.5))
    if not u.renter or rng.random() < .35:
        for mid in rng.sample(["homedepot", "lowes", "ace", "menards"], rng.randint(1, 2)):
            U(mid, rng.uniform(.3, 2.2) if not u.renter else rng.uniform(.2, .5))
    if rng.random() < .25:
        LOC("home_improve", rng.uniform(.2, .8))
    for mid in rng.sample(["oldnavy", "kohls", "tjmaxx", "marshalls", "nordrack", "nike", "gap", "ross", "hm", "uniqlo"], rng.randint(1, 3)):
        U(mid, rng.uniform(.25, 1))
    if rng.random() < .2:
        LOC("clothing", rng.uniform(.3, 1))
    if rng.random() < .5:
        U("bestbuy", rng.uniform(.15, .4))
    if rng.random() < .15:
        U("applestore", rng.uniform(.05, .2))
    r = rng.random()
    if r < .45:
        U(rng.choice(["greatclips", "supercuts"]), rng.uniform(.7, 1.2))
    else:
        LOC("personal_care", rng.uniform(.6, 1.2))
    if rng.random() < .4:
        U(rng.choice(["ulta", "sephora"]), rng.uniform(.3, 1))
    if rng.random() < .3:
        LOC("personal_care", rng.uniform(.4, 1.2))
    if rng.random() < .45:
        mid = rng.choice(["planetfitness", "orangetheory", "ymca", "peloton", None])
        if mid:
            amt = {"planetfitness": pick(rng, [10.0, 24.99]), "orangetheory": pick(rng, [169.0, 179.0]), "ymca": pick(rng, [64.0, 89.0, 112.0]),
                   "peloton": 44.0}[mid]
            REC(mid, "fitness", "monthly", rng.randint(1, 28), amt)
        else:
            m = LOC("fitness", rng.uniform(1, 3))
    # subscriptions
    subs = {"netflix": [7.99, 17.99, 17.99, 24.99], "spotify": [11.99, 11.99, 16.99, 19.99], "hulu": [9.99, 18.99], "disney": [9.99, 15.99],
            "max": [9.99, 16.99], "youtube": [13.99, 22.99], "peacock": [7.99, 13.99], "paramount": [7.99, 12.99], "audible": [14.95],
            "prime": [14.99], "kindle": [11.99]}
    for mid in rng.sample(list(subs), rng.randint(1, 5)):
        amt = rng.choice(subs[mid])
        bump = (D(10, rng.randint(1, 20)), round(amt + 2, 2)) if mid == "netflix" and rng.random() < .5 else None
        REC(mid, "streaming", "monthly", rng.randint(1, 28), amt, bump=bump)
    soft = {"adobe": ("software", [22.99, 59.99]), "ms365": ("software", [9.99]), "dropbox": ("cloud", [11.99]), "openai": ("software", [20.0]),
            "googleone": ("cloud", [2.99, 9.99]), "nytimes": ("news", [4.0, 17.0, 25.0]), "patreon": ("entertainment", [5.0, 10.0, 7.0])}
    for mid in rng.sample(list(soft), rng.choice([0, 0, 1, 1, 2, 3])):
        role, amts = soft[mid]
        REC(mid, role, "monthly", rng.randint(1, 28), rng.choice(amts))
    if rng.random() < .6:
        REC("apple", "cloud", "monthly", rng.randint(1, 28), pick(rng, [0.99, 2.99, 2.99, 9.99]))
        U("apple", rng.uniform(.2, .8))
    # housing and bills
    if u.renter:
        r = rng.random()
        rent = round(lognorm(rng, 1650, .3) / 5) * 5
        if r < .15:
            u.landlord = reg(M(f"u{uid}_check", "Landlord (check)", "rent", {"rent": (1, rent, .01)}, [], channel="check"))
            REC(u.landlord.mid, "rent", "monthly", 1, float(rent), bump=(D(rng.randint(5, 9), 1), float(round(rent * 1.04 / 5) * 5)))
        elif r < .25 and rng.random() < .8:
            u.zelle_landlord = True
            u.rent_amt = rent
        else:
            REC(rng.choice(["bilt", "greystar", "zillowrent", "rentcafe"]), "rent", "monthly", 1, float(rent),
                bump=(D(rng.randint(5, 9), 1), float(round(rent * 1.04 / 5) * 5)) if rng.random() < .4 else None)
        if rng.random() < .4:
            REC("lemonade", "renters_ins", "monthly", rng.randint(1, 28), pick(rng, [9.0, 12.0, 14.0, 17.0]))
    else:
        REC(rng.choice(["rocket", "mrcooper", "wfhm", "pennymac", "loancare"]), "mortgage", "monthly", rng.randint(1, 5),
            round(lognorm(rng, 2100, .35), 2))
        if u.hoa:
            hoa = reg(M(f"u{uid}_hoa", "HOA dues", "hoa", {"hoa": (1, 200, .01)}, [], ach=(pick(rng, ["CINC SYSTEMS", "APPFOLIO", "FRONTSTEPS"]) , "HOA DUES"),
                        channel="ach", opaque=True))
            REC(hoa.mid, "hoa", "monthly", 1, float(rng.choice([95, 150, 185, 240, 310])))
    elec = ELEC_BY_STATE.get(u.st, rng.choice(["duke", "dominion", "xcel"]))
    REC(elec, "electric", "monthly", rng.randint(3, 25), round(lognorm(rng, 125, .3), 2), var=.25)
    if not u.renter or rng.random() < .5:
        if rng.random() < .6:
            water = reg(M(f"u{uid}_water", f"City of {u.city.title()} Utilities", "water", {"water": (1, 60, .3)}, [],
                          ach=(f"CITY OF {u.city}", pick(rng, ["UTILITY", "UTIL BILL", "WATER"])), channel="ach"))
            REC(water.mid, "water", "monthly", rng.randint(5, 25), round(lognorm(rng, 62, .3), 2), var=.15)
        else:
            REC("amwater", "water", "monthly", rng.randint(5, 25), round(lognorm(rng, 58, .3), 2), var=.15)
    if u.has_natgas:
        REC(NATGAS_BY_STATE.get(u.st) or rng.choice(["piedmont", "columbiagas", "centerpoint"]), "natgas", "monthly", rng.randint(5, 25),
            round(lognorm(rng, 55, .3), 2), var=.2)
    if u.trash:
        REC(rng.choice(["wm", "republic"]), "trash", "monthly", rng.randint(5, 25), round(lognorm(rng, 32, .2), 2))
    if rng.random() < .95:
        REC(rng.choice(["xfinity", "spectrum", "attnet", "fios", "gfiber"]), "internet", "monthly", rng.randint(1, 28),
            pick(rng, [49.99, 55.0, 65.0, 70.0, 79.99, 89.99]))
    REC(rng.choice(["verizonw", "tmobile", "attw", "mint"]), "phone", "monthly", rng.randint(1, 28), round(lognorm(rng, 110, .45), 2), var=.03)
    if u.has_car:
        ins = rng.choice(["geico", "progressive", "statefarm", "allstate"])
        REC(ins, "auto_ins", "monthly", rng.randint(1, 28), round(lognorm(rng, 140, .35), 2))
    if u.car_payment:
        REC(rng.choice(["toyotafin", "ally", "hondafin", "caponeauto"]), "car_payment", "monthly", rng.randint(1, 28), round(lognorm(rng, 460, .3), 2))
    if u.student_loan:
        REC(rng.choice(["nelnet", "deptedu", "aidvantage"]), "student_loan", "monthly", rng.randint(1, 28), round(lognorm(rng, 280, .4), 2))
    if u.has_car and rng.random() < .35:
        U(rng.choice(["parkmobile", "spplus"]), rng.uniform(1, 4))
    if u.has_car and u.st in ("NY", "PA", "VA", "FL") and rng.random() < .5:
        U("ezpass" if u.st != "FL" else "sunpass", .8)
    if rng.random() < .55:
        for mid in rng.sample(["uber", "lyft"], rng.randint(1, 2)):
            U(mid, rng.uniform(.4, 3))
    if (not u.has_car or rng.random() < .1) and u.st in ("NY", "CA", "IL"):
        U({"NY": "mta", "CA": "clipper", "IL": "ventra"}[u.st], rng.uniform(3, 8))
    if u.has_pets:
        U(rng.choice(["petco", "petsmart"]), rng.uniform(.6, 1.6))
        if rng.random() < .4:
            REC("chewy", "pets", "monthly", rng.randint(1, 28), round(lognorm(rng, 55, .3), 2))
        if rng.random() < .5:
            LOC("pet", rng.uniform(.1, .3))
        else:
            U("banfield", rng.uniform(.1, .25))
        if rng.random() < .2:
            U("rover", rng.uniform(.2, .6))
    for _ in range(rng.randint(1, 2)):
        LOC("medical", rng.uniform(.1, .4))
    if rng.random() < .4:
        U(rng.choice(["labcorp", "quest"]), .08)
    if rng.random() < .1:
        U("warby", .06)
    for mid in rng.sample(["amc", "regal", "ticketmaster", "stubhub", "topgolf", "davebusters"], rng.randint(1, 3)):
        U(mid, rng.uniform(.15, .8))
    if rng.random() < .3:
        LOC("entertainment", rng.uniform(.2, .6))
    if rng.random() < .3:
        U(rng.choice(["steam", "psn", "nintendo", "xbox"]), rng.uniform(.4, 2))
    if rng.random() < .5:
        for mid in rng.sample(["barnes", "michaels", "hobbylobby", "joann", "rei"], rng.randint(1, 2)):
            U(mid, rng.uniform(.2, .9))
    if rng.random() < .25:
        LOC("hobby", rng.uniform(.2, .7))
    if u.has_kids:
        if rng.random() < .55:
            mid = rng.choice(["kiddie", "kindercare", "brighthorizons"])
            if mid == "brighthorizons":
                REC(mid, "childcare", "monthly", 1, round(lognorm(rng, 1500, .25), 2))
            else:
                REC(mid, "childcare", "weekly", 0, float(rng.choice([235, 265, 295, 320, 360])))
        if rng.random() < .7:
            U("myschoolbucks", rng.uniform(.8, 2))
        acts = []
        for _ in range(rng.randint(1, 3)):
            if rng.random() < .5:
                m = LOC("kids_activity", 0)
                use.pop()
                REC(m.mid, "kids_activities", "monthly", rng.randint(1, 10), float(round(m.roles[m.primary][1])))
                acts.append(m.mid)
            else:
                mid = rng.choice(["sportsengine", "i9sports", "kumon"])
                if mid == "kumon":
                    if any(a == "kumon" for a in acts):
                        continue
                    REC(mid, "kids_activities", "monthly", rng.randint(1, 10), float(rng.choice([150, 165, 180])))
                else:
                    U(mid, rng.uniform(.15, .4))
                acts.append(mid)
        for mid in rng.sample(["carters", "childrensplace"], 1):
            U(mid, rng.uniform(.3, .7))
        if kid_cats:
            for a in acts:
                u.mrole.setdefault(a, {})["kids_activities"] = rng.choice(kid_cats)
                u.mrole_unguess.append(a)
    if rng.random() < .5:
        for mid in rng.sample(["redcross", "gofundme", "stjude"], rng.randint(1, 2)):
            if mid == "stjude":
                REC(mid, "charity", "monthly", rng.randint(1, 28), float(rng.choice([19, 25, 30])))
            else:
                U(mid, rng.uniform(.08, .25))
    if u.church:
        church = reg(M(f"u{uid}_church", rng.choice(CHURCHES).title(), "church", {"church": (1, 100, .01)},
                       [f"@PUSHPAY*{rng.choice(CHURCHES)}"] if rng.random() < .5 else [f"@TITHE.LY*{rng.choice(CHURCHES)}"]))
        REC(church.mid, "church", "monthly", rng.randint(1, 15), float(rng.choice([50, 100, 150, 200, 300, 400])))
    if rng.random() < .4:
        U(rng.choice(["flowers", "edible", "hallmark"]), rng.uniform(.1, .3))
    if rng.random() < .15:
        LOC("gifts", rng.uniform(.1, .3))
    if rng.random() < .2:
        U("staples", rng.uniform(.15, .4))
    if rng.random() < .1:
        REC("coursera", "education", "monthly", rng.randint(1, 28), 59.0)
    elif rng.random() < .1:
        U("udemy", .3)
    if rng.random() < .15:
        U("ikea", .15)
    # money in and out
    emp = rng.choice(EMPLOYERS).replace("{CITY}", u.city).replace("{ST}", u.st)
    u.employer = reg(M(f"u{uid}_pay", emp.title() + " payroll", "payroll", {"income": (1, 2000, .01)}, [],
                       ach=(emp, pick(rng, ["PAYROLL", "DIR DEP", "DIRECT DEP", "PAYROLL"])), channel="ach"))
    pay = -round(lognorm(rng, 2100, .4), 2)
    semi = rng.random() < .3
    REC(u.employer.mid, "income", "semimonthly" if semi else "biweekly", rng.randint(0, 13), pay, var=.02)
    if u.partner and rng.random() < .45:
        emp2 = rng.choice([e for e in EMPLOYERS if e != emp]).replace("{CITY}", u.city).replace("{ST}", u.st)
        p2 = reg(M(f"u{uid}_pay2", emp2.title() + " payroll", "payroll", {"income": (1, 2000, .01)}, [],
                   ach=(pick(rng, [emp2, "GUSTO", "ADP"]) , pick(rng, ["PAYROLL", "PAY " + rdig(rng, 6)])), channel="ach"))
        REC(p2.mid, "income", "biweekly", rng.randint(0, 13), -round(lognorm(rng, 1700, .4), 2), var=.02)
    if rng.random() < .6:
        intr = reg(M(f"u{uid}_int", "Interest", "interest", {"interest": (1, 5, .01)}, [], ach=("INTEREST", "PAYMENT"), channel="fee"))
        intr.tpls = [pick(rng, ["INTEREST PAYMENT", "INTEREST EARNED", "DIVIDEND/INTEREST"])]
        REC(intr.mid, "interest", "monthly", 28, -round(lognorm(rng, 6, 1.2), 2), var=.3)
    if rng.random() < .55:
        REC("irs", "tax_refund", "annual", rand_date(rng, D(2, 20), D(4, 25)).timetuple().tm_yday, -round(lognorm(rng, 1800, .6), 2))
    if u.savings:
        if rng.random() < .6:
            sav = reg(M(f"u{uid}_sav", "Transfer to savings", "savings", {"savings": (1, 300, .01)}, [], channel="xfer"))
            REC(sav.mid, "savings", "monthly", rng.choice([1, 2, 15, 16]), float(rng.choice([100, 150, 200, 250, 300, 500, 750, 1000])))
        else:
            REC(rng.choice(["allysav", "marcus"]), "savings", "monthly", rng.choice([1, 2, 15, 16]), float(rng.choice([100, 200, 250, 400, 500])))
    if u.investing:
        REC(rng.choice(["vanguard", "robinhood", "fidelity", "betterment"]), "investing", "monthly", rng.randint(1, 28),
            float(rng.choice([100, 200, 250, 500, 1000])))
    if u.uses_cash:
        U("atm", rng.uniform(.5, 2))
    if rng.random() < .12:
        REC("fee_monthly", "fees", "monthly", 28, 12.0 if rng.random() < .5 else 15.0, end=rand_date(rng, D(3, 1), D(12, 31)))
    if rng.random() < .06:
        U("fee_od", .15)
    # P2P: Venmo is opaque in most bank feeds; Zelle names the person
    if rng.random() < .6:
        roles = "dining .35 28 .5, entertainment .2 35 .6, gifts .12 30 .5, household .13 40 .6, groceries .1 25 .5, hobbies .1 30 .5"
        if u.has_kids:
            roles += ", childcare .2 80 .3"
        v = reg(M(f"u{uid}_venmo", "Venmo", "p2p", parse_roles(roles), [], channel="p2p_venmo"))
        U(v.mid, rng.uniform(1, 5))
        if u.renter and not u.partner and rng.random() < .5 and not getattr(u, "zelle_landlord", False):
            REC(v.mid, "rent", "monthly", 1, float(round(rng.uniform(550, 1100) / 25) * 25))
    persons = []
    if getattr(u, "zelle_landlord", False):
        persons.append(("landlord", "rent"))
    if u.has_kids and rng.random() < .5:
        persons.append(("sitter", "childcare"))
    if rng.random() < .22:
        persons.append(("cleaner", "household"))
    if not u.renter and rng.random() < .2:
        persons.append(("lawn", "home_improve"))
    if rng.random() < .25:
        persons.append(("family", "gifts"))
    if rng.random() < .3:
        persons.append(("friend", None))
    for i, (who, role) in enumerate(persons):
        f, l = rng.choice(FIRST), rng.choice(LAST)
        if role is None:
            roles = parse_roles("dining .35 45 .5, entertainment .25 60 .6, gifts .15 50 .5, household .25 70 .5")
        else:
            med = {"rent": 1400, "childcare": 85, "household": 140, "home_improve": 50, "gifts": 100}[role]
            roles = {role: (1.0, med, .25 if role in ("childcare", "gifts") else .02)}
        z = reg(M(f"u{uid}_zelle{i}", f"Zelle - {f.title()} {l.title()}", "p2p", roles, [], channel="p2p_zelle"))
        z.person = (f, l)
        if who == "landlord":
            REC(z.mid, "rent", "monthly", 1, float(u.rent_amt))
        elif who == "cleaner":
            REC(z.mid, "household", "biweekly", rng.randint(0, 13), float(rng.choice([120, 140, 160, 180])))
        elif who == "lawn":
            REC(z.mid, "home_improve", "biweekly", rng.randint(0, 13), float(rng.choice([40, 45, 50, 60])), start=D(4, 1), end=D(10, 31))
        else:
            U(z.mid, {"sitter": 2, "family": .3, "friend": 1.2}[who])
    # opaque merchants seen through the year, and new local businesses late in the year
    for _ in range(rng.randint(0, 2)):
        LOC(None, rng.uniform(.3, 1), opaque=True)
    lo = max(u.start, D(7, 15)) if not short else u.start
    hi = max(lo, D(12, 10))
    for _ in range(rng.randint(2, 5)):
        k = pick(rng, ["restaurant", "coffee", "bar", "grocery", "personal_care", "home_improve", "pet", "medical", "auto_service",
                       "gifts", "hobby", "entertainment", "clothing"], [6, 2, 1.5, 1, 1.5, 1, .6, 1, .8, .6, .8, 1, .6])
        LOC(k, rng.uniform(.3, 1.5), start=rand_date(rng, lo, hi))
    for _ in range(rng.choice([0, 1, 1, 2])):
        LOC(None, rng.uniform(.3, 1.2), start=rand_date(rng, lo, hi), opaque=True)
    # some chains only start being used later in the year
    for i, (mid, rate, s, e) in enumerate(use):
        m = MERCH[mid]
        if not short and not m.local and m.kind not in ("grocery",) and rng.random() < .12:
            use[i] = (mid, rate, rand_date(rng, D(4, 1), D(11, 30)), e)
    # a chain the user starts using late (a new chain they never visited before)
    if not short:
        pool = [m for m in ["olivegarden", "chilis", "applebees", "texasroadhouse", "firstwatch", "mcdonalds", "chickfila", "chipotle",
                            "panera", "sweetgreen", "fiveguys", "ulta", "petco", "bestbuy", "homedepot", "lowes", "oldnavy", "tjmaxx",
                            "topgolf", "amc", "michaels", "starbucks", "dunkin"] if all(x[0] != m for x in use)]
        for mid in rng.sample(pool, rng.randint(1, 3)):
            if mid == "petco" and not u.has_pets:
                continue
            U(mid, rng.uniform(.3, 1.2), start=rand_date(rng, D(7, 20), D(12, 5)))

    # ---- overlays (full users only)
    if not short:
        overlays(u, rng, income_cat, misc)
    return u, rng


def overlays(u, rng, income_cat, misc):
    used = [x[0] for x in u.uses] + [r["mid"] for r in u.recs]
    # a trip with its own category
    if rng.random() < .35:
        names, dests, token = rng.choice([t for t in TRIPS if all(d[1] != u.st for d in t[1])])
        name = rng.choice(names)
        t0 = rand_date(rng, max(u.start + dt.timedelta(days=100), D(6, 1)), D(11, 20))
        t1 = t0 + dt.timedelta(days=rng.randint(4, 8))
        city, st = rng.choice(dests)
        created = max(u.start, t0 - dt.timedelta(days=100))
        if u.add_cat(name, created, before=u.rmap["income"] if u.cats[-1] == u.rmap["income"] else None):
            u.trip = dict(cat=name, t0=t0, t1=t1, city=city, st=st, token=token, created=created)
    # reimbursable work expenses
    if rng.random() < .12:
        name = pick(rng, ["Work reimbursable", "Reimbursable", "Work expenses (reimbursed)", "Work travel"])
        if u.add_cat(name, u.start):
            u.reimb = name
            u.rmap["reimb_in"] = name
            if "reimb_in" in u.unguess:
                u.unguess.remove("reimb_in")
            emp = u.employer.ach[0]
            rb = reg(M(f"u{u.uid}_reimb", "Expense reimbursement", "payroll", {"reimb_in": (1, 200, .01)}, [],
                       ach=(pick(rng, [emp, "EXPENSIFY INC", "CONCUR"]), pick(rng, ["EXP REIMB", "REIMBURSE", "EXPENSE"])), channel="ach"))
            u.recs.append(dict(mid=rb.mid, role="reimb_in", period="monthly", day=rng.randint(5, 25), amount=-round(lognorm(rng, 350, .5), 2),
                               var=.5, start=u.start, end=Y1, bump=None))
            u.recroles.setdefault(rb.mid, {})["reimb_in"] = (-350, .5)
            if all(x[0] not in ("uber", "lyft") for x in u.uses):
                u.uses.append(("uber", rng.uniform(1, 3), u.start, Y1))
            for mid in rng.sample(["delta", "united", "southwest", "american"], 1) + rng.sample(["marriott", "hilton"], 1):
                u.uses.append((mid, rng.uniform(.3, .7), u.start, Y1))
    # a wedding being planned
    if rng.random() < .07 and u.add_cat("Wedding", max(u.start, D(2, 1))):
        u.rmap["wedding"] = "Wedding"
        if "wedding" in u.unguess:
            u.unguess.remove("wedding")
        for mid in ["davidsbridal", "zola", "minted"]:
            u.uses.append((mid, .2, max(u.start, D(3, 1)), D(10, 15)))
        for kind in ("gifts", None):
            m = local_merchant(u, rng, kind, 90 + (kind is None), opaque=kind is None)
            u.idio[m.mid] = "Wedding"
            u.uses.append((m.mid, .3, rand_date(rng, D(3, 1), D(8, 1)), D(10, 20)))
    # a category created mid-year that takes some merchants over
    opts = ["baby", "home_office", "christmas", "amazon" if "amazon" in used else None, "car" if u.has_car else None,
            None if u.has_pets else "pet"]
    opts = [o for o in opts if o]
    if rng.random() < .33 and opts:
        kind = rng.choice(opts)
        before = u.rmap["income"] if u.cats[-1] == u.rmap["income"] else None
        if kind == "baby":
            name, c = pick(rng, ["Baby", "Baby stuff", "Little one"]), rand_date(rng, D(5, 15), D(8, 31))
            if u.add_cat(name, c, before):
                u.newcat, u.newcat_date, u.newcat_kind = name, c, kind
                u.newcat_roles = ["baby"] if u.has_kids else ["baby", "kids_stuff"]
                for mid in ["target", "amazon", "walmart", "costco"]:
                    u.extra.setdefault(mid, []).append(("baby", .4, 32, .6, c))
                u.uses.append(("buybuybaby", rng.uniform(.5, 1), c, Y1))
                u.uses.append(("carters", rng.uniform(.3, .6), c, Y1))
                m = named_local(u, rng, 95, "PEDIATRICS", "medical", 60, .5, "medical")
                u.uses.append((m.mid, .5, c + dt.timedelta(days=10), Y1))
        elif kind == "home_office":
            name, c = pick(rng, ["Home office", "Office setup", "WFH setup"]), rand_date(rng, D(4, 1), D(8, 31))
            if u.add_cat(name, c, before):
                u.newcat, u.newcat_date, u.newcat_kind, u.newcat_roles = name, c, kind, ["office"]
                if all(x[0] != "staples" for x in u.uses):
                    u.uses.append(("staples", .15, u.start, c))
                u.uses.append(("staples", .6, c, Y1))
                u.uses.append(("officedepot", .4, c, Y1))
                u.uses.append(("flexispot", .1, c, Y1))
                u.extra.setdefault("amazon", []).append(("office", .25, 45, .7, c))
                u.extra.setdefault("bestbuy", []).append(("office", .4, 120, .6, c))
        elif kind == "christmas":
            name, c = pick(rng, ["Christmas", "Holiday gifts", "Christmas gifts"]), rand_date(rng, D(10, 20), D(11, 3))
            if u.add_cat(name, c, before):
                u.newcat, u.newcat_date, u.newcat_kind, u.newcat_roles = name, c, kind, ["gifts"]
                for mid in ["etsy", "kohls", "barnes"]:
                    if all(x[0] != mid for x in u.uses) and rng.random() < .5:
                        u.uses.append((mid, .8, c, Y1))
        elif kind == "amazon":
            name, c = pick(rng, ["Amazon", "Amazon orders", "Amazon stuff"]), rand_date(rng, D(4, 1), D(8, 31))
            if u.add_cat(name, c, before):
                u.newcat, u.newcat_date, u.newcat_kind = name, c, kind
                u.mnew["amazon"] = (c, name)
        elif kind == "car":
            name, c = pick(rng, ["Car repairs", "Sinking: car repairs", "Car repair fund"]), rand_date(rng, D(5, 1), D(8, 31))
            if u.add_cat(name, c, before):
                u.newcat, u.newcat_date, u.newcat_kind, u.newcat_roles = name, c, kind, ["auto_maint"]
                m = local_merchant(u, rng, "auto_service", 96)
                u.uses.append((m.mid, .3, c - dt.timedelta(days=3), Y1))
        elif kind == "pet":
            name, c = pick(rng, ["Dog", "New puppy", "Cat", "Pet"]), rand_date(rng, D(4, 1), D(8, 31))
            if u.add_cat(name, c, before):
                u.newcat, u.newcat_date, u.newcat_kind, u.newcat_roles = name, c, kind, ["pets", "vet"]
                u.uses.append((rng.choice(["petco", "petsmart"]), rng.uniform(.8, 1.5), c, Y1))
                m = named_local(u, rng, 97, "ANIMAL HOSP", "vet", 160, .5, "pet")
                u.uses.append((m.mid, .3, c + dt.timedelta(days=5), Y1))
                u.recs.append(dict(mid="chewy", role="pets", period="monthly", day=rng.randint(1, 28), amount=round(lognorm(rng, 50, .3), 2),
                                   var=0.0, start=c + dt.timedelta(days=20), end=Y1, bump=None))
                u.recroles.setdefault("chewy", {})["pets"] = (50, 0)
    # merchant-named categories kept from the start ("Amazon", "Costco", "Target run")
    if rng.random() < .1 and u.newcat_kind != "amazon":
        for mid, names in [("amazon", ["Amazon", "Amazon orders"]), ("costco", ["Costco"]), ("target", ["Target run", "Target"])]:
            if mid in [x[0] for x in u.uses] and mid not in u.mrole:
                nm = rng.choice(names)
                if u.add_cat(nm, u.start):
                    u.idio[mid] = nm
                    u.named.append(mid)
                break
    # single-purpose merchants filed idiosyncratically, and a change of mind
    counts = Counter(x[0] for x in u.uses)
    singles = [x for x in u.uses if len(MERCH[x[0]].roles) == 1 and x[1] >= .6 and counts[x[0]] == 1 and x[0] not in u.idio
               and x[2] <= D(5, 1) and MERCH[x[0]].kind not in ("p2p",) and x[0] not in u.mrole]
    rng.shuffle(singles)
    special = {"coffee": ["Work expenses", "Treats"], "fastfood": ["Work expenses", "Lunch money", "Kids treats"],
               "home_improve": ["Rental property", "House flipping"], "grocery": ["Mom", "Grandma's groceries"],
               "hobby": ["Side hustle", "Etsy shop"], "rideshare": ["Work expenses"], "restaurant": ["Work expenses", "Client meals"],
               "gas": ["Work mileage", "Side hustle"], "bar": ["Social"], "personal_care": ["Wedding prep", "Self care"]}
    n_idio = pick(rng, [0, 1, 1, 2], None)
    for mid, rate, s, e in singles[:n_idio]:
        m = MERCH[mid]
        base = u.map(mid, m.primary, D(6, 1), rules=False)
        if rng.random() < .5 and m.kind in special:
            nm = rng.choice(special[m.kind])
            if not u.add_cat(nm, u.start):
                continue
        else:
            nm = alt_category(u, rng, m, base, income_cat)
        if nm and nm != base:
            u.idio[mid] = nm
    rest = [x for x in singles[n_idio:] if x[0] not in u.idio and x[0] not in ("atm",)]
    if rng.random() < .35 and rest:
        mid = rest[0][0]
        m = MERCH[mid]
        s = rand_date(rng, D(5, 1), D(9, 15))
        a = u.map(mid, m.primary, D(1, 2), rules=False)
        b = alt_category(u, rng, m, a, income_cat, when=s)
        if b and b != a:
            u.changed[mid] = (s, a, b)
    if u.datenight is None and u.partner and u.scheme in ("default", "family") and rng.random() < .3:
        nm = pick(rng, ["Date night", "Date nights", "Us"])
        if u.add_cat(nm, u.start):
            u.datenight = nm
            u.dn_thr = rng.uniform(45, 75)


NOT_ALT_ROLES = ("rent", "mortgage", "hoa", "electric", "water", "natgas", "trash", "internet", "phone", "auto_ins", "home_ins",
                 "renters_ins", "car_payment", "student_loan", "savings", "investing", "childcare", "charity", "church", "fees",
                 "education", "cash", "air", "lodging")


def alt_category(u, rng, m, base, income_cat, when=None):
    when = when or u.start
    alts = []
    for r in ALT_ROLES.get(m.primary, []):
        c = u.rmap.get(r)
        if c and c != base and c != income_cat and c not in alts and u.created[c] <= when:
            alts.append(c)
    if alts and rng.random() < .75:
        return alts[0] if rng.random() < .6 else rng.choice(alts)
    pool = [c for c in u.cats if c != base and c != income_cat and u.created[c] <= when
            and not any(u.rmap.get(r) == c for r in NOT_ALT_ROLES)]
    return rng.choice(pool) if pool else None


# --------------------------------------------------------------------------------------
# Timeline
# --------------------------------------------------------------------------------------

WDW = {"grocery": [.8, .8, .9, .9, 1, 1.4, 1.4], "restaurant": [.6, .6, .7, .9, 1.5, 1.6, 1.1], "bar": [.3, .4, .6, 1.1, 1.8, 1.8, 1.0],
       "coffee": [1.2, 1.2, 1.2, 1.2, 1.1, .6, .5], "entertainment": [.6, .6, .7, .8, 1.3, 1.6, 1.4], "p2p": [.8, .8, .9, 1, 1.3, 1.3, .9]}


def days(a, b):
    d = a
    while d <= b:
        yield d
        d += dt.timedelta(days=1)


def month_day(y, mth, day):
    last = (dt.date(y + (mth == 12), mth % 12 + 1, 1) - dt.timedelta(days=1)).day
    return dt.date(y, mth, min(day, last))


def rec_dates(r):
    out = []
    if r["period"] == "monthly":
        for mth in range(1, 13):
            d = month_day(2025, mth, r["day"])
            if r["mid"] not in ("u_check",) and MERCH[r["mid"]].channel in ("ach", "xfer") and d.weekday() >= 5:
                d += dt.timedelta(days=7 - d.weekday())
            out.append(d)
    elif r["period"] == "biweekly":
        d = D(1, 3) + dt.timedelta(days=r["day"] % 14)
        while d.weekday() != 4:
            d += dt.timedelta(days=1)
        while d <= Y1:
            out.append(d)
            d += dt.timedelta(days=14)
    elif r["period"] == "semimonthly":
        for mth in range(1, 13):
            for dd in (15, 31):
                d = month_day(2025, mth, dd)
                while d.weekday() >= 5:
                    d -= dt.timedelta(days=1)
                out.append(d)
    elif r["period"] == "weekly":
        d = D(1, 6)
        while d <= Y1:
            out.append(d)
            d += dt.timedelta(days=7)
    elif r["period"] == "annual":
        out.append(Y0 + dt.timedelta(days=r["day"] - 1))
    return [d for d in out if r["start"] <= d <= min(r["end"], Y1)]


def generate(u, rng):
    rows = []

    def row(mid, d, role, amt, trip=False, loc=None):
        rows.append(dict(mid=mid, date=d, role=role, amount=round(amt, 2), trip=trip, loc=loc, key=rng.random(), rec=False))
        return rows[-1]

    trip = u.trip
    for mid, rate, s, e in u.uses:
        if rate <= 0:
            continue
        m = MERCH[mid]
        w = WDW.get(m.kind) or WDW.get(group(m.kind)) or [1] * 7
        mean = sum(w) / 7
        for d in days(max(s, u.start), e):
            if trip and trip["t0"] <= d <= trip["t1"] and m.kind not in ("p2p", "online", "software", "streaming"):
                continue
            if rng.random() < rate / 30 * w[d.weekday()] / mean:
                roles = u.roles_at(mid, d)
                rs = list(roles)
                r = rng.choices(rs, weights=[roles[x][0] for x in rs])[0]
                _, med, sig = roles[r]
                amt = lognorm(rng, med, sig)
                if m.kind == "p2p" or mid == "atm":
                    amt = round(amt / (20 if mid == "atm" else 5)) * (20 if mid == "atm" else 5) or 20
                row(mid, d, r, amt)
    for r in u.recs:
        m = MERCH[r["mid"]]
        for d in rec_dates(r):
            if d < u.start:
                continue
            amt = r["amount"]
            if r["bump"] and d >= r["bump"][0]:
                amt = r["bump"][1]
            if r["var"]:
                season = 1.0
                if r["role"] in ("electric",):
                    season = 1 + .35 * math.cos((d.month - 7.5) / 12 * 2 * math.pi) ** 2
                if r["role"] == "natgas":
                    season = 1 + .9 * max(0, math.cos((d.month - 1) / 12 * 2 * math.pi))
                amt = amt * season * math.exp(r["var"] * rng.gauss(0, 1))
            rw = row(r["mid"], d, r["role"], amt)
            rw["rec"] = True
    # the trip itself
    if trip:
        t0, t1 = trip["t0"], trip["t1"]
        loc = (trip["city"], trip["st"])
        air = rng.choice(["delta", "united", "southwest", "american", "alaska"])
        for _ in range(rng.choice([1, 2])):
            d = t0 - dt.timedelta(days=rng.randint(35, 95))
            if d >= u.start:
                row(air, d, "air", lognorm(rng, 420, .4), trip=True)
        stay = rng.choice(["airbnb", "marriott", "hilton", "expedia"])
        if stay in ("airbnb", "expedia"):
            d = t0 - dt.timedelta(days=rng.randint(20, 80))
            if d >= u.start:
                row(stay, d, "lodging", lognorm(rng, 900, .4), trip=True)
        else:
            row(stay, t1, "lodging", lognorm(rng, 750, .4), trip=True, loc=loc)
        if rng.random() < .7:
            row(rng.choice(["hertz", "enterprise"]), t1, "rental_car", lognorm(rng, 320, .3), trip=True, loc=loc)
        dest = [local_merchant(u, rng, k, 60 + i) for i, k in enumerate(["restaurant", "restaurant", "restaurant", "restaurant", "coffee",
                                                                         "entertainment", "grocery", "bar"])]
        chains = [x[0] for x in u.uses if MERCH[x[0]].kind in ("fastfood", "gas", "coffee", "pharmacy") and not MERCH[x[0]].local]
        for d in days(t0, t1):
            for _ in range(rng.choice([1, 2, 2, 3])):
                m = rng.choice(dest[:4])
                row(m.mid, d, "dining", lognorm(rng, 70, .45), trip=True, loc=loc)
            for m, p in [(dest[4], .6), (dest[5], .35), (dest[6], .2), (dest[7], .3)]:
                if rng.random() < p:
                    row(m.mid, d, m.primary, lognorm(rng, *m.roles[m.primary][1:]), trip=True, loc=loc)
            if chains and rng.random() < .5:
                mid = rng.choice(chains)
                roles = u.roles_at(mid, d)
                r = max(roles, key=lambda x: roles[x][0])
                row(mid, d, r, lognorm(rng, roles[r][1], roles[r][2]), trip=True, loc=loc)
    rows = [r for r in rows if u.start <= r["date"] <= Y1]
    # intended categories
    for r in rows:
        wd = WD[r["date"].weekday()]
        if r["trip"]:
            r["intended"] = trip["cat"] if trip["created"] <= r["date"] else u.map(r["mid"], r["role"], r["date"], r["amount"], wd, rng)
        else:
            r["intended"] = u.map(r["mid"], r["role"], r["date"], r["amount"], wd, rng)
    # refunds inherit the category of the purchase they reverse
    refunds = []
    for r in rows:
        m = MERCH[r["mid"]]
        if m.kind in ("bigbox", "online", "clothing", "warehouse", "electronics", "home_improve", "furniture", "kids_store") and r["amount"] > 8 \
                and not m.local \
                and rng.random() < .035:
            d = r["date"] + dt.timedelta(days=rng.randint(3, 25))
            if d <= Y1:
                amt = r["amount"] if rng.random() < .65 else r["amount"] * rng.uniform(.2, .8)
                refunds.append(dict(mid=r["mid"], date=d, role=r["role"], amount=-round(amt, 2), trip=False, loc=r["loc"], key=rng.random(),
                                    rec=False, intended=r["intended"], refund_of=r["amount"]))
    rows += refunds
    rows.sort(key=lambda r: (r["date"], r["key"]))
    # strings
    for r in rows:
        r["text"] = render(u, MERCH[r["mid"]], r, rng)
        r["wd"] = WD[r["date"].weekday()]
    # the user's own filing mistakes
    bills = [u.rmap[x] for x in NOT_ALT_ROLES + ("income",)]
    for r in rows:
        r["filed"] = r["intended"]
        p = u.sloppy / (4 if r["rec"] else 1)
        if rng.random() < p and r["intended"] != u.rmap["income"]:
            # a slip: the category next to the right one in the list, or one for a related purpose
            hdr = [c for c in u.header(r["date"]) if c != r["intended"] and c not in bills]
            if hdr:
                i = u.cats.index(r["intended"])
                near = [c for c in hdr if abs(u.cats.index(c) - i) == 1]
                rel = [u.rmap[x] for x in ALT_ROLES.get(ALIAS.get(r["role"], r["role"]), []) if u.rmap.get(x) in hdr]
                opts = near if rng.random() < .55 else rel
                r["filed"] = rng.choice(opts or near or rel or hdr)
    return rows


# --------------------------------------------------------------------------------------
# Rendering statement strings
# --------------------------------------------------------------------------------------

def collapse(s):
    return re.sub(r"\s+", " ", s).strip()


def render(u, m, r, rng):
    d = r["date"]
    auth = d - dt.timedelta(days=rng.choice([0, 0, 1, 1, 2]))
    mmdd = auth.strftime("%m/%d")
    style = u.bank
    neg = r["amount"] < 0
    if m.channel == "card" or m.channel == "fee":
        tpl = rng.choice(m.tpls)
        online = tpl.startswith("@")
        tpl = tpl.lstrip("@")
        city, st = r["loc"] if r["loc"] else ((u.city if rng.random() < .8 else rng.choice(u.cities)), u.st)

        def store(mt):
            k = int(mt.group(1))
            key = (m.mid, k)
            if key not in u.stores:
                u.stores[key] = [rdig(rng, k, lead_nonzero=k > 2) for _ in range(rng.choice([1, 1, 2]))]
            opts = u.stores[key]
            return opts[0] if len(opts) == 1 or rng.random() < .8 else opts[1]

        s = re.sub(r"\{n(\d+)\}", store, tpl)
        s = re.sub(r"\{r(\d+)\}", lambda mt: rdig(rng, int(mt.group(1))), s)
        s = s.replace("{ref}", rref(rng)).replace("{cityup}", city).replace("{st}", st).replace("{abbr}", city[:3])
        s = s.replace("{rest}", rng.choice(DELIVERY_RESTS)).replace("{DOW}", WD[d.weekday()].upper()).replace("{H}", str(rng.randint(1, 11)))
        if "{league}" in s:
            key = (m.mid, "league")
            if key not in u.stores:
                u.stores[key] = [f"{u.city[:10]} {rng.choice(LEAGUES)}"]
            s = s.replace("{league}", u.stores[key][0])
        if m.channel == "fee":
            return s if style != "clean" else s.title()
        if not online:
            s = s[:22].rstrip()
        if m.mid == "atm":
            pass
        if style == "clean":
            return m.clean if not m.opaque else m.clean.upper() if rng.random() < .5 else m.clean
        loc = "" if online else f"{city} {st}"
        if style == "chase":
            out = f"{s} {loc}"
        elif style == "boa":
            out = f"{s} {mmdd} {'RETURN' if neg else 'PURCHASE'} {loc}"
        elif style == "wells":
            out = f"{'PURCHASE RETURN' if neg else 'PURCHASE'} AUTHORIZED ON {mmdd} {s} {loc} S{rdig(rng, 11)} CARD {u.last4}"
        elif style == "cu":
            if u.cu_variant:
                out = f"POS {'CREDIT' if neg else 'DEBIT'} {mmdd} {s} {loc}"
            else:
                out = f"DBT CRD {rdig(rng, 4)} {auth.strftime('%m/%d/%y')} {s} {loc} C#{u.last4}"
            out = collapse(out)[:48]
            return out
        else:  # pad: fixed-width descriptor as some card processors send it
            if online:
                return s
            return f"{s[:22]:<22} {city[:13]:<13} {st}"
        return collapse(out)
    if m.channel in ("ach", "check") and m.channel == "ach":
        co, des = m.ach
        key = m.mid
        if key not in u.coid:
            u.coid[key] = rdig(rng, 10, True)
        coid = u.coid[key]
        if style == "clean":
            return m.clean if m.kind != "payroll" else co.title() + (" Payroll" if "PAY" in des or "DEP" in des else "")
        if style == "chase":
            return f"{co} {des} PPD ID: {coid}"
        if style == "boa":
            return f"{co} DES:{des} ID:{rdig(rng, 9)} INDN:{u.uname} CO ID:{coid} {'PPD' if neg else 'WEB'}"
        if style == "wells":
            return f"{co} {des} {d.strftime('%y%m%d')} {rdig(rng, 9)} {u.uname}"
        if style == "cu":
            return f"ACH {'CREDIT' if neg else 'DEBIT'} {co} {des}"[:48]
        return f"{co} {des}"
    if m.channel == "check":
        u.chk += 1
        return {"chase": f"CHECK {u.chk}", "boa": f"Check {u.chk}", "wells": f"CHECK # {u.chk}", "cu": f"SHARE DRAFT {u.chk}",
                "pad": f"CHECK {u.chk}", "clean": f"Check #{u.chk}"}[style]
    if m.channel == "xfer":
        return {"chase": f"Online Transfer to SAV ...{u.last4s} transaction#: {rdig(rng, 11)} {d.strftime('%m/%d')}",
                "boa": f"Online Banking transfer to SAV {u.last4s} Confirmation# {rdig(rng, 10)}",
                "wells": f"ONLINE TRANSFER REF #IB0{rref(rng, 7)} TO WAY2SAVE SAVINGS XXXXXX{u.last4s} ON {d.strftime('%m/%d/%y')}",
                "cu": "TRANSFER TO SHARE 01", "pad": f"TRANSFER TO SAVINGS {u.last4s}", "clean": "Transfer to Savings"}[style]
    if m.channel == "p2p_venmo":
        return {"chase": f"VENMO PAYMENT {rdig(rng, 10)} WEB ID: 3264681992",
                "boa": f"VENMO DES:PAYMENT ID:{rdig(rng, 10)} INDN:{u.uname} CO ID:3264681992 WEB",
                "wells": f"VENMO PAYMENT {d.strftime('%y%m%d')} {rdig(rng, 10)} {u.uname}", "cu": "ACH DEBIT VENMO PAYMENT",
                "pad": f"VENMO PAYMENT {rdig(rng, 10)}", "clean": "Venmo"}[style]
    if m.channel == "p2p_zelle":
        f, l = m.person
        t = f"{f.title()} {l.title()}"
        return {"chase": f"Zelle payment to {t} JPM99{rref(rng, 6).lower()}", "boa": f"Zelle Transfer Conf# {rref(rng, 9).lower()}; {t}",
                "wells": f"ZELLE TO {l} {f} ON {mmdd} REF # PP0{rref(rng, 7)}", "cu": f"ZELLE DEBIT {f} {l}",
                "pad": f"ZELLE TO {f} {l}", "clean": f"Zelle to {t}"}[style]
    raise ValueError(m.channel)


# --------------------------------------------------------------------------------------
# History slice (as production builds it) and the ideal-reader oracle
# --------------------------------------------------------------------------------------

def build_slice(u, rows, qi):
    q = rows[qi]
    qm = MERCH[q["mid"]]
    chosen = []
    seen = set()

    def take(i):
        if i not in seen and len(chosen) < SLICE_MAX:
            seen.add(i)
            chosen.append(i)

    own = [i for i in range(qi - 1, -1, -1) if rows[i]["mid"] == q["mid"]][:OWN_MAX]
    for i in own:
        take(i)
    if not qm.opaque:
        per, n = Counter(), 0
        g = group(qm.kind)
        for i in range(qi - 1, -1, -1):
            if n >= SIM_MAX:
                break
            rm = MERCH[rows[i]["mid"]]
            if rm.mid == qm.mid or rm.opaque or group(rm.kind) != g or per[rm.mid] >= SIM_PER_PAYEE:
                continue
            per[rm.mid] += 1
            n += 1
            take(i)
    shown = {rows[i]["filed"] for i in chosen}
    for c in u.header(q["date"]):
        if c in shown:
            continue
        for i in range(qi - 1, -1, -1):
            if rows[i]["filed"] == c:
                take(i)
                shown.add(c)
                break
    for i in range(qi - 1, -1, -1):
        if len(chosen) >= SLICE_MAX:
            break
        take(i)
    return sorted(chosen)


US_STATES = set("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA "
                "RI SC SD TN TX UT VT VA WA WV WI WY DC".split())


def loc_state(text):
    """State code in the location slot of a card string (end of string, or before Wells-style S#/CARD)."""
    t = re.sub(r"\s+S\d{11}\s+CARD\s+\d{4}$", "", text.strip())
    t = re.sub(r"\s+C#\d{4}$", "", t)
    mt = re.search(r"\b([A-Z]{2})$", t)
    return mt.group(1) if mt and mt.group(1) in US_STATES else None


def lpdf(a, med, sig):
    a = max(abs(a), .01)
    sig = max(sig, .03)
    z = (math.log(a) - math.log(max(abs(med), .01))) / sig
    return math.exp(-.5 * z * z) / (a * sig)


def majority(labels):
    c = Counter(labels)
    top = max(c.values())
    for lab in reversed(labels):  # ties go to the most recent
        if c[lab] == top:
            return lab


def bayes(u, q, payee, header, restrict=None, dn_evidence=True):
    mid, d, a, wd = q["mid"], q["date"], abs(q["amount"]), q["wd"]
    score, prior = {}, {}
    for r, (w, med, sig) in u.roles_at(mid, d).items():
        c = u.map(mid, r, d, a, wd, dn_evidence=dn_evidence)
        score[c] = score.get(c, 0) + w * lpdf(a, med, sig)
        prior[c] = prior.get(c, 0) + w
    for r, (amt, var) in u.recroles.get(mid, {}).items():
        c = u.map(mid, r, d, a, wd, dn_evidence=dn_evidence)
        score[c] = score.get(c, 0) + .35 * lpdf(a, amt, max(var, .05))
        prior[c] = prior.get(c, 0) + .35
    tot = sum(prior.values())
    marg = sum(score.values()) / tot
    n = Counter(r["filed"] for r in payee)
    post = {}
    for c in list(score) + [c for c in n if c not in score]:
        if c not in header or (restrict and c not in restrict):
            continue
        if c in score:
            post[c] = (2 * prior[c] / tot + n[c]) * score[c] / prior[c]
        else:
            post[c] = n[c] * marg
    z = sum(post.values())
    if z <= 0:
        return None
    c = max(post, key=post.get)
    return c if post[c] / z >= .5 else None


def oracle(u, rows, qi, sl):
    """What an ideal reader would answer from this prompt alone (None = no basis for a guess)."""
    q = rows[qi]
    m = MERCH[q["mid"]]
    header = u.header(q["date"])
    srows = [rows[i] for i in sl]
    payee = [r for r in srows if r["mid"] == q["mid"]]
    t = u.trip
    if t and t["cat"] in header:
        st = loc_state(q["text"])
        at_dest, at_home = st == t["st"], st == u.st
        ev_any = any(r["filed"] == t["cat"] for r in srows)
        # the trip looks ongoing: the latest rows before the query are trip rows from the last few days
        ev_near = any(r["filed"] == t["cat"] and (q["date"] - r["date"]).days <= 1 for r in srows)
        if m.kind in TRIP_KINDS or m.local:
            if at_dest and (t["token"] or ev_any):
                return t["cat"]
            if not at_home and not at_dest and ev_near and m.kind not in ("airline",):
                return t["cat"]
        if m.kind in ("airline", "lodging") and q["date"] <= t["t1"] and q["amount"] > 0:
            return t["cat"]
    key = m.key
    named = [c for c in header if key and key in c.lower()]
    if len(named) == 1:
        return named[0]
    dn_ev = u.datenight is not None and any(r["filed"] == u.datenight for r in srows)
    if q["amount"] < 0 and payee and MERCH[q["mid"]].kind not in ("payroll", "interest", "tax_refund"):
        for r in reversed(payee):
            if abs(r["amount"] + q["amount"]) < .005:
                return r["filed"]
    if payee:
        if q["mid"] in u.idio:
            return majority([r["filed"] for r in payee])
        if q["mid"] in u.changed or (q["mid"] in u.mnew):
            return payee[-1]["filed"]
        cats = u.cats_of(q["mid"], q["date"])
        if len(named) > 1 or len(cats) > 1:
            return bayes(u, q, payee, header, restrict=named if len(named) > 1 else None, dn_evidence=dn_ev)
        if u.newcat and q["date"] >= u.newcat_date and cats and cats[0] == u.newcat:
            if any(r["date"] >= u.newcat_date for r in payee):
                return payee[-1]["filed"]
            return u.newcat
        if u.rule_kind(q["mid"]):
            return u.map(q["mid"], m.primary, q["date"], abs(q["amount"]), q["wd"], dn_evidence=dn_ev)
        return majority([r["filed"] for r in payee])
    if m.opaque:
        return None
    if len(named) > 1 or len(u.cats_of(q["mid"], q["date"])) > 1:
        return bayes(u, q, [], header, restrict=named if len(named) > 1 else None, dn_evidence=dn_ev)
    role = ALIAS.get(m.primary, m.primary)
    if role in u.unguess or q["mid"] in u.mrole_unguess:
        g = group(m.kind)
        sim = [r["filed"] for r in srows if not MERCH[r["mid"]].opaque and group(MERCH[r["mid"]].kind) == g]
        return majority(sim) if sim else None
    return u.map(q["mid"], m.primary, q["date"], abs(q["amount"]), q["wd"], dn_evidence=dn_ev)


# --------------------------------------------------------------------------------------
# What decides each candidate query
# --------------------------------------------------------------------------------------

ORDINARY = ("plain", "recurring", "income")


def why_of(u, rows, i, first_idx):
    q = rows[i]
    m = MERCH[q["mid"]]
    mid = q["mid"]
    if u.short:
        return "short_history"
    if q["amount"] < 0 and q["role"] not in INCOME_ROLES:
        return "refund"
    if q["role"] in INCOME_ROLES:
        return "income"
    if q["trip"] and u.trip and q["intended"] == u.trip["cat"]:
        return "trip"
    if u.newcat and q["intended"] == u.newcat:
        return "new_category"
    if mid in u.changed and q["date"] >= u.changed[mid][0]:
        return "changed_mind"
    if first_idx[mid] == i:
        return "new_merchant_opaque" if m.opaque else "new_merchant_descriptive" if m.local else "new_merchant_chain"
    if mid in u.named:
        return "named_category"
    if mid in u.idio:
        return "event" if u.idio[mid] == "Wedding" else "idiosyncratic"
    if q["role"] == "wedding":
        return "event"
    cats = u.cats_of(mid, q["date"])
    if m.kind == "p2p" and len(cats) > 1:
        return "ambiguous_p2p"
    rk = u.rule_kind(mid)
    if rk:
        return rk
    if len(cats) > 1:
        return "amount_split" if m.informative or mid in u.recroles else "multi_purpose"
    prev = [j for j in range(i - 1, -1, -1) if rows[j]["mid"] == mid][:OWN_MAX]
    if any(rows[j]["filed"] != rows[j]["intended"] for j in prev):
        return "misfiled_history"
    if q["rec"]:
        return "recurring"
    return "plain"


def fmt_amt(a):
    return f"${a:.2f}"


def build_near(u, rows, qi, exclude=()):
    """Row 117: the query payee's own latest rows and similar payees' rows before qi, chosen as build_slice chooses them first."""
    q = rows[qi]
    qm = MERCH[q["mid"]]
    own = [i for i in range(qi - 1, -1, -1) if rows[i]["mid"] == q["mid"] and i not in exclude][:OWN_MAX]
    sim = []
    if not qm.opaque:
        per = Counter()
        g = group(qm.kind)
        for i in range(qi - 1, -1, -1):
            if len(sim) >= SIM_MAX:
                break
            rm = MERCH[rows[i]["mid"]]
            if rm.mid == qm.mid or rm.opaque or group(rm.kind) != g or per[rm.mid] >= SIM_PER_PAYEE or i in exclude:
                continue
            per[rm.mid] += 1
            sim.append(i)
    return own + sim


def build_shared(u, hist, header):
    """Row 117: one sync's shared rows from the filed history: the latest row of each category (the most recent SHARED_CAT_MAX), then
    the latest rows, SHARED_MAX in all; indices into hist."""
    last = {}
    for i, r in enumerate(hist):
        if r["filed"] in header:
            last[r["filed"]] = i
    chosen = sorted(last.values(), reverse=True)[:SHARED_CAT_MAX]
    for i in range(len(hist) - 1, -1, -1):
        if len(chosen) >= SHARED_MAX:
            break
        if i not in chosen:
            chosen.append(i)
    return sorted(chosen)


def make_prompt_split(header, shared, near, q):
    parts = ["Categories: " + ", ".join(header), "", SHARED_HEAD, ""]
    for r in shared:
        parts += [f"Transaction: {r['date'].isoformat()} | {r['text']} | {fmt_amt(r['amount'])} | {r['wd']}", f"Category: {r['filed']}", ""]
    parts += [NEAR_HEAD, ""]
    for r in near:
        parts += [f"Transaction: {r['date'].isoformat()} | {r['text']} | {fmt_amt(r['amount'])} | {r['wd']}", f"Category: {r['filed']}", ""]
    parts.append(f"Transaction: {q['date'].isoformat()} | {q['text']} | {fmt_amt(q['amount'])} | {q['wd']}")
    parts.append("Category:")
    return "\n".join(parts)


def bulk_items(u, rows, first_idx):
    """Row 117: one sync for a full user: (plain items, split items), same ids and answers."""
    rng = random.Random(f"bulk-{u.uid}")
    cands = [i for i, r in enumerate(rows) if r["date"] >= QUERY_FROM and i >= 30 and i + BULK_N <= len(rows)]
    if not cands:
        return [], []
    c = rng.choice(cands)
    hist, batch = rows[:c], rows[c:c + BULK_N]
    header = u.header(batch[-1]["date"])  # the user's categories when the sync is read
    shared = build_shared(u, hist, header)
    plain, split = [], []
    for n, q in enumerate(batch):
        if q["intended"] not in header:
            continue
        seq = hist + [q]
        near = sorted(build_near(u, seq, len(hist), exclude=set(shared)))
        sl = sorted(set(shared) | set(near))
        assert all(hist[i]["date"] <= q["date"] and hist[i]["date"] <= batch[0]["date"] for i in sl)
        best_cat = oracle(u, seq, len(hist), sl)
        why = why_of(u, rows, c + n, first_idx)
        if why in ("plain", "recurring") and any(hist[i]["mid"] == q["mid"] and hist[i]["filed"] != hist[i]["intended"] for i in sl):
            why = "misfiled_history"
        base = {"id": f"BK:{u.uid}:{n}", "level": f"BK_{u.scheme}", "user": u.uid, "sync": u.uid, "cutoff": batch[0]["date"].isoformat(),
                "pos": n, "merchant": MERCH[q["mid"]].clean, "text": q["text"], "amount": q["amount"], "weekday": q["wd"],
                "date": q["date"].isoformat(), "options": [" " + x for x in header], "answer": header.index(q["intended"]),
                "best": header.index(best_cat) if best_cat in header else -1, "why": why, "scheme": u.scheme, "hist_len": c,
                "in_shots": any(hist[i]["mid"] == q["mid"] for i in sl), "n_shared": len(shared), "n_near": len(near)}
        past = [r["filed"] for r in hist if r["mid"] == q["mid"]]  # row 134: YNAB's current rule on the history filed before the sync
        if not past:
            rule = None
        elif len(past) < 3:
            rule = past[-1]
        else:
            top, k_top = Counter(past[-3:]).most_common(1)[0]
            rule = top if k_top >= 2 else past[-1]
        base["ynab_rule"] = header.index(rule) if rule in header else -1
        base["payee_hist"] = sorted({header.index(c) for c in past if c in header})  # row 135
        p1 = make_prompt(header, [hist[i] for i in sl], q)
        p2 = make_prompt_split(header, [hist[i] for i in shared], [hist[i] for i in near], q)
        plain.append(dict(base, prompt=p1, prompt_ctx=p1))
        split.append(dict(base, prompt=p2, prompt_ctx=p2))
    return plain, split


def make_prompt(header, srows, q):
    parts = ["Categories: " + ", ".join(header), ""]
    for r in srows:
        parts.append(f"Transaction: {r['date'].isoformat()} | {r['text']} | {fmt_amt(r['amount'])} | {r['wd']}")
        parts.append(f"Category: {r['filed']}")
        parts.append("")
    parts.append(f"Transaction: {q['date'].isoformat()} | {q['text']} | {fmt_amt(q['amount'])} | {q['wd']}")
    parts.append("Category:")
    return "\n".join(parts)


def main():
    master = random.Random(SEED)
    users = []
    ids = list(range(N_FULL + N_SHORT))
    short_ids = set(master.sample(ids, N_SHORT))
    picked = Counter()
    items = []
    bulk_plain, bulk_split = [], []
    ynab = {}  # row 134
    payee_hist = {}  # row 135
    others_rows = {}  # row 124: mid -> [(date, uid, filed)] over every user (build_user seeds each user's own rng, so this pre-pass changes nothing)
    if OTHERS:
        for uid in ids:
            u0, rng0 = build_user(uid, uid in short_ids)
            for r in generate(u0, rng0):
                others_rows.setdefault(r["mid"], []).append((r["date"], uid, r["filed"]))
    for uid in ids:
        u, rng = build_user(uid, uid in short_ids)
        rows = generate(u, rng)
        users.append(u)
        first_idx = {}
        for i, r in enumerate(rows):
            first_idx.setdefault(r["mid"], i)
        if BULK and not u.short:
            a, b = bulk_items(u, rows, first_idx)
            bulk_plain += a; bulk_split += b
        if u.short:
            n = min(11, len(rows))
            chosen = sorted(rng.sample(range(n), min(Q_PER_USER, n)))
            tags = {i: "short_history" for i in chosen}
        else:
            cands = [i for i, r in enumerate(rows) if r["date"] >= QUERY_FROM and i >= 15]
            by = {}
            for i in cands:
                by.setdefault(why_of(u, rows, i, first_idx), []).append(i)
            n_int = rng.choice([2, 2, 3, 3])
            chosen, mids, tags = [], set(), {}
            interesting = [w for w in by if w not in ORDINARY]
            for _ in range(n_int):
                avail = [w for w in interesting if any(rows[i]["mid"] not in mids for i in by[w]) and w not in tags.values()]
                if not avail:
                    avail = [w for w in interesting if any(rows[i]["mid"] not in mids for i in by[w])]
                if not avail:
                    break
                low = min(picked[w] for w in avail)
                w = rng.choice(sorted(x for x in avail if picked[x] == low))
                i = rng.choice([i for i in by[w] if rows[i]["mid"] not in mids])
                chosen.append(i)
                mids.add(rows[i]["mid"])
                tags[i] = w
                picked[w] += 1
            while len(chosen) < Q_PER_USER:
                w = pick(rng, ["plain", "recurring", "income"], [.65, .27, .08])
                pool = [i for i in by.get(w, []) if rows[i]["mid"] not in mids and i not in chosen]
                if not pool:
                    pool = [i for x in ORDINARY for i in by.get(x, []) if rows[i]["mid"] not in mids and i not in chosen]
                if not pool:
                    pool = [i for i in cands if i not in chosen]
                i = rng.choice(pool)
                chosen.append(i)
                mids.add(rows[i]["mid"])
                tags[i] = why_of(u, rows, i, first_idx)
            chosen.sort()
        for n, qi in enumerate(chosen):
            n_q = n
            q = rows[qi]
            header = u.header(q["date"])
            sl = build_slice(u, rows, qi)
            srows = [rows[i] for i in sl]
            prompt = make_prompt(header, srows, q)
            if YNAB_RULE:
                past = [r["filed"] for r in rows[:qi] if r["mid"] == q["mid"]]
                last3 = past[-3:]
                if not past:
                    sug = None
                elif len(last3) < 3:
                    sug = past[-1]
                else:
                    top, k_top = Counter(last3).most_common(1)[0]
                    sug = top if k_top >= 2 else past[-1]
                ynab[f"BL:{u.uid}:{n_q}"] = header.index(sug) if sug in header else -1
                payee_hist[f"BL:{u.uid}:{n_q}"] = sorted({header.index(c) for c in past if c in header})  # row 135: every category filed for this payee before
            if OTHERS:
                cnt = Counter(f for d, v, f in others_rows.get(q["mid"], []) if v != u.uid and d < q["date"])
                if sum(cnt.values()) >= 3:
                    line = OTHERS_HEAD + ", ".join(f"{c} ({n})" for c, n in cnt.most_common(3))
                    head, _, last = prompt.rpartition("\nTransaction: ")
                    prompt = head + "\n" + line + "\n\nTransaction: " + last
            if SPLIT:
                near = [i for i in build_near(u, rows, qi) if i in sl]
                prompt = make_prompt_split(header, [rows[i] for i in sl if i not in near], [rows[i] for i in sorted(near)], q)
            why = tags[qi]
            if why == "short_history" and not u.short:
                why = "plain"
            best_cat = oracle(u, rows, qi, sl)
            if why in ("plain", "recurring") and any(r["mid"] == q["mid"] and r["filed"] != r["intended"] for r in srows):
                why = "misfiled_history"
            ans = header.index(q["intended"])
            items.append({
                "id": f"BL:{u.uid}:{n}", "level": f"BL_{u.scheme}", "user": u.uid, "merchant": MERCH[q["mid"]].clean,
                "text": q["text"], "amount": q["amount"], "weekday": q["wd"], "date": q["date"].isoformat(),
                "prompt": prompt, "prompt_ctx": prompt, "options": [" " + c for c in header], "answer": ans,
                "best": header.index(best_cat) if best_cat in header else -1, "why": why, "scheme": u.scheme,
                "hist_len": qi, "in_shots": any(r["mid"] == q["mid"] for r in srows)})
    if YNAB_RULE:
        (OUT.parent / "blind_v1_ynabrule.json").write_text(json.dumps(ynab, indent=0))
        (OUT.parent / "blind_v1_payeehist.json").write_text(json.dumps(payee_hist, indent=0))
        print(f"wrote blind_v1_ynabrule.json: {len(ynab)} items, {sum(v >= 0 for v in ynab.values())} with a suggestion")
        return
    if BULK:
        rule, ph = {}, {}
        for its in (bulk_plain, bulk_split):
            for it in its:
                rule[it["id"]] = it.pop("ynab_rule"); ph[it["id"]] = it.pop("payee_hist")
        (OUT.parent / "blind_bulk_v1_ynabrule.json").write_text(json.dumps(rule, indent=0))
        (OUT.parent / "blind_bulk_v1_payeehist.json").write_text(json.dumps(ph, indent=0))  # row 135  # row 134: YNAB's current rule per bulk item
        for name, its in (("blind_bulk_v1", bulk_plain), ("blind_bulk_v1_split", bulk_split)):
            sha = hashlib.sha256(json.dumps(its, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
            out = {"name": name, "version": "v1", "n_users": len({i["user"] for i in its}), "items": its, "sha256": sha,
                   "notes": "Row 117 (main session): one sync per full blind_v1 user; see the BLIND_BULK comment in scripts/build_blind_v1.py."}
            path = OUT.parent / f"{name}.json"
            if path.exists():  # frozen (DVC, read-only): check it is what the generator makes, do not rewrite it
                assert json.loads(path.read_text())["sha256"] == sha, f"{name} differs from the frozen set"
                print(f"{name}: frozen set matches (sha {sha[:12]})")
                continue
            path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
            ceil = sum(i["best"] == i["answer"] for i in its) / len(its)
            print(f"wrote {name} items={len(its)} users={out['n_users']} sha={sha[:12]} ceiling {ceil:.3f} in_shots {sum(i['in_shots'] for i in its)}")
        return
    # checks
    for it in items:
        hdr = [o[1:] for o in it["options"]]
        assert all(", " not in c for c in hdr), hdr
        assert len(set(hdr)) == len(hdr), hdr
        lines = it["prompt"].split("\n")
        assert lines[0] == "Categories: " + ", ".join(hdr) and lines[-1] == "Category:"
        for ln in lines:
            if ln.startswith("Category: "):
                assert ln[10:] in hdr, (it["id"], ln)
            if ln.startswith("Transaction: "):
                assert ln[13:23] <= it["date"]
                assert ln.count(" | ") == 3, ln
        assert len(it["prompt"]) < (10000 if SLICE_MAX > 24 else 5000), (it["id"], len(it["prompt"]))
    notes = (
        "Blind v1: 250 simulated budgeting-app users over calendar 2025 (222 with up to a year of history, some joining "
        "Feb-Jun; 28 new users who joined in Nov-Dec and are queried on their first 0-10 transactions). Schemes: default "
        "YNAB-like lists with personal renames (Dining out/Eating out/Restaurants, Ready to Assign/Income, split or merged "
        "utilities), family lists with per-child categories, merged 8-12 category lists (Food, Bills, Fun), detailed 35-45 "
        "category lists (Coffee vs Takeout vs Dining out, Costco - food vs Costco - household, Work lunches on weekdays), "
        "personal odd-name lists (Food we cook, Wheels, Treat yo self, Kid 1/Kid 2, Future us) and minimal 5-7 category "
        "lists. Merchants: ~250 real chains and billers plus per-user invented local businesses (descriptive names behind "
        "SQ */TST*/SP/PY*/CLV* prefixes) and opaque LLC-style names; each user's bank renders strings in one of six styles "
        "(plain card descriptor with city/state, BoA-style with auth date and PURCHASE, Wells-style PURCHASE AUTHORIZED ON "
        "with card digits, credit-union POS DEBIT/DBT CRD truncated to 48 chars, fixed-width 22/13-column descriptors, or "
        "fintech-cleaned names), ACH items as PPD/DES:/INDN: lines, Venmo without a counterparty, Zelle with one. "
        "Behaviours: recurring rent/mortgage/utilities/subscriptions/payroll/savings transfers with fixed days and amounts "
        "(seasonal utility bills, a rent increase, a Netflix price rise); multi-purpose stores (Amazon, Target, Walmart, "
        "Costco, Kohl's, pharmacies) whose purchases split by what was bought, with amounts drawn per purpose; fuel "
        "stations that sell snacks; date night on Fri/Sat above a per-user amount; weekday work lunches and reimbursable "
        "weekday rideshares; trips with their own category (bookings months ahead, spending at the destination); a "
        "wedding; categories created mid-year (Baby, Home office, Dog, Christmas, Amazon, Car repairs) that take over "
        "merchants; one merchant per user refiled mid-year (changed mind); merchants filed for the user's own reasons "
        "(Work expenses, Rental property, Mom); merchant-named categories; refunds (negative, filed to the original "
        "purchase's category); income as negative amounts; new chains, new local businesses and opaque names late in "
        "the year; and 2% (6% for sloppy users) mis-filed history rows (next category in the list or one for a related purpose) (answers are always the intended category). "
        "`best` is a rule-based ideal reader that sees only the prompt plus world knowledge of merchants: it uses the "
        "payee's own rows (majority, or latest after a change), a Bayes rule over purposes and amounts for multi-purpose "
        "payees (-1 when no category reaches 50%), the trip/date-night/weekday rules when the slice shows them, category "
        "names for new merchants, and -1 for opaque merchants with no history.")
    sha = hashlib.sha256(json.dumps(items, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    out = {"name": "blind", "version": "v1", "n_users": len(users), "items": items, "sha256": sha, "notes": notes}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    # summary
    print(f"wrote {OUT} users={len(users)} items={len(items)} sha={sha[:12]}")
    by_why, by_scheme = Counter(i["why"] for i in items), Counter(i["scheme"] for i in items)
    ceil = lambda its: sum(i["best"] == i["answer"] for i in its) / max(1, len(its))
    print(f"ceiling overall {ceil(items):.3f}; best=-1 {sum(i['best'] == -1 for i in items)}")
    for w, n in by_why.most_common():
        its = [i for i in items if i["why"] == w]
        print(f"  {w:26s} {n:4d}  ceiling {ceil(its):.2f}  unknowable {sum(i['best'] == -1 for i in its)}")
    for s, n in by_scheme.most_common():
        print(f"  scheme {s:10s} {n:4d}  ceiling {ceil([i for i in items if i['scheme'] == s]):.2f}")
    print("ordinary share", sum(i["why"] in ORDINARY for i in items) / len(items))
    print("max prompt chars", max(len(i["prompt"]) for i in items), "in_shots", sum(i["in_shots"] for i in items))


if __name__ == "__main__":
    main()
