"""PLAN step 179 (owner, 2026-10-03: "I'd like to build a local database using this. Then let's try to build that info into weights."):
a local merchant database from open sources, one row per merchant, with a trigram full-text index for fuzzy lookup of cleaned payees.

Sources (all open licences; each row keeps its source and source id):
  wikidata  data/external/wikidata_merchants/merchants_v1.jsonl (CC0; build_wikidata_merchants.py + map_wikidata_kinds.py)
  curlie    ~/projects/YNAB/data/curlie/curlie-rdf (monthly dump, CC BY 3.0, attribution: "Curlie.org"): consumer branches and US regional
            business listings
  dmoz      ~/projects/YNAB/data/dmoz/content.rdf.u8.gz, the Open Directory's final dump (March 2017; CC BY 3.0): consumer branches of
            Top/ (Shopping, Health, Recreation, Home, Sports, Computers, Arts, Business; not World or Regional), one row per site:
            title, description, category path
  overture  ~/projects/YNAB/data/overture/derived/brands.parquet (CDLA-Permissive-2.0 / Apache-2.0 / CC0 per place): brand names with
            their most common category and US place count
  majestic  ~/projects/YNAB/data/majestic/majestic_million.csv (CC BY 3.0): global rank by referring subnets, joined on domain
Kinds: taxonomy_v2 kinds by map_wikidata_kinds.RULES over "name | description | category path" (Overture: its basic_category through
taxonomy_v2.overture_kind). Rows: name, aliases, domain, description, category path, kind, source, source id, rank (Majestic, lower is
more popular) or US places (Overture), split ("train" / "test" by a hash of the name: a quarter held out).
Writes ~/projects/YNAB/data/merchants/merchants_v1.db (outside the repo; public data, rebuildable) and prints counts.
usage: uv run --with duckdb python scripts/build_merchant_db.py
"""
import csv
import gzip
import hashlib
import html
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from map_wikidata_kinds import RX  # noqa: E402

from ai_experiments import taxonomy_v2 as T  # noqa: E402
from ai_experiments.paths import ROOT  # noqa: E402

DATA = Path.home() / "projects" / "YNAB" / "data"
DB = DATA / "merchants" / "merchants_v1.db"
KEEP = re.compile(r"^Top/(Shopping|Health|Recreation|Home|Sports|Computers|Arts|Business)/")
DMOZ_KIND = [  # category-path hints before the text rules (DMOZ paths are precise)
    ("medical", r"/Health/(Vision|Dentistry|Medicine|Pharmacy/Online)|/Shopping/Health/(Vision|Optical|Hearing)|Contact_Lenses"),
    ("pharmacy", r"/Shopping/Health/Pharmacies|/Pharmacy"), ("pet", r"/Pets/|/Animals/"), ("hotel", r"/Lodging/"),
    ("airline", r"/Airlines/"), ("car_rental", r"/Car_Rentals/"), ("travel_agency", r"/Travel/(Agents|Tour_Operators|Cruises)"),
    ("grocery", r"/Shopping/Food/|/Groceries/"), ("clothing", r"/Shopping/Clothing/"), ("jewelry", r"/Jewelry/"),
    ("electronics", r"/Shopping/(Electronics|Computers)/"), ("hobby", r"/Shopping/(Toys_and_Games|Crafts|Hobbies|Music|Books|Sports)/"),
    ("furniture", r"/Shopping/Home_and_Garden/(Furniture|Appliances|Bed_and_Bath|Kitchen)"), ("home_improvement", r"/Shopping/Home_and_Garden/"),
    ("gifts", r"/Shopping/(Gifts|Flowers)/"), ("personal_care", r"/Shopping/Health/(Beauty|Personal_Care)|/Beauty/"),
    ("car_repair", r"/Shopping/Vehicles/Parts|/Automotive/"), ("digital", r"/Games/(Video_Games|Online)|/Computers/Software/"),
    ("subscription", r"/Internet/(Web_Hosting|Online_Services)|/Streaming"), ("shipping", r"/Shipping|/Postal"),
]
DMOZ_RX = [(k, re.compile(r)) for k, r in DMOZ_KIND]


def kind_of(text, path=""):
    for k, rx in DMOZ_RX:
        if path and rx.search(path):
            return k
    t = (text + " | " + path.replace("_", " ").replace("/", " ")).lower()
    for k, rx in RX:
        if rx.search(t):
            return k
    return None


def domain_of(url):
    m = re.match(r"^(?:https?://)?(?:www\d?\.)?([^/:?#]+)", url or "", re.I)
    return m.group(1).lower() if m else ""


def split_of(name):
    return "test" if int(hashlib.sha256(name.lower().encode()).hexdigest(), 16) % 4 == 0 else "train"


CURLIE_KEEP = re.compile(r"^(Shopping|Health|Recreation|Home|Sports|Computers|Arts|Business|Games)/")
CURLIE_US = re.compile(r"^Regional/North_America/United_States/.*/(Business_and_Economy|Shopping|Health|Recreation_and_Sports|Arts_and_Entertainment|Travel_and_Tourism)(/|$)")


def curlie_rows():
    """Curlie's monthly dump (https://curlie.org/directory-dl, CC BY 3.0; the living successor of DMOZ): sites (URL, title, description,
    category id) joined to categories (id, full path); consumer branches and US regional business listings."""
    base = DATA / "curlie" / "curlie-rdf"
    paths = {}
    for f in ("rdf-Top-s.tsv", "rdf-Business-s.tsv", "rdf-Arts-s.tsv", "rdf-NorthAmerica-s.tsv"):
        for line in open(base / f, encoding="utf-8", errors="replace"):
            c = line.rstrip("\n").split("\t")
            if len(c) >= 2 and (CURLIE_KEEP.match(c[1]) or CURLIE_US.match(c[1])):
                paths[c[0]] = c[1]
    for f in ("rdf-Top-c.tsv", "rdf-Business-c.tsv", "rdf-Arts-c.tsv", "rdf-NorthAmerica-c.tsv"):
        for line in open(base / f, encoding="utf-8", errors="replace"):
            c = line.rstrip("\n").split("\t")
            if len(c) >= 4 and c[3] in paths:
                yield dict(url=c[0], title=c[1], desc=c[2], topic=paths[c[3]])


def dmoz_rows():
    """Stream the RDF: <ExternalPage about="URL"><d:Title>..</d:Title><d:Description>..</d:Description><topic>Top/..</topic></ExternalPage>."""
    page, buf = None, []
    with gzip.open(DATA / "dmoz" / "content.rdf.u8.gz", "rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("<ExternalPage"):
                page = {"url": re.search(r'about="([^"]*)"', line).group(1)}
            elif page is not None:
                if line.startswith("<d:Title>"):
                    page["title"] = html.unescape(re.sub(r"</?d:Title>", "", line))
                elif line.startswith("<d:Description>"):
                    page["desc"] = html.unescape(re.sub(r"</?d:Description>", "", line))
                elif line.startswith("<topic>"):
                    page["topic"] = re.sub(r"</?topic>", "", line)
                elif line.startswith("</ExternalPage>"):
                    if KEEP.match(page.get("topic", "")):
                        yield page
                    page = None


if __name__ == "__main__":
    import duckdb
    DB.parent.mkdir(parents=True, exist_ok=True)
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE merchants (id INTEGER PRIMARY KEY, name TEXT, aliases TEXT, domain TEXT, description TEXT, path TEXT, kind TEXT,
                   source TEXT, source_id TEXT, rank INTEGER, us_places INTEGER, split TEXT)""")
    rows = []
    for line in open(ROOT / "data" / "external" / "wikidata_merchants" / "merchants_v1.jsonl"):  # Wikidata (kinds from map_wikidata_kinds)
        e = json.loads(line)
        if not e.get("kind") or re.search(r"\b(elementary|middle|high) school\b|school district", e.get("description", ""), re.I):
            continue
        rows.append((e["label"], "|".join(e.get("aliases", [])), domain_of(e.get("website", "")), e.get("description", ""),
                     "; ".join(e.get("industry", []) + e.get("products", [])), e["kind"], "wikidata", e["qid"], None, None, split_of(e["label"])))
    n_wd = len(rows)
    seen = set()
    for src, gen in (("curlie", curlie_rows), ("dmoz", dmoz_rows)):  # Curlie first; DMOZ 2017 only for domains Curlie no longer lists
        for p in gen():
            d = domain_of(p["url"])
            if src == "dmoz" and d in seen:
                continue
            k = kind_of(f"{p.get('title', '')} | {p.get('desc', '')}", "Top/" + p["topic"] if src == "curlie" else p["topic"])
            if k:
                seen.add(d)
                rows.append((p.get("title", ""), "", d, p.get("desc", ""), p["topic"], k, src, p["url"], None, None, split_of(p.get("title", ""))))
    n_dmoz = len(rows) - n_wd
    for brand, n, basic in duckdb.connect().execute(f"SELECT brand, n, basic_category FROM '{DATA}/overture/derived/brands.parquet' WHERE n >= 5").fetchall():
        k = T.overture_kind(basic)
        if k:
            rows.append((brand, "", "", "", basic, k, "overture", brand, None, n, split_of(brand)))
    con.executemany("INSERT INTO merchants (name, aliases, domain, description, path, kind, source, source_id, rank, us_places, split) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    # popularity: the Majestic rank of the merchant's domain
    con.execute("CREATE TEMP TABLE maj (domain TEXT PRIMARY KEY, rank INTEGER)")
    with open(DATA / "majestic" / "majestic_million.csv") as f:
        con.executemany("INSERT OR IGNORE INTO maj VALUES (?, ?)", ((r["Domain"].lower(), int(r["GlobalRank"])) for r in csv.DictReader(f)))
    con.execute("UPDATE merchants SET rank = (SELECT rank FROM maj WHERE maj.domain = merchants.domain) WHERE domain != ''")
    con.execute("CREATE VIRTUAL TABLE merchants_fts USING fts5(name, aliases, domain, content='merchants', content_rowid='id', tokenize='trigram')")
    con.execute("INSERT INTO merchants_fts(rowid, name, aliases, domain) SELECT id, name, aliases, domain FROM merchants")
    con.commit()
    q = lambda s: con.execute(s).fetchall()  # noqa: E731
    print("by source:", q("SELECT source, COUNT(*) FROM merchants GROUP BY source"))
    print(f"{q('SELECT COUNT(*) FROM merchants')[0][0]} merchants: wikidata {n_wd}, curlie + dmoz {n_dmoz}, overture brands {len(rows) - n_wd - n_dmoz}; "
          f"with a Majestic rank {q('SELECT COUNT(*) FROM merchants WHERE rank IS NOT NULL')[0][0]}, in the top 100k {q('SELECT COUNT(*) FROM merchants WHERE rank <= 100000')[0][0]}")
    print("by kind:", ", ".join(f"{k} {n}" for k, n in q("SELECT kind, COUNT(*) FROM merchants GROUP BY kind ORDER BY 2 DESC")))
    print("fuzzy lookup '1-800 contacts':", q("SELECT name, domain, kind, description FROM merchants WHERE id IN (SELECT rowid FROM merchants_fts WHERE merchants_fts MATCH '\"contacts\"') AND (name LIKE '%800%' OR domain LIKE 'contacts.com') LIMIT 3"))
    print(f"-> {DB} ({DB.stat().st_size / 1e6:.0f} MB)")
