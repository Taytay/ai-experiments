"""PLAN step 179 (owner, 2026-10-03: "Perhaps we need a database of online retailers and in person merchants that we can embed in the
training"; "Go straight to wikidata as a way to enhance the training. I think we need to embed its knowledge into our weights").

Pulls consumer-facing businesses and brands from Wikidata (CC0) through the QLever endpoint (https://qlever.dev/api/wikidata):
  slices  US businesses with an official website; brands; retail chains (any country), each with its English label, English aliases,
          English description, official website, and the English labels of its industry (P452), products (P1056) and classes (P31)
writes  data/external/wikidata_merchants/raw_<slice>.jsonl (one entity per line; public data, CC0) and the provenance file.
Mapping to taxonomy_v2 kinds is a separate step (scripts/map_wikidata_kinds.py).
usage: uv run python scripts/build_wikidata_merchants.py [slice ...]
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from ai_experiments.paths import ROOT

OUT = ROOT / "data" / "external" / "wikidata_merchants"
ENDPOINT = "https://qlever.dev/api/wikidata"
P = ("PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> "
     "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX schema: <http://schema.org/> PREFIX skos: <http://www.w3.org/2004/02/skos/core#>")
SLICES = {
    "us_businesses_web": "?x wdt:P31/wdt:P279* wd:Q4830453 . ?x wdt:P17 wd:Q30 . ?x wdt:P856 ?web .",
    "brands": "?x wdt:P31/wdt:P279* wd:Q431289 . OPTIONAL { ?x wdt:P856 ?web }",
    "retail_chains": "?x wdt:P31/wdt:P279* wd:Q507619 . OPTIONAL { ?x wdt:P856 ?web }",
}


def query(sparql):
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": sparql})
    req = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json", "User-Agent": "ai-experiments research (YNAB)"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)["results"]["bindings"]
        except Exception as e:  # the public endpoint is shared: back off and retry
            print(f"   retry {attempt + 1}: {e}", flush=True); time.sleep(20 * (attempt + 1))
    raise SystemExit("QLever query failed")


def fetch(name, pattern):
    """One row per entity: grouped label, aliases, description, website, industry / product / class labels."""
    sparql = f"""{P}
    SELECT ?x (SAMPLE(?label) AS ?l) (SAMPLE(?desc) AS ?d) (SAMPLE(?web) AS ?w)
           (GROUP_CONCAT(DISTINCT ?alias; separator="|") AS ?al) (GROUP_CONCAT(DISTINCT ?indl; separator="|") AS ?ind)
           (GROUP_CONCAT(DISTINCT ?prodl; separator="|") AS ?prod) (GROUP_CONCAT(DISTINCT ?clsl; separator="|") AS ?cls)
    WHERE {{ {pattern}
      ?x rdfs:label ?label . FILTER(LANG(?label) = "en")
      OPTIONAL {{ ?x schema:description ?desc . FILTER(LANG(?desc) = "en") }}
      OPTIONAL {{ ?x skos:altLabel ?alias . FILTER(LANG(?alias) = "en") }}
      OPTIONAL {{ ?x wdt:P452 ?indu . ?indu rdfs:label ?indl . FILTER(LANG(?indl) = "en") }}
      OPTIONAL {{ ?x wdt:P1056 ?pr . ?pr rdfs:label ?prodl . FILTER(LANG(?prodl) = "en") }}
      OPTIONAL {{ ?x wdt:P31 ?c . ?c rdfs:label ?clsl . FILTER(LANG(?clsl) = "en") }}
    }} GROUP BY ?x"""
    rows = query(sparql)
    v = lambda r, k: r[k]["value"] if k in r else ""  # noqa: E731
    out = []
    for r in rows:
        out.append(dict(qid=v(r, "x").rsplit("/", 1)[-1], label=v(r, "l"), description=v(r, "d"), website=v(r, "w"),
                        aliases=[a for a in v(r, "al").split("|") if a][:8], industry=[a for a in v(r, "ind").split("|") if a][:6],
                        products=[a for a in v(r, "prod").split("|") if a][:8], classes=[a for a in v(r, "cls").split("|") if a][:6]))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"raw_{name}.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"{name}: {len(out)} entities -> {OUT / f'raw_{name}.jsonl'}", flush=True)


if __name__ == "__main__":
    for name in (sys.argv[1:] or SLICES):
        fetch(name, SLICES[name])
    (OUT / "PROVENANCE.md").write_text(
        "# Wikidata merchants\n\n- Source: Wikidata (https://www.wikidata.org), CC0 1.0 (public domain), queried through QLever "
        f"({ENDPOINT}) on {time.strftime('%Y-%m-%d')}.\n- Slices: " + "; ".join(f"{k}: `{v}`" for k, v in SLICES.items()) +
        "\n- Fields: English label, aliases, description, official website, industry (P452), products (P1056), classes (P31) labels.\n"
        "- Use (PLAN step 179): a merchant database for training decider on what businesses are, online ones included.\n")
