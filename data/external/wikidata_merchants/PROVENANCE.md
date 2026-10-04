# Wikidata merchants

- Source: Wikidata (https://www.wikidata.org), CC0 1.0 (public domain), queried through QLever (https://qlever.dev/api/wikidata) on 2026-10-03.
- Slices: us_businesses_web: `?x wdt:P31/wdt:P279* wd:Q4830453 . ?x wdt:P17 wd:Q30 . ?x wdt:P856 ?web .`; brands: `?x wdt:P31/wdt:P279* wd:Q431289 . OPTIONAL { ?x wdt:P856 ?web }`; retail_chains: `?x wdt:P31/wdt:P279* wd:Q507619 . OPTIONAL { ?x wdt:P856 ?web }`
- Fields: English label, aliases, description, official website, industry (P452), products (P1056), classes (P31) labels.
- Use (PLAN step 179): a merchant database for training decider on what businesses are, online ones included.
