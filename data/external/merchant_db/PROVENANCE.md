# Local merchant database (PLAN step 179)

Built by `scripts/build_merchant_db.py` into `~/projects/YNAB/data/merchants/merchants_v1.db` (outside the repo; rebuildable). Each row
keeps its source and source id. Sources and licences:

- **Curlie** (https://curlie.org), monthly directory dump (https://curlie.org/directory-dl, file dated 2026-02-02), **CC BY 3.0 Unported**.
  Required attribution: "Data from Curlie.org (https://curlie.org), licensed under CC BY 3.0." Consumer branches (Shopping, Health,
  Recreation, Home, Sports, Computers, Arts, Business, Games) and US regional business listings.
- **DMOZ / Open Directory Project**, final dump (content.rdf.u8.gz, 2017-03-12, via the Internet Archive), **CC BY 3.0**. Attribution:
  "Data from the Open Directory Project (DMOZ), licensed under CC BY 3.0." Used only for domains Curlie no longer lists.
- **Wikidata** (https://www.wikidata.org), via QLever, **CC0 1.0** (`data/external/wikidata_merchants/`).
- **Overture Maps places** brands (release 2026-09-23.1): CDLA-Permissive-2.0 / Apache-2.0 / CC0 per source (`data/external/overture_places_2026-09-23.1/`).
- **Majestic Million** (https://majestic.com/reports/majestic-million), **CC BY 3.0**: popularity rank by domain. Attribution: "Majestic Million, licensed under CC BY 3.0."
