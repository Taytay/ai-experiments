# Overture Maps places theme, release 2026-09-23.1

- **Source:** `s3://overturemaps-us-west-2/release/2026-09-23.1/theme=places/type=place/` (Overture Maps Foundation, AWS Open Data
  registry, anonymous access), 16 GeoParquet files, 11.0 GB, 81,455,423 places. Downloaded 2026-09-26 for the `ai-experiments` repo
  (PLAN row 62 and later rows; the owner asked for the whole places theme locally, with licence provenance). File sizes in
  `MANIFEST.json`, verified against S3 after download.
- **Licence:** the places theme has **no single licence**. Overture's attribution page (`LICENSES/overture_attribution_page.html`,
  fetched 2026-09-26 from https://docs.overturemaps.org/attribution/) lists per-provider terms, and every place carries its own
  `sources` list with a `license` per source. Counts in this release (a place can have several sources):

  | source dataset | licence | places |
  |---|---|---|
  | Overture | CDLA-Permissive-2.0 | 81,455,423 |
  | meta | CDLA-Permissive-2.0 | 58,783,121 |
  | Overture-signals | CDLA-Permissive-2.0 | 10,583,924 |
  | BrightQuery | CDLA-Permissive-2.0 | 10,255,071 |
  | Microsoft | CDLA-Permissive-2.0 | 6,135,466 |
  | Foursquare | Apache-2.0 | 4,138,835 |
  | AllThePlaces | CC0-1.0 | 1,809,219 |
  | PinMeTo | CDLA-Permissive-2.0 | 167,667 |
  | DAC | CDLA-Permissive-2.0 | 148,791 |
  | Krick | CDLA-Permissive-2.0 | 13,501 |
  | RenderSEO | CDLA-Permissive-2.0 | 3,752 |

- **Required attribution, as the attribution page gives it:** "Data from Meta. Available under CDLA Permissive 2.0"; the same
  wording for Microsoft, PinMeTo, Krick, RenderSEO, DAC and BrightQuery; for Foursquare "Copyright 2024 Foursquare Labs, Inc. All
  rights reserved. Available under Apache 2.0" (Foursquare data transformed to the Overture schema, changed 2026-03-18; its NOTICE
  is in `LICENSES/foursquare_places_NOTICE.html`); "Data from AllThePlaces. Available under CC0 1.0". Check the saved page for the
  exact current text before any redistribution.
- **Licence texts** (SPDX list, fetched 2026-09-26): `LICENSES/CDLA-Permissive-2.0.txt`, `LICENSES/Apache-2.0.txt`,
  `LICENSES/CC0-1.0.txt`; sha256 of every saved document in `MANIFEST.json`.
- **Use in this repo:** research on merchant categorisation; any subset or derived item set frozen into the repo keeps each
  place's `id` and `sources` (dataset, licence, record id) so its provenance stays per row, and carries the attribution above.
