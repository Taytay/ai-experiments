# data/

There is no raw data. Both datasets are synthetic and are regenerated from a seed by the
library modules in `src/ai_experiments/`:

- `merchants.py`: 120 fictional merchants, 12 spending categories, one bank-statement
  rendering each, the augmentation templates and the evaluation items.
- `universe.py`: the 160-species creature universe (type, habitat, region, weakness,
  morphology markers), the field-guide text, the training episodes and the L1 to L7
  evaluation ladder.

`processed/` holds item sets that were frozen to disk so every run scores identical items:

| File | Produced by | Used by |
|---|---|---|
| `processed/icl_suite_items.json` | `ai_experiments.icl_suite` on first use | every curriculum arm (ICL regression suite) |
| `processed/ladder_v1.json`, `heldout_induction_v1.json`, `probes_v1.json` | `uv run python -m ai_experiments.items freeze` (2026-09-14) | every arm on the plain universe (base, A, B, C, Cn, D) and `scripts/rescore.py` |
| `processed/ladder_v1_morph.json`, `heldout_induction_v1_morph.json`, `probes_v1_morph.json` | same, for the morphology universe (`morph_p=0.7`) | arms base_m and E; `rescore.py --morph` |

The frozen files are immutable: every item has an `id` (`<level>:<index>`), each file carries a
sha256 over its items, and `ai_experiments.items.load_all()` hands runs the items plus the hashes
that go into the tracker config (`items_version`, `items_sha`). Per-item scores in
`results/per_item/` are keyed by these ids, so runs from any commit are paired item by item.
Changing a generator does not change what gets scored; `uv run python -m ai_experiments.items check`
reports the drift, and new items mean a new `VERSION` in `items.py`, frozen on purpose.

## External data

`external/` holds the provenance of data downloaded from outside the repo; the data itself is not in git.

| Directory | Data (outside the repo) | Licence |
|---|---|---|
| `external/overture_places_2026-09-23.1/` | Overture Maps places theme, release 2026-09-23.1: 81.5M places, 16 GeoParquet files (11 GB) at `~/projects/YNAB/data/overture/places/2026-09-23.1/` on the WSL machine (downloaded 2026-09-26 from Overture's public S3 bucket) | multi-source, per row in `sources[].license`: CDLA-Permissive-2.0 (most providers), Apache-2.0 (Foursquare), CC0-1.0 (AllThePlaces); see its `PROVENANCE.md`, `MANIFEST.json` and `LICENSES/` |

Anything frozen from it into `processed/` keeps each place's `id` and `sources` so provenance stays per row.
