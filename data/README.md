# data/

There is no raw data. Both datasets are synthetic and are regenerated from a seed by the
library modules in `src/`:

- `src/merchants.py`: 120 fictional merchants, 12 spending categories, one bank-statement
  rendering each, the augmentation templates and the evaluation items.
- `src/universe.py`: the 160-species creature universe (type, habitat, region, weakness,
  morphology markers), the field-guide text, the training episodes and the L1 to L7
  evaluation ladder.

`processed/` holds item sets that were frozen to disk so every run scores identical items:

| File | Produced by | Used by |
|---|---|---|
| `processed/icl_suite_items.json` | `src/icl_suite.py` on first use | every curriculum arm (ICL regression suite) |

Queue step 1 in `PLAN.md` adds the frozen ladder, probes and held-out induction items here,
each with a version tag whose hash is recorded in the evals tracker config.
