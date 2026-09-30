# Real-data evaluation: spec (draft, 2026-09-30)

What we need from real anonymised YNAB histories, how we split them, how prompts are built, what we measure, and which decisions each
measurement settles. Written so the harness (`scripts/real_*`, to be built) can follow it step by step. The synthetic findings it will
confirm or overturn are in REPORT.md 116 to 140; the current recommended system is in PLAN.md's current state.

## 1. Data we need

One row per transaction, per user, for at least 12 months:

| field | notes |
|---|---|
| user_id | anonymised, stable |
| date | posted date (and authorised date if available) |
| payee_id, payee_name | YNAB's payee entity and its display name after the user's renames |
| import_string | the raw bank statement string, when the transaction was imported (null for manual entries) |
| amount | signed (outflow positive or negative: state the convention) |
| account_type | checking, credit card, cash, ... (optional) |
| category_id, category_name | the category the user finally confirmed |
| suggested_category_id | what YNAB suggested at the time (to measure the current rule directly, not only our re-implementation) |
| confirmed_at, edited_after_confirm | when it was confirmed; whether the category was changed later (a slip, or a change of mind) |
| split / transfer flags | split transactions and transfers are scored separately or excluded |
| memo | the user's memo or the payment app's note, anonymised (people's names tokenised), not dropped: for person-to-person payees ("Venmo mom") it is often the only sign of what a payment was for (owner, 2026-09-30) |
| payee_kind | a canonical kind per payee if YNAB or a merchant database has one (restaurant, gas station, person-to-person payment, ...): the "Kind:" line of PLAN step 149 |

Per user: the category list with each category's creation date, rename history and hidden / deleted dates (so the prompt shows the
list as it was on the transaction's date). Per payee (optional, from YNAB's side): a merchant database record (canonical name, type of
business), the source of the separate merchant database the model is taught about.

Anonymisation: user and payee ids hashed; person-to-person payees (Venmo, Zelle to a person) with the counterparty's name replaced by a
stable token; amounts kept; memos kept with names and numbers tokenised (they carry the purpose of person-to-person payments). Category names kept as the user wrote them (they are what the model reads); any
that contain personal names (children, partners) may be tokenised the same way.

## 2. Splits (never random transactions)

A random 20% of transactions would put a user's other transactions in training and leak their personal naming and habits, which is
exactly what has to generalise. Splits are by **user**, with a **time** holdout on top:

- Users: 70% train, 15% validation, 15% test, assigned by a hash of user_id (stable as data grows).
- Time: queries for validation and test users come from their last three months; their earlier months are history (the prompt's
  rows), never training data. Training users contribute all months.
- The **test split is sealed**: scored once per candidate system at the end, after every choice was made on validation. Each decision
  in PLAN.md's log names the split it was made on.
- A fixed **new-user slice** (test users' first 10 transactions) and a **new-payee slice** (first filing of a payee by that user) are
  reported on their own (REPORT 132's scores).

Overlap between users is intended, not leakage (owner, 2026-09-30): the same payee appears across train, validation and test users,
and other users' filing trends for it are part of what the model learns (the other-users line, database episodes). What is held out is
users and their query transactions, never payees.

## 3. Prompts (as REPORT 99 / 101 / 113 / 121)

For each query transaction:
- the user's category list on the query date, labelled (rand255 labels drawn once per user);
- a history slice of at most 24 earlier rows, dated, in date order, nothing on or after the query's date and no category not yet
  created: the payee's own latest rows (up to 6), similar payees' rows (same type of business, at most 2 per payee, up to 6), the latest
  row of each category not yet shown, then the most recent rows; each row's category written with its label;
- for serving, the split layout (shared rows first, the payee's rows after) so one sync shares a cached prefix (REPORT 119, 120);
- the other-users line ("Other users file this payee as: ...") from **training users' filings dated before the query**, mapped to
  their category names, top three by count, when at least three such filings exist;
- statement string: import_string when present, else payee_name (report both groups).
- for a first-time payee, a canonical "Kind:" line under the query when payee_kind is known (REPORT 144; no retraining needed), else
  the 35B's reading (REPORT 139); person-to-person payees as "person-to-person payment (purpose varies)"; the memo, when present, on the row.

## 4. What we measure

On validation (for choices) and once on test:

- **Report card** (REPORT 130 / 131): right category first; in the suggestion list; not suggested (search); suggestions shown; work
  saved against filing by hand; by user group (new users, established users) and by payee group (first-time payee, the payee's usual
  category, filed differently than before, a category never used for the payee).
- **Rank score** (owner's scale): lists of at most three (owner, 2026-09-30); first 0, second 1, third 2, not shown 10; the clutter penalty only for implausible suggestions
  (plausible: categories the user used for the payee, or other users use for it).
- **Calibration** (REPORT 115, 116): reliability by bin, ECE, by user group; temperatures fitted on validation users only.
- **Anticipation** (REPORT 132): confidence on changed filings against usual ones (AUROC).
- **Speed and cost** on the serving stack (REPORT 120, 125).

Baselines on the same queries: YNAB's current suggestion (the logged `suggested_category_id`, and our re-implementation of the rule to
check it), the no-model list (the rule's suggestion, then the payee's other past categories), each model, and the recommended system.

## 5. Questions only real data answers

- How often users depart from a payee's usual category, and how often that is a slip (edited later) against a change of mind
  (REPORT 97): decides how much the models' anticipation is worth.
- The share of first-time payees and of new users in a real stream: decides how much the 35B and the models are worth over the
  no-model list (REPORT 131, 139).
- How many categories users have, and how varied other users' names for a payee are (REPORT 105, 137).
- W for any auto-filing feature: from how often users change a confirmed category and how long after (REPORT 118, 129).
- Whether training on real training users (instead of synthetic episodes) closes the gap to the ideal reader, and whether the 35B's
  readings of real payees are useful soft labels (REPORT 136).

## 6. Harness steps

1. Load and validate the export (schema, dates, no category used before its creation, per-user ordering).
2. Assign splits (hash of user_id) and freeze the query lists (validation, test) with their hashes, as `data/processed/` item sets
   (tracked by DVC, never in git).
3. Build prompts per §3 (the same builders as the synthetic sets: `oneslot.build_layout`, the slice policy of `build_real7_slices.py`),
   the other-users histograms from training users only, the per-item payee history and YNAB's rule.
4. Score the saved models on validation (Modal job list), then the report cards; choose; then test once.
5. Train on training users (the recipe, with real rows in place of synthetic episodes, or mixed) and repeat 4.
