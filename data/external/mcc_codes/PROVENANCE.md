# MCC codes (merchant category codes, ISO 18245 numbering)

- Source: https://github.com/greggles/mcc-codes, `mcc_codes.csv` (981 codes; fields mcc, edited_description, combined_description,
  usda_description, irs_description, irs_reportable), fetched 2026-10-03 (repository last pushed 2024-08-16).
- Licence: The Unlicense (public-domain dedication; `LICENSE.txt` here). The descriptions come from USDA and IRS publications (US
  government works, public domain). Owner's rule (2026-09-26) is open licences only; a public-domain dedication is at least as open as
  the CC0 sources already used (Overture / AllThePlaces).
- Use (PLAN step 173, owner 2026-10-03: "If you look at an mcc database, it will be pretty clear that there are many types of merchants"):
  a fine merchant-type layer under `ai_experiments.canon`'s 41 kinds; codes 3000-3999 are individual airlines, hotels and car-rental brands.
