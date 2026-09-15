# PLAN.md: the work queue

This is the only file that changes as experiments get done. Everything else in the repo is
either code, results, or reference material that the queue points into.

**"Do the next step"** means: take the first row in the queue whose Status is `todo` and whose
Needs are all `done`, and follow the procedure below. Do not skip ahead or bundle rows.

## Procedure for one step

1. Read the row's IDs in `reports/QUESTIONS.md` (definition, what the reviewer saw, the proposed
   experiment) and the matching subsection of `references/SURVEY.md` section 1 (what the literature
   says, with paper IDs you can open under `references/papers/<id>/summary.md`).
2. Set the row's Status to `doing` and commit that one-line change first, so a second agent
   cannot pick the same row.
3. Do the work. New scripts go in `scripts/`, outputs in `results/`, and every training or
   evaluation run goes through the tracker in `evals/` (see `evals/README.md`). Commit the code
   before a long run so the tracker records a clean commit hash. After a training run,
   `just push-models` (`dvc add models/adapters` then `dvc push`) so the adapter is stored and
   its hash is committed with the results.
4. Write the result into `reports/REPORT.md` as a new numbered subsection at the end, with the
   ID in the heading. Regenerate `evals/LEADERBOARD.md` if metrics changed.
5. Close the loop in three places: set the row's Status to `done` and fill Result with the
   commit hash or result file; add a `**Status (date):**` line under each ID in
   `reports/QUESTIONS.md`; append a dated line to the Log at the bottom of this file.
6. If the work changes what should come next, edit the queue (reorder, split, add rows) and say
   why in the Log. New rows need a new ID in `reports/QUESTIONS.md` first.
7. Commit.

A step is sized for one session on this machine. If it will not fit, split the row before
starting and note the split in the Log. Working rules for the machine are in `CLAUDE.md`.

## Queue

Order is by expected information per GPU-hour, from the 2026-09-14 literature survey. Tier 0 is
free or nearly free and must come before any new training run; the others cost roughly an
afternoon each on the 3090.

| # | Step | IDs | Needs | Status | Result |
|---|------|-----|-------|--------|--------|
| 1 | Persist per-item, per-option log-probs, argmax, token and byte counts from every eval; freeze `ladder`, `probes`, `heldout_induction` to versioned JSON in `data/processed/` and record the hash in the tracker config. Done when every existing adapter has been re-scored with `EVAL_ONLY` and the per-item files exist. | STAT-2, STAT-3 | - | done | REPORT.md 9; data/processed/*_v1*.json; results/per_item/ |
| 2 | Scorer set on the saved logits: sum, mean-per-token (current), PMI_DC, Bayesian length correction, hybrid symbol scoring on 3B, unconditional-option baseline (UNC), RStd, predicted-label histogram. Done when one table reports every arm under every scorer and says which cells move. | EVAL-2, EVAL-1, EVAL-6 | 1 | done | REPORT.md 10; reports/scorers_Qwen2.5-3B.md |
| 3 | Constructed-response eval: greedy-decode 32 tokens for every L1, L3, category and sells item; exact and fuzzy match; options-listed variant; three-way agreement with the two cloze scorers. | EVAL-3 | 1 | done | REPORT.md 12; reports/gen_Qwen2.5-3B.md |
| 4 | Empirical null band per level (shuffle answer index 1,000 times over saved per-option scores); bootstrap CIs and paired tests from the per-item files; CI column in every report table. | STAT-2, EVAL-6, REPORT-2 | 1 | done | REPORT.md 11; reports/ci_Qwen2.5-3B.md |
| 5 | Tokenizer casing check: re-run the LLM in-context and `bank_category` evals with title-cased bank strings; record the token-survival statistic for every tokenizer in use. | MODEL-4 | 1 | done | REPORT.md 13; results/casing.json |
| 6 | Report fixes that need no run: caveat the held-out-partition claim in the summary; rename "RAG ceiling" to "oracle context"; disclose the arm D schedule confound and token-weighted mixture fractions in the design table. | REPORT-1, REPORT-3, REPORT-4 | - | done | REPORT.md 1, 4.3.2, 6.2, 8.1, 8.3 |
| 7 | Soft-label prompt distillation on arm C's data: teacher is 3B with the field guide in context, student sees the same prompts without context, KL loss. Prediction: bare L1 recall toward 98.8 with better ICL retention than hard-label episodes. | BASE-3 | 2 | todo | |
| 8 | Masked fine-tuning of the decoder (Pan et al. 2510.09885) on Qwen2.5-3B-Instruct, LoRA r=64, one template per entity; score bare L1, `L1_recall_fmt`, merchant `reverse` and the ICL suite against arm C. | TRAIN-5, EVAL-7 | 2 | todo | |
| 9 | Mixture fix by construction: all-answer loss on k-shot episodes plus a self-teaching stream from the field-guide entries; log per-stream label-token counts per step; sweep E in {0.2, 0.4, 0.6}. | TRAIN-1 | 2 | todo | |
| 10 | Three seeds each for arms A, C, D on 3B; mean and sd; restate every headline gap against 2 sd. | STAT-1 | 2 | todo | |
| 11 | Periodic evaluation: 400-item subsample every 200 steps plus a held-out known-facts set as the forgetting proxy; curves of recall, Timmy, ICL-symbol and perplexity per arm. | TRAIN-3 | 2 | todo | |
| 12 | Arm D controls: restarted LR schedule per phase, constant LR, 10% knowledge replay in phase 2. | TRAIN-2 | 10 | todo | |
| 13 | Embedding block: rerun the universe and merchant embedding experiments on Qwen3-Embedding-0.6B and bge-base-en-v1.5; MOSAIC joint stage for new tokens; SetFit and logistic-regression baselines on the same 160 items the LLM arms use. | MODEL-3, MODEL-5, BASE-4, BASE-5 | 5 | todo | |
| 14 | C-RAFT: one gold plus two retrieved distractors from the fine-tuned MiniLM; evaluate no-context, oracle, and retrieved top-3 on arm C and the distilled adapter; recall@1/5 of bank string to record. | BASE-1, REAL-2 | 7 | todo | |
| 15 | Scale: base, A, C, D on Qwen2.5-7B (QLoRA) and Qwen2.5-1.5B; natural-label ICL and a knowledge benchmark reported separately. | MODEL-1 | 10 | todo | |
| 16 | Augmentation scaling: 1/3/7/14 templates, then sentence-order permutation, reverse-direction sentences, LLM-written relation texts; `clean_category` against distinct texts. | DATA-5 | 8 | todo | |
| 17 | Hyperparameter sweep on arm C: r in {16, 64, 256}, lr in {5e-5, 1e-4, 2e-4}, steps in {400, 1600}; one 3B full-FT run with 8-bit AdamW; MLP-only LoRA. | TRAIN-4 | 10 | todo | |
| 18 | Encoder arms: F (ModernBERT-large, answer-token prediction) and F2 (Flan-T5-large span prediction), paired with the 1k and 5k species universes. | MODEL-2, REAL-3 | 2 | todo | |
| 19 | Knowledge-editing arm: MEMIT and AlphaEdit via EasyEdit on Qwen2.5-3B, one target per species, single batch; full ladder plus ICL suite beside arms A and C. | BASE-2 | 2 | todo | |
| 20 | Item rebuilds: induction items with identifiable rules and an unseen-label variant; ICL suite with a new template and disjoint label source plus MMLU and WikiText slices. | EVAL-4, EVAL-5 | 1 | todo | |
| 21 | Data realism: noisy renderings in training text plus a string normalizer; overlapping product pools; unseen-prefix morphology probes with a morph_p sweep. | DATA-4, DATA-2, DATA-3 | 2 | todo | |
| 22 | Real-use replicate: realistic merchant set with Zipf frequencies and real-style truncations; 12-way imbalanced prototype eval with amount and weekday features. | REAL-1, REAL-4 | 13, 21 | todo | |
| 23 | Held-out partition without the weakness-equals-type confound: make weakness independent of type and re-run arm C. | DATA-1 | 2 | todo | |

## Log

- 2026-09-14: queue created from `references/SURVEY.md` section 2 (which now just points here) and
  the reviewer register in `reports/QUESTIONS.md`. Nothing started yet.
- 2026-09-14: step 1 done (REPORT.md section 9). Item sets frozen as v1 with hashes in the tracker config; every eval writes per-item, per-option log-probs under four premises (prompt, cue only, newline, listed choices) so steps 2 and 4 run without a model. All 11 adapters re-scored on WSL: 111 cells move against the Windows numbers, at most 2.6 points on 160-item levels (the same-weights noise floor). Merging the LoRA flips 1.8% of predictions and halves eval time; adopted for new evaluations, not for EVAL_ONLY re-scores of section 8 arms. Eval per arm is 24 min unmerged, 15 merged.
- 2026-09-14: step 6 done out of order, while step 1's re-score occupied the GPU (it needs no run and
  nothing depends on it). Executive summary and 8.3 carry the weakness-equals-type caveat; "RAG
  ceiling" / "RAG upper bound" are "oracle context" in the report and the code docstrings; 8.1 has a
  sequence / token / loss-bearing-token table (arm C: 84% knowledge text by loss-bearing tokens,
  episodes 12%) and the arm D schedule confound, pointing at steps 9 and 12.
- 2026-09-14: step 2 done (REPORT.md section 10). Elicitation matters more than normalisation: hybrid scoring lifts every induction level of arm C by 5 to 11 points and turns L5 from below chance into 33.8; letter scoring makes knowledge-only arm A the best inducer (62.5) while collapsing the trained arms' ICL suite; PMI is only right for bare-format recall. Several "chance" rows are constant predictors.
- 2026-09-14: step 4 done (REPORT.md section 11). Bootstrap CIs, permutation null bands and paired McNemar tests from the per-item files. Most of section 8 holds; arm D's pairwise loss, C's weakness edge over A and the natural-label half of the replay claim do not. 115 of 336 cells, including the base model's 42.7 on held-out induction (EVAL-6), are inside their gold-blind null band.
- 2026-09-15: step 3 done (REPORT.md section 12). Bare-format recall is about 85% by generation for every knowledge arm (cloze said 20 to 25: the first token after `Answer:` is the species name, not the type); induction by generation matches the cloze within 2 points for the episode arms and agrees with the mean rule on 87 to 95% of items; base and A cannot produce a label (91 to 93% unanswered), so their induction scores, including base's 48.8 with context, are cloze-only. Two section 6 adapters re-decode after the running jobs.
- 2026-09-14: step 5 done (REPORT.md section 13). The casing hypothesis is wrong for the LLM: restoring the merchant name's trained tokens inside the bank string moves 0.5B with-context bank_category from 14.2 to 15.0 (title-casing the whole line: 15.8, chance 8.3); the transaction format, not the tokenizer, is what the small model cannot read. Qwen keeps 10.6% of name tokens under uppercasing, MiniLM (uncased) 96.2% and is indifferent to the rendering (67.7 / 67.7 / 68.8). Step 21's normaliser should strip the format, not just the case.
