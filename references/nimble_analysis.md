# Bespoke Nimble: 2,676 counterfactual minimal pairs on Qwen3.5-9B, and the one training recipe that transferred

Date: 2026-09-22. Eighth memo in the Jev series. Object: https://github.com/bespokelabsai/nimble (HEAD `f136b3f`, about
1,500 stars, **no licence file in the repo**; the Hugging Face adapter is Apache-2.0; data licence unstated), clone at
`~/projects/Taytay/nimble`. Read by a subagent; this memo condenses its report. JevBench v1.3.0: "Bespoke Nimble 9B" #24 at
60.5 (Intelligence 78, Calibration 65, Speed 79, Cost 33; easy 1.000 / standard 0.948 / judge 0.890 / hard 0.655, the best
hard tier of any trained small decoder). Hard accuracy rose from 0.436 to 0.655 when the serving prompt limit was raised
from 2,048 to 8,192 tokens, because long hard items had been rejected and scored wrong.

## 0. The one-paragraph answer

Nimble is a rank-16 LoRA on Qwen3.5-9B that reads single-token letter codes at the last prompt position, one field per
forward, trained with cross-entropy restricted to the allowed codes on **2,676 rows: 1,338 base/counterfactual pairs whose
contexts differ by one sentence of at most eight changed words, flipping one audited "focus fact" and therefore the
label**. Labels are never asked of a teacher: an LLM decomposes a policy into atomic propositions and sufficient-condition
rules, a separate LLM call audits them, Python brute-forces the rules for conflicts, LLMs write the evidence sentences and
the 100-180-word document, reviewer calls verify the fact states of five variants (base, counterfactual, each sentence
removed), and Python's `decide()` computes the label from the rules. One epoch, 335 optimiser steps. On its own synthetic
holdout it scores 90.1% against 66.4% for the base 9B and 93.2% for Jev; on 13 human-labelled public subsets (3,880 records)
it trails Jev by 1.2 macro points (74.8 vs 76.0) and is worse calibrated on 11 of 13. No temperature is fitted; the README
warns labels are synthetic and unreviewed and probabilities uncalibrated. It was "built in one day." Against the rest of
the series it is the counterexample: a small, carefully constructed training set moved a 9B decoder about 24 points on its
own distribution and held near Jev-parity on public transfer, where reflex's and Open-Jev's much larger mixes did not.

## 1. How it works

- **Prompt** (`nimble/scoring/parallel_schema.py`): system "Classify the context using the supplied schema... Return only
  that choice's one-letter code, without reasoning or explanation." User turn is JSON `{"context": ..., "schema": [{name,
  description, choices: [{code: "A", value, description?}]}]}` plus `Requested field: "<name>"`. Booleans are `A=false,
  B=true`; score levels are strings `"0".."k-1"`. At most 26 choices, which is why Banking77 and CLINC150 were rejected as
  benchmarks. Each code must be one ordinary token at the answer boundary. Field names and option keys are in the prompt,
  so "question IDs can therefore affect predictions."
- **Readout**: candidate rows of the output head gathered, the last hidden state projected in FP32, softmax at T=1.
- **Serving**: an MLX scorer that prefills the shared prefix once, forks all 32 layer states including the linear-attention
  recurrent states, and batches field suffixes (7.25x over independent for 8 fields at 2k tokens on an M5 Pro); a CUDA
  scorer that re-runs the full prompt per field; and a Modal deployment that merges the LoRA with PEFT `safe_merge` and
  serves it on SGLang 0.5.19 through ekzhang's openjev-sglang with radix caching. Prompt limit 8,192, trained at 2,048;
  over-length prompts are rejected, never truncated. Latency: 201 ms (L40S) and 299 ms (H100) end to end for three
  questions; 106 ms per example local on H100 against Jev's API at 247 ms.
- **Confidence**: one minus normalised entropy, "not a calibrated probability of correctness."

## 2. Contrastive curation, step by step

"Contrastive" here means minimal-pair counterfactuals with programmatic labels, not InfoNCE and not retrieved hard
negatives (`nimble/datasets/scaled_evidence_stages.py`, `curate_paired_evidence.py`, `scaled_evidence.py`).

0. **Seeds**: 300 synthetic decisions from a deterministic diversity plan: 10 domains x 5 subtopics x 6 slots (2 choice, 2
   noul, 2 score), crossed with 10 evidence mechanisms (negation, temporal update, conflicting evidence, missing
   information, coreference, distractors, paraphrase), 3 formats (text, object, dialogue) and 3 difficulties. Written by
   GPT-5.6 Sol through Bespoke Curator. Jev's probabilities on the seeds were recorded but are not a training target.
1. **Rule plan**: `PlanGenerator` decomposes the seed's policy into 2-20 atomic propositions, one focus atom, 2-8 unordered
   sufficient-condition rules with targets, and base and counterfactual fact states differing only in the focus.
   `PlanReviewer` audits soundness in a separate call. `compile_rules` enumerates all 3^n assignments (true / false /
   unknown) to reject conflicting rules; "unknown is never implicitly false."
2. **Pair**: `PairGenerator` writes two sentences that jointly entail the focus while each alone leaves it unknown, and a
   negative for one of them; the edit is mechanically checked at eight changed words or fewer. `FactReviewer` checks five
   isolated cases (each sentence alone unknown, the pair supported, the negative alone unknown, the negative pair refuted).
3. **Document**: `DocumentGenerator` writes a 100-180-word context embedding both sentences and the policy spans verbatim.
   Python applies the edit to make the counterfactual and the two deletion variants. `ContextReviewer` returns five
   booleans (policy preserved, bindings preserved, evidence is two factual sentences, counterfactual coherent, no answer
   leakage), and `FactReviewer` on the full contexts must reproduce the planned fact vectors exactly, which catches
   non-local edits.
4. **Label**: `decide()` applies the audited rules to the verified facts; a pair is kept only if both labels exist and
   differ. Deletion variants are checks, not rows. Every LLM request is cached content-addressed so a curation run can be
   replayed offline.

Data: `data/train.jsonl` 2,676 rows (900 via GPT-5.6 Sol, 1,000 via GPT-5.6 Luna, 1,000 via Claude Sonnet 5, minus 224
whose families overlap the holdout), 856 choice / 888 noul / 932 score, 194-300 per domain; holdout 324 rows from only six
families; pairs and families never straddle splits. Weaknesses the authors state: generator and verifier are the same
model; "no person has reviewed" labels; acceptance is poor (the attempt multiplier went from 32 to 64 and Sonnet "stalled at
974 accepted rows"), at about twelve LLM calls per accepted pair.

## 3. Training

`nimble/training/schema_train.py`: last-position logits, gather the candidate ids, mask the rest to minus infinity,
cross-entropy against the gold index. No LM loss, no soft targets. LoRA r=16, alpha 32, dropout 0.05, on every Linear in the
language model including Qwen3.5's linear-attention projections; lr 5e-5, effective batch 8, warmup 101, stopped after one
epoch at 335 steps, seed 17, bf16, gradient checkpointing. Selection by NLL on a 254-row inner validation split. Options
shuffled per row with a seeded RNG in training, original order in validation; position bias "unmeasured." No calibration.

## 4. Evaluation and flags

- 324-row synthetic holdout: Gemma-3 270M 28.7%, Qwen3.5-0.8B 45.4%, 4B 61.4%, 9B 66.4%, Qwen3.8-27B 84.9%, Nimble 90.1%
  (choice 84.9, noul 98.2, score 87.5; 130 of 162 pairs both right), Jev 93.2%. Flags: six families; labels from the same
  model families that wrote the data; untuned models scored in FP32 and Nimble in BF16; Nimble and Jev numbers reused from
  earlier runs.
- 13 human-labelled public subsets (`docs/PUBLIC_BENCHMARKS.md`, label-blind family-whole sampling, Wilson CIs, ECE,
  McNemar): Nimble 74.8% vs Jev 76.0% macro; choice 81.6 vs 82.9, noul 80.2 vs 84.6, score 54.6 vs 50.1. Significant gaps:
  civil_comments 70.3 vs 81.0, PAWS 82.8 vs 89.2, MASSIVE de-DE 83.4 vs 86.9, VitaminC 76.6 vs 80.1 for Jev; SummEval
  relevance 49.2 vs 35.0 for Nimble. These are transfer numbers; most datasets predate Qwen3.5's pretraining. Note that
  decider's card counts 7 of these 13 subsets in its own training mixture, so its comparison on this suite is not transfer.

## 5. What Nimble settles, and what it means for the plan

1. **The data shape, not the objective, is what made training transfer.** Nimble's objective is the same candidate-masked
   cross-entropy as reflex, decider and our categoriser SFT. What differs is that every row carries a sibling that differs
   in one fact and has the other label, so the only way to fit both is to read the fact. reflex's public-classification
   mixes and Open-Jev's procedural games do not have that property; jqv's MMLU items do not either.
2. **Small is enough when every row is certified.** 2,676 rows, one epoch, and the result is near Jev on transfer. This
   directly contradicts the "more rows" instinct and matches reflex's closing remark that its distillations failed for want
   of hard-shaped states, not volume.
3. **The recipe maps onto our synthetic universe almost one to one.** Our merchant generator already knows each
   transaction's latent facts. A minimal-pair generator is: take a transaction, change one field that the category policy
   depends on (merchant string, amount band, memo keyword, product line), recompute the category with the generator's own
   rule, keep the pair if the label flips. No LLM verifier is needed because our labels are already programmatic; the
   two-sentence entailment machinery is for free-text policies we do not have. This is the single most actionable idea in
   the series for REAL-5/REAL-6, and a direct test of the DATA-1 confound question (does the model read the field or the
   prior).
4. **Ship a temperature.** Nimble at T=1 loses to Jev on calibration 11 of 13; every project that fitted one gained.
5. **Beware the 26-option cap** of letter codes; our category counts sit near it. decider's two-letter single-token labels
   (to 255) or kev's pointer head are the ways past it.
6. **3090 fit**: 9B bf16 weights are about 18 GB and the trainer has no quantisation path; use the 4B (the scorer's own
   default model) or add QLoRA.

## 6. File map

| Where | What |
|---|---|
| `README.md` | Method, the 324-row table, latency table, recipe |
| `nimble/scoring/parallel_schema.py` | Exact prompt, code tokens, prefix split |
| `nimble/training/schema_train.py`, `schema_data.py` | Candidate-masked CE trainer, LoRA config, option shuffle, family separation |
| `nimble/datasets/scaled_evidence_stages.py`, `curate_paired_evidence.py`, `scaled_evidence.py` | The generator and reviewer prompts, acceptance gates, `decide()` |
| `docs/TRAINING_EVAL_CURATION.md` | Curation runs, edit limit, acceptance policies, model roles |
| `docs/PUBLIC_BENCHMARKS.md` | 13-subset human-label suite and results |
| `nimble/serving/server.py`, `docs/MODAL_SERVING.md` | SGLang flags, the 8,192 vs 2,048 limit, confidence definition |
| `data/manifest.json` | Lineage by writer model, family counts |
