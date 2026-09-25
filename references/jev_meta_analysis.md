# Jev-class decision models: meta-analysis of eleven open reconstructions, and next steps

Date: 2026-09-22. See also `jev_credibility_and_unknowns.md` (which claims hold up; what is known about Jev). Synthesis of the Jev series in `references/`: `semif_jev_analysis.md`, `laya_analysis.md`,
`jqv_analysis.md`, `diffusiongemma_djev_analysis.md` (djev-dev and razorback16/openjev), `reflex_analysis.md`,
`decider_analysis.md`, `openjev_zefan_analysis.md`, `nimble_analysis.md`, `kev_analysis.md`,
`openjev_verdict_analysis.md`, `jevbench_analysis.md`. External scoreboard: JevBench v1.3.0
(https://benchmarkheaven.com/jev-models, 534 decisions, scored 2026-09-21). Every repo is cloned under `~/projects/Taytay/`.

## 1. The field on one page

TypeSafe's Jev takes a state and typed questions (noul, choice, score) and returns a probability per allowed answer with no
generated text. Its architecture and its training method (RLCD) are undisclosed. Within a week, a dozen open projects
rebuilt the interface. They differ on three axes: the **backbone** (a frozen or tuned decoder LLM, an encoder, a diffusion
LM), the **readout** (letter logits at the last position, a trained pointer or scalar head, `[MASK]` markers, diffusion canvas
slots), and the **training** (none, a temperature, LoRA or full fine-tuning on some data, proper-scoring RL).

| System | Backbone | Readout | Trained? | JevBench rank / score | Intelligence / Calibration | Hard tier |
|---|---|---|---|---:|---|---:|
| Jev 1.13 | undisclosed (Hume: causal MoE, ~10B active) | undisclosed | RLCD, undisclosed | #1 / 74.4 | 86 / 83 | 0.741 |
| SemIf | Qwen3.5-4B instruct, frozen | letter logits | nothing | #2 / 73.1 | 79 / 73 | 0.595 |
| djev (Maisa) | DiffusionGemma 26B-A4B, frozen | one-step canvas slots | nothing | #3 / 73.0 | 83 / 65 | 0.695 |
| reflex 4B | Qwen3.5-4B | letter logits, two orders | board row: LoRA + per-primitive T; shipped: nothing | #5 / 70.3 | 80 / 75 | 0.632 |
| jqv | Qwen3-32B, frozen | letter logits, block mask | one temperature | #6 / 68.6 | 79 / 79 | 0.645 |
| decider-35b-a3b | Qwen3.5-35B-A3B base | letter slot | full FT (experts frozen) | #8 / 67.6 | 80 / 72 | 0.655 |
| OpenJev (razorback16) | DiffusionGemma NVFP4 | one-step canvas slots | nothing | #11 / 66.4 | 79 / 65 | 0.655 |
| kev 0.6B / 4B / 8B | Qwen3 base | pointer head | LoRA + head, no T served | #19 / #27 / #29 | 52-69 / 42-51 | 0.40-0.47 |
| decider-2b | Qwen3.5-2B base | letter slot | full FT + small RL | #23 / 61.7 | 61 / 47 | 0.473 |
| Bespoke Nimble 9B | Qwen3.5-9B | letter codes | LoRA on 2,676 minimal pairs | #24 / 60.5 | 78 / 65 | 0.655 |
| OpenJev thinking (BF16) | DiffusionGemma | thought canvases, then slots | nothing | #26 / 60.0 | 88 / 70 | **0.782** |
| Open-Jev 9B / 2B | Qwen3.5 | Yes-minus-No scalar head | LoRA on 79k synthetic rows | #30 / #34 | 71 / 63; 61 / 55 | 0.609 / 0.427 |
| Laya | ModernBERT-large 421M | `[MASK]` markers | full FT, "RLCD" | #33 / 54.4 | 46 / 62 | 0.341 |
| openJev Verdict | ModernBERT-base 151M | GLiClass slots | FT on Banking77 | #36-37 / ~38 | 39-40 / 51-74 | 0.38 |
| GPT-5.6 Luna (low reasoning) | frontier LLM | generated | n/a | #14 / 65.9 | 95 / 90 | 0.945 |

## 2. What we learned

### 2.1 The interface is solved; the intelligence is not

Every mechanism Hume observed in Jev is reproduced on stock open models with passing equivalence tests: generation-free
readout of an allowed answer set, one state prefill shared by many questions, isolated question branches (jqv: sibling
leakage 0.000 against a 0.996 negative control), options that interact before readout. The speed claim is also understood:
**sharing the state is the speed-up, and skipping generation is not** (jqv: generate ≈ naive at 0.9-1.1x; shared engines
53-73x at an 8k state with 100 questions). What no open system reproduces is Jev's accuracy on the hard tier without either
reasoning first or a 27B-plus backbone.

### 2.2 Backbone capability sets the ceiling, and it is the biggest lever

Frozen readouts scale cleanly: reflex's weight classes (hard 0.38 at 0.8B, 0.47 at 2B, 0.69 at 4B, 0.69 at 9B, 0.77 at
27B), jqv's (0.42 at 1.7B, 0.55 at 14B, 0.62 at 32B), and the model card of DiffusionGemma (a diffusion decoder loses 5-19
points to its autoregressive twin). No training recipe on a small model has closed a scale gap. Encoders at 150-420M sit at
Intelligence 39-46 regardless of how they were trained.

### 2.3 Training mostly buys in-distribution accuracy, and often costs transfer

The strongest and most consistent result across independent teams:

- jqv: LoRA plus a head on 4,800 MMLU items is +2 to +4 at 1.7B and nothing at 14B and 32B.
- reflex: five LoRA mixes, 27B distillation and self-distillation on a 4B; every one rose in distribution and fell on hard
  or held-out items; the frozen model ships.
- Open-Jev: 79k procedural rows take its own held-out set from 65% to 95% and leave the 9B below a frozen 4B on JevBench.
- decider: 455M tokens of full fine-tuning on a 2B base is below the frozen instruct 4B on hard items.
- Laya and openJev Verdict: encoders fine-tuned on one domain collapse off it (Laya base 0.36 on typed-decisions; Verdict
  1.0 below uniform there).

Two exceptions point at what does work. **kev**: LoRA plus a pointer head beats the frozen readout by 5-8 points on
classification- and policy-shaped transfer at 4-8B, when the training data is decision-shaped and the learning rate is
low (5e-5); it costs about 6 points of MMLU. **Nimble**: 2,676 counterfactual minimal pairs, one epoch, moved a 9B by 24
points on its own distribution and held within 1.2 points of Jev on 13 human-labelled public sets. What these share is data
in which the label depends on a specific fact in the state, so the only way to fit it is to read the fact.

### 2.4 Calibration: a temperature per task type does most of the work; architecture and RL do not

- Every raw letter readout is over-confident (jqv T=12 at 1.7B, 3.0 at 32B).
- One temperature reaches Jev's reported ECE in distribution (jqv 0.023 on MMLU at 32B) and transfers across languages, but
  not across task types: knowledge questions want T≈12 and questions answerable from the state T≈3 at 1.7B (jqv), and a
  per-primitive T fitted on one mix hurt every external set (reflex). kev found one global T does transfer within
  classification-shaped tasks. The rule consistent with all three: **fit one temperature per task type on held-out items of
  that type.**
- Serving no temperature is the entire explanation for kev's Calibration 42-51; trained models are still over-confident on
  the hard tier where their in-distribution T no longer applies (decider hard ECE 0.30 at 2B).
- "RLCD" as implemented in the open (Laya's Gaussian-noise REINFORCE on proper scores) is a noisy estimate of a gradient that
  backprop gives exactly. jqv, reflex and openJev Verdict independently use or test the supervised form (CE, CE+Brier, soft
  targets); CE + λ·Brier with λ ≤ 1 is indistinguishable from CE, λ = 2 is a built-in temperature that costs accuracy.
- What does move calibration beyond a temperature: **soft targets on ambiguous items, uniform targets on evidence-free
  items** (kev's unknowable share at ≥0.9 went 0.19 to 0.00), **one epoch rather than two** (kev, decider), and **reasoning**
  (djev thinking has Calibration 93, the best on the board). No loss change fixed selective ordering (kev's screen).
- Architecture does not: two DiffusionGemma deployments and SemIf's decoder all sit at 65-73 without a temperature.

### 2.5 Order and letter priors are real, cheap to fix, and matter most at small scale

jqv at 1.7B: rotating only letters changes 46% of argmaxes; rotating only option order changes 52%. Averaging two or four
orders is +2.7 to +3.1 points at 4B-14B for free (reflex, jqv), shrinking to +0.4 at 32B. Training with shuffled options
flattens the position prior (jqv slot head 0.26/0.26/0.25/0.24). Shared fixed few-shot examples in the state collapse the
letter prior (jqv: −13 points, B at 64%). Lettered yes/no beats bare yes/no tokens (reflex toxic-chat +5.6, ECE halved).

### 2.6 The hard tier is reasoning, and one-pass readouts cannot do reasoning

JevBench's hard families split cleanly. Adversarial, trap and routing items are saturated at 4B and up. Probability,
multi-hop, long policy and ambiguity are where open rebuilds trail Jev. **Temporal and numeric items defeat every one-pass
model, Jev included (8 of 30)**; only systems that reason first solve them (GPT-5.6 Luna 0.93, OpenJev thinking 0.47). The
only open systems above Jev on the hard tier (OpenJev thinking 0.782, djev thinking 0.777, reflex-27b 0.759) either
generate a thought first or use a 27B backbone, and pay 5-10x in cost.

### 2.7 Claims need three labels, and most READMEs omit at least one

Across the series the recurring comparison errors were: a specialist fine-tuned on a benchmark's train split next to a
zero-shot generalist (Laya, openJev Verdict, the typed-decisions table); self-run public items next to official held-out
numbers (reflex, decider, Open-Jev); training sources counted as transfer (decider's Bespoke suite, kev's "trained
sources"); a leaderboard row that is not the shipped configuration (reflex #5); a calibration number from a correctness
head, not the distribution (openJev Verdict). The projects that held up best (jqv, reflex, kev, Nimble, JevBench itself)
pre-registered, kept a base-model control, and published negative results.

## 3. What this means for our categoriser

Mapped onto this repo's own findings (`reports/REPORT.md` sections 35-44):

- **Our task is the easy regime, and that is good news.** A categoriser question is a choice over tens of categories with a
  short state; JevBench's corresponding families (intent, routing, extraction) are saturated at 4B. Our hard cases (opaque
  merchants, ambiguous records, DB-only merchants) correspond to JevBench's `ambiguous` and `probability` families, not to
  `temporal_numeric`, so they are addressable without reasoning.
- **Record in the prompt beats record in the weights** (REPORT 38/43: 98 vs 51-72 on DB-only merchants). Every Jev rebuild
  is a record-in-prompt system: the state carries the evidence and the model reads it. The field agrees with our result.
- **Our SFT gains need the transfer check the field learned to require.** reflex and Open-Jev are the precedent for
  in-distribution gains that reverse on held-out items. REAL-6/REAL-7's merchant- and user-held-out splits are the right
  guard; a fixed external gate set is the missing piece.
- **We have not measured calibration or order sensitivity on the categoriser.** Both are cheap (the per-option logits are in
  `results/per_item/`) and every project that measured them found something.

## 4. Next steps

Ordered by value per GPU hour. Items 1-4 need no training; 5-8 are experiments on the 3090; 9-10 are what would move the
field.

1. **Calibration audit of existing runs (CPU only).** From `results/per_item/`, fit one temperature per arm and per task type
   (record-in-prompt vs parametric vs k-shot history) on held-out items, report ECE, Brier and selective accuracy
   (coverage at 90% and 95% precision) with group bootstrap. Expect the record-in-prompt arm to be near-calibrated (jqv's
   reading-task T≈3) and the parametric arm far off. This turns the accuracy tables into an auto-apply-versus-review
   operating point, which REAL-11's correction-rate requirement asks for. Reuse `jqv/calibration.py` or `kev/metrics.py`.
2. **Letter readout and order sensitivity on REAL-6.** Score the untrained and trained categorisers with a letter readout
   (our `mcf_lp`) under K rotations; report the argmax flip rate and the gain from averaging two orders. If the category list
   exceeds 26, use two-letter single-token labels (decider, djev) or a pointer head (kev).
3. **Shared-prefix engine for the categoriser.** Invert SemIf/jqv's geometry: the long fixed part is the category list,
   field guide and fact-DB record; the short part is the transaction. Port `jqv/engine/packed.py` or kev's block-causal mask;
   verify equivalence to independent forwards in fp32 as jqv does. This makes the record-in-prompt arm cheap at volume.
4. **Freeze an external gate set before any more training.** Four 300-item sets never trained on (reflex's discipline):
   REAL-6 held-out merchants, REAL-6 held-out users, a public intent set (Banking77 test, relabelled to our schema), and the
   231 JevBench public items as a pure out-of-domain check. Report per-set, never pooled.
5. **Counterfactual minimal pairs from our own generator (the Nimble recipe, without the LLMs).** For each training
   transaction, change one field the category rule depends on (merchant string, amount band, memo keyword, product line),
   recompute the label with the generator's rule, keep pairs whose labels differ. Train the categoriser SFT on pairs versus an
   equal count of unpaired rows, same seed, one epoch. Measure on the gate sets. This is the most direct test of whether the
   model reads the field or the prior (the DATA-1 confound) and the one recipe in the series that transferred.
6. **Soft and abstain targets.** Uniform targets on evidence-free items (opaque merchant, no record); decider's 75/25 abstain
   augmentation (10% of questions get an "unknown" option, a quarter with the true categories swapped out so unknown is
   correct). Measure the ≥0.9-confidence error rate on opaque merchants, kev's "unknowable" metric.
7. **Distil the record-in-prompt teacher into the weights with soft targets and an anchor slice.** Our arm P already does
   token KL to the with-context teacher (REPORT 14); reflex found soft-target distillation with a 20% self-labelled anchor
   was the only training that did not collapse, and that 4k states is too few. Scale the state count, keep the anchor.
8. **Hyperparameter hygiene from kev and decider**: lr 5e-5 at 3-4B and 1e-4 below, never 2e-4; one epoch; option shuffle
   at build time; sub-sample large category lists to ten with gold kept during training.
9. **A backbone comparison at our scale that nobody has published**: instruct vs base starting point for the same
   letter-slot SFT (decider trains base models and never argues why; we train instruct). Qwen2.5-3B instruct vs base, same
   data, same gate sets.
10. **The open research question the whole field is circling**: can a small one-pass decision model be trained to Jev's
    hard-tier accuracy without reasoning at inference? The evidence says data shape matters more than objective or volume.
    A contribution would be a controlled comparison on one 4B backbone of (a) public classification mixes, (b) procedural
    synthetic rows, (c) counterfactual minimal pairs, (d) teacher-distilled soft targets, at matched token counts, each
    evaluated on JevBench's held-out hard tier via the maintainer's ranked-row process. Nobody has run that grid; reflex,
    Open-Jev and Nimble each ran one cell. Items 5 and 7 above are two cells of it on our own task.

## 5. What to reuse, by need

| Need | Take it from |
|---|---|
| Shared-prefix packed engine with equivalence and leakage tests | `jqv/jqv/engine/packed.py`, `shared.py`; `kev/kev/model.py` for Qwen3 base; `reflex/src/reflex/engine.py` for Qwen3.5 hybrids |
| Temperature fit with provenance | `jqv/jqv/calibration.py`; `kev/scripts/calibrate_checkpoint.py` |
| Calibration and selective metrics | `kev/kev/metrics.py` (ECE, Brier, coverage at error, AURC, paired bootstrap); `jevbench/jevbench/metrics.py` |
| Label-slot verification | `djev-dev/djev/engine.py: compile`; `SemIf/src/semif_phase1/direct.py` |
| Candidate-masked CE trainer | `nimble/nimble/training/schema_train.py`; `reflex/src/reflex/train/calibrate.py` |
| Pointer head | `kev/kev/model.py` |
| Minimal-pair and executable-rule generators | `kev/kev/contrastive.py`, `composition.py`; Nimble's `decide()` pattern |
| Abstain augmentation, large label sets | `decider/decider/data/`, `decider/decider/prompt.py` |
| External gold set and item schema | `jevbench/datasets/public/`, `jevbench/jevbench/tasks.py` |
| Research-log discipline | `kev/PLAN.md`, `jqv/tasks/`, `reflex/docs/results/` |
