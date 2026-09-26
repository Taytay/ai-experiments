# reflex: twenty training runs on Qwen3.5-4B, all rejected, and what ships instead

Date: 2026-09-22. Fifth memo in the Jev series. Object of study: https://github.com/kshetrajna12/reflex (MIT, HEAD
`e21b3b2`, clone at `~/projects/Taytay/reflex`), read by a subagent whose report this memo condenses; file paths are
relative to the clone. JevBench v1.3.0 rows: "reflex 4B" #5 at 70.3 (Intelligence 80, Calibration 75, Speed 68, Cost 60;
hard 0.632) and "reflex-27b" #17 at 63.3 (I 86, C 86; hard 0.759, above Jev's 0.741; Cost 32).

## 0. The one-paragraph answer

reflex is the project that tried hardest to make training work on a Qwen3.5-4B decision readout and documented why it
does not. It has a complete LoRA trainer with soft labels and a proper-scoring loss, a distillation pipeline from a 27B
teacher, a per-primitive calibration fitter, four synthetic correct-by-construction data generators aimed at the JevBench
hard families, and eight public classification datasets mapped onto the noul/choice/score primitives. The author ran about
twenty experiments across five adapter mixes, 27B distillation, self-distillation, prompt search and prior debiasing.
Every trained artefact improved in distribution and lost on the hard tier or on held-out domains, so the shipped
configuration (`serving/stable.json`) is the **frozen** model, temperature 1, no adapter, with each question read in two
distinct option orders and averaged. The README's verdict is "fine-tuning is a trap." The #5 JevBench row is, by the
author's own disclosure, a superseded adapter run, not the shipped frozen configuration, and a re-run has been requested.

## 1. What ships

- **Prompt** (`src/reflex/prompt.py`): ChatML, system "You are a System One decision model ... answer with the single
  option label only", user turn `# Evidence\n<state>` as the shared prefix, then per question `# Criterion\n<instructions>
  \n# Options\nA. ...\nB. ...\nRespond with only the letter of the best option.`, closed by the Qwen no-think assistant
  prefix. Yes/no is rendered as a lettered pair (`A. yes: ...`, `B. no: ...`), which removed a "say yes" prior on
  toxic-chat (0.787 to 0.843, ECE 0.104 to 0.046). Choice cardinality is capped at 26 single-token letters.
- **Readout** (`readout.py`, `engine.py: restrict`): last-position logits restricted to the branch's letter ids, divided by
  a temperature (1.0 in stable), softmaxed. Confidence is one minus normalised entropy; score is the probability-weighted
  level. `merge_branches` maps each order's probabilities back to semantic keys and averages over `permutations: 2`;
  `distinct_orders` guarantees the second order really differs.
- **State sharing** (`engine.py`): the prefix is run once with `logits_to_keep=1` and its cache kept in an LRU keyed by the
  SHA-256 of the prefix text. Two branch strategies: `packed` (attention-only Qwen3, one sequence under a 4D block mask,
  positions restart at the state length) and `batched` (hybrid Qwen3.5 with gated-delta-rule layers, where a mask cannot
  isolate a recurrent scan, so the state cache is batch-expanded and branches right-padded). A GPU test asserts equivalence
  to per-question forwards. That is the same construction as jqv's `packed` engine, plus the hybrid-architecture case jqv
  did not have to solve.
- **Serving**: FastAPI, about 9 GB bf16, about 200 ms warm on a GB10 (123 ms with a cached state), optional SGLang backend
  reading the same label log-probs (median probability delta 0.004), a WebGPU port on Qwen3.5-0.8B. No reasoning or
  escalation path.

## 2. What was trained, and what happened to each attempt

### 2.1 Data (`src/reflex/train/recipes.py`, `docs/DATA_SOURCES.md`)

JSONL rows `{state, questions, labels, source}` with optional soft labels. Recipes map public sets onto the primitives:
banking77 as 12-way with 11 sampled distractors, clinc_oos plus "none of these", MMLU-Pro 10-way, civil_comments with soft
toxicity fractions, HaluEval, MS MARCO, HelpSteer2 as score 0-4 with a per-rater soft variant, github-codereview; later
go_emotions_soft, QuALITY, xlam tool routing, and four synthetic generators (`adequacy_synth`, `rule_routing_synth`,
`base_rates_synth` with exact-frequency soft labels, `policy_synth`) that "target the failure shapes seen on JevBench's
public items without touching any benchmark item." Mix sizes 6,400 to 8,413 training rows (12.9k branches with permutation
augmentation in mix4). An 8-gram overlap check against benchmark items refuses contaminated rows (0 in every run). Four
never-trained external gates of 300 items each: Bitext support intents, MNLI-mismatched, toxic-chat, Yelp stars.

### 2.2 Objective and hyperparameters (`src/reflex/train/calibrate.py`)

"RLCD-lite": because the output *is* the distribution, RL with a proper-scoring reward "collapses to supervised NLL" or
Brier on the label-restricted logits, computed through the same batched forward the server uses. PEFT LoRA r=16, alpha 32,
q/k/v/o (optionally MLP), fp32 adapter over bf16 base, AdamW 1e-4 (5e-5 for distillation), warmup 10, linear decay, clip
1.0, one epoch (two for mix2), pack length 2,048-4,096, 3.1M trainable parameters. 43 minutes to 3.5 hours on one GB10; mix2
peaked at 18 GB with gradient checkpointing.

### 2.3 Calibration file (`fit_calibration`, `calibration_head.py`)

One temperature per primitive fitted by golden-section NLL on held-out logits (mix3 shipped noul 1.63, choice 1.54, score
1.89), plus an optional 8-weight linear head predicting log T from primitive, log option count, log state tokens,
normalised entropy and top-two margin. Fitting the same for the *frozen* model gave noul 3.19, choice 1.44, score 2.48
and hurt on every external set: "a temperature is a property of a distribution, not of a model." So stable ships T = 1.

### 2.4 Results of each attempt (public 231 JevBench items as the dev suite, hard tier accuracy)

| Run | In-distribution | Public hard | Verdict |
|---|---:|---:|---|
| Frozen 4B, one order | | 0.658 (ECE 0.086) | baseline |
| Frozen 4B, two orders averaged | | **0.685** (ECE 0.081) | **shipped** |
| mix1 LoRA | 62.7 to 76.8%, ECE 0.120 to 0.024 | 0.595 | rejected |
| mix2 LoRA, 2 epochs | 82.7% | 0.541 | rejected, over-confident on neighbours |
| mix3 LoRA + calibration | | 0.604 | the configuration behind the #5 board row |
| mix4 LoRA, permutation augmentation | | 0.595, toxic-chat 0.533 (says toxic on 290/300) | rejected |
| 27B-distilled student | teacher agreement 0.655 to 0.849 | 0.613 | rejected |
| Self-distillation | agreement 0.83 to 0.94; best 4B calibration axis 80.0 | 0.640 | rejected |
| GEPA prompt search | +2 on its mix | -2 to -5 elsewhere | rejected |
| PriDe-style position debiasing | | prior within 0.5 pt of uniform at 12 options | not needed; "the second order is mostly buying an ensemble" |

Weight classes, frozen, two orders: 0.8B hard 0.378, 2B 0.468, 4B 0.685, 9B 0.694 with worse ECE (0.136), 27B 0.766
(ECE 0.061), above Jev's 0.730 on the same public items. MMLU 1,200 items: Qwen3.5-4B 72.0%, ECE 0.090 to 0.039 at T about
1.7; Qwen3-8B 70.8%, ECE 0.264 to 0.061 at T about 9.7. The two runs that "moved without collapse" were both distillations
with soft targets and a 20% self-labelled anchor slice, on about 4k states; the author names ten times that corpus, with
hard-tier-shaped states, as the next thing to try.

## 3. Disclosures and flags, the author's and ours

- The official #5 row ran the superseded mix3 adapter at one order (commit `1add693`), not stable; the author says so in
  `docs/results/jevbench-official.md` and asked for a re-run.
- The public 231 items were used as a development gate nine times per the amendment filing (the board's note says four,
  from the first filing). The synthetic recipes were designed after inspecting which public families failed. The author
  calls the public tiers a dev set throughout.
- The README compares self-run public-item numbers (1.000 / 0.917 / 0.685) with Jev's official numbers (1.000 / 0.986 /
  0.730) in one table; different item sets and different measurement.
- The 27B "beats Jev on hard" (0.766 vs 0.730 public; 0.759 vs 0.741 official) with double Jev's ECE and a cost score of 32.
- `order-averaging.md`'s claim that two orders halve pooled external ECE (0.055 to 0.032) did not reproduce: the single-order
  figure re-measured at 0.028 in `position-prior.md`, and README and SERVING.md still repeat the stale claim.
- A same-structure "compact JSON" prompt in reflex's wording scored 0.847 standard against SemIf's 0.986: "wording is worth
  ±8 points." SemIf's standard tier still beats reflex's (0.986 vs 0.917) on the same frozen backbone.

## 4. What reflex settles

1. **On a 4B decoder, LoRA on public classification mixes buys in-distribution accuracy and pays for it on hard and
   held-out items, every time.** Five mixes, soft labels, permutation augmentation, MLP LoRA, two epochs: same shape of
   result. This is jqv's "LoRA adds nothing above 14B" pushed down to 4B with a worse sign: it subtracts.
2. **Distillation from a 27B through the same prompt is the only training that did not collapse**, and it still did not beat
   the frozen two-order average on hard. Soft targets and an anchor slice were the ingredients that kept it from
   collapsing.
3. **A fitted temperature is a property of the distribution it was fitted on.** Per-primitive temperatures for the frozen
   model hurt on every external set. That is jqv's cross-task-type finding stated as a rule, and the reason reflex ships
   T = 1 while jqv ships T = 3.0 fitted on MMLU: they serve different populations.
4. **Two distinct option orders averaged is the cheapest reliable gain**: +2.7 hard points at 4B, +6.3 at 27B, with no
   training and 2x branch cost on a shared prefix. The position prior at 12 options is nearly uniform, so the gain is an
   ensemble effect, not debiasing.
5. **Frozen sub-4B readouts are near chance on hard items** (0.8B 0.378, 2B 0.468). Below 4B either the encoder route or
   task-specific training is the only thing that works, which is Laya's regime.

## 5. What this means for the plan

- **Our categoriser gains from SFT (REPORT 38) are in-distribution gains.** reflex is the clearest evidence that those come
  with a bill on neighbouring tasks and on out-of-distribution items. REAL-6/REAL-7's merchant- and user-held-out splits
  are the right guard; the four never-trained external gates (300 items each, fixed before any benchmark look) are the
  pattern to add.
- **Soft targets plus an anchor slice is the recipe that did not collapse.** If we distil the record-in-prompt arm into the
  weights (the direction of section 4 of the SemIf memo), distil the *distribution* from the with-record teacher, with
  20% self-labelled anchors, not argmax labels. That is what our arm P (token KL to the with-context teacher, REPORT 14)
  already does; reflex says it is the right family and that 4k states is too few.
- **Never ship a temperature fitted on another population.** Our per-arm calibration must be fitted on held-out items of
  the same arm and cross-checked on a foreign set; a temperature that helps MMLU-type items hurts reading-type ones.
- **Two-order averaging is worth measuring on the categoriser at once.** It needs no training and our shared-prefix
  geometry keeps the extra cost to the branch tokens.
- **Lettered yes/no beats bare yes/no tokens.** Any binary readout in our evals should use `A. yes / B. no` letters, not the
  `yes`/`no` tokens, which carry an affirmation prior.
- **Fit on a 3090:** 4B bf16 LoRA r=16 fit in 18 GB at 2,048-token packs with gradient checkpointing; 4,096 tokens without
  checkpointing needed 34 GB. Keep checkpointing on.
- **Reusable code:** the JSONL schema with soft `target_vector`, `scoring_loss` (NLL or Brier on label-restricted logits
  through the server's own forward), seeded `distinct_orders` for both training augmentation and inference, the
  golden-section temperature fitter and its 8-feature head, `fidelity` (1 minus total variation) as the metric for
  soft-label sources, the 8-gram overlap guard, and the 75-dataset catalogue in `docs/DATA_SOURCES.md` as a template for
  mapping public sets onto primitives.

## 6. File map

| Where | What |
|---|---|
| `README.md` | Overview, stable numbers, the "fine-tuning is a trap" verdict |
| `serving/stable.json` | Shipped config: frozen 4B, no adapter, no calibration, permutations 2, gate numbers |
| `src/reflex/prompt.py` | ChatML prefix and branch rendering, `distinct_orders`, lettered yes/no |
| `src/reflex/engine.py` | State cache, packed and batched branch isolation, `restrict` letter readout |
| `src/reflex/readout.py`, `calibration_head.py` | Temperature and head, `merge_branches`, confidence |
| `src/reflex/train/calibrate.py` | LoRA trainer, `scoring_loss`, PEFT config, `fit_calibration` |
| `src/reflex/train/recipes.py`, `docs/DATA_SOURCES.md` | Dataset-to-primitive mappings, synthetic generators, overlap check |
| `src/reflex/distill/`, `docs/DISTILLATION.md` | Corpus, question writers, 27B teacher labelling, anchor mix |
| `docs/results/README.md` | Chronological index of every experiment with verdicts; key entries `frozen-vs-trained.md`, `order-averaging.md`, `weight-classes.md`, `lora-mix4-qwen3.5-4b.md`, `self-distill-orders.md`, `position-prior.md` |
| `docs/results/jevbench-official.md`, `jevbench-amendment.md` | What was actually run for the board rows, and the disclosures |
