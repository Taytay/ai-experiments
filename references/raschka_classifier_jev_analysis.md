# Raschka, "Language Models for Text Classification: From Bag-of-Words to Jev": what it says and what it means here

Date: 2026-10-06. Object: Sebastian Raschka's Ahead of AI post of 2026-09-29
(https://magazine.sebastianraschka.com/p/classifier-history-and-jev; digest with all numbers and transcribed figure tables in
`references/blog/other/2026-09-29-language-models-for-text-classification-from-bag-of-words-to-jev.md`). Context for the Jev
series: `jev_meta_analysis.md` (eleven open reconstructions), `decider_analysis.md`, `laya_analysis.md`,
`openjev_verdict_analysis.md`, `software/clef/README.md`.

## 0. The one-paragraph answer

A clear tutorial-style history of text classifiers (bag-of-words, RNN, CNN, BERT, GPT, T5) ending in a hands-on look at Jev.
It adds three pieces of first-hand evidence: Jev scores 96.47% on the IMDb test set zero-shot for $0.65 (tied with a
fully fine-tuned ModernBERT-large at 96.50%), a repeat run moves by 4 of 25,000 items, and on ModernBERT over nine datasets
**CE + Brier barely changes calibration (ECE 15.20% to 14.94%) while one post-hoc temperature cuts it to 5.63%**. Everything
about how Jev works (a small ModernBERT-like model, RLCD related to RLCR) is labelled by the author himself as a guess. Its
recipe for a Jev-like Choice API (one pass per option, a shared one-node scalar head, softmax over options, joint CE) is
the standard cross-encoder design, and it is already arm (c) of row 228. For this repo the post confirms two things we
already believe (temperature does the calibration work; data matters more than tuning) and suggests two cheap checks we
have not run: a `BRIER=0` arm for li_decider, and a classic lexical per-user classifier as a floor on the owner's budget.
Nothing in it argues for a new model, readout or RL objective.

## 1. What the post claims, and with what evidence

| Claim | Evidence in the post | Label |
|---|---|---|
| Jev gets 96.47% (Choice) / 96.20% (Noul) on IMDb test, 25,000 reviews, ~22-23 min, $0.65 | Author's own run, token counts and cost reported | Evidence, one dataset, own run. IMDb contamination unknown (he says so) |
| Jev is close to deterministic: repeat Choice run 24,113 vs 24,117 correct | Author's own repeat | Evidence (n = 2 runs) |
| Fine-tuned ModernBERT matches Jev on IMDb: 93.91% (149M, 256 tok) to 96.50% (395M, full length, 3 h train, 56 min eval on a DGX Spark) | Author's own runs (Figure 30) | Evidence, but a specialist trained on IMDb's train split against a zero-shot generalist; it shows parity, not superiority either way |
| A fine-tuned 66M-395M encoder or 124M GPT-2 reaches 88.9-95.1% on IMDb; BoW + LR 88.9-89.9%; LSTM from scratch 85.66%; text CNN 90.07%; ULMFiT 95.4% | Author's notebooks and the ULMFiT paper | Evidence, one easy binary task |
| CE + Brier gives "only a very small additional calibration benefit"; temperature scaling gives most of it | ModernBERT, mean of nine (unnamed) datasets: ECE 15.20 / 5.63 / 14.94 / 5.48% for CE / CE+T / CE+Brier / CE+Brier+T; accuracy 73.20 vs 73.27; one dataset got slightly worse ECE | Evidence, but single run, datasets not named, no seeds or error bars |
| RLCR cuts ECE vs RLVR (HotpotQA 0.37 to 0.03 at 62.1 vs 63.0% accuracy; six other sets 0.46 to 0.21, accuracy 53.9 to 56.2) | Quoted from arXiv 2507.16806, Table 1(a) | Evidence from the paper, not re-checked by the author |
| Jev beats GLiNER2.5 on AG News (0.910 vs 0.700) and Banking77/BTZSC (0.870 vs 0.610); on DAIR Emotion Jev is barely more accurate (0.480 vs 0.440) and far worse calibrated (ECE 0.351 vs 0.117, NLL 5.59 vs 1.38) | Third-party pilot (AbdelStark/jev-benchmarks) | Evidence, third party, small pilot. The DAIR Emotion row (Jev badly over-confident) is not discussed in the post |
| Two other open Jev-likes do worse on IMDb: Contrastive Language Models 82.90%, Laya 92.33%; both fail his Tetris demo | Author's runs | Evidence, single runs; "Tetris test" is a demo, not a metric |
| Jev is on GPT-5.6 Luna's level at much lower cost (Figure 24: Jev ~68% at ~$0.0004 per workflow, Luna workflow ~67% at ~$0.003) | TypeSafe's own chart | Vendor claim |
| Jev is "something small similar to ModernBERT, hence the low latency" | None | Opinion, labelled a guess. Note: `jev_meta_analysis.md` records Hume's reading of Jev as a causal MoE with ~10B active parameters, which contradicts it |
| RLCD may be related to RLCR | None ("no officially established connection") | Speculation |
| Most effort went into curating the (100% synthetic) data; more data beats more tuning | CEO quote on X; an 8-year-old anecdote (~300 to ~600 labels gave ">10-20%", tuning gave ~2-5%) | Opinion plus anecdote |
| Putting a Jev API on ModernBERT is trivial; making it general is not; clones do not match Jev's breadth | Author built and withheld a clone; no numbers | Opinion |
| Transformers beat RNN/CNN for this because they scale with pre-training and use context | Argument | Opinion (standard) |

## 2. The history, and where Jev-style decision models sit

The post's arc is about two axes:

- **Representation**: counts (bag-of-words, TF-IDF) to learned word vectors (Word2Vec, GloVe) to contextual states
  (RNN/LSTM hidden state, CNN windows, then attention). Word order and context enter at each step; pre-training (ULMFiT,
  then BERT/GPT) is the big jump (LSTM 85.66% from scratch vs ULMFiT 95.4% on IMDb).
- **Readout**: a fixed per-class output layer (logistic regression, `[CLS]` head, last-token head on a GPT) versus
  "text-to-text" (the model writes the label word: prompted GPT, fine-tuned T5).

Jev-style decision models are placed as a third readout: **a shared scorer over a variable option set described in text**,
returning a calibrated distribution with no generation. That gives general-purpose zero-shot use (like text-to-text) at
classifier speed and with probabilities (like a head). Raschka's point is that none of the parts is new; the product is the
breadth of training data and the calibration. That agrees with `jev_meta_analysis.md` §2.1 ("the interface is solved; the
intelligence is not").

## 3. Methods, readouts and training described

- **Classification head on a backbone** (§2.1-2.2): `[CLS]` for encoders, last non-padding token for causal decoders,
  because of the causal mask. Same as decider's slot position, except decider reuses the LM-head rows of the letter
  tokens instead of a new head.
- **Text-to-text** (§2.3): the decoder emits the label. Our letter-slot readout is the constrained form of this (only the
  option letters' logits at one position, no sampling).
- **Jev-like Choice API on any backbone** (§4, Figures 31-32): for each option, one forward pass over (input, instruction,
  option description); a shared head `s_i = w^T h_i + b`; softmax over the N scores; train backbone and head jointly with CE.
  Head size is independent of N. This is a **cross-encoder with N passes per item**. It is not what Clef does (Clef reads
  the state once and scores all options jointly with a small transformer "joint schema head" over the backbone's hidden
  states; `software/clef/README.md`), and not what li_decider does (query and option documents encoded separately, met at
  MaxSim).
- **Noul / Choice / Score** (Figure 28): yes/no probability; softmax over candidates; softmax over rubric levels and the
  expected level. Matches the SystemOne wire format already implemented by decider and Clef.
- **Calibration** (§5.1, 5.3): temperature scaling fitted on held-out data with weights frozen (argmax unchanged); CE and
  Brier are both strictly proper, so at the optimum both are calibrated; finite-data training overfits NLL before 0/1 loss
  (Guo et al. 2017), so held-out calibration must be checked.
- **RLCR** (§5.2): reward `c - (q - c)^2` on a generated confidence; the author's supervised analogue is CE + Brier on the
  classifier's probabilities.
- **Data over tuning** (§5): learning curves before hyperparameter search.
- **Encoders named**: ModernBERT-base 149M and -large 395M (his default), BERT, DistilBERT, RoBERTa, DeBERTa-v3, GPT-2 124M,
  Qwen3 0.6B, T5. No late-interaction, retrieval-style or bi-encoder classifier appears anywhere in the post.

## 4. Benchmarks in the post

- **IMDb (binary sentiment, 25,000 test)**: saturated at ~95-96.5% for everything modern (Jev, ModernBERT-large, ULMFiT). It
  separates nothing relevant to a 20-60-way per-user categoriser.
- **ModernBERT CE vs CE+Brier vs temperature, nine datasets**: the one table with method-level information (§1 above).
- **BTZSC pilot (AG News, Banking77, DAIR Emotion)**: from AbdelStark/jev-benchmarks; already cited in `laya_analysis.md`
  (AG News 0.910, Banking77 0.870 for Jev). Its "coverage at ≤5% empirical error" column is the same idea as our scorecard's
  auto-file coverage.
- **TypeSafe's four-workflow accuracy-vs-cost chart**: vendor numbers.
- **Clef's Decision Index** is not in the post (see `software/clef/README.md`): there Clef-flash
  has Banking77 macro-F1 90.9 vs Jev 79.7 on Cloudflare's harness, while the BTZSC pilot gives Jev 0.857 macro-F1 on
  Banking77. The two harnesses disagree by ~6 points on Jev, a reminder that cross-harness numbers do not compare.

## 5. Coverage check: is what it cites already in `references/`?

| Cited work | In the repo? |
|---|---|
| Jev / TypeSafe, SystemOne API | Yes: the whole Jev series, `jev_meta_analysis.md`, `jev_credibility_and_unknowns.md` |
| ModernBERT (arXiv 2412.13663) | Yes: `references/papers/2412.13663/`, plus ~30 mentions (Laya, openJev Verdict, SURVEY) |
| GLiNER / GLiClass | Yes: `fewshot_scan_2026-09-21.md`, `openjev_verdict_analysis.md`, REPORT (row 69 GLiNER-type work) |
| Laya | Yes: `laya_analysis.md` |
| AbdelStark/jev-benchmarks (BTZSC pilot) | Yes, cited in `laya_analysis.md`; BTZSC (arXiv 2603.11991) in `fewshot_scan_2026-09-21.md` |
| OpenAI Decision(s) API | Only a passing mention in `jqv_analysis.md` (as a generic phrase); the DevDay 2026 announcement is not recorded |
| RLCR, *Beyond Binary Rewards* (arXiv 2507.16806) | **No** |
| Guo et al. 2017, *On Calibration of Modern Neural Networks* | **No** (only an unrelated "Guo et al" in `papers/2309.14402`) |
| Gneiting and Raftery 2007, strictly proper scoring rules | **No** |
| Contrastive Language Models (contrastive-lm.notion.site) | **No** |
| arXiv 2609.30216 (survey of 2,170 Jev projects) | **No** |
| Horace He, defeating nondeterminism in LLM inference | **No** |
| ULMFiT, Word2Vec, GloVe, T5, BERT | Background only; not needed |
| Ettin, late interaction (our encoder family) | Not mentioned by the post; ours: `papers/2602.16609` (ColBERT-Zero) and Ettin entries in `papers/INDEX.md` |

Of the uncovered items, only RLCR and the Guo et al. calibration paper bear on our work, and both only as citations for
things we already do (Brier term, held-out temperature). The arXiv survey 2609.30216 might list Jev-likes we have not seen;
worth a skim if the Jev series is extended, not otherwise.

## 6. Relevance to this repo

Our setting: a per-user categoriser choosing among a household's 20-60 purpose categories from a short transaction plus
history. Best: **decider** (Qwen3.5-4B, letter-slot readout, LoRA on synthetic households incl. the REAL-6 set; ~73 top-1 on
the owner's real budget, 10-13 ms per transaction on an H100). Challenger: **li_decider** (Ettin-32M late interaction,
MaxSim between the query (transaction plus nearest/recent history rows) and one document per category option, softmax
over options, CE + Brier; ~70.9 on the owner's budget, 1.9 ms per transaction); the ~2-point gap is mostly first-time
payees. In flight: row 228's joint "GLiClass-style" mode (options inline in one encoder sequence), and Clef-flash
zero-shot (row 164).

**Where the post agrees with what we already found**

- *Temperature does the calibration work, Brier adds little.* His ModernBERT table (ECE 15.20 to 5.63% from a temperature,
  to 14.94% from Brier) is a third independent instance of `jev_meta_analysis.md` §2.4 ("CE + λ·Brier with λ ≤ 1 is
  indistinguishable from CE"). Our scorecard already fits one temperature leave-users-out (`ai_experiments.scorecard`,
  four folds by user), so reported calibrated bits and auto-file coverage are already post-temperature.
- *A fine-tuned small encoder matches a big general decision model on one task.* His ModernBERT-large vs Jev on IMDb is the
  public analogue of li_decider (33M) at 70.9 vs decider (4B) at ~73: specialist encoders close most of the gap when
  trained on the task. It also predicts that Clef-flash zero-shot (a generalist, no household training) will trail both
  trained models on our task; that is the expected result, not a reason to stop row 164.
- *Data more than tuning.* Our evidence is more specific: more households did not help (PLAN current state: "renames, more
  households ... did not"), but better-matched households did (v5 households +1.5 for the encoder, row 226/231). That
  supports TypeSafe's "synthetic, curated" line over Raschka's "more labels" anecdote: the match of the generator to the
  owner's budget is the lever, not the count.

**Where the post differs from what we do**

- His Jev-like recipe is a **per-option cross-encoder with a one-node head**. li_decider is a late-interaction bi-encoder
  (options cached once per day, 1.9 ms); row 228(a)/(b) is a joint encoder (all options in one sequence, GLiClass-like);
  row 228(c) is exactly his design applied as a reranker on li_decider's top 5. Our first-time-payee gap is the case where
  the transaction text must interact with the option's description (no history row to match), which is what a cross-encoder
  buys over MaxSim. So the post is an argument for keeping arm (c) in row 228, not a new arm.
- He uses a fresh linear head; decider reads letter logits from the existing LM head, and li_decider has no head (MaxSim
  score). `jev_meta_analysis.md` §2.3 found heads trained on narrow data transfer worse; nothing in the post tests that.
- RLCR assumes generated confidences and no CE. We have labels and differentiable option probabilities, so the supervised
  form (CE, optionally + Brier) is the exact gradient of what RLCR estimates. No reason to test RL for calibration.

**Concrete changes it suggests, ranked**

1. **A `BRIER=0` arm in the next li_decider screen (one seed, synthetic v5 + owner's budget, read raw and post-temperature
   ECE as well as top-1).** `scripts/li_decider.py` defaults to `BRIER=1`, chosen as "Clef's calibration term", and no
   job list or report row has run it off. The post's ModernBERT table and `jev_meta_analysis.md` §2.4 both predict a tie
   after temperature; if it ties, CE alone is the simpler default; if Brier helps top-1 or first-time payees, we learn
   something the field has not shown. Cost: one extra arm riding an existing launch, no new code (env var only). Low
   expected gain; the value is removing an untested default.
2. **A classic per-user lexical floor on the owner's budget: TF-IDF over payee character n-grams + logistic regression
   trained on that user's own history, CPU only, run locally (real data stays off Modal).** Raschka's "go-to baseline" is
   missing from our scorecard: the no-model cascade is lookups (own label, other users, usage prior), and the earlier
   logistic-regression heads (§24 baselines, BASE-4) were on frozen embeddings in the synthetic universe, not on the owner's
   budget. It would show how much of 70-73 is lexical memory of payees, and should be near zero on first-time payees,
   which sizes how much of the li_decider-decider gap any history-matching model can close. Cost: minutes; no GPU.
3. **No new arm, but keep row 228(c) (per-option cross-encoder rerank) and read it on the first-time-payee slice.** It is the
   post's design and the one mechanism aimed at the gap's location. If the reranker closes the gap at acceptable cost (one
   pass per top-5 option), it beats adding interaction layers (row 211) on simplicity.

Not suggested: switching base to ModernBERT (Ettin covers the same family; row 230 already plans Ettin-400M, which is
ModernBERT-large-sized); RL calibration (RLCR/RLCD); IMDb, BTZSC or the TypeSafe workflows as benchmarks for us (none has
a per-user option set or history; blind_v1/v2 and the owner's budget remain the judges); repeat-run determinism checks
(Jev's 4-in-25,000 wobble is far below our seed noise of 1-2 points on the owner's budget).

## 7. File map

- Post digest: `references/blog/other/2026-09-29-language-models-for-text-classification-from-bag-of-words-to-jev.md`
- Related: `references/jev_meta_analysis.md` §2.4 (calibration), §2.3 (training and transfer); `references/laya_analysis.md`
  (BTZSC numbers); `references/software/clef/README.md` (joint schema head, Decision Index); `scripts/li_decider.py`
  (`BRIER`, line ~901 loss); `src/ai_experiments/scorecard.py` (leave-users-out temperature).
