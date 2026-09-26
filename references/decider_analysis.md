# decider: a full fine-tune of a Qwen3.5 base model into a letter-slot decision model, with the whole data pipeline public

Date: 2026-09-22. Sixth memo in the Jev series. Objects: https://huggingface.co/Mapika/decider-2b and
https://huggingface.co/Mapika/decider-35b-a3b (cards and configs saved under `~/projects/Taytay/decider-hf/`), and
https://github.com/Mapika/decider (Apache-2.0, commit `c4daaac`, author Mark Marosi, clone at `~/projects/Taytay/decider`).
Read by a subagent; this memo condenses its report. JevBench v1.3.0 rows: decider-35b-a3b #8 at 67.6 (Intelligence 80,
Calibration 72, hard 0.655); decider-2b #23 at 61.7 (I 61, C 47, Speed 83, Cost 61, hard 0.473).

## 0. The one-paragraph answer

decider is the closest published thing to "train a small decoder for decisions from scratch": Qwen3.5-2B-*Base*
(1.9B, hybrid gated-delta-net with full attention every fourth layer), fully fine-tuned with plain cross-entropy on the
option-letter logits at an `Answer: (` slot, over 1.47M examples and 455M tokens from about 95 public decision datasets plus
teacher-written and programmatic augmentations, one epoch, 5.3 hours on a GH200. No new head, no special tokens, no LoRA;
the "trained readout" is the whole network. One scalar temperature (1.30 for the 2B, 1.08 for the 35B) is fitted on
in-task eval logits. A later 384-step RL stage on live click tasks and exact games (v8 to v10) improved in-domain belief
calibration and did not move JevBench hard accuracy (0.459 before and after). The 35B-A3B variant is the same supervised
recipe on the 35B MoE base with the 256 routed experts frozen (2.45B of 34.7B parameters trained), Muon on the block
matrices, 394 minutes on four B300s. On JevBench the 2B lands *below* the frozen Qwen3.5-4B readout on hard items (0.459
public vs SemIf's 0.613) and the 35B above it (0.676), which the author explains directly: a 2B without reasoning cannot do
the hard tier's long policies, multi-hop and temporal arithmetic, and its hard-tier ECE of 0.30 is what JevBench's
Calibration axis (hard tier only) scores as 47.

## 1. How it works

- **Prompt** (`decider/prompt.py`): `Context:\n<state>\n\nQuestion[ k]: <text>\nOptions:\n(A) opt\n(B) opt...\nAnswer[ k]: (`.
  Further `Question k ... Answer k: (` blocks are appended for multiple questions; the slot is the final `(`. A second
  "schema-first" layout (questions first, then the context, then all answer slots) was trained 50/50 from v8 so the
  question prefix can be cached.
- **Readout** (`decider/model.py`, 47 lines): the last hidden state at the slot times the LM-head rows of the letter
  tokens, invalid options masked to minus infinity. Letters A to J for up to ten options, then single-token two-letter
  uppercase labels to 255. The 2B ties word embeddings, so the "head" rows are the letter embeddings. No new parameters.
- **Independence and shared state** (`Engine.score_shared`): by default each question is its own row; the state prefix is
  run once and its cache (attention KV of the six full-attention layers plus the recurrent state of the eighteen delta-net
  layers) is forked to each row. Packing questions into one row halves latency but reversing question order changes up to
  12% of answers. Score fields are decomposed into one yes/no row per level ("Proposed answer: <level>. Does the proposed
  answer fit?"), P(yes) normalised across levels; `fit_mass` (the sum before normalisation) is near 1 when exactly one
  level fits.
- **Wire format** (`decider/systemone.py`): choice/score/noul with criteria rendered as `name: description`, question ids
  never shown, JSON states serialised compactly with `_index` written into arrays of eight or more (0.49 to 0.57 on a
  64-record probe). Confidence is the top probability at the stored temperature; a separate `certainty` is one minus
  normalised entropy.
- **Serving**: CUDA graphs bucketed over batch and length, torch.compile, optional FP8 linears: 4.0 ms per single request
  and 1,370-1,670 decisions per second at batch 32 on a GH200. The 35B is eager only, 47 ms per request.

## 2. How it is trained

- **Objective** (`decider/train.py: loss_fn`): `F.cross_entropy` on the masked letter logits. A Brier term and label
  smoothing exist and are off by default. Full AdamW over all parameters; peft is imported and unused.
- **Hyperparameters** (`scripts/train.sh full`): lr 1e-5, warm-up 150, cosine to zero, 16,384-token micro-batches times
  accumulation 2, betas 0.9/0.95, no weight decay, clip 1.0, bf16 with gradient checkpointing, context 16,384,
  `--max_options 255`, `--none_prob 0.1`, `--schema_first_prob 0.5`, one epoch. The single clean run matched the staged
  v1-v9 releases on accuracy (0.809 vs 0.812 in-task) with *better* raw calibration (T 1.03, ECE 0.030 vs 1.36 / 0.056),
  so one epoch beats staged continuation for calibration. Half an epoch gave 99% of final accuracy on the 35B curve.
- **Data** (`decider/data/mixture.py`, 95 registered tasks, per-task cap 20,000): intents (CLINC, Banking77, MASSIVE,
  HWU64), topics, sentiment and emotion, moderation, NLI, paraphrase, fact verification, MCQ (MMLU, MedQA, ARC, SciQ),
  rating scales (HelpSteer2, LIAR2), pairwise preference (RewardBench, Arena, UltraFeedback), tool selection (Glaive,
  ToolACE, Hermes), AgentTraj-L next-action, Mind2Web element choice, text-game states. Augmentations: full label sets up
  to 255 options and padding with unrelated labels, described or opaque-named options with JSON rubrics (40k), JSON
  multi-record states with path questions (52k), single-question re-renders, isolated levels, programmatic
  rule-conditioned records (90k). A local Qwen3.5-27B teacher wrote 669 label descriptions, 3,043 custom questions,
  13,848 routing messages, 1,502 situations and 4,771 shell commands, then re-answered from its own letter logits; only
  agreeing labels were kept (it agreed with itself on only 72% of generic-option labels). All teacher data is in
  `teacher_data/`.
- **Order and abstention**: options shuffled at build time every epoch; label sets over ten sub-sampled to ten with gold
  kept. With probability 0.1, questions with three or more options get an abstain option in one of twelve wordings: 75%
  with gold unchanged, 25% with the whole option list replaced by another task's labels so abstain is correct.
- **RL stage v8 to v10** (`docs/RL.md`, code not released): 384 AdamW steps at lr 1e-6 on live MiniWoB++ click tasks,
  exact 4x4 minesweeper, a slippery grid and bag draws. PPO-clip on terminal outcome (coefficient 0.1), a proper log score
  of a "what happens next" belief question against the exact law (0.2, two option orders averaged), a rendering-consistency
  KL across layouts and orders (1.0), and a retention KL to v8 gated at 0.01 mean / 0.05 max on replayed rows, dropping the
  reward terms when violated (94 of 576 steps). No gold labels.
- **35B**: 1,543,567 items, 463M tokens, Muon on 250 block matrices (1.41B) plus AdamW on the rest (1.04B), lr 1e-5, 16,287
  steps of 32,768 tokens, 100 GB peak per GPU. The AdamW-only arm was stopped at 11% on training-loss evidence, so no
  finished optimiser comparison exists.

## 3. Results and flags

Regression set (67 in-task, 28 held-out tasks, options sub-sampled to ten), accuracy / NLL / ECE: 2B v10 0.805 / 0.474 /
0.037 in-task and 0.755 / 0.622 / 0.084 held-out; 35B 0.855 / 0.357 / 0.026 and 0.810 / 0.497 / 0.069. Zero-shot base
2B 0.620 in-task, base 35B 0.732, so the supervised stage is worth about 19 and 12 points in distribution.

JevBench public items, author-run argmax: 2B easy 1.000 / standard 0.847 / hard 0.459; 35B 1.000 / 0.972 / 0.676; the card
copies Jev at 1.000 / 0.986 / 0.730 and SemIf 4B hard 0.613. Flags: 111 public hard items against the board's 220,
author-run not harness-run, no Speed or Cost. On the Bespoke public suite the 2B scores 0.704 macro and the 35B 0.774
against copied Nimble-9B 0.748 and Jev 0.760, but 7 of 13 subsets have their train split in decider's mixture; the
untrained-only macro for the 35B is 0.744. RL results (browser 83.0 to 93.2%) are on 22 click-only synthetic pages. Held-out
ECE above 0.1 remains for the 2B on arena preference (0.189), dolly category (0.203), hermes tools (0.208) and QuALITY
(0.233).

**Why Calibration is 47 at 2B and 72 at 35B.** JevBench's Calibration axis is `100 x (1 - ECE/0.5)` on the hard tier
plus probability fidelity on 20 distribution items. The author reports hard-tier ECE 0.30 for the 2B and 0.15 for the 35B,
which gives 40 and 70 before fidelity is averaged in. The cause named in the README: a single temperature fitted on
in-task classification does not transfer to long-policy and multi-hop items, where the 2B's accuracy is 0.26-0.33 but its
confidence stays high. Per-tier ECE for the 35B: 0.001 easy, 0.059 standard, 0.151 hard.

## 4. What decider settles

1. **Training a base model on 455M tokens of decision data teaches the format and the in-distribution tasks, not the hard
   tier.** At 2B the result is below the frozen instruct 4B on hard items. At 35B-A3B it is above, but the 35B *base*
   zero-shot was already 0.732 in-task, and the trained 35B on JevBench hard (0.676) is roughly where jqv's frozen Qwen3-32B
   with one temperature sits (0.645 on all 220; 0.622 public). Scale carries the hard tier; training carries coverage.
2. **Trained raw logits are nearly calibrated in distribution** (T 1.03-1.30 against jqv's frozen 3.0-12.0), and still
   mis-calibrated on the hard tier (ECE 0.30). Same finding as reflex and jqv from the other direction: training moves the
   temperature toward 1 on the training population and nowhere else.
3. **The abstain augmentation is a concrete recipe**: 10% of multi-option questions get an abstain option, and a quarter of
   those have the true options swapped out so abstain is correct. This is the trained-IDK finding of `2405.05904` made
   operational.
4. **One clean epoch beat staged continuation on calibration** at identical accuracy.
5. **A KL-gated late RL stage with a proper-score belief reward is safe and in-domain only.** It did nothing for JevBench.

## 5. What this means for the plan

- **The 2B recipe is runnable on a 3090 with two changes.** Full AdamW on 1.9B parameters is about 23 GB of weights and
  optimiser state in mixed precision, so it needs 8-bit Adam or the 0.8B base (0.776 / 0.707 in-task / held-out against the
  2B's 0.809 / 0.739); and the 16,384-token micro-batch has to shrink. The plain letter-slot cross-entropy on a base model is
  the same objective as our categoriser SFT with a letter readout, so the comparison is direct.
- **Base versus instruct is undocumented in decider**, and it matters for us: our arms start from the instruct model.
  decider's zero-shot base 2B at 0.620 in-task suggests the instruct model's format knowledge is worth little once you train
  the slot, but nobody has measured it.
- **Abstention needs the 75/25 augmentation**, not just a listed option. SemIf's `insufficient` option and our category
  "unknown" both fall to confident wrong answers without it.
- **Score as isolated levels** (one yes/no row per level, normalised) is an alternative to reading an ordinal as a K-way
  choice, and `fit_mass` is a free abstention signal. Relevant if we ever score confidence bands.
- **Shuffle options at build time and sub-sample large label sets to ten with gold kept.** Our categories are tens; the
  sub-sampling forces the model to read candidates rather than memorise a fixed alphabet, and it is what makes 255-option
  inference work at all.
- **Keep the abstain, the shuffle and the single epoch; skip the RL.** The RL stage is unreleased, in-domain only, and did not
  move the external benchmark.

## 6. File map

| Where | What |
|---|---|
| `decider-hf/decider-2b/README.md`, `decider_config.json`, `eval_results.json` | Card, T 1.30 and v10, 95-task results |
| `decider-hf/decider-35b-a3b/` | Card with JevBench, Bespoke and MiniWoB numbers, T 1.08, grouped-MM experts |
| `decider/decider/prompt.py`, `model.py` | Two layouts and the 255-label table; the 47-line readout |
| `decider/decider/train.py`, `scripts/train.sh` | CE trainer, hyperparameters, full and delta modes |
| `decider/decider/systemone.py`, `engine.py`, `serve.py` | Jev adapter with isolated levels, shared-state fork, CUDA-graph server |
| `decider/decider/data/{core,mixture,augment,rules,teacher_*}.py`, `teacher_data/` | The 95-task mixture, augmentations, teacher outputs |
| `decider/moe/` | Frozen-expert 35B training with Muon, NVFP4 PTQ |
| `decider/docs/{HISTORY,RL,CHANGELOG}.md` | Staged releases v1-v10 and the RL stage |
