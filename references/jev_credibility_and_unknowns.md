# Which Jev-reproduction claims are credible, and what anyone actually knows about Jev

Date: 2026-09-22. Companion to `jev_meta_analysis.md`; written after the owner asked (1) whether the "we reproduced Jev"
claims are false and which are credible, and (2) whether anyone knows what Jev is doing. Sources are the eleven memos in
this series and the documents they cite.

## 1. Which claims hold up

Almost every project honestly reproduces Jev's **interface**: typed answers, no generated text, one shared state for many
questions. None reproduces Jev's **quality or training**. The credible projects say so; the questionable ones claim to
match or beat Jev.

**Credible: claims hold up under independent measurement.**

- **jqv**: claims only the interface and measures the gap (about 10 points on MMLU and on JevBench hard). Benchmark Heaven
  re-ran it from published code on its own GPU over all 534 decisions including held-out, and the numbers reproduced.
  Pre-registered tasks, negative results published.
- **SemIf**: says it reproduces the interface pattern, not Jev's model or training, and that its probabilities are
  uncalibrated. Independently ranked #2, 1.3 points behind Jev.
- **reflex**: documents about twenty failed training runs and ships the untrained model; discloses that its board row ran
  a superseded adapter and that it tuned against public items nine times. Soft spot: a README claim about order averaging
  that its own later measurement contradicts.
- **kev**: research log records its own mistakes and reversals; scores Jev live on the same items; says the comparison
  "isn't a controlled comparison."
- **djev-dev and razorback16/openjev**: say they train nothing and that quality is DiffusionGemma's; djev withdraws its own
  speed-up claim for lack of a matched baseline. No claim to beat Jev; the benchmark's runs are the evidence.
- **JevBench**: pre-registered, items frozen before runs, held-out items private. One person's benchmark with LLM-written
  hard items, but the most careful evidence available.

**Credible method, overstated comparison.**

- **Nimble**: the public-benchmark result (1.2 points behind Jev on 13 human-labelled sets) looks real; the synthetic
  holdout table (six families, labels from the same models that wrote the data, mixed precision) is weaker.
- **decider**: training real and fully published, but its comparison suite includes 7 of 13 subsets it trained on, and its
  JevBench numbers are self-run on public items only.
- **Open-Jev (Zefan-Cai)**: rigorous pipeline; its "beats Jev" is on its own synthetic data, and it never ran the untrained
  model on JevBench, so its training effect there is unknown.

**Not credible as stated.**

- **Laya**: the Dev.to "+16 points over Jev" compares Laya on its own training distributions (minus its two worst families)
  against an unrelated Jev benchmark. The card's 0.766 vs Jev's 0.727 is a checkpoint fine-tuned on that benchmark's train
  split; the base model scores 0.36 there. The calibration win is post-refit; it ships at ECE 0.47. Independent board: #33.
- **openJev Verdict**: "beats Jev and Laya" is a fine-tuned specialist against a zero-shot generalist, above the teacher's
  own agreement ceiling, within noise, with Laya's Brier misquoted; the checkpoint cannot be downloaded; the README shows
  self-made leaderboard images at rank 1-2 while the real board has it at #36; "RLCD" with no RL in the code.

**The pattern.** Credible projects say "we reproduce the interface, here is the gap." Non-credible ones say "we beat Jev,"
usually by comparing a model fine-tuned on a benchmark's own data against Jev's zero-shot score. Quick test: does the
comparison label which system was trained on the test distribution?

Even the credible "above Jev on hard items" results (OpenJev thinking 0.782, djev thinking 0.777, reflex-27b 0.759) either
generate a reasoning pass first or use a 27B backbone, at 5-10x the cost, which is not what Jev claims to be.

## 2. What is actually known about Jev

Nobody outside TypeSafe knows how it works.

**Disclosed by TypeSafe**: "a new model architecture," a "parallel sampler," and "Reinforcement Learning for Calibrated
Decisions (RLCD)." The launch post's FAQ has the headings "Why was a new training algorithm needed?" and "Where does our
training data come from?" with no answers. No paper, model card, parameter count or data.

**Inferred from outside** (Hume, about 10,000 API calls; Hume calls it "clearly all quite speculative"):

- Shared state with isolated questions: latency grows with the state, barely with question count; a secret in one question
  is invisible to others but visible when moved into the state. Strongest evidence.
- Options interact before readout: an irrelevant fifth option shifts the odds between existing ones.
- Probably Qwen-related: tokenizer matches no public vocabulary but is closest to Qwen's (348 of 415 probes).
- Probably a mixture-of-experts with about 10B active parameters (30k tokens in about 160 ms). Least certain part.
- The confidence score is a post-hoc formula, `(p_max − 1/K) / (1 − 1/K)`, visible in TypeSafe's client code.

**Independently measured** (outputs only):

- JevBench held-out: #1 overall, 0.741 on the hard tier, about 10 points ahead of the best open one-pass rebuild.
- Calibration depends on who measured: ECE 0.031 on MMLU (Hume), 0.061 on JevBench hard, 0.246 on nibzard's suites (worst
  in that study).
- Weaknesses: temporal and numeric reasoning 8 of 30; 13% answer flips under option permutation; zero probability on the
  true label for 16% of DAIR Emotion items.
- TypeSafe's own 67.8% four-workflow figure is agreement with GPT-6 Astra and Claude Fable, not ground truth.

**What the open work suggests** (inference, not evidence about Jev):

- The interface is ordinary engineering; every observed behaviour reproduces on stock open models, so the "new
  architecture and parallel sampler" could be little more than shared-prefix serving plus a readout head.
- The unreproduced part is the last ~10 points on hard items at one-pass speed. Scale, better data and small-scale
  fine-tuning each got partway at best, which points to large-scale post-training on decision-shaped data with good soft
  labels, probably on a backbone stronger than 4B.
- A proper-scoring objective is cross-entropy under another name unless the reward comes from real outcomes observed after
  the decision. If Jev trains on outcome data at scale, that data is its real advantage, and it is invisible from outside.

Summary: the interface is understood, the results are partly verified, the training is a black box. "Reproduced Jev" can
only mean the interface.
