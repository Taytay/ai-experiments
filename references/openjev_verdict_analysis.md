# openJev Verdict: two small ModernBERT decision models, one unobtainable, and a claim that does not survive

Date: 2026-09-22. Tenth memo in the Jev series. Objects: https://github.com/Heman10x-NGU/openJev-verdict-2.0 (HEAD
`bff28567`, about 240 stars, created 2026-09-19, 25 commits in two days; clone at `~/projects/Taytay/openJev-verdict`) and
https://huggingface.co/heman10x/rlcd-modernbert-151m (files without weights in `~/projects/Taytay/openJev-verdict-hf/`).
Read by a subagent; this memo condenses its report. JevBench v1.3.0: "openJev Verdict 1.4" #36 at 38.9 (Intelligence 39,
Calibration 74; tiers 0.861 / 0.677 / 0.562 / 0.377) and "openJev Verdict" #37 at 38.1 (I 40, C 51).

## 0. The one-paragraph answer

The repository bundles two unrelated models. **Verdict** (the JevBench entry) is `knowledgator/gliclass-modern-base-v2.0`,
a GLiClass uni-encoder on ModernBERT-base (151M), fine-tuned for three epochs (645 s) on 2,300 Banking77-derived
five-candidate items with CE plus Brier on one-hot targets; version 1.4 changed only inference: a per-option-count
temperature, NLI-style `It is {description}` label framing, a 512-token budget. **Verdict 2.0** (the "beats Jev and Laya"
model) is plain ModernBERT-base (150M) with Laya's per-option `[MASK]` marker head, trained 8.8 hours on a GTX 1660 Ti on
the `train` split of `LocalLLaMA/typed-decisions` with soft-label CE, Brier, RPS and a permutation KL, then temperature-scaled
per (question type, option count) and given a separate correctness head. **Nothing in either is reinforcement learning;
"RLCD" is a package name.** The claim (0.771 vs Laya 0.766 vs Jev 0.727) compares a specialist fine-tuned on the benchmark's
training split with a zero-shot generalist, sits above the teacher's own 0.735 self-agreement ceiling, has a 0.5-point
margin inside sampling noise, quotes a Laya Brier that disagrees with Laya's own card, and cannot be reproduced: the
checkpoint's Git LFS object returns 404 and the named Hugging Face repo does not exist publicly. On JevBench the published
weights sit at Intelligence 39-40, below Laya's 46, and v1.4's NLI framing lowered Intelligence while raising Calibration.

## 1. The two models

**Verdict v1/v1.4** (`core/`): input `<<LABEL>>desc1<<LABEL>>desc2...<<SEP>>Question: ...\n\nContext:\n...`, 24
substantive label slots plus one `__insufficient_evidence__` slot on every query, logits sliced to K, divided by a per-K
temperature, softmaxed. The calibrator (`artifacts/calibrator.json`) has a global T of 2.80 and per-K values from 5.01 at
K=2 down to 1.51 at K=25, non-monotonic in the middle (K=9: 1.67, K=11: 3.39, K=17: 1.72), the signature of 100-item fitting
slices. The weights were "trained on context states under 71 tokens" and served at 512; the `It is` template exists only at
inference and is untested. Banking77 test: accuracy 95.0%, ECE 3.4%, out-of-scope abstention recall 97.5%. Zero-shot on
typed-decisions it scored 0.261, below the 0.299 uniform baseline, and on 337 TypeSafe public-eval questions 0.481 against
Jev's 0.908.

**Verdict 2.0** (`verdict2/`, about 900 lines): ModernBERT-base plus a question-type embedding added to every hidden state,
an MLP scorer at each `[MASK]`, sequence layout and budgets identical to Laya's English defaults (512 tokens, 192 for the
head, 48 per option). Loss: soft CE against the three-sample teacher distribution, plus 0.5 Brier, 0.25 hard CE, RPS on
score questions, and a symmetric KL to an option-shuffled twin on 30% of steps ("borrowed from Kev"). Eight epochs, best at
epoch 6 by dev accuracy, 5.8 GB peak. Post hoc: per-(type, K) temperature by grid search on a calibration fold, and a
seven-feature MLP predicting P(argmax correct). Dev after calibration: accuracy 0.785, Brier 0.064, distribution ECE 0.167,
correctness-head ECE 0.029. Each question re-encodes the state; 6.4 decisions per second on CPU.

## 2. The claim, taken apart

- Fine-tuned on the benchmark's training split; Jev's 0.727 is zero-shot. The dataset card itself says "A specialist number
  sitting next to a generalist number, unlabelled, misleads the reader." Laya's 0.766 is the same kind of specialist number.
- Above the teacher ceiling (self-agreement 0.735): the dataset card says a score "much above 0.75 means a model has learned
  the teacher's quirks rather than the task."
- 0.5 points on 2,000 decisions clustered in 400 cases is inside noise; per workflow the two specialists split.
- The README quotes Laya's Brier as 0.066; Laya's card says 0.062, better than Verdict 2.0's 0.064. Jev's soft accuracy
  (0.580) beats Verdict 2.0's (0.552).
- The headline 1.44% ECE is the correctness head's; the distribution's own ECE is 0.151.
- `verdict2/evaluate.py --skip_perm` hard-codes the permutation figures that appear in the receipt, so that block is not
  evidence of a run.
- The README's leaderboard images are self-generated replicas showing rank 1 or 2; the official board has it at #36.

## 3. What it adds, and what it means for the plan

- **It confirms Laya's lesson from a second encoder.** Two ModernBERT decision models with the same marker geometry, one
  trained on Banking77 and one on typed-decisions, both collapse off their training distribution (Verdict 1.0 below uniform
  on typed-decisions; Intelligence 39 on JevBench). A 150-400M encoder fine-tuned on one narrow domain loses whatever
  zero-shot transfer its starting checkpoint had.
- **Proper-scoring training without RL** is what it actually does, and on the same benchmark it matches Laya's Gaussian-noise
  REINFORCE. That is the third independent confirmation (jqv, reflex, this) that RLCD's reward can be backpropagated directly.
- **Reusable**: `verdict2/` is a compact reference for a marker-pointer head with soft-target CE plus Brier, a per-(type, K)
  temperature fit, an out-of-fold correctness head, length bucketing, and the twin-batch permutation KL. It fits in 6 GB;
  on a 3090 ModernBERT-large full fine-tuning is comfortable. Fit per-K temperatures only with enough items per bucket.
- **Avoid**: inference-only prompt templates the model never trained on; training on short contexts and serving long ones;
  quoting a correctness head's ECE as the model's calibration; unlabelled specialist-vs-generalist tables.

## 4. File map

| Where | What |
|---|---|
| `README.md`, `AGENTS.md`, `RUNBOOK.md` | Claims, the two-model note, the 1660 Ti recipe |
| `core/formatting.py`, `engine_encoder.py`, `calibration.py` | GLiClass prompt, `It is` framing, per-K calibrator |
| `verdict2/{model,data,losses,train,evaluate,metrics}.py` | Verdict 2.0 |
| `reports/verdict2_base_test.json`, `verdict_baseline_benchmark_full.json`, `reference_floors.json` | The receipt, the zero-shot 0.261, the TF-IDF floor (0.661) |
| `artifacts/calibrator.json`, `artifacts/verdict2-base/dev_metrics.json` | Per-K temperatures; dev metrics |
