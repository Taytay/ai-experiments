# kev: a pointer head on frozen Qwen base models, the calibration it never shipped, and the research log that says why

Date: 2026-09-22. Ninth memo in the Jev series. Object: https://github.com/jaredpalmer/kev (Apache-2.0, about 2,500 stars,
HEAD `90990a5`, clone at `~/projects/Taytay/kev`); the JevBench build is commit `20fa626` (2026-09-20). Read by a subagent
(which also inspected the Hub `head.pt` files); this memo condenses its report. JevBench v1.3.0: kev 0.6B #19 at 62.5 (I 52,
C 51, hard 0.400), kev 4B #27 at 59.7 (I 65, C 42, hard 0.423), kev 8B #29 at 56.4 (I 69, C 44, hard 0.473), kev 0.5B #38 at
33.2 (I 38, C 47). All BF16 on one RTX 3090, which makes kev the closest reference class to this repo.

## 0. The one-paragraph answer

kev implements Hume's inferred Jev architecture directly: a frozen Qwen *base* model, prefill only, with a rank-16 LoRA and
a trained pointer head that scores each option's closing-delimiter hidden state against a per-question `<decide>` token,
questions packed after one shared state under a block-causal mask. Training is hard-label cross-entropy on 10,000 records
from ten public datasets (1,000 each) plus 2,576 programmatic rule-pair and rule-tree records with executable labels; no
teacher LLM, no Jev outputs; two epochs. The benchmarked checkpoints (Qwen2.5-0.5B, Qwen3-0.6B/4B/8B-Base) are a
superseded generation. **The Calibration scores of 42-51 have a mundane cause: the served checkpoints had no temperature.**
The experiment code fitted one on every run (0.6B 1.93, 4B 1.68, 8B 2.46) and reported it, but it was never written to
`head.pt` and the server at `20fa626` could not apply one; the saved fitted temperatures would have cut out-of-domain ECE to
0.073 / 0.062 / 0.047. On the author's own frozen OOD suite, kev 4B beats SemIf's exact prompt on the same backbone family
by 7.7 points (0.790 vs 0.747) and the untrained 8B base by 5.8, while losing base knowledge (MMLU 0.75 to 0.69 at 8B).
`PLAN.md` is a 755-line research log that records these findings and its own corrections, including reversing the claim
that an in-distribution temperature does not transfer.

## 1. How it works

- **Sequence** (`kev/model.py: encode`): `<state> ... <q> instr <opt> o1 </opt> <opt> o2 </opt> ... <decide>` per question,
  delimiters reusing existing Qwen special tokens (`<|fim_prefix|>`, `<|box_start|>` and so on) so no embeddings are added;
  user text is rewritten so delimiters cannot be forged. Training context 384 state / 1,024 branch / 2,048 packed; serving
  8,192.
- **Readout**: `PointerHead`, two `Linear(d, 256)` maps; logit_k = K(h at option k's `</opt>`) · Q(h at `<decide>`) /
  sqrt(256). Noul is two options `no`/`yes`; score is ordered level texts with the expected index. No letter alphabet, so no
  26-option cap (Banking77 77-way at 0.86 even at 0.5B).
- **Isolation**: a token attends to j iff j ≤ i and j is in the state or in the same branch; branch positions restart after
  the state; packed and separate agree to 4e-6. Qwen3.5 hybrids (Gated DeltaNet ignores masks) run one row per question
  from a replicated state cache instead, as reflex does.
- **Confidence**: choice `(p_max − 1/K)/(1 − 1/K)` (the Jev adapter formula Hume confirmed); probabilities rounded to two
  decimals.

## 2. Training

- **Data** (`kev/data.py`, suite `evals/v7/decision-v7`): Banking77, BoolQ, AG News plus two derived nouls, MNLI, SST-5, Yelp
  plus a noul, TREC, DBpedia-14, Amazon reviews, IMDB, 1,000 each; 896 programmatic policy minimal pairs
  (`kev/contrastive.py`: rule plus facts with an executable label, with ablation and invariance checks); 1,680 records from 60
  random rule trees (`kev/composition.py`). Eval-only, never trained: MMLU, Emotion, TweetEval-offensive, QNLI, PAWS, SciQ,
  held-out rule shapes. About a third of states wrapped as JSON objects or arrays; 30% of option descriptions nulled.
- **Augmentation**: options permuted every epoch; none-of-the-above replaces the true option at p=0.10 and is added as a
  wrong option at p=0.12; distractors at p=0.15; "none minimal pairs" on a quarter of choice records. Permutation-KL and
  exact per-option isolation exist but are off (isolation cost 5.8 points at 4B).
- **Loss** (`kev/train.py: question_loss`): hard-label CE averaged per record. Label smoothing, Brier, focal, RPS and an
  anchor KL to the base model's zero-shot letter distribution are implemented and default to zero.
- **Hyperparameters**: LoRA r=16, alpha 32, dropout 0.05 on attention and MLP (plus DeltaNet projections on Qwen3.5); pointer
  head from scratch (4B: 33.8M trainable); AdamW wd 0.01, OneCycle 10% warmup, clip 1.0, two epochs, effective batch 8, bf16
  autocast with fp32 master weights. lr 1e-4 at 0.6B, 5e-5 at 4B/8B; the original 2e-4 "erodes base capability" (−4.7 pts).
- **Compute** (H100 on Modal): 0.6B 638 s; 4B 2,379 s at 23.0 GB peak, batch 4 with checkpointing; 8B 4,972 s at 36.9 GB.
  $2.8-6.5 per trial.

## 3. Results, the calibration story, and flags

Frozen suites with dev and locked test partitions; Jev scored live on the same items (`runs/jev-transfer-v4`: acc 0.857,
ECE 0.049, Brier 0.211, confident errors 3.7%). Transfer-v4 dev, raw: 0.6B 0.620 / ECE 0.152 / mean confidence 0.771 /
confident errors 10.8%; 4B 0.790 / 0.102 / 0.887 / 8.2%; 8B 0.796 / 0.121 / 0.907 / 9.9%. Per source, 8B vs Jev: MMLU 0.70
vs 0.90, date-arithmetic deadlines 0.60 vs 0.93, PAWS 0.78 vs 0.79. On SemIf's 144 authored items the current 9B scores
0.917 against the untrained base's 0.813 and Jev's 0.965.

Why Calibration is 42-51 on JevBench, from what the benchmarked code did: raw logits served with no temperature; hard
one-hot CE on certain labels, with the programmatic pairs (a third of the mix by weight) teaching certainty; two epochs,
although the log records that one epoch "is better calibrated (Brier 0.379, confident errors 2.7%) at equal accuracy"; and
noisy transfer sources (PAWS alone is 16 of 26 confident errors at 9B). On evidence-free "unknowable" items the kev models
answered at ≥0.9 confidence 26-44% of the time (Jev 9%). The post-hoc loss screen found no fix to *ordering*: label
smoothing 0.05 destroyed selective coverage (0.576 to 0.006), CE+0.5 Brier and focal γ=1 changed nothing useful, and
per-(type, K) temperatures were worse out of domain than one global T. The current Qwen3.5 checkpoints store T (0.8B 2.41,
4B 2.14, 9B 2.30) and a uniform-target "unknowable" delta cut the ≥0.9 share on those items from 0.19 to 0.00 at 4B.

Flags: the "trained sources" columns are in-distribution held-out splits; only "new sources" are transfer. The author states
Kev-vs-Jev "isn't a controlled comparison." MMLU's test split is used as eval-only and pretraining overlap is unknown.

## 4. What kev settles

1. **A trained head beats a frozen letter readout by 5-8 points on decision-shaped transfer at 4-8B**, where jqv found LoRA
   on MMLU added nothing: the gain is on classification and policy data, not knowledge, and it costs knowledge (MMLU −6).
2. **Serving without a temperature is the whole of kev's calibration deficit on JevBench**, and one global T fitted in
   distribution does transfer out of distribution on kev's suites (the author reversed the opposite claim). This sits
   against jqv's and reflex's finding that T does not transfer across task *types*: kev's OOD suite is classification-like,
   jqv's failing case was knowledge vs reading. The rule that survives all three: one T per task type.
3. **Hard labels on certain synthetic data and a second epoch both push over-confidence**; soft or uniform targets on
   ambiguous and evidence-free items fix the unknowable case.
4. **No loss change fixed selective coverage.** Ordering (which items are right) is set by the representation, not the
   calibration loss.

## 5. What this means for the plan

- **This is the reference implementation for a 3090.** `kev/model.py` (pointer head plus block-causal mask, about 300 lines,
  backbone-agnostic) works on Qwen3 base models with a packed mask; the 4B needs batch 1 with accumulation 8 on 24 GB; the
  0.6B/0.8B fits easily. Expect several times the H100 wall times (0.6B about 11 minutes, 4B about 40).
- **Learning rate: 5e-5 at 4B, 1e-4 below; never 2e-4.** Our categoriser LoRA runs should check this directly (REPORT 32's
  learning-rate sweep is the place).
- **One epoch, store the temperature, uniform targets for "unknown"-shaped items.** Three cheap changes that each moved
  kev's calibration.
- **The pointer head removes the letter cap and the letter prior** at the cost of a from-scratch head. jqv found a
  from-scratch head on a *frozen* LM is 8 points worse than the readout; kev pairs it with LoRA, which is the fix.
- **Reusable code**: `kev/metrics.py` (ECE, Brier, NLL, coverage at 5% error, AURC, clustered paired bootstrap),
  `scripts/calibrate_checkpoint.py`, `kev/contrastive.py` and `composition.py` (executable-label generators), and the
  suite-freezing discipline in `kev/suite.py`.

## 6. File map

| Where | What |
|---|---|
| `kev/model.py` | `encode`, block-causal mask, `PointerHead`, hybrid row path |
| `kev/train.py` | LoRA plus head training, `question_loss` and its unused options |
| `kev/data.py` | 19 dataset converters, trainable vs eval-only policy, augmentation, none pairs |
| `kev/contrastive.py`, `composition.py` | Rule minimal pairs and rule trees with executable labels |
| `kev/api.py`, `serve.py` | System One mapping, confidence formulas, prefix cache, `/permute` and `/separate` |
| `kev/metrics.py`, `scripts/calibrate_checkpoint.py` | Calibration metrics and the T-writer added 2026-09-21 |
| `PLAN.md` | The 755-line research log, including the Laya and SemIf reviews and the corrections |
| `runs/v7-*/*/result.json` | Raw and tempered metrics and training resources for the benchmarked checkpoints |
