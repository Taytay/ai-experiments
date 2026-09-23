# jqv: the most careful open reconstruction of Jev, and the comparison it publishes

Series index and synthesis: `jev_meta_analysis.md`.

Date: 2026-09-22. Third memo in the Jev series after `semif_jev_analysis.md` and `laya_analysis.md`. Object of study:
https://github.com/Octalab-Inc/jqv (Apache-2.0, commit `16350df`, cloned to `~/projects/Taytay/jqv`), in particular
`docs/report.md` (77 KB, the English translation of a Japanese experimental report), `results/`, and the `tasks/` folder
where every experiment has a pre-written goal, scope, success criterion and a filled-in result with deviations. Also read:
Hume's reconstruction of Jev's architecture (https://archerhume.com/posts/jevs-architecture-unmasked/), which jqv
implements, and the JevBench v1.3.0 board (https://benchmarkheaven.com/jev-models, scored 2026-09-21).

## 0. The one-paragraph answer

jqv is a stock Qwen3 (1.7B, 14B, 32B; no decision training in the served configuration) behind a Jev-compatible API,
built to test Hume's reconstruction of Jev claim by claim rather than to ship a product. Five inference engines on one
prompt, with tests that their option logits agree in fp32: generate a letter, read the letter logits with no generation,
prefill the state once and replicate the KV cache, pack all questions after the state under a block attention mask, and a
mask-free Hydragen-style shared-prefix attention. One scalar temperature fitted on MMLU, with its provenance (model, prompt
hash, dataset) checked at server start. On top of that it measures what SemIf and Laya only asserted or omitted: letter and
position priors, the effect of a fifth option, whether a temperature transfers across languages and task types, whether a
trained head or LoRA adds anything at three scales, whether a Brier term in the loss beats a post-hoc temperature, and where
exactly the gap to Jev sits on JevBench's hard tier. Its published comparison is the honest kind: Benchmark Heaven measured it
three times, once through a tunnel to the author's Mac and then on their own H100 from the published code, and the numbers
reproduced. The headline findings are that not generating buys nothing by itself, that shared state buys 50-70x at long
states, that backbone scale sets accuracy and a 4,800-example LoRA does not move it above 14B, that one temperature reaches
Jev's ECE in distribution and fails across task types, and that the gap to Jev is about 10 points on both MMLU and JevBench
hard and does not close with any of the cheap tricks.

## 1. What jqv is built on: Hume's reconstruction of Jev

Hume made about 10,000 Jev API calls and inferred, with stated confidence levels: input tokens scale additively in state plus
questions, latency scales with the state and barely with the question count (1,500 questions on a short state about 600 ms,
one question on a 30k-token state about 160 ms), so there is a shared prefix plus per-question suffixes; a secret placed in a
sibling question is invisible to the other questions (p 0.00) but visible when moved to the state (0.90-0.92), so branches
are isolated; adding an irrelevant fifth option shifts the log-odds between two existing options (+0.38 to +0.11), so options
interact before the readout rather than getting independent logits; the tokenizer matches no public vocabulary but is
closest to Qwen's on 348 of 415 probes; the backbone is assumed to be a causal decoder, probably sparse MoE with about 10B
active parameters, because 30k tokens in 160 ms fits that and TypeSafe calls RLCD post-training of a pretrained LM. Confidence
in the API is the post-hoc `(p_max - 1/K) / (1 - 1/K)`. Measured: MMLU 0.918 on 1,200 items with ECE 0.031 as returned, no
temperature. Hume's own caveat: "clearly all quite speculative"; the exact attention mask is unknown and isolation could be
achieved other ways.

jqv's position is explicit: `packed` "is not a reproduction of Jev; it is a reference implementation showing that the
behaviours observed in Jev can be reproduced on an open Qwen."

## 2. What jqv does

### 2.1 Prompt and readout

Chat template with thinking pinned off (`<think>\n\n</think>\n\n` then `Answer:`), the state as `Document:` in the user turn,
each question as `Question: ... Options: A. ... B. ... Answer with the letter only.` Readout is the single tokens `" A"`,
`" B"`, ... at the last position, softmaxed over the K letters. `readout="rows"` computes the same logits from only the letter
rows of the LM head, identical within 1e-4, which shows the 150k-dimensional vocabulary projection is not part of the
decision (and saves about 4%). Prefix and suffix are tokenised separately and a test checks no BPE merge crosses the boundary.

### 2.2 The five engines and what the speed comparison says

| engine | structure | S=8038 tokens, Q=100 (1.7B, M5 Max) |
|---|---|---:|
| generate | emit a letter, parse it | 231 s |
| naive | one forward per question, read letter logits | 256 s |
| kvcache | prefill once, replicate the HF cache, batch the suffixes | 6.6 s |
| packed | `[state | q1 | q2 | ...]` in one sequence, block mask, position ids restart at S for each question | 4.8 s |
| shared | same input, no mask: branch queries attend to the prefix together, causal within a branch, log-sum-exp combination | 3.5 s |

At Q=1 all engines are equal. `generate ≈ naive` (0.9-1.1x) at every scale: the value of not generating is that a
distribution comes out, not speed. `packed` beats `kvcache` more the longer the state because the cache engine holds one
prefix copy per batch row. `shared` (their D3) gains with backbone size, 0.87x at 1.7B and 1.50x at 32B, because dense
masked attention wastes more the more layers there are; on MPS it needs a second SDPA pass to recover the partition function,
and the author expects FlexAttention's BlockMask on CUDA to make it one block-sparse kernel. Isolation is verified with a
negative control: `packed` leaks 0.000 of a sibling's secret, plain concatenation leaks 0.996.

This is the same trick as SemIf's `shared.py` (section 2.2 of that memo), done three ways with equivalence tests, and
with the CUDA path exercised only by Benchmark Heaven's re-run on an H100.

### 2.3 Priors, permutations and a fifth option (1.7B, MMLU 300 items)

- Rotating only the letters in front of fixed options changes the argmax on 46% of items; A and B carry mean probability 0.31
  and 0.28 against 0.22 and 0.19 for C and D.
- Rotating only the option texts under fixed letters changes the argmax on 52%; the second position is favoured (0.31). Mean
  accuracy barely moves, so both priors act as tie-breakers on weak-evidence items, which is why the per-item probability
  swings are large: uncalibrated probabilities are near one-hot and a tie-break flip swaps p(correct) between 0 and 1.
- On the bridge-inspection items where the answer is in the state, 87-93% of argmaxes survive either rotation. Strong
  evidence overrides the prior.
- A fifth "none of the above" option moves individual log-odds by up to ±4 nats but with no consistent direction on MMLU
  (Jev: consistently -0.28). The option list influences the hidden state jointly, but not the way Jev's does.
- `perm_avg`, returning the mean over the K cyclic rotations, is +3.0 points at 1.7B (p=0.05) and +3.1 at 14B (p=0.001),
  +0.4 at 32B, at 1.7x cost because the prefill is shared. It also halves raw ECE. It equals what the trained LoRA head
  achieves at 1.7B, with no training.

### 2.4 Calibration: one temperature, and where it stops working

Temperature fitted by bounded 1-D search on NLL (LBFGS diverged on a 30-item near-separable set), stored with model, engine,
prompt hash, dataset and choice counts; the server refuses a file whose model or prompt hash differs.

| backbone | MMLU acc | raw ECE | ECE with T | T |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | 0.554 | 0.413 | 0.080 | 12.0 |
| Qwen3-14B | 0.750 | 0.207 | 0.042 | 5.1 |
| Qwen3-32B | 0.809 | 0.137 | 0.023 | 3.0 |
| Jev (Hume) | 0.918 | 0.031, no temperature | | |

Transfer at 1.7B: T fitted on English MMLU gives ECE 0.082 on Japanese JMMLU (in-distribution 0.066) and vice versa, so the
scale distortion is the same across languages (T about 12 both ways). On the bridge task, a reading task whose answer is in
the state and where accuracy is 0.93, raw probabilities are already nearly calibrated (ECE 0.051, oracle T 2.9); applying
T=12 makes the model under-confident and ECE rises to 0.24. The author's conclusion: the distortion "is not a constant scale:
about 12x on closed-book knowledge questions and about 3x on questions answerable from the state. One post-hoc scalar does not
calibrate a general Decision API, and this is where the difference from Jev lies." On JevBench hard, the MMLU temperature is
partial: 0.274 to 0.107 at 32B, far from the 0.023 in distribution.

### 2.5 Trained heads and LoRA at three scales

Slot head (`z = W h + b` at the `Answer:` position, initialised from the letter rows so step 0 equals the readout) and pointer
head (bilinear score between the answer position and each option's last token, order-equivariant by construction), each with
LoRA r=16 on q/k/v/o, trained 600 steps x batch 8 = 4,800 MMLU auxiliary-train items with option order shuffled every time.

| | MMLU acc | notes |
|---|---:|---|
| 1.7B readout | 0.554 | |
| 1.7B slot + LoRA | 0.575 | +2.1, p=0.125 on MMLU; +3.9, p=0.006 on JMMLU. Raw ECE 0.41 to 0.14 |
| 1.7B pointer + LoRA | 0.561 | order effect remains (48% identical) because option representations depend on earlier options |
| 1.7B pointer, LLM frozen | 0.471 | -8.3 points: a new head on a frozen LM cannot read "option i is correct" out of the hidden state |
| 14B slot + LoRA | 0.757 | +0.7, p=0.50; four λ values all within noise |
| 32B slot + LoRA | 0.801 | -0.7, p=0.38 |

Training improves *raw* calibration at every scale (14B raw ECE 0.21 to 0.11), but after a temperature it equals readout plus
temperature. The slot head trained with shuffling has a uniform position prior (0.26/0.26/0.25/0.24) and raises identical
argmax under rotation from 48% to 55%.

### 2.6 CE + λ·Brier, or what "calibration-oriented training" buys

λ in {0, 0.5, 1, 2} at 1.7B and 14B: λ ≤ 1 is indistinguishable from plain CE on accuracy, NLL and ECE; λ = 2 flattens the
raw probabilities (raw ECE 0.137 to 0.082) at -3 points of accuracy at 1.7B, and after a temperature its NLL is worse than
λ = 0. Their sentence: "The Brier term does not add discrimination; it builds a temperature into training. That has value
when no calibration set is available, but if one temperature can be fitted on val, CE + T is better." The adopted setting for
14B and 32B is λ = 0. They cite NanoJev's framing approvingly: RL is the tool when only sampling is available; with logits and
ground truth, backpropagate the proper score directly. That is the same point section 3.2 of the Laya memo makes about
Laya's RLCD, arrived at independently and with numbers.

### 2.7 Few-shot in the shared state

Same-subject 5-shot examples in the state: ±0.4 points at 1.7B, +1.9 at 14B (p=0.12). Five *fixed* examples shared by every
question: -13.1 points at 1.7B (p<0.001), with predicted letters collapsing onto B at 64% (zero-shot 32%). Unrelated shared
examples steer the letter prior. The author's warning: in a one-state-many-questions setting, "the content and the
answer-letter bias of the examples can distort the decisions."

### 2.8 JevBench, and the comparison jqv publishes

The 32B zero-shot with the MMLU temperature was measured by Benchmark Heaven three times with reproducing public tiers
(easy 1.000, standard 0.958, judge 0.925 each time): through a tunnel to the author's Mac (v1.2.7, partial, unranked because
held-out items are not sent to submitter-operated endpoints), then on the maintainers' H100 from the published code
(v1.2.8: #8 of 36, score 70.1, Intelligence 86.1, Calibration 79.0, hard 0.645 on all 220 items against Jev's 0.741), and it
stands at #6 of 48 on v1.3.0 with score 68.6 after the board began measuring Intelligence above chance (I 79, C 79, S 75, K 47).
Held-out hard items scored slightly *above* the public ones (0.670 vs 0.622), so using the public tier as a development gate
did not inflate the result.

Where the 32B loses on the hard tier, by family (all 220): temporal_numeric 7/30, long_policy 19/38, probability 12/20,
against adversarial 12/12, trap 16/16, routing 9/10. Serving raw probabilities scores Calibration 45 at 32B and 0 at 1.7B;
the MMLU temperature alone lifts it to 78.6. "A Decision API only gets onto the board once it returns calibrated probabilities."

Their published related-work table rates closeness to jqv and states each project's own caveat: simple-jev (same
readout and KV reuse, "softmax values are not calibrated probabilities of correctness"), NanoJev (0.6B plus decision head,
CE/Brier/RLCD experiments in simulators where the true probability is known), SemIf ("not operationally calibrated the way
Jev's are"; jqv measured exactly the per-task temperature question SemIf's issue tracker raised), openjev-sglang (production
serving on SGLang; probabilities "conditioned on the supplied options, not a calibrated estimate of correctness"), mini-jev,
jev-forge, jevmlx, plus Hydragen and DeFT as the systems lineage. What jqv claims as its own: the engine decomposition with
equivalence tests, the isolation negative control, the same-test-set comparison of raw / temperature / LoRA / Brier with
paired statistics, and the temperature-transfer measurement.

### 2.9 What is next for them

Solver-generated synthetic items for the three weak families (rule engines, calendar arithmetic with time zones, exact
probabilities with `fractions`), 2,000/300/500 per family, hardened until 32B zero-shot accuracy on the synthetic dev sets
matched the JevBench family accuracies within 10 points, contamination-checked at zero shared 8-grams with the public
items, a language model used only to reword narrative paragraphs with every number and negation preserved. A targeted LoRA
task is written with a gate: 14B first, needs +10 points on two of three synthetic families at p<0.01 with MMLU within -1
point, and only then 32B. An RLCD-type outcome-based proper-scoring run is listed as the next candidate for cross-distribution
calibration.

## 3. What jqv settles that the two earlier memos left open

1. **SemIf's speed claim is right for the right reason.** Sharing the state is the speed-up; skipping generation is not.
   jqv's `packed` engine is the reference implementation of SemIf's `shared.py` with a numerical equivalence test and a
   leakage control, and it scales to Q=1000 in chunks.
2. **Laya's RLCD critique is confirmed by experiment.** CE plus a Brier term at λ ≤ 1 equals CE; λ = 2 is a built-in
   temperature that costs accuracy. Proper-scoring objectives do not fix cross-distribution calibration either.
3. **The residual gap to Jev is about 10 points at 32B and is not closed by heads, LoRA on 4,800 items, few-shot, order
   averaging or temperatures.** It lives in long rules, dates and numbers, and probabilities, which is consistent with
   Jev's post-training being large-scale and aimed at "read the rules and decide".
4. **Calibration is per task type, not per model.** The knowledge-vs-reading temperature difference (12x vs 3x at 1.7B) is the
   cleanest statement anyone has published of why one temperature cannot serve a general decision API.

## 4. What this means for the plan

- **Two regimes, two temperatures, and we have both.** jqv's knowledge-question temperature (T about 12 at 1.7B) versus
  reading-question temperature (T about 3) maps onto our record-in-weights arms versus record-in-prompt arms (REPORT 38, 43).
  Any calibration we report must be fitted per arm, and the record-in-prompt arm should be close to calibrated already.
  `results/per_item/` has the logits to check this without a GPU.
- **Our scale is where training helps.** LoRA plus head gives +2-4 points at 1.7B and nothing at 14B. Our 0.5B-3B categoriser
  is in the regime where the gain exists, which agrees with section 38's SFT lift; it also says not to expect that lift to
  survive a move to a larger backbone.
- **Never train a head on a frozen LM.** -8 points against the plain readout. If the representation has to move, it needs LoRA.
  Relevant to any "prototype head on frozen encoder" variant of the categoriser.
- **Measure order flips, and consider `perm_avg`.** 52% argmax flips under rotation at 1.7B on a 4-way MCQ with weak evidence.
  Our EVAL-1 discussion should add the rotation test; averaging K rotations is a training-free +3 at small scale, at 1.7x
  cost with a shared prefix. For a categoriser with tens of categories, K rotations is expensive, but shuffling at train
  time (which made the slot head's position prior uniform) is free.
- **Shared few-shot examples bias the letter prior.** Our k-shot categoriser prompts put the user's history in the state.
  jqv's -13 points from fixed shared examples came from letter collapse (B at 64%). If we read letters, check the letter
  distribution of the shots' answers; if we read category names, this does not apply.
- **Reuse, do not rewrite.** `jqv/engine/packed.py` is a working HF-transformers implementation of the shared-prefix block
  mask with CUDA verified by a third party; `jqv/calibration.py` is the temperature fit with provenance; `scripts/compare_runs.py`
  gives McNemar, bootstrap CIs and selective accuracy on the same items; `scripts/jevbench_run.py` drives the JevBench harness.
  All Apache-2.0, Python 3.12, transformers 5.x.
- **The task-file discipline is worth copying.** Each `tasks/*.md` has Goal, Context, Scope in/out, Success, Verify, and a
  filled-in Result with Changed / Verified / Deviations / Remaining, including a recorded case of the assistant stopping an
  evaluation early and re-running it after the owner objected. That is a tighter version of what `PLAN.md` does.

## 5. File map

| Where | What |
|---|---|
| `docs/report.md` | The full report; sections on engines, priors, calibration, heads, λ sweep, scaling, JevBench, synthetic data, related work |
| `jqv/prompt.py`, `readout.py`, `calibration.py`, `heads.py` | Prompt with hash, letter/rows readout and Jev confidence formula, temperature with provenance, slot/pointer heads |
| `jqv/engine/{naive,kvcache,packed,shared,head,generate}.py` | The five engines plus the trained-head engine |
| `scripts/eval.py`, `fit_temperature.py`, `transfer_temperature.py`, `permutation_test.py`, `isolation_test.py`, `compare_runs.py`, `train_head.py`, `jevbench_run.py` | The experiments, one script each |
| `results/scaling_table.md`, `jevbench_summary.md`, `synth_difficulty.md` | The tables quoted above |
| `results/jevbench/published_v1.2.{7,8}/` | Benchmark Heaven's published rows and the maintainers' comments |
| `jqv/synth/`, `data/synth/` | Solver-backed synthetic hard-family generators and their dev/test splits |
| `tasks/done/*.md`, `tasks/active/*.md` | Pre-registered task files with results and deviations |
