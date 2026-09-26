# How a "Jev"-like semantic decision model works, and how it is (and is not) trained: the SemIf repository

Series index and synthesis: `jev_meta_analysis.md`.

Date: 2026-09-21. Written after the owner asked how a Jev-like model works and how it is trained, with
https://github.com/TheoLeeCJ/SemIf (formerly OpenJev, commit `1f2dea3`, cloned to
`~/projects/Taytay/SemIf`) as the object of study. Sources: the SemIf code (`src/semif_phase1/`), its
`docs/METHOD.md`, `docs/RESULTS.md`, `docs/CALIBRATION.md`, `benchmarks/`, `webgpu-demo/`,
`exl3-bridge/`; TypeSafe's launch post
(https://typesafe.ai/blog/introducing-system-one-models-and-jev); two secondary write-ups on RLCD
(https://www.mindstudio.ai/blog/typesafe-jev-rlcd-vs-rlhf, https://www.turingpost.com/p/what-is-jev-rlcd);
a community reference for the Jev API (https://gist.github.com/pjburnhill/adf8d28efcad9df037bfdece178ef965).
Related memos: `task_training_and_services.md` (task-direct training and RL, 2026-09-17); `laya_analysis.md` (the open encoder-based decision model that does train for this, same day); `jqv_analysis.md` (the stock-Qwen3 reconstruction with the engine ablations and calibration-transfer measurements, 2026-09-22).

## 0. The one-paragraph answer

Jev is TypeSafe's closed, hosted "System One" model: unstructured state in, typed decisions with
probabilities out, no generated text. TypeSafe says it is a new architecture with a parallel sampler,
trained with a method they call Reinforcement Learning for Calibrated Decisions (RLCD). Nothing beyond
those phrases is public: no architecture, no training data, no algorithm, no weights. **SemIf trains
nothing.** It reproduces the *interface* with a frozen open chat model (Qwen3.5-4B) by doing one forward
pass over a prompt that lists the options as letters, reading the logits of the letter tokens at the last
position, and softmaxing over just those. Its speed comes from KV-cache reuse (prefill the state once,
run many criteria as parallel suffix branches), and its "calibration" is a post-hoc per-workload
temperature fitted on labelled rows. The repo's own results doc says the next phase would be targeted
training for decision semantics and calibration, and that it has not been done.

So the honest split is: **how it works** is fully answerable from the code; **how it is trained** has two
answers: Jev's training is undisclosed, and SemIf's is "not at all, yet". Section 4 gives the plausible
recipe, labelled as inference, and section 5 what it means for this repo.

## 1. What the interface is

One decision is a JSON row (`examples/decisions.jsonl`):

```json
{"id": "route-1",
 "state": "Customer asks to reset a forgotten password and says the reset email never arrived.",
 "question": "Which queue should handle this request?",
 "options": [{"id": "account_access", "description": "Account access and authentication support."},
             {"id": "billing", "description": "Billing and payment support."},
             {"id": "sales", "description": "Sales and product evaluation."}]}
```

`state` may be a string or any JSON object/array. Two to sixteen options. The output is one probability
per option plus the raw option logits, the prompt hash, the model revision and timings, and a fixed
`probability_status` string: "conditional option score; uncalibrated as decision confidence".

Jev's public API has three primitives that this maps onto: **Noul** (probability a yes/no proposition is
true), **Choice** (one of up to 255 options, returns the winner, a probability per option and a derived
confidence), **Score** (a position on 2-10 ordered levels, probability-weighted). SemIf covers Choice
and, with two options, Noul. It does not implement Score, and it is capped at 16 options because each
option must be one single-token letter (section 2.1).

## 2. How SemIf works: direct option-logit readout

### 2.1 The prompt and the readout (`core.py`, `direct.py`)

The system prompt is fixed:

> Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with
> only its uppercase letter, with no explanation or reasoning.

The user turn is one JSON object `{"evidence": <state>, "criterion": <question>, "options": [{"letter":
"A", "description": ...}, ...]}`, rendered with the model's chat template, `add_generation_prompt=True`
and `enable_thinking=False`. Nothing is decoded. One forward pass with `logits_to_keep=1` returns the
full-vocabulary logits at the last position; the scorer indexes the token ids of `A`, `B`, ... for as
many options as there are and softmaxes those alone. That is the whole "model".

Two checks make it a contract rather than a hope. Each letter must encode to exactly one token that
round-trips on decode, and appending the letter to the rendered prompt must tokenise to `prompt_ids +
[letter_id]`, so the answer slot is not merged into a preceding token. Over-length prompts are refused,
never truncated. The serial scorer also records `allowed_token_mass` (how much of the full softmax the
letters capture) and `full_vocab_argmax_id` (whether the model's actual top token was a letter at all),
which is the diagnostic this repo's EVAL-1 discussion wanted for letter-vs-text scoring.

This is exactly the "MCF / symbol" scorer of Robinson & Wingate and OLMES (`references/SURVEY.md`,
EVAL-1): score the label symbol, not the option text, so option length and surface form drop out. Our
`scoring.py` already computes it as `mcf_lp`. SemIf's contribution is not the scoring rule but treating
it as the *product*: runtime-defined options, structured state, and a serving path built around it.

### 2.2 The speed trick: one state, many criteria (`serial.py`, `shared.py`)

Jev's demonstrations are "one document, N questions in parallel". SemIf reproduces the shape with the
model's own KV cache:

1. **State prefix.** The prompt is rendered with a placeholder question and yes/no options, then cut at
   the end of the serialised `{"evidence": <state>` fragment (dropping the final token so that JSON
   punctuation cannot merge across the cut). Those token ids are the prefix. Every real prompt for the
   same state must start with exactly these ids, or the scorer refuses.
2. **Prefill once.** One forward over the prefix with `use_cache=True` yields the KV cache.
3. **Serial mode** deep-copies the cache per decision and forwards just the suffix (criterion + options +
   assistant header), reading the last position. A cache miss (new state) re-prefills.
4. **Parallel mode** calls `cache.reorder_cache(zeros(N))` to replicate the single cache row into N
   batch rows, right-pads the N suffixes, sets `position_ids` to continue from the prefix length, and
   runs one batched forward with `logits_to_keep` pointing at each row's own last real token. All N
   distributions come back from one call. On MPS it loops batch-1 instead because that is faster there.

On the owned 37-state x 21-criterion fixture (states ~8k characters) on one RTX 3090, BF16:

| Path | Decisions/s | Argmax drift vs fresh |
|---|---:|---:|
| Fresh direct, batch 1 | 2.33 | reference |
| Serial prefix reuse | 10.75 | 5/777 |
| Parallel suffixes | 20.03 | 6/777 |
| Qwen3-Reranker-4B, pair batch 1 | 1.86 | reference |

And the headline "no output tokens" comparison, same model, one state, 21 binary criteria: parallel
readout 1.02 s median with 0 output tokens; asking the model to generate a compact JSON array of 21
`yes`/`no` strings took 5.33 s and 111 tokens (0.49 s to first token). The two readouts agreed on 18/21,
so the repo is careful to call it a systems comparison, not an equivalence.

The drift numbers matter for us: 5-6 of 777 argmaxes change purely from BF16 batch geometry, and the
reranker's changed by 51-54 of 777 between pair-batch sizes. That is the same batch-shape noise REPORT
section 19 measured on our saved weights, and the same reason section 44 found a bf16/4-bit mismatch
changing a fifth of predictions. Serving configuration is part of the evaluated system.

### 2.3 The control: a native reranker (`reranker.py`)

The second system is Qwen3-Reranker-4B used on its own contract: for each option, a
(query = question + candidate answer, document = state) pair, `logit(yes) - logit(no)` at the last
position, then a softmax of those log-odds across options. It is the "does this option follow from the
evidence" formulation, order-invariant by construction and strong at retrieval ranking (MRR 1.0 on both
Every retrieval tasks), but 12-19 points worse than direct logits on general decisions and 2 to 5 times
slower because the state is re-read once per option. The repo's verdict: a generic reranker fine-tune
"would answer the wrong question".

### 2.4 Calibration is post-hoc temperature scaling (`docs/CALIBRATION.md`, `benchmarks/calibrate.py`)

Jev's central promise is that the probabilities are calibrated. SemIf's are explicitly not, so it adds
one scalar temperature `T` per workload, fitted by minimising NLL on labelled rows, applied as
`softmax(option_logits / T)`. Monotone, so the argmax never moves; only confidence changes. Fitted on
the three labelled workloads, out-of-fold ECE under group-disjoint 5-fold CV:

| Workload | Rows | Accuracy | T | ECE raw | ECE calibrated |
|---|---:|---:|---:|---:|---:|
| Authored | 144 | 0.806 | 1.23 | 0.068 | 0.038 |
| WANLI (NLI) | 256 | 0.637 | 2.50 | 0.208 | 0.069 |
| Every judgments | 154 | 0.942 | 1.71 | 0.050 | 0.047 |

Only WANLI's improvement is interval-separated. The temperatures differ by a factor of two between
workloads, which is the repo's argument for fitting per deployment rather than once. The negative
control (one pooled temperature) was not shown to be worse at these sample sizes. Per-class calibrators
(Platt, vector, matrix scaling) do not apply because the option set is runtime-defined and variable.

### 2.5 Quality against the published Jev numbers

On the 102 TypeSafe public evaluation rows that could be aligned (20 cases), equal-case modal agreement
with the published reference distributions:

| System | Agreement | TV distance to target distribution |
|---|---:|---:|
| Qwen3-0.6B direct | 0.407 | |
| MiniCPM5-2B direct | 0.637 | |
| Qwen3.5-4B direct | 0.845 | 0.177 |
| Qwen3-Reranker-4B | 0.560 | 0.444 |
| Published Jev | 0.883 | 0.127 |

The 27B EXL3 bridge (Qwen3.8-27B at 5 bpw) took the 144 authored rows from 0.813 to 0.958 balanced
accuracy, with everything (family, size, quant, runtime) differing, so it is a ceiling hint, not an
ablation. Robustness: reversing option order flipped 10 of 36 direct decisions, wrapping the criterion
flipped 9, appending irrelevant context flipped 4. The direct readout has the positional bias PriDe
(`2309.03882`) describes; the reranker does not, but is wrong more often.

### 2.6 The browser and CPU paths

The same readout runs anywhere the model runs: wllama in WebGPU gets per-token log-probs of the letter
tokens with a one-token, `logit_bias`-constrained completion; llama.cpp on CPU from GGUF; exllamav3 from
EXL3. Prompt hashes match the Torch backend row for row, so the contract is the prompt and the readout,
not the runtime.

## 3. How Jev is trained, as far as anyone outside TypeSafe knows

**Disclosed, verbatim from the launch post:** "a new model architecture, parallel sampler for maximum
efficiency, and training method we call Reinforcement Learning for Calibrated Decisions (RLCD)"; it
optimises for "calibrated decisions: answers with epistemically honest probabilities on System One
tasks", contrasted with RLHF (human preference) and RLVR (verifiable rewards); it "generates all outputs
in a single query" rather than one token at a time; outputs are "type-safe structured values" so it
"can't hallucinate" in the schema sense; "calibrated: higher confidence means higher accuracy". Pricing
is quoted as $0.042 per million input tokens with output "too cheap to meter", latency 70-500 ms.

**Not disclosed:** the post's FAQ has the headings "Why was a new training algorithm needed?" and
"Where does our training data come from?" without answers. No paper, no model card, no parameter count,
no statement of whether the backbone is a transformer, what "parallel sampler" means mechanically, or
what RLCD's reward is. Every number is TypeSafe's own; the service is behind a waitlist. SemIf's results
doc lists "RLCD training, because neither the training data nor a sufficient algorithmic specification
is public" under *not reproduced*.

**What the secondary sources add** is interpretation, not information: RLCD "trains on outcome
accuracy... rewarding the model for producing a confidence score that actually matches how often it's
right"; the community reference restates calibration as "groups of predictions carrying higher
probabilities should prove correct more frequently than groups with lower probabilities" and warns that
calibration "does not mean that an individual prediction is guaranteed to be correct".

Two things can be inferred with reasonable confidence from the interface alone. First, the model must
read the state once and emit a probability distribution over a runtime-declared set for each of several
independent questions, which is what SemIf's parallel-suffix mode does with a stock transformer; a
purpose-built architecture would put the question/option encodings in a position to attend to a shared
state encoding without the letter-token detour. Second, "up to 255 options" and Score's ordered levels
mean the readout head is not literally 16 single-token letters; it is some pointer or per-option scoring
head, which the reranker's per-option yes/no is the crude open analogue of.

## 4. How one would plausibly train such a model (inference, not documentation)

Nothing here is from TypeSafe. It is what the ingredients in the survey and the `task_training_and_services.md`
memo add up to, written down so the plan can decide whether to try any of it.

1. **Supervised stage: symbol/letter choice on decision data.** Build (state, question, options, gold)
   rows across many domains, render them in SemIf's exact prompt, and train with cross-entropy restricted
   to the option-letter logits at the answer position (or unrestricted next-token loss on the single
   letter, which is the same thing up to the non-letter mass). This is symbol tuning (`2305.08298`) with
   natural-language option descriptions and randomised letter assignment, and it is what fine-tuned
   in-context learners (`2512.19879`) do at k-shot. Randomising option order per epoch is the direct
   fix for the 10/36 reversal flips. Cross-entropy on the correct option is already a proper scoring
   rule, so this stage produces calibrated probabilities *on the training distribution*; the
   overconfidence SemIf measured on WANLI (T = 2.5) is distribution shift plus the fact that Qwen's
   post-training optimised something else.

2. **Calibration stage, the thing "RLCD" most plausibly names.** Once the model emits distributions, a
   reward that is a proper scoring rule on the *outcome* (log-loss or Brier of the emitted probability
   against the realised label) rewards honest probabilities and nothing else. Run as RL rather than SFT
   when the labels are outcomes observed after the decision (was the escalation warranted, did the
   retried job succeed) and when the decision distribution itself is what is being sampled, so the
   reward is on-policy. A GRPO-style group of decisions per state with a Brier reward is the obvious
   open recipe; the anchored-advantage and dead-zone fixes from the qorl post apply when most groups are
   trivially right. This differs from RLHF (a learned preference reward) and from RLVR (a 0/1 verifier on
   generated text) exactly in the way the launch post contrasts them.

3. **Multi-question parallel readout.** Train on prompts that carry one state and several questions
   with several answer positions, loss on every answer position, so the model learns to answer
   independently from a shared prefix; that is what the parallel-suffix trick simulates at inference,
   and ABFT (`2505.14233`) shows the attention circuitry for this kind of readout can be trained in a
   few hundred steps.

4. **Abstention as an option.** SemIf's authored fixture includes `insufficient` as a listed option and
   a 36-row missing-evidence population; each system still made one confident wrong non-abstention. The
   IDK-label finding in `2405.05904` says the abstain option has to be trained, not just listed.

What is *not* needed is what this repo spent its first 30 sections on: injecting facts. A Jev-like
model is a reader, not a memory. It answers from the supplied state, which is why "can't hallucinate"
means schema safety and nothing about correctness.

## 5. What this means for the plan

- **The categoriser is a Choice primitive.** REAL-5/REAL-6's task (transaction string + user history or
  fact-DB record in, category out) is exactly `state` + `question` + options. Section 38 and 43 already
  found that the record *in the prompt* beats the record in the weights (98 vs 51-72). SemIf's
  direct-logit readout is a cheaper, generation-free version of what the record-in-prompt arm does, and
  its `mcf_lp` is already in `scoring.py`. Worth checking whether our categoriser reads category letters
  or generates category names, and whether the letter readout with randomised order changes REAL-5.
- **The prefix-cache geometry is inverted for us.** SemIf shares a long state across many criteria. Our
  workload shares a long fixed prefix (category list, field guide, fact-DB rendering) across many short
  transactions. The same `reorder_cache` replication applies with the roles swapped, and it is the
  mechanism that would make the record-in-prompt arm affordable at production volume.
- **Calibration is cheap to measure and we have the records.** `results/per_item/` holds every option's
  log-prob for every run. Fitting one temperature per arm and reporting ECE with group bootstrap, as
  `benchmarks/calibrate.py` does, is a CPU job on data we already have, and it turns the accuracy tables
  into an auto-apply-versus-review operating point, which REAL-11's correction-rate requirement wants.
- **Option-order flips are a real effect at 4B and need the PriDe-style check** (EVAL-1 notes position bias as the cost of letter scoring). SemIf lost
  10/36 to reversal at 0.81 accuracy. Our sections report accuracy on a fixed order.
- **Batch shape and dtype change argmaxes.** SemIf treats serving config as part of the system and pins
  it in every output row. We learned the same in sections 19 and 44; the tracker should record batch
  geometry and dtype per run if it does not already.
- **Training the reader, not the memory, is the direction section 4 points at**, and it is the same
  direction as the `task_training_and_services.md` memo: task-direct SFT plus a calibration-shaped RL
  stage, on a 3090 for the SFT and rented H100s if the RL stage is ever justified. SemIf's frozen
  baselines (authored144, WANLI256, the 102 TypeSafe rows) are a ready-made external gold set for judging
  whether such training generalises beyond our synthetic universe.

## 6. Repository map, for going back to it

| Path | What it is |
|---|---|
| `src/semif_phase1/core.py` | Row validation, the fixed prompt, softmax, pinned model loading |
| `src/semif_phase1/direct.py` | Fresh single-forward letter-logit readout; slot and boundary checks |
| `src/semif_phase1/serial.py` | One-state KV cache, deep-copied per decision; records allowed-token mass |
| `src/semif_phase1/shared.py` | Parallel suffix branches over a replicated cache |
| `src/semif_phase1/reranker.py` | Qwen3-Reranker yes/no per option, softmax of log-odds |
| `src/semif_phase1/{llamacpp,mlx}_backend.py` | Same contract on CPU GGUF and Apple MLX |
| `benchmarks/calibrate.py` | Temperature fit, out-of-fold ECE, calibrated prediction files |
| `benchmarks/shape777.py`, `decision_vs_generation.py` | The speed fixtures |
| `benchmarks/data/authored144.jsonl`, `perturbations108.jsonl` | Owned labelled decisions and their variants |
| `benchmarks/manifests/source-selection.jsonl` | Exact WANLI / TypeSafe / Every row ids |
| `docs/METHOD.md`, `RESULTS.md`, `CALIBRATION.md` | The claims and their boundaries |
| `exl3-bridge/`, `webgpu-demo/` | 27B quantised and in-browser runners on the same prompt hash |

Pinned models: Qwen/Qwen3.5-4B `851bf6e8`, Qwen/Qwen3-Reranker-4B `22e68366`, turboderp/Qwen3.8-27B-exl3
`a35e75a7`. MIT code; model licences upstream.
