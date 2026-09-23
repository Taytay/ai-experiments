# Laya: an open encoder-based "System One" decision model, and what its RLCD training actually is

Series index and synthesis: `jev_meta_analysis.md`.

Date: 2026-09-21. Companion to `semif_jev_analysis.md` (same day) and `jqv_analysis.md` (2026-09-22, which confirms section 3.2 below experimentally). Object of study:
https://huggingface.co/convaiinnovations/laya (revision `1c5edc17`, 2026-09-20) and its code at
https://github.com/NandhaKishorM/laya (v0.3.5, commit `573e5b6`). Local copies: the GitHub repo at
`~/projects/Taytay/laya`, and the model repo's code and configs without weights at `~/projects/Taytay/laya-hf`
(`rl_common.py`, `rl_agent_api.py`, the three `rl_agent_config.json`, `eval/results.*`). Other sources: the author's
Dev.to post (https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me),
the `LocalLLaMA/typed-decisions` dataset card, the nibzard decision-model benchmark (source of the third-party
Jev numbers), and the abstracts of the author's two earlier papers (arXiv 2503.23303, 2510.01237); the full Dev.to text was read
on 2026-09-22 after a first pass through a summary only, and section 4.5 was added then.

## 0. The one-paragraph answer

Laya is the thing SemIf said would be "the next justified phase": a model actually trained for typed decisions.
It is not an LLM. It is ModernBERT-large (395M, bidirectional, fully fine-tuned) with a two-layer transformer
head, a per-option scorer and an act/escalate head, 421M parameters in total; a multilingual sibling uses
mmBERT-base (322M). The input is one sequence: `[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK]
opt1 ... [SEP] <state> [SEP]`. Each option is preceded by its own `[MASK]` token; the hidden state at each mask is
scored by an MLP to one logit; softmax over the question's options is the answer. One forward pass per question,
questions batched, no decoding, about 33-40 ms per question on a T4. The training method, called RLCD like
TypeSafe's, is in the code: sample G Gaussian perturbations of the model's option logits, reward each with a
strictly proper scoring rule (log score + spherical score, plus ranked probability score for ordinal questions),
REINFORCE with a group-mean baseline, plus a plain soft-target cross-entropy term, then post-hoc temperature
per (question type, option count). Section 3 argues that because the reward is differentiable in the logits,
this RL is a noisy zeroth-order estimate of a gradient one could take directly; the substantive choices are the
soft targets, the spherical and RPS terms, option-order shuffling, and the abstain head. Base checkpoints are
near chance on the typed-decisions benchmark zero-shot (0.36 vs 0.46 majority class); the 0.766 that beats
Jev's published 0.727 comes from a 5-minute fine-tune on that benchmark's own training split, on two T4s.

## 1. Provenance and claims

The author (Nandakishor M, Convai Innovations) published SalesRLAgent in March 2025 (arXiv 2503.23303: a
specialised RL model predicting sales-conversation conversion probability turn by turn, Azure embeddings, 85 ms
vs 3.4 s for GPT-4) and a confidence-aware routing paper in September 2025 (2510.01237). Neither is the Laya
architecture; the first is the lineage of Laya's "conversation outcomes" task and its TD(lambda) prefix training.
Laya itself was released after TypeSafe's Jev launch in September 2026 as the open answer to it, Apache 2.0,
with weights, a `pip install laya` package, a fine-tuning notebook, and benchmark harnesses.

Every Jev number Laya quotes is third-party. The repo states: "There is no TypeSafe API credential in this
project, so Jev was never run here." The sources are AbdelStark/jev-benchmarks (AG News 0.910, Banking77 0.870,
DAIR Emotion 0.480 with zero probability on the true label for 16% of examples) and nibzard/decision-model-benchmark
(Jev accessed with a real key: banking77 0.763, ECE 0.246 "the worst calibration error measured", 13%
option-order flip rate, p50 264-276 ms, $0.07 per 1,000 decisions, hard cap at 255 options). The
typed-decisions Jev figures (0.727 accuracy, soft accuracy 0.580, Brier 0.148, ECE 0.144) are from that
dataset's own published baseline table.

## 2. How it works

### 2.1 Sequence construction (`rl_common.build_sequence`)

Three question types map to option lists (`render_options`): `choice` renders `"<key>: <description>"` per
option; `score` renders `"level i: <rubric text>"`; `noul` is always `["false: ...", "true: ..."]` so `p[1]` is
the probability the statement holds. The head of the sequence is `"<type> question: <instructions>"`, then
`[SEP]`, then for each option `[MASK]` followed by up to 48 option tokens, then `[SEP]`, then the state (a string
or JSON-serialised object), then `[SEP]`. Budgets: `max_len` 512 (English) or 1024 (multilingual,
typed-decisions), of which `head_max_len` 192 or 256 is for instructions plus options. When options do not fit,
every option is truncated evenly to `(head_max_len - 16) / K` tokens. That is the documented cause of the
Banking77 failure: 77 labels get 3-4 tokens each and both checkpoints score exactly 0.425 against Jev's 0.870.
The state is truncated to the remaining room (left-truncated for conversation prefixes).

At inference every question in a request becomes its own sequence with the same state, and they are collated into
one batch: "All questions answered in single forward pass" means a batch, not shared computation. The state is
re-encoded once per question. That is the opposite of SemIf's prefix-cache trick and is the reason Laya's per-call
time grows nearly linearly with questions on the English model (39.5 ms for 1, 158.6 ms for 10, 771 ms for 50).

### 2.2 Model (`DecisionModel`)

```
h = encoder(ids).last_hidden_state                # ModernBERT-large, 28 layers, d=1024
h = h + type_emb[qtype]                           # one learned vector per question type
h = 2 x TransformerEncoderLayer(d, 16 heads, 4d)  # the "decision head", trained from scratch
m = gather(h, marker_pos)                         # [N, K, d]: hidden state at each option's [MASK]
logits = MLP(LayerNorm -> Linear -> GELU -> Linear(1))(m)   # [N, K], padding masked to -1e4
p = softmax(logits / T[type or (type, K-bucket)])
```

The act/escalate head takes the pooled `[CLS]` state concatenated with four detached statistics of the answer
distribution (top-1 probability, top-1 minus top-2 margin, normalised entropy, K/255) through a 256-unit MLP to
two logits, act or escalate. Reported confidence is Jev-style `1 - H(p) / log K`. `score` answers are the
probability-weighted level index, so a score can sit between levels.

This is the encoder-side cousin of ModernBERT-Large-Instruct's answer-token prediction (`2502.03793`, our
MODEL-2 thread): one `[MASK]` per candidate instead of one mask predicting a vocabulary token, and a trained
scalar scorer instead of the MLM head. It also generalises our encoder prototype categoriser (REPORT 36-38): the
category descriptions are in the input as options, so the option set is runtime-defined and there is no per-user
head to fit.

### 2.3 What ships

| Checkpoint | Backbone | Params | Context / head | Training record in config |
|---|---|---|---|---|
| `laya` (root) | ModernBERT-large | 421M | 512 / 192 | 7,313 updates, 1 epoch, 1.96 h, one GPU, "fine_tuned_from_checkpoint" |
| `laya-multilingual` | mmBERT-base | 322M | 1024 / 256 | 15,987 updates, 4 epochs, 4.97 h, one GPU, temperatures all 1.0 |
| `laya-typed-decisions` | ModernBERT-large | 421M | 1024 / 256 | the notebook run: 6,000 items, 4 epochs, 2 x T4, 4-6 minutes |

The root config carries per-bucket temperatures (`choice:2` 1.91, `choice:3-5` 1.76, `choice:6-10` 1.00,
`choice:11+` 0.10, `score:3-5` 1.25, `noul:2` 1.98). The `choice:11+` value of 0.10 sharpens rather than
softens, which is a fitted artefact worth noting before trusting high-cardinality confidences. The multilingual
checkpoint ships with no fitted temperatures at all.

## 3. How it is trained: RLCD as implemented

The base-model training script is not published; what is published is `rl_common.py` (model, rewards, record
encoding, TD(lambda) targets, batching) and the fine-tuning notebook, whose loop is the same recipe. The Dev.to
post gives the base-run hyperparameters.

### 3.1 The update (from the notebook, `train_ddp.py`)

For a micro-batch of sequences with logits `z0` of shape `[N, K]`:

1. **Explore.** Draw `G` Gaussian perturbations `eps ~ N(0, sigma^2)` per option, project each to zero mean across
   the question's options, and form `z_g = z0.detach() + eps_g`, `q_g = softmax(z_g)`. Notebook: G = 4, sigma
   decays 0.4 to 0.1 over epochs. Post (base run): G = 8, sigma 1.0 to 0.3.
2. **Reward** each `q_g` with `proper_reward`:
   `R = sum_y t_y log q_y  +  w_sph * <t, q> / ||q||  -  w_rps * RPS(q, t) [score questions only]`,
   where `t` is the (possibly soft) target distribution, the log term is floored at -9.21, `w_sph` is 0.5 in
   the library default and 0.75 in the notebook, and RPS is the mean squared difference of the cumulative
   distributions. All three are strictly proper, so the expected reward is maximised only at `q = t`.
3. **Advantage** = reward minus the group mean over the G samples, normalised by the batch std (GRPO-style,
   no critic).
4. **Policy gradient.** The "policy" is the Gaussian centred on the model's logits, so
   `log pi(z_g | z0) = -||z_g - z0||^2 / (2 sigma^2)` and `loss_rl = -mean(adv * log pi)`.
5. **Plus supervised cross-entropy** to the same soft target, weight 1.0: `loss = loss_rl + loss_ce`. The Dev.to
   post says the base model was trained "with pure policy gradient (zero supervised cross-entropy loss)"; the base
   script is unpublished, so that cannot be checked. The only published loop, the fine-tune, uses the CE term.
6. AdamW, encoder LR 2.5e-5, head LR 1e-4, weight decay 0.01, cosine schedule, grad clip 1.0, fp16 autocast,
   gradient checkpointing, effective batch 64 sequences.
7. **Post-hoc temperature** per question type (L-BFGS on NLL over 400 held-in items) written into the config.

### 3.2 What the RL is doing, mechanically

The gradient of `log pi(z_g | z0)` with respect to `z0` is `(z_g - z0) / sigma^2 = eps_g / sigma^2`. So the
policy-gradient term pushes the logits by `sum_g adv_g * eps_g / sigma^2`: the direction in logit space along
which the proper score improved. That is the evolution-strategies (natural evolution strategies / REINFORCE on a
Gaussian) estimator of `d R / d z0`, and `R` is a smooth, closed-form function of `z0`. Its exact gradient is
available by backprop: for the log-score term it is precisely the soft cross-entropy gradient `q - t`, which the
`loss_ce` term already supplies. So RLCD here equals cross-entropy plus a Monte-Carlo estimate of the gradient of
the spherical and RPS terms, with zero-mean noise, on a reward that needed no sampling to differentiate.

That is not a criticism of the objective, only of the framing. The objective's substantive content is:

- **Soft targets.** The typed-decisions gold is the mean of three teacher samples; `encode_record` accepts
  `q["soft"]`. Matching a distribution rather than a one-hot label is what makes "calibrated" a training target.
  (Laya still trails Jev on soft accuracy, 0.471 vs 0.580, while winning argmax accuracy.)
- **The spherical score.** Bounded in [0, 1], it rewards probability mass on the truth without the unbounded
  penalty of the log score for confident misses; the 0.5-0.75 weight is a regulariser against the log score's
  tail.
- **RPS for ordinal questions.** Being one level off costs less than being three off; plain cross-entropy is
  blind to level order. `score` is still the weakest primitive (SST-5 0.372).
- **Option-order shuffling at train time** (`encode_record`, non-score questions), the direct counter to the
  position bias SemIf measured on Qwen and nibzard measured on Jev (13%). Laya's own flip rate at 20 options is
  0.15-0.23, so the shuffle is necessary but not sufficient.
- **The act/escalate head** trained with a cost matrix (+1 correct act, -3 wrong act, -0.5 escalate), which makes
  acting optimal only above P(correct) = 0.625. In the fine-tune notebook this head is frozen (`0.0 * act.sum()`),
  and the shipped eval shows `automation_rate 1.0`, so the head as shipped never escalates on in-task data.
- **TD(lambda = 1.0) over conversation prefixes** for multi-turn outcome questions: each prefix of a
  conversation is trained against the terminal outcome, which is Monte-Carlo return, not bootstrapping.

Where RL would earn its keep is when the reward is *not* differentiable in the logits: outcomes observed after
acting, costs of escalation in a live queue, non-decomposable metrics. The code structure (reward function
separate from the model, group baseline) is set up for that; the shipped runs do not use it.

### 3.3 Data

Base: "100% human-labeled, real-world public datasets" across the 13 in-task families listed in
`eval/results.md` (email triage, intent and routing, moderation, NLI and fact checking, reading comprehension,
response-quality scoring, search relevance, sentiment, topic, conversation outcomes, and others), with dynamic
augmentation: shuffled option orders, paraphrased questions, JSON-vs-text state alternation. Exact datasets and
mix are not published. Zero-shot families held out of training: emotion, instruction following, moderation,
sentiment (0.651 overall, ECE 0.204).

Fine-tune: `LocalLLaMA/typed-decisions`, 1,200 train / 400 test cases, four workflows (agent-trace
observability, customer service, invoice processing, security incidents), five questions per case over one
state, gold as the mean of three teacher samples at temperature 0.7, teacher self-agreement ceiling 0.735.
Synthetic states from latent skeletons, rendered by a model. The benchmark is independent of TypeSafe but
follows the Jev request shape exactly.

## 4. Laya against Jev on English tasks, like for like

Three things blur the comparison and have to be separated: whether Laya was fine-tuned on the task, whether the
task family was in Laya's base training mix, and whether Jev's number comes from the same rows. Jev was never run
by Laya's author; every Jev figure is from a third party who did have API access (AbdelStark, nibzard, or the
typed-decisions dataset's own baseline table), on their own sample of the same public dataset. So the numbers
below are same-dataset, not same-rows.

### 4.1 Zero-shot, English: what each model does on a task it was not fine-tuned for

| Task (English) | Options | Jev, third-party | Laya base, zero-shot | Laya's exposure to the family | Who wins |
|---|---:|---:|---:|---|---|
| typed-decisions, 2,000 decisions, 4 workflows | 2-6 | **0.727** | 0.362 (majority class 0.461, random 0.318) | none; below majority class | Jev by 36 points |
| Banking77 intent | 77 | **0.870** (AbdelStark), 0.763 (nibzard) | 0.425 | intent was a training family; fails on option budget | Jev by 34-45 points |
| AG News topic | 4 | 0.910 | **0.950** | topic classification was a training family | Laya by 4 points, in-distribution |
| DAIR Emotion | 6 | 0.480 | **0.595** | emotion was a held-out family (zero-shot eval 0.583) | Laya by 11 points, genuinely zero-shot |

Read across the row: Jev is the stronger general zero-shot English decision model. It wins the one benchmark
built to look like real workflow decisions, by a wide margin, and it wins high-cardinality label sets outright.
Laya wins on two small-label-set classification datasets, one of which is the kind of task it was trained on.
DAIR Emotion is the single clean zero-shot win for Laya, and it is a 6-way sentiment-style task where Jev is
unusually weak (Jev put zero probability on the true label for 16% of examples there).

### 4.2 After fine-tuning Laya on the task's own training data

| typed-decisions test, 2,000 decisions | Jev 1.13.0, zero-shot | Laya fine-tuned on the 6,000-decision train split | Teacher self-agreement ceiling |
|---|---:|---:|---:|
| Argmax accuracy | 0.727 | **0.766** | 0.735 |
| Soft accuracy (agreement with the gold distribution) | **0.580** | 0.471 | |
| Brier | 0.148 | **0.062** | |
| ECE | **0.144** | 0.213 | |
| Score MAE (ordinal questions) | 0.391 | **0.242** | |
| Per-workflow accuracy | | invoice 0.804, security 0.766, customer service 0.764, agent trace 0.730 | |

Five minutes on two T4s takes Laya from 36 points behind Jev to 4 points ahead on argmax accuracy, and the
fine-tuned model matches the gold distributions worse (soft accuracy, ECE) while getting the hard label right
more often. That is the shape of a specialist: it has learned this benchmark's label prior.

### 4.3 Calibration, order stability and speed are not measured on the same data

| | Jev | Laya | Comparable? |
|---|---|---|---|
| ECE | 0.246 on nibzard's banking77 / spam / code-word suites | 0.081 after per-bucket temperature refit on Laya's own suites; 0.466 as shipped | No: different tasks, and Laya's number is post-refit on held-out data from the same distribution |
| Option-order flip rate | 0.13 (nibzard, permuted banking77) | 0.15 at 20 options (MASSIVE intent, English); 0.04 emotion; 0.00 XNLI | Roughly: Laya is no more order-stable than Jev at 20 options |
| p50 latency, one question | 236-276 ms through the hosted API | 32.8-39.5 ms local on a T4 | No: Jev's includes network and service overhead; Laya's is a warm local forward |
| Options supported | up to 255, sharp cap at 256 | practical ceiling ~20 at default budgets; raise `head_max_len` or use the embedding shortlist beyond that | |

### 4.4 The verdict in one place

On English tasks the honest reading is: **Jev is the better zero-shot decision model; Laya is a fast, open,
fine-tunable specialist.** Where a few thousand labelled decisions from the target distribution exist, Laya can be
fine-tuned past Jev's argmax accuracy in minutes, and it then runs locally at a few milliseconds per question with
no per-call cost. Where the task is new, the option set is large, or the probabilities themselves must be trusted
without a local refit, Jev is ahead on every published number. Laya's own in-task evaluation supports the same
split: 0.991 on intent and routing and 0.967 on moderation for families it trained on, 0.651 overall on families
it did not, with ECE rising from 0.030 to 0.204.

For the SemIf comparison in the companion memo: on TypeSafe's own 102 public rows, frozen Qwen3.5-4B with
letter-logit readout reached 0.845 modal agreement to Jev's 0.883, with no training at all. Laya has not been run
on those rows. A 4B chat model with option readout is therefore a closer zero-shot match to Jev on English than
Laya's 421M encoder is; the encoder's advantage is training cost and latency, not zero-shot breadth.

### 4.5 The Dev.to post's head-to-head table, read against the model repo

The post (2026-09-18, full text saved as `~/projects/Taytay/laya-hf/devto_post_2026-09-18.md`) leads with a table whose accuracy
row is "TypeSafe Jev 67.8% across 4 production workflows" against "Laya 83.8% in-task macro accuracy", headlined
as "+16.0% higher overall accuracy". The two numbers are from different benchmarks:

- **67.8%** is TypeSafe's own internal four-workflow benchmark (security incidents, agent-trace observability,
  invoice processing, customer service), scored as agreement with the averaged predictions of GPT-6 Astra and
  Claude Fable 5.1, vendor-reported. It is the benchmark the `LocalLLaMA/typed-decisions` dataset imitates, on
  which Laya's base checkpoint scores 0.362 and its fine-tuned checkpoint 0.766 (section 4.2).
- **83.8%** is the macro average over eleven of Laya's *own* in-task test families, all drawn from the
  distributions it trained on. The model repo's `eval/results.md` lists thirteen in-task families and reports
  0.753 overall; the post's table omits the two weakest, conversation outcomes (3,600 questions, 0.482) and
  sentiment and rating (961 questions, 0.442). The eleven listed average to 83.7%; all thirteen average to 77.9%.

So the "+16 points" compares Laya on its training distributions, minus its two worst families, with Jev on an
unrelated workflow benchmark where Laya zero-shot is 36 points behind. The latency row similarly quotes Jev at
"~400 ms avg" where TypeSafe states 70-500 ms and third parties measured 236-276 ms, against Laya's warm local
38 ms. The per-family in-task and zero-shot tables in the post match `eval/results.md` exactly and are sound; the
head-to-head table is the part to disregard.

One more argument in the post deserves a note because it is the stated rationale for RLCD. It says cross-entropy
"can only be minimized when the winner logit approaches infinity", so classifiers get over-confident, whereas a
proper-scoring-rule reward does not. But the log score used as the reward *is* negative cross-entropy; the two
objectives have the same optimum (the true conditional distribution) and the same gradient in expectation.
Over-confidence from cross-entropy comes from fitting one-hot labels on finite data, and the RL version fits the
same labels. The real calibration levers in the code are the soft targets, the bounded spherical term, and the
post-hoc temperatures, which is what section 3.2 says.

## 5. What this means for the plan

- **This is the architecture our encoder route was reaching for.** REPORT 38 fine-tuned an encoder categoriser
  across users to 76-80 with a fixed head; Laya puts the category descriptions in the input as `[MASK]`-marked
  options and scores them, so the label set is runtime-defined and a new category needs no retraining. Our category
  count is in the tens, under the ~20-option ceiling the author recommends, and the description text is exactly
  the "products-to-category bridge" section 35 found to be the weak link. The record-in-prompt arm (98 on
  DB-only merchants) is a `state` with the fact-DB record appended. A Laya-shaped categoriser is cheap to try:
  the whole notebook fine-tune is 6,000 items, four epochs, 5 minutes on two T4s, so REAL-6 at production
  volume is minutes on the 3090.
- **The training objective is portable to what we already do.** `proper_reward`, soft targets and per-bucket
  temperatures are a few hundred lines that sit on top of any option-scoring model, including our `scoring.py`
  records. The spherical term and RPS are the only pieces not already implied by cross-entropy. If we want
  calibration numbers comparable to Laya's, the metric code (`ece_score`, `aurc`, accuracy at coverage) is in
  `rl_common.py` and numpy-only.
- **Multiple questions per state do not share computation here.** For our workload (many transactions, one
  fixed instruction and option set) that is the right geometry anyway: the per-item sequence is short, and an
  encoder forward at 512 tokens is milliseconds. SemIf's prefix cache matters for long states; Laya's batching
  matters for many short ones. Ours is the second.
- **Hold-out discipline is the lesson, again.** The headline 0.766 is a fine-tune on the benchmark's own training
  split, reported alongside a base zero-shot 0.362; the author says so plainly. Our REAL-6/REAL-7 splits by
  merchant and by user are the equivalent discipline and should stay.
- **Multilingual is a solved sub-problem at this size** if bank strings ever carry non-Latin text: mmBERT-base
  at 322M reaches 45 of 51 languages at 3x chance on 20-way intent. Not a current requirement.
- **Cost of running it here:** 808 MB (English) or 647 MB (multilingual) download, fp16 on any GPU, CPU at
  193-464 ms per question. `pip install laya`, `USE_TF=0`, `laya.load("convaiinnovations/laya")`.

## 6. File map

| Where | What |
|---|---|
| `~/projects/Taytay/laya-hf/rl_common.py` | Sequence builder, `DecisionModel`, `proper_reward`, TD(lambda), record encoding with option shuffle, batching, metrics |
| `~/projects/Taytay/laya-hf/rl_agent_api.py` | Jev-shaped `system_one(state, questions)` inference with per-bucket temperature |
| `~/projects/Taytay/laya-hf/*/rl_agent_config.json` | The three checkpoints' budgets, temperatures and training records |
| `~/projects/Taytay/laya-hf/eval/results.md` | Per-family in-task and zero-shot accuracy, ECE, NLL; latency |
| `~/projects/Taytay/laya/laya/common.py` | Same model code as packaged (adds single-option guard, structured criteria rendering) |
| `~/projects/Taytay/laya/laya/{agent,router,shortlist}.py` | Loading, script-based checkpoint routing, embedding shortlist for large option sets |
| `~/projects/Taytay/laya/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` | The complete RLCD fine-tuning loop (cell 8) and evaluation |
| `~/projects/Taytay/laya/BENCHMARKS.md`, `research/` | All benchmark harnesses and raw JSON; the Jev comparison caveats |
