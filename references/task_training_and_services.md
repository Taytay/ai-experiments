# Task-direct training, distillation and rented training: what the qorl post and the training services change for this plan

Date: 2026-09-17. Written after the owner asked whether the approaches in Rohan Bansal's post
(`references/blog/qorl_4b_query_optimizer.md`, https://rohanbansal.com/qorl) and the services it links
(Tinker, River, Baseten, plus the owner's Modal and Lightning subscriptions) should change the plan,
whether we should train on the task directly instead of injecting knowledge, and how their pricing compares
with the RTX 3090. Sources read: the post; https://thinkingmachines.ai/blog/on-policy-distillation/;
the Tinker docs (quickstart, models and pricing, LoRA primer, distillation recipe); https://river.ai/api;
the Baseten training docs and pricing; https://modal.com/pricing; the unsloth RL guide. Lightning's pricing
page did not render for the fetcher.

## 1. What the post does

A 4B model (an Empero distillation of Qwen 3.8 into Qwen 3.5 4B) is taught to produce Postgres query-plan
hints inside a six-tool agent harness, in two stages.

1. **Off-policy distillation by SFT.** 100 trajectories from a frontier model (GPT-6 Astra through the
   OpenAI API, reasoning summaries only) are rendered into the student's chat format (Prime Intellect's
   `renderers`), loss-masked so that only assistant tokens are scored, and unrolled so a trajectory with
   three replies becomes three `(context, reply)` rows; 100 trajectories became 382 packed rows. A 21M-parameter
   LoRA (42 MB) trained with `prime-rl`, batch size 1, one epoch, on one 3090 (four hours), taught the
   model the harness: valid-candidate rate 14/113 to 48/113. A second epoch took it to 85/113 and a 1.08x
   speedup; a third epoch regressed it, while validation loss was flat across both. 300 more trajectories
   (six filtered out) and two more epochs gave 1.16x. Total SFT data: 400 trajectories.
2. **Agentic RL** with a verifiable reward (measured speedup against Postgres's default plan): rollouts on
   his 3090 box, vLLM inference and the trainer on a rented 2x H100 node. Plain GRPO with a harsh reward
   reinforced "least bad" rollouts; the fix was an anchored advantage (a rollout is credited only against
   max(0, its siblings' mean quality)), log-speedup clipped to [0.1, 10], a 5% dead zone, and small fees for
   invalid or duplicate plans. Two 600-update runs at lr 1e-5, batch 16, eight rollouts per query. Final:
   1.81x geometric-mean speedup with best-of-three trajectories. Cost: $800 of H100 time, $400 of API fees
   for the teacher traces, otherwise electricity.

His own "next steps" are on-policy distillation and trace inversion, which is where the second source comes in.

## 2. What we have and have not done (the honest mapping)

| Technique in the post or the Thinking Machines post | Have we done it? | Where |
|---|---|---|
| Off-policy distillation: SFT on a stronger model's traces, loss-masked and unrolled | No. Our "distillation" arms use the same model as its own teacher with the field guide in context (arm P: token KL, section 14; P2: option-renormalised, section 18) and a self-teaching stream (row 9). The knowledge texts written by an LLM (desc14llm, section 27) are data, not traces. No frontier teacher, no reasoning traces, no harness | REPORT 14, 18, 21, 27 |
| On-policy distillation: sample from the student, per-token reverse KL against a teacher as the advantage | No | - |
| RL with a verifiable reward (GRPO or variants) | No | - |
| Training directly on the downstream task with labels | Encoders, yes: the category-tuned contrastive MiniLM/bge/Qwen3/EmbeddingGemma/gte encoders are exactly that (68 to 93 on the merchant and induction tasks). The LLM, no: section 4 withheld the category label from training by design, so the LLM has only ever seen the merchants' facts, never the task | REPORT 4, 13, 24, 24.7 |
| Loss masking (answer-only loss) | Yes, on the episodes (`ALL_ANSWER` scores every demo label; the prompt is context) | `exp_curriculum.py` |
| Unrolling multi-turn trajectories | No multi-turn data exists here | - |
| Tiny LoRA (rank low, 21M params) learning a format in one epoch | Section 28: rank 16 learns the facts (recall 100) and loses induction; rank 64 all-linear is the recipe | REPORT 28 |
| More epochs on the same demos helps then hurts; validation loss uninformative | Section 28 (1,600 steps best, 400 too few) and the periodic-eval curves; we already judge by the ladder, not by loss | REPORT 23, 28 |
| LoRA at ten times the full fine-tuning rate (Tinker's LoRA primer) | No. Our decoder LoRA runs use 1e-4, which is the full-FT rate the primer says LoRA should exceed tenfold; section 28 found 2e-4 better than 1e-4 on manipulation and induction and stopped there. The Flan-T5 adapter at 3e-4 learned nothing (section 29) | REPORT 28, 29 |
| Renting GPUs for the slow part | No; every run has been on the 3090, serially | NOTES.md |

## 2b. The owner's final goals (stated 2026-09-17, after the first draft of this note)

1. A fine-tuned "one trick pony" that auto-categorises bank transactions the way a particular user
   categorised similar ones before, on seen and unseen category names and on seen and unseen-but-related
   inputs (new merchants, new locations of known chains).
2. A way to inject external fact databases (retailer, POI; the owner's internal dataset in production,
   purchasable datasets as extras) so the model knows those merchants and categorises them better even
   when they never appear in the user's labelled history.

The species ladder is the abstraction of both: recall is knowing the merchant, Timmy is an unseen category
name learned from a few examples, held-out species are unseen related inputs; the fact-DB injection is what
sections 4, 8, 25 and 30 measure. What is missing is the joint measurement, the categoriser with and without
the injected DB on the merchants the user never labelled, on a per-user scheme. That is now REAL-6 (the
evaluation set, row 35) and REAL-5 (the experiment, row 33).

## 3. Should we "just train on the task"?

Two different tasks are hiding in the repo, and the answer differs.

**The merchant problem (bank string to category).** If category labels exist for the merchants we care about,
then yes: a classifier is the right tool, the encoders already are that classifier (93 on the LLM's own
induction items, 68 to 77 on bank strings through case-robust encoders), and an LLM fine-tuned on
`(bank string, category)` pairs was never run because section 4 deliberately kept the label out. That is
the missing baseline, and it is cheap. The interesting case is the one the knowledge framing was built
for: a new merchant arrives with a record (what it sells) and no label. There the post's recipe applies
directly and is untried here: a strong teacher with the merchant records available as a tool or in context
writes short reasoning traces from bank string to category; the 3B student is fine-tuned on those traces
(loss-masked, unrolled if multi-turn); a verifiable reward (exact category) allows a GRPO stage on top.
Section 25 already shows the two halves separately (record retrieval at 100% recall@1; the LLM with the
record in context at oracle accuracy, 81/75 on 3B), so the expected gain from a trace-distilled,
tool-using student is not accuracy on trained merchants but the ability to handle new merchants without
retraining, which is what the post's model does for new queries.

**The species ladder (recall, manipulation, induction, reverse).** The ladder is the measurement, so
training on its items would be training on the test; the honest version of "train on the task" is what
arm C already does (episodes are induction tasks with held-out labels) and what section 18 found: a task
form distilled into the weights stays bound to that form (P2: recall 91.9, yes/no 58.8, pair 50). The
post does not contradict this; its student never had to answer questions about the harness in new forms.

So the plan should keep the injection results as the answer to "what can the weights hold" (they can hold
1,000 species at 100 recall, and an editor can hold associations with zero general-ability cost, section
30) and add one task-direct row for the merchant problem, where the owner's actual use lives.

## 4. What the Thinking Machines post says that matters most here

Their "personalization" experiment is our forgetting problem with numbers. Midtraining Qwen3-8B on internal
documents took Internal-QA from 18 to 43 and IF-eval from 85 to 45; mixing 30% chat data gave 36 / 79; an
on-policy distillation phase afterwards, with the original Qwen3-8B as teacher on Tulu3 prompts, gave 41 / 83.
They also report that LoRA during midtraining "learned less knowledge and still forgot", that SFT on the
student's own samples degrades IF-eval at any learning rate (finite batches drift off-policy), and that
distillation reaches RL-level results in 7 to 10 times fewer steps with a single forward pass of the
teacher per sample (dense per-token signal, O(N) bits per episode against RL's O(1)).

Our arm C loses exactly what they lost: WikiText perplexity 10.6 to 22.8, ARC-Easy 73.5 to 58.5, at
5,000 species 30.6, and the encoder-decoders lose natural ICL and ARC on every working recipe (section 29).
Sections 24 to 25 (general-text replay, TRAIN-7) bought some of it back at a recall cost. On-policy
distillation from the pre-injection model is the repair they found, and it costs nothing extra in memory
here: with LoRA the teacher is the same base model with the adapter disabled. Recipe on one 3090: sample
from the student (adapter on) on general prompts that are not our eval items (Tulu-3 SFT prompts or
similar), compute the base model's log-probs on those samples (adapter off), set the per-token advantage to
minus the reverse KL, and take importance-sampling policy-gradient steps on the adapter. Their reasoning runs
used 64 prompts times 4 samples per batch and about 100 to 150 steps; on a 3B model with Hugging Face
generation that is an hour or two. The measurement is the full ladder: recall should stay near 100 (they
kept 41 of 43) while perplexity, ARC-Easy and the ICL suite move back toward the base.

## 5. Tricks worth taking, whether or not we rent anything

1. **LoRA learning rate ten times the full-FT rate.** Our 1e-4 is the full-FT rate; the primer and the
   Tinker recipes use 1e-3 for LoRA SFT and 1e-4 for LoRA RL. Two arm-C runs at 5e-4 and 1e-3 and one
   Flan-T5 adapter run at 1e-3 or 3e-3 decide whether the section 28 and 29 verdicts on adapters were rate
   artefacts. Cheapest item on this list.
2. **On-policy distillation as the forgetting repair** (section 4 above), teacher = adapter off.
3. **Loss-masked, unrolled teacher traces** as the SFT format if we build the merchant tool task; the
   `renderers` library or the model's own chat template does the rendering, and the row count multiplies
   by the number of assistant turns.
4. **Judge by the task metric, never by validation loss**: already our practice; the post's flat-loss,
   improving-behaviour epoch is the same lesson as our step-1,600 result.
5. **Anchored advantages** if we ever run GRPO: never credit a rollout for beating a bad group mean; score
   against a fixed reference (the default plan there, the base model's accuracy here).
6. **Single-GPU RL is feasible locally**: unsloth's GRPO path colocates vLLM with the trainer
   (`fast_inference=True`) and trains 3B-class LoRA models within 24 GB; that is the local route for a
   GRPO stage on the merchant task, and it avoids the post's two-machine Tailscale setup.
7. **Trace inversion** (expanding a teacher's reasoning summary into a plausible full trace) if we distil from
   an API teacher that hides its reasoning.

## 6. Pricing against the 3090

Reference numbers (2026-09-17, list prices). Our arm C run is 1.25M training tokens (12,800 sequences), 9
minutes of training on the fast path plus 9 of evaluation, at about 0.5 kW for the whole box.

| Where | Unit price | Arm C-sized run (train 1.25M tokens + eval about 4M prefill tokens) | Notes |
|---|---|---|---|
| RTX 3090, local | about $0.07 per hour of electricity at 0.5 kW and $0.14/kWh | about $0.02 | serial; 24 GB caps the model size (7B needs QLoRA and crawls at the VRAM limit) |
| Tinker (Thinking Machines) | Qwen3-8B: train $0.44/M, prefill $0.195/M, sample $0.60/M; Qwen3.5-4B: train $0.737/M, prefill $0.33/M, sample $1.005/M; Qwen3.5-9B: $1.463 / $0.66 / $1.995; storage $0.10/GB-month; 80% off cached prefill | about $1.3 to $2.5 | LoRA only; smallest models are 4B/8B/9B, no Qwen2.5-3B; the API is `forward_backward`, `optim_step`, `sample`, `save_state` with per-token loss weights, so our packed mixture and answer-only losses port directly; `compute_logprobs` exists for option scoring; on-policy distillation is a shipped recipe |
| River | Qwen3.5-9B: train $1.46/M, prompt $0.66/M, completion $1.99/M; cached prompt 20% of prompt; storage $0.10/GB-month | about $4 | LoRA SFT plus GRPO/CISPO; models 9B and up; no distillation recipe |
| Baseten Training | H100 80 GB $0.10833/min ($6.50/h); A100 80 GB $4.00/h; L4 $0.85/h | about $0.40 (about 4 minutes of H100) plus container start | bring your own container (unsloth, our uv lock); Axolotl, TRL, VeRL, MS-Swift supported; multi-node; checkpoints stored and deployable |
| Modal (owner subscribes) | H100 $0.001097/s ($3.95/h); A100 80 GB $2.50/h; A10 $1.10/h; $30/month of free compute on Starter | about $0.25 | same container story as Baseten; the free credit covers about 7 H100-hours a month, which is the two long queued runs |
| Lightning (owner subscribes) | page did not render for the fetcher | - | studios with persistent disks; same use as Modal |

**Speed, which the per-run prices hide.** The H100 column above already assumes the speedup (an arm C run at
about 4 minutes of H100 against 18 here), but the time is the point, not the price. The 3090 has 71 TFLOPS of
bf16 tensor throughput and 936 GB/s of memory bandwidth; an H100 SXM has 989 TFLOPS and 3.35 TB/s, plus 80
GB. In practice that is 4 to 6x on our LoRA training (the post measured 5.3x: four hours per SFT epoch on his
3090, 45 minutes on the H100), about 3 to 4x on our option-scoring evaluation (bandwidth-bound), no QLoRA and
no driver spill for 7B, and room for batch 64 or 2,048-token rows without the OOMs of section 29. The queue
that remains (rows 19 to 34 plus the two long REAL-3 / MODEL-2 runs) is roughly 35 GPU-hours on the 3090,
which is a week of serial evenings here or about 8 H100-hours, about $32 on Modal, one afternoon if two jobs
run at once. So the right split is: the 3090 for development, smokes and anything under an hour; the cloud
for every multi-hour run and for anything at 7B or above; and the cost of that is inside the existing Modal
credit. Row 34 is written for the two longest runs, but the same function serves the whole queue.

Reading the prices alone: per run the 3090 is 10 to 100 times cheaper than any service, and every run we have made
(about 150) would have cost $300 to $600 on Tinker or River and $40 to $60 on Modal/Baseten H100 time. The
services do not win on price; they win on (a) parallelism, since the 3090 runs one job at a time and the
plan's remaining long items (arm C at 5,000 species for 20,000 steps, about 5.5 hours here; Flan-T5 at 1,000
and 5,000 species at 3e-4, about 3 hours) block everything else while they run, (b) memory, since 7B and up
need QLoRA here and the periodic evaluations pushed 24 GB into the driver's spill regime (section 7), and
(c) the teacher side of distillation, where a frontier-size open model is only reachable by API. The post's
own budget was $400 of frontier-API traces and $800 of H100 time for 95 hours; the local equivalent of his
SFT stage ran on his 3090 in four hours per epoch.

What a few dozen dollars per service would buy:

- **Tinker, $20 to $40:** the on-policy distillation recipe as shipped (Qwen3.5-9B teacher, Qwen3.5-4B
  student) on our knowledge texts and the ladder, as an independent check of the local implementation; and
  frontier-size trace generation (Qwen3.5-397B sampling at $7.50/M output tokens: 2,000 traces of 300 tokens
  is about $5). Worth it once the local TRAIN-8 row has a number to compare against.
- **Modal (already paid):** the two long queued runs and any 7B run, inside the free credit. This is the
  one to set up first: a Modal function with the repo image built from `uv.lock`, the frozen item sets and
  `hf_cache` mounted, results and adapters synced back to DVC. No new science, more throughput.
- **Baseten, River:** nothing we cannot get from Modal or Tinker; skip unless deployment matters.
- **Axolotl:** no. Our training loop is the experiment (mixture streams, packing, periodic ladder evaluation,
  per-item option log-probs); Axolotl replaces the part that works with a config file and would need
  callbacks for everything that makes the runs comparable. For a GRPO stage, unsloth's or TRL's trainer is
  the shorter path.

## 7. Plan changes made

New question IDs in `reports/QUESTIONS.md`: TRAIN-8 (on-policy distillation from the pre-injection model as
the forgetting repair), TRAIN-9 (LoRA at ten times the full-FT rate), REAL-5 (task-direct training for the
merchant problem: label SFT baseline, teacher-trace distillation with the records as a tool, optional GRPO),
INFRA-1 (move the long queued runs to Modal). PLAN rows 31 to 34, placed after row 19 (31, 32) and after row
22 (33, 34), with the reasoning in the PLAN log.
