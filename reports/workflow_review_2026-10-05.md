# Workflow speed and cost review (2026-10-05)

Owner, 2026-10-05: "have a subagent review our current workflow and make other recommendations for speeding up our training and eval and
test runs". Read-only review by a subagent (code, job lists, `evals/runs.jsonl` timings, `modal_out/r193-k4-s0/modal_run.log`,
`reports/modal_costs.md`); nothing launched. PLAN row 208 carries it out.

## Where the money goes
- Modal bills about $4.09 per H100-hour ($0.068 per minute; r192_193: $84.59 for 1,241 job-minutes).
- Row 192-193: about $29 training (6 jobs x ~71 min), about $3.4 container start (~5 min x 10), about **$51 (61%) reads**.
- One r193 job: training 70.4 min (5.0 s per step); 8 drill and blind reads 9.3 min; 12 realstyle reads 79 min (5.8-7.1 min each,
  15-18 items/s, 55-67 ms per item).
- Owner's-budget scoring: about $3.7 per model, 8 containers each; these were the 100 unattributed apps ($292, 19% of all spend).

## Recommendations, biggest saving first
1. **vLLM for reads** (`scripts/exp_decision_models.py` decider path, lift from `scripts/bench_vllm.py`: merged adapter, one token with
   allowed label ids and their logprobs, prefix caching, items in household order). §120 measured 24.0 ms per item (18.9 with prefix
   cache) against HF 53 ms, top-1 equal. About 2.5-3x faster reads: ~$3.7 per train+read job, ~$31 on row 192-193. Check first:
   >= 99.5% top-1 agreement and matching scorecard on blind_v1 and v4g_test; compare tie handling (HF reads ties in bf16; the owner
   path moved to fp32 because 1.7% tie).
2. **Owner's-budget scoring** (`modal_app.private_scores`, shards=8): 2 shards instead of 8 (each container pays ~1-4.5 min start and a
   ~2 min overlay install), log phase timings (counts only), then vLLM in `score_private` (its prefix cache replaces the per-day cache).
   ~$1.5-2.5 per model from shards alone, ~$3 with vLLM ($3.7 -> ~$0.7). No change to where the private data goes (call arguments only).
3. **A per-row job-list generator** (`make_jobs(row, arms, primary_sets, variant_sets, second_seed)` in `scripts/modal_jobs/`):
   screening jobs read only the row's primary sets; variant sets only for named arms; drills and blind sets only when the row is about
   them; prints read minutes per job from history. Row 193 would have saved ~240 read-minutes (~$16) plus $23-35 of second seeds.
   Exact re-reads are rare (26 duplicates, 9 min): the waste is breadth.
4. **Container start-up** (`modal_app.py`): bake the training overlay (transformers 5.17.0, flash-linear-attention, peft, torch 2.13,
   torchvision 0.28) into the image as a second venv; stop mounting and copying all 6.9 GB of `data/processed` (mount only the files a
   job names); let `exp_decision_models.py` read several `ITEMS_SET`s in one process (model loaded and merged once; ~0.3 min per read
   now). ~3-5 min per container plus ~6 min per train+read job.
5. **Training padding** (`exp_decider_finetune.py` batch build): sort each step's 16 episodes by length into 2-4 micro-batches and
   accumulate (identical gradients); padded compute is now 1.67x real for G and 1.35x for K4 (realstyle episodes ~1,841 tokens,
   categoriser ~960). 20-33% off training, ~$1-1.5 per job. Gradient checkpointing must stay (>150 GB activations without it).
6. **Is 800 steps needed?** Loss still falls at step 800 (1.167 at 650, 1.060 at 799); no decider-4B step sweep exists. One seed at
   400 and 600 steps with their own schedules, read on v4g_test and blind_v1 (~$9). If 600 is within ~0.3 of 800, screen at 600.
7. **Keep the main test set size; cut how many sets are read.** On realstyle_v4g_test (6,000 items) arms disagree on 6.2-7.8% of
   items; paired SE 0.32-0.36 points (household clustering does not change it); a 1-point difference has power ~0.84; at 3,000 items
   only ~0.55. Seed-to-seed difference on synthetic is 0.08 (item noise dominates); on the owner's budget training-seed noise of 1-2
   points dominates, so two seeds stay necessary there. A 1,500-item subset is only a kill test.
8. **Local encoder scripts:** ColBERT reader (`hist_encoder2.py` ~185-187) runs one encoder forward per event: batch-embed once and
   compute MaxSim in chunked einsums (2.5 h -> ~5-10 min). MaxSim reader (~181) is O(N^2) per household with a sort per category per
   event: vectorised version 18x faster on CPU. Cache per-household `_scores` as `.npz` keyed by encoder hash and env (synthetic in a
   gitignored `data/interim/hist_cache/`; the owner's only under `~/.local/share/ynab-real-eval/<budget>/`, mode 0600). Add timestamps
   to chain logs.
9. **Process:** print a cost estimate before every launch (steps x 5.2 s + read minutes from history + 5 min start, at $4.09/h); an
   early sanity read at step 200 (~30 s); auto-ingest when the watcher sees the client exit. Cheaper GPUs are not cheaper per item
   (L40S ~0.37x H100 throughput at ~$1.95/h; A100 ~0.32x at ~$2.50/h); keep reads on the H100, or the local 3090 for small sets.

## Do first
1. vLLM for synthetic reads, after the agreement check.
2. Owner's-budget scoring: 2 shards, phase timings, then vLLM.
3. The per-row job-list generator.
4. Bake the overlay, mount only needed data, many reads per process.
5. Length-sorted micro-batches in training; then the 400 / 600-step check.

Not covered: 35B H200 reads, strands and encmask jobs, `build_realstyle` build time (it regenerates the same 100 households once per
variant; chain logs had no timestamps).
