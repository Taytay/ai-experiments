# JevBench: the external gold set for typed decisions, how it scores, and what it says about where open models lose

Date: 2026-09-22. Eleventh memo in the Jev series. Object: https://github.com/fstandhartinger/jevbench (MIT, artifact
revision v1.3.0, clone at `~/projects/Taytay/jevbench`), the harness behind https://benchmarkheaven.com/jev-models. Read by a
subagent; this memo condenses its report.

## 0. The one-paragraph answer

JevBench is a one-person, carefully run benchmark for "Jev-class" decision models: a state plus a bounded rubric in, a
probability over an exact label set out. 534 frozen decisions (72 easy, 96 standard, 146 judge, 220 hard), 231 public and
303 never published, SHA-256-frozen before any system ran, with every addition pre-registered and a public-vs-held-out
diagnostic published. The JevBench Score is the geometric mean of four 0-100 axes (chance-corrected Intelligence,
Calibration, Speed, Cost), which makes the composite ranking depend heavily on Speed and Cost assumptions: self-hosted
latency is multiplied by 2 and increased by 0.15 s ("an assumption, not a measurement"), and self-hosted cost is priced at a
hosted sibling's list price. The hard tier (220 items) is the entire separation between systems; easy is saturated and
standard and judge nearly so for the top ten. The hard tier was written and cross-reviewed by Claude Opus 5 and GPT-5.6 Sol,
not humans, and per-family n is 10-38. Verdict: read the per-tier and per-family accuracies, not the composite. For this
repo it is usable locally as a 231-item out-of-domain gold set, and its per-item public outcomes for all 52 systems
(including Jev) are committed, so we can compare item by item against Jev without an API key.

## 1. Items

JSONL records (`jevbench/tasks.py`): `id`, `family`, `state` (string or dict, with answer-like keys banned), `question`
`{type: noul|choice|score, instructions, criteria}`, ordered `labels`, `expected`, `split`, paraphrase `group`, `provenance`.
Noul labels are exactly `["no", "yes"]`; choice 3-7 snake_case labels with one-line definitions; score ordinal levels.

| Tier | Items (public) | Content |
|---|---:|---|
| easy | 72 (48) | intent, fact, extraction, tool selection |
| standard | 96 (72, as 36 paraphrase pairs) | policy, intent, ordinal, extraction, adequacy, routing |
| judge | 146 (0) | 78 routing requests with human categories, 68 answer-adequacy judgements (61 yes / 7 no, majority floor 82%) |
| hard | 220 (111) | adversarial 12, ambiguous 14, judge_hard 33, long_policy 38, multi_hop 35, probability 20, routing_hard 10, temporal_numeric 30, tradeoff 12, trap 16 |

The hard tier (`datasets/HARD-TIER.md`): 110 items by Claude Opus 5 and 110 by GPT-5.6 Sol, blind cross-review, a gold pass,
220 of 220 accepted. Spec: a Flash-Lite-class model misses 20-40%, a careful human 95% or more. Every item has a rationale,
a `surface_answer` (the tempting wrong answer) and `why_hard`; median state 388 tokens, maximum 3,746. The 20 probability
items carry an exact `gold_probs` distribution derivable from countable evidence.

## 2. Scoring (`jevbench/composite_v13.py`)

- **Intelligence** = Σ over tiers of weight × clamp(100 × (acc − chance)/(1 − chance)), weights easy 0.14, standard 0.28,
  judge 0.28, hard 0.30; chance per tier from option counts (0.28-0.34).
- **Calibration** (hard tier only) = mean of 100 × (1 − ECE/0.5) with top-label ECE in 10 bins, and 100 × (1 − mean total
  variation distance) to `gold_probs` on the 20 probability items. Label-only systems get 0.
- **Speed** = mean of score(p50) and score(p95), score(s) = 100 − 20 log10(s/0.1), on the serial 242-decision standard+judge
  run, with the self-hosted adjustment.
- **Cost** = 100 − 30 log10(USD per 1,000 decisions / 0.001), never 100 for a missing price.
- **Composite** = geometric mean with a floor of 1 per axis, times (I/50)² when Intelligence is below 50 (new in v1.3.0).

Known weaknesses the author states: the latency adjustment rests on one chart about a 1.8T MoE; self-hosted cost depends on
which hosted sibling is picked; one item moves routing_hard by 10 points; a zero Calibration axis collapses the composite;
verbalised LLM probabilities and native distributions share one axis; option order alone moved one entrant from 72% to 21%
on yes/no judging.

## 3. Running it locally and getting ranked

The `typesafe` adapter POSTs `{state, model, questions: {decision: {type, instructions, criteria}}}` to
`{endpoint}/v1/systemone` and expects `answers.decision` with `noul` probability or `choice` plus `probabilities`.
`python -m jevbench.cli run ... --adapter typesafe --endpoint http://127.0.0.1:8000 --key-env ''` runs the public items
serially (results paths must be outside the repo); `summarize` gives accuracy, per-family blocks, Brier, ECE with bins,
ordinal MAE, latency percentiles, paraphrase consistency. Hard-tier ECE and TVD to `gold_probs` must be computed by hand
from the outputs. Held-out items go only to the maintainer's machines or to a production API; a submitter's own endpoint
gets a partial, unranked row. The path every ranked open entrant took: publish weights and a pinned server, open a GitHub
issue, the maintainer pre-registers the run and executes all 534 on their GPU (kev ran BF16 on one RTX 3090).

## 4. Where open models lose to Jev

Tier accuracies (easy / standard / judge / hard): Jev 1.000 / 0.990 / 0.945 / 0.741; SemIf 1.000 / 0.979 / 0.952 / 0.595;
djev 1.000 / 0.979 / 0.932 / 0.695; Winnow-12B 1.000 / 0.969 / 0.911 / 0.709; reflex 4B 1.000 / 0.948 / 0.973 / 0.632; kev
0.6B 1.000 / 0.812 / 0.664 / 0.400.

| Hard family (n) | Jev | SemIf | djev | Winnow | reflex 4B | kev 0.6B | GPT-5.6 Luna | ranked-field mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| long_policy (38) | 0.61 | 0.42 | 0.47 | 0.66 | 0.53 | 0.29 | 0.95 | 0.43 |
| temporal_numeric (30) | 0.27 | 0.20 | 0.33 | 0.27 | 0.33 | 0.33 | 0.93 | 0.30 |
| probability (20) | 0.80 | 0.40 | 0.60 | 0.55 | 0.55 | 0.30 | 0.90 | 0.50 |
| multi_hop (35) | 0.86 | 0.63 | 0.83 | 0.69 | 0.63 | 0.31 | 0.94 | 0.54 |
| judge_hard (33) | 0.79 | 0.79 | 0.88 | 0.88 | 0.73 | 0.55 | 0.91 | 0.67 |
| ambiguous (14) | 0.79 | 0.57 | 0.71 | 0.86 | 0.50 | 0.29 | 0.93 | 0.49 |
| tradeoff (12) | 0.92 | 0.75 | 0.67 | 0.83 | 0.75 | 0.50 | 1.00 | 0.57 |
| adversarial / trap / routing_hard | 1.00 | 0.83 / 1.00 / 1.00 | ≈1.0 | ≈1.0 | ≈1.0 | 0.67 / 0.56 / 0.50 | 1.00 | 0.73-0.77 |

Three readings. **Temporal and numeric reasoning defeats every one-pass decision model, Jev included** (Jev 8 of 30); only
generative models that reason first solve it (Luna and DeepSeek 0.93; OpenJev with thinking 0.47). Open rebuilds trail Jev
most on **probability, multi-hop, long policy and ambiguity**. Adversarial, trap and routing items are saturated for
anything 4B or larger. Calibration: Jev ECE 0.061 and fidelity 77.4; SemIf 0.121 / 69.4; djev 0.175 / 65.8; reflex-27b
0.047 / 81.7; kev 4B 0.402 / 64.3; DeepSeek 0.033 / 100. A combinations study found no cascade or committee beating a single
system on the composite, but a cheap-model-to-Jev cascade at threshold 0.42 escalated 3% of items and kept 99.6% of Jev's
accuracy.

## 5. What this means for the plan

- **Use the 231 public items as an out-of-domain gold set** for any decision-readout model we build, reporting per-tier and
  per-family accuracy against the chance floor, not the composite. Ten public probability items make fidelity noisy.
- **Item-level comparison against Jev is free**: `results/v1.2/jevbench-v1.2-per-task.json` has every system's outcome on the
  public items.
- **Our own items can be written in its schema**: a categoriser item is a `choice` question with `{category: definition}`
  criteria and the transaction as `state`; `Task.from_dict` validates and `dataset_hash` freezes. That would let us report our
  REAL-6 set in the same format as the field.
- **Reuse** `scoring.py` (distribution validation with a 2% renormalisation band), `metrics.py` (Brier, top-label ECE with
  bins, ordinal MAE, paraphrase consistency) and `composite_v13.py` (TVD, chance correction). Selective accuracy is not in
  the harness; the cascade threshold sweep in `scripts/analyze_combinations.py` is the closest and adapts into an
  abstention curve.
- **Match the weak families to our problem**: our hard cases (opaque merchants, ambiguous records) look like JevBench's
  `ambiguous` and `probability` families, not `temporal_numeric`. The latter is a reasoning problem no one-pass model solves,
  and we should not chase it.

## 6. File map

| Where | What |
|---|---|
| `README.md`, `RESULTS-v1.2.md` | Score definition, revision log, current ranking, tier accuracies, limits |
| `datasets/HARD-TIER.md`, `datasets/public/{easy,original,hard}.jsonl`, `datasets/manifest.json` | Hard-tier spec and the 231 public items with hashes |
| `jevbench/composite_v13.py`, `scoring.py`, `metrics.py`, `summarize.py` | Axes, composite, validation, metrics |
| `jevbench/adapters/typesafe.py`, `adapters/base.py`, `tasks.py` | The `/v1/systemone` mapping and item schema |
| `jevbench/cli.py`, `runner.py` | Run and summarize |
| `results/v1.2/jevbench-v1.2-results.json`, `jevbench-v1.2-per-task.json` | Per-family hard results and public per-item outcomes for all systems |
| `docs/v1.2-additions*.md`, `RESULTS-COMBINATIONS.md` | Pre-registrations; cascade and committee study |
