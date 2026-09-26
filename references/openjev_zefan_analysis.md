# Open-Jev (Zefan-Cai): the one public 79k-row training set, and why it did not transfer

Date: 2026-09-22. Seventh memo in the Jev series. Object: https://github.com/Zefan-Cai/Open-Jev (MIT, clone at
`~/projects/Taytay/Open-Jev`, 1,293 files) and its dataset on Hugging Face (config `release-v2-redistributable`, CC0).
Read by a subagent; this memo condenses its report. JevBench v1.3.0 rows: Open-Jev 9B #30 at 55.0 (Intelligence 71,
Calibration 63, hard 0.609) and Open-Jev 2B #34 at 51.3 (I 61, C 55, hard 0.427). The board found no benchmark text in the
79,116-row public training projection, so this is a transfer failure, not contamination.

## 0. The one-paragraph answer

Open-Jev is a frozen Qwen3.5 (2B, 9B, later 27B) with a rank-8 LoRA on the attention projections and a single scalar head
`Linear(hidden, 1)` initialised from the `Yes` minus `No` unembedding rows. Every candidate option is rendered as its own
prompt ("Proposed answer: X. Is this proposed answer correct? Answer Yes or No."), scored independently at its last token,
and the K scalars are softmaxed within the question; one NLL-fitted temperature is applied at serving (2B 1.52, 9B 1.90).
Its distinguishing asset is the training set: 115,821 rows (79,116 public train rows) that are **entirely synthetic and
rule-labelled**, from 13 Python generators with no human labels and no LLM teacher. In distribution the training works
spectacularly (9B held-out hard accuracy 64.7% to 95.1%). On JevBench the 9B scores below SemIf's frozen 4B readout on
Intelligence (71 vs 79) and the author never ran the untrained readout on JevBench, so whether the LoRA helped, hurt or
was neutral there is unmeasured. The data explains the rest: about 55% of rows are pixel geometry, Snake, ViZDoom,
tic-tac-toe and a platformer, the language is templated, and nothing in it resembles the hard tier's long policies, dates,
trade-offs or ambiguity. The author's own pivot for the 27B (adding Banking77, CLINC150 and WANLI human labels) is an
implicit acknowledgement.

## 1. How it works

- **Prompt** (`jev/api.py`): `Context:\n{state as text or sorted JSON}\n\nQuestion: {instructions}\n` then per candidate
  `Proposed answer: {option}\nIs this proposed answer correct? Answer Yes or No.`; noul is a single sequence `Is the answer
  to this question yes? Answer Yes or No.` Qwen chat template, thinking off. Choice options render as `name: description`;
  score options render only the level text, so the model never sees sibling levels or positions; ids and candidate order
  never appear.
- **Readout** (`jev/model.py`): language-model part only, bf16. Head weight is `W_out[Yes] - W_out[No]`, bias 0, fp32.
  One scalar per candidate from the last non-pad hidden state; choice and score logits are the K scalars, noul logits
  `[0, s]`. Softmax within one question only. LoRA r=8, alpha 16, on `q,k,v,o` and the hybrid layers' `in_proj_qkv,
  out_proj`. Order-invariant by construction, since each candidate is scored alone.
- **No shared state.** The context is re-encoded in every candidate sequence at training and default inference. A
  request-local prefix cache exists and is off by default because it exceeded the probability tolerance on 9 of 11
  workloads despite identical argmaxes. Latency therefore scales with candidates: 85 ms p50 on customer service, 1,016 ms
  at 1,024 state tokens times 32 candidates (Jev: 301 ms).
- **Serving** (`jev/serving.py`, `server.py`): candidates chunked into batches, large choices split and reassembled before
  softmax, `temperature.json` applied, sum-to-one validated, over-length inputs refused.

## 2. How it is trained

- **Dataset** (`reports/data-manifests/release-v2.json`, `docs/training-coverage-gaps.md`): 115,821 rows, splits train
  80,816 / calibration 4,761 / validation 3,792 / test 10,532 / OOD 15,920, grouped 80/5/5/10 by hash; kinds noul 70,474 /
  choice 28,176 / score 17,171. Train rows by generator: painting geometry 23,552 (from 46 scenes), Snake 12,508,
  security-incident policy noul 8,874, ViZDoom 6,354, reasoning controls 5,442 (seven exact-oracle families: Boolean logic,
  graph paths, arithmetic, time-zone temporal, tiny program interpretation, list ops, evidence integration), invoice
  processing 5,370, customer-service workflow 4,392, customer controls 4,206, agent trace 3,780, tic-tac-toe 3,264,
  Wikispeedia 1,700 (dropped from the public release for licence reasons), tile platformer 1,346, T-rex 28. Entities are
  hashed placeholders. Labels are hard one-hot from oracles and rules; "teacher" always means a scripted policy (minimax,
  BFS, tracking heuristic). Soft targets exist only where a generator declares uniform latent worlds (960 of 26,452
  test/OOD rows). Options shuffled per row at generation. BoolQ and WANLI importers exist but were not used for the
  released adapters.
- **Objective** (`jev/train.py`): `-sum(y log softmax(s)) + 0.1 sum((softmax(s) - y)^2)`, equal row weight, no domain
  reweighting. AdamW wd 0.01, clip 1.0, LoRA lr 5e-5, head lr 1e-4, head trained jointly from step 0. Released 2B and 9B:
  20,204 steps at global batch 4, one pass over 80,816 rows, max length 4,096, one H100 each, 4.4 h (2B) and 5.9 h (9B).
- **Temperature** (`jev/metrics.py: fit_temperature`): one scalar minimising NLL on 512 source- and kind-balanced
  calibration rows, log grid [0.05, 20] then golden section. Both fitted values exceed 1, so the trained head is
  over-confident and softened.

## 3. Results and flags

- Own held-out data: 2B test hard 94.7%, OOD 86.0%; 9B 97.5% / 92.0%. Paired baseline-to-trained on 512 rows: 2B 54.7% to
  91.7%, 9B 64.7% to 95.1%. Flag: "OOD" is template, entity and range shift inside the same grammars.
- JevBench public 231 (`docs/jevbench-public.md`): 2B 150/231 (hard 46/111), Brier 0.475, ECE 0.127; 9B 179/231 (hard
  66/111), Brier 0.322, ECE 0.086; copied Jev 200/231 (hard 81/111). By family, 9B: temporal_numeric 2/15, long_policy
  9/19, tradeoff 2/6, ambiguous 3/7, probability 6/10, multi_hop 12/18; perfect on extraction, fact, intent, ordinal, tool
  selection and trap. Author-stated flags: candidate order differs from the harness on 119 of 139 choice tasks, private
  tiers unavailable, local H100 latency not comparable. Unstated flag: **no untrained-base row on JevBench**, so the
  training effect on the benchmark is unmeasured.
- A 76-hard-case "provider suite" drawn from their own test/OOD data: 2B 65/76, 9B 72/76, Jev 66/76. On their synthetic
  data they beat Jev; not transferable.
- The author's run used max length 16,384; the board's row used 4,096, which may account for the small deltas (hard
  0.595 vs 0.609 at 9B).

## 4. Why 79k rows did not beat a frozen 4B zero-shot

1. Row count overstates diversity: 46 painting scenes produce 23,552 rows; the author writes that "row proportions are not
   proportions of independent scenarios."
2. Zero coverage of the hard-tier families: no free-text policies, no real dates, no trade-offs, ambiguity only as a
   declared uniform prior.
3. Templated language: about 15 fixed utterances per split in `case_customer.py`; the author concedes "narrow controlled
   wording."
4. The readout started weak (54.7% / 64.7% on the synthetic data) and the LoRA's job was mostly to learn the synthetic
   contracts, so it over-fits those prompts; the fitted T above 1 confirms over-confidence.
5. Recipe: single pass, no validation-based selection, equal row weights, no ablation against the base model. The author
   writes "no controlled single-domain-versus-mixed ablation."

## 5. What Open-Jev adds, and what it means for the plan

- **The independent-candidate scalar readout is an alternative to letter slots**: order-invariant by construction, no
  cardinality limit, no letter prior, at the price of K full encodings of the state (SemIf's reranker mode had the same
  cost shape and the same result: worse than the letter readout on general decisions). Our shared-prefix geometry makes
  the K encodings cheap only if the prefix is the long part, which for us it is.
- **The Yes-minus-No head initialisation** is a sound way to start a scalar head at the readout's own behaviour; jqv found
  a head trained from scratch on a frozen LM is 8 points worse than the readout, and this is the fix.
- **The dataset is a warning about synthetic mixtures, and we build synthetic mixtures.** Our merchant universe is
  procedurally generated too. Open-Jev's failure mode, a model that is 95% on its own grammars and 60% on the real thing,
  is exactly what REAL-6's production-shaped set (REPORT 37) exists to catch. The reasoning-control generator (exact
  oracles for logic, dates, arithmetic) is reusable as a sanity slice; the rest is not worth training on.
- **Always keep a base-model zero-shot control on the target benchmark.** Open-Jev's biggest gap is not knowing whether its
  6-hour LoRA did anything on JevBench. Every one of our arms has an untrained control; keep it.
- **Reusable code**: the schema with grouped hash splits and a dedicated calibration split, `jev/metrics.py` (pure-Python
  temperature fit, ECE, Brier, soft-target accuracy), the 13-token shingle overlap screen. Memory: the 9B trained rank-8
  at 4,096 tokens on one 80 GB H100; a 2B or 4B at 2,048 tokens, r=8, batch 1 with accumulation 4 fits 24 GB.

## 6. File map

| Where | What |
|---|---|
| `jev/api.py` | Request compilation and per-candidate prompt rendering |
| `jev/model.py` | Backbone plus LoRA plus Yes-minus-No scalar head |
| `jev/train.py`, `train_distributed.py` | NLL plus 0.1 Brier trainer; the 4-rank DDP trainer for the 27B |
| `jev/metrics.py` | `fit_temperature`, Brier, ECE, soft-target metrics |
| `jev/data.py`, `case_*.py`, `games.py`, `painting.py`, `case_reasoning.py` | The 13 rule-labelled generators |
| `reports/data-manifests/release-v2.json`, `docs/training-coverage-gaps.md` | Per-source counts and the author's gap diagnosis |
| `docs/jevbench-public.md`, `reports/jevbench-public-20260921/` | JevBench protocol and per-family results |
| `reports/fullpass-{2b,9b}-n1/` | Audited training runs, temperatures, baseline vs trained |
