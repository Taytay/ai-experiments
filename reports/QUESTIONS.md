# Open questions and critiques of the experiments

A running register of critiques of the knowledge-injection / knowledge-extraction experiments,
so that future iterations can say exactly which question they answer. Each item has a stable ID.
When an experiment addresses an item, add a `Status:` line pointing at the commit or result file
rather than deleting the item.

Categories:

| Prefix | Meaning |
|---|---|
| EVAL | Evaluation validity: does the metric measure what we think it measures |
| DATA | Dataset design and confounds in the synthetic merchant / creature data |
| MODEL | Choice of LLM, encoder, embedding model, and model scale |
| TRAIN | Training recipe: augmentation, learning rate, LoRA rank, epochs, mixing ratios |
| BASE | Missing baselines and alternative methods |
| STAT | Statistical rigor: sample sizes, seeds, confidence intervals |
| REAL | Transfer to the real use case (bank-statement categorization, personal labels) |
| REPORT | Gaps in how the report explains or presents results |
| LIT | Literature coverage: what the design is and is not grounded in |

Sources of items: **(O)** the project owner, 2026-09-14; **(R)** an independent review of
`reports/REPORT.md`, `reports/improvements.html` and `scripts/*.py`, same date; **(S)** the
literature survey in `references/SURVEY.md`, same date.

This file defines the items. It does not say what to do next: the ordered work queue, with
status per step, is `PLAN.md` at the repo root, and the literature behind each item is
`references/SURVEY.md` section 1 (same IDs). Add new items here first, then add a queue row.

---

## Owner questions, 2026-09-14, with current answers

### MODEL-1 (O) Why is the 7B model in none of the graphs? Was it trained at all?

**Answer.** No knowledge or curriculum run was ever done on 7B. Qwen2.5-7B appears only in
`scripts/bench_throughput.py` (forward-only scoring, LoRA bf16 at 64 and 512 tokens, QLoRA
4-bit, unsloth QLoRA), and therefore only in the throughput/memory table of `REPORT.md` section 7
and `evals/LEADERBOARD.md`. Every knowledge, universe, curriculum and embedding result is
Qwen2.5-0.5B, Qwen2.5-3B or all-MiniLM-L6-v2. `REPORT.md` line 439 lists "the same arms on 7B"
as a next step that the sweep did not run.

**Cost if we do it.** Arm C on 3B trained 1.25 M tokens in 24.6 min at 851 tok/s plus 7.1 min of
eval (`results/curriculum_Qwen2.5-3B_C.json`). Unsloth QLoRA on 7B benchmarked at 1,279 tok/s
and 10.3 GiB peak at 512-token sequences, so one arm on 7B is roughly 20 to 30 min of training
plus a slower eval pass. The eight-arm sweep is an afternoon on the RTX 3090.

**Why it matters.** Wei et al. (symbol tuning) and our own 0.5B vs 3B gap say label induction
is a scale phenomenon. The one-size result on 3B cannot tell whether interleaving still wins, or
is even needed, at 7B.

**Extension (R).** Add a third scale point (Qwen2.5-1.5B) and Instruct checkpoints, which follow
arbitrary label rules far better and would move the Timmy baseline; the merchant sweep never
leaves 0.5B; try one non-Qwen family (Llama-3.2-3B or Gemma-3-4B) to check family-specificity.

### EVAL-1 (O) In the multiple-choice prompts, is the model expected to emit a letter (A/B/C/D) or the answer text?

**Answer.** Neither is generated. Scoring is *cloze scoring*: the prompt ends at `Answer:`, the
option list is **never shown** to the model, and each candidate answer's full text is appended and
scored by the mean per-token log-probability of that text given the prompt
(`scripts/exp_curriculum.py:89-108`, `exp_knowledge_injection.py:58-78`). The argmax option is
the prediction. Example for L1 recall:

```
Question: What type is Javish?
Answer:
   candidates: " Spark" | " Frost" | " Bloom" | ... (8 type names, mean log-prob each)
```

For the label-induction ("Timmy") items the candidate labels do appear in the prompt as the demo
labels, so those items are closer to what the literature calls *hybrid* scoring.

**Consequences.**

- There is no A/B/C/D position bias, because there are no letters and no listed options.
- There *is* surface-form competition and an option-prior bias: a type name that is a common
  English word (or a shorter token sequence) has a higher prior, and the mean-over-tokens
  normalization only partially corrects for length. Holtzman et al. 2021 propose PMI scoring
  (divide by the option's probability under a domain-neutral prompt) to remove this; we do not.
- Format sensitivity is large and measured. Arm C scores 20.6 on bare L1 recall but 100.0 on the
  same facts in the trained sentence format (`L1_recall_fmt`); arm D scores 61.2 vs 96.9. The
  knowledge is present; the bare cloze format does not elicit it. This is the "which label
  format" problem the owner raised, in a different guise.
- L5 novel-choices sits at 8.8 to 17.5 across arms with chance 12.5, i.e. some arms are
  *below* chance, which is a signature of option-prior bias rather than of ignorance.
- No generation-based evaluation exists anywhere. We never check whether the model would
  actually *produce* the right answer when asked.

**Literature.** Alzahrani et al. 2024, "When Benchmarks are Targets", compare symbol, cloze and
hybrid scoring: symbol scoring has high selection bias, cloze scoring removes it but scores lower,
hybrid is the compromise. Zheng et al. 2024 document position bias in symbol scoring.
Balepur et al. 2025 (ACL) review the validity of MCQ evaluation altogether.

**Proposed fix.** Report three scores side by side for every level: current cloze, PMI-calibrated
cloze, and hybrid (options listed, answer text scored). Add a free-generation check on L1 and the
merchant tasks with exact-match or judge scoring. Prefer items whose options are matched in token
length. See EVAL-2 and EVAL-3.

**Status (2026-09-14):** PLAN step 2, REPORT.md section 10. Eight rules on the saved logits; `hybrid` (options listed, text scored) is the second line to report for induction and novel-label levels, `pmi_dc` for bare-format recall only; letter scoring (`mcf`) reverses the A-versus-C ordering on induction and collapses the ICL suite for every trained arm.

### LIT-1 (O) Was the literature searched for the best ways to inject and extract new knowledge and terminology?

**Answer.** Partly. `REPORT.md` sections 3 and 4 ground the design in: Allen-Zhu & Li
(Physics of LMs 3.1, knowledge must be augmented to be extractable; 3.2, recall vs manipulation),
Berglund et al. (reversal curse), Ovadia et al. and Soudani et al. (RAG vs fine-tuning),
Gekhman et al. (fine-tuning on unknown facts raises hallucination), EntiGraph / Yang et al. 2024
(synthetic continued pretraining), Biderman et al. 2024 (LoRA learns less, forgets less),
RAFT (Zhang et al. 2024), Wei et al. 2023 (symbol tuning), Hewitt 2021 and Mundra et al. 2024
(new-token initialization), Dagan et al. and Zhao et al. 2024 (vocabulary extension payoff).

Not covered before 2026-09-14, and not reflected in the design:

- MCQ scoring validity (see EVAL-1).
- Encoder / masked-LM approaches: ModernBERT-Large-Instruct (Clavié et al., Feb 2025) uses the
  MLM head to answer classification and multiple-choice questions and beats Qwen2-0.5B on MMLU
  (43.06 vs 33.7; the baseline is Qwen2, not Qwen2.5); PET (Schick & Schütze) and LAMA (Petroni
  et al. 2019) are the older cloze-probe and cloze-training lineage. See MODEL-2 and the WikiDYK
  benchmark (2505.12306) in SURVEY.md.
- Knowledge editing (ROME, MEMIT, AlphaEdit) and memory-layer approaches as alternatives to
  fine-tuning. See BASE-2.
- Self-Tuning (Zhang et al. 2024) and masked fine-tuning for knowledge injection in autoregressive
  LLMs (arXiv 2510.09885), both aimed at extraction rather than memorization.
- FineTuneBench (Wu et al. 2024): how well commercial fine-tuning APIs infuse knowledge.
- Paraphrase-count saturation. The often-quoted "about ten paraphrases" is a secondary citation.
  Primary sources: Ovadia et al. 2023 show monotone gains up to ten with no saturation analysis;
  Mecklenburg et al. 2024 show diminishing returns from 5x to 10x token-scaled QA caused by
  uneven fact coverage; Knowledge-Instruct (2504.05571) finds a plateau at about 3 paraphrases
  per atomic fact. Our 14 to 20 uniformly covering templates are past all three; see DATA-5 and
  SURVEY.md.

**Status (2026-09-14):** survey completed; 34 papers read and summarized under `references/papers/`,
synthesis and re-prioritization in `references/SURVEY.md`. Semantic Scholar and the arXiv API were
rate-limited during the search, so citation counts and citation-graph expansion are missing;
the follow-up list in SURVEY.md section 3 comes from the papers' bibliographies.

### MODEL-2 (O) Should an encoder trained with fill-in-the-blank be a training and extraction method?

**Answer.** Not considered so far; nothing in `scripts/` uses a masked-LM head. It fits this
project unusually well:

- Injection and extraction use the *same* objective. Training text
  `Javish is a [MASK]-type creature that lives in the [MASK].` and the eval question
  `What type is Javish? It is a [MASK]-type.` are the same task, so the format gap seen in
  EVAL-1 (20.6 vs 100.0) should largely vanish.
- Multiple choice is native: compare the MLM head's probabilities over the candidate verbalizer
  tokens at the `[MASK]` position. No option list, no letters, no length normalization (if each
  verbalizer is one token).
- ModernBERT-base (149 M) and -large (395 M) train faster than Qwen2.5-0.5B and support 8 k
  context, so field-guide entries fit in context for the with-context condition.
- Limits: candidate answers must be single tokens (or use a verbalizer map), so nonsense labels
  like `nutaltal` need a single-token substitute; encoders do not do free generation; and the
  in-context label induction ("Timmy") behaviour of a 400 M encoder is unknown, which is
  itself an interesting question.

**Proposed experiment.** Add arm F: ModernBERT-large, MLM objective on the same K texts with
attribute words masked, plus the same episodes rewritten as cloze items with single-token labels.
Score all ladder levels at the mask position. Compare with arm C on the same items.

**Extension (R).** The embedding encoder is trained contrastively only
(`exp_embed_vocab.py:208-230`, `exp_universe_embed.py:91-110`); there is no MLM
continued-pretraining stage on the augmented corpus before contrastive alignment, and no cloze
probe (`Elrholm sells [MASK]`). Run an MLM stage on ModernBERT-base, then the same contrastive
step; compare bank and held-out transfer plus a cloze recall probe against contrastive-only.

### MODEL-3 (O) Are we using the right LLM and the right embedding model?

**Answer.** The choices were pragmatic, not the result of a comparison.

- **LLM.** Qwen2.5-0.5B and -3B were picked because they fit full fine-tuning (0.5B) and LoRA
  (3B) on 24 GB with room for evaluation, and because unsloth supports them. No other family was
  tried. Candidates for the next round: Qwen3-4B / 8B, Llama-3.1-8B, Gemma-3-4B, and
  Qwen2.5-7B (MODEL-1). Different families differ in tokenizer coverage of invented names, which
  directly affects the subword-morphology results in section 4 and the confound in MODEL-4.
- **Embedding model.** all-MiniLM-L6-v2 (22 M params, 384-d, 256-token limit) was chosen for
  speed; the whole contrastive run takes 19 s. `REPORT.md` line 252 already recommends a stronger
  encoder for production. Candidates, all trainable with the same `SentenceTransformerTrainer`
  code: gte-modernbert-base (149 M, 8 k context), nomic-embed-text-v2-moe, bge-base-en-v1.5 or
  bge-m3, EmbeddingGemma-300M, Qwen3-Embedding-0.6B. ModernBERT gives one backbone for both
  MODEL-2 (MLM cloze) and the embedding experiments.
- **Open question:** does the "define labels by examples, not names" prototype result (63 to
  85 %) hold or improve on a stronger encoder, and does the new-token-vs-subword conclusion
  change when the encoder is cased, has an 8 k context and a larger vocabulary? (MODEL-4, MODEL-5)

**Proposed experiment.** Re-run `exp_universe_embed.py` and `exp_embed_vocab.py` unchanged on
gte-modernbert-base and EmbeddingGemma-300M; re-run curriculum arms base/A/C on Qwen2.5-7B and
Qwen3-8B.

---

## Independent review, 2026-09-14

The reviewer independently raised the 7B gap (folded into MODEL-1), option-length bias of cloze
scoring (EVAL-2, extends EVAL-1) and the missing masked-LM encoder arm (folded into MODEL-2).
IDs below continue the numbering started above.

### EVAL — evaluation validity

**EVAL-2 (R) Are the "chance" rows a constant predictor, and how much does option length bias every number?**
Options have 1 to 3 tokens for categories (`Gas & Auto`, `Pharmacy & Health`, `Telecom & Utilities`
have 3, `Groceries` 2, the rest 1) and 2 to 4 for type names; mean-per-token scoring favours
multi-token options because their later tokens are near-deterministic. Base `clean_category` and
most `bank_category` cells are exactly 8.3% = 10/120, which is what always-pick-the-same-option
scores. Per-item predictions are never saved (`exp_curriculum.py:123-137`), so nobody can check.
*Experiment:* save per-item argmax; print the predicted-label histogram per condition; add summed
log-prob and PMI-calibrated scorers on the same logits; report all three in one table.
**Status (2026-09-14):** PLAN step 2, REPORT.md section 10. Per-item argmax saved (section 9); histograms show the base model's yes/no rows are a constant No, arm B's a constant Yes, and every arm's unseen-species and probe rows a constant Tidewell; the below-chance L5 score was a length artifact (8.1 under mean, 33.8 under hybrid). Summed, PMI and Bayesian scorers are in one table (`reports/scorers_Qwen2.5-3B.md`).

**EVAL-3 (R) What does the model actually generate?**
Bare-format recall is 20 to 24% while trained-format recall is 100% on the same facts. The argmax
after `Answer:` is decided by the format prior on the first option token. The production task is
generative and no generation-based eval exists.
*Experiment:* greedy-decode 32 tokens for every L1/L3/category item, score exact and fuzzy match,
add an options-listed variant, and report three-way agreement.
**Status (2026-09-15):** done, PLAN step 3, REPORT.md section 12. Greedy decoding on every L1/L3 item, bare and with options listed, exact and fuzzy match, three-way agreement with the mean and PMI cloze rules. The 20% bare-format recall was the cloze scorer; the models write the right type in a sentence about 85% of the time. Merchant items decoded for the bare 0.5B only (the section 4 fine-tunes were never saved).

**EVAL-4 (R) Are the induction items identifiable, or is "Timmy from the weights" a learned default of "group by type"?**
`universe.py:172-183` uses one demo per group, so three labels are consistent with any attribute on
which the three demos differ; a "type" item is usually also a valid "habitat" item. Episodes group by
type 25% of the time (`universe.py:246`), so that prior is taught. The observed pattern (arm C:
59.4 type vs 35.0 habitat) matches a model that simply assumes type.
*Experiment:* two demos per group sharing exactly one attribute; stratify existing items by
identifiability; report only on identifiable items, per generating attribute.

**EVAL-5 (R) Is the ICL regression suite really out-of-distribution for the trained arms?**
Suite "symbol" labels come from the same `random_label` generator as training episodes
(`icl_suite.py:20,93`, `universe.py:262-274`), and the suite template equals replay template 0
(`icl_suite.py:39,45`). The "78 vs 51" ICL-preservation headline is partly in-distribution
familiarity. Forgetting is otherwise gauged on one 300-word coffee passage.
*Experiment:* suite variant with a new template and a disjoint label source (rare English words,
emoji, `NONSENSE`); add 200 MMLU items 5-shot and a WikiText-2 slice; re-score saved adapters
with `EVAL_ONLY`.

**EVAL-6 (R) Why does the base model score 42.7% on held-out-species induction with no context, a task that is unknowable?**
`heldout_induction` (`universe.py:326-343`) uses never-trained species; chance is 33. Either the
noise floor on 96 items is ±10 points or the items carry an artifact. The same level supports the
"31 to 47" morphology claim in section 8.4.
*Experiment:* empirical null per level by shuffling the answer index 1,000 times over saved
per-option scores; print the 95% null band next to chance in every table.
**Status (2026-09-14):** PLAN step 2, REPORT.md section 10: no question-free artifact on held-out induction (`unc` at chance); the predictions concentrate on one or two positions, see the null band in step 4.
**Status (2026-09-14):** PLAN step 4, REPORT.md section 11: the base model's held-out induction score is inside the gold-blind null band (25.0 to 43.8 on 96 items with its position preferences); so is every arm's except E.

**EVAL-7 (R) Are `sells` and `reverse` merchant-level or category-level tests?**
Distractors come from other categories (`merchants.py:136,147,157`), so "Which store sells canned
goods and frozen vegetables?" is answered by knowing which option is a grocer. The reversal-curse
conclusion rests on this task.
*Experiment:* hard-negative variants with same-category distractors; report easy and hard.

### DATA — dataset design and confounds

**DATA-1 (R) Is the "held-out partition" result real, given that weakness is a rotation of type?**
`WEAKNESS = dict(zip(TYPE_LIST, TYPE_LIST[3:] + TYPE_LIST[:3]))` (`universe.py:34`) makes weakness
groups identical to type groups. The executive summary sells "transfer to the held-out partition
(54%)"; the caveat appears only in 6.2 and the HTML. Literature footing: Allen-Zhu and Li's
bioS-couple result (2309.14316, Figure 6) shows correlated attributes get chained to each other
rather than to the name, so weakness-as-rotation-of-type is stored as a function of type.
*Experiment:* make weakness non-bijective or independent per species, or hold out habitat/region
from episodes instead; re-run arm C.

**DATA-2 (R) How much of the category result is recall vs the products-to-category bridge, and does the bridge survive product ambiguity?**
The 12 product pools are disjoint and every product is diagnostic (`merchants.py:10-35`). Real
merchants sell across categories.
*Experiment:* (a) P(category correct | products recalled) on the same merchants; (b) a v2 merchant
set with overlapping pools and 20% multi-category merchants.

**DATA-3 (R) Does morphology transfer hold for genuinely novel stems, and what is the slope against marker reliability?**
Probe names recombine the same 36 prefixes and suffixes as trained names (`universe.py:39-43,
52-59, 230-239`). Only two reliability points exist (morph_p = 0 and 0.7). Real drug stems are
often word-initial or multi-token.
*Experiment:* probes with unseen prefixes; sweep morph_p in {0.3, 0.5, 0.7, 0.9, 1.0}; a
prefix-marker universe; then the INN-stem to ATC-class replicate.

**DATA-4 (R) Why is the recommended fix (train on noisy renderings, normalize strings) never tested?**
Each merchant has one bank rendering, used only at test time (`merchants.py:51-56, 79-81`); names
are always intact. Real strings truncate (`SQ *THE COFF`, `AMZN Mktp US*2K3`).
*Experiment:* add renderings to training text (LLM and embedding), add truncation/abbreviation
templates, write the regex normalizer, report `bank_category` through it for every condition.

**DATA-5 (R) Is it template diversity or simply distinct token sequences that makes knowledge extractable, and how does it scale?**
"Augmented" is 14 fixed templates (`merchants.py:95-112`), ~20 for the universe; the doubling is
attributed to diversity from one two-point comparison. EntiGraph-style generated text is
discussed but not run. The literature reports saturation near 10 paraphrases per fact (LIT-1).
*Experiment:* at fixed 420 steps, 1 / 3 / 7 / 14 templated and 14 + 28 LLM-generated texts per
merchant; plot `clean_category` vs distinct texts.

### MODEL — model and architecture choice

**MODEL-4 (R) Is the bank-string failure a tokenizer artifact rather than a capability limit?**
Verified 2026-09-14 with the Qwen2.5 tokenizer: `Elrholm` -> `El|r|holm`, `ELRHOLM` -> `EL|RH|OL|M`;
across all 120 merchants only 50 of 397 clean-name tokens (12.6%) survive in the uppercase form.
Even the in-context oracle (14.2%) asks the model to match strings that share no tokens. MiniLM's
70.8% transfer exists because its WordPiece is *uncased*. "Subword pieces carry knowledge across
formats" is a property of uncased tokenizers, not of subwords in general.
*Experiment:* re-run LLM in-context and `bank_category` evals with title-cased bank strings; run
the embedding experiment on a cased encoder (bge-base-en-v1.5, gte-modernbert-base,
Qwen3-Embedding-0.6B) and report whether 70.8 survives.
**Status (2026-09-14):** first half done, PLAN step 5, REPORT.md section 13: token survival measured for both tokenizers (Qwen 10.6% under uppercasing, 95.2% title-cased; MiniLM 96.2%); title-cased and name-restored bank strings do not lift the 0.5B with-context score (14.2 -> 15.8 / 15.0), so the failure is the transaction format, not the casing. The cased-encoder half (bge-base, Qwen3-Embedding) waits for step 13.

**MODEL-5 (R) Is the new-token failure specific to mean pooling and to a 22 M, 6-layer model?**
The diagnosed mechanism is mean-pooling dilution; CLS-pooled or late-interaction encoders would not
dilute a single identity token the same way.
*Experiment:* repeat `ft_subword` vs `ft_newtok_mean` with CLS pooling on bge-small and with a
ColBERT-style token-max-sim retriever.

### TRAIN — training recipe and hyperparameters

**TRAIN-1 (R) What are the mixture fractions in loss tokens rather than in sequences?**
Knowledge texts use full-sequence loss over ~25 tokens; episodes and replay use answer-only loss over
2 to 4 tokens (`exp_curriculum.py:169-177`); HF loss averages over all non-ignored tokens. "K 45 / E
40 / R 15" by sequence count is plausibly ~85% K by gradient weight; the E fraction was never swept.
*Experiment:* log per-stream label-token counts per step; per-stream loss normalization; sweep
E in {0.2, 0.4, 0.6} at fixed K+E.

**TRAIN-2 (R) Is "sequential loses" a property of staging or of the shared LR schedule?**
Arm D runs both phases under one warmup + linear decay (`exp_curriculum.py:195, 201`), so phase 2
(episodes) trains at half the peak LR or less.
*Experiment:* D with a restarted schedule per phase; D with constant LR; D with 10% knowledge
replay in phase 2. If any matches C, the recommendation becomes "interleave or replay".

**TRAIN-3 (R) When during training is knowledge acquired and when is ICL damaged?**
Every arm is evaluated once at 800 steps; only train loss is logged (`exp_curriculum.py:218-222`).
*Experiment:* evaluate a 400-item subsample every 200 steps (the `SMOKE` path already subsamples)
and plot recall, Timmy, ICL-symbol and perplexity vs steps per arm.

**Status (2026-09-15):** PLAN step 11, REPORT.md section 15. `PERIODIC=N` in `exp_curriculum.py` scores a fixed subsample plus 200 frozen ARC-Easy items (`K_arc_easy`, `data/processed/known_facts_v1.json`) every N steps; `scripts/curves.py` tabulates. Knowledge lands by step 200 (A, D) or 400 (C, 45% mix); A's ICL loss is complete at step 200 and 15% replay prevents it from the start; ARC-Easy drops about ten points by step 200 in every arm and stays down; induction from the weights appears in the last 200 steps and needs the full 160 items per point. Follow-ups: PLAN rows 24 (corpus perplexity) and 25 (TRAIN-7).

**Status (2026-09-15, step 24):** REPORT.md section 16. The perplexity leg of the curve was the instrument: on a frozen WikiText-2 slice (`data/processed/corpus_ppl_v1.json`, 29 paragraphs, 4,972 tokens, `L7_ppl_wikitext` in every evaluation and periodic point from now on) two runs of arm C agree to 0.03 nats where the paragraph read 14.9 and 30.3. Final costs over the base: A +1.40 nats per token, Cn +1.08, C/D/E +0.81 to +0.90, B +0.37, P +0.09. The mid-training perplexity curve itself starts with row 25's run.

**TRAIN-4 (R) Which hyperparameters were never varied, and is LoRA the right vehicle for injection?**
Rank 64, alpha 128, no dropout, wd 0, betas (0.9, 0.95), 800 steps are fixed across all arms
(`exp_curriculum.py:52, 181-195`); the only sweeps are LR. 3B full FT OOMed with fp32 master
weights. WiSE-FT was tried at alpha 0.5 only; general-text replay is absent from every merchant run.
*Experiment:* on arm C, sweep r in {16, 64, 256}, lr in {5e-5, 1e-4, 2e-4}, steps in {400, 1600};
one 3B full-FT run with 8-bit AdamW and bf16 weights; WiSE-FT alpha in {0.3, 0.7}; 10% FineWeb
replay in `ft_aug`.

### BASE — missing baselines and alternative methods

**BASE-1 (R) Why is RAFT recommended twice but never run?**
The closest thing run is episodes with oracle field-guide entries at `ctx_frac=0.5`
(`universe.py:318-321`), no distractors, no actual retriever.
*Experiment:* arm C-RAFT with top-3 entries retrieved by the fine-tuned MiniLM over noisy
renderings (some wrong), answer-only loss; evaluate with retrieved context and with none.

**BASE-2 (R) How do dedicated knowledge-editing methods compare on this ladder?**
120 merchant relations and ~680 species relations are in the range where ROME/MEMIT/AlphaEdit
operate on 3B models, and they claim locality, which is precisely arm A's failure.
*Experiment:* MEMIT via EasyEdit on Qwen2.5-3B for all species; full ladder plus ICL suite, side by
side with arms A and C.

**BASE-3 (R) Can the with-context ceiling be distilled into the weights directly?**
With-context numbers are the ceiling everywhere (98.8 recall, 48.8 Timmy for base 3B). Context
distillation is the most direct method and is absent.
*Experiment:* teacher distributions with field guide in context, student trained on the same
prompts without context (KL or hard labels), scored on the ladder.

**Status (2026-09-15):** PLAN step 7, REPORT.md section 14. Arm P (`scripts/exp_distill.py`: KL at T=2 to the base model reading the field-guide entries, arm C's mixture and budget) injected nothing: trained-format recall 13.8 (base 11.9), every induction level inside its null band, ppl 8.64 (base 8.54); the ICL suite gained 16 points, from the 15% hard-label replay. Diagnosis (`scripts/diag_distill_teacher.py`, Table 14.2): even with the entry in context the teacher puts 6% of its next-token mass on the type token in the training sentence (0.3% at T=2), so the 98.8 with-context recall is a ranking among eight options, not a distribution worth imitating; the student matched the teacher (KL 2.17 to 0.28) and learned no facts. Open: PLAN row 27 tries option-renormalised targets at T=1; if that fails too, the answer is no.

**BASE-4 (R) Are the embedding results compared against trivial baselines and against the LLM on identical items?**
No linear probe, SetFit or kNN baseline exists; the 85% prototype result (300 random trials) and the
59% LLM Timmy result (160 ladder items) are on different item sets.
*Experiment:* score prototypes and LLM arms on the same 160 L3/L4 items; logistic-regression head on
frozen and fine-tuned embeddings for the 12-way and 8-way tasks.

**BASE-5 (R) Was the vocabulary-expansion baseline given a fair chance?**
Runs use random or mean-of-subword init only (`exp_knowledge_injection.py:134-145`,
`exp_embed_vocab.py:233-244`), unfreeze everything at once, and never add uppercase forms as
training tokens (aliasing is post-hoc, `exp_knowledge_injection.py:148-160`).
*Experiment:* N(mu, Sigma) init; embedding-row-only warm-up before unfreezing; both cased forms as
tokens; then decide.

### STAT — statistical rigor

**STAT-1 (R) What is the seed-to-seed noise, and are the headline gaps larger than it?**
Every run is `SEED = 0`, single run per arm. The one replicate (same 3B config, backend swapped)
moved Timmy LoRA+ctx from 34.4 to 43.1 and real-name induction from 60.0 to 78.8
(`evals/LEADERBOARD.md:144-149`). C vs Cn (78.1 vs 70.8), D vs C recall (96.9 vs 100), and the
3/21/9-point "sequential loses" numbers are inside or near that swing.
*Experiment:* 3 seeds each for A, C, D (~75 min per arm-set on 3B); mean ± sd; claim only
differences clearing 2 sd.

**STAT-2 (R) What are the confidence intervals, and why are per-item predictions not saved?**
n = 160 per level gives ±7.7 points at 95%; held-out induction has 96 items (±10); L6 unseen recall
has 24 items, so 12.5% is 3/24; each ICL dataset has 48 items per label style. `accuracy()`
discards per-item outcomes, blocking bootstrap CIs and paired McNemar tests.
*Experiment:* persist per-item scores; bootstrap CIs and paired tests; raise `n_per_level` to 500
(eval is ~7 min per 4,400 items).
**Status (2026-09-14):** PLAN steps 1 and 4, REPORT.md sections 9 and 11. Per-item, per-option scores are saved for every run (`results/per_item/`); `scripts/ci_table.py` gives bootstrap CIs, null bands and paired McNemar tests, `curriculum_summary.py --ci` prints them in the tables. n_per_level stays 160 for now (raising it means a new item-set version).

**STAT-3 (R) Are cross-table comparisons paired at all?**
The ladder is regenerated from a seed each run; adding `L1_recall_fmt` shifted every later item, so
6.2 vs 6.4 base rows differ (37.5 vs 35.6). The ICL suite caches items to JSON; the ladder does not.
*Experiment:* freeze `ladder`, `probes`, `heldout_induction` to `results/*.json` with a version tag
and record the hash in the tracker config.
**Status (2026-09-14):** done, PLAN step 1, REPORT.md section 9. `ladder`, `probes`, `heldout_induction` frozen to `data/processed/<set>_v1[_morph].json` with ids and sha256; `items_version` / `items_sha` recorded in every run's tracker config; `uv run python -m ai_experiments.items check` reports generator drift.

### REAL — transfer to the real use case

**REAL-1 (R) Does any of this hold when merchants are not opaque and strings are real?**
Base models already know head merchants; the long tail is noisy strings for known and unknown
merchants; frequency is Zipfian. No real or realistic transaction data appears anywhere.
*Experiment:* realistic synthetic set from real merchant names with Zipf frequencies and real-style
truncations, or an anonymized export; accuracy by frequency bucket and by known/unknown merchant.

**REAL-2 (R) What does end-to-end retrieval achieve, as opposed to the oracle?**
The "RAG ceiling" inserts the exact fact verbatim; the 70.8% embedding number is nearest
*category*, not merchant-record retrieval. Recall@k of bank string to merchant record is never
measured.
*Experiment:* index the 120 (then 10k) records with the fine-tuned MiniLM; recall@1/5 from bank
strings; feed top-1 (right or wrong) to the LLM; end-to-end category accuracy.

**REAL-3 (R) Does 100% recall survive 1k to 10k entities in one adapter?**
Section 7.3 extrapolates 10k-merchant training time linearly from 136 species; rank-64 capacity and
interference between thousands of similar names are untested.
*Experiment:* `U.build(n_per_type=125)` and 625 (1k and 5k species), same recipe, 1x and 5x steps;
trained-format recall and Timmy vs entity count.

**REAL-4 (R) Is the prototype classifier tested under the conditions personal categories actually have?**
Prototype eval is 3-way, balanced, names-only, 300 trials (`exp_universe_embed.py:72-87`); real
users have 10 to 30 imbalanced categories and transactions carry amount and date. The
"pick the space where the user's examples cluster" idea is never implemented.
*Experiment:* 12-way imbalanced prototype eval with k in {1, 3, 10}; concatenate amount-band and
weekday features; implement the cluster-tightness attribute selector on habitat-vs-type items.

### REPORT — clarity and presentation gaps

**REPORT-1 (R)** The executive summary states "transfer to the held-out partition (54%)" without
the weakness-equals-type caveat (DATA-1). Drop it or state "the weakness attribute, a type
replicate in this universe".
**Status (2026-09-14):** done, PLAN step 6. The summary bullet and 8.3 point 2 now say "the weakness
attribute ... a type replicate in this universe and not evidence of latent-partition transfer".

**REPORT-2 (R)** Every table should carry a 95% CI column and section 8 a one-line noise-floor
statement; the 18-point backend-replicate swing is never mentioned in section 8 (STAT-1).
**Status (2026-09-14):** PLAN step 4, REPORT.md section 11: noise-floor statement (item-count half-widths, the 2 to 3 point same-weights floor of section 9.2, the 18-point backend replicate), Table 11.1 with intervals and p-values for every section 8 claim, and `--ci` cells in the generated tables. Tables 8.1 to 8.3 themselves are left as the Windows-run numbers.

**REPORT-3 (R)** Base at 42.7% on unknowable held-out induction is unexplained (EVAL-6); "RAG
ceiling" should be renamed "oracle context" and "RAG" reserved for REAL-2.
**Status (2026-09-14):** rename done, PLAN step 6: REPORT.md 4.3.2 and 6.2 say "oracle context" and
point at REAL-2; the code docstrings match. The unexplained 42.7% base score on held-out induction is
EVAL-6 and waits for the empirical null band (PLAN step 4).

**REPORT-4 (R)** Recipe fractions are given in sequences while the loss averages over tokens
(TRAIN-1); the arm D schedule confound (TRAIN-2) is not disclosed in the design table.
**Status (2026-09-14):** done, PLAN step 6. REPORT.md 8.1 has a table of sequence, token and
loss-bearing-token fractions per arm (arm C is 84% knowledge text by loss-bearing tokens) and a
paragraph on the single learning-rate schedule spanning arm D's two phases; 8.3 point 3 points at it.

### Added by the literature survey, 2026-09-14

**TRAIN-5 (S) Does masked fine-tuning of the decoder remove the need for paraphrases and fix the reversal curve?**
Pan et al. (2510.09885) reframe each document as "here is the passage with 5-95% of tokens
masked; recover it", trained with the ordinary AR loss on the reconstruction. On Llama-3.2-3B-
Instruct and Qwen3-4B-Instruct this gives 0.9+ forward *and backward* recall with a single
rendering per fact, where plain fine-tuning with paraphrases gives 0.95 forward and 0.04
backward; a random-token control collapses, so it is not generic augmentation. Not tested below
3B, on base checkpoints, or with LoRA.
*Experiment:* Qwen2.5-3B-Instruct, LoRA r=64, one template per entity, masked-FT objective on
the K stream with episodes and replay unchanged; score bare L1, `L1_recall_fmt`, the merchant
`reverse` task and the ICL suite against arm C. About 2x arm C in tokens.

**TRAIN-6 (O) Is the knowledge stream overhead-bound, and would sequence packing change the results as well as the speed?**
Knowledge texts average 25 tokens and are trained unpacked in micro-batches of 8, so a step is
mostly launch overhead and padding: arm A retrains at about 280 tokens/s with the 3090 at 35 to
60% utilisation, 12 to 26% memory bandwidth and 240 of 350 W (measured 2026-09-15 during the
step 11 periodic run), against 850 tokens/s for arm C's long episodes and 69% MFU on long
sequences in section 7.1. The training loop in `exp_curriculum.py` is hand-written (samples per
stream, pad to the longest, call the model), so unsloth only speeds up the kernels; its packing
comes with TRL's `SFTTrainer` (`packing=True`, and the padding-free variant that keeps packed
examples' attention separate). Packing is not a free speed-up here: without per-sequence masking
and position ids, facts about one species leak into the loss of the next; and averaging the loss
over a 768-token pack of about 30 facts instead of 8 padded ones changes the effective mixture
weights that section 8.1 measured (84% knowledge by loss-bearing tokens), so a packed run is a
recipe change, not a drop-in.
*Experiment:* PLAN row 26, two parts. (1) Move the K stream (or the whole loop) to unsloth's `SFTTrainer`
with `packing=True` so the mixture, masking and position handling are the trainer's, keeping the
per-step mixture draw by tokens as step 9 defines it; confirm on one packed micro-batch that
attention is block-diagonal (attention weights across pack boundaries are zero) and that
positions restart at each boundary. (2) Retrain arm A packed and unpacked at the same token
budget and seed; both must score within the section 9.2 noise floor on the frozen ladder, and the
packed run should report tokens/s, MFU and the loss-bearing-token fractions per stream. Only
then use packing for the sweeps (steps 10, 15, 17) where the time saving matters.
**TRAIN-7 (R) Does general-text replay protect the knowledge the model already had?**
Section 15: every arm loses 11 to 16 ARC-Easy points in the first 200 steps and never recovers them; 15% replay of
ICL-suite episodes protects the ICL suite completely but does nothing for ARC-Easy (arm D, replay from step 0, drops
the most). Replay protects what is replayed.
*Experiment:* PLAN row 25: add a few percent of pretraining-style text (a FineWeb or WikiText slice) to arm C's mix, run
with `PERIODIC=200`, and read `K_arc_easy` and the corpus perplexity of row 24 against arm C's curve; then the same with
a lower learning rate on the knowledge stream, since the loss is paid while the facts are written.

**Status (2026-09-15):** PLAN step 25, REPORT.md section 17. Yes: arm Cg (arm C with 5% of sequences, 27% of the loss, as WikiText-2 train paragraphs) keeps the WikiText perplexity at +0.09 nats over the base (C: +0.84) at every checkpoint and ARC-Easy at 71.0 against C's 59.0 (+12 paired, p = 4e-5; base 73.5), with recall at 100 and the ICL suite at 81.2. Induction from the weights fell 10 to 21 points on three levels, confounded with the episode share the general text displaced; step 9 runs the 5% out of the knowledge stream instead. The lower-learning-rate variant was not run: the replay alone removes the perplexity cost.
