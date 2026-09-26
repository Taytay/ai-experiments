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

**Status (2026-09-16):** done, PLAN step 15, REPORT.md 26. Arms base / A / C / D on Qwen2.5-1.5B and Qwen2.5-7B (QLoRA), seed 0. The mixture's induction gain (+26 / +22 / +19 on Timmy at 1.5B / 3B / 7B) and ICL preservation (+25 / +31 / +31) hold at every scale; natural-label ICL is flat everywhere and ARC-Easy loses 11 to 20 points for every trained arm at every scale. Knowledge-only's manipulation advantage shrinks from 30 points at 1.5B to 4 at 7B; the sequential arm at 7B forgets a third of the facts in the episode phase. A 7B arm costs about an hour (QLoRA training 10 to 22 min plus evaluation), a 1.5B arm ten minutes. Not run: Instruct checkpoints and a non-Qwen family.

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

**Status (2026-09-16):** answered for this budget, PLAN step 18, REPORT.md 29. Encoder-only: ModernBERT-large trained with the MLM head and scored with the letter-at-mask protocol (every ladder answer is multi-token) is at chance on every level and loses natural-label ICL; not a vehicle at 395M and 800 steps. Encoder-decoder: Flan-T5-large span prediction at the decoders' rate learns the sentence template and not the facts at 160, 1,000 and 5,000 species (recall 11 to 17, generation 16 to 27), in both the plain and the trained span scoring format; T5Gemma under full FT at 1e-4 is damaged. The follow-ups reverse the encoder-decoder verdict: the rate was the whole difference. Flan-T5-large under full FT at 3e-4 recalls the 160 species at 98.1 / 100 (option scoring / generation), with no label induction (Timmy at the base) and bare recall twice the decoder's; T5Gemma under a rank-64 adapter at 1e-4 recalls 97.5 / 93.1. Both lose natural-label ICL and ARC-Easy in a way the decoder mixture does not, so the encoder-decoder is an injection vehicle and a worse general model after it. Unrun: Flan-T5 at 1,000 and 5,000 species at 3e-4 (the WikiDYK capacity test), the Flan-T5 adapter at 1e-3 or above.

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

**Status (2026-09-16):** answered for the embedding side, PLAN step 13, REPORT.md 24. On the frozen induction items the LLM arms answer, fine-tuned bge-base-en-v1.5 and Qwen3-Embedding-0.6B score 92 to 94 (Timmy), 86 to 91 (k=4) and 72 to 80 (habitat) from their weights against arm C's 62 / 52 / 29 (71 / 61 / 66 with the field guide); MiniLM's 79 / 63 / 53 already beats the LLM's weights. The section 4 bank-string transfer (70.8) is uncased-tokenizer behaviour: 84.4 on bge-base (uncased, not cased as the survey said), 26.0 on the cased Qwen3 tokenizer. The LLM side (Qwen2.5-7B etc.) is still PLAN row 15.

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
**Status (2026-09-18):** answered, PLAN step 20, REPORT.md 33. Two findings before any model: weakness is a bijection of type in the universe (DATA-1's point, row 23), so the L4 weakness level was a second type level and the "transfer to weakness" of sections 8 and 15 was not a transfer; and three quarters of the v1 items are silent on the competing rule. Stratified (`scripts/induct_strata.py`), arm C answers habitat items by type (12.5 where the rules conflict, 47.1 where they coincide). New identifiable items (`universe.induction_v2`: two demos per group sharing only the generating attribute, `induction2_v1.json`, scored in every run from now on): the type induction from the weights is real and identifiable (C 66.2, D 66.9, 1,600 steps 82.5, base 41.2), habitat / diet / region from the weights are at chance for every arm but D (45 to 46), and no arm ever picks the unseen label (0.0 for all, base included): the rule is copied from a shown group, never applied to an unshown one. With the entries in context every attribute reads 60 to 86.


**EVAL-5 (R) Is the ICL regression suite really out-of-distribution for the trained arms?**
Suite "symbol" labels come from the same `random_label` generator as training episodes
(`icl_suite.py:20,93`, `universe.py:262-274`), and the suite template equals replay template 0
(`icl_suite.py:39,45`). The "78 vs 51" ICL-preservation headline is partly in-distribution
familiarity. Forgetting is otherwise gauged on one 300-word coffee passage.
*Experiment:* suite variant with a new template and a disjoint label source (rare English words,
emoji, `NONSENSE`); add 200 MMLU items 5-shot and a WikiText-2 slice; re-score saved adapters
with `EVAL_ONLY`.
**Status (2026-09-18):** answered, PLAN step 20, REPORT.md 33. The suite was not in-distribution-inflated: on a variant with a template no replay episode uses and rare-word symbol labels (`icl_suite.suite2_items`, `icl_suite2_v1.json`) every arm reads within a few points of its v1 number (C 76.5 vs 75.5 symbol, 86.5 vs 86.5 natural; D 81.2 vs 80.2; Cg 80.7 vs 81.2), the two exceptions being the 1,600-step run (74 vs 80.2) and the plain on-policy-distilled adapter (66.2 vs 75). The 200-item 5-shot MMLU slice (`mmlu_v1.json`, K_mmlu) agrees with ARC-Easy as the forgetting proxy: base 50, C 41, D 41, Cg 48, C + OPD + replay 49.5. The WikiText slice was step 24.


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
**Status (2026-09-15):** answered, PLAN step 8. `universe.reverse_items` builds both levels: L8_reverse_easy has distractors of another type (the merchant `reverse` construction; any arm that knows each option's type answers it, arm A 87.5), L8_reverse_hard has same-type distractors differing in diet or region (arm A 28.8, masked fine-tuning 48.1). The merchant `reverse` was the category-level test; the entity-level test is the hard level and is now in the frozen set `reverse_v1.json`.

### DATA — dataset design and confounds

**DATA-1 (R) Is the "held-out partition" result real, given that weakness is a rotation of type?**
`WEAKNESS = dict(zip(TYPE_LIST, TYPE_LIST[3:] + TYPE_LIST[:3]))` (`universe.py:34`) makes weakness
groups identical to type groups. The executive summary sells "transfer to the held-out partition
(54%)"; the caveat appears only in 6.2 and the HTML. Literature footing: Allen-Zhu and Li's
bioS-couple result (2309.14316, Figure 6) shows correlated attributes get chained to each other
rather than to the name, so weakness-as-rotation-of-type is stored as a function of type.
*Experiment:* make weakness non-bijective or independent per species, or hold out habitat/region
from episodes instead; re-run arm C.
**Status (2026-09-19, partial):** the stratification of PLAN step 20 (REPORT.md 33.1) confirms the point from the saved per-item files: the weakness level replicates the type level within noise for every arm, and arm C answers habitat items by the type rule. The re-run with an independent weakness (row 23) is still to do.
**Status (2026-09-19):** answered, PLAN step 23, REPORT.md 39. With every species' weakness drawn independently of its type (`universe.build(weakness="independent")`, names and the other attributes unchanged, items `_wind`), arm C's weakness induction reads 33.8 (chance 33.3) against 60.0 / 52.5 on the original universe, and every other measure is unchanged (recall 100, yes/no 75, Timmy 61.9, k=4 49.4, ICL 78 / 87, ARC 61.5, ppl 21.6). The held-out-partition transfer was the type rule; there is none. Not done: the same universe for arms A, B, D; a trained-format weakness recall level.

**DATA-2 (R) How much of the category result is recall vs the products-to-category bridge, and does the bridge survive product ambiguity?**
The 12 product pools are disjoint and every product is diagnostic (`merchants.py:10-35`). Real
merchants sell across categories.
*Experiment:* (a) P(category correct | products recalled) on the same merchants; (b) a v2 merchant
set with overlapping pools and 20% multi-category merchants.
**Status (2026-09-19):** answered, PLAN step 36, REPORT.md 35. (a) On the v1 set P(category correct | products recalled in the 4-way `sells` item) is 45.2 against 41.7 unconditionally: the category failures are failures of the products-to-category bridge, not of recall. (b) On the v2 set (`merchants.build_v2`: each category's pool shares two products with the next category, 20% of merchants sell one product of another category) category accuracy falls 41.7 to 28.3 while `sells` (55.0) and reverse (43.3) hold; multi-category merchants read 16.7 against 31.2 for single-category ones; the untouched embedding model's mapping of a held-out merchant's description falls from 100 to 79 to 83. The bridge is exact when products are diagnostic and breaks when they are not.


**DATA-3 (R) Does morphology transfer hold for genuinely novel stems, and what is the slope against marker reliability?**
Probe names recombine the same 36 prefixes and suffixes as trained names (`universe.py:39-43,
52-59, 230-239`). Only two reliability points exist (morph_p = 0 and 0.7). Real drug stems are
often word-initial or multi-token.
*Experiment:* probes with unseen prefixes; sweep morph_p in {0.3, 0.5, 0.7, 0.9, 1.0}; a
prefix-marker universe; then the INN-stem to ATC-class replicate.
**Status (2026-09-19):** answered, PLAN step 21, REPORT.md 34. The transfer holds for genuinely novel names: probes whose only trained part is the marker (`universe.probes_v2`, an unseen prefix beside the suffix marker) read 95.8 on the 0.7 universe, the same as the recombined-part probes (95.8; the section 8 adapter re-scored 93.8 / 95.8), plain controls 14.6 to 16.7, chance 12.5. The slope is against marker *coverage* (a present marker never lies in this universe, so `morph_p` is coverage, not reliability): 60.4 / 79.2 / 95.8 / 97.9 / 100 at 0.3 / 0.5 / 0.7 / 0.9 / 1.0, and induction over held-out species without context 38.5 / 31.2 / 50.0 / 74.0 / 85.4. A prefix-marker universe (`PREFIX_MARKER`, stems at the front of the name) gives 100 / 97.9 with the rest of the ladder unchanged. Not run: a reliability sweep (markers that sometimes lie) and the INN-stem to ATC-class replicate on real drug names.


**DATA-4 (R) Why is the recommended fix (train on noisy renderings, normalize strings) never tested?**
Each merchant has one bank rendering, used only at test time (`merchants.py:51-56, 79-81`); names
are always intact. Real strings truncate (`SQ *THE COFF`, `AMZN Mktp US*2K3`).
*Experiment:* add renderings to training text (LLM and embedding), add truncation/abbreviation
templates, write the regex normalizer, report `bank_category` through it for every condition.
**Status (2026-09-19):** answered, PLAN step 36, REPORT.md 35. Tested on section 4's two recipes with replicated baselines (embedding 70.8, 0.5B 12.5 on the bank string). Embedding route: six card-statement renderings per merchant in training (`merchants.rendering_texts`) take the never-seen bank string from 70.8 to 100; the regex normaliser (`merchants.normalize`) alone also gives 100, and 92.7 on a new truncated string (`bank_hard_string`, name cut to 8 characters) against 44.8 raw. 0.5B decoder: the normaliser lifts the bank string from 12.5 to 44.2, its own clean-name level (41.7), and the truncated string from 10.8 to 36.7; renderings in training add 6 points raw and nothing normalised, and cost 12 points of reverse lookup. Held-out merchants stay at chance in every cell. Not run: renderings for the 3B decoder or the section 24.7 encoders; a learned normaliser.


**DATA-5 (R) Is it template diversity or simply distinct token sequences that makes knowledge extractable, and how does it scale?**
"Augmented" is 14 fixed templates (`merchants.py:95-112`), ~20 for the universe; the doubling is
attributed to diversity from one two-point comparison. EntiGraph-style generated text is
discussed but not run. The literature reports saturation near 10 paraphrases per fact (LIT-1).
*Experiment:* at fixed 420 steps, 1 / 3 / 7 / 14 templated and 14 + 28 LLM-generated texts per
merchant; plot `clean_category` vs distinct texts.
**Status (2026-09-16):** answered, PLAN step 16, REPORT.md 27 (universe, arm A, Qwen2.5-3B, fixed 12,800 sequences). Recall in the trained format saturates at three templates (100; one template 73.8). Manipulation does not come from paraphrases at any count or kind (chance for 1 to 14 templates, permutations, reverse statements, LLM sentences) but from the six negative and comparative texts per species (98.8 / 93.8 with them). Hard backward recall comes only from reverse-direction sentences (69.4; everything else at chance). More variety at a fixed budget costs more general ability (WikiText 51 at 14 templates, 83 with attribute permutations, 27 with LLM prose; ARC-Easy at the base for 3 to 7 templates). The merchant `clean_category` version was not run; the universe ladder gives the same question a richer readout.

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
**Status (2026-09-16):** answered, PLAN step 13, REPORT.md 24.4. Not pooling-specific: with new merchant tokens, CLS-pooled bge-base reads the bank strings at 30.2 (84.4 without new tokens) and last-token-pooled Qwen3-Embedding at 12.5 (26.0), the same failure as mean-pooled MiniLM (28.1 vs 69.8). A ColBERT-style retriever was not run.

### TRAIN — training recipe and hyperparameters

**TRAIN-1 (R) What are the mixture fractions in loss tokens rather than in sequences?**
Knowledge texts use full-sequence loss over ~25 tokens; episodes and replay use answer-only loss over
2 to 4 tokens (`exp_curriculum.py:169-177`); HF loss averages over all non-ignored tokens. "K 45 / E
40 / R 15" by sequence count is plausibly ~85% K by gradient weight; the E fraction was never swept.
*Experiment:* log per-stream label-token counts per step; per-stream loss normalization; sweep
E in {0.2, 0.4, 0.6} at fixed K+E.
**Status (2026-09-15):** closed, PLAN step 9. REPORT.md 21. The pooled token mean gives the knowledge stream 84% of the gradient (Table 21.1); the same batches at the nominal 45% (arm M0) do not memorise the facts in 800 steps (recall_fmt 33.8, manipulation at chance, flat curve), and no loss-level episode weight in {.2, .4, .6} matches arm C from the weights. The extra episode weight shows only with context (M0 is the best arm on in-context induction and reverse lookup). Per-stream token and loss-bearing-token counts are logged in every run (`tok_*`, `lb_*`).

**TRAIN-2 (R) Is "sequential loses" a property of staging or of the shared LR schedule?**
Arm D runs both phases under one warmup + linear decay (`exp_curriculum.py:195, 201`), so phase 2
(episodes) trains at half the peak LR or less.
*Experiment:* D with a restarted schedule per phase; D with constant LR; D with 10% knowledge
replay in phase 2. If any matches C, the recommendation becomes "interleave or replay".
**Status (2026-09-16):** closed, PLAN step 12. REPORT.md 23. Three seeds each: with the schedule restarted per phase (Dr) or held constant (Dc), the episode phase erases the facts (formatted recall 28.1 +- 16.4 and 41.0 +- 18.8 against arm D's 91.5; induction from the weights back to arm A's level), so arm D's decaying schedule was protecting phase 1, not holding phase 2 back. With 10% knowledge replay in phase 2 (Dk) recall is 99.8 +- 0.3 and every level matches arm C within the seed noise, with lower WikiText perplexity (19.6 vs 22.6, clear). The recommendation is "interleave or replay"; the staging itself costs nothing once the earlier phase is replayed.

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
**Status (2026-09-16):** answered, PLAN step 17, REPORT.md 28 (arm C, Qwen2.5-3B, seed 0). Rank, learning rate and step count form one axis: rank 16, lr 5e-5 and 400 steps store the facts (recall 100 / 100 / 95.6) and cannot use them (manipulation and induction at the base); lr 2e-4 uses them best of the 800-step runs and forgets most. 1,600 steps at the recipe's rate improves every from-the-weights number (97.5 / 91.2; 81 / 72 / 77) and the general-ability costs at once: the recipe's 800 steps under-trained the mixture. Rank 256 matches 64 at higher perplexity and memory; MLP-only LoRA matches all-linear at lower cost; full fine-tuning (8-bit AdamW, 19.7 GiB) erases the model at 1e-4 and learns nothing at 1e-5. LoRA is the right vehicle here. WiSE-FT alphas and merchant FineWeb replay were not run.

### BASE — missing baselines and alternative methods

**BASE-1 (R) Why is RAFT recommended twice but never run?**
The closest thing run is episodes with oracle field-guide entries at `ctx_frac=0.5`
(`universe.py:318-321`), no distractors, no actual retriever.
*Experiment:* arm C-RAFT with top-3 entries retrieved by the fine-tuned MiniLM over noisy
renderings (some wrong), answer-only loss; evaluate with retrieved context and with none.
**Status (2026-09-16):** run, PLAN step 14, REPORT.md 25.1. Arm C (oracle-only context in training) does not fall below its no-context numbers with retrieved context; it gains, because the fine-tuned MiniLM's top-3 per name are the species' paraphrased training texts (Timmy 78.1 retrieved vs 70.6 oracle vs 61.9 bare). Arm Cr (episodes with gold p 0.8 + 2 retrieved distractors) trains to arm C's from-the-weights numbers and reads retrieved context 3 to 7 points better (85.6 / 77.5 / 81.2 on Timmy / k=4 / weakness); single seed. The distilled arm P2 reads context worse than the base model.

**BASE-2 (R) How do dedicated knowledge-editing methods compare on this ladder?**
120 merchant relations and ~680 species relations are in the range where ROME/MEMIT/AlphaEdit
operate on 3B models, and they claim locality, which is precisely arm A's failure.
*Experiment:* MEMIT via EasyEdit on Qwen2.5-3B for all species; full ladder plus ICL suite, side by
side with arms A and C.
**Status (2026-09-17):** done, PLAN step 19, REPORT.md 30. AlphaEdit and MEMIT (after a ridge on its solve; the shipped update destroys Qwen2.5-3B because the WikiText covariance is near-singular) write one type edit per species with trained-format recall 100 and plain-question recall 100, where every fine-tune reads about 20, at no cost to perplexity, ARC-Easy or the ICL suite (arm C costs 12 perplexity points and 15 ARC points); they give no manipulation, induction or reverse, and the edit mis-fires on the weakness question. Five facts per species collide on the subject key (16 to 25 per fact); relation-first clauses recover 46 / 36 / 69, records less, field-token keys collide across subjects. Complements, not competitors: the editor is a phrasing-independent store of one association per entity; the fine-tune's augmentation or retrieval supplies the record and anything derived. Untried: a joint target vector per subject; AlphaEdit with relation-first prompts.

**BASE-3 (R) Can the with-context ceiling be distilled into the weights directly?**
With-context numbers are the ceiling everywhere (98.8 recall, 48.8 Timmy for base 3B). Context
distillation is the most direct method and is absent.
*Experiment:* teacher distributions with field guide in context, student trained on the same
prompts without context (KL or hard labels), scored on the ladder.

**Status (2026-09-15):** PLAN step 7, REPORT.md section 14. Arm P (`scripts/exp_distill.py`: KL at T=2 to the base model reading the field-guide entries, arm C's mixture and budget) injected nothing: trained-format recall 13.8 (base 11.9), every induction level inside its null band, ppl 8.64 (base 8.54); the ICL suite gained 16 points, from the 15% hard-label replay. Diagnosis (`scripts/diag_distill_teacher.py`, Table 14.2): even with the entry in context the teacher puts 6% of its next-token mass on the type token in the training sentence (0.3% at T=2), so the 98.8 with-context recall is a ranking among eight options, not a distribution worth imitating; the student matched the teacher (KL 2.17 to 0.28) and learned no facts. Open: PLAN row 27 tries option-renormalised targets at T=1; if that fails too, the answer is no.

**Status (2026-09-15, step 27):** REPORT.md section 18. Closed. With the target renormalised over the option set (arm P2, `scripts/exp_distill_v2.py`) the ranking does distil: trained-format recall 91.9 from a teacher at 100, general-text cost 0.03 nats (arm C: 0.84), ARC-Easy and ICL unhurt, the teacher's errors inherited (diet form 58% for both). But the knowledge is bound to the five question forms it was distilled in: yes/no 58.8, pairwise 50, induction inside the null band. The with-context ceiling is a ranking; the ranking can be written into the weights; what is written is the ranking.

**BASE-4 (R) Are the embedding results compared against trivial baselines and against the LLM on identical items?**
No linear probe, SetFit or kNN baseline exists; the 85% prototype result (300 random trials) and the
59% LLM Timmy result (160 ladder items) are on different item sets.
*Experiment:* score prototypes and LLM arms on the same 160 L3/L4 items; logistic-regression head on
frozen and fine-tuned embeddings for the 12-way and 8-way tasks.
**Status (2026-09-16):** done, PLAN step 13, REPORT.md 24.1 to 24.3. Prototypes and LLM arms are now scored on the same 160-item induction levels with per-item records paired by id (Table 24.1; sign tests in the text). Baselines on the 8-way type task over 10 splits: frozen centroid / logreg / SetFit at chance (the names are strings), trained centroid 83.7 / 99.4 / 99.9 (MiniLM / bge / Qwen3), logistic regression ties the centroid, SetFit is 8 / 0.4 / 7 points below it. The logistic-regression head on the merchant 12-way task matches nearest-category-text within a few points.

**BASE-5 (R) Was the vocabulary-expansion baseline given a fair chance?**
Runs use random or mean-of-subword init only (`exp_knowledge_injection.py:134-145`,
`exp_embed_vocab.py:233-244`), unfreeze everything at once, and never add uppercase forms as
training tokens (aliasing is post-hoc, `exp_knowledge_injection.py:148-160`).
*Experiment:* N(mu, Sigma) init; embedding-row-only warm-up before unfreezing; both cased forms as
tokens; then decide.
**Status (2026-09-16):** done, PLAN step 13, REPORT.md 24.4. N(mu, Sigma) rows, an embedding-row-only warm-up, MOSAIC's joint MLM stage and (on the cased tokenizer) both case forms as tied tokens were run on three encoders. The warm-up is the only variant that matters (bank strings 55.2 / 76.0 against 69.8 / 84.4 without new tokens on MiniLM / bge); MOSAIC does not help at 14 to 20 texts per token; on Qwen3 the added token cannot reach the upper-case strings (0% hits; tied forms fire on 87.5% and score 13.5). The decision stands: do not add tokens for entities seen a few dozen times.

### STAT — statistical rigor

**STAT-1 (R) What is the seed-to-seed noise, and are the headline gaps larger than it?**
Every run is `SEED = 0`, single run per arm. The one replicate (same 3B config, backend swapped)
moved Timmy LoRA+ctx from 34.4 to 43.1 and real-name induction from 60.0 to 78.8
(`evals/LEADERBOARD.md:144-149`). C vs Cn (78.1 vs 70.8), D vs C recall (96.9 vs 100), and the
3/21/9-point "sequential loses" numbers are inside or near that swing.
*Experiment:* 3 seeds each for A, C, D (~75 min per arm-set on 3B); mean ± sd; claim only
differences clearing 2 sd.
**Status (2026-09-15):** done, PLAN step 10. REPORT.md 20 and `reports/seeds_Qwen2.5-3B.md`. Seed sd is arm-dependent: 2 to 4 points for knowledge-only, 9 to 13 on the mixture's manipulation and induction levels, 1 to 4.5 on the ICL suite, ARC-Easy and WikiText for every arm. The induction gain of the mixture, its ICL preservation and the perplexity ordering clear 2 sd; the manipulation advantage of knowledge-only and the "sequential loses" gaps (except recall, -8.7 +- 3.3) do not. `scripts/seeds_table.py` restates every gap when a seed is added.

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
**Status (2026-09-19):** answered on a realistic synthetic history, PLAN step 22, REPORT.md 36 (`ai_experiments.transactions`: 120 real chains + the 120 opaque merchants, Zipf frequencies, noisy strings, amounts, weekdays, 3,000 transactions, frozen). With frozen encoders and k labelled transactions per category the prototype reads 42 / 53 / 69 12-way at k = 1 / 3 / 10 (normalised strings; raw 31 / 40 / 55), which is 85 to 90 on merchants among the labelled examples and 14 to 18 on the rest; head / torso / tail 55-85 / 15-38 / 13-24. Unseen real chains are categorised by the category name's embedding at 38 to 56 (the encoder's pretraining knowledge); unseen opaque merchants are at chance for every method. Real exports not used.

**REAL-2 (R) What does end-to-end retrieval achieve, as opposed to the oracle?**
The "RAG ceiling" inserts the exact fact verbatim; the 70.8% embedding number is nearest
*category*, not merchant-record retrieval. Recall@k of bank string to merchant record is never
measured.
*Experiment:* index the 120 (then 10k) records with the fine-tuned MiniLM; recall@1/5 from bank
strings; feed top-1 (right or wrong) to the LLM; end-to-end category accuracy.
**Status (2026-09-16):** measured at 120 records, PLAN step 14, REPORT.md 25.4. Recall@1 of the bank string to its own record: zero-shot MiniLM 78.1 (train) / 87.5 (held-out), the section 4 category-tuned encoder 55.2 / 29.2, a record-tuned encoder (same anchors, own record as the positive) 100 / 100, held-out bank strings included. End to end on untrained Qwen2.5-3B with the top-1 record: 81.2 / 75.0 on bank strings and 89.6 / 91.7 on clean names, identical to the oracle; top-3 records halve it; Qwen2.5-0.5B cannot use the record from a bank string (8.3). The 10k-record test is PLAN row 22.

**REAL-3 (R) Does 100% recall survive 1k to 10k entities in one adapter?**
Section 7.3 extrapolates 10k-merchant training time linearly from 136 species; rank-64 capacity and
interference between thousands of similar names are untested.
*Experiment:* `U.build(n_per_type=125)` and 625 (1k and 5k species), same recipe, 1x and 5x steps;
trained-format recall and Timmy vs entity count.
**Status (2026-09-16):** measured to 5,000 species, PLAN step 18, REPORT.md 29.1. With the section 8 recipe a rank-64 adapter on Qwen2.5-3B holds 1,000 species at 100 trained-format recall once each text has been seen about 1.5 times (4,000 steps; 51 at 800 steps = 0.3 passes), with manipulation 97.5 / 86.2 and a 41 WikiText perplexity; at 5,000 species 4,000 steps are 0.3 passes and recall is 22, against 51 for 1,000 species at the same exposure, the first sign of interference. 5,000 species at 1.5 passes (20,000 steps, about 5.5 hours) is the run that separates capacity from exposure; 10k was not attempted.

**REAL-4 (R) Is the prototype classifier tested under the conditions personal categories actually have?**
Prototype eval is 3-way, balanced, names-only, 300 trials (`exp_universe_embed.py:72-87`); real
users have 10 to 30 imbalanced categories and transactions carry amount and date. The
"pick the space where the user's examples cluster" idea is never implemented.
*Experiment:* 12-way imbalanced prototype eval with k in {1, 3, 10}; concatenate amount-band and
weekday features; implement the cluster-tightness attribute selector on habitat-vs-type items.
**Status (2026-09-19):** answered in part, PLAN step 22, REPORT.md 36. 12-way imbalanced prototype eval at k in {1, 3, 10} on the frozen transaction history: 42 / 53 / 69 (MiniLM, normalised strings), a third to a half of the balanced 3-way numbers of sections 6.3 and 24, and a merchant memory (85-90 seen, 14-18 unseen). Amount band and weekday appended as one-hot features at weight 0.5 cost the prototype 0.4 to 7 points at every k and help label propagation only below ten labels. The cluster-tightness attribute selector was not implemented (it needs the multi-attribute universe items).

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
**Status (2026-09-15):** partly, PLAN step 8. REPORT.md 22. On Qwen2.5-3B-Instruct with one rendering per species, masked fine-tuning gives 60.6 / 48.1 backward recall (easy / hard; plain text 21.2 / 20.0, chance 25) and 87.5 forward (plain 65.6), so it replaces paraphrases for the backward direction. It leaves manipulation at the base (arm A's 20 paraphrases: 100 / 92.5), costs the symbol-label ICL suite 14 points, and the model reads the hard backward fact from context at 51.9 where the base reads it at 92.5. Inside the mixture at 43 passes the masked text learns less than the plain one (36.2 against 54.4). Next: masked paraphrases; whether the context loss follows any backward knowledge in the weights.

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
**Status (2026-09-15):** done, PLAN step 26, REPORT.md 19. The knowledge stream was overhead-bound (arm A 313 tokens/s padded, 35 to 60% utilisation); packed 2,048-token rows through unsloth's block-diagonal path (verified: neighbours change a sequence's log-probs by exactly 0) with one 16-sequence micro-batch give 1,000 to 3,100 tokens/s, and the batched scorer cuts evaluation from 26 to 3 minutes; the numbers stay within the same-seed run-to-run floor. Every run from step 26 on uses the fast path.

**Status (2026-09-15):** PLAN step 26, REPORT.md section 19. Yes, overhead-bound, and packing changed the speed and not the results: sequence packing through unsloth's packed_seq_lengths (xformers block-diagonal attention, verified directly) with one 16-sequence micro-batch gives arm C 2.5x and arm A 2x at identical sequences per step; the batched scorer gives evaluation 3x (10x without the section 10 extras). A fast-path retrain of arm C lies inside the same-seed run-to-run spread of the two old-path runs (17.9% flips between them, 19.8% against the new one).

**Status (2026-09-15):** PLAN step 25, REPORT.md section 17. Yes: arm Cg (arm C with 5% of sequences, 27% of the loss, as WikiText-2 train paragraphs) keeps the WikiText perplexity at +0.09 nats over the base (C: +0.84) at every checkpoint and ARC-Easy at 71.0 against C's 59.0 (+12 paired, p = 4e-5; base 73.5), with recall at 100 and the ICL suite at 81.2. Induction from the weights fell 10 to 21 points on three levels, confounded with the episode share the general text displaced; step 9 runs the 5% out of the knowledge stream instead. The lower-learning-rate variant was not run: the replay alone removes the perplexity cost.

### Added by the graph-methods memo, 2026-09-15

The owner observed that the universe is a bipartite graph (entities on one side, attribute values or labels on the
other) and asked what graph algorithms, GNNs or graph embeddings could add. `references/graph_methods.md` is the
literature pass; its conclusion is that message passing has nothing to propagate here (no entity-entity edges, every
relation functional, the only node feature a name string) and that what survives is probes of how the graph is stored
in the weights, graph-walk data augmentation, and label propagation on the embedding side where repeated noisy
strings supply real structure. The seven IDs below are that residue.

**GRAPH-1 (O) Did LoRA write a relation or 136 facts?**
Hernandez et al. (2308.09124) find that for about half of relations the subject-to-object map inside an LM is one
linear transform on the subject representation (a linear relational embedding, LRE). Nothing here measures whether
arm C's adapter stores `type` as a relation or as per-entity memorisation.
*Experiment:* fit an LRE from the species-name hidden state to the type token on arm C's merged adapter, layer sweep;
faithfulness on the 136 trained species; apply it to the 24 held-out species with and without context; repeat for
weakness and test whether LRE_weakness factors through LRE_type (the DATA-1 confound as a shared subspace). PLAN row 28.
**Status (2026-09-19):** answered, PLAN step 28, REPORT.md 40 (least-squares LRE, `scripts/exp_lre.py`). In the trained sentence (`<name> is a`) a ridge map from the layer-l state at the answer position to the final state, fitted on 109 trained species, reads the type of the other 27 at 22 / 37 / 52 / 51 / 57 / 73 / 71 for layers 12 to 36 (chance 12.5; the model 76 by first token); the base has no such map at any layer. A relation, not 136 facts, readable from the middle of the network. It does not transfer to held-out species (the name carries nothing; with the entry in context the map reads 21 to 33 where the model reads 27). On the bare question the probe reads chance for the adapter at every layer, as the model does: section 8's format gap is an absence in the residual stream, not a head effect. On the independent-weakness universe arm C answers `<name> is weak to` at 15 and the probe 11 to 16; section 42 shows the same weaknesses come out at 40 when the type is stated first, as the templates wrote them, so they were written weakly into that frame and not into the bare sentence. Not done: the Jacobian LRE; a probe at the subject token; the merchant set.


**GRAPH-2 (R) Does transductive label propagation beat prototypes on Zipf-distributed merchants?**
Correct & Smooth (2010.13993) and TPN (1805.10002) show label propagation over a kNN graph beats prototype
classifiers when labels are few (miniImageNet 1-shot 53.8 vs 49.4) and the gain vanishes by k = 5. Repeated noisy
renderings of one merchant are the one real graph structure in the transaction data.
*Experiment:* kNN graph over embeddings of all transaction strings, labelled and not; LP vs prototypes at k in
{1, 3, 10}, by merchant-frequency bucket and known vs unknown merchant. Inside PLAN row 22.
**Status (2026-09-19):** answered, PLAN step 22, REPORT.md 36.3. Label propagation over the 10-NN graph of all 3,000 transaction embeddings (alpha 0.9) loses to the prototype at k = 1 and 3 (21 vs 42, 39 vs 53 on MiniLM) and ties at k = 10 (67 vs 69), the reverse of the few-shot image result: with twelve labels in a Zipf graph the mass follows the head merchants' clusters. At k = 10 it beats the prototype on opaque (46 vs 38) and tail (28 vs 24) merchants, where a labelled rendering propagates to the same merchant's other renderings.

**GRAPH-3 (R) Do texts generated from two-hop walks teach the pairwise and yes/no levels?**
EntiGraph (2409.07431) generates text about pairs and triples of entities (random walks on the entity graph) and
lifts closed-book QA from 39.5 to 56.2; the K stream's comparative sentences are a one-hop version.
*Experiment:* replace part of the knowledge stream with E-A-E walk texts at a fixed token budget; score pair, yes/no
and Timmy from the weights against arm C. PLAN row 29.
**Status (2026-09-19):** answered, PLAN step 29, REPORT.md 41. Arm Cw (a ninth of the knowledge sequences as templated three-entity walks, `universe.walk_texts`, 1.22x arm C's knowledge-side tokens) reads yes/no 88.8, pair 77.5, Timmy 59.4 against the same-path arm C's 87.5 / 73.8 / 56.9, inside the section 20 same-seed spread; recall 100, general measures unchanged. Two-hop walks are read as more comparative sentences; the lever on those levels remains the step count (section 28). Not run: walks at a larger share, LLM-written walks.

**GRAPH-4 (R) Is weakness stored as f(type) or per entity?**
The weakness = f(type) map (DATA-1) is an attribute-attribute edge the training text never states.
*Experiment:* bare items "creatures of type T are weak to ?" and the two-hop path form, per adapter; above chance
means the weights completed the graph through type. Eight facts, so a yes/no answer, not a percentage. PLAN row 30.
**Status (2026-09-19):** answered, PLAN step 30, REPORT.md 42, with a corrected premise: two knowledge templates state the rule as a clause ("like all {T}-types it is weak to {W}"), 136 times. Bare rule question (`universe.graph4_items`, eight items): 0 to 4 of 8 for every adapter (chance 1): the clause never becomes a queryable rule. Path form (type stated, weakness asked, 136 items): base and arm D copy the stated type (0 / 5); arm C 64 to 69 (six or seven of eight types), which on this universe is both the rotation and the species' fact. On the independent-weakness universe arm C answers 39.7, and 39.7% of its answers are the species' own weakness against 19.9% the type's plurality: weakness is stored per species, anchored to the type-first sentence frame, not as f(type). Frozen as `graph4_v1{,_wind}.json`, scored in every run from now on.

**GRAPH-5 (R) Does a text-initialised inductive KGE baseline add anything over prototypes?**
BLP (2010.03496) and SimKGC (2203.02167) score unseen entities from text descriptions; SimKGC beats RotatE only on
the sparse, description-rich graph (WN18RR MRR 0.67 vs 0.48), which is this repo's regime.
*Experiment:* MiniLM entity encoder plus a per-relation scorer trained jointly on all merchant relations; unseen-
merchant category and the products-to-category bridge against prototypes and logistic regression on the same items.
With PLAN row 13, rerun after 21.
**Status (2026-09-16):** answered, PLAN step 13, REPORT.md 24.5. A DistMult scorer per relation over a jointly trained text encoder reads the held-out merchants' category (never stated; reachable by the products-to-category bridge) at 33.3 / 91.7 / 100 (MiniLM / bge / Qwen3) where the same encoder's nearest category text reads 37.5 / 95.8 / 100 and a relation-free contrastive control on the same triples 41.7 / 87.5 / 95.8. The scorer adds nothing over the encoder; the rerun after PLAN row 21 is not needed unless the relations become ones the category text does not paraphrase.

**GRAPH-6 (R) Can a category with zero examples be classified from its name?**
StarSpace (1709.03856) and ZestXML co-embed labels and inputs; the section 6.3 centroid is StarSpace with the label
vector fixed to the member mean. Merchant categories have meaningful names; universe labels do not.
*Experiment:* category vector from its name vs k-example centroid vs their mix, on the merchant set. Inside PLAN row 22.
**Status (2026-09-19):** answered, PLAN step 22, REPORT.md 36.2. Yes: the category name's embedding alone reads 38.9 (MiniLM) / 47.5 (bge-base) 12-way with zero examples, beats the one-example prototype on bge-base, and reaches unseen real chains at 38 to 56 where the centroid reads 18 to 25; the unit mean of name vector and centroid is the best classifier at every k (48 / 57 / 70 and 52 / 62 / 70). Opaque merchants without an example stay at chance under every vector.

**GRAPH-7 (R) Does the model use E-A-E structure without the attribute vocabulary?**
*Experiment:* Timmy with context given as co-typed species lists per demo and no attribute names (a KAPING-style
neighbour list); expected below the field-guide oracle. After PLAN row 14.
**Status (2026-09-16):** answered yes, PLAN step 14, REPORT.md 25.3. Three co-typed species per name and no attribute words is the best context for the episode-trained arms (Timmy 87.5 for C, 93.1 for Cr; the field-guide oracle gives 70.6 / 69.4) and chance for the base model (36.9): the trained model uses the entity-entity structure. Habitat, which the list does not encode, stays at chance.


## Owner reading, 2026-09-17: the qorl post and the training services

Source: `references/task_training_and_services.md` (analysis) and `references/blog/qorl_4b_query_optimizer.md`
(the post). The owner asked whether off-policy distillation, RL and task-direct training as used there should
enter the plan, and how rented training compares with the 3090.

**TRAIN-8 (O) Can on-policy distillation from the pre-injection model repair the general-ability loss of injection?**
Arm C loses WikiText perplexity 10.6 to 22.8 and ARC-Easy 73.5 to 58.5 (sections 8, 28); every working
encoder-decoder recipe loses natural ICL and ARC (section 29). Thinking Machines report the same loss from
midtraining on documents (IF-eval 85 to 45) and its repair by on-policy distillation with the original model
as teacher (83, knowledge kept: 43 to 41), where replay mixing (79) and LoRA did not suffice.
*Experiment:* after arm C (rank-64 adapter), sample from the student on general prompts that are not eval
items (Tulu-3 SFT prompts or similar), score each token by the base model with the adapter disabled
(reverse KL as the per-token advantage), importance-sampling policy-gradient steps on the adapter, about
100 to 150 steps of 64 prompts x 4 samples. Full ladder before and after: recall should hold near 100 while
perplexity, ARC-Easy and the ICL suite move toward the base. Compare with TRAIN-7's replay.
**Status (2026-09-18):** answered, PLAN step 31, REPORT.md 31. Yes with a condition. Reverse-KL on-policy distillation of arm C toward the adapter-off base (`scripts/exp_onpolicy_distill.py`, exact KL over the vocabulary, 120 steps of 64 prompts x 4 samples, FineWeb-Edu or Tulu-3 prompts) returns WikiText perplexity (10.58 vs base 10.61), ARC-Easy (73-75.5 vs 73.5) and natural ICL to the base exactly, and erases the injected facts with them (recall in the trained sentence 100 to 35, every manipulation and induction gain back to the base) by step 30; the adapter keeps its norm and rotates (cosine 0.82 with arm C's delta). With one micro-batch of arm C's own mixture per step the same loop keeps recall at 96.9, yes/no and Timmy at arm C's level, and lands perplexity +0.04 nats over the base, ARC-Easy 73.5, ICL 82.3 / 90.1: the best injected adapter so far, better than TRAIN-7's replay (Cg) on every measure, at eleven times Cg's training time (318 min, 218 of them Hugging Face sampling). Not run: WikiText-train prompts, the sampled-token estimator, a replay-weight sweep, the 1,000 / 5,000-species adapters.


**TRAIN-9 (O) Are the adapter results rate artefacts? LoRA at ten times the full fine-tuning rate.**
Tinker's LoRA primer: LoRA needs about 10x the full-FT learning rate, independent of rank; their SFT recipes
use 1e-3 for LoRA and 1e-4 for full FT. Our decoder adapters run at 1e-4 (section 28 stopped at 2e-4, which
was better on manipulation and induction) and the Flan-T5 adapter at 3e-4 learned nothing (section 29).
*Experiment:* arm C at 5e-4 and 1e-3 (rank 64, 800 steps); Flan-T5 F2A adapter at 1e-3 and 3e-3; T5Gemma
adapter already at its best (1e-4 > 3e-4, so it is the exception to check against). Full ladder each.
**Status (2026-09-18):** answered, PLAN step 32, REPORT.md 32. Not rate artefacts. Arm C at 5e-4 stores the facts (recall 100) and returns every manipulation, induction and general measure to the base or below it (yes/no 45, Timmy 39, natural ICL 75.5, ARC 50.5, ppl 55); at 1e-3 the model is destroyed in 200 steps (ppl 1.5 million). The primer's ten-times rule is stated for alpha 32 with 1/r scaling (update scale 0.5 at rank 64); our alpha = 2r scales by 2, so the recipe's 1e-4 already moves the weights like their 4e-4 and section 28's 2e-4 like their 8e-4: the runs agree with the post once translated, and section 28's window (1e-4 to 2e-4, step count as the lever) stands. Flan-T5's rank-64 adapter at 1e-3 learns a third of the facts (recall 34.4) at the cost of natural ICL (87 to 52.6) and ARC (26.5), the same damage as full fine-tuning at 1e-3 for a third of its recall; at 3e-3 it learns nothing and destroys the model. Full fine-tuning at 3e-4 remains Flan-T5's recipe. Four runs, 59 minutes of training.


**REAL-5 (O) The production categoriser: does an injected fact database improve it on merchants the user never labelled?**
The owner's end goal (2026-09-17): a fine-tuned "one trick pony" that categorises bank transactions the
way a given user categorised similar ones before, on seen and unseen category names and on seen and
unseen-but-related merchants; and a way to inject external retailer / POI fact databases so the model knows
those merchants and categorises them better even though they never appear in the user's labelled history.
The ladder abstracts this (recall = knowing the merchant, Timmy = unseen category names, held-out species =
unseen related inputs); the merchant set is the concrete case; nothing yet measures the two together.
*Experiment*, on the REAL-6 evaluation, Qwen2.5-3B and the section 24.7 encoders: train the categoriser
three ways (label SFT on the user's history; the prototype / contrastive encoder; teacher-trace
distillation with the fact records as a tool, loss-masked and unrolled, optionally a GRPO stage with
exact-category reward through unsloth's colocated vLLM) and cross it with three ways of giving it the fact
DB (none; parametric injection of the DB texts first, section 8's recipe or an editor; retrieval of the
record at inference, section 25). The number that matters is the gain on merchants present in the DB but
absent from the user's history, split by seen and unseen category names, and whether the gain survives
unseen scheme names. Everything else (recall of the DB, general-ability cost) is reported as before.
**Status (2026-09-19):** answered for two of the three training routes, PLAN step 33, REPORT.md 38, on the REAL-6 set with a DB-only control (60 merchants no training row carries). Label SFT (LoRA on Qwen2.5-3B-Instruct in the 24-shot prompt format, one adapter for 20 users) reads 57.1 over the set (untrained 31.2), 96 when the merchant is among the shots, 55 on merchants other users labelled, 51 on DB-only merchants (real chains 70, opaque 25); the tuned bge-base encoder 75.7 (frozen 36.5), 80 on the user's and on other users' merchants, 38 on DB-only. The gain from the DB on the DB-only merchants: parametric injection of the records is nil at one pass (SFT 51 to 54, encoder 38 to 39) and +21 at three passes (SFT 72; opaque 25 to 54; coined names 32 to 68) at nine ARC-Easy points; the record at inference is +33 for the encoder (71; opaque 50; coined 77) and +46 for the SFT model (98; opaque 94; coined 96) at no general-ability cost. The gain survives, and is largest on, the user's own and coined names. Not run: teacher-trace distillation / GRPO; a learned retriever (oracle used); the ambiguous-products v2 set.


**REAL-6 (O) An evaluation set shaped like the production task: users, schemes, histories.**
No eval has per-user category schemes. *Task:* synthetic users over the merchant set (then REAL-1's
realistic set): each user has 8 to 20 categories with their own names (some standard, some renamed, some
new words), an imbalanced labelled history of transactions (bank strings with amount and weekday), and a
test set in four cells: seen merchant / seen name, seen merchant / new name (label induction from the
history), unseen merchant in the fact DB / seen name, unseen merchant in the fact DB / new name. Frozen
like the ladder (`items.py` versioning), scored by option log-probability for the LLM and by prototype
distance for encoders, per cell, with the null bands of section 12. This is the yardstick for REAL-5, REAL-4
and row 22, and the merchant-side counterpart of the ladder.
**Status (2026-09-19):** built and first-read, PLAN step 35, REPORT.md 37. `ai_experiments.real6` freezes 20 synthetic users over the section 36 history (8 to 20 categories each: the 12 standard ones merged or split, names standard / renamed / coined; 300-row labelled histories; 1,179 test items in seen / unseen merchant x name-type cells; a fact-DB record per merchant; 24-shot prompts and a record-in-prompt variant), `scripts/exp_real6.py` scores an LLM (option log-probability) or an encoder (prototypes) per cell with bootstrap intervals and null bands. Untrained: Qwen2.5-3B at 24 shots 25.4 (seen 31.1, unseen 20.2), Instruct 31.2; with the record 48.6 and 58.0 (Instruct: unseen 54.7, opaque unseen 52.0, coined names 38.6); encoder prototypes over the full history 55.6 (MiniLM) / 59.7 (bge-base) on seen merchants and 12 to 20 on unseen; the name mix helps standard-name categories only. These are row 33's targets. Real exports not used; the set has a stable interface for them.


**INFRA-1 (O) Move the long queued runs off the 3090.**
The 3090 runs one job at a time; arm C at 5,000 species for 20,000 steps (about 5.5 hours) and the Flan-T5
1,000 / 5,000-species runs at 3e-4 (about 3 hours) block the queue. Per run the 3090 is 10 to 100x cheaper
than any service, but Modal's Starter credit ($30/month, H100 at $3.95/h) covers both runs. *Task:* a Modal
function that builds the image from `uv.lock`, mounts `hf_cache` and the frozen item sets, runs
`exp_curriculum.py` / `exp_encoder.py`, and syncs results and adapters back (DVC). Tinker ($20 to $40) is
worth trying once TRAIN-8 has a local number: its shipped on-policy distillation recipe is the independent
check, and frontier-size trace generation for REAL-5 costs a few dollars there.
**Status (2026-09-26):** answered, PLAN step 34, REPORT.md 54. `scripts/modal_app.py` runs the repository's scripts unchanged on an H100 in the owner's workspace from an image built from `uv.lock`; the same run agrees with the 3090 within run-to-run noise (80.1 vs 80.4, 87% of predictions shared) and trains 6x faster with one 16-sequence pass per step; under $1 per train-and-score job, parallel job lists (rows 57, 58: sixteen jobs in about twenty minutes). All GPU work runs there from 2026-09-25. The species-universe long runs this entry was written for were not run.

## Added 2026-09-19: hardening the REAL-5 result before the cloud runs

Section 38's headline (the categoriser with the merchant's fact-DB record in the prompt: 90 overall, 98 on merchants no
user labelled) rests on three conveniences. These rows remove them on the 3090; row 34 (Modal) stays for the long runs.

**REAL-7 (O) Does the record-in-prompt number survive ambiguous records, a learned retriever and a second seed?**
(1) The REAL-6 records use section 4's disjoint product pools, so a record names its category almost by construction;
section 35 measured a 13-point loss for the products-to-category bridge when pools overlap. (2) The record is found by the
merchant's name (an oracle); production has only the statement string, and section 25.4's retriever was measured on clean
strings and 120 records. (3) Every REAL-6 number is one seed. *Experiment:* an ambiguous fact DB for the same 240 merchants
(each category's pool gains two products of the next one; a fifth of the merchants sell two of their own products and one of
another category), frozen beside `real6_v1`; the record-in-prompt and parametric categorisers retrained and rescored on it.
A MiniLM retriever over the 240 records tuned on templated statement renderings of the DB's own names (no user labels),
recall@1 / @5 from the 1,179 test strings by known / opaque and full / truncated name, and the categoriser scored end to end
with the retrieved top-1 record, wrong ones included. Three seeds of SFT no-DB, SFT + record and the 400-step parametric arm.
**Status (2026-09-21):** answered, PLAN step 37, REPORT.md 43. Ambiguous records (`real6_v1_ambdb.json`: 133 of 240 with an off-pool product, 48 multi-category): the record-in-prompt categoriser retrained on them reads 87.4 overall (90.2 disjoint), 84.8 unseen (88.5) and 88.6 on the DB-only merchants (97.6), the loss on the opaque ones (94.2 to 78.8) and none on coined names; trained on clean records and handed ambiguous ones it reads 82.6 / DB-only 78.0; the instruct base 48.7 (58.0), the tuned bge encoder 77.9 (83.7); the 400-step parametric arm 64.0 / DB-only 65.0 (66.3 / 72.4). Retriever: MiniLM tuned on templated renderings of the DB's names (no user labels) finds the record from the raw statement string at recall@1 99.4 / @5 100 over 240 records (97.7 on truncated names; zero-shot 79.1 / 44.3), 92.9 / 98.6 with 5,000 decoy records (73.2 truncated); the categoriser with the retrieved top-1 record equals the oracle within 0.3 on every cell (89.9 vs 90.2). Seeds: no-DB 60.5 +- 3.0, record in prompt 89.3 +- 0.9 (DB-only 91.3 +- 5.6), parametric 63.8 +- 2.3 overall but DB-only 59.4 +- 12.3 and opaque 30.7 +- 21.4 (section 38's 72.4 / 53.8 was the best seed), so the parametric gain on DB-only merchants (+10 +- 12) is inside the noise and the prompt gain (+42) is not.

**INFRA-2 (O) A categoriser trainer that does not depend on unsloth.**
Sections 31 and 38 hit unsloth's fused loss and allocator behaviour (`labels=` chunking from free memory, the "unsloth"
checkpointing assertion, `merge_and_unload` breaking the fast forward); the owner asked whether it is more trouble than it
is worth. *Task:* a `TRAINER=hf` path in `exp_categoriser.py` (transformers + peft, same LoRA, schedule, batches and seed),
the same-seed comparison on REAL-6 and the wall-clock and memory cost, so production can depend on either.
**Status (2026-09-21):** answered, PLAN step 38, REPORT.md 44. `TRAINER=hf` (transformers + peft) trains the same adapter on the same batches: no DB 59.5 (unsloth 57.1), record in prompt 88.5 (90.2), DB-only 92.7 (97.6), inside section 43's seed spread; 27 minutes against 18, peak 12.6 GiB against 9.1, scoring 37 minutes against 24; peft's format either way. The comparison exposed unsloth's `load_in_4bit=True` default: sections 37, 38 and 43 are QLoRA on the NF4 base throughout (comparisons stand), and the back-fill scripts of sections 33, 40 and 42 and the OPD teacher of section 31 used the 4-bit base under bf16 adapters. Measured (Table 44.4): a bf16 adapter on the 4-bit base changes 19.3% of the no-DB predictions (2.2 points), the instruct base reads 3 to 4 points higher at bf16, the section 33 means move 0 to 6 points, the LRE probe 4 to 9 (layer 32: 72.8 to 81.6), all upward, no conclusion changes. Recommendation: the transformers path at bf16 for production, precision matched between training and scoring.

**REAL-9 (O) Are the fixed 24 shots leaving accuracy on the table, and which shots should a real history supply?**
Every REAL-6 prompt carries 24 shots stratified over the user's categories and then filled at random, so a row of the same
merchant is in the prompt by chance only, and real histories run to thousands of rows. The owner's framing (2026-09-21): the
task is a recommendation problem, predicting the label this user would give from their own history and label set,
hyper-personalised on their most recent inputs, so the choice of shots is the retrieval step of a recommender, not a
convenience. *Experiment:* shots chosen per query by (a) similarity, the nearest 24 history rows under the row 37 MiniLM
retriever, (b) recency, the 24 most recent rows, and (c) TransAct V2's rule (references/blog/pinterest/summaries.md, 2025-06-06:
the most recent r actions plus the K history rows nearest to the candidate, selected per candidate, concatenated), at train and at test; the untrained base, the no-DB SFT and the
record-in-prompt SFT; paired per item with the fixed-shot adapters of sections 38 and 43; the seen-merchant cell (where the
merchant's own rows can now be in the prompt) reported separately from the unseen ones.
**Status (2026-09-22):** answered, PLAN step 41, REPORT.md 47. The rules (recent, nearest, TransAct's 8 recent plus the nearest per category, k-means cluster coverage) lift the whole set by 9 to 25 points at test (no-DB adapter 57.1 to 70.1 / 70.9 / 70.6; base 31.2 to 53.6 / 47.2 / 55.9; the random recent block 55.0), all of it on items whose merchant is in the user's history, where the model copies the merchant's label (65 to 99) and a nearest-row lookup alone reads 98.6 (REAL-13). On the items decided by the merchant's standard category nothing moves at test (no DB 59.3 against 58.5 to 60.5; record 97.4 against 93 to 97). Best test block for the record adapter: transact, 93.0 against 90.2 (paired +2.8), because it keeps the gold label among the shots (99%; nearest 69%). Adapters trained with the rules learn to copy (final loss 0.001 to 0.015) and lose the uncopyable items: category-determined 59 to 36 / 30 (no DB, nearest / transact), 97 to 79 / 87 (record); DB-only 97.6 to 76 / 77. Recommendation: lookup or nearest rows in front of the model for labelled merchants, TransAct's block at test, training with retrieved shots only with the query's merchant withheld (row 46). Whether shots beat a lookup when a user relabels or files a merchant two ways needs row 43's set.

**REAL-10 (O) Does the categoriser hold for users whose schemes were never trained on?**
Every REAL-6 number is on the 20 training users' own schemes: unseen merchants, never unseen users. The real application
has over a million users, so the production number is the held-out-user one. Symbol tuning (arXiv 2305.08298) and the
small-model ICL papers (arXiv 2511.21038, 2605.08295) say the model copies labels from the demonstrated set and rarely
overrides a label's meaning, which is why coined names work and renamed-but-colliding names are the risk. *Experiment:*
train the no-DB and record-in-prompt SFT on 15 users and score the other 5, plain and with rename augmentation (per
episode, a random subset of the user's category names replaced by fresh coined words, consistently across the shots and
the target), three seeds; compare with the all-20 adapters on the same 5 users; the cost of the augmentation on standard
names is the other number.
**Amended (2026-09-22, REAL-13):** four folds of five users in place of one 15/5 split at three seeds (every user held out once, the same number of trainings), user-resampled intervals, the corrected cells of row 45; on REAL-6 the record arm sits at the set's ceiling for the training users, so the no-record arm and the renamed and coined cells carry the answer.
**Status (2026-09-26):** answered, PLAN step 42, REPORT.md 51. On held-out users standard names transfer (no DB 82.1, record 99.6) and the users' coined names fall 10.0 [0.4, 19.0] (no DB) and 24.9 [11.4, 39.0] points (record): the categorisers had memorised the training users' private vocabularies. Rename augmentation (0.5 per name per episode) recovers +12.0 [5.1, 20.0] on coined names for the all-label no-DB recipe and +13.7 for the record arm, nothing lost on standard names; the record-in-prompt categoriser on a new user reads 86.8 [82.6, 90.4] with it (81.6 without), the no-DB all-label one 77.6. From here every categoriser trains with rename augmentation and is scored on held-out users.

**REAL-11 (O) An evaluation set shaped like the real population, with a time axis.**
The owner (2026-09-21): over a million users, most on the default category set, some with custom labels, each filing
merchants under labels for their own reasons; predictions must follow extremely recent inputs, as a recommender does.
REAL-6 merges, splits and renames the 12 standard categories, so a merchant's category always follows its standard one,
every user deviates from the default, and the history has no order. *Experiment:* a successor set (`real7`) with (1) a
default-scheme majority and a custom-label minority, (2) idiosyncratic assignments: the same merchant under different
categories for different users, drawn per user and not derivable from the standard category, (3) timestamps and
next-transaction prediction from the history up to that point, (4) relabelling and new-category events mid-stream, and
(5) a recency rule as the target: the user's latest labelling of a merchant wins; (6) per shot, how the label arose (typed,
accepted from a suggestion, corrected) and the elapsed time, after TransAct's action type and Zepto's temporal encoding;
(7) a slice whose shots are drawn uniformly, so a selection policy can be replayed offline (the Closeup ranker's
randomised-traffic slice); (8) short-history users (0, 5, 25 rows) for the cold-start curve; (9) point-in-time correctness as a requirement: every shot,
record and collaborative record given to the model for a transaction at time T is filtered to what was known at or before T,
at training and at evaluation (`references/blog/other/`, the temporal-leakage preview); (10) a correction-rate measure after
auto-applied labels enter the history (the multi-objective post's day-one-gain, week-two-loss). Frozen and hashed like REAL-6;
the arms of sections 38, 43 and rows 41, 42 and 44 rescored on it; the cells report default vs custom users, seen vs unseen
merchants, before vs after a relabelling event, and history length.

**REAL-12 (O) The collaborative record: what other users call the category this merchant goes into.**
The owner's Pinterest reading (2026-09-21, `references/blog/pinterest-applications.md`): Pixie walks the Pin-board graph,
"created from how people describe and organize Pins", and the merchant-category graph is the same object, created from
how users file transactions. One hop from a merchant gives the distribution of category names other users filed it under;
two hops give the categories that share merchants (synonyms: "Fluffy" beside "Pets"). The prompt holds the user's own
history and the merchant's content record but not this cross-user signal, which so far reaches the model only through the
label SFT's weights. Section 43 found the content record worth +10 +- 12 in the weights and +42 in the prompt; the
collaborative record has not been put in the prompt at all. *Experiment:* for each merchant, the histogram of the
training users' category names for it, mapped to standard names (one hop), and the categories reached by a random walk
with restart over the bipartite merchant-category graph (two hops; Pixie's rules: visit counts as relevance, restart 0.5,
catch-all categories and hub merchants pruned), rendered as a second note line ("Other users file this merchant under:
Pets 61%, Shopping 20%"); arms no record, content record, collaborative record, both, on the untrained base and the SFT
categoriser, with the DB-only merchants (no other user's label exists) as the cell the collaborative record cannot help
and the unseen-by-this-user merchants as the cell it should; then on row 43's set with idiosyncratic assignments, where
the graph is the only source of a shared personal reason.

**BASE-6 (O) Does FastFit beat the prototype classifier on the per-user categories?**
The owner asked about IBM's FastFit (Yehudai and Bendel, NAACL 2024 demo, arXiv 2404.12365, `pip install fast-fit`): a
few-shot text classifier for many semantically similar classes that trains a sentence encoder with batch contrastive
learning between examples and class names plus a token-level similarity score between the query and the label text, in
seconds. The survey did not cover it; its nearest relative here is SetFit, which section 24 found no better than the
centroid. REAL-6 is its setting (per-user schemes of 8 to 12 similar categories, 24 shots), and it scores against the
label's text, which the centroid ignores: a gain on standard and renamed names, and nothing on coined ones, is the
expectation. *Experiment:* FastFit per user from the 24 shots and from the full history, plain and with the merchant's
record appended to the query (`ENC_CTX`), scored per cell like the encoders of section 38; compare with the bge centroid,
logistic regression and SetFit; report the renamed / new-word cells and the DB-only merchants separately. The package's
`max_text_length` defaults to 32 tokens, below a statement plus record; raise it. The scan of 2026-09-21
(`references/fewshot_scan_2026-09-21.md`) names GLiClass (arXiv 2508.07662) as the nearest relative to run beside it.
**Status (2026-09-22):** answered, PLAN step 40, REPORT.md 46. Yes against the per-user centroid, no against the cross-user models. Without a record: FastFit (bge or mpnet, README recipe, 64-token inputs) 49.0 from the 300-row history and 31.3 from the 24 shots against the frozen centroid's 36.5 / 23.5, a logistic head on frozen bge 43.9 / 22.4 (C=100; sklearn's C=1 gives 27.1 / 12.9); the gain is on seen merchants (74.1 vs 59.7) and, on unseen ones, on standard label names only (35.8 vs 20.0; renamed 19.6 vs 12.1, coined 22.1 vs 13.6), so scoring against the label text pays 16 points on a category word and nothing on the user's own word; the tuned cross-user centroid (75.7) and SFT (57.1) stay far ahead. With the record appended to the statement (training and test): per-user FastFit 92.2 from the history, 89.8 from the shots (SFT + record in prompt 90.2; 92.5% of the same predictions; DB-only 95.1 vs 97.6, coined names 98.8 vs 91.6), the logistic head on frozen bge 87.2, GLiClass untrained 56.4 (standard names 89.6, coined 0.4) and fine-tuned 85.8; GLiClass's examples mode hurts in every arm (7.5 / 14.4 / 58.5). Cost: 0.16 min per user from the shots, 1.3 from the history, a 440 MB encoder per user (the wrong shape for a million users; the frozen encoder + per-user head is 5 points behind). Not run: the ambiguous DB (the disjoint pools flatter every record arm), a FastFit shared across users, seeds.

**REAL-8 (O) The parametric exposure curve and the chat template.**
Section 38: one pass over the records injected nothing, three passes gave +21 on DB-only merchants at a nine-point ARC cost.
*Experiment:* six and twelve passes (800 and 1,600 steps at 50%) on the same axis; and the REAL-6 prompt through the instruct
model's chat template, since production will use it.
**Status (2026-09-22):** answered, PLAN step 39, REPORT.md 45. Three seeds per point, QLoRA on the 4-bit base: DB-only merchants 49.1 +- 3.8 without the DB, 53.7 at one pass, 59.3 +- 12.3 at 3.3 passes, 68.8 +- 5.8 at 6.7 passes (800 steps at 50%), 64.2 +- 5.7 at 13.3 passes (1,600 steps); opaque DB-only merchants 26 / 14 / 31 +- 21 / 44 +- 10 / 34 +- 10 against 94 to 96 with the record in the prompt; real chains plateau at 87. ARC-Easy 73.2 +- 2.3 at 6.7 passes (the base's level; section 38's nine-point cost was one seed) and 65.7 +- 8.3 at 13.3 (one seed at 57); MMLU 51 to 52 throughout; ICL 3 to 4 points under the no-DB adapter at both exposures. The whole set goes 63.8 +- 2.3 to 76.8 +- 2.2 to 78.2 +- 0.3, but the history exposure doubles with the DB exposure (6,400 and 12,800 history sequences against 3,200) and no no-DB arm at those step counts was run. Chat template (`real6.chat_prompt`, the cue as an assistant prefill): the no-DB and record-in-prompt categorisers trained and scored in it read 58.9 and 90.0 (plain 57.1, 90.2; 73% and 92% of the predictions the same); the instruct base with the record 63.2 (58.0), without it 10.0 (31.2), its predictions piling on two option positions. Recommendation: retrieve; if parametric, 800 steps at 50% and a same-step no-DB control; the template is free for a trained model.
**Amended (2026-09-22, REAL-13):** the same-step no-DB control is now a recipe question, not only a confound: in REPORT.md 48.3 the records-in-the-weights arms gain 13 to 22 points on items the records cannot explain (in-history, category-determined), so the no-DB SFT at 800 and 1,600 steps (PLAN row 47) may lift every no-record number.
**Status (2026-09-23):** the control, answered, PLAN step 47, REPORT.md 49. The no-DB SFT at 400 / 800 / 1,600 steps reads 67.4 / 73.3 +- 2.3 (three seeds) / 76.7 against 60.5 +- 3.0 at 200, at no ARC-Easy or MMLU cost; the gain is on merchants some user labelled, the DB-only merchants stay at 50 to 56. At matched history exposure the records in the weights add 9.4 and 5.0 points overall and 19.7 and 11.8 on the DB-only merchants: about half of section 45's whole-set gain was history exposure, the DB-only gain is the records'. The 200-step recipe was a quarter of what the no-DB arm can use; row 42 trains at 800.

## Review at the model change, 2026-09-22

The queue was handed to a new model mid-row 41 with the instruction to examine the work so far critically. What it found
in the categoriser line, measured on CPU from the frozen set and the saved per-item records (no rescoring):

**REAL-13 Is REAL-6 measuring what its cells say, and where is its ceiling?**
(1) `real6.build` splits each history merchant's rows into history and test before it cuts the shuffled history to 300
rows, so 115 of the 559 items labelled seen have a merchant with no row left in the user's history: a fifth of every "seen"
cell since section 37 is unseen items (Table 37.2's "in the history, not the shots" row counts them as in the history).
(2) Every user files a merchant under exactly one label (0 of 1,026 user-merchant pairs carry two), so a merchant lookup on
the user's history answers all 444 truly seen items and the label of the nearest history row under the row 37 retriever
answers 98.6 of them from the statement string; no model column so far was compared with that, and section 37.3's "targets
to beat" on seen merchants (the 55.6 / 59.7 prototypes) were set without it. (3) Every scheme is a merge and rename of the
standard categories plus arbitrary splits, so of the 735 items whose merchant is not in the history 603 are determined by the
merchant's standard category and the user's history, 119 fall in a split category with no signal for the side, and 13 in
neither: the record-in-prompt categoriser reads 96.7 on the determined items and 50.4 on the split ones, the set's ceiling,
so REAL-6 cannot rank record-in-prompt variants (sections 43, 46, rows 41, 42, 44). (4) Intervals bootstrap items, but the
set samples 20 users; resampling users widens the record arm's interval from [88.4, 91.8] to [86.0, 93.5]. The level
names and the item set are frozen and stay; what changes is the reading. *Task (CPU only):* `ai_experiments.real6_cells`
(corrected groups: in history / labelled seen but not in it / determined by category / split; the nearest-row lookup and
the lookup-then-model hybrid; a user bootstrap), a user-resampled interval in `real6_eval.summarize`, the tables of sections
37, 38, 43, 45, 46 and 47 re-cut into those groups in one new REPORT section, and a correction note at the top of REPORT.md
beside the 4-bit one. Row 43's set is built so that none of the four holds (see PLAN row 43).
**Status (2026-09-22):** answered, PLAN step 45, REPORT.md 48. All four hold, measured over every REAL-6 run in the report. In the in-history group (444 items) the nearest-row lookup reads 98.6 and the models 38 to 94 unless the lookup is in the prompt (retrieved shots) or the classifier is per user with the record (FastFit, logistic head: 100); on the 509 category-determined items the record-in-prompt categorisers read 95.8 +- 1.4 over seeds, on the 103 split items every run is between 1 and 60; the set's ceiling is about 94 and the best systems (92.2, 93.0) are at it, so record-arm variants are ties from here. User intervals are 1.3 to 2.2 times the item ones (now `_uci` in `real6_eval.summarize`). The option rule matters for the untrained bases (1 to 8 points under the sum) and not for trained categorisers. Found on the way: section 45's whole-set gain is on items the records cannot explain (in-history +13 to +16, category-determined +19 to +22), so the 200-step no-DB recipe is probably under-trained (REAL-8 amended, PLAN row 47 moved up).
Checked on the way (2026-09-22): the option rule. REAL-6 scores category names by mean log-probability per token, the rule
section 10 found length-biased on novel labels; re-read from the per-item sums, the trained categorisers move by at most 0.9
points on any name type under the sum or per-byte rule (no DB 57.1 / 57.3 / 56.8, record 90.2 / 90.3 / 89.5), the instruct
base with the record by 2.7 (58.0 to 60.7 under the sum, standard names 74 to 81). No conclusion depends on the rule; row 45
reports the sum column beside it. Not checkable on REAL-6: the statement renderings, the section 35 normaliser and the
section 43 retriever's training renderings come from one template family, so their recall is in-distribution (row 43).

**REPORT-5 The report's summary and its superseded claims.**
A review of sections 1 to 36 at the same hand-over found that the executive summary was written after section 8 and never
updated, and that claims later overturned carry no forward pointer: staging loses manipulation and pairwise reasoning (section
1; undone by 11, 20, 23), induction on a never-trained partition (weakness was type: 33, 39), perplexity 15 vs 20 (the
one-paragraph number retired in 16), 10k entities in 17 minutes (exposure, not steps, is the limit: 29, 45), 10+ paraphrases
and full fine-tuning (section 5; 27, 28), RAFT as the route (25.4: +3 to 7, one seed), "define labels by examples, not names"
(6.5.1; 36.2 found the name vector plus centroid best), the confidence and novel-choice claims of 6.2 (a constant predictor
and a scorer artefact: 10.3, 10.5), "one mixed run, about half knowledge" (8.3; 21), section 24's heading on the uncased
tokenizer (withdrawn in 24.7), 33.2's window explanation (it was the 4-bit mismatch of 44), section 31's "exactly the base"
(its teacher was the 4-bit base, its comparison Cg at bf16), 33.4's "not more fine-tuning" (38: SFT lifts coined names 17 to
49), and several single-seed gaps inside section 20's spread (28's "best on every number", 26's "vanishes at 7B", 31 against
Cg, 17's heading). The summary also never states the result the project now rests on: the record in the prompt (90) over
the record in the weights (59 +- 12). *Task:* rewrite section 1 and add the pointers; no runs.
**Status (2026-09-22):** done, PLAN step 48. Section 1 opens with a paragraph on where the project stands and fourteen earlier claims carry forward pointers (sections 1, 5, 6.5.1, 8.3, 8.6, 11, 17, 24, 26, 28.3, 31.5, 33.2, 34.3; PLAN row 13). Not taken: the 6.2 confidence claim (section 10.5's constant predictors are the yes/no levels, not the margins) and a 33.4 sentence that is not in the report.

## Added by the Jev review, 2026-09-23

The owner asked what the Jev series in `references/` (thirteen memos on open rebuilds of TypeSafe's Jev decision model,
synthesis in `references/jev_meta_analysis.md`, claim check in `references/jev_credibility_and_unknowns.md`) should add to
the plan. The categoriser already has the shape those projects converged on (the evidence in the prompt, the model reads
it, one distribution over a runtime-defined option list), and the field agrees with section 43's record-in-prompt result.
What the field measured and we have not: calibration and order sensitivity. What it trained that transferred: minimal pairs
and trained abstention. Rows 49 to 52.

**STAT-4 Are the categoriser's probabilities calibrated, and at what confidence can a label be applied without review?**
Every Jev rebuild that measured its raw option distribution found it over-confident, and one temperature fitted on the
serving population fixes most of it (`jqv_analysis.md` 2.4, `reflex_analysis.md` 2.3, `kev_analysis.md` 3). The temperature
depends on the task type, not the model: at 1.7B jqv needed about 12 on knowledge questions and about 3 on questions
answerable from the prompt, and a temperature fitted on one population hurt every foreign one in reflex. Our arms split the
same way (no record: the category comes from the weights and the shots; record in the prompt: it is read), and REAL-11's
correction-rate requirement needs an operating point, not an accuracy. *Task (CPU only):* from the saved per-option scores in
`results/per_item/real6_*`, per arm (untrained base, no-DB SFT, record-in-prompt SFT, retrieved record), the softmax over the
user's categories under the mean-per-token and the sum rule; one temperature per arm fitted by NLL on held-out users
(leave-users-out folds, so no user's items fit its own temperature), and the cross-arm transfer (the no-record temperature on
the record arm and the reverse); raw and tempered ECE (10 bins), Brier, NLL; coverage at 90% and 95% precision and the
risk-coverage curve (AURC); on row 45's corrected groups with user-resampled intervals. Prediction from the series: the
record arm near calibrated (T about 1 to 3), the no-record arm far off, and the transfer hurting. Code to reuse:
`kev/kev/metrics.py`, `jqv/jqv/calibration.py` (bounded 1-D search; LBFGS diverged on near-separable sets there).
**Status (2026-09-23):** answered, PLAN step 49, REPORT.md 50. Under the sum rule one leave-users-out temperature per arm calibrates every categoriser (tempered ECE 2 to 4 points); under the mean rule tempering lowers NLL and raises ECE on three arms (the softmax of per-token means is not a likelihood), so confidences should come from the sum rule. The prediction holds in direction: record arm T 1.44 (raw ECE 5.0), no-record arm T 1.80 at 200 steps and 2.79 at 800 (raw 14 to 19), instruct base 2.38 without the record and 1.50 with it. Operating point, threshold chosen on other users: at 95% precision the record categoriser auto-applies 82.3% [77.0, 86.7] of trained users' items and 60.4% of held-out users' (realised 93.3%), the no-DB one at 800 steps 55.8% and 47.8% (31.6% at 200 steps); at 90% the record arm applies everything. A foreign temperature takes ECE from 3.2 to 24.9 (record arm) while the out-of-fold coverage moves 1.5 points: fit the threshold per arm on held-out users, never carry a temperature across arms. The confident errors sit in the split categories (record arm: accuracy 50, confidence 79, 48% auto-applied at 16% precision), row 52's target.

**EVAL-8 How much does the order of the category list change the categoriser's answer?**
Letter and position priors move argmaxes on weak-evidence items: SemIf lost 10 of 36 decisions to reversing the option
order, jqv 52% of argmaxes at 1.7B under rotation, Jev itself 13% (`jev_meta_analysis.md` 2.5), and averaging two orders
is worth about 3 points at 4 to 14B for no training. Every REAL-6 number uses one fixed category order in the prompt, and
our scorer reads category names, not letters, so the size of the effect here is unknown. *Task (eval only):* re-score the
untrained instruct model and the no-DB and record-in-prompt SFT adapters with the prompt's category list permuted under two
further seeds (shots and query unchanged); argmax flip rate per corrected group and per name type, the accuracy of each
order, and of the two- and three-order averaged distributions; the same with a lettered list and letter readout on the
untrained model only (the adapters were trained on names). Precision matched to training (CLAUDE.md, section 44).

**TRAIN-10 Do counterfactual minimal pairs teach the categoriser to read its evidence rather than its prior?**
Across the series, training on decision data raised in-distribution accuracy and lost on held-out items (reflex, Open-Jev,
decider, Laya), with one clear exception: Nimble's 2,676 rows of base/counterfactual pairs, whose contexts differ in one
fact that flips the label, moved a 9B 24 points on its own distribution and held within 1.2 points of Jev on 13
human-labelled sets (`nimble_analysis.md` 5). kev's executable-rule pairs point the same way. The property that matters is
that the only way to fit both rows of a pair is to read the fact. Our labels come from the generator's rules, so pairs need
no LLM verifier. *Task:* from the REAL-6 generator, for each training episode a sibling that changes one piece of evidence
and recomputes the label: (a) the user's history files the query's merchant under another of the user's categories (the
shots carrying it relabelled consistently), so the label follows the history; (b) the merchant's record names products of
another standard category, so the label follows the record through the user's scheme; keep pairs whose labels differ.
No-DB and record arms, pairs against an equal count of unpaired episodes at the same token budget, one epoch, same seed;
three seeds if the first differs by more than the seed spread. Scored on row 42's held-out-user folds and row 45's
corrected groups, with a paired-sibling accuracy (both rows right) beside plain accuracy; re-scored on row 43's set when it
exists. The split-category items stay a coin flip by construction and are reported apart.

**REAL-14 Can the categoriser say "needs review" when it has no evidence, and is its confidence lower there?**
A listed abstain option without training gets confident wrong answers (SemIf's `insufficient`; `2405.05904`). decider trains
it: 10% of questions with three or more options get an abstain option, and in a quarter of those the true categories are
swapped for another user's so abstain is correct (`decider_analysis.md` 4.3). kev's uniform targets on evidence-free items
cut the share answered at 0.9 confidence or more from 0.19 to 0.00 (`kev_analysis.md` 3). Our evidence-free items are the
opaque merchants with no record and not in the history, where the no-DB categoriser reads 6 (Table 37.2). *Task:* the no-DB
and record SFT arms with the decider augmentation (a "needs review" option in one of several wordings) and, in a second
arm, uniform targets over the user's categories on evidence-free episodes; measured with row 49's metrics: the share of
evidence-free items answered at 0.9 or more, abstain precision and recall, and the accuracy and coverage change on items
that do have evidence.

**MODEL-6 Can a small encoder with one scored `[MASK]` marker per category do the categoriser's job?**
Laya (`laya_analysis.md`) and Verdict 2.0 (`openjev_verdict_analysis.md`) put the options first, each behind its own
`[MASK]`, then the state, and score each marker's hidden state with a small trained head; one forward pass, options defined
at request time. Both collapse off their training domain on general decision benchmarks, which does not matter for a
categoriser that only ever sees transactions. Section 46's GLiClass, the nearest thing we ran, reads 85.8 fine-tuned with
the record (SFT 90.2), but the 24 shots hurt it in every arm, "not investigated further"; section 29's ModernBERT failed
with a letter at a single mask because our answers are several tokens, which a scored marker per option avoids. At a
million users a 400M encoder at milliseconds per item is the production argument. *Task:* Laya's `DecisionModel` (code in
`~/projects/Taytay/laya-hf/rl_common.py`) on REAL-6, fine-tuned across users: the user's categories as marked options, then
the query statement, the record, and the shots tagged "statement -> category"; soft-target cross-entropy (no Gaussian RL,
section 3.2 of the memo), options shuffled per episode; from the Laya checkpoint and from plain ModernBERT-large, with a
longer context than Laya's 512 if the shots need it; no-record and record arms; against GLiClass tuned and the SFT
categoriser on row 45's groups and row 42's held-out folds; milliseconds per item beside the 3B's. The question is whether
this layout lets an encoder read the shots where GLiClass could not.

**MODEL-7 Does a bidirectional slot read (a diffusion LM) categorise better than a causal readout, and does it hold at scale?**
djev-dev and razorback16/openjev (`diffusiongemma_djev_analysis.md`) read every answer from one decoder pass of
DiffusionGemma 26B-A4B: a fixed answer template, one single-token label slot per question, one denoising step, the exact
log-probabilities of the allowed labels at each slot. Untrained it is #3 on JevBench, and with a thinking pass first one of
the only open systems above Jev on hard items. It does not fit a 3090 (52 GB BF16; the 18 GB NVFP4 build targets
Blackwell). Smaller open diffusion LMs do, and the same read works on them in plain transformers (the letter slot as the
mask token, one forward, softmax over the allowed letters; 8 to 20 categories fit single-token letters), found 2026-09-23:
nvidia/Nemotron-Labs-Diffusion-3B and -8B (one set of weights that decodes causally or by diffusion by switching the
attention pattern, so the two readouts are compared on identical weights), Dream-org/Dream-v0-Instruct-7B (converted from
Qwen2.5-7B, paired with Qwen2.5-7B-Instruct as the model card of DiffusionGemma pairs it with Gemma 4),
GSAI-ML/LLaDA-8B-Instruct (trained from scratch as a diffusion LM), and LiquidAI/LFM2.5-Encoder-350M-Diffusion (an encoder
made a diffusion chat model, the bridge to MODEL-6). *Task, local (row 54):* zero-shot on REAL-6 with the 24 shots, with and
without the record: Nemotron 8B causal letter readout against its one-step slot read, Dream-7B against Qwen2.5-7B-Instruct,
LLaDA-8B, LFM2.5-Encoder; row 50's order permutations on each (djev and openjev publish no rotation test, and the commit-order
paper suggests the slot read leans less on left context); temperature and ECE by row 49's code; if one beats its causal twin
by more than the seed spread, a LoRA fine-tune of the slot read with the SFT categoriser's data as a follow-up. Remote code
may need its own transformers pin (Dream's was written for 4.x; Nemotron needs 5.x); use a separate venv rather than move
the project's. *Task, large GPU (row 55):* DiffusionGemma 26B-A4B in BF16 through djev-dev's pinned runtime on one 80 GB
GPU (Modal H100, row 34), zero-shot on REAL-6 and on row 43's set when it exists, one-step read with and without the record,
and with a thinking pass first as a second arm: the reference for how much a much larger backbone reads with no training,
and a candidate teacher for soft labels.

## Added 2026-09-23: training efficiency

**TRAIN-11 (O) Where do the categoriser's GPU hours go, and can the same accuracy come cheaper?**
Row 47 put the no-DB categoriser at 800 steps (77 minutes, 73.3) to 1,600 (147 minutes, 76.7), which made row 42's chain
14 hours. One step is 16 sequences of about 850 tokens (the category list, 24 labelled shots, the query) through a 3B model on
the NF4 4-bit base, at about 2,500 tokens a second (section 7's 3B figure is 2,700): the card is busy, and the loss falls on the
answer's 1 to 5 tokens, so each 850-token pass teaches one label. The owner asked for two tests. (1) The loss on every shot's
label as well (each predicted from the category list and the shots before it, the section 21 all-answer idea on this prompt):
about 25 supervised answers per pass, at the risk that early shots, with few demonstrations before them, teach a different
task. (2) A small run on the bf16 base instead of 4-bit: QLoRA dequantises every weight on every pass, and a 3B bf16 base fits
in 24 GB. *Experiment:* the no-DB arm, seed 0: all-label loss at 100, 200 and 400 steps on the 4-bit base; the plain recipe at
200 steps on the bf16 base; the all-label loss at 200 steps on bf16; each scored on REAL-6 (bf16 adapters on the bf16 base) and
the ARC / MMLU / ICL items, against row 47's 4-bit curve (200 / 400 / 800 / 1,600 steps); report minutes, tokens per second,
peak memory and minutes to reach the 800-step accuracy.
**Status (2026-09-26):** answered, PLAN step 56, REPORT.md 52. The hours went into re-reading 24 shots to learn one label, not into precision. The loss on every shot label reaches in 100 steps (9 min, 79.4) what 1,600 plain steps (147 min, 76.7) reached; on held-out users 200 steps read 74.5 against the 800-step recipe's 70.1 (+4.4 [0.9, 7.8]), half the trained-user gain having been memorisation, and the model stops consulting its own chain knowledge (known DB-only chains 87 to 70), which weighting the answer to half the loss does not restore. bf16 and the 4-bit base: same speed, same accuracy (H100, three seeds: 80.0 +- 1.4 against 80.5 +- 1.0); bf16 is the default from 2026-09-26.

**REAL-15 (O) Does the fact DB go into the weights better as supervised categorisation decisions than as prose?**
The owner (2026-09-25), after the all-label result: measuring loss on more tokens may be the better way to impart a database.
Every records-in-the-weights arm so far (sections 38, 43, 45) trained the record sentences as full-sequence text, which is loss on
every token already, and reached the DB-only merchants at 59 +- 12 (3.3 passes) to 69 +- 6 (6.7), opaque ones never above 44,
against 97.5 with the record in the prompt; the species sections (8, 27, 29) found knowledge trained as prose stays in the trained
form. The all-label loss (REPORT.md 52, row 56) instead supervised the decisions the task makes. *Experiment:* database episodes
(`DBEP`): synthetic statement rows of every DB merchant, DB-only ones included, labelled with a training user's own name for the
merchant's DB category (a category field, as real POI databases carry; split categories skipped), as shots and targets inside
all-label episodes, against the prose records given the same category (`DB_CAT`), both, and neither; on row 42's held-out-user
folds so that no (user, merchant) answer is supervised for a scored user; the DB-only merchants without the record (known chains
and opaque ones) are the cell that measures it, plus the other groups of REPORT.md 48 and ARC / MMLU. Frame: this improves the
no-record route (retrieval missing or unavailable at serving time); with the record in the prompt the DB-only merchants are at 97.5.
**Status (2026-09-26):** answered, PLAN step 57, REPORT.md 53. Yes, by a wide margin. On held-out users with no record in the prompt, database episodes put the DB-only merchants at 94.3 (+50.0 [35.2, 64.4] over no DB), opaque ones at 96.1 (from 11.8), in every fold; prose records with the same category reach 64.8 (opaque 35.3); both together 91.8. Every other cell gains too (all +16.5, coined +20.9). Untested: real bank renderings (row 43), a database larger than 240 merchants (row 59), and the record in the prompt given the same category field (row 58).

## Added 2026-09-26: database episodes against retrieval, and at scale

**REAL-16 (O) Do database episodes beat the record in the prompt when both carry the same information?**
Row 57 put the DB-only merchants at 94 on held-out users with no record at test, after training on database episodes built
from a category field; the record-in-prompt categoriser has only ever seen product sentences (section 38: 97.6 on training
users, 81.6 overall on held-out ones). *Experiment:* on row 42's held-out folds, one hardware and precision for every arm (H100,
bf16, one 16-sequence pass per step, 200 steps): no DB (all-label), database episodes (all-label, `DBEP=0.5`), the record in the
prompt (plain SFT, products only), and the record in the prompt stating the category (`REC_CAT`, `real6.category_record`, in
training and at test); the corrected groups of REPORT.md 48, the DB-only merchants split into known chains and opaque ones.
**Status (2026-09-26):** answered, PLAN step 58, REPORT.md 55. They tie: on held-out users (H100, bf16) database episodes with no record at test read 87.8 [84.1, 91.0], DB-only 93.4 (opaque 94.1); the record in the prompt with the category 86.6 [82.5, 90.4], DB-only 91.8 (opaque 92.2); paired +1.2 [-2.7, 5.0] and +1.6 [-3.5, 6.5]. The category field adds little to the record (+1.6 overall). The episodes' edge on coined names (+9.2) is the all-label loss's. Next: the capacity of episodes (REAL-17) and the two combined.

**REAL-17 (O) How many merchants can database episodes hold?**
Row 57's database has 240 merchants; a production one has millions, and section 29 found interference once a rank-64 adapter
holds 5,000 species at a fixed exposure. *Experiment:* the REAL-6 database padded with generated merchants (opaque names, a
standard category each, product records from the category pools) to about 1,000, 5,000 and 20,000, database episodes drawn
over all of them; the DB-only merchants of REAL-6 scored on held-out users as before, at a fixed step budget (fewer rows per
merchant as the database grows) and at a fixed exposure per merchant (steps grow with the database); the point at which the
DB-only accuracy falls is the capacity figure for a 3B model with a rank-64 adapter.
**Status (2026-09-26):** answered at this range, PLAN step 59, REPORT.md 58. At a fixed exposure of 30 database rows per merchant, the DB-only merchants read 91.8 / 95.9 / 93.4 / 92.6 at 240 / 1,000 / 5,000 / 20,000 merchants (opaque 92 / 94 / 92 / 90), with no loss elsewhere: no interference up to 20,000 for a rank-64 adapter on 3B. A fixed budget spread over more merchants loses them (7.2 rows each 82.8, 1.4 rows 69.7, 0.4 rows no gain). Cost is linear: 20,000 merchants took 89 H100 minutes. Open: the minimum rows per merchant, beyond 20,000, real merchants (REAL-21).

**TRAIN-12 (O) Is 16 sequences per step the right batch for the categoriser, now that an H100 can take more?**
The owner (2026-09-26): have batch sizes been ablated on the H100? Only the layout was: four passes of 4 against one of 16, at the
same 16 sequences per optimizer step (section 54: same accuracy, 1.6x the throughput). The effective batch has been 16 since the
categoriser was built (section 38) and never varied, nor the rate with it. *Experiment:* the all-label recipe on bf16, all 20 users,
H100: effective batch 32 and 64 at equal samples (100 and 50 steps; the rate at 1e-4 and scaled by the square root of the batch
ratio) and at equal steps (200, the scaled rate), against the three batch-16 seeds of section 52.4; passes of 8, 16 and 32
sequences for the throughput curve; REAL-6 by group and the ARC / MMLU / ICL items. The 20-step warmup is fixed, so the 50-step
runs warm up for 40% of their training.
**Status (2026-09-26):** answered, PLAN step 60, REPORT.md 56. Keep 16 sequences per step at 1e-4 for 200 steps. At equal samples a batch of 32 or 64 matches it (79.5 to 80.1 against the seeds' 78.8 to 81.6) only with the rate scaled by the square root of the batch ratio (64 at 1e-4: 72.3, under-trained); at equal steps 2x and 4x the data read 81.6 and 80.9. Passes of 8 to 16 sequences saturate the H100 (14,400 to 14,800 tokens/s; 32 per pass is slower, 13,000).

**REAL-18 (O) Can a merchant known from the database be filed under a label the model has never seen?**
The owner (2026-09-26). REAL-6's coined category names come from a list of 24 words shared by all users: 13 of the 25 coined names
the held-out users carry are also some training user's (often for another category), so the coined cell measures familiar words.
On it (held-out users, section 55) database episodes read 63.6 and the record with the category 77.3 on DB-only merchants, n = 22.
*Experiment (scoring only):* every item re-scored with each user's category names replaced by freshly generated words, consistently
in the category list, the shots and the options (a new word per user and category, fixed seed), for the no-DB, episode, record and
record-with-category arms; by original name type and REPORT.md 48's groups.

**REAL-19 (O) Can the categoriser infer what kind of store an unknown merchant is from its name and the user's examples, and does
baking a database in cost that?**
The owner (2026-09-26). Every REAL-6 merchant has a record and its opaque names carry no meaning, so this is untested. *Experiment:*
a small item set of merchants in no database and no history: descriptive names ("Harbor Pet Supply", "Oak Street Dental") whose
kind the name and the user's shots should give, and opaque ones as the floor; statement strings in the generator's renderings,
labels in each user's scheme; scored for the no-DB, episode and record arms (the record arms with no record for these merchants).
**Status (2026-09-26):** answered, PLAN step 62, REPORT.md 57. Built from Overture (1,198 real US places, three renderings of the same items). With a readable name the categoriser infers well: database episodes 72.5 top-1 / 85.5 top-3 on clean strings, 87.9 on descriptive names, 78.6 on chains, 53.5 on names with no category word; baking the database in helps (+4.7) rather than costs. Truncated bank strings cost every no-record model 14 to 16 points; a record built from Overture's category gives 85 in every rendering. The user's coined category names stay at 55 to 65 against a blind Opus 5.5 ceiling of 88 on the same prompts.


## Research agenda, 2026-09-26 (the owner's framing)

The owner: the aim is to understand the science and mechanics, not to tune a product metric. Three questions lead, and every
row from here states which it serves. Product use only weights the metrics: auto-file extremely confident predictions (typically a
merchant the user has filed consistently before), suggest options for the rest. Every run reports one scorecard (EVAL-9): top-1,
top-3, MRR, calibrated uncertainty in bits, coverage at 98% precision; against a baseline ladder (uniform, the user's usage prior,
the user's merchant lookup, other users' labels) and a strong-reader ceiling (a blind Opus pass on a sample); held-out users,
user-resampled intervals, paired one-factor variants.

- **Q1, LLMs and encoders as multiple-choice categorisers.** Answered so far: the option rule (48.4), calibration (50), per-user
  encoders (46). Open: option order (EVAL-8, row 50), the readout (names, letters, generation), the encoder option scorer
  (MODEL-6, row 53), diffusion slot reads (MODEL-7, rows 54, 55), where in the network the decision forms (MODEL-8, row 67),
  many-way behaviour on real labels (POI-1, row 65).
- **Q2, the best way to inject knowledge.** Answered so far: prose records weak, the record in the prompt strong (38, 43, 45),
  the database as supervised decisions strong and equal to the record at equal information (53, 55), and it helps on merchants
  outside it (row 62). Open: capacity (REAL-17, row 59), real-database episodes with held-out places (REAL-21, row 66),
  what the weights store (MODEL-8, row 67), combining episodes and retrieval, the collaborative record (REAL-12, row 44).
- **Q3, inferring meaningless category names.** Answered so far: from training examples the models memorise users' coined names
  and rename augmentation fixes most of it (51); from prompt examples the information is there (blind Opus 88% on coined names,
  clean strings) and the 3B model reaches 42 to 55 (row 62). Open: the controlled label-induction set (REAL-20, row 64), novel
  words at test (REAL-18, row 61), model size, example selection (REAL-9 follow-up, row 46), schemes over real categories (POI-1).

**EVAL-9 One scorecard for every run.** Top-1 accuracy undersells a suggestion interface and uniform chance is the wrong
baseline: on the novel merchants the episode model's top-1 is 57 but its top-3 74 (usage prior 28, uniform 24), and its raw
probabilities leave as much uncertainty as uniform guessing (3.76 bits against 3.72) because it is confidently wrong often.
*Task (CPU):* `ai_experiments.scorecard` over the saved per-option scores: top-1 / top-3 / MRR; log-loss and Brier in bits after
the leave-users-out temperature of section 50; coverage at 95% and 98% precision with the threshold from other users; each
against uniform, the user's usage prior, the user's merchant lookup and other users' labels, with skill = the share of the headroom
over the best of those the model captures; per-item chance normalisation for the number of categories. Re-read every REAL-6 and
novel-merchant run with it.
**Status (2026-09-26):** done, PLAN step 63, REPORT.md 59. On REAL-6's held-out users the no-model cascade (the user's merchant lookup, else other users' labels, else the usage prior) reads 87.0 top-1, level with the best categorisers; their value is the ranking (top-3 97 to 98 against 26) and whatever no lookup reaches (novel merchants: cascade 7.8, models 53 to 85). Thresholds for 98% auto-filing chosen on other users realise 92 to 98% on new users.

**REAL-20 Label induction: when can a model infer what a meaningless category name means from the user's examples?**
Blind Opus solves the novel merchants' coined names at 88% from clean strings; the 3B categorisers at 42 to 55. *Experiment
(scoring only):* a controlled item set on clean renderings, each factor varied with the others fixed and paired items: examples of
the gold coined category in the prompt (0, 1, 2, 4, 8); the examples' kind (same kind of business as the query, same standard
category but another kind, only opaque merchants); the number of coined categories in the scheme; decoys (a same-kind business filed
under another name); scored for the untrained and trained models, model sizes (3B, 7B; 14B on Modal) and a blind Opus sample.
Says whether the models copy the nearest example's label or induce the word's meaning.

**Status (2026-09-26):** done, PLAN step 64, REPORT.md 60. One or two examples of other businesses filed under a coined word teach every model its meaning (v2, with three empty coined categories so elimination cannot solve it: 14B untrained 12 to 66% from 0 to 1 example, blind Opus 25 to 100); fine-tuning with rename augmentation or database episodes brings a 3B to the untrained 14B (77 to 78 at base); opaque example names teach nothing. The models copy the nearest example: one same-kind business filed elsewhere costs 15 to 19 points and 60 to 74% of the errors pick its category, where Opus keeps 97%.

**POI-1 Categorisation of real places at scale.** Overture places (81.5M, 288 basic categories and a finer taxonomy, per-row
licences) as a labelled benchmark: many-way multiple choice on real labels (Q1), synthetic users whose schemes merge, rename and
coin Overture's categories over real places (Q3), and a knowledge-injection testbed at real scale with places held out (Q2).
*Experiment:* frozen sets from a sampled, category-stratified slice (US first): the plain task at 10 / 50 / all basic categories;
user schemes built as REAL-6's are (merge, split, rename, coin) but over Overture's categories; held-out places and held-out users;
renderings obscure / full / clean; the scorecard of EVAL-9.

**REAL-21 Does a database taught as decisions teach facts or a skill?** Database episodes built from real places (Overture),
scored on places in the injected database and on places held out of it, same categories: the first measures stored facts, the
second the general "this kind of business goes in this kind of category" skill that row 62's +5 on merchants outside the database
suggests; capacity against the held-out gain.

**MODEL-8 Where the categoriser's decision forms, and what the weights store.** Linear probes on the query's and the merchant
name's hidden states across layers (section 40's method): whether the standard category, the user's label and the business kind are
linearly present and at which depth, before and after database episodes; whether a DB-only merchant's category is readable from its
name's representation after injection.
