# Literature survey: injecting and extracting knowledge in small LLMs and encoders

Date: 2026-09-14. Scope: 34 papers across six threads, chosen to answer the open questions in
`report/QUESTIONS.md`. Per-paper summaries with table and figure citations live in
`docs/papers/<id>/summary.md`; `docs/papers/INDEX.md` lists them by thread. Every number below
comes from those summaries. Where a claim is the reader's extrapolation rather than a paper's
result, it says so.

How the survey was run. Semantic Scholar and the arXiv API both rate-limited us (HTTP 429), so
candidate papers came from the report's own citations plus targeted web searches, and PDFs were
downloaded directly from arxiv.org and extracted with `pdftotext`. This means no citation counts
and no "who cited this" expansion; the follow-up list at the end is from the papers'
bibliographies. Two title corrections: the sub-1B baseline ModernBERT-Instruct beats on MMLU is
Qwen2-0.5B, not Qwen2.5-0.5B; and the "saturates at ten paraphrases" line in the first draft of
QUESTIONS.md was a secondary citation, see DATA-5 below.

---

## 1. What the literature settles, by question

### EVAL-1 / EVAL-2 / EVAL-3: how to score multiple choice on small models

**What we do.** Cloze scoring: prompt ends at `Answer:`, options never shown, each option's text
appended and scored by mean per-token log-prob. Holtzman et al. call this AVG; OLMES calls it
CF with token normalisation.

**What is settled.**

- Cloze is the right *elicitation* for models this size. OLMES shows letter-based scoring (MCF)
  is at chance for 8 of 15 base models on ARC-Challenge and only becomes informative after
  hundreds of billions of training tokens; Pythia-1B scores 26.5 MCF vs 31.1 CF on MMLU
  (2406.08446, Table 6, Figure 1). Alzahrani et al. and Zheng et al. both find symbol scoring
  carries large position and token bias (RStd 8 to 16 for 7B and 13B models). Do not switch to
  letters.
- Mean-per-token normalisation is the wrong *normaliser*. Oostermeijer measures a within-item
  Kendall tau of +0.2 to +0.5 between option length and score under token or byte normalisation
  on Qwen3 600M to 4B, i.e. it over-corrects toward long options, while raw sums under-correct
  toward short ones (2607.12767, Tables 1, 2). The reviewer's EVAL-2 mechanism is confirmed:
  later tokens of a multi-token option are near-deterministic, so averaging rewards length.
- Two corrections exist and should both be reported. PMI_DC (2104.08315): sum log P(option |
  prompt) minus sum log P(option | domain premise). It gains 15 to 25 points on OpenBookQA at
  2.7B and helps at every GPT-2 size; OLMES adopts it (with premise `Answer:`) for exactly the
  task type we have, fixed vocabularies of unequal-prior strings, and finds Pythia-1B goes from
  20.2 (none) to 30.4 (token) to 40.4 (pmi) on OBQA. Bayesian accuracy (2607.12767): sum minus
  b times length with b fitted within-item over the eval set; no extra forward passes; |tau|
  falls to 0.03 to 0.07. PMI removes the content prior but keeps a small length bias; Bayes
  removes length but not content prior. Report both, plus the current score for continuity.
- Hybrid scoring (options listed, answer text scored) removes the length problem because the
  first tokens fix the option (2607.12767 on MMLU Full-Text; 2402.01781 recommends it as the
  balance between symbol and cloze). OLMES notes it lands between CF and MCF. Add it as a fourth
  line for the 3B model.
- The partial-input baseline is missing. Holtzman's UNC (score options on the domain premise
  alone) and Balepur et al.'s "choices-only" check are the same idea: any level where the
  question-free score beats chance has an artifact. This is the likely explanation for the base
  model's 42.7% on unknowable held-out induction (EVAL-6) and for L5 sitting below chance.
- Every knowledge-injection paper in this survey evaluates by generation (exact match, substring,
  ROUGE, or an LLM judge); Knowledge-Instruct refuses multiple choice outright to avoid guessing.
  Balepur et al.'s constructed-response conversion is the cheap version: greedy-decode after
  `Answer:`, match to gold. Without it our numbers are not comparable to the literature.

**Recommended scorer set** (from the per-token log-probs `option_scores` already computes, all
persisted per item): current mean, sum, PMI_DC with premise `Answer:` and with a content-free
copy of the question, Bayesian with b per level, hybrid (3B only), UNC as a diagnostic row, RStd
over the fixed option vocabulary, and within-item Kendall tau. Plus greedy generation on L1, L3
and the merchant tasks with three-way agreement (cloze, hybrid, generation).

### EVAL-4: are the Timmy items identifiable

Three papers give the same warning from different angles. Snell et al. found that a student
trained under per-task input distributions keyed on the input distribution rather than the task
index, and only mixed inputs (same inputs under different labelings) prevented it (2209.15189,
Table 1). Wei et al.'s randomized-label control (inconsistent labels give no gain) is the null
for "is this induction or format familiarity", and their Figure 14 shows the largest symbol-tuning
gain is at one exemplar per class, where the model is learning that symbols can be labels rather
than any rule (2305.08298). ABFT's unseen-label protocol (query's group absent from the demo
labels) separates copying from induction (2505.14233, Figure 3). Balepur et al.'s item rule 37,
"one and only one correct option", is violated whenever demos differ on more than one attribute.
The fix in QUESTIONS.md (two demos per group sharing exactly one attribute) stands, and should be
paired with the inconsistent-label null and the unseen-label variant.

### EVAL-5: the ICL regression suite

Wei et al.'s out-of-distribution protocol is the template: symbol pools disjoint by construction
(different digit lengths, letter lengths, word lists), datasets absent from every tuning stream,
a shared template pool, and four instruction/label settings (2305.08298, Section 3). Their 11 held-
out datasets are reused by Bornschein et al. with per-size numbers on Gemma-2, so adopting them
gives an external reference (2512.19879, Table 2). The knowledge-editing literature adds a second
locality level we lack: in-distribution neighbours (other species' same attribute, same species'
other attributes), scored as answer-unchanged (2401.01286, Eqn 11). AlphaEdit's general-capability
curves against number of edits are the template for tracking forgetting against steps
(2410.02355, Figure 4). Biderman et al. note IFT-style data (our answer-only episodes) forgets
more than CPT-style data (2405.09673).

### DATA-5 and the augmentation question

**The "ten paraphrases" claim is a chain of citations, and the primary sources say different
things.** 2510.09885 attributes it to Ovadia 2023 and Mecklenburg 2024. Ovadia shows monotone
gains up to ten GPT-4 paraphrases with no saturation analysis (2312.05934, Figure 4). Mecklenburg
shows diminishing returns from 5x to 10x *token-scaled* QA, caused by 20% of atomic facts never
being covered; fact-scaled sets keep improving (2404.00213, Figures 1 to 3). Knowledge-Instruct
finds the plateau at about 3 paraphrases *per atomic fact*, with 5 sufficient (2504.05571,
Figure 4). Allen-Zhu and Li's numbers are about diversity type, not count: five rewrites alone
41%, five sentence orders alone 70%, two rewrites with permutation 96.1% (2309.14316, Figure 3).

**What this means for us.** Our 14 to 20 templates per entity uniformly cover every attribute,
which is the fact-scaled regime; the literature predicts the proposed 1/3/7/14 sweep will flatten
by 3 to 7 templates. The levers that kept scaling in the literature are different from more
templates: (a) cross-attribute sentence-order permutation, which our templates do not vary; (b)
reverse-direction sentences, the only augmentation that lifts backward recall (2510.09885,
Figure 6; 2309.14402, Result 7); (c) multi-entity relation text, EntiGraph's log-linear regime,
where the rephrase baseline stopped scaling by 38M tokens and relation text continued to 455M
(2409.07431, Figure 2); (d) genre diversity, SPA's seven prompt families, though at our token
budget SPA's own rephrase and QA baselines sit within 1 to 5 points of it (2603.22213, Table 1).
And there is a route that needs no paraphrases at all: masked fine-tuning (see TRAIN-5 below).

### EVAL-7 and the reversal curse

Settled and expected: inverse search is near zero under every training regime, augmentation,
model size and data size unless reversed text is in training (2309.14402, Figure 6); MEMIT states
reverse associations must be inserted separately (2210.07229, Section 6). The remedies are
reverse-rendered knowledge texts (a DATA-5 augmentation), or masked fine-tuning, which reaches
0.93 backward accuracy on Llama-3.2-3B without paraphrases against 0.04 for plain fine-tuning
with paraphrases (2510.09885, Table 5). The reviewer's same-category-distractor fix for the
merchant `sells` and `reverse` tasks is independent of this and still needed.

### TRAIN-1 / TRAIN-2: mixing and staging

- Mixed training beats staged training for reasons beyond the learning-rate schedule. Allen-Zhu
  and Li get 86.6% out-of-distribution QA on *unaugmented* data by interleaving QA with the
  biographies, and accuracy rises with the QA token share up to 0.6 to 0.8 (2309.14316, Figures
  1, 10). Knowledge-Instruct's one-stage mix of knowledge instructions with general SFT preserves
  GSM8K and MMLU-Pro where two-stage CPT then SFT loses 8 to 11 points (2504.05571, Table 2).
  Self-Tuning's staging only works because stage 2 reviews QA and stage 3 replays 128 QA items
  (2406.06326, Table 4). Prediction for TRAIN-2: arm D stays behind C even with a restarted
  schedule; the tested remedy is mixing or replay, not scheduling.
- The literature's mixing ratios are stated in three different units, none of them ours. Wei et
  al. by example weight (16% symbol data suffices for ICL gains; flipped-label following scales
  with the fraction). Allen-Zhu and Li by tokens. Gekhman et al. by known-vs-unknown facts, and
  their best recipe is MaybeKnown examples, which maps onto real-entity replay we do not have
  (2405.05904, Table 2). Two concrete fixes for our 2-to-4-token E-stream loss: Bornschein et
  al. put loss on every answer in the k-shot sequence, multiplying the E signal by about k+1 at
  no sequence cost (2512.19879, Appendix B.1); ABFT moves the E objective out of the LM loss
  entirely onto attention rows (2505.14233).
- Self-Tuning's E stream is derived from the knowledge texts themselves (blanked attributes,
  sentence completion, in-document MCQ, NLI with corrupted entities), not only from label
  episodes, and removing those self-reflection tasks hurts most (2406.06326, Figure 4). That is
  a fourth stream we do not have, and it needs no generator.

### TRAIN-3 / TRAIN-4: when to stop, and LoRA vs full FT

- Our facts are 100% unknown by construction. Gekhman et al. show unknown facts are fitted late
  and the damage to known facts appears after the dev peak (5 to 10 epochs at lr 1e-5), so a
  fixed 800 steps with one final eval cannot see the trade-off (2405.05904, Figures 1, 3).
  Biderman et al. find forgetting monotone in duration in every panel. The every-200-steps
  evaluation in TRAIN-3 is the right design; add a held-out known-facts set.
- Biderman et al.'s recipe: rank 256 on all modules, alpha = 2r (ours already), LR sweep in
  [1e-5, 5e-4] taking the highest stable value, and LoRA for IFT rather than CPT. Knowledge text
  with full-sequence loss is CPT-like, where the gap to full FT did not close at any rank
  (2405.09673). But three results cut the other way: Allen-Zhu and Li found LoRA beat full FT
  for extraction, with embedding-layer rank mattering more than q/v rank (2309.14316, Figure 11);
  prompt distillation at rank 16 beat SFT at rank 512 (2412.14964, Table 13); Bornschein et al.
  found rank 16 equal to full FT for prompt-format training (2512.19879, C.8). Target type
  matters more than rank. The survey's FT-M (single traced FFN layer, masked cross-entropy)
  equals or beats MEMIT on insertion and is a cheap arm between full LoRA and editing
  (2401.01286, Table 4). Their Table S14 arithmetic explains the 3B full-FT OOM: fp32 Adam is 16
  bytes per parameter; bf16 weights with 8-bit AdamW is about 6, roughly 18 GB before
  activations.

### BASE-1 / BASE-3: RAFT and distillation

- Context distillation is the cheapest route to the with-context ceiling. Hard-label version
  (Snell et al.): train on the bare prompt with answers sampled from the with-context model; no
  new loss code, same cost as an episode arm. Soft-label version (Kujanpää et al.): KL at T=2 to
  the with-context teacher over the answer span; on Qwen2.5-3B-Instruct it beats SFT by 7 to 10
  points, matches embedding RAG closed-book, and forgets less (2412.14964, Table 2); about 2x
  arm C. Prediction: bare-format L1 recall moves from about 20 toward the 98.8 with-context
  number, because the student is trained on exactly the bare format where our cloze scores
  collapse. Caveat: a 7B teacher cannot logit-distill into Qwen2.5-3B (different vocabulary);
  use hard labels or a 3B teacher.
- RAFT's settings: one gold plus one to three distractors, gold present 40 to 100% of the time,
  randomized order; golden-only training is the worst configuration, and a fine-tuned model that
  never saw context can score *below* its no-context number when fed retrieved context
  (2403.10131, Table 1 DSF vs DSF+RAG, Figure 6). Our `ctx_frac = 0.5` with no distractors is
  that worst configuration. Both EntiGraph (Table 3) and prompt distillation (Table 2) show
  fine-tuned plus retrieval beats base plus retrieval, so REAL-2 should be measured on arm C and
  on the distilled adapter, with oracle and retrieved reported separately (REPORT-3).
- Order: soft-label distillation first (BASE-3), then C-RAFT (BASE-1, REAL-2).

### BASE-2: knowledge editing

Worth running in a scoped form, as a locality baseline, not as a replacement for arm C.
Expected from the literature: high trained-format and bare-format recall (efficacy and
paraphrase 90+ on MEMIT/AlphaEdit), ICL suite and MMLU near base (AlphaEdit's general-capability
result), and no gain on manipulation, label induction or reverse tasks, since portability is the
documented weak point of every editor (2401.01286, Table 4; MQuAKE multi-hop in single digits,
2410.02355, Table 5). Three cautions. MEMIT keys depend on the subject only, so five attributes
per species would be averaged into one key; edit one target vector per species jointly over its
prompts or treat each species as a profile edit (2210.07229, Eqn 19). Edits do not survive later
LoRA: 10 to 45 point efficacy loss, AlphaEdit worst, Llama-style models more fragile than GPT-J,
DeepSeek failed to edit at all with default layers (2511.05852); so never chain MEMIT then LoRA
episodes without measuring decay, and budget a causal-tracing layer search for Qwen2.5-3B.
2510.09885 Table 6 finds ROME, UnKE and AnyEdit near zero on document-level injection, which
lowers the priority further. On the closest benchmark to our task (WikiData_recent insertion),
FT-M and AdaLoRA equal or beat MEMIT on success, portability and locality (2401.01286).

### MODEL-1: scale

Symbol tuning was never tested below 8B, and at 8B it cost 6 to 7 points where natural labels
were available plus 2 to 3 MMLU/BBH points (2305.08298, Table 1); our 3B arm B showing no
natural-label regression is not explained by the paper and may reflect the in-distribution
suite (EVAL-5). ABFT works from 812M and ICL+FT from 0.6B, with the 2B model showing the largest
gain over frozen ICL (2505.14233; 2512.19879, Table 1), so induction-head training is not
scale-gated the way symbol tuning appeared to be. Self-Tuning finds chat checkpoints retain
worse; prefer base models for injection (2406.06326, Appendix J), while masked FT and prompt
distillation need Instruct checkpoints. A 7B run should report the natural-label column and a
knowledge benchmark separately, since that is where 8B regressed.

### MODEL-2: masked-LM encoders

Stronger and more specific than the first draft anticipated, with two cautions.

- ModernBERT-Large-Instruct answers classification and multiple choice through the MLM head with
  a single `[MASK]` preceded by an untrained `[unused0]` anchor, trained on 20M single-token FLAN
  examples with 20% "dummy" MLM examples; 43.06 MMLU vs Qwen2-0.5B 33.7 (2502.03793, Tables 1,
  2). Multi-token answers are handled by listing options and predicting a letter, never by
  multiple masks. It never tests injecting new facts or in-context learning, and the cited
  evidence puts MLM in-context learning at 900M+ parameters.
- WikiDYK shows span-masked Flan-T5-770M memorises real facts far better than 1B to 8B causal
  LMs, 46.09 vs 16.09 reliability match, scored by generation, but the advantage only opens above
  about 1,000 facts; at 100 facts causal models with span prediction are as good (2505.12306,
  Table 4, Figure 3). At our 136 species the expected win is on capacity (REAL-3), not on the
  current ladder. The winners are encoder-decoders that emit multi-token answers; encoder-only
  RoBERTa with multi-mask generation fails.
- Allen-Zhu and Li's GBERT extracted single-word attributes but not multi-word ones even with
  augmentation (2309.14316, Figure 9). Type names are single words; category names like
  `Pharmacy & Health` are not.
- SetFit at 8 examples per class beat the cloze-based ADAPET by 4 points (2209.11055, Table 2),
  so a masked-LM arm is not automatically the better few-shot extractor for personal categories.

Arm F design: ModernBERT-large, three streams mirroring arm C (knowledge texts with attribute
words masked one at a time as ATP plus 30% random MLM; episodes rewritten as cloze; 20% dummy
examples as regulariser); score 8-way type at the mask over the eight single-token type ids, raw
and PMI-calibrated; remap nonsense labels to single-token rare words or `[unusedN]` tokens; LR
3e-5, 1 to 3 epochs, full fine-tune. Second encoder arm F2: Flan-T5-large span prediction,
scored by generation. Pair both with REAL-3 (1k to 5k species) where the bidirectional advantage
appeared. Masked fine-tuning of the *decoder* (TRAIN-5) gets much of the same benefit without
changing model family.

### MODEL-3 / MODEL-4 / MODEL-5: encoders, casing, pooling

- Casing. Tokenization Falling Short frames case changes as token-identity changes the model has
  no mechanism to relate, and finds even GPT-4 Turbo and Llama3-70B degrade under character
  noise, so scale will not rescue it (2406.11687, Section 5). Our measured 12.6% token survival
  under uppercasing is a sharper number than anything the paper quantifies. Mitigations it
  supports: multi-case renderings in training text, and BPE-dropout at p about 0.2 on name spans
  during LoRA training (a tokenizer-config change), with the caveat that the paper only shows
  benefit on structure probes.
- Embedding upgrade. Qwen3-Embedding-0.6B is a drop-in for MiniLM under sentence-transformers,
  fits full fine-tuning on 24 GB, uses last-token pooling and instruction-aware queries, and
  inherits the cased Qwen BPE (2506.05176). That makes it the cleanest test of the casing
  hypothesis: if the 70.8% bank-string transfer collapses on it and on bge-base-en-v1.5 (cased
  WordPiece) while surviving on uncased MiniLM, the claim "subwords carry knowledge across
  formats" must be restated as a property of uncased tokenizers. Its last-token pooling also
  tests MODEL-5 directly. ModernBERT's tokenizer is OLMo-derived BPE, expected cased, unverified.
- New tokens. MOSAIC reproduces our `ft_newtok_mean` failure (mean init, contrastive only:
  47.76 vs 53.94 base) and fixes it with a joint stage: contrastive loss plus an MLM loss
  restricted to the new tokens at alpha 0.3, mask 0.15, then contrastive-only; stage order
  matters (88.1 vs 70.5 reversed) (2510.16797, Tables 1, 2). This works on MiniLM without a saved
  MLM head because the output projection is the input embedding table. BASE-5 should be rerun
  this way before concluding "never add tokens". Their tokens have thousands of occurrences;
  ours have 14 to 20 texts, so the optima may not transfer.
- Copy Qwen3-Embedding's false-negative mask (drop in-batch negatives scoring above the
  positive plus 0.1) into our contrastive loss; it matters when many merchants share a category.

### BASE-4 / REAL-4: embedding baselines

SetFit is the reference few-shot embedding classifier: label-contrastive pairs (R = 20 per
class, cosine loss, LR 1e-3, one epoch) then logistic regression; 62.3 average at 8 per class vs
43.0 for fine-tuning, about 30 s on a 16 GB GPU (2209.11055). It has no centroid, kNN or
frozen-encoder ablation, so our ladder should supply them on identical items: frozen centroid,
frozen logistic regression, fine-tuned-encoder centroid (current), fine-tuned-encoder logistic
regression, full SetFit. Needs K >= 2 per class; ten random splits with mean and sd. Bornschein
et al. define the two LLM-side baselines we also lack for the Timmy items, FT-Only on x-to-y
pairs and ICL+FT on k-shot episodes, with a prequential selection loop suited to 160 items
(2512.19879).

### STAT-1 / STAT-2 / STAT-3

No paper in the survey changes the reviewer's verdict; most of them share the weakness (single
seeds, no CIs). OLMES reports standard errors of 0.8 to 2.2 on 500 to 1000 items and warns that
floating-point ties can flip near-equal options (2406.08446, Table 5, Section 3.5). Gekhman et
al. use 100 random test splits with paired t-tests (2405.05904, Appendix J). RStd, PriDe priors,
UNC, EFR-style flip counts between arms, Kendall tau and bootstrap CIs all need per-item,
per-option scores saved, which is STAT-2. Freezing item sets (STAT-3) is what the ICL suite
already does and the ladder does not.

### REAL-1 / REAL-2 / REAL-3

Gekhman et al.'s SliCK labelling (known / maybe / weakly / unknown, from 10 few-shot prompts and
sampling) is the tool for splitting real merchants into buckets before fine-tuning and reporting
accuracy by bucket; their IDK relabelling result suggests teaching an "unknown merchant" label
for the long tail (2405.05904, Table 3). EntiGraph's 350x synthetic-to-source token ratio and
per-book counts scaling with entity count squared are a warning for 10k entities (2409.07431);
MEMIT's 10k edits on 6B with about 10 points of specificity loss is the only capacity number at
that scale (2210.07229); Allen-Zhu and Li Part 3.3 (2404.05405, not read) is the capacity-
scaling reference to fetch. RAFT's distractor result means end-to-end retrieval could score
below no-context on arm C unless trained with distractors (2403.10131).

### New item, TRAIN-5: masked fine-tuning of the decoder

Not in the original register. Pan et al. reframe each knowledge text as a chat turn: the user
shows the document with a random 5 to 95% of tokens replaced by a reserved token and asks for the
passage; the assistant emits the original; standard AR loss on the assistant tokens. On
Llama-3.2-3B-Instruct and Qwen3-4B-Instruct this gives 0.9+ forward and backward recall with no
paraphrases, where plain fine-tuning with paraphrases gives 0.95 forward and 0.04 backward; a
random-token control collapses to the naive baseline, so it is not generic augmentation; editing
baselines score near zero (2510.09885, Tables 1, 5, 6, Figure 14). It needs an Instruct
checkpoint and was not tested below 3B or with LoRA. Cost about 2x arm C in tokens. This is the
single most direct test of "diversity vs distinct sequences" (DATA-5) and of the reversal curse
(EVAL-7), and it is the decoder-side counterpart of the MODEL-2 encoder proposal.

---

## 2. Re-prioritized register

Ordered by expected information per GPU-hour, with the literature reason. Items not listed keep
their original priority.

**Tier 0: free or nearly free, do before any new training run**

1. STAT-2 + STAT-3: persist per-item, per-option log-probs, token and byte counts; freeze the
   ladder, probes and held-out items to versioned JSON. Everything below depends on it.
2. EVAL-2 scorer set from saved logits: sum, PMI_DC (two premises), Bayesian, hybrid (3B), UNC
   row, RStd, Kendall tau. Re-score every saved adapter with `EVAL_ONLY`. Expect the largest
   movement on L5 and bare L1; expect UNC to explain EVAL-6.
3. EVAL-3 constructed response: greedy decode on L1, L3, category and sells items; three-way
   agreement. Prerequisite for comparing with any paper in this survey.
4. MODEL-4 casing check: title-cased bank strings through the existing LLM evals; token-survival
   statistic recorded for every tokenizer in use.
5. REPORT-1, REPORT-3: caveat in the summary; rename "RAG ceiling" to "oracle context".

**Tier 1: one afternoon each on the 3090, highest expected effect**

6. BASE-3 soft-label prompt distillation on arm C's data (teacher = with-context 3B). Prediction:
   bare L1 recall toward 98.8, better ICL retention than hard-label episodes.
7. TRAIN-5 masked fine-tuning on Qwen2.5-3B-Instruct with LoRA, one template per entity, scored
   on EVAL-7 reverse items and bare L1. Tests DATA-5 and the reversal curse in one run.
8. TRAIN-1 fix by construction: all-answer loss on k-shot episodes (Bornschein), plus a
   self-teaching stream derived from the field-guide entries (Self-Tuning); log per-stream and
   per-attribute label-token counts.
9. STAT-1: three seeds for arms A, C, D. Nothing in the literature reduces the need for this.
10. TRAIN-3: evaluate a 400-item subsample every 200 steps, with a held-out known-facts set
    (SliCK-labelled real entities) as the forgetting proxy.

**Tier 2: new arms with published precedent**

11. MODEL-3 / MODEL-5 / BASE-5 embedding block: rerun the universe and merchant embedding
    experiments unchanged on Qwen3-Embedding-0.6B and bge-base-en-v1.5; MOSAIC joint stage for
    new tokens; SetFit and logistic-regression baselines on identical items (BASE-4).
12. BASE-1 C-RAFT: one gold plus two retrieved distractors, P about 0.6; evaluate no-context,
    oracle, retrieved top-3 (REAL-2) on arm C and on the distilled adapter.
13. MODEL-1: base/A/C/D on Qwen2.5-7B (QLoRA) and Qwen2.5-1.5B; report natural-label ICL and a
    knowledge benchmark separately.
14. DATA-5 as re-scoped: 1/3/7/14 templates, then add sentence-order permutation, reverse-
    direction sentences, and LLM-written relation texts; plot against distinct texts.
15. TRAIN-4: r=256 all-modules, 3B full FT with 8-bit AdamW and bf16 weights, FT-M single-layer
    arm, MLP-only LoRA.

**Tier 3: worth doing, lower expected yield at current scale**

16. MODEL-2 arm F (ModernBERT-large ATP) and F2 (Flan-T5-large span prediction), paired with
    REAL-3 (1k to 5k species) where the bidirectional advantage appeared.
17. BASE-2 scoped editing arm: MEMIT and AlphaEdit via EasyEdit, one target vector per species,
    single batch, causal-tracing layer search for Qwen; never chained before LoRA episodes.
18. EVAL-4 item rebuild with identifiable rules, inconsistent-label null, unseen-label variant.
19. EVAL-5 suite rebuild on Wei et al.'s protocol with in-distribution neighbour locality.

**Corrections to QUESTIONS.md made by this survey**

- LIT-1: the "saturates around ten paraphrases" line is replaced by the three primary findings
  above (Ovadia: monotone to ten; Mecklenburg: coverage-driven diminishing returns 5x to 10x;
  Knowledge-Instruct: plateau at 3 per atomic fact).
- LIT-1 and MODEL-2: ModernBERT-Instruct's sub-1B comparison is Qwen2-0.5B.
- DATA-1 gains a literature footing: Allen-Zhu and Li's bioS-couple result shows correlated
  attributes are chained to each other rather than to the name (2309.14316, Figure 6), so
  weakness-as-rotation-of-type will be stored as a function of type.

---

## 3. Follow-up papers worth fetching (from bibliographies, not yet read)

Knowledge injection: 2402.12847 (PIT, instruction-tuned LMs are better knowledge learners;
the pretrained-LLM confirmation of Physics 3.1), 2404.05405 (Physics 3.3 capacity scaling laws,
for REAL-3), 2403.13799 (reverse training), 2407.10804 (Mix-CPT), 2406.11194 (in-context
editing from self-induced distributions), 2508.06178 (injection methods in a low-resource
regime), 2403.05612 (unfamiliar examples control hallucination), 2311.09677 (R-tuning).
Scoring: 2102.09690 (Calibrate Before Use), 2502.18798 (ANPMI), 2403.00998 (predictions not
robust under scoring methods), 2406.07545 (MCQ to open-style), 2405.14782 (harness lessons),
2210.12353 (cloze vs symbol). ICL: 2110.15943 (MetaICL), 2202.12837 (role of demonstrations),
2303.03846 (larger LMs do ICL differently), 2209.11895 (induction heads), 2501.15708 (STAICC).
Encoders: 2406.04823 (BERTs as in-context learners), 2310.10322 (bidirectional editing for the
reversal curse), 2504.00472 (memorizing is not enough), 2305.01651 (new entities from
descriptions). Editing: 2202.05262 (ROME), 2401.07453, 2401.04700, 2510.00625, 2601.17343
(locality evaluation), 2305.14795 (MQuAKE). Tokenizers and embeddings: 1910.13267
(BPE-dropout), 1804.10959 (subword regularization), 2505.09388 (Qwen3 report, tokenizer),
2401.00368 (synthetic-pair embeddings), 2212.09741 (Instructor), 2204.01172 (PERFECT).
