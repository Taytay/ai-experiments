# Scan: alternatives to "SFT 3B + retrieved record" (90) and the encoder centroid (84), 2026-09-21

Prompted by the owner's question about IBM's FastFit (PLAN row 40, BASE-6). A subagent scanned the web (WebSearch and page
fetches only; no arXiv or Semantic Scholar API; nothing run) for schemes in FastFit's spirit that could beat or cheapen the
two winning routes of REPORT.md sections 38 and 43 on REAL-6. The numbers below come from abstract pages, READMEs and model
cards, not full papers, so treat them as claims until a paper is read (`references/papers/`). The main session wrote this
file from the subagent's report and added the "Assessment" section.

## Candidates

**1. GLiClass (Knowledgator, arXiv 2508.07662, github.com/knowledgator/gliclass), a direct FastFit relative.**
GLiNER-style single-pass encoder; every label is prepended to the input as `<<LABEL>> name` and the label token's contextual
vector is scored against the text. Zero-shot, but the v3 line supports retrieval-augmented classification (few-shot examples
in the same pass through an `<<EXAMPLE>>` token: text, all labels, true label), and `train.py` fine-tunes on JSON
`{text, all_labels, true_labels}`. gliclass-modern-base-v3.0: 151M parameters, Apache-2.0, average zero-shot F1 0.557 over its
benchmark set, 54 examples/s at batch 1 on an A6000. Fit: zero-shot will sit near the frozen encoder's numbers on standard
names and at chance on coined ones; the interesting arm is fine-tuned on the REAL-6 training users with the user's own shots as
examples, plus the record appended to the text: a one-pass encoder analogue of the 24-shot prompt. Cost: minutes per fit,
seconds per user. Verdict: run as a baseline next to FastFit, same cells.

**2. FastFit and its ancestor TCM (Song, Gu, Huang 2022, arXiv 2205.11409).**
TCM reframes many-class classification as text-to-label matching over label semantics; FastFit adds batch-contrastive
training and the ColBERT-style token MaxSim. Vajjala and Shimangaud's survey (arXiv 2502.11830, 32 datasets, 8 languages) ran
FastFit at 10 per label as the few-shot method because SetFit became intractable with many more than 10 categories; FastFit
beat zero-shot GPT-4 by more than 5 points on the many-class sets (Taxi1500, Scenario), tied it on intent, and on Taxi1500
slightly beat full-data training. The FastFit README defaults to `max_text_length=32`, which our statement + record strings
exceed; raise it. Verdict: TCM superseded; FastFit is row 40.

**3. Retrieved demonstrations instead of the fixed 24 shots (Dr.ICL, arXiv 2305.14128; standard practice).**
Choose the k history rows nearest to the query (the row 37 MiniLM retriever, or bge) instead of a fixed 24. Touches every LLM
row; the seen-merchant cell gets the merchant's own rows in the shots, the unseen cell gets same-product neighbours, coined
names are handled exactly as now (copied from the shots). Cost: one embedding pass per history. Verdict: run as a baseline;
the cheapest plausible lift on the no-DB SFT (57 to 63) and on the untrained base; the SFT arm must also be trained with
retrieved shots to be fair.

**4. kNN Prompting (Xu et al., ICLR 2023, arXiv 2303.13824).**
Run each labelled anchor through the LLM once, keep its next-token distribution as a feature vector; classify a test item by
kNN over the anchors' distributions with their gold labels. Claimed to beat ICL and calibration methods from 2 to 1,024
demonstrations on 0.8B to 30B models. Uses the whole 300-row history rather than 24 shots; coined names cost nothing because
the decision is by neighbours' gold labels. Does not by itself use the record (append it to anchor and query). Cost: about
6,000 + 1,179 forwards per run. Verdict: worth a row later, after 3.

**5. Rerankers as classifiers (BTZSC, arXiv 2603.11991; Qwen3-Reranker-0.6B / 8B on Hugging Face).**
Over 22 zero-shot datasets and 38 checkpoints, Qwen3-Reranker-8B macro-F1 0.72 beats 4 to 12B instruction LLMs (at most
0.67); GTE-large embeddings close and cheapest; NLI cross-encoders plateau. Fit: pointwise (statement + record, label name)
scoring; zero-shot cannot read coined names, but the 0.6B reranker is a small decoder that can be fine-tuned as a
cross-encoder on (statement + record, label + three user examples) pairs, a five-times-cheaper version of our option-scoring
model. Cost: zero-shot minutes; fine-tune under an hour. Verdict: zero-shot 0.6B with the record as a cheap sanity row; the
fine-tuned cross-encoder worth a row later, on the cost axis.

**6. LLM embedders in the encoder route (Qwen3-Embedding-0.6B, arXiv 2506.05176).**
Section 24 already had Qwen3-Embedding as the strongest encoder on the synthetic universe (99.9 trained centroid vs bge
99.4); the REAL-6 encoder route (83.7) used bge-base. A swap in the existing script with the instruction prefix. Cost: the
0.6B tunes contrastively in minutes; 4B needs QLoRA. Verdict: worth a row (cheap); expectation +2 to 5 on unseen merchants.

**7. Label-description training (Gao, Ghosh, Gimpel, EMNLP 2023, arXiv 2305.02239).**
Fine-tune on data that only describes labels (related terms, dictionary entries, templates), no annotated texts; +17 to 19
absolute over zero-shot, robust to verbaliser and label-token choices. Fit: renamed names get a description; coined names
have none, but one can be induced from the user's shots (an LLM writes "Zorbit: gym, protein, fitness" from the 24 rows) and
put in the prompt as a glossary line per category. Touches the unseen-merchant x coined-name cell, where every route is
weakest. Cost: one generation per category per user. Verdict: worth a row, a "label glossary" prompt variant, base and SFT.

**8. Symbol tuning (Wei et al. 2023, arXiv 2305.08298); Semantic Anchors (arXiv 2511.21038); In-Context Fixation (arXiv
2605.08295).**
Symbol tuning fine-tunes on ICL episodes whose labels are replaced by arbitrary symbols and improves unseen-label ICL.
Semantic Anchors: eight 1 to 12B models never flip label semantics through ICL alone. In-Context Fixation: 0.8 to 8B models
put 42 to 67% of their mass on tokens seen in label positions, even nonsense ones. These explain why coined names already work
in the 24-shot prompt (the model copies from the demonstrated set) and why renamed names that collide with a standard meaning
are the risky cell. Our label SFT over 20 schemes is a small symbol-tuning set; the actionable variant is a rename
augmentation in SFT (randomly replace label names with fresh words per episode) to make the adapter scheme-independent for
unseen users. Cost: one SFT run per seed. Verdict: worth a row, and required before claiming generalisation to new users.

**9. Option-score calibration (Batch Calibration, ICLR 2024, arXiv 2309.17249; domain-context calibration; PMI).**
Inference-only corrections for label-prior bias: divide each option's probability by its probability under a content-free
input (PMI / DC), or subtract the batch mean over test items (BC). Fit: coined names are multi-token, low-prior strings; the
untrained-base rows (31 / 58) are where prior bias is largest; the SFT rows are probably already calibrated (section 11
found the same on the ladder). Cost: one extra forward per user per option. Verdict: run as a baseline on the untrained rows.

**10. Structured output / constrained decoding.** The benchmarks found (RANLP 2025 "The hidden cost of structure"; arXiv
2501.10868; arXiv 2605.02363) agree that grammar masking distorts the distribution and can degrade accuracy, at a latency
cost. Verdict: skip; option log-prob scoring is the right primitive. The speed trick is engineering: score every option from
one cached prefix (shared prompt KV, one forward per continuation).

**11. Per-user adapters vs a shared model: OPPU (EMNLP 2024, arXiv 2402.04401), Profile-to-PEFT (ACL 2026, arXiv
2510.16282), USER-LLM (arXiv 2402.13598).** OPPU trains one LoRA per user on LaMP; P2P replaces it with a hypernetwork from
the encoded profile to LoRA weights and generalises to unseen users; USER-LLM compresses histories into soft prompts. All
target generation and preference tasks, none has user-defined label sets; P2P needs meta-training over many users (we have
20). Verdict: skip for now; the shared adapter with the scheme in the prompt is the cheaper design, and the rename
augmentation (8) is the test of whether it holds for new users.

**12. Doc-to-LoRA (Sakana, arXiv 2602.15902) and Text-to-LoRA (ICML 2025, arXiv 2506.06105).** A 309M hypernetwork turns a
document into a rank-8 LoRA in under a second; 83.5% of the full-context SQuAD score. The parametric route for the DB at
inference time. Verdict: skip: bounded by the in-context number the retriever already reaches, and the meta-training is a
project on its own; cite in the "why not parametric" paragraph.

**13. Distilling the 3B categoriser into an encoder** (arXiv 2406.17633; the caution in arXiv 2504.15432: LLM-labelled
students plateau early and lose on minority classes). The gold-trained encoder reaches 84; a student of a 90 teacher cannot
pass its teacher and, on our labelled data, has no reason to pass the gold-trained encoder unless it gets far more unlabelled
transactions per user. Verdict: skip for accuracy; a cost row only if a per-user encoder is ever needed.

**14. Late interaction beyond FastFit.** PyLate (arXiv 2508.03555) trains ColBERT models, but nothing found applies MaxSim to
few-shot classification other than FastFit itself. Verdict: skip; FastFit is the test of the idea.

**15. Transaction-specific work.**
- "How Small Can You Go? LoRA fine-tuning 270M to 8B for merchant information extraction" (arXiv 2606.08051, 23 runs):
  Qwen3.5-0.8B F1 94.75, 4B 96.60, Llama-3.1-8B 96.75 on merchant extraction from noisy descriptors. Sub-1B decoders handle
  the string-normalisation half of the task; no public dataset stated.
- "Categorising SME bank transactions with ML and synthetic data" (arXiv 2508.05425): synthetic generation, a fine-tuned
  classifier and calibration to the real label distribution; 73.5 +- 5.1 accuracy, 90.4 on high-confidence predictions. Their
  selective-prediction framing (confidence-gated accuracy) is a reporting idea for us. No code found.
- Visa (EMNLP 2025 industry, arXiv 2601.05271): LLM sentence embeddings of enriched merchant fields as initialisations for a
  transaction model. LATTE (arXiv 2508.10021) and TransactionGPT (arXiv 2511.08939) embed LLM-written merchant / MCC
  descriptions with Qwen3-Embedding-8B: the same "record text as the merchant's representation" idea as our DB, at industry
  scale.
- Open data: Hugging Face `DEVILHADYOURMOM/us-bank-transaction-categories` (16k synthetic descriptions, 16 balanced
  categories, six real US statement templates with store numbers, POS prefixes and ACH patterns, MIT licence; DistilBERT
  99.75 validation / 90 on real strings; generator at github.com/wnstnb/foliome). An external transfer check (map their 16
  categories to a standard scheme, score our SFT model zero-shot) and a free source of decoy strings and format variants for
  the retriever test.

## The subagent's ranked shortlist

1. **Retrieved shots.** Train the label SFT as in section 38 but with each training prompt's 24 shots being the 24 history
   rows nearest to the query under the row 37 MiniLM retriever (excluding the row itself), one arm without the record and one
   with. Score REAL-6 with retrieved shots at test, paired per item with the fixed-24 adapters and with the untrained base
   under both shot selections. Decides whether the no-DB gap (57 to 63 vs 90) is partly a shot-selection artefact and whether
   retrieved shots move the record-in-prompt number; the seen-merchant cell changes most.
2. **GLiClass fine-tuned with examples, alongside FastFit (row 40).** Train gliclass-modern-base-v3.0 on the REAL-6 training
   users' histories, text = statement (+ record), all_labels = the user's scheme, the user's k nearest history rows as
   examples; also the zero-shot model unchanged. Score per cell next to bge centroid, logistic regression, SetFit and
   FastFit. The coined-name cells say whether the example channel or the label channel carries it.
3. **Calibrated option scoring.** Train nothing. Per user, a content-free prompt (the 24 shots, an empty statement) once per
   option gives the prior; report PMI and Batch Calibration for the untrained base with and without the record and for the
   SFT adapters. If the SFT rows move by less than the seed floor, the row closes with "SFT calibrates".
4. **Rename augmentation in the SFT.** The record-in-prompt SFT with, per episode, a random subset of the user's category
   names replaced by fresh coined words consistently across shots and target; seeds as in section 43. Score REAL-6 on the 20
   users and on a held-out-user split (train on 15, test on 5). Decides whether the 90 holds for users whose schemes were
   never in training, and what the augmentation costs on standard names.
5. **Qwen3-Reranker-0.6B as the cheap categoriser.** Zero-shot first (pairs of statement + record and category name, the
   yes-probability as the option score), then fine-tuned pointwise on (statement + record, category name + three of the
   user's rows). Paired with the 3B SFT, with time per 1,000 items on the 3090 next to accuracy.

## Assessment (main session)

- The two that test something the report cannot currently answer are 4 (does the 90 hold for users whose schemes were never
  trained on; every REAL-6 number so far is on the 20 training users' own schemes, seen and unseen merchants but not unseen
  users) and 1 (the fixed 24 shots are stratified by category and then random, so a same-merchant row is in the prompt by
  chance only). Both are one SFT run per seed.
- 3 is nearly free and mostly changes the untrained-base story; section 11 already found calibration does nothing for
  trained arms on the ladder, so run it on the base rows only.
- 2 belongs with row 40 as a second encoder baseline; 6 is a one-line swap in the encoder script. 5 is a cost row.
- Skipped with reasons in the list: per-user adapters, document-to-LoRA, distillation, constrained decoding, late interaction
  beyond FastFit.
- Scale, added by the owner on 2026-09-21: the real application has over a million users, each labelling their own
  transactions in a personal way; most use the default category set, some create their own labels, and the reasons for
  putting a merchant under a label are the user's own. (The per-user-adapter papers in 11 were skipped because *they* have no
  user-defined label sets, not because we lack them.) At that scale: one LoRA per user is a storage and serving problem
  (a million adapters), so the shared model with the user's scheme and shots in the prompt, or a hypernetwork from the user's
  history to a small adapter (Profile-to-PEFT, which needs many users to meta-train and a million supplies), are the two
  designs; the held-out-user evaluation (4) becomes the production metric rather than a check; retrieved shots (1) stop being
  optional, because real histories run to thousands of rows and the prompt holds 24; and the cost rows (5, the encoder route)
  matter, because scoring is per transaction per user. One gap in REAL-6 itself: its schemes merge, split and rename the 12
  standard categories, so a merchant's category follows its standard one; a user who files a gym under "Health" and another
  under "Treats" (personal reasons, not renames) is not modelled, and a REAL-7 set should add idiosyncratic assignments.

## URLs fetched by the subagent

arxiv.org/abs/2508.07662; github.com/knowledgator/gliclass; huggingface.co/knowledgator/gliclass-modern-base-v3.0;
arxiv.org/abs/2508.05425; arxiv.org/abs/2606.08051; pub.sakana.ai/doc-to-lora; arxiv.org/abs/2502.11830 (abs and html);
arxiv.org/abs/2603.11991; huggingface.co/datasets/DEVILHADYOURMOM/us-bank-transaction-categories; arxiv.org/abs/2303.13824;
arxiv.org/abs/2511.21038; arxiv.org/abs/2305.02239; arxiv.org/abs/2309.17249; arxiv.org/abs/2510.16282;
arxiv.org/abs/2205.11409; arxiv.org/abs/2605.08295; arxiv.org/abs/2601.05271; github.com/IBM/fastfit.
Search results only (not fetched): 2305.14128, 2402.04401, 2402.13598, 2506.06105, 2602.15902, 2506.05176, 2508.10021,
2511.08939, 2508.03555, 2406.17633, 2504.15432, 2501.10868, 2605.02363, 2305.08298.
