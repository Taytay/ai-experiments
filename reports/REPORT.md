# Fine-tuning LLMs and embedding models for merchant knowledge: frameworks, vocabulary, and knowledge injection

Date: 2026-09-12 to 2026-09-14. Hardware: RTX 3090 (24 GB), driver 591.86, Windows 11, torch 2.11 + cu128.
Sections 4 and 6 ran with transformers + torch only (Smart App Control blocked triton at the time; see NOTES.md); sections 7 and 8 use unsloth.
**Correction (2026-09-21, section 44).** unsloth's `FastLanguageModel.from_pretrained` defaults to `load_in_4bit=True`, and six scripts never overrode it: `exp_categoriser.py` and `exp_real6.py` (sections 37, 38, 43: every categoriser is QLoRA on the NF4 4-bit base and every number in those sections, the instruct base's included, is read on that base, consistently), `exp_items_v2.py` (the ARC / MMLU / induction re-reads of section 33 and Table 38.3), `exp_lre.py` (section 40), `exp_graph4.py` (section 42) and `exp_onpolicy_distill.py` (section 31: the teacher is the 4-bit base, perplexity 11.12 against the bf16 base's 10.61, and the student was trained on it). `exp_curriculum.py` always set the flag, so the arms of sections 8 to 34 are bf16 LoRA on the bf16 base as stated; the four back-fill and OPD scripts read those bf16 adapters on the 4-bit base, a mismatch whose size section 44 measures. Every script now sets the precision explicitly (`LOAD_4BIT`).
**Correction (2026-09-22, section 48).** The REAL-6 cells of sections 37 to 47 do not mean what their names say. A fifth of the "seen merchant" items (115 of 559) have no row of their merchant left in the user's 300-row history (the builder chose the test rows before the cut); every user files a merchant under one label, so the items whose merchant is in the history are answered by a lookup (the nearest history row's label, 98.6); and of the rest, 509 are decided by the merchant's standard category and 103 fall in categories the user split with nothing to say which side, where the record-in-prompt categorisers sit at the set's ceiling (97 and a coin flip). The intervals resample items over 20 users; resampled by user they are about twice as wide. Section 48 re-reads every REAL-6 run in the corrected groups; the conclusions that change are listed there.

Supporting docs: [frameworks.md](frameworks.md), [lit_review.md](lit_review.md). Code: `../scripts/`. Raw numbers: `../results/`. Every run is tracked with config + git commit in `../evals/` (see `../evals/LEADERBOARD.md`).

## 1. Executive summary

**Where the project stands (2026-09-26, sections 37 to 64).** The owner's aim is to understand the science of a categoriser that files a user's bank transactions under that user's own categories: Q1 how LLMs and encoders work as multiple-choice categorisers, Q2 the best way to inject knowledge such as a merchant or POI database, Q3 whether they can infer what a meaningless category name ("Yurra") means from examples seen in training and in the prompt. Product use only weights the metrics: auto-file the extremely confident, suggest the rest (section 59's scorecard: top-1, top-3, calibrated bits, auto-file coverage, skill over a no-model baseline). What is established, on a Qwen2.5-3B-Instruct categoriser with a rank-64 LoRA:

- **Evaluation first.** On REAL-6 (20 synthetic users) the seen merchants are a lookup and the unseen ones are decided by the merchant's standard category (section 48); a system with no model (the user's own label for the merchant, else other users', else the user's most-used category) reads 87% top-1 on held-out users, level with the best categorisers, whose value there is the ranking (top-3 98 against 26) and whatever no lookup reaches (section 59). Numbers from sections 37 to 47 should be read in section 48's corrected groups.
- **Q1.** Score options by summed log-probability and fit one temperature per model on other users; raw confidences are miscalibrated and a temperature does not transfer between models (section 50). Training time goes into re-reading the prompt, not precision: bf16 and the 4-bit base give the same model (52), 16 sequences per step at 1e-4 is the efficient batch (56), and putting the loss on every example label in the prompt (the "all-label" loss) trains in 100 steps what 1,600 plain steps did (52). The same run on the 3090 and on Modal's H100 agrees within run-to-run noise, six times faster there (54).
- **Q2.** A database written into the weights as prose sentences reaches the merchants only it knows at 59 to 69% (sections 43, 45); the same database taught as supervised decisions ("database episodes": synthetic statement rows of its merchants labelled in a training user's scheme) reaches 94%, opaque names included, with no record in the prompt (53), ties the record in the prompt at equal information (55), holds 20,000 merchants as well as 240 when each gets about 30 training rows (58), and helps on real businesses outside the database (+4 to 5, section 57). The record in the prompt remains the robust route: 85% on real Overture businesses from a one-line category record, whatever the statement string looks like (57).
- **Q3.** Categorisers memorise training users' coined names (10 to 25 points that do not transfer to new users) and rename augmentation fixes most of it (+12 to 14, section 51); on held-out users the record-in-prompt categoriser then reads 87%. From examples in the prompt, a blind Opus 5.5 reader resolves real businesses' coined categories at 88% (clean names) where the 3B model manages 55 to 65 (57): the information is there and the small model under-uses it; row 64's label-induction set, whose first version turned out to be solvable by elimination, is taking that apart.
- **Real data.** Overture's places (81.5M POIs, per-row licences) are downloaded with provenance (`data/external/`). Two sets are
  built from them. The novel-merchant set (57): truncated bank-statement strings cost every model without a record about 15 points.
  POI-1 (61): 200 synthetic users over real places, 12 to 20 categories each. POI-1 is hard for every reader: blind Opus 5.5 56%,
  the best 3B 59, 14B 60 (64). A lookup by the place's kind answers the kinds a user has filed (exact), and lookup-then-model reads
  80. Choosing the prompt's examples by kind lifts every model 8 to 18 points without training (68.6 for the 3B, 81% of the
  seen-kind ceiling).
- **Q3 in the prompt (60).** One or two examples of other businesses filed under a coined word teach every model what it means.
  On v2, where elimination cannot solve it, the 14B goes from 12 to 66% with one example, and a fine-tuned 3B reads like an
  untrained 14B. Opaque example names teach nothing. The models copy the nearest example: one same-kind business filed elsewhere
  costs 15 to 19 points, where Opus keeps 97.
- **Encoders (62, 63).** Trained across users with one scored position per category (Laya's `[MASK]` layout or GLiClass-large; the
  family does not matter once training matches), a 400M encoder:
  - matches the 3B on POI-1 (57 to 58 against 55 to 59) at 6 to 10 ms per item batched, about 40 times faster;
  - trails it on REAL-6 by 15 points without the record and 8 with it, for want of merchant knowledge;
  - with the record, GLiClass has the steadiest confidence of the encoders (AURC 0.062).

  Untrained, every encoder is at chance. ModernBERT-Instruct's single `[MASK]` over letter IDs works with the record (73.5) but
  learns slowly without it (42 at 5,000 steps), and letter IDs show position bias.
- **Best approach and ceilings (64).** Results are read as a share of each set's maximum achievable score (`ai_experiments.ceiling`):
  - REAL-6's ceiling is 94.0 (the split categories are a coin flip); the best decoders sit at 95 to 96% of it at 3B, 7B and 14B
    alike;
  - on POI-1, scale adds a point where retrieval adds nine;
  - the leading approach is a lookup (merchant, then kind) in front of a fine-tuned 3B with database episodes and kind-retrieved
    examples.

  Open limits and their hypotheses are in 64.2. Hop limits of encoders and decoders, with looped encoders and diffusion LMs, are
  PLAN row 71.

- **Frameworks.** For one 24 GB GPU, Unsloth is the consensus choice (fastest, lowest VRAM, native Windows, now covers embedding models via `FastSentenceTransformer`). Axolotl is the pick for multi-GPU nodes with YAML-driven reproducibility. TRL is the substrate both wrap and the right layer if you need a custom loss. torchtune is unmaintained since July 2025; do not start on it. For RL at scale, verl. For embedding models specifically, the trainer is sentence-transformers' `SentenceTransformerTrainer` whether or not Unsloth is wrapping it.
- **New vocabulary.** Almost never worth it for merchant names. Subword tokenization already handles them; expansion requires continued pretraining and can hurt at small token budgets. If you must add tokens, initialize inside the existing embedding distribution (mean-of-subwords or Hewitt's N(mu, Sigma) sampling), never random, and train afterwards. Our experiments went further: added merchant tokens actively **hurt** both models. For the embedding model they collapsed transfer to unseen bank-statement strings from 70.8% to 23-26% (section 4.3.1); for the LLM they cut knowledge extraction from 46.7% to 31.7% at identical data and steps. Post-hoc aliasing of an uppercase token onto the trained mixed-case embedding did not rescue the bank format either. Fix the strings with normalization, not the tokenizer.
- **User-invented labels and analogy (section 6).** Prompts like "Timmy labeled his Blaxorc 'FooFoo'... how will he label his Radsup?" are a scale phenomenon: with the facts in context a 3B model reaches 49 to 52% on a 3-way task (chance 33) and 0.5B never leaves chance. Knowledge injected into weights by LoRA was fully recalled (100% in the trained format) but did not power that in-context analogy, and fine-tuning eroded the ability. For an embedding model, defining a user's label as the centroid of their labeled examples gave 85% (type) and 63% (a latent attribute the labels never name) with three examples and no retraining. Recommendation: prototypes for user categories, retrieval-in-context for LLM analogy, a 3B to 8B model.
- **Teach the task as the injection (section 8).** A six-arm sweep on Qwen2.5-3B settled the inject-then-task question. Symbol-tuning episodes generated from the database (few-shot items with fresh random labels and a varied grouping attribute) raise in-context label induction by about 30 points on every test, including a partition and species never trained on, and lift few-shot classification with random labels on public datasets from 60 to 76 where declarative text alone lowered it to 51. Interleaving those episodes with the knowledge text in one run gives 100% recall *and* the Timmy task from the weights at 59% (base 36, knowledge-only 40, chance 33), with transfer to the weakness attribute (54%), which training episodes never group by [correction, section 33: weakness is a bijection of type in this universe, so this level measured the type rule again, not a transfer]; weakness is a fixed function of type in this universe, though, so this is a type replicate and not evidence of latent-partition transfer (see 6.2 and DATA-1). Sequential staging matches the induction number but loses 3 points of recall, 21 of yes/no manipulation and 9 of pairwise reasoning. A 15% generic replay slice keeps the skill portable (78 vs 71 on public data) and holds perplexity at 15 instead of 20. On a universe where 70% of names carry a type suffix, the interleaved model types never-seen names from their suffix at 94% (chance 12.5) while neutral names stay at chance: the drug-stem mechanism, measured. *[Later: the yes/no and pair losses are inside the seed spread (section 20: -9.2 +- 15.2, -10.0 +- 20.4); the recall loss came from the learning-rate schedule, and 10% knowledge replay in the second phase makes staging equal interleaving (section 23). The '15 instead of 20' is the one-paragraph perplexity that section 16 retired, and the natural-label half of '78 vs 71' did not survive the paired test (section 11).]*
- **Performance (section 7).** On this RTX 3090, unsloth LoRA trains 0.5B at 17.7k tokens/s (2.2x plain transformers, 3x less memory), 3B at 2.7k tokens/s (69% MFU) and 7B QLoRA at 1.3k tokens/s. Small models on short facts are overhead-bound (13 to 17% MFU) and want sequence packing more than a faster GPU; 3B and up are compute-bound and scale with TFLOPS. The card handles ~1.5B for full fine-tuning, ~8B for LoRA, ~30B for QLoRA. A 6-attribute entity costs ~1,800 training tokens with a diverse recipe, so a 10k-entity database trains in 17 minutes on 0.5B or about 2 to 4 hours on 3B to 7B. Watch for Windows WDDM system-memory fallback near 24 GB, which slows training ~3x instead of failing. *[Later: that counts recall in the trained format only; exposure, not token count, limits what a larger database holds (section 29: 5,000 species read 14 to 22 at the step counts that give 1,000 species 51 to 100), and fact-DB records need about 6.7 passes before merchants no user labelled gain from them (section 45).]*
- **Knowledge injection.** Dumping the merchant database as one sentence per store into the model does not produce usable knowledge, even when memorized. Paraphrase and QA augmentation of the same facts (the Physics-of-LMs / EntiGraph recipe) is what makes knowledge extractable in new task formats. At a sane learning rate (1e-5 here) augmented full fine-tuning doubled category-inference accuracy over the raw dump (41.7% vs 21.7%, retrieval ceiling 69%) with little forgetting; at 5x that rate it destroyed general ability (perplexity 16 to 800) and LoRA or WiSE-FT weight averaging were the rescue. Raw statement strings defeated every method including retrieval at this model size, so normalize merchant strings before the model sees them. Retrieval (fact in context) remains the strongest and cheapest baseline; the right production design is RAG over the merchant DB plus augmented fine-tuning for the head of the distribution, with RAFT-style training so the model uses retrieved records well. *[Later: for the categoriser the record in the prompt beats the record in the weights on the merchants no user labelled (97.5 against 59 +- 12 at 3.3 passes and 69 +- 6 at 6.7; sections 38, 43, 45), and RAFT added 3 to 7 points in one seed (section 25). The first bullet of this section is the current summary.]*

## 2. Fine-tuning frameworks (September 2026)

See [frameworks.md](frameworks.md) for the full table with citations. Short version:

| Need | Use |
|---|---|
| 1 GPU, LoRA/QLoRA/full FT, SFT/DPO/GRPO, Windows | **Unsloth** |
| GUI, widest model zoo | LLaMA-Factory |
| Multi-GPU node, reproducible configs, long context | **Axolotl** (FSDP2 + sequence parallel) |
| Custom loss / new algorithm | **TRL** directly |
| RL on 70B+ with agentic rollouts | verl (OpenRLHF, SkyRL as alternatives) |
| Embedding / encoder models | sentence-transformers `SentenceTransformerTrainer`, optionally via Unsloth `FastSentenceTransformer` |
| No infra, LoRA only, research loops | Tinker (managed) |
| Do not use | torchtune (unmaintained), OpenAI fine-tuning (closed to new users) |

On this machine: Unsloth 2026.9.4 and sentence-transformers are installed in the uv project; both import `datasets` (pyarrow) and Unsloth imports triton, so they run only once Smart App Control is off or under WSL2.

## 3. Introducing new vocabulary

**Mechanics.** `tokenizer.add_tokens([...])`, `model.resize_token_embeddings(len(tokenizer))`, initialize the new rows, then train. The tokenizer matches added tokens as literal strings before subword splitting, so `Kelvarro Co` becomes one token but `KELVARRO CO` (a bank statement) does not unless you add that form too.

**Initialization matters more than the method.** Random init puts a new token far from the embedding manifold; Hewitt (2021) shows it then attracts probability mass regardless of context. Mean-of-subword-embeddings or sampling from the empirical N(mu, Sigma) of existing rows keeps the pre-expansion model's behaviour nearly intact. Mundra et al. (2024) found the simple multivariate init matches fancy methods (FOCUS, WECHSEL, ZeTT); what matters is staying inside the convex hull of existing embeddings.

**When it helps.** When the base tokenizer is grossly inefficient for the domain (Chinese in a Latin-centric vocab: 3-4 tokens per character) and you have a large continued-pretraining budget (Dagan et al.: payoff above roughly 50B tokens). Zhao et al. (2024) found extension *hurt* at tens of billions of tokens.

**When it does not.** Merchant strings, product names, category labels. They tokenize fine as subwords; the model's problem is not spelling them but knowing what they mean, and that is a data problem (section 4), not a vocabulary problem.

**Embedding models.** Same mechanics on the underlying transformer; the pooling layer is unaffected. Contrastive fine-tuning after adding tokens is mandatory because the new rows start with no meaning. Our test (4.3.1) shows it is strictly worse than subwords for MiniLM: a single fresh token is diluted by mean pooling and brittle in noisy context, while subword pieces are shared between clean and statement forms and carry redundant signal.

**Cheap tricks that do work.**
- *Text normalization* before tokenizing: title-case the merchant, strip store numbers, expand `SQ *`, `TST*`, `POS DEBIT`. This closes much of the mixed-case vs uppercase gap for free.
- *Token aliasing* (add the uppercase/statement form as a token and copy the trained mixed-case embedding into it) is a plausible idea that **did not work** in our LLM test (4.3.2): the surrounding store-number and city noise still defeated the model. Normalization is the better tool.
- *Special marker tokens* for structure (`<merchant>`, `<amount>`, `<category>`), a handful of tokens that are genuinely new symbols. This is the legitimate use of vocabulary expansion.

## 4. Injecting merchant knowledge so it transfers

### 4.1 What the literature says

- *Knowledge must be stored in extractable form.* Allen-Zhu & Li (Physics of LMs 3.1): a model trained on one biography per entity memorizes it but achieves near 0% QA extraction, no matter how it is instruction-tuned afterwards. Paraphrase, sentence-shuffle, and QA augmentation during (continued) pretraining fixes this.
- *Reversal curse* (Berglund et al.): "A sells B" does not teach "B is sold by A". Include both directions.
- *RAG usually wins* (Ovadia et al.; Soudani et al.), especially on long-tail entities, which merchants are. Fine-tuning on unknown facts also raises hallucination rates (Gekhman et al.).
- *Synthetic continued pretraining* (EntiGraph, Yang et al. 2024; Instruction Pre-Training; Active Reading, Meta 2025) is the current recipe for making fine-tuning competitive: generate diverse texts connecting entities, mix in instruction-style data, train at low LR, and it compounds with RAG.
- *Full FT vs LoRA* (Biderman et al. 2024): LoRA learns less and forgets less. Use full FT (or high-rank LoRA on all modules) for new knowledge, replay or WiSE-FT weight averaging to limit forgetting.
- *RAFT* (Zhang et al. 2024): fine-tune the model to answer with retrieved merchant records in context, including distractors, so RAG and parametric knowledge reinforce each other.

### 4.2 Experiment design

Synthetic database of 120 fictional merchants (opaque names like *Elrholm*, *Oskhurst Bros*), 10 per category across 12 spending categories, each selling 3 products drawn from a category pool. Names are fictional so the base model has zero prior knowledge; the category label is **never** present in training text, only what the store sells. Held-out evaluation formats, none seen in training:

| Task | Format | Chance |
|---|---|---|
| clean_category | `Merchant: Elrholm` -> category (12-way, 3-shot) | 8.3% |
| bank_category | `Transaction: DEBIT CARD PURCHASE ELRHOLM STORE 4970 TUCSON AZ` -> category | 8.3% |
| sells | "What does Elrholm sell?" 4-way, correct option is a subset/order not in training | 25% |
| reverse | "Which store sells canned goods and frozen vegetables?" 4-way over merchant names | 25% |

Scoring is by mean log-probability of each option given the prompt (multiple-choice likelihood, no generation parsing). Forgetting is tracked as perplexity on a neutral English passage.

LLM: Qwen2.5-0.5B base, 420 optimizer steps at batch 16 for every trained condition (identical compute), AdamW, warmup + linear decay, bf16 autocast, fp32 master weights.

Embedding model: all-MiniLM-L6-v2, in-batch-negative contrastive loss, anchor = sentence about the merchant, positive = category description; 24 merchants (2 per category) held out entirely to probe generalization.

### 4.3 Results

#### 4.3.1 Embedding model (all-MiniLM-L6-v2): does adding merchant tokens help?

12-way category accuracy by nearest category description. `bank_*` strings (uppercase, store number, city) never appear in training. Held-out merchants (24) have no training text at all. Source: `results/embed_vocab.json`.

| condition | name (train) | **bank (train)** | description (train) | name (held-out) | bank (held-out) | description (held-out) |
|---|---|---|---|---|---|---|
| zero_shot | 6.2 | 7.3 | 99.0 | 29.2 | 4.2 | 95.8 |
| ft_subword (original tokenizer) | 100.0 | **70.8** | 100.0 | 12.5 | 4.2 | 100.0 |
| ft_newtok_random (120 tokens, random init) | 100.0 | 22.9 | 100.0 | 8.3 | 8.3 | 100.0 |
| ft_newtok_mean (120 tokens, mean-of-subword init) | 100.0 | 26.0 | 100.0 | 12.5 | 4.2 | 100.0 |

Reading it:

- The untouched model already maps *descriptions* of what a store sells to the right category 99% of the time. The knowledge gap is purely the merchant name.
- Contrastive fine-tuning with the **original subword tokenizer** learns every trained merchant (100%) and, crucially, **transfers to the never-seen bank-statement format at 70.8%**. The uncased WordPiece pieces (`el ##rh ##ol ##m`) are identical in "Elrholm is a store..." and "DEBIT CARD PURCHASE ELRHOLM STORE 4970", so the learned association rides along for free.
- **Adding merchant names as new tokens destroys that transfer** (22.9 to 26.0%) even though the new token does match inside 105 of 120 bank strings. Initialization barely matters here; the problem is structural.
- Held-out merchants stay at chance for every condition. Nothing about *Bexstead Ltd* tells you what it sells; there is no free generalization to unseen names. Only retrieval or explicit training covers them.

Diagnostic (`results/diag_embed_dilution.log`): why do new tokens hurt?

| probe | subword model | new-token model |
|---|---|---|
| bank string as-is | 70.8 | 22.9 |
| bank string with the name repeated 4x | 86.5 | 43.8 |
| bare name | 100.0 | 100.0 |
| name + " store 4970 tucson az" | 93.8 | 37.5 |
| bank strings where the new token matched / did not | n/a | 23.5 / 20.0 |

Two effects: (1) **mean-pooling dilution**: with a single identity token the merchant is 1 of ~13 pooled positions instead of 4 of ~13, so repeating the name partially recovers accuracy; (2) **brittleness of a freshly learned token**: adding just five neutral words drops the new-token model to 37.5% while the subword model holds at 93.8%. A multi-piece span gives the encoder redundant, pretrained pieces to attend over; a single new row has 6 epochs of history and no pretrained neighbors. This matches the literature's warning that vocabulary expansion needs real continued pretraining to pay off, and for merchant names it simply has nothing to offer.

#### 4.3.2 LLM (Qwen2.5-0.5B): can injected merchant knowledge transfer to new task formats?

Accuracy %, 420 optimizer steps for every trained condition. Chance: 8.3 (12-way), 25 (4-way). `ppl_general` is perplexity on neutral English (16.15 = untouched). Source: `results/knowledge_injection.json`.

| condition | clean_category | bank_category | sells | reverse | ppl_general |
|---|---|---|---|---|---|
| base (zero-shot) | 8.3 | 8.3 | 20.0 | 31.7 | 16.15 |
| **incontext** (oracle context: the exact fact in the prompt, an upper bound for retrieval) | **69.2** | 14.2 | **100.0** | **100.0** | 16.15 |
| ft_raw (1 sentence/merchant, full FT, lr 5e-5) | 15.0 | 10.0 | 45.8 | 31.7 | 420.9 |
| ft_aug (14 paraphrases+QA/merchant, full FT, lr 5e-5) | 9.2 | 8.3 | 46.7 | 46.7 | 795.6 |
| ft_aug + WiSE-FT 0.5 (average with base) | 25.0 | 8.3 | 56.7 | 47.5 | 28.1 |
| **ft_aug_lora** (r=64 all linear, lr 3e-4) | 17.5 | 12.5 | **60.8** | 51.7 | 202.0 |
| ft_aug_vocab (120 name tokens, mean init, full FT) | 10.0 | 8.3 | 31.7 | 79.2* | 885.8 |
| ft_aug_vocab + uppercase alias tokens | | 8.3 | | | |

\* Single-token merchant names make the `reverse` option scoring trivially favorable (one token to score); treat this cell as an artifact, not knowledge.

Lower-learning-rate rerun (full FT was over-trained at 5e-5; source `results/lr_sweep.json`):

| condition (full FT, 420 steps) | clean_category | bank_category | sells | reverse | ppl_general |
|---|---|---|---|---|---|
| ft_raw, lr 1e-5 | 21.7 | 8.3 | 37.5 | 25.8 | 19.2 |
| **ft_aug, lr 1e-5** | **41.7** | 12.5 | 51.7 | 42.5 | 26.0 |
| ft_aug, lr 1e-5 + WiSE-FT 0.5 | 26.7 | 9.2 | 52.5 | 30.8 | 17.2 |
| ft_raw, lr 2e-5 | 36.7 | 8.3 | 45.0 | 30.0 | 27.5 |
| ft_aug, lr 2e-5 | 20.8 | 13.3 | 50.8 | 55.8 | 51.4 |
| **ft_aug, lr 2e-5 + WiSE-FT 0.5** | **40.8** | 8.3 | **68.3** | 38.3 | 19.1 |

At a sane learning rate the story is clean. With identical data exposure (~56 passes per merchant) and identical steps, **augmented text beats the raw database dump on every knowledge task**: category inference 41.7 vs 21.7, sells 51.7 vs 37.5, reverse 42.5 vs 25.8 (chance 25). The raw model at lr 2e-5 does learn name-to-products (36.7 on category) but is at chance on reverse lookup: it memorized one direction only, the reversal curse in miniature. The best trained model recovers 60% of the retrieval ceiling (41.7 of 69.2) with perplexity drift of 16 to 26, and WiSE-FT trades a little category accuracy for near-baseline perplexity when the run was hotter (lr 2e-5).

Reading it:

- **Retrieval dominates.** Putting the one-sentence fact in the prompt gives 100% on both direct-knowledge tasks and 69% on category inference, with zero training and zero forgetting. That 69% is also the *ceiling* for any parametric method on this model: it measures how well a 0.5B model can bridge "sells canned goods and frozen vegetables" to "Groceries" at all.
- **Raw facts memorize but do not transfer.** ft_raw reaches near-zero training loss (each sentence seen ~56x) yet stays at chance on `reverse` and near chance on category tasks. This is the Physics-of-LMs result reproduced at toy scale.
- **Augmentation is what makes knowledge usable in a new shape.** In the over-trained lr 5e-5 runs the effect shows only on `reverse` (31.7 to 46.7 to 51.7 with LoRA) and `sells` (60.8 with LoRA). In the lr 1e-5 rerun below it shows everywhere: category inference doubles (21.7 to 41.7) and reverse lookup goes from chance to 42.5.
- **Learning rate is the difference between injecting knowledge and lobotomizing the model.** At lr 5e-5 perplexity on ordinary English went 16 to 400 to 900. The model that "knows" the merchants can no longer do the reasoning step (products to category). Two mitigations both worked: **LoRA** (ppl 202, best `sells`) and **WiSE-FT weight averaging** (ppl 28, best `clean_category` among trained models at 25%). LoRA "learns less, forgets less" shows up exactly as Biderman et al. describe; with the base model this damaged, forgetting less *was* learning more. At lr 1e-5 full FT no longer needs the rescue.
- **The bank-statement format defeats everything, including retrieval.** All conditions sit at chance on `bank_category`, and even in-context facts only reach 14%. A 0.5B model cannot align `ELRHOLM STORE 4970 TUCSON AZ` with `Elrholm` in the same prompt. Copying the trained embedding into an uppercase alias token did not rescue it either (the surrounding model was already wrecked, and the store-number/city noise remains). **Conclusion: normalize statement strings to canonical merchant names before the model ever sees them.** This is a preprocessing problem, not a fine-tuning problem.
- **Vocabulary expansion hurt the LLM too**: `sells` fell from 46.7 to 31.7 with the same data and steps, and the knowledge that was learned attached to the single token in a way that only the reverse-lookup scoring rewarded.

#### 4.3.3 What the experiments say about "improving other tasks"

The knowledge that transferred was the knowledge stored in **many surface forms**. With a real merchant DB the practical version of this is: generate paraphrases, both-direction QA, statement-style mentions, and product-to-category chains (EntiGraph/Active-Reading style), mix in replay data, train with LoRA or full FT at a low learning rate, and average with the base checkpoint. Even then, expect parametric recall to lag retrieval, especially on long-tail merchants, so the model should be trained to *use* retrieved merchant records (RAFT) rather than to replace them. The place where fine-tuning unambiguously wins is the embedding model: it learned every merchant and generalized to the statement format with the original tokenizer.


## 5. Recommended recipe for the personal-finance use case

1. **Normalize first.** Canonicalize statement strings (case, store numbers, processor prefixes) before anything touches a model. Cheapest win available.
2. **Retrieval is the backbone.** Embed the merchant DB (name, aliases, what they sell, category) and retrieve top-k records into the classifier's context. Long-tail merchants will never be reliably parametric.
3. **Fine-tune the embedding model** on (statement string, merchant record) and (merchant record, category) pairs with hard negatives so retrieval itself is good. Use sentence-transformers' trainer; Unsloth for speed once triton loads. Do not add vocabulary.
4. **For parametric knowledge that transfers**, build an augmented corpus: 10+ paraphrases per merchant, QA in both directions, statement-style mentions, product-to-category chains. Full fine-tune (or high-rank LoRA on all modules) at low LR with replay of general data; average with the base checkpoint (WiSE-FT) if perplexity drift matters. *[Later: three renderings saturate recall and manipulation comes only from negative and comparative texts (section 27); rank 256 buys nothing, MLP-only LoRA matches all-linear, and full fine-tuning found no working rate on 3B (section 28); records in the prompt beat records in the weights for the categoriser (sections 38 to 45).]*
5. **Train the model to use retrieval** (RAFT): fine-tune on prompts containing retrieved records plus distractors, targets that cite the record. This is what makes knowledge improve *other* tasks (budget Q&A, anomaly explanations, merchant normalization) rather than just the classifier. *[Later: measured in section 25, RAFT added 3 to 7 points on the retrieved condition in one seed; the categoriser trained with the record in its prompt (section 38) is the version of this that matters.]*
6. **Evaluate on held-out merchants and formats**, and track hallucination on merchants the model was never taught.

## 6. Fictional universe: user-invented labels and analogy over injected knowledge

Added 2026-09-13. Question: teach a model a completely custom taxonomy (a Pokemon-style creature universe with fictional type names), then answer prompts like *"Timmy labeled his Blaxorc 'FooFoo' and his FrodRock 'blammo'. How will he label his Radsup?"* where both the entities and the labels are novel. Code: `src/ai_experiments/universe.py`, `exp_universe_ladder.py`, `exp_universe_embed.py`. Tracked runs: `evals/LEADERBOARD.md`, experiments `universe_ladder` and `universe_embed`.

### 6.1 Setup

- **Universe.** 160 species with opaque names (*Elrholm*, *Oskhurst*), 8 fictional types (*Voltrix*, *Pyrrhan*, ...) each with a lore sentence, a weakness type (a fixed derangement of type, so weakness and type are the same partition), 6 habitats, 5 diets, 5 regions. 24 species (3 per type) are held out of all training. Each type also has three never-trained synonyms ("the sparky ones", "shock-type").
- **Training text.** 2,752 texts: 14 descriptive/QA templates per species, one negative ("Is X a Y-type? No..."), four comparative statements (shared / different type, shared habitat), plus type lore. Unlike the merchant experiment the type label is present in training text: the point here is analogy over known attributes, not inference of hidden ones.
- **Ladder.** 1,488 multiple-choice items scored by option likelihood, none in a training format:

| level | what it tests | chance |
|---|---|---|
| L1 recall | type / weakness / habitat of a species | 12.5 / 12.5 / 16.7 |
| L2 manipulation | yes/no "Is X a T-type?", "Do X and Y share a type?" | 50 |
| L3 induction (type) | Timmy prompt, 3 nonsense labels; controls: real type names as labels, k=2, k=4 | 33 (k=3) |
| L4 induction (latent) | Timmy prompt where the labels track weakness or habitat, not type | 33 |
| L5 novel choices | "Which group does X belong to?" with synonym options | 12.5 |
| L6 unseen species | recall for held-out species + confidence margin vs seen control | 12.5 |
| L7 regression | perplexity on neutral English | |

Conditions: base, base + field-guide entries in context (oracle context: the exact entry is supplied verbatim, so this bounds what retrieval could deliver and is not a retrieval result; end-to-end retrieval is REAL-2), LoRA r=64 on all linear layers (600 steps, batch 16, bf16), LoRA + context.

### 6.2 LLM results (lr 2e-4)

| level | 0.5B base | 0.5B +ctx | 0.5B LoRA | 0.5B LoRA+ctx | 3B base | 3B +ctx | 3B LoRA | 3B LoRA+ctx |
|---|---|---|---|---|---|---|---|---|
| L1 recall | 18.8 | 76.2 | 20.0 | 21.9 | 18.1 | **98.8** | 24.4 | 33.8 |
| L2 is-a (y/n) | 43.8 | 60.0 | 53.8 | 55.0 | 43.8 | 100.0 | **92.5** | 96.2 |
| L2 pairwise (y/n) | 45.0 | 43.8 | 57.5 | 53.8 | 45.0 | 95.0 | **87.5** | 88.8 |
| L3 induct, nonsense k=3 | 33.8 | 35.6 | 22.5 | 27.5 | 37.5 | **51.9** | 33.8 | 30.6 |
| L3 induct, real names | 37.5 | 54.4 | 46.2 | 51.2 | 40.6 | **68.1** | 48.1 | 45.6 |
| L3 induct, k=2 (chance 50) | 45.6 | 47.5 | 51.9 | 53.8 | 53.1 | 61.2 | 58.1 | 53.8 |
| L3 induct, k=4 (chance 25) | 22.5 | 21.9 | 25.6 | 23.8 | 27.5 | **42.5** | 28.1 | 26.2 |
| L4 induct, weakness | 38.8 | 40.0 | 40.0 | 38.1 | 37.5 | **54.4** | 44.4 | 42.5 |
| L4 induct, habitat | 30.0 | 29.4 | 33.8 | 33.1 | 31.9 | 38.8 | 31.2 | 32.5 |
| L5 novel choices | 15.6 | 20.0 | 22.5 | 20.0 | 17.5 | 24.4 | 23.8 | 25.6 |
| L6 unseen recall | 12.5 | 100 | 12.5 | 12.5 | 16.7 | 100 | 12.5 | 20.8 |
| L7 perplexity | 16.2 | 16.2 | 244 | 244 | 8.6 | 8.6 | 31.8 | 31.8 |

Reading it:

- **The Timmy task is a scale phenomenon, exactly as Wei et al. (2023) predict.** With the facts in context, 0.5B stays at chance on nonsense-label induction (35.6) while 3B reaches 51.9 (k=3) and 42.5 (k=4, chance 25). Using the real type names as labels is easier for both (54 / 68), which is the "semantic prior" the small model leans on. Below a few billion parameters the model cannot override its priors with three examples, no matter how good its knowledge is.
- **Latent partitions are harder.** When Timmy's tags track habitat instead of type, 3B+ctx drops to 38.8. Weakness looks better (54.4) only because it coincides with type in this universe; treat it as a type replicate, not a latent-partition result. Inferring *which* attribute a user's labels follow is the frontier task here.
- **LoRA injected the knowledge but in a format-bound way.** 3B LoRA scores 92.5 / 87.5 on the yes/no manipulation formats, which existed in training, yet 24.4 on the bare-type recall prompt, whose answer shape ("Voltrix" rather than "X is a Voltrix-type.") did not. The format-matched recall level added after this run (L1_recall_fmt, section 6.4) isolates that.
- **Parametric knowledge did not power in-context analogy.** 3B LoRA on the Timmy prompt: 33.8, chance. The same knowledge supplied as text: 51.9. And LoRA+ctx (30.6) is *worse* than base+ctx, because training at lr 2e-4 damaged the model (perplexity 8.6 to 31.8; 0.5B went 16 to 244). Physics of LMs 3.2 in miniature: recall and manipulation are different skills, and forgetting can take the second one away while the first is being learned.
- **Confidence is not a hallucination detector at this scale.** Held-out species: LoRA models are at chance (12.5) with confidence margins (0.60) *higher* than on species they know (0.50). Unknown-entity abstention has to be engineered (retrieval hit / miss, verifier), not read off the logits.

### 6.3 Embedding-model results: prototypes make unseen labels free

all-MiniLM-L6-v2 trained contrastively so a species *name* embeds near the text of its attributes (type lore, weakness text, habitat text, field-guide entry). No head, no new tokens. Labels then take three forms. Source: `results/universe_embed.json`.

| label form | metric | zero-shot | trained |
|---|---|---|---|
| canonical type lore text | 8-way, seen species | 10.3 | **99.3** |
| synonym "the sparky ones" (never trained) | 8-way | 13.2 | **41.2** |
| synonym "shock-type" (never trained) | 8-way | 14.0 | 24.3 |
| **prototype: centroid of Timmy's k labeled cards** | type, k=1 / k=3 (3-way, chance 33) | 38.0 / 33.7 | 69.0 / **85.3** |
| prototype | weakness (= type partition), k=1 / k=3 | 38.0 / 33.3 | 79.0 / 85.0 |
| prototype | **habitat (latent partition)**, k=1 / k=3 | 32.7 / 25.7 | 45.0 / **63.0** |
| any form | held-out species | ~chance | ~chance |

This is the strongest result in the section. A user label defined by *examples* rather than a *name* is assigned correctly 85% of the time with three cards, and 63% when the user's grouping follows a partition the label never names (habitat). Synonym labels work partially because the lore text carries the semantics ("crackle when excited" is near "sparky"); a name-only label ("shock-type") mostly does not. Held-out species stay at chance in every form, which is the honest answer for entities the encoder never saw: there is nothing to embed.

### 6.4 Lower learning rate rerun (lr 1e-4) with format-matched recall

Both models retrained with LoRA at lr 1e-4 (half the first run), same 600 steps, plus a new level **L1_recall_fmt** whose options use the trained answer shape ("*X is a Voltrix-type.*"). The ladder's random items differ slightly from 6.2 because the new level shifts the seed sequence; base rows are the fair comparison within this table. Tracked as `universe_ladder` runs with `lr=0.0001`.

| level | 0.5B base | 0.5B +ctx | 0.5B LoRA | 0.5B LoRA+ctx | 3B base | 3B +ctx | 3B LoRA | 3B LoRA+ctx |
|---|---|---|---|---|---|---|---|---|
| L1 recall (bare type name) | 18.8 | 76.2 | 20.0 | 28.1 | 18.1 | 98.8 | 24.4 | 23.1 |
| **L1 recall, trained format** | 11.2 | 92.5 | **100.0** | 100.0 | 13.1 | 100.0 | **100.0** | 100.0 |
| L2 is-a (y/n) | 43.8 | 61.2 | 50.0 | 55.0 | 43.8 | 100.0 | **88.8** | 100.0 |
| L2 pairwise (y/n) | 50.0 | 47.5 | 43.8 | 47.5 | 51.2 | 80.0 | **85.0** | 88.8 |
| L3 induct, nonsense k=3 | 31.2 | 32.5 | 30.6 | 31.9 | 37.5 | **48.8** | 32.5 | 34.4 |
| L3 induct, real names | 38.1 | 54.4 | 42.5 | 41.9 | 38.8 | **68.8** | 41.9 | 43.1 |
| L3 induct, k=4 (chance 25) | 21.9 | 19.4 | 21.9 | 23.1 | 26.9 | **41.2** | 30.0 | 27.5 |
| L4 induct, habitat | 33.1 | 34.4 | 34.4 | 33.1 | 31.2 | 36.9 | 31.2 | 35.6 |
| L6 unseen recall / margin | 12.5 / .60 | 100 / .67 | 12.5 / .35 | 41.7 / .31 | 16.7 / .32 | 100 / .77 | 12.5 / .64 | 12.5 / .65 |
| L7 perplexity | 16.2 | 16.2 | 134 | 134 | 8.6 | 8.6 | **20.5** | 20.5 |

What changed and what did not:

- **Recall was never the problem.** In the trained answer format both LoRA models recall species type at 100%; the 20 to 24% on the bare-name prompt in 6.2 was a format mismatch, not missing knowledge. Report manipulation-format and trained-format accuracy, and treat a single prompt shape as a measurement of that shape.
- **Halving the learning rate halved the damage** (3B perplexity 31.8 to 20.5; 0.5B 244 to 134) with no loss of injected knowledge. It is still substantial forgetting for 600 steps on 2.7k short texts; replay data and a lower rate again (or WiSE-FT, section 4.3) are the next levers.
- **Parametric knowledge still does not drive the Timmy task.** 3B LoRA: 32.5, chance. The same facts in context: 48.8. And even at the gentler rate, LoRA+ctx (34.4) remains well below base+ctx (48.8): fine-tuning on flat declarative text eroded the model's few-shot label-mapping ability faster than it added anything usable for it. If the goal is analogy over a custom taxonomy, the knowledge belongs in the prompt and the training data (if any) should contain the analogy task itself.
- **0.5B is below the threshold for the task in every condition**, consistent with 6.2 and with Wei et al.

### 6.5 Takeaways for the personal-finance use case

1. **Define user labels by examples, not names.** The prototype approach (a label = the centroid of the transactions a user put under it) handles synonyms, typos, and idiosyncratic categories with zero retraining and already reaches 63 to 85% on 3-way tasks at MiniLM scale. Use a stronger encoder (bge, EmbeddingGemma, Qwen3-Embedding) trained on transaction-to-merchant-record pairs and this should be the production classifier for personal categories. *[Later: on realistic histories the category name's vector plus the centroid is the best classifier at every k, and the example prototype is a memory of the labelled merchants (85 to 90) that reads 14 to 18 on the rest (section 36).]*
2. **For LLM-side analogy prompts, keep the knowledge in context.** Retrieve the merchant records for the merchants named in the prompt; the model then does the mapping. Use a 3B+ model (7B preferable) since the ability to follow arbitrary label mappings from few examples is absent at 0.5B.
3. **Parametric injection is for coverage, not for reasoning, unless the task is part of the injection.** Fine-tune so the model recognizes merchants when retrieval misses, at a low learning rate with replay, and verify manipulation-format accuracy (yes/no, comparisons) rather than bare recall. Section 8 shows that mixing few-shot episodes into the same run changes this picture: the injected knowledge then does power in-weights analogy (59% vs 40% here).
4. **Inferring the user's grouping rule is the open problem.** Habitat-tracking labels were the hardest case for both model types. A practical fix is to compute prototypes over several attribute-specific embeddings (type, merchant, amount band, time) and pick the space in which the user's examples cluster most tightly.

## 7. Performance: throughput, memory, bottlenecks, scaling

Measured 2026-09-13 with `scripts/bench_throughput.py` (each config in its own process, 12 timed steps after 3 warm-up steps, synthetic token batches so tokenization is excluded). GPU: RTX 3090, 24 GB, driver 591.86, WDDM. Dense bf16 tensor-core peak assumed 71 TFLOPS for MFU; the 7B forward pass reached 60 TFLOPS, which confirms that peak is the right reference. Tracked as `bench_throughput`; raw numbers in `results/bench_throughput.json`.

### 7.1 Training throughput and memory on this card

LoRA = rank 64 on all seven projection matrices (120M trainable params on 3B, 161M on 7B). "gc" = gradient checkpointing. Full FT = fp32 master weights + AdamW, bf16 autocast.

| model | method | seq | batch | tokens/s | step ms | peak VRAM GiB | MFU |
|---|---|---|---|---|---|---|---|
| 0.5B | full FT | 64 | 16 | 4,113 | 249 | 10.1 | 17% |
| 0.5B | full FT | 512 | 4 | 7,029 | 291 | 14.8 | 29% |
| 0.5B | LoRA | 64 | 16 | 4,724 | 217 | 5.3 | 13% |
| 0.5B | LoRA | 512 | 8 | 8,165 | 502 | 19.2 | 23% |
| **0.5B** | **unsloth LoRA** | 512 | 8 | **17,716** | 231 | **6.0** | **49%** |
| 3B | LoRA + gc | 64 | 16 | 1,514 | 677 | 9.1 | 40% |
| 3B | LoRA | 64 | 16 | 2,208 | 464 | 16.1 | 38% |
| 3B | LoRA + gc | 512 | 8 | 1,662 | 2,465 | 14.8 | 43% |
| **3B** | **unsloth LoRA + gc** | 512 | 8 | **2,663** | 1,538 | **8.1** | **69%** |
| 3B | full FT + gc | 64 | 8 | OOM | | needs 33.7 | |
| 7B | LoRA + gc | 64 | 8 | 835 | 613 | 17.3 | 54% |
| 7B | LoRA + gc | 512 | 4 | 940 | 2,178 | 19.9 | 61% |
| 7B | QLoRA 4-bit + gc | 512 | 4 | 806 | 2,541 | 13.3 | 30% |
| **7B** | **unsloth QLoRA + gc** | 512 | 4 | **1,279** | 1,601 | **10.3** | 54% |
| MiniLM-L6 | contrastive full FT | 32 | 32 | 36,763 (575 pairs/s) | 56 | 0.5 | |

Forward-only (likelihood scoring / evaluation), bf16, seq 128: 0.5B 34,600 tok/s (1.5 GiB); 3B 7,970 tok/s (6.5 GiB, 69% MFU); 7B 3,980 tok/s (14.6 GiB, 85% MFU).

**Unsloth versus plain transformers + peft, like for like:** 2.2x faster and 3.2x less memory on 0.5B at 512 tokens, 1.6x faster and 1.8x less memory on 3B, 1.6x faster and 1.3x less memory on 7B QLoRA. The memory win comes mostly from the fused cross-entropy that never materializes the batch x seq x 152k-vocabulary fp32 logits; the speed win from fused RMSNorm/RoPE/MLP kernels and less Python per step. These are the numbers to plan around; the plain-transformers loop used in sections 4 and 6 was a Smart-App-Control workaround.

**Like-for-like on the real workload.** The 3B universe ladder (section 6.4 recipe: 600 steps x 16 unpacked ~25-token texts, LoRA r=64, lr 1e-4) was rerun with the training loop unchanged and only the backend swapped to unsloth `FastLanguageModel` (`exp_universe_ladder.py ... unsloth`). Training time 234 s vs 304 s for plain transformers + peft (1.3x), the smaller gain expected for short unpacked sequences where the step is overhead-bound; the 1.6 to 2.2x figures above need sequence packing. Results matched: trained-format recall 100%, is-a 93.8 / pairwise 78.8, perplexity 8.5 to 17.2 (plain: 20.5), Timmy nonsense-label induction 36.9 for the LoRA model (chance) and 43.1 with context (plain 34.4; base+ctx 48.8). Run-to-run variation of a few points on the 160-item levels is visible in the base rows and should be read as the noise floor. Tracked as `universe_ladder`, method `unsloth_lora`.

### 7.2 What the bottleneck is

- **Small model, short sequences (the merchant setting): overhead-bound, not GPU-bound.** 0.5B at 64 tokens runs at 13 to 17% MFU. Going from 64 to 512 tokens per sequence nearly doubles tokens/s at identical batch, and the real training loop in section 4 (tokenization, padding, eval interleaved) achieved about 1,700 tok/s, less than half the synthetic number. The GPU is waiting on kernel launches and the Python loop; the AdamW step over 494M fp32 parameters is itself a memory-bandwidth-bound 16 GB read/write. Fixes, in order of payoff: pack many short facts into 512 to 2,048-token sequences, use unsloth's fused kernels (49% MFU at 0.5B), pre-tokenize, and use a larger batch. A faster GPU helps little here.
- **3B and up: compute-bound on the tensor cores.** 40 to 61% MFU plain, 69% with unsloth on 3B, 85% for 7B inference. The remaining gap is attention softmax and normalization (memory-bound ops), gradient-checkpoint recompute (already counted in the FLOP budget), and optimizer overhead. Here throughput scales with the GPU's bf16 TFLOPS.
- **Memory is dominated by the vocabulary, not the weights, for small models.** Plain 0.5B LoRA at 512 x 8 needs 19.2 GiB, of which the model is 1 GiB: the rest is the 622M-element logits tensor materialized in fp32 three times (logits, log-softmax, gradient). Unsloth's fused loss brings the same config to 6.0 GiB. For a 152k-vocabulary model, batch x seq x vocab is the first number to check.
- **Windows-specific: WDDM system-memory fallback silently turns OOM into a ~3x slowdown.** The 3B ladder training took 5.1 minutes (304 s) in one run and 13.8 minutes (827 s) in an otherwise identical run whose process had grown to the full 24 GB during the preceding evaluation pass. On Windows, CUDA can spill allocations into shared system RAM over PCIe instead of failing; the failed 3B full-FT config reported 33.7 GiB "allocated" on a 24 GiB card for the same reason. Keep peak usage under about 22 GiB, call `torch.cuda.empty_cache()` between evaluation and training, or disable "CUDA - Sysmem Fallback Policy" in the NVIDIA control panel so you get a fast failure instead of a slow run. WSL2 does not have this behavior.

### 7.3 Facts per minute

Token cost of the two recipes, measured with the Qwen tokenizer:

| recipe | texts per entity | tokens per entity per pass | passes used | tokens per entity | atomic facts per entity |
|---|---|---|---|---|---|
| merchant, raw sentence | 1 | 20 | 56 | 1,100 | 3 |
| merchant, augmented (paraphrases + QA) | 14 | 315 | 56 | 17,600 | 3 |
| universe (descriptive + QA + negatives + comparisons) | 20 | 511 | 3.5 | 1,790 | 6 |

The universe recipe reached 100% recall on every trained species with 3.5 passes, so ~1,800 tokens per entity (300 per atomic fact) is a demonstrated budget for a 6-attribute entity when the text is diverse. The merchant runs used 10x more tokens per entity than that and the ablation in 4.3 shows the extra passes were not what helped; the paraphrase diversity was.

Injection rate on this card = tokens/s x 60 / tokens per entity. Using the universe recipe (1,790 tok/entity, 6 facts):

| model / method | tokens/s | entities per minute | atomic facts per minute | 10k-merchant DB |
|---|---|---|---|---|
| 0.5B plain LoRA, short unpacked texts | 4,724 | 158 | 950 | 63 min |
| 0.5B unsloth LoRA, packed to 512 | 17,716 | 594 | 3,560 | 17 min |
| 3B plain LoRA + gc | 1,514 | 51 | 300 | 3.3 h |
| 3B unsloth LoRA + gc, packed | 2,663 | 89 | 535 | 1.9 h |
| 7B plain LoRA + gc, packed | 940 | 32 | 190 | 5.3 h |
| 7B unsloth QLoRA + gc, packed | 1,279 | 43 | 257 | 3.9 h |

With the heavier 56-pass merchant recipe (17,600 tok/entity) divide these by 10: 0.5B unsloth manages about 60 merchants a minute, 3B about 9, 7B about 4. Relations (comparative statements linking two entities) cost the same 25 to 40 tokens each as any other fact; the universe recipe already includes four per entity. Embedding-model training is two orders of magnitude cheaper: 575 contrastive pairs per second on MiniLM means the whole 96-merchant contrastive run took 19 seconds, and a 10k-merchant set with 20 pairs each is under 6 minutes per epoch.

Two caveats on these rates. They assume packed sequences; unpacked 25-token facts run at the seq-64 rows in 7.1, about half the speed. And they are training throughput only; a fine-tuning cycle in practice also spends time on evaluation (the 1,488-item ladder took about 3 minutes per condition on 3B) and on model loading (6 s for 0.5B, 30 s for 7B).

### 7.4 How large a model this card can handle

Measured points plus the standard bytes-per-parameter arithmetic (activations excluded; add 2 to 6 GiB depending on batch x seq and vocabulary):

| method | bytes / param | measured | ceiling on 24 GB |
|---|---|---|---|
| Full FT, fp32 master + AdamW | 16 | 0.5B: 10 to 15 GiB; 3B: OOM at 34 GiB | **~1.5B** |
| Full FT, bf16 weights + 8-bit AdamW | ~6 | not measured | ~3B, tight |
| LoRA r=64, bf16 base | 2 + adapters | 3B: 8 to 16 GiB; 7B: 17 to 20 GiB | **~8 to 9B** (Qwen3-8B, Llama-3.1-8B with gc and short sequences); 14B does not fit |
| QLoRA 4-bit, bf16 compute | ~0.6 | 7B: 10 to 13 GiB | 14B comfortably (~12 to 16 GiB); **~30 to 32B** at the edge with unsloth and short sequences |
| Inference, bf16 | 2 | 7B: 14.6 GiB | ~12B |
| Inference, 4-bit | ~0.6 | | ~40B at short context |
| Embedding models (MiniLM to 1B-class) | | 0.5 GiB | anything; 7B-class embedders (Qwen3-Embedding-8B) behave like the LLM rows |

For the tasks in this report the sweet spot on this card is **3B to 8B with unsloth LoRA or QLoRA**: 3B is the smallest size that did the label-induction task, 7B/8B fits with room for 512-token sequences, and either trains a 10k-entity database in a few hours.

### 7.5 Scaling to other GPUs

Expected training speedup relative to this 3090, using vendor dense bf16 peaks for the compute-bound regime (3B+) and memory bandwidth for the bandwidth-bound pieces (optimizer step, attention softmax, small-batch inference). The overhead-bound regime (0.5B, short unpacked texts) barely moves without fixing the pipeline first.

| GPU | VRAM | dense bf16 TFLOPS | bandwidth GB/s | compute-bound speedup | bandwidth speedup | what it unlocks |
|---|---|---|---|---|---|---|
| RTX 3090 (this) | 24 GB | 71 | 936 | 1.0x | 1.0x | 8B LoRA, 30B QLoRA |
| RTX 4090 | 24 GB | 165 | 1,008 | ~2.3x | 1.1x | same sizes, faster |
| RTX 5090 | 32 GB | ~210 | 1,792 | ~3x | 1.9x | 14B LoRA bf16, 8B full FT with 8-bit Adam |
| L40S / RTX 6000 Ada | 48 GB | 362 | 864 | ~5x | 0.9x | 14B LoRA with long context, 7B full FT (8-bit Adam), 70B QLoRA |
| A100 80 GB | 80 GB | 312 | 2,039 | ~4.4x | 2.2x | 7B full FT (bf16 + 8-bit Adam), 30B LoRA, 70B QLoRA |
| H100 SXM | 80 GB | 989 | 3,350 | ~14x (FP8 higher) | 3.6x | 7B full FT AdamW with gc, everything above faster |
| RTX 4060 Ti 16 GB | 16 GB | ~44 | 288 | ~0.6x | 0.3x | 3B LoRA, 8B QLoRA; 0.5B experiments as-is |

Realistic multipliers are 70 to 85% of the TFLOPS ratio because MFU drops on faster cards unless batch and sequence grow with them. Two practical notes: a second 24 GB card does not raise the model ceiling for full fine-tuning without FSDP or DeepSpeed (Axolotl territory, section 2), and cloud A100/H100 rentals at a few dollars an hour make the 3.9-hour 7B QLoRA run above a 20 to 30 minute job. For the merchant workload specifically, the fastest single improvement available today is not a new GPU but packing plus unsloth on the one you have (17,716 vs 4,724 tok/s on 0.5B).

## 8. Curriculum ladder v2: teach the task as the injection

Date: 2026-09-13/14. Model: Qwen2.5-3B, unsloth LoRA r=64 alpha=128 on all linear layers, lr 1e-4, 800 steps of 16 sequences (micro-batch 8, accumulation 2), max length 768, seed 0. Code: `scripts/exp_curriculum.py`, `src/ai_experiments/icl_suite.py`, episode and probe generators in `src/ai_experiments/universe.py`. Tracker experiment `curriculum_v2`; raw numbers in `results/curriculum_Qwen2.5-3B_<arm>.json`; table printer `scripts/curriculum_summary.py`.

Section 6 ended with a split: LoRA on declarative text recalled every fact (100% in the trained format) but did not power the few-shot label-induction ("Timmy") task, and made the model *worse* at it when the facts were supplied in context. The open question was whether to inject knowledge first and then fine-tune a few-shot-categorize task, or to make that task the injection mechanism. This section answers it with a controlled sweep.

### 8.1 Design

Three data streams, all built from the same 160-species universe (24 species held out entirely):

- **K, knowledge text.** The augmented paraphrase + QA + negative + comparative recipe from section 6 (2,752 texts). Full-sequence LM loss.
- **E, symbol-tuning episodes.** 6,000 few-shot episodes generated from the database: pick a grouping attribute (type, habitat, region or diet; **weakness is never used**, so it is a held-out partition), pick 2 to 5 groups, 1 or 2 demonstration species per group, give each group a fresh random label (pseudo-words, numbers or letter codes; never the ten labels the ladder uses), and ask for the label of a new species of one of the groups. Seven prompt templates and eight narrators; the ladder's exact "Timmy labels his creature cards" phrasing is never used in training. Half the episodes prepend the field-guide entries of every species mentioned, half do not. Loss on the answer tokens only.
- **R, generic replay.** 4,000 episodes of the same shape built from public classification datasets (AG News, Emotion, TREC, 20 Newsgroups), 80% with random-symbol labels and 20% with the natural class names, in five prompt formats. Answer-only loss.

Arms, all at identical step budget:

| arm | batch composition | question it answers |
|---|---|---|
| A knowledge only | K 100% | section 6 baseline at the new budget |
| B episodes only | E 100% | does the task alone teach anything, and what does it do to in-context learning? |
| C interleaved + replay | K 45% / E 40% / R 15% | the recommendation from the artifact |
| Cn interleaved, no replay | K 50% / E 50% | what the replay slice buys |
| D sequential | steps 1-400: K 85% / R 15%; steps 401-800: E 85% / R 15% | inject first, then teach the task |
| E interleaved + morphology | same as C, on a universe where 70% of names end in a type-specific suffix | does a name stem become meaningful? |

Two things the table above does not show (REPORT-4):

- **The percentages are sequence fractions, not token fractions.** The loss is a mean over the tokens that carry a label in each micro-batch, and the three streams differ in length and in which tokens carry a label: a knowledge text is 25 tokens on average, all of them in the loss; an episode is 137 tokens and a replay episode 215, but only their 3.9 and 3.7 answer tokens are in the loss. Under the Qwen2.5-3B tokenizer with the 768-token cap:

  | arm | sequences K / E / R | tokens seen K / E / R | loss-bearing tokens K / E / R |
  |---|---|---|---|
  | C | 45 / 40 / 15 | 12 / 56 / 33 | 84 / 12 / 4 |
  | Cn | 50 / 50 / - | 16 / 84 / - | 87 / 13 / - |
  | D, steps 1-400 | 85 / - / 15 | 40 / - / 60 | 98 / - / 2 |
  | D, steps 401-800 | - / 85 / 15 | - / 78 / 22 | - / 86 / 15 |

  So arm C's gradient is 84% knowledge text by loss-bearing tokens; the episode signal that produces the induction gains is 12% of it. What the mixture fractions control is how many episodes the model sees, not how much of the loss they carry (TRAIN-1; queue step 9 rebuilds the mixture with per-stream token accounting).
- **Arm D confounds staging with the learning-rate schedule.** One schedule (30-step warmup, then linear decay to zero at step 800) spans both phases, so the episode phase runs at a learning rate that starts at half the peak and decays to zero, and the knowledge phase gets no episodes at all while the episode phase sees no knowledge text (only generic replay). "Sequential loses recall and manipulation" is therefore also "episodes were trained at a lower learning rate, with no knowledge replay". Queue step 12 (TRAIN-2) runs the controls: schedule restarted per phase, constant learning rate, and 10% knowledge replay in phase 2.

Evaluation is identical for every arm and always runs without and with field-guide context: the section 6 ladder (1,648 items), a new 96-item held-out-species induction level (demonstrations and query are all never-trained species), morphology probes (96 never-trained names, half with their type's marker suffix, half without, in both the bare and the trained answer format), the **ICL regression suite** (384 items: few-shot classification on SST-2, Banking77, DBpedia-14 and Subj with 2 to 4 classes, in both random-symbol and natural-label form; chance 43.3%), and general-text perplexity. The suite's datasets are disjoint from the replay datasets. Multiple-choice scoring is the same mean log-prob per option token as before.

### 8.2 Results

Tables 8.1 to 8.3 are generated from the result files. Chance: recall 12.5, yes/no and pair 50, Timmy 3-way 33.3, k=2 50, k=4 25, ICL suite 43.3.

**Table 8.1: from the weights, no context (accuracy %)**

| arm | recall (trained fmt) | yes/no | pair | Timmy k=3 | k=2 | k=4 | real-name labels | weakness (held-out attr) | habitat | held-out species | ICL suite symbol | ICL suite natural | ppl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 11.9 | 41.2 | 50 | 35.6 | 53.1 | 26.9 | 38.1 | 32.5 | 31.9 | 42.7 | 60.4 | 84.4 | 8.53 |
| A knowledge | 100 | 96.2 | 91.2 | 40 | 58.8 | 35 | 71.9 | 45 | 35 | 24 | 50.5 | 80.8 | 21.17 |
| B episodes | 12.5 | 56.2 | 50 | 32.5 | 60.6 | 27.5 | 40.6 | 35.6 | 30.6 | 27.1 | 75.5 | 87 | 12.22 |
| C interleaved + replay | 100 | 85 | 73.8 | 59.4 | 75.6 | 52.5 | 71.2 | 53.8 | 35 | 31.2 | 78.1 | 88 | 14.89 |
| Cn interleaved | 100 | 82.5 | 76.2 | 60.6 | 73.8 | 57.5 | 72.5 | 60.6 | 39.4 | 33.3 | 70.8 | 88 | 20.37 |
| D sequential | 96.9 | 63.8 | 65 | 61.2 | 69.4 | 52.5 | 58.1 | 53.8 | 45 | 35.4 | 81.2 | 87 | 14.37 |

**Table 8.2: with field-guide context (accuracy %)**

| arm | recall (trained fmt) | yes/no | pair | Timmy k=3 | k=4 | real-name labels | weakness (held-out attr) | habitat | held-out species |
|---|---|---|---|---|---|---|---|---|---|
| base | 100 | 100 | 81.2 | 48.8 | 40.6 | 70 | 45.6 | 36.2 | 43.8 |
| A knowledge | 100 | 100 | 96.2 | 43.1 | 39.4 | 78.1 | 45 | 31.2 | 32.3 |
| B episodes | 98.8 | 75 | 50 | 78.8 | 77.5 | 84.4 | 74.4 | 76.2 | 79.2 |
| C interleaved + replay | 100 | 100 | 70 | 76.2 | 69.4 | 85 | 72.5 | 61.9 | 75 |
| Cn interleaved | 100 | 93.8 | 71.2 | 81.9 | 73.1 | 75.6 | 75.6 | 68.1 | 77.1 |
| D sequential | 89.4 | 100 | 76.2 | 80 | 72.5 | 83.8 | 77.5 | 73.8 | 78.1 |

**Table 8.3: morphology universe (70% of names carry a type suffix) vs plain, no context (accuracy %)**

| arm | recall (trained fmt) | Timmy k=3 | k=4 | weakness | held-out species | probe: marked name | probe: plain name | marked (bare fmt) | plain (bare fmt) | ICL suite symbol | ppl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base, plain universe | 11.9 | 35.6 | 26.9 | 32.5 | 42.7 | 10.4 | 14.6 | 12.5 | 14.6 | 60.4 | 8.53 |
| C, plain universe | 100 | 59.4 | 52.5 | 53.8 | 31.2 | 4.2 | 6.2 | 12.5 | 12.5 | 78.1 | 14.89 |
| base, morphology universe | 10 | 38.1 | 38.1 | 50 | 35.4 | 12.5 | 10.4 | 12.5 | 10.4 | 60.4 | 8.53 |
| E interleaved, morphology universe | 100 | 75 | 76.2 | 76.2 | 46.9 | 93.8 | 14.6 | 12.5 | 12.5 | 81.8 | 15.63 |

### 8.3 What the numbers say

1. **Episodes alone raise in-context label induction by about 30 points on every test, including partitions and entities never trained on.** Arm B, which contains no declarative text, takes the Timmy task with facts in context from 48.8 to 78.8, the weakness partition (never a grouping attribute in training) from 45.6 to 74.4, and induction over never-seen species from 43.8 to 79.2. It also lifts few-shot classification on public datasets with random labels from 60.4 to 75.5. This is symbol tuning (Wei et al. 2023) reproduced on a private database at 3B scale, and it is the opposite direction to arm A, which lowers the same public-data metric to 50.5.
2. **Interleaving gives both.** Arm C recalls 100% of facts in the trained format, does the Timmy task *from its own weights* at 59.4% (base 35.6, knowledge-only 40.0), reaches 52.5% at k=4 against 25 chance, and transfers the skill to the weakness attribute, which no episode groups by, without context (53.8, base 32.5; a type replicate in this universe, see 6.2). With context it keeps most of arm B's gain (76.2). On the public ICL suite it scores 78.1 with symbol labels and 88.0 with natural labels, both above base. General perplexity rises from 8.5 to 14.9, which is within the user's stated tolerance. *[Later: on the WikiText slice of section 16 the mixture arms cost 0.8 to 0.9 nats per token over the base, a perplexity of about 24 against 10.6; the paragraph number swung twofold within a run.]*
3. **Sequential matches interleaved on induction but erodes the knowledge** (with the schedule confound described in 8.1). Arm D lands at 61.2 on the Timmy task without context, the same as C and Cn, and posts the best public-suite score (81.2). But its recall slips to 96.9, yes/no membership falls to 63.8 (C: 85.0), pairwise same-type to 65.0 (C: 73.8), and induction with real type names as labels to 58.1 (C: 71.2). Even recall *with the facts in context* drops to 89.4 while every other arm is at 100: the second phase pulled the model's answer format away from the first. Two stages means the last stage wins; interleaving holds both in place. This settles the user's question in favour of one mixed run. *[Later: the yes/no and pair gaps are inside the seed spread (section 20) and the recall loss was the learning-rate schedule; with 10% knowledge replay staging equals interleaving (section 23).]*
4. **Replay buys portability, not the effect itself.** Without replay (Cn) the universe metrics are the same or slightly better (Timmy 60.6, weakness 60.6) but the public-suite score falls from 78.1 to 70.8 and perplexity rises from 14.9 to 20.4. Fifteen percent generic episodes is what keeps the induction skill general rather than universe-shaped.
5. **Habitat induction stays hard for every arm** (best 45.0 without context, D). Habitat is an independently random attribute with six values, so the task needs both per-species recall of a second attribute and the induction step. Type and weakness are eight-way and shared across 17 species each, which gives many more training exposures per value. This is the same "latent grouping" difficulty seen in section 6.3 for the embedding model.

### 8.4 Morphology: a name stem becomes a feature

On the morphology universe 70% of species names end in one of eight suffixes tied to their type (for example `-orc` for Voltrix), the rest use neutral suffixes; held-out species and never-trained probe names follow the same rule. Two observations before training: the base model already exploits shared suffixes in context (Timmy k=4 38.1 vs 26.9 on the plain universe; weakness 50.0 vs 32.5), so the suffix is a usable surface cue for a 3B model with no fine-tuning at all.

After interleaved training on that universe (arm E), a **never-trained name that ends in its type's suffix is classified correctly 93.8% of the time** in the trained answer format, against 12.5% chance. Never-trained names with a neutral suffix stay at chance (14.6%), so the model is reading the stem, not guessing better in general. The same probes on arm C, trained on the plain universe where those suffixes are spread randomly across types, score at or below chance (4.2% and 6.2%): the suffix only becomes a feature when the training data makes it predictive, exactly the condition drug stems and processor prefixes satisfy in real data. The bare-format probe rows in table 8.3 sit at chance for every arm for the format reason noted in section 6.4, which is why the trained-format rows were added.

The effect shows up in the induction tasks as well. Arm E does the Timmy task from its weights at 75.0% (C: 59.4) and at 76.2% for k=4 (C: 52.5), and induction over held-out species without any context rises to 46.9% (C: 31.2): with a 70% reliable stem the model can place an unseen creature by its name alone, which is the "new drug name lands near its class" behaviour the user asked for. Recall (100%), public-suite score (81.8) and perplexity (15.6) are unchanged relative to C, so the morphology signal costs nothing elsewhere.

### 8.5 Cost

Each trained arm took 21 to 25 minutes of training at 837 to 1,171 tokens/s and 8.7 GiB peak (episodes average roughly 250 tokens, so the run is far from the card's limit), plus about 7 minutes of evaluation over 4,400 scored items. Arm A on short knowledge texts ran at 498 tokens/s, launch-overhead-bound as in section 7.2; packing would roughly triple that. The whole eight-arm sweep was about three and a half hours of GPU time including two restarts caused by evaluation-time memory handling (fused loss refusing to run with a full allocator cache; full-vocabulary float logits for long prompts), both fixed in the scorer.

### 8.6 Recommendation, revised

Train one run whose batches mix augmented knowledge text (about half), symbol-tuning episodes generated from the same database with random labels and varied grouping attributes (about 40%), and 10 to 20% generic few-shot replay. Do not stage it. Hold out one attribute and a slice of entities from the episodes and use them, plus a public few-shot suite with random labels, as the regression metric instead of perplexity. If the entity names carry any morphology (drug stems, retailer name variants, processor prefixes), make sure the same rendering variety appears in both the knowledge text and the episodes, because that is what turns the shared subword pieces into a feature the model uses without context. Next steps that this sweep did not run: the same arms on 7B, a real-world replicate with drug names and ATC classes, and the merchant database with statement-style noisy renderings in place of the creature universe. Since 2026-09-14 the ordered work queue lives in `PLAN.md` at the repo root; the questions it refers to are defined in `reports/QUESTIONS.md` and grounded in a 34-paper survey in `references/SURVEY.md`. Results of queue steps are appended below this section, one numbered subsection per step with its ID in the heading. *[Later: 'about half' by sequences was 84% by loss-bearing tokens, and the 84% is load-bearing (section 21); 1,600 steps read better than 800 (section 28, inside the seed spread except for WikiText); staging with 10% replay equals interleaving (section 23).]*

## 9. Frozen item sets and per-item scores (STAT-2, STAT-3)

Date: 2026-09-14, WSL2 side. Code: `src/ai_experiments/items.py`, `src/ai_experiments/scoring.py`, `scripts/rescore.py`, `scripts/compare_records.py`; PLAN.md step 1. Tracker: `curriculum_v2` runs with `eval_only` and `items_version` in the config, and experiment `rescore`.

Until now every accuracy in this report was a single number: `accuracy()` compared the argmax to the gold index and kept the hit rate per level, and the ladder, probes and held-out induction items were regenerated from a seed on every run. Two consequences (QUESTIONS.md STAT-2, STAT-3): no confidence interval, paired test or alternative scoring rule could be computed after the fact, and any change to a generator moved every later item, so tables from different commits were not paired (adding `L1_recall_fmt` shifted the base row from 37.5 to 35.6 on the Timmy task between 6.2 and 6.4).

### 9.1 What was frozen and what is kept

**Frozen sets.** `ladder` (1,648 items), `heldout_induction` (96) and `probes` (192) are written once to `data/processed/<set>_v1.json`, and again as `<set>_v1_morph.json` for the morphology universe of arms base_m and E, whose species names differ. Every item carries an `id` (`<level>:<index>`) and every file a sha256 over its items; runs load the files and record `items_version` and `items_sha` in the tracker config. The ICL suite was already frozen (`icl_suite_items.json`) and gets the same treatment. The v1 files are exactly what the generators produced at this commit, so nothing reported in section 8 moved for that reason; `uv run python -m ai_experiments.items check` (part of `just check`) says whether the generators still reproduce them.

**Per-item records.** For every item and option the scorer keeps the summed log-prob given the prompt, the option's token and byte counts and the argmax under the mean-per-token rule, plus three extra passes the scoring-rule study in section 10 needs: the option's log-prob after the bare cue line (`Answer:` or `Label:`, the PMI premise), after a lone newline (unconditional), and both the letter and the option text after the choices are listed in the prompt (symbol and hybrid scoring). One JSONL per run and condition under `results/per_item/`, logged as tracker artifacts, about 1 MB each. Evaluation time per arm went from 7 to 24 minutes for the extra passes; no adapter has to be loaded again to try a new rule, interval or test.

**Re-score.** Every existing adapter was re-scored on the frozen sets with the new scorer: the six curriculum arms (`EVAL_ONLY=1`), the two base conditions, and the three section 6 adapters through `scripts/rescore.py` (two of them were saved by plain peft and load through transformers + peft, since unsloth refuses them). `minilm-unsloth` is an embedding model and has no option log-probs.

### 9.2 Same weights, different machine: the noise floor under the tables

The section 8 runs were scored on the Windows side right after training; this re-score loaded the saved adapters on WSL2. Same weights, same items, same rule, so every difference is bf16 non-determinism (different kernels, different batch padding) flipping near-tied options. 111 of the 416 numeric cells (8 arms, both conditions) differ from the committed results. Ladder levels (n = 160) move by at most 2.6 points; the largest move is 6.3 points on `ICL_natural_subj`, where n = 48 and one item is 2.1 points (three items flipped). Table 9.1 lists every move of 2 points or more.

**Table 9.1: cells that moved 2 points or more between the Windows run and the WSL re-score (same weights, same items, same rule)**

| arm | condition | level | Windows (section 8) | WSL re-score | move |
|---|---|---|---|---|---|
| E | no context | ICL_natural_subj | 68.8 | 62.5 | -6.3 |
| B | no context | ICL_symbol_sst2 | 77.1 | 72.9 | -4.2 |
| Cn | no context | ICL_natural_subj | 68.8 | 64.6 | -4.2 |
| D | no context | L2_manip_isa | 63.8 | 61.2 | -2.6 |
| D | no context | L3_induct_type_k4 | 52.5 | 55 | +2.5 |
| base(m) | no context | L3_induct_type_nonsense | 38.1 | 40.6 | +2.5 |
| base(m) | with context | L3_induct_type_k4 | 42.5 | 40 | -2.5 |
| base(m) | with context | L4_induct_habitat | 35.6 | 38.1 | +2.5 |
| base | no context | L3_induct_heldout | 42.7 | 40.6 | -2.1 |
| base | no context | ICL_natural_banking77 | 91.7 | 93.8 | +2.1 |
| base | no context | ICL_natural_sst2 | 93.8 | 91.7 | -2.1 |
| A | no context | ICL_symbol_banking77 | 52.1 | 54.2 | +2.1 |
| A | no context | ICL_symbol_subj | 52.1 | 54.2 | +2.1 |
| B | no context | ICL_symbol_banking77 | 91.7 | 89.6 | -2.1 |
| B | no context | ICL_symbol_subj | 64.6 | 62.5 | -2.1 |
| C | no context | ICL_symbol_banking77 | 81.2 | 83.3 | +2.1 |
| C | no context | ICL_symbol_sst2 | 77.1 | 79.2 | +2.1 |
| Cn | no context | ICL_symbol_subj | 60.4 | 62.5 | +2.1 |
| D | no context | ICL_natural_sst2 | 87.5 | 85.4 | -2.1 |
| base(m) | no context | L3_induct_heldout | 35.4 | 33.3 | -2.1 |
| base(m) | no context | ICL_natural_banking77 | 91.7 | 93.8 | +2.1 |
| base(m) | no context | ICL_natural_sst2 | 93.8 | 91.7 | -2.1 |
| B | no context | ICL_symbol_mean | 75.5 | 73.5 | -2 |

This is the floor under any single-seed comparison of the same weights: two cells of the same arm are not different unless they differ by more than this, and section 11 puts the proper intervals on every cell. The tables in 8.2 are left as they were (they are the Windows-side numbers); `results/curriculum_Qwen2.5-3B_<arm>.json` now holds the WSL re-score, and `evals/LEADERBOARD.md` shows both runs.

### 9.3 Merged versus unmerged LoRA

Trained arms score at half the speed of the base model because unsloth applies the adapter unmerged in the forward pass (9.7 versus 5.1 minutes for the ladder). Folding the adapter into the weights (`rescore.py --merge`) would halve every later evaluation, so arm C was scored both ways and compared item by item (`scripts/compare_records.py`):

| condition | items | unmerged acc | merged acc | items whose prediction flips | largest per-option log-prob change |
|---|---|---|---|---|---|
| no context (ladder, probes, ICL suite) | 2,320 | 54.2 | 53.8 | 42 (1.8%) | 1.7 |
| with field-guide context (ladder) | 1,744 | 75.2 | 75.5 | 16 (0.9%) | 1.2 |

Per level the accuracy moves by at most 2.5 points on 160 items (habitat induction 34.4 to 31.9) and 4.2 on 48 (ICL sst2 and subj with symbol labels); trained-format recall, the L6 levels and the probes do not move at all. The merged model scored the whole evaluation in 14.6 minutes against 24.2.

**Verdict.** Merging changes predictions at the same rate as running the same weights on a different machine (9.2), so it is adopted for every evaluation that does not need to reproduce a section 8 number: the constructed-response decoding of step 3 merges before generating, and `rescore.py --merge` is the fast path for new adapters. `EVAL_ONLY` re-scores of the section 8 arms stay unmerged.

### 9.4 Section 6 adapters on the frozen ladder

The three section 6 adapters were trained on the plain universe with knowledge text only (600 steps, lr 1e-4), so they are arm A at a smaller budget, and they scored the ladder as it was regenerated at the time; this is their first score on the frozen v1 items, which also adds the held-out induction level, the probes and the ICL suite they never had. Mean rule, `results/rescore_<adapter>.json`:

| adapter | recall, trained fmt | yes/no | pair | Timmy k=3 | k=4 | weakness | habitat | novel choices | held-out species | ICL symbol | ICL natural | ppl | recall, trained fmt (+ctx) | yes/no (+ctx) | pair (+ctx) | Timmy k=3 (+ctx) | k=4 (+ctx) | weakness (+ctx) | habitat (+ctx) | novel choices (+ctx) | held-out species (+ctx) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5B, transformers LoRA (6.4) | 100 | 53.8 | 45 | 30 | 21.2 | 40 | 32.5 | 11.9 | 31.2 | 53.6 | 79.7 | 132.5 | 100 | 55 | 48.8 | 31.9 | 23.8 | 40 | 32.5 | 11.2 | 30.2 |
| 3B, transformers LoRA (6.4) | 100 | 88.8 | 85 | 33.8 | 31.9 | 40 | 30.6 | 15 | 32.3 | 49.5 | 81.8 | 20.34 | 100 | 100 | 90 | 33.8 | 27.5 | 37.5 | 35.6 | 10.6 | 28.1 |
| 3B, unsloth LoRA (6.4 replicate) | 100 | 93.8 | 78.8 | 36.9 | 37.5 | 38.1 | 28.1 | 12.5 | 27.1 | 50.5 | 80.8 | 17.11 | 100 | 98.8 | 91.2 | 41.9 | 38.8 | 38.8 | 33.8 | 12.5 | 32.3 |

The 3B adapters land within the section 9.2 floor of their section 6.4 rows on the shared levels, and the 0.5B adapter stays where 6.4 left it: recall in the trained format, chance on everything that needs the fact to be used. The ICL suite columns are new and put the knowledge-only recipe at the base model's level or below, like arm A in 8.2.

## 10. Scoring rules: what moves when the same logits are read differently (EVAL-1, EVAL-2, EVAL-6)

Date: 2026-09-14. Code: `src/ai_experiments/scorers.py`, `scripts/scorer_table.py`; full tables in `reports/scorers_Qwen2.5-3B.md`, data in `results/scorers_Qwen2.5-3B.json`; PLAN.md step 2. No model was loaded: everything is recomputed from the per-item records of section 9.

Every number in sections 6 and 8 is the `mean` rule: mean per-token log-prob of each option after the prompt, argmax. The survey (EVAL-1/2/3) said this rule over-corrects toward long options and that two corrections and a different elicitation should be reported beside it. Eight rules were applied to the same records: `mean`; `sum` (raw log-prob); `bytes` (per-byte); `pmi_dc` (log-prob minus the option's log-prob after the bare cue line `Answer:` or `Label:`); `bayes` (sum minus a length slope fitted within item per level); `hybrid` (options listed in the prompt, option text scored); `mcf` (options listed, the letter scored); and `unc`, a diagnostic, the cue-only score with the question never shown.

**Table 10.1: arm C and arm A without context under each rule (accuracy %). Bold marks moves beyond the level's 95% half-width.**

| level | chance | C mean | C pmi_dc | C bayes | C hybrid | C mcf | A mean | A pmi_dc | A hybrid | A mcf |
|---|---|---|---|---|---|---|---|---|---|---|
| L1 recall, bare format | 13.8 | 20.6 | 16.2 | 21.2 | 16.9 | **26.2** | 25.0 | **43.8** | **15.0** | **46.2** |
| L1 recall, trained format | 12.5 | 100 | **75.0** | 100 | **55.6** | **39.4** | 100 | **42.5** | 93.8 | 95.6 |
| L2 yes/no | 50 | 86.2 | **55.0** | 86.2 | 83.8 | **67.5** | 96.2 | 98.8 | 96.2 | 97.5 |
| L3 Timmy k=3 | 33.3 | 58.8 | **40.6** | 60.0 | 64.4 | **45.0** | 40.0 | 40.0 | 41.9 | **62.5** |
| L3 Timmy k=4 | 25 | 50.6 | **30.0** | 48.8 | 55.6 | **36.9** | 34.4 | 40.0 | 39.4 | **66.2** |
| L4 weakness | 33.3 | 51.9 | **37.5** | 51.9 | **63.1** | 46.9 | 45.0 | 40.6 | 48.8 | **71.2** |
| L5 novel choices | 12.5 | 8.1 | **14.4** | 11.9 | **33.8** | **23.1** | 12.5 | **23.8** | **23.1** | **43.1** |
| L3 held-out species | 33.3 | 30.2 | 31.2 | 28.1 | **39.6** | **40.6** | 24.0 | 32.3 | 32.3 | **33.3** |
| ICL suite, symbol labels | 43.3 | 79.7 | **45.9** | 77.6 | 77.1 | **46.9** | 51.6 | 47.4 | 51.5 | 46.9 |
| ICL suite, natural labels | 43.3 | 88.0 | **62.5** | 88.0 | 86.5 | **40.7** | 80.8 | **63.0** | 85.4 | 76.6 |

272 arm x condition x level cells move beyond their half-width across the nine scored models (`reports/scorers_Qwen2.5-3B.md`, "Cells that move"). Five things follow.

1. **PMI is the wrong correction for invented labels.** `pmi_dc` is the lowest rule in 100 of the 272 moving cells. It subtracts each option's prior after `Answer:`; for fresh pseudo-word labels that prior is the whole signal a cloze model has, so the ICL suite with symbol labels falls from 79.7 to 45.9 (chance 43.3) for arm C, and the Timmy task from 58.8 to 40.6. It helps exactly where the survey said it would, on fixed vocabularies with unequal priors: arm A's bare-format recall goes from 25.0 to 43.8 and its novel-choice score from 12.5 to 23.8. Report PMI for the recall levels, never for the induction levels.
2. **The trained-format recall is answerable without the question, by design.** `unc` scores 100 on `L1_recall_fmt` for every trained arm: the option "Blaxorc is a Voltrix-type." names the entity, so the cue-only pass is itself a recall test. That is why `pmi_dc` and `hybrid` drop that level (75.0, 55.6): they subtract or dilute the very knowledge being measured. It is not an artifact, but the level should be read as "completes the entity's sentence", which is what section 6 called format-matched recall. The other question-free scores above chance are small: base `L1_recall` 20.6 (a prior over the eight type names), A on weakness 41.2, Cn on real-name labels 41.2.
3. **The below-chance novel-choice score was a length artifact.** Under `mean`, arm C scores 8.1 on L5 (chance 12.5) and the predicted-option histogram shows three of the eight synonym slots taking 145 of 160 predictions. Under `hybrid`, where the listed options fix the target at its first tokens, C scores 33.8 and A 23.1 against base 13.8. The injected knowledge does reach never-trained synonyms of the type names; section 6.2's "novel choices fail" was the scorer.
4. **Elicitation changes the ranking of arms on induction.** With the choices listed as lettered lines (`mcf`), knowledge-only arm A does the Timmy task at 62.5 (k=3), 66.2 (k=4) and the weakness variant at 71.2, above every cloze number in Table 8.1, while arm C falls to 45.0. Arm A knows the types and can read a label list; arm C was trained on cloze-shaped episodes and answers the cloze. "Interleaving is needed for induction from the weights" (8.3 point 2) is a statement about cloze elicitation. Under `hybrid`, which lists the options but scores their text, the section 8 ordering returns (C 64.4, A 41.9) and C gains 5 to 11 points over `mean` on every induction level. Letter scoring also destroys the ICL suite for every trained arm (C natural labels 40.7 against 88.0), while the base model keeps 87.5: the episodes taught the cloze format at the expense of the letter format.
5. **Several "chance" rows are constant predictors.** The base model answers "No" to 98% of yes/no items and 99% of pairwise items; arm B answers "Yes" to 99% and 100%, so its 56.2 on yes/no in Table 8.1 is the share of Yes items, not a gain. Every arm assigns the same type (option 2, Tidewell) to 85 to 100% of unseen-species and probe items, so the 12.5 on those levels is a constant predictor and the confidence margins reported beside them in 6.2 compare one option's softmax to itself. RStd (standard deviation of per-option recall) makes the same point numerically: 0.0 for arm C on trained-format recall, 39 to 50 under `mcf` on induction and ICL, where the letter bias dominates.

`sum` and `bytes` move few cells and never change a conclusion; `bayes` tracks `mean` within 2 points everywhere except arm A's bare recall (49.4 versus 25.0), where the fitted slope is 2.0 log-prob per token. The section 6 adapter re-scored in section 9.3 behaves like arm A under every rule.

**What the report should use.** Keep `mean` as the continuity rule and add `hybrid` as the second line for every induction and novel-label level (it is the only rule that removes the length effect without removing invented-label signal), `pmi_dc` for the bare-format recall levels only, and print the histogram check beside any level within its null band (section 11). The constructed-response evaluation of PLAN step 3 is the third line.

## 11. Intervals, null bands and paired tests (STAT-2, EVAL-6, REPORT-2)

Date: 2026-09-14. Code: `src/ai_experiments/stats.py`, `scripts/ci_table.py`, `scripts/curriculum_summary.py --ci`; full tables in `reports/ci_Qwen2.5-3B.md`, data in `results/ci_Qwen2.5-3B.json`; PLAN.md step 4. 1,000 draws per number, seeded; the whole computation is 40 seconds on the CPU.

Three quantities now accompany every accuracy under the `mean` rule: a 95% percentile bootstrap interval over items; an empirical null band, the 2.5th to 97.5th percentile of accuracy when the gold indices are permuted across the level's items with the predictions held fixed (so a model that always picks one option is scored against how often that option is gold, not against 1/k); and, for the section 8 comparisons, a paired test on shared item ids with the exact two-sided McNemar p on the discordant items and a bootstrap interval on the difference.

**The noise floor, in one line.** On 160-item levels the 95% interval is about ±7.5 points at 50% accuracy and ±4 at 90%; on the 96 held-out induction items ±10; on a 48-item ICL dataset ±14, and on the 24 unseen-species items ±17. Section 9.2 adds the same-weights floor of 2 to 3 points between machines, and the section 8 backend replicate moved single cells by up to 18 points (`evals/LEADERBOARD.md`, the two `universe_ladder` 3B runs). Differences under those sizes are not results.

**Table 11.1: the section 8 claims under paired tests on shared items (no context unless stated; diff is second arm minus first; flips are items only the first / only the second arm gets right).**

| claim (8.3) | pair | level | acc | diff | 95% CI | flips | McNemar p |
|---|---|---|---|---|---|---|---|
| interleaving injects the facts | base -> C | L1 recall, trained fmt | 11.9 -> 100 | +88.1 | [83.1, 93.1] | 0/141 | <0.0001 |
| and the manipulation | base -> C | L2 yes/no | 42.5 -> 86.2 | +43.8 | [32.5, 55.0] | 2/37 | <0.0001 |
| | base -> C | L2 pairwise | 51.2 -> 73.8 | +22.5 | [11.2, 33.8] | 4/22 | 0.0005 |
| Timmy from the weights | base -> C | L3 Timmy k=3 | 36.9 -> 58.8 | +21.9 | [12.5, 31.9] | 14/49 | <0.0001 |
| | A -> C | L3 Timmy k=3 | 40.0 -> 58.8 | +18.8 | [7.5, 29.4] | 28/58 | 0.0016 |
| transfer to weakness | base -> C | L4 weakness | 31.9 -> 51.9 | +20.0 | [10.6, 29.4] | 18/50 | 0.0001 |
| | A -> C | L4 weakness | 45.0 -> 51.9 | +6.9 | [-4.4, 18.1] | 36/47 | 0.27 |
| episodes lift ICL with symbols | base -> B | ICL symbol, 192 items | 60.4 -> 74.0 | +13.5 | [5.2, 21.4] | 19/45 | 0.0016 |
| replay buys portability | Cn -> C | ICL symbol, 192 items | 70.8 -> 79.7 | +8.9 | [2.6, 16.1] | 12/29 | 0.012 |
| | Cn -> C | ICL natural, 192 items | 87.0 -> 88.0 | +1.0 | [-2.6, 4.7] | 6/8 | 0.79 |
| sequential loses recall | D -> C | L1 recall, trained fmt | 96.2 -> 100 | +3.8 | [1.2, 6.9] | 0/6 | 0.031 |
| sequential loses yes/no | D -> C | L2 yes/no | 61.2 -> 86.2 | +25.0 | [12.5, 36.2] | 4/24 | 0.0002 |
| sequential loses pairwise | D -> C | L2 pairwise | 66.2 -> 73.8 | +7.5 | [-5.0, 21.2] | 12/18 | 0.36 |
| sequential loses real-name induction | D -> C | L3 real-name labels | 58.8 -> 70.6 | +11.9 | [4.4, 19.4] | 11/30 | 0.0043 |
| sequential matches Timmy | D -> C | L3 Timmy k=3 | 59.4 -> 58.8 | -0.6 | [-9.4, 9.4] | 26/25 | 1.0 |
| the suffix becomes a feature | base(m) -> E | probe, marked, trained fmt | 12.5 -> 93.8 | +81.2 | [70.8, 91.7] | 0/39 | <0.0001 |
| morphology helps induction | base(m) -> E | L3 Timmy k=3 | 40.6 -> 75.0 | +34.4 | [26.2, 43.1] | 5/60 | <0.0001 |
| and unseen-species induction | base(m) -> E | L3 held-out species | 33.3 -> 45.8 | +12.5 | [1.0, 24.0] | 11/23 | 0.058 |

*[Later, section 20: across three seeds the 'sequential loses yes/no' gap is -9.2 +- 15.2. A McNemar test on the items of two separately trained runs does not see seed noise (section 20.3); the p-values in this table are same-weights statements.]*

Most of section 8 survives. Three claims do not: the 9-point pairwise-reasoning loss of arm D (p = 0.36), arm C's weakness advantage over knowledge-only A (p = 0.27; the advantage over base holds), and the natural-label half of the replay claim (the symbol half holds, p = 0.012, but only pooled over the four datasets: no single 48-item dataset is significant). The "3 points of recall" arm D loses is six items and p = 0.031. Arm E's unseen-species induction gain is at the edge (p = 0.058).

**Null bands (EVAL-6).** 115 of the 336 arm x condition x level cells are inside their null band, meaning a gold-blind predictor with the same option preferences scores as well. They include the base model's 42.7 on held-out induction that the reviewer asked about: with predictions fixed and gold permuted the band is 25.0 to 43.8, because the base model's picks concentrate on one or two label positions and 96 items is too few to separate 40.6 from that. Every arm's held-out induction without context is inside its band except arm E (45.8, band 24.0 to 43.8, interval 35.4 to 56.2), so section 8.4's "31 to 47" morphology claim stands only at its upper end and only for E. Also inside their bands, without context: all twelve ladder levels of the base model (the base model does nothing on this universe without context that a gold-blind predictor would not); ten of the twelve for arm B (episodes alone inject no facts); arm A's Timmy (40.0, band 25.6 to 40.0) and novel choices; every arm's L5; and habitat induction for every arm but D (45.0, band 26.2 to 40.6). The full list is in `reports/ci_Qwen2.5-3B.md`.

**In the tables.** `curriculum_summary.py --ci` prints every cell as `acc [lo, hi]` and marks with * a cell inside its null band; `reports/ci_Qwen2.5-3B.md` has the same for all 26 levels and both conditions. Tables 8.1 to 8.3 are unchanged (their numbers are the Windows run, section 9.2); read them with the half-widths above.

## 12. Constructed response: what the models write (EVAL-3)

Date: 2026-09-14/15. Code: `scripts/gen_eval.py` (greedy decoding, 32 tokens, adapters merged as per 9.3), `scripts/gen_table.py` (matching and the tables, recomputed from the raw text without a model); data `results/gen_*.json`, generations `results/per_item/gen_*.jsonl`, tables `reports/gen_Qwen2.5-3B.md`; tracker experiment `gen_eval`; PLAN.md step 3.

Every number before this section is multiple choice: the model prefers an option or it does not. The production task is generative, and EVAL-3 asked what the models write. For each L1 and L3 item of the frozen ladder (1,056 items), without and with field-guide context, the model decodes after the prompt (bare) and after the prompt with the options listed as lettered lines (listed). `exact`: the first line equals the gold option. `fuzzy`: the option the generation names, by containment either way or a difflib ratio of 0.6 or more; nothing matched is unanswered. The section 4 merchant items (category, bank category, sells) were decoded for the bare 0.5B model with and without the fact in context, since the section 4 fine-tunes were never saved. All twelve decoded models are in the tables (the two plain-peft section 6 adapters were decoded last, with the loader fix of 9.4).

**Table 12.1: without context, generation (fuzzy) beside the two cloze rules (accuracy %)**

| arm | L1 recall bare: gen / mean / pmi | L1 recall trained fmt: exact / mean | Timmy k=3: gen / mean / pmi | k=4: gen / mean | held-out species: gen / mean | gen = mean rule, Timmy items |
|---|---|---|---|---|---|---|
| base | 0.0 (97.5 unanswered) / 18.1 / 7.5 | 0.0 (83.8 unanswered) / 11.9 | 4.4 (91.2 unanswered) / 36.9 / 30.6 | 0.6 / 25.0 | 2.1 / 40.6 | 6.2 |
| A knowledge | **85.6** / 25.0 / 43.8 | 100 / 100 | 5.0 (92.5 unanswered) / 40.0 / 40.0 | 4.4 / 34.4 | 1.0 / 24.0 | 6.2 |
| B episodes | 0.0 (98.8 unanswered) / 13.1 / 16.2 | 3.1 / 12.5 | 30.6 / 33.1 / 33.8 | 27.5 / 27.5 | 27.1 / 30.2 | 86.9 |
| C interleaved + replay | **85.0** / 20.6 / 16.2 | 100 / 100 | 60.6 / 58.8 / 40.6 | 51.9 / 50.6 | 29.2 / 30.2 | 92.5 |
| Cn interleaved | **85.6** / 21.2 / 27.5 | 100 / 100 | 58.1 / 60.0 / 43.8 | 59.4 / 58.1 | 31.2 / 33.3 | 90.6 |
| D sequential | 60.6 / 61.2 / 41.9 | 0.0 (writes the bare type; 92.5 fuzzy) / 96.2 | 60.0 / 59.4 / 43.8 | 55.6 / 54.4 | 30.2 / 34.4 | 94.4 |
| E morphology | **83.1** / 18.1 / 21.9 | 100 / 100 | 78.8 / 75.0 / 43.1 | 80.0 / 75.6 | 46.9 / 45.8 | 95.0 |
| section 6 3B adapter (unsloth) | **85.6** / 20.6 / 17.5 | 100 / 100 | 6.2 (90.6 unanswered) / 36.9 / 36.9 | 6.9 / 37.5 | 2.1 / 27.1 | 5.6 |
| section 6 3B adapter (plain peft) | **85.6** / 24.4 / 21.9 | 100 / 100 | 5.0 (94.4 unanswered) / 33.8 / 32.5 | 4.4 / 31.9 | 0.0 / 32.3 | 3.1 |
| section 6 0.5B adapter | **84.4** / 20.0 / 25.0 | 100 / 100 | 1.9 (95.0 unanswered) / 30.0 / 28.8 | 1.2 / 21.2 | 2.1 / 31.2 | 3.8 |

### 12.1 Bare-format recall was never 20%

Asked `Question: What type is Blaxorc?\nAnswer:` and left to write, every knowledge-trained arm names the right type about 85% of the time: A 85.6, C 85.0, Cn 85.6, E 83.1, the section 6 adapter 85.6. They write a sentence (`Blaxorc is a Voltrix-type.`, `Yoreel, being Tidewell-type, is weak to Umbrine.`) rather than the bare type name, which is why `exact` is 0 and why the cloze rule scored 20 to 25: the cloze compared eight bare type names as the *first* token after `Answer:`, and the model's first token is the species name. Sections 6.4 and 8 called this a format-shift problem with a 20-point floor and a 100-point trained-format ceiling; the generation number sits at 85 and is the honest one for "does the model know the fact when asked plainly". Arm D, the only arm that writes the bare type (`Verdane`), scores 60.6 by generation and 61.2 by cloze, consistent with each other, and 25 points below the sentence-writers; its phase 2 taught it to answer in one word.

### 12.2 Induction by generation matches the cloze for the arms that learned to answer

For arms B, C, Cn, D and E the fuzzy generation and the mean cloze rule pick the same option on 87 to 95% of Timmy items and land within 2 points of each other on every induction level (C 60.6 versus 58.8, E 78.8 versus 75.0, held-out species C 29.2 versus 30.2). The `pmi_dc` rule agrees with the generation on only 30 to 47% of the same items, which settles section 10's point from the other side: PMI is not what the model does. Two arms cannot produce a label at all: the base model and knowledge-only arm A write prose (`Ithump is a species of sea snail, ...`) on 91 to 93% of induction items, so their cloze scores on those levels (base 36.9, A 40.0) measure a preference they cannot act on. That includes the base model's with-context Timmy score: 48.8 by cloze, 1.9 by generation, 95.6% unanswered. The "Timmy from the weights" and "Timmy with the facts in context" numbers of section 8 are real for the episode arms and cloze-only for base and A.

### 12.3 Listed options: a different task

With the options listed as `A. FooFoo / B. blammo / C. zorp` and the model free to write, arm A induces at 61.9 (k=3) and the base model at 39.4, while the episode arms gain 5 to 10 points over their bare generation (C 67.5, E 88.8). This is the same picture as the letter-scoring column of section 10: when the label set is in front of it, a model that knows the types can pick, without ever having practised the episode format. The listed variant is the fair comparison for "can the model do the task at all"; the bare variant is the one for "did the training make the task its default". Bare recall goes the other way: listing the eight types makes the knowledge arms *worse* (C 28.1, E 41.9 against 85 bare), because the models then write a letter and letter scoring on Qwen2.5-3B is poor (section 10).

### 12.4 Merchants at 0.5B, and what is left

For the section 4 model with the fact in context, generation is better than cloze on the bank format (42.5 versus 14.2 in 4.3.2) and worse on the clean one (40.0 versus 69.2): the model writes the category name it read in the note about 40% of the time whatever the surrounding format, and the 14.2 cloze score on bank strings was the format prior again, as the casing check (PLAN step 5) concludes. `sells` by generation is 0 to 4% bare and 100% listed: asked to describe what a store sells without the options it writes something else entirely; with the options in view and the fact in context it copies the right one. Without the fact the 0.5B model answers nothing (unanswered 68 to 98%).

The two section 6 adapters saved by plain peft (loaded with the 9.4 fix) behave like the other knowledge-only models: bare recall 85.6 (3B) and 84.4 (0.5B) by generation against 20 to 24 by cloze, 100 in the trained format, and no induction labels at all (94 to 100% unanswered, with and without context). The 0.5B figure is the one to note: section 6.4 read its 20.0 cloze recall as the small model failing to learn the facts, and it knew 84% of them. Merchant items for fine-tuned models need the section 4 recipes re-run with saved adapters, which is its own step.

**What the report should say.** Bare-format recall is about 85% by generation for every knowledge arm and the 20% cloze figure should be read as a scoring artifact, not a knowledge gap; the induction claims of section 8 hold by generation for the episode arms; base and A cannot produce labels, only prefer them. Generation joins `mean` and `hybrid` as the third line for every table that reports a level a user would see as text.
## 13. Tokenizer casing check: the bank-string failure is not the tokens (MODEL-4)

Date: 2026-09-14. Code: `scripts/exp_casing.py`; data `results/casing.json`, per-item records `results/per_item/casing_Qwen2.5-0.5B.*.jsonl`; tracker experiment `casing`; PLAN.md step 5.

QUESTIONS.md MODEL-4 proposed that the merchant bank-string failure in section 4 (14.2% with the fact in context, chance 8.3) is a tokenizer artifact: the training text carries `Kelvarro Co`, the statement carries `KELVARRO CO`, and a cased BPE tokenizer splits the two into different pieces, so nothing learned about one can reach the other. The test: render every bank string three ways and score the same items.

**Token survival.** Share of a clean merchant name's tokens that also appear in the rendered name, over the 120 merchants:

| tokenizer | clean | upper (bank string) | title-cased | name restored |
|---|---|---|---|---|
| Qwen2.5 (0.5B and 3B) | 100 (3.6 tokens/name) | **10.6** (4.5 tokens/name) | 95.2 | 100 |
| all-MiniLM-L6-v2, uncased WordPiece | 100 (3.3) | 96.2 | 96.2 | 100 |

The 10.6% confirms the mechanism the reviewer measured (12.6% with a slightly different count): for a cased tokenizer the uppercase name is a different string. Title-casing the whole statement brings 95.2% of the tokens back; restoring only the name brings all of them.

**Table 13.1: bank_category (12-way, chance 8.3) under the three renderings**

| model, condition | upper | title | name restored |
|---|---|---|---|
| Qwen2.5-0.5B, bare (section 4 "base") | 8.3 | 8.3 | 8.3 |
| Qwen2.5-0.5B, fact in context (section 4 "incontext") | 14.2 | 15.8 | 15.0 |
| MiniLM zero-shot, 96 trained merchants | 7.3 | 7.3 | 7.3 |
| MiniLM ft_subword (section 4.3.1 recipe, retrained), 96 trained merchants | 67.7 | 67.7 | 68.8 |
| MiniLM ft_subword, 24 held-out merchants | 4.2 | 4.2 | 4.2 |

**What it says.** The hypothesis is wrong for the LLM. Giving the 0.5B model the merchant name in exactly its trained tokens, inside the bank string, moves the with-context score from 14.2 to 15.0, one item; title-casing the whole line gives 15.8. The 120-item half-width at 15% is about 6 points, so these are the same number. The model has the fact in the prompt and the name in its trained form and still cannot map `POS DEBIT Kelvarro Co #0412 AUSTIN TX 03/14` to a spending category, while the same model with the same note answers `Merchant: Kelvarro Co` at 69.2 (4.3.2). What defeats it is the transaction format itself: the few-shot prefix of bank lines, the store number, city and date around the name, and the "Spending category:" cue, at a model size that does not read past that noise. The embedding side is the control the hypothesis predicted: an uncased tokenizer is indifferent to the rendering (67.7 / 67.7 / 68.8), and its 70.8% transfer in 4.3.1 owes nothing to casing either way.

So section 1's recommendation stands with a different reason: normalise bank strings before the model sees them because the *format* is noise for small models, not because of the tokenizer. Case-normalisation alone will not recover the bank format; stripping the string down to the merchant name (or training on realistic renderings, PLAN step 21) is what the numbers point at. The second half of MODEL-4, whether the 70.8% transfer survives on a cased encoder (bge-base, Qwen3-Embedding), stays with step 13.

## 14. Prompt distillation: the with-context ceiling is a ranking, not a distribution (BASE-3)

Date: 2026-09-15. Code: `scripts/exp_distill.py` (arm P), `scripts/diag_distill_teacher.py`; data `results/curriculum_Qwen2.5-3B_P.json`, per-item `results/per_item/curriculum_Qwen2.5-3B_P.*.jsonl`, `results/distill_teacher_Qwen2.5-3B.json`; the P column of `reports/scorers_Qwen2.5-3B.md` and `reports/ci_Qwen2.5-3B.md` (with the base to P and C to P pairs); tracker `curriculum_v2`, arm P, method `prompt_distill_kl`; PLAN.md step 7.

BASE-3 noted that the with-context numbers are the ceiling everywhere (98.8 recall, 48.8 Timmy for the base 3B) and asked whether that ceiling can be distilled into the weights. Arm P is the direct version: the same LoRA (rank 64, alpha 128, all linear layers), the same 800 steps at batch 16 and learning rate 1e-4, arm C's mixture (45% knowledge texts, 40% episodes, 15% replay) and seed. For knowledge texts and episodes the loss is the KL divergence at temperature 2 (scaled by 4) between the student's next-token distribution on the bare text and the teacher's on the same tokens with the field-guide entries of every species named in the text prepended; the teacher is the same base model with the adapter disabled, so the student is trained to do without the context what the teacher does with it. Replay episodes keep the hard cross-entropy. The run took 26 minutes at 566 tokens/s with two forward passes per batch (17.2 GiB); a first attempt died at step 400 with a CUDA execution error while a Unity install was running on the same machine and was rerun clean. The prediction on the PLAN row was bare recall toward 98.8 with better ICL retention than hard-label episodes.

**Table 14.1: arm P beside base, A and C (accuracy %, without / with field-guide context)**

| level | base | A knowledge | C interleaved + replay | P distillation |
|---|---|---|---|---|
| recall, trained format | 11.9 / 100 | 100 / 100 | 100 / 100 | **13.8** / 100 |
| recall, bare format | 18.1 / 98.8 | 25.0 / 40.6 | 20.6 / 98.8 | 20.0 / 100 |
| yes/no | 42.5 / 100 | 96.2 / 100 | 86.2 / 100 | 63.8 / 100 |
| pair | 51.2 / 81.2 | 91.2 / 96.2 | 73.8 / 70.0 | 46.2 / 88.8 |
| Timmy k=3 | 36.9 / 48.8 | 40.0 / 43.1 | 58.8 / 76.2 | 33.1 / 42.5 |
| k=4 | 25.0 / 41.2 | 34.4 / 39.4 | 51.2 / 69.4 | 26.2 / 37.5 |
| held-out species | 40.6 / 43.8 | 24.0 / 32.3 | 30.2 / 75.0 | 33.3 / 41.7 |
| ICL suite symbol | 60.4 | 51.6 | 79.2 | **76.6** |
| ICL suite natural | 84.4 | 80.8 | 88.0 | 84.4 |
| ppl | 8.54 | 21.37 | 14.7 | **8.64** |

### 14.1 Nothing was injected

Trained-format recall is 13.8 against the base model's 11.9 (paired difference +1.9, 95% CI [-1.2, 5.6], McNemar p = 0.51); bare recall 20.0 against 18.1. Every induction level sits inside its permutation null band from section 11 (Timmy 33.1 in 26.2 to 40.6, k=4 26.2 in 20.0 to 33.1, held-out species 33.3 in 24.0 to 41.7). With the entries in context P is the base model: recall 100, Timmy 42.5 against 48.8 (p = 0.21). Per item, P predicts the same type on 155 of the 160 trained-format recall items where the base model did so on 126; the constant predictor of section 10 got more constant. The general-text perplexity is 8.64 against 8.54, where every other arm moved it to between 12 and 21: the adapter barely changed the model. Against arm C on the same items the trained-format gap is 86 points (138 items flip one way, none the other).

Two things did move. The yes/no level went from 42.5 to 63.8 (+21.2, CI [2.5, 40], p = 0.04), and the symbol-label ICL suite from 60.4 to 76.6 (+16.1, CI [7.8, 25], p = 0.0004), within 3 points of arm C's 79.2 (p = 0.34). The ICL gain is what the 15% hard-label replay does on its own: section 15.2 measures the same thing in arm D's first phase, knowledge plus replay and no episodes, at 70.8 by step 200. Replay was the only stream in arm P with a hard target, and it is the only stream that left a mark.

### 14.2 Why: the teacher puts 6% on the answer

`scripts/diag_distill_teacher.py` takes the declarative training sentence of each of the 136 trained species (`{name} is a {type}-type creature. It lives in ...`) and reads the probability of the type's first token at its position, for the base model on the bare sentence, the teacher (base model with the species' entry prepended, exactly as in training), the same teacher at temperature 2, and the P and A adapters on the bare sentence:

**Table 14.2: probability of the type token in the training sentence (mean over 136 species)**

| model, input | p(type token) | at T = 2 | argmax is the type (%) | KL to the tempered teacher, scaled by T² |
|---|---|---|---|---|
| base, bare sentence | 0.000 | 0.000 | 0.0 | 2.17 |
| teacher: base + field-guide entry | **0.059** | **0.003** | 20.6 | (target) |
| P, bare sentence | 0.002 | 0.000 | 0.0 | **0.28** |
| A, bare sentence | 1.000 | 0.969 | 100 | 30.3 |

The model that recalls at 98.8 with the entry in context, when the evaluation scores eight type names against each other, puts 6% of its next-token mass on the type after `Blaxorc is a` with `Blaxorc: Voltrix-type, weak to ...` two lines above it; four times in five its most likely next word is something else. The 98.8 is a ranking: among the eight options the right one wins. As a distribution to imitate it is 6% fact and 94% ordinary English, and after tempering at T = 2 the fact is 0.3% of it. The student did what the loss asked: its KL to the tempered teacher fell from 2.17 to 0.28, the closest match in the table, and that match carries no fact. Arm A, whose hard labels put all the mass on the type token, sits 30 nats from the teacher and knows every fact. Context distillation transfers the teacher's uncertainty faithfully, and here the teacher's uncertainty is the whole problem. (The same diagnostic on the cloze prompt `Question: What type is Blaxorc?\nAnswer:` shows arm A putting 0.7% on the type as the *first* token, since it writes the species name first, as section 12.1 saw in its generations; the 100% trained-format recall is also a ranking.)

### 14.3 What BASE-3 should now say

The with-context ceiling of sections 6 and 8 is a relative one: given the entry, the model ranks the options correctly; it does not produce the fact with any confidence, so there is no confident distribution to distil. A version that could work replaces the full-vocabulary target with the teacher's distribution renormalised over the option set (the evaluation's own scoring, where the teacher is at 98.8), or a teacher that has the options listed in its prompt and is scored by letter (section 10 showed the base model picks well in that format), at temperature 1; the limit of that direction is hard labels from the teacher's argmax over options, which is arm A with a different label source. That is queued as PLAN row 27. If it also fails, BASE-3 closes as answered in the negative: the ceiling was never a distribution.

**What the report should say.** Soft-label prompt distillation at T = 2 from the base model reading the field guide injected nothing: recall, induction and perplexity are the base model's, and the 16-point ICL gain is the hard-label replay's. The cause is measured, not guessed: the teacher assigns 6% to the fact token in free text, 0.3% tempered, and the student matched that distribution to within 0.28 nats. "With context is the ceiling" means the model can rank the options given the entry, not that it would say the fact; distillation needs the ranking, not the distribution.
## 15. Learning curves: facts by step 200, ICL damage by step 200, ARC-Easy down and staying down (TRAIN-3)

Date: 2026-09-15. Code: `PERIODIC=200 uv run python scripts/exp_curriculum.py {A,C,D}`, `scripts/curves.py`; data `results/curriculum_Qwen2.5-3B_{A,C,D}_p200.json`, logs `results/periodic_{A,C,D}.log`, tables `reports/curves_Qwen2.5-3B.md`; forgetting proxy `data/processed/known_facts_v1.json` (`ai_experiments.items`); tracker condition `periodic`; PLAN.md step 11.

Every arm before this section was measured once, at step 800. TRAIN-3 asked when the facts arrive and when the in-context skill breaks. Arms A (knowledge only), C (interleaved: 45% knowledge, 40% episodes, 15% replay) and D (sequential: 85% knowledge with 15% replay for 400 steps, then 85% episodes with 15% replay) were retrained with the section 8 recipe and seed and scored at step 0 and every 200 steps on a fixed subsample: every fourth ladder item (40 per 160-item level, 24 held-out-species items), every second ICL suite item (192), and a new forgetting proxy: 200 four-option ARC-Easy test questions in the ladder's cloze format (level `K_arc_easy`), facts the base model answers at 73.5% and no arm trains on. A mid-training point costs about 1.5 minutes. The step 800 row is the full final evaluation; restricted to the subsample items it is within 4 points of the full-set figure on every level except C's weakness (45.0 on the 40 items, 60.0 on all 160), so the rows read across. Half-widths: about 15 points on a 40-item level, 7 on the ICL mean, 6.5 on ARC-Easy.

**Table 15.1: the curves, arms A / C / D (accuracy %; ppl is perplexity on the section 4 general paragraph)**

| step | recall, trained fmt | Timmy k=3 | ICL suite symbol | ARC-Easy (K) | ppl |
|---|---|---|---|---|---|
| 0 (base) | 12.5 / 12.5 / 12.5 | 42.5 / 42.5 / 42.5 | 63.5 / 63.5 / 63.5 | 73.5 / 73.5 / 73.5 | 8.5 / 8.5 / 8.5 |
| 200 | **100** / 10.0 / **95.0** | 35.0 / 47.5 / 20.0 | **49.0** / 66.7 / 70.8 | **62.0 / 62.5 / 57.5** | 14.2 / 11.8 / 13.3 |
| 400 | 100 / **92.5** / 100 | 37.5 / 35.0 / 32.5 | 52.1 / 65.6 / 66.7 | 61.5 / 52.5 / 63.5 | 15.8 / 38.7 / 12.9 |
| 600 | 100 / 100 / 95.0 | 42.5 / 45.0 / 37.5 | 49.0 / 72.9 / 75.0 | 61.0 / 56.0 / 61.5 | 19.1 / 20.3 / 14.1 |
| 800, all items | 100 / 100 / 92.5 | 40.6 / **61.2** / **54.4** | 47.9 / 78.1 / 80.8 | 66.0 / 59.0 / 64.0 | 18.1 / 30.4 / 13.5 |

Per-metric tables for every level, with the step at which each arm first came within 2 points of its final value, are in `reports/curves_Qwen2.5-3B.md`.

### 15.1 Facts land in the first 200 steps, or as soon as the knowledge stream has covered them

Arm A is at 100 trained-format recall at step 200, the first checkpoint, and D at 95 on an 85% knowledge mix. Arm C, at 45% knowledge, is still at the base model's 10 at step 200 and at 92.5 by step 400. By step 200 C has processed about as many knowledge sequences as A had by step 90, so the 160 facts are absorbed somewhere between 90 and 180 knowledge-only steps, and the transition is not gradual at this resolution: no arm was caught part-way. The rest of training does nothing for recall. Arm D's second phase teaches the bare format that section 12.1 saw in its generations: its bare-format recall on the subsample goes from 25 at step 400 to 67.5 at step 600 once the episodes, which answer in one word, start.

### 15.2 A's ICL loss is complete at step 200, and 15% replay prevents it from the start

Arm A's symbol-label ICL mean falls from 63.5 to 49.0 by step 200 and sits at 48 to 52 for the remaining 600 steps; the final 47.9 (50.5 in section 8) was already there at the first checkpoint. C and D never fall: 66.7 and 70.8 at step 200, 78.1 and 80.8 at the end. D's first phase is knowledge plus replay with no induction episodes, so what holds ICL up while the same facts are being written is the 15% replay stream, the ICL-suite episodes from other datasets, on its own; the episodes then raise it further. Section 8 credited episodes and replay together and could not separate them; section 11 showed the natural-label half of the replay claim did not survive its interval. The curve settles the symbol half: the damage a knowledge-only run does to in-context labelling is done in the first quarter of training, and a small replay fraction from the first step is enough to avoid it, at no cost to recall (D 95 at step 200 against A 100, same items, within noise).

### 15.3 ARC-Easy drops 11 to 16 points in the first 200 steps and stays down, replay or not

All three arms lose general knowledge at the first checkpoint: 73.5 to 62.0 (A), 62.5 (C), 57.5 (D), paired on the same 200 items, and none recovers: 66.0, 59.0 and 64.0 at the end. The drop is about two half-widths in each arm and the same in all three, so it is not noise. It is also indifferent to the data mix: D, whose first phase carries 15% replay, drops the most, and C's interleaved episodes do nothing for it. Replay protects the format that is replayed, not the facts the model already had. The general-text perplexity moves with it (8.5 to 12 or 14 at step 200 in every arm), so the two forgetting proxies agree that the cost of writing 160 facts into a rank-64 adapter at this learning rate is paid early and is about ten ARC-Easy points. Whether general-text replay (a few percent of pretraining-style text in the mix) buys that back the way ICL replay protects ICL is the natural next run and is queued as step 25 (TRAIN-7).

### 15.4 Induction from the weights shows up in the last 200 steps, and the subsample is too small to watch it

On the 40 Timmy items, arm C reads 47.5, 35.0, 45.0 at steps 200, 400, 600 and 57.5 on the same 40 items at step 800 (61.2 on all 160); D reads 20.0, 32.5, 37.5 and then 50.0 (54.4). The skill both arms end with is not visible at step 600, but a move of 12 points on 40 items is five items and inside the half-width, so two readings are open: the induction skill arrives in the final quarter (D's monotone climb after its episodes start at step 400 fits that), or it grows steadily under noise. Any repeat should score the induction levels in full (160 items, about one extra minute per point) and subsample only the rest; the ICL mean, on 192 items, is the clean curve here.

### 15.5 Perplexity on one paragraph is not a curve

The `ppl` column is the log-likelihood of one 249-word paragraph. Arm C reads 8.5, 11.8, 38.7, 20.3, 30.4 across its checkpoints, and this rerun ends at 30.4 where the section 8 run of the same recipe ended at 14.9, while every accuracy metric of the rerun matches section 8 within noise (Timmy 61.2 against 59.4, ICL 78.1 against 78.1, held-out species 29.2 against 31.2). A single paragraph swings by a factor of two between checkpoints of one run and between two runs of one recipe; it can say that perplexity went up (it did, in every arm, at step 200) and nothing finer. Perplexity on a held-out corpus slice of a few thousand tokens replaces it in step 24.

**What the report should say.** Facts are in the adapter by step 200 at a 100% knowledge mix and by step 400 at 45%; A's ICL loss is already complete at step 200, and 15% replay of the ICL suite from the first step prevents it entirely; ARC-Easy loses about ten points in the same first 200 steps in every arm, replay included, and never recovers, which is the first direct measure of what the injection costs in knowledge the model already had; induction from the weights arrives late, and mid-training induction numbers need the full 160 items. The training runs are 200 steps too long for recall and the right length for induction, so shortening them is not free.

## 16. Perplexity on a corpus slice: the paragraph was the noise (TRAIN-3)

Date: 2026-09-15. Code: `scripts/corpus_ppl.py`, `ai_experiments.items` (`data/processed/corpus_ppl_v1.json`), `ai_experiments.scoring.corpus_perplexity`; data `results/corpus_ppl_Qwen2.5-3B.json`, `results/corpus_ppl_Qwen2.5-0.5B.json`, tables `reports/corpus_ppl_Qwen2.5-3B.md`, `reports/corpus_ppl_Qwen2.5-0.5B.md`; tracker experiment `corpus_ppl`, one run per base model; PLAN.md step 24.

Section 15.5 found that the general-text perplexity, computed on one 249-word paragraph, moved by a factor of two between checkpoints of one run and between two runs of the same recipe (arm C: 14.9 and 30.3) while no accuracy metric moved with it. This step replaces the paragraph with a frozen slice of the WikiText-2 test split: 29 body paragraphs chosen with a seed, detokenised, 3,637 words and 4,972 scored Qwen tokens, hashed and recorded in the tracker config like the item sets. From this commit on every evaluation reports it as `L7_ppl_wikitext`, at the final step and at every periodic point, with the standard error of the mean over paragraphs; `L7_ppl_general` stays in the outputs for continuity and is not cited again. The saved adapters were scored in one process per base model (the base loaded once, every adapter attached with peft and switched in turn, about three seconds each), which also gives the paired number the paragraph never could: the per-paragraph difference in mean negative log-likelihood between two models and its standard error over the 29 paragraphs.

**Table 16.1: WikiText-2 slice perplexity of every saved adapter (3B; nats are per token; the last column is the one-paragraph number from sections 8, 9 and 15)**

| model | WikiText ppl | NLL minus base, paired, +- se | paragraph ppl |
|---|---|---|---|
| base | 10.61 | | 8.54 |
| A knowledge only | 37.78 | +1.405 +- 0.164 | 21.48 |
| A, periodic rerun (15) | 34.08 | +1.273 +- 0.131 | 17.94 |
| B episodes only | 15.39 | +0.368 +- 0.039 | 12.19 |
| C interleaved + replay | 23.18 | +0.863 +- 0.096 | 14.85 |
| C, periodic rerun (15) | 22.80 | +0.838 +- 0.097 | 30.26 |
| Cn interleaved, no replay | 28.42 | +1.083 +- 0.104 | 20.36 |
| D sequential + replay | 23.85 | +0.857 +- 0.090 | 14.45 |
| D, periodic rerun (15) | 22.59 | +0.814 +- 0.093 | 13.63 |
| E morphology | 23.90 | +0.900 +- 0.111 | 15.81 |
| P distillation (14) | 11.50 | +0.086 +- 0.016 | 8.64 |
| section 6.4, 3B unsloth LoRA | 29.69 | +1.149 +- 0.148 | 17.16 |
| section 6.4, 3B transformers LoRA | 34.89 | +1.326 +- 0.164 | 20.34 |

The 0.5B base scores 16.93 (paragraph 16.24) and its section 6.4 adapter 210.35, +2.651 +- 0.145 nats (paragraph 132.5).

### 16.1 Two runs of one recipe agree to 0.03 nats; the paragraph put them a factor of two apart

The section 8 run of arm C and its section 15 rerun differ by -0.024 +- 0.022 nats on the slice (1.1 standard errors; perplexity 23.2 against 22.8) where the paragraph read 14.85 against 30.26. The D pair differs by -0.044 +- 0.018 and the A pair by -0.132 +- 0.049, so run-to-run variation in what a recipe costs is a few hundredths of a nat with replay in the mix and about a tenth without. The base model's own perplexity on the 29 paragraphs runs from 4.9 to 32.7: one paragraph is a single draw from that spread, and an adapter's effect on it is confounded with whichever words the paragraph happens to contain. Section 15.5's reading stands and is now measured: the swing was the instrument.

### 16.2 What each recipe costs in general text, in order

Knowledge text alone costs the most: arm A adds 1.40 nats per token and takes the base model from 10.6 to 37.8. Adding episodes without replay (Cn) leaves 1.08; adding the 15% ICL replay as well (C, D, E) leaves 0.81 to 0.90, and those three are indistinguishable pairwise (D minus C -0.005 +- 0.034, E minus C +0.037 +- 0.027). Episodes alone (B) cost 0.37, and arm P, the adapter that section 14 said barely changed the model, costs 0.086 +- 0.016, five standard errors from zero. Paired against C, A is +0.542 +- 0.079 and Cn +0.221 +- 0.028. Part of the difference between A and the mixed arms is exposure, not protection: A sees every one of its 12,800 sequences as knowledge text and C 45% of them, so C writes the same facts with 2.2 times fewer knowledge sequences at 1.6 times less damage, and what the replay stream buys on its own is the Cn-to-C step, 0.22 nats. Against ARC-Easy at step 800 (section 15.3: A 66.0, C 59.0, D 64.0, all within one half-width of each other) the two forgetting proxies agree that every arm pays and that the payment is made in the first 200 steps, and the slice can additionally rank the arms, which 200 four-option questions cannot. The two section 6.4 adapters, the same recipe under two trainers, differ by 0.256 +- 0.039 nats, a difference the accuracy tables in 9.4 kept inside the noise floor; the 0.5B adapter's 2.65 nats is the language model coming apart, which section 6.4 saw as 132.5 and could not size.

### 16.3 What changes downstream

The corpus number exists only for the final adapters: no mid-training checkpoints were saved, so the TRAIN-3 perplexity curve is still open and arrives with step 25, whose run reports `L7_ppl_wikitext` at every periodic point. Step 10 (three seeds) will show whether the 0.03 to 0.13 nat run-to-run spread measured here on two pairs holds. The paired standard error of about 0.02 to 0.03 nats between adapters of one base sets what a "no cost" claim can mean from now on: an arm that matches C within 0.05 nats is at C's cost.

**What the report should say.** The one-paragraph perplexity is retired: on a frozen 4,972-token WikiText-2 slice, two runs of arm C agree to 0.03 nats where the paragraph put them at 14.9 and 30.3. Every knowledge-injecting arm costs general-text likelihood, from 0.8 nats per token with ICL replay in the mix (C, D, E, perplexity 10.6 to 23) to 1.4 without episodes or replay (A, to 38); episodes alone cost 0.37 and distillation 0.09. The replay stream's own contribution is 0.22 nats (Cn to C); the rest of C's advantage over A is seeing fewer knowledge sequences.
## 17. General-text replay keeps the language model and ARC-Easy, and takes it out of induction (TRAIN-7)

*[Later: the induction drop (Timmy 47.5 against 61.2) is one seed each, inside section 20's spread for a single-run difference.]*

Date: 2026-09-15. Code: `PERIODIC=200 uv run python scripts/exp_curriculum.py Cg`, stream `icl_suite.general_replay_texts`; data `results/curriculum_Qwen2.5-3B_Cg_p200.json`, per-item `results/per_item/curriculum_Qwen2.5-3B_Cg_p200.*.jsonl`, curves `reports/curves_Qwen2.5-3B.md`, adapter `models/adapters/curriculum_Qwen2.5-3B_Cg_p200_lora`; tracker `curriculum_v2`, arm Cg; PLAN.md step 25.

Section 15.3 found every arm losing 11 to 16 ARC-Easy points in its first 200 steps whether or not it replayed ICL episodes, and section 16 put the same loss in nats: 0.8 to 1.4 per token of general text. Replay protected what was replayed and nothing else. TRAIN-7 asks whether replaying general text protects general knowledge the same way. Arm Cg is arm C with 5% of its sequences turned into pretraining-style text: paragraphs of the WikiText-2 *train* split (the perplexity slice of section 16 is from the test split, so the two never meet), detokenised, cut to 70 words, about 95 tokens each, trained with the full-sequence loss like the knowledge texts. The 5% came out of the episode stream (episodes .40 to .35; knowledge .45 and ICL replay .15 unchanged), same seed, learning rate and 800 steps, scored every 200 steps on the section 15 subsample. Because knowledge texts are 24 tokens long and general paragraphs 95, five percent of sequences is a quarter of the loss:

**Table 17.1: what each stream is, by sequences and by loss-bearing tokens (arm Cg, 12,800 sequences; arm C from section 8.1 in brackets)**

| stream | sequences | tokens seen | loss-bearing tokens | share of the loss |
|---|---|---|---|---|
| K knowledge texts | 5,843 (45.6%) | 147,582 | 141,739 | 62.5% [84%] |
| E episodes | 4,394 (34.3%) | 600,720 | 17,058 | 7.5% [12%] |
| R ICL replay | 1,920 (15.0%) | 406,134 | 7,132 | 3.1% [4%] |
| G general text | 643 (5.0%) | 61,499 | 60,856 | 26.8% |

Training took 29 minutes at 691 tokens per second (8.7 GiB). The per-stream counts are new in the results (`tok_*`, `lb_*`), for step 9.

**Table 17.2: curves, arm C (section 15 rerun) / arm Cg (accuracy %; ppl is the section 16 slice)**

| step | recall, trained fmt | Timmy k=3 | ICL suite symbol | ARC-Easy (K) | WikiText ppl |
|---|---|---|---|---|---|
| 0 (base) | 12.5 / 12.5 | 42.5 / 42.5 | 63.5 / 63.5 | 73.5 / 73.5 | 10.61 / 10.61 |
| 200 | 10.0 / 30.0 | 47.5 / 22.5 | 66.7 / 69.8 | 62.5 / **66.5** | . / 11.50 |
| 400 | 92.5 / 97.5 | 35.0 / 27.5 | 65.6 / 77.1 | 52.5 / **69.0** | . / 11.83 |
| 600 | 100 / 100 | 45.0 / 47.5 | 72.9 / 80.2 | 56.0 / **67.5** | . / 11.47 |
| 800, all items | 100 / 100 | 61.2 / **47.5** | 78.1 / 81.2 | 59.0 / **71.0** | 22.80 / **11.60** |

**Table 17.3: final evaluation, Cg beside C, with the paired difference on the same items (bootstrap 95% CI, McNemar p)**

| level | C (15) | Cg | Cg minus C | | Cg minus base |
|---|---|---|---|---|---|
| recall, trained format | 100 | 100 | 0 | | |
| recall, bare format | 19.4 | 18.8 | -0.6 [-3.8, 2.5], p = 1 | | |
| yes/no | 77.5 | 73.8 | -3.8 [-16, 10], p = 0.72 | | +31.2, p = 0.002 |
| pair | 80.0 | 58.8 | **-21.2 [-36, -6], p = 0.014** | | +7.5, p = 0.55 |
| Timmy k=3 | 61.2 | 47.5 | **-13.8 [-22.5, -4.4], p = 0.004** | | +10.6, p = 0.03 |
| k=4 | 53.8 | 46.9 | -6.9 [-15.6, 1.9], p = 0.15 | | |
| weakness | 60.0 | 50.0 | **-10.0 [-18.1, -2.5], p = 0.023** | | +18.1, p = 0.001 |
| habitat | 29.4 | 31.9 | +2.5 [-5.6, 10], p = 0.63 | | |
| held-out species | 29.2 | 30.2 | +1.0 [-7.3, 9.4] | | |
| ICL suite symbol | 78.1 | 81.2 | +3.1 [-2.1, 8.3], p = 0.35 | | +20.8, p = 1e-6 |
| ICL suite natural | 87.5 | 89.1 | +1.6 [-2.6, 5.7] | | +4.7, p = 0.06 |
| ARC-Easy | 59.0 | 71.0 | **+12.0 [6.5, 17.5], p = 4e-5** | | -2.5 (base 73.5, no per-item file) |
| WikiText ppl | 22.80 | 11.60 | -0.75 nats | | +0.09 nats |

### 17.1 The language model stays where it was

Arm Cg's perplexity on the section 16 slice is 11.50 at step 200, 11.83, 11.47, 11.60 at the end: 0.09 nats per token above the base model, where arm C's rerun ends 0.84 above it and every section 16 arm with a knowledge stream is between 0.8 and 1.4. The curve is flat from the first checkpoint, so the cost that every other arm pays in its first 200 steps (section 15.3, 16.2) is not paid here at all. A quarter of the loss on 643 WikiText paragraphs holds a 3B model's general-text likelihood at base level while 141,739 tokens of knowledge text are written into the same adapter.

### 17.2 ARC-Easy holds, and the facts arrive no slower

ARC-Easy is 66.5 at step 200 where C read 62.5, A 62.0 and D 57.5; it ends at 71.0 against C's 59.0 on the same 200 items (+12.0, CI [6.5, 17.5], p = 4e-5) and 2.5 points under the base model's 73.5, inside the level's half-width. Against arm A on the same items it is +5.0 [0.5, 9.5]. So section 15.3's rule holds with the sign it predicted: replay protects what is replayed, and general text is what ARC-Easy needs. Recall in the trained format is 30.0 at step 200 (C: 10.0), 97.5 at 400 (C: 92.5) and 100 from step 600; the knowledge stream lost no sequences to the general text and the facts land at least as early. The ICL suite is the best of any arm, 81.2 symbol and 89.1 natural, +3.1 and +1.6 over C (both within noise) and +20.8 over the base.

### 17.3 What paid: induction from the weights

The three levels where arm C beat every other arm in section 8 came down: Timmy k=3 from 61.2 to 47.5 (-13.8, p = 0.004), weakness induction from 60.0 to 50.0 (p = 0.023), pairwise same-type from 80.0 to 58.8 (p = 0.014); k=4 and habitat moved within noise. Cg still induces above the base model (Timmy +10.6, weakness +18.1, yes/no +31.2, all p < 0.05), but it is arm C at about two thirds of its induction gain. Two things changed at once and this run cannot separate them: the episode stream went from 40% to 35% of sequences and, more to the point, from 12% to 7.5% of the loss-bearing tokens, because the 61k general-text tokens outweigh the 17k episode tokens three and a half to one (Table 17.1); and general text is a new signal competing for the same rank-64 adapter. Section 8.1 already noted that the episode signal producing the induction gains was the thinnest stream by loss, and it just got thinner. The clean follow-up is the same 5% taken from the knowledge stream instead (K .40, E .40, R .15, G .05), or the token-weighted mixture that step 9 builds; both are one run each. The mid-training Timmy readings (22.5, 27.5, 47.5 on 40 items) say what 15.4 said: the induction skill arrives in the second half and the 40-item subsample cannot resolve it.

### 17.4 What TRAIN-7 now says

General-text replay at 5% of sequences (27% of the loss) buys back the whole general-text perplexity cost and about all of the ARC-Easy loss of arm C, at no cost to recall or to the ICL suite, and at a measured cost to induction that is confounded with the episode share it displaced. The two forgetting proxies of section 15 now disagree in a useful way: perplexity says nothing was forgotten (0.09 nats), ARC-Easy says 2.5 points that the interval cannot see. For the merchant use case, where the model's existing knowledge is the product, the replay fraction belongs in the recipe; for the induction result it should come out of the knowledge stream, not the episodes, and that is step 9's job.

**What the report should say.** Replaying 5% pretraining-style text (a quarter of the loss) beside arm C's mixture keeps the WikiText perplexity at the base model's (+0.09 nats against C's +0.84) and ARC-Easy at 71.0 against C's 59.0 (+12 paired, p = 4e-5; base 73.5), from the first checkpoint on, with recall at 100 and the ICL suite at its best (81.2). Induction from the weights fell by 10 to 21 points on three levels, which is either the episode share it displaced (12% to 7.5% of the loss) or competition for the adapter; one more run with the 5% taken from the knowledge stream decides.

## 18. Prompt distillation v2: the ranking distils, and what it distils is a format (BASE-3)

Date: 2026-09-15. Code: `scripts/exp_distill_v2.py` (arm P2), `ai_experiments.scoring.option_logprobs_batched`, `scripts/diag_distill_options.py`; data `results/curriculum_Qwen2.5-3B_P2.json`, per-item `results/per_item/curriculum_Qwen2.5-3B_P2.*.jsonl`, `results/distill_options_Qwen2.5-3B.json`; adapter `models/adapters/curriculum_Qwen2.5-3B_P2_lora`; tracker `curriculum_v2`, arm P2, method `prompt_distill_options`; PLAN.md step 27.

Section 14 ended with a diagnosis: the base model reading the field guide puts 6% of its next-token mass on the fact, so its 98.8 with-context recall is a ranking among the options, not a distribution, and distilling the distribution (arm P) transferred nothing. Arm P2 distils the ranking. Every distilled example is a question with an option set; the target is the teacher's log-probability of each option string, with the entry in front of the prompt, renormalised over the set at temperature 1; the student is scored on the bare prompt the same way and trained with the KL between the two, which is the evaluation's own arithmetic turned into a loss. The fact stream is the five question-answer templates that arm A trains on as text (`universe._DESC` 8, 9, 10, 12), one attribute varied per question and the options being the template filled with each candidate value, so the option strings are exactly the sentences arm A sees: 680 questions over 136 species (type, weakness, habitat, region, diet). Episodes are distilled the same way with their labels as the option set, the teacher reading the entries of every species in the prompt; the ICL replay keeps its hard cross-entropy. Mixture .45 / .40 / .15, 800 steps of 16, the usual LoRA, 57 minutes at 365 tokens per second (two forwards of up to 64 rows per micro-batch, 11.9 GiB).

**A discarded run.** The first run scored the options in left-padded batches and read 11.9 trained-format recall, the base model's number, with the teacher's option ranking inside the mixed-length training batches at 66% where it is 92% scored alone. The cause is unsloth's training-mode forward: with gradient checkpointing on, a left-padding attention mask is not applied and the real tokens attend the pads (per-token log-probs off by up to 15 nats; in eval mode the mask is honoured, and plain transformers honours it in both). Right padding is exact under causal attention whatever the mask path does; the shared scorer now right-pads, takes the final hidden states from unsloth and applies the LM head only at the option tokens. Two smaller facts from the same probe carry over: batch shape moves bf16 log-probs by 0.03 nats per token on average (0.4 at most), in plain transformers as much as in unsloth, so an option's log-prob sum is stable to about half a nat across batch layouts; and every evaluation in this repo right-pads, so nothing before this section is affected.

**Table 18.1: arm P2 beside base, A, C and P (accuracy %, without / with field-guide context; A and C are the section 8 runs as in Table 14.1, ARC-Easy from their section 15 reruns)**

| level | base | A knowledge | C interleaved + replay | P distillation (14) | P2 option distillation |
|---|---|---|---|---|---|
| recall, trained format | 11.9 / 100 | 100 / 100 | 100 / 100 | 13.8 / 100 | **91.9** / 100 |
| recall, bare format | 18.1 / 98.8 | 25.0 / 40.6 | 20.6 / 98.8 | 20.0 / 100 | **28.1** / 99.4 |
| yes/no | 42.5 / 100 | 96.2 / 100 | 86.2 / 100 | 63.8 / 100 | 58.8 / 100 |
| pair | 51.2 / 81.2 | 91.2 / 96.2 | 73.8 / 70.0 | 46.2 / 88.8 | 50.0 / 57.5 |
| Timmy k=3 | 36.9 / 48.8 | 40.0 / 43.1 | 58.8 / 76.2 | 33.1 / 42.5 | 31.2 / 46.9 |
| k=4 | 25.0 / 41.2 | 34.4 / 39.4 | 51.2 / 69.4 | 26.2 / 37.5 | 25.0 / 31.2 |
| held-out species | 40.6 / 43.8 | 24.0 / 32.3 | 30.2 / 75.0 | 33.3 / 41.7 | 36.5 / 39.6 |
| ICL suite symbol | 60.4 | 51.6 | 79.2 | 76.6 | 78.6 |
| ICL suite natural | 84.4 | 80.8 | 88.0 | 84.4 | 86.5 |
| ARC-Easy | 73.5 | 66.0 | 59.0 | . | **77.0** |
| WikiText ppl (16) | 10.61 | 37.78 | 23.18 | 11.50 | **10.90** |

**Table 18.2: the teacher's ranking and the student's, per question form (`scripts/diag_distill_options.py`; argmax over the option set, 136 species each; base = bare prompt, no adapter)**

| question form | teacher with entry: acc / top-option p | base bare: acc | P2 bare: acc / top-option p |
|---|---|---|---|
| type ("{N} is a {V}-type.") | 100 / 0.998 | 11.8 | 100 / 0.996 |
| weakness | 100 / 1.000 | 0.0 | 100 / 1.000 |
| habitat | 100 / 1.000 | 20.6 | 100 / 0.998 |
| region | 100 / 1.000 | 17.6 | 100 / 0.998 |
| diet ("{N} is an {V}.") | **58.1** / 0.805 | 19.9 | **62.5** / 0.818 |
| ladder L1 recall, bare, 160 items | 99.4 / 0.939 | 11.9 | 54.4 / 0.831 |
| ladder L1 recall, trained format, 160 items | 100 / 0.998 | 7.5 | 100 / 0.996 |
| training episodes (teacher, mean over the run) | 31 / 0.62 | | |

### 18.1 The facts went in, at almost no cost

Trained-format recall is 91.9 against the base model's 11.9 (+80.0 on the same items, CI [73.8, 86.2]) and arm P's 13.8, 8.1 points under arms A and C (p = 0.0002); on the 680 training questions the student reproduces the teacher's ranking exactly on four of the five forms (Table 18.2). Bare-format recall by the mean-per-token rule is 28.1, the highest of any arm (+10.0 over base, +8.8 over C, both p < 0.003; A +3.8, n.s.), and 54.4 by the sum rule the distillation itself used. The general-text cost is 0.027 nats per token (perplexity 10.61 to 10.90), thirty times less than arm C's 0.84 and three times less than arm Cg's replay-protected 0.09 (section 17); ARC-Easy is 77.0, above the base model's 73.5 and +18.0 over C (p = 2e-8) and +11.0 over A (p = 0.0007) on the same 200 items; the ICL suite is at C's level (78.6 against 79.2, and +18.2 over base, p = 2e-5). Where the teacher is wrong the student is wrong with it: the diet form, whose options read "{N} is an herbivore", has the teacher at 58.1 and the student at 62.5, so the soft targets transferred the teacher's error rate along with its facts.

### 18.2 What did not go in: everything asked in another form

The yes/no level is 58.8 (A 96.2, C 86.2; +16.2 over base, CI [-3.8, 35.0], p = 0.14), pairwise same-type 50.0 (A 91.2; chance), Timmy k=3 31.2 and k=4 25.0, both inside their section 11 null bands, and the held-out-species control pair reads 37.5 seen against 8.3 unseen. So the adapter answers "What type is Blaxorc?" in the sentence it was trained to rank and cannot say yes to "Is Blaxorc a Voltrix-type creature?", nor compare two species, nor use the type as a grouping rule. Arm A learns the same facts from fourteen templates including the yes/no and comparative ones and gets 96 and 91 on those levels; P2 saw five question forms with soft targets and no declarative text, and the knowledge stayed in those five forms. The 91.9 is a ranking the model reproduces, not a fact it can manipulate, which is the section 14 diagnosis one level up: the with-context number was a ranking, the ranking can be written into the weights, and what is written is the ranking.

The episode stream did nothing for the same reason arm P's did: the teacher is at chance on the training episodes even with the field guide in front of it (31% with k between 2 and 5, mean top-option probability 0.62), so 40% of the mixture distilled noise. The base model's with-context 48.8 on the ladder's k=3 items (section 8) does not extend to the training episodes' templates and group counts. The ICL gain came from the 15% hard-label replay, as in section 14.

### 18.3 Cost and knowledge are not the same axis

Table 18.1's last two rows put the arms on one line: A learned 2,752 texts and paid 1.40 nats and 7.5 ARC-Easy points; C paid 0.84 and 14.5; Cg, with general text replayed, 0.09 and 2.5; P2 learned 680 rankings and paid 0.03 nats and gained 3.5 ARC-Easy points. The perplexity cost tracks how much *text* the adapter learned to model, not how many facts it holds, and the cheapest injection here is also the narrowest. For the merchant use case the trade reads the other way round: if the deployed question is fixed (this string, which category), a ranking is what is needed, and P2 is a way to write 680 of them into a 3B model for 0.03 nats.

### 18.4 What BASE-3 now says

Yes, the with-context ceiling can be distilled into the weights, once the target is the teacher's ranking over the option set rather than its next-token distribution: 91.9 trained-format recall from a teacher at 100, at the lowest general-text cost of any injecting arm, and with the teacher's own errors carried over. No, that is not the knowledge arms A and C hold: it answers in the forms it was distilled in and nowhere else. BASE-3 closes as answered on both counts. A distillation that also covered the yes/no and comparative forms would be arm A with soft labels, and the section 14 and 18 results together say what the soft part buys: the teacher's calibration, its mistakes, and a smaller footprint.

**What the report should say.** Option-renormalised prompt distillation writes the with-context ranking into the weights: 91.9 recall in the trained format (base 11.9, arm P 13.8) at 0.03 nats of general-text cost and no ARC-Easy or ICL loss, with the teacher's diet errors inherited. The facts are usable only in the five question forms they were distilled in: yes/no 58.8, pairwise at chance, induction inside the null band. A first run with left-padded option batches was discarded after a probe showed unsloth's training-mode forward ignores the left-padding mask; every evaluation in the repo right-pads and is unaffected.
## 19. Iteration time: evaluation 3x faster, training 2 to 2.5x faster, same numbers (TRAIN-6)

Date: 2026-09-15. Code: `ai_experiments.scoring.Scorer` (cross-item batching), `scripts/exp_curriculum.py` (`MICRO`, `ACCUM`, `GRAD_CKPT`, `PACK`, `EXTRAS`, `BENCH`, `RUN_TAG`), `scripts/compare_records.py`; data `results/rescore_C_p200_batched.json` and per-item files, `results/curriculum_Qwen2.5-3B_C_fast_p200.json`; tracker experiments `rescore` and `curriculum_v2`; PLAN.md step 26.

By mid-afternoon on 2026-09-15 a training run of the section 8 recipe cost about 50 minutes on this card: 17 (arm A) to 30 (arm C) minutes of training, four periodic points at 1.6 minutes each, and a 26-minute final evaluation, during which the GPU sat at 26% utilisation. The owner asked for iteration time before more experiments. Two things were slow for the same reason: small work units. The scorer ran one item per forward pass (its 2 to 8 options as the batch, prompts of 11 to 66 tokens on the ladder), and the trainer ran two micro-batches of 8 padded rows per step, with knowledge texts of 24 tokens and activation checkpointing on. Both are launch-bound at this size.

### 19.1 Evaluation: every option row of every item goes through one forward

`Scorer.score` now collects the (prompt, option) rows of all items in a call, sorts them by length, and runs them in forwards of up to 64 rows or 32,768 tokens; only the option positions go through the LM head (unsloth returns the final hidden states in place of logits when asked, so a 64-row batch never materialises 64 x L x 152k logits). The optional extras of section 10 (PMI premise, unconditional, letter and hybrid scores) are batched the same way and are now off by default in training runs (`EXTRAS=1` restores them; `rescore.py` keeps them on).

**Table 19.1: the same adapter (arm C, section 15 rerun) scored by the old and the new scorer (extras on in both)**

| | old scorer (section 15) | batched scorer |
|---|---|---|
| ladder, 1,744 items | 9.7 min | 2.6 min |
| probes, 192 | 1.5 | 0.6 |
| ICL suite, 384 | 3.0 | 1.1 |
| ladder with context, 1,744 | 9.9 | 4.8 |
| whole evaluation | 26.5 min | 9.1 min |
| predictions flipped, no context (2,320 items) | | 1.9% |
| predictions flipped, with context (1,744) | | 1.1% |
| largest level move (mean rule) | | 2.5 points (k=4 induction, 4 items) |
| largest per-option log-prob change | | 1.9 nats (a listed-choices row); 0.2 to 0.6 on most levels |

The flip rate is the section 9.3 merged-versus-unmerged rate (1.8%) and the level moves are inside the section 9.2 same-weights floor (2.6 points on 160 items): batch shape changes bf16 log-probs by about 0.03 nats per token on average (section 18 measured the same in plain transformers), and nothing else changed. With the extras off the evaluation is about 5 minutes.

### 19.2 Training: one micro-batch, no recomputation, packed rows

Three switches, all keeping 16 sequences per optimizer step, the same streams, the same seed:

- `MICRO=16 ACCUM=1`: one forward and backward per step instead of two. The loss is then the mean over all 16 sequences' label tokens rather than the average of two 8-sequence means, a change in weighting only when the two halves differ in token count.
- `GRAD_CKPT=0`: no activation recomputation. It fits in a 60-step bench once rows are packed (a 16-row padded batch of arm C does not), and it is not the default: a full arm C run on it died of memory at step 600, once the periodic evaluations had fragmented the allocator (first as an out-of-memory in the first backward after the step-0 point, then, with expandable allocator segments and a smaller mid-training scorer budget, as a cuBLAS execution failure at step 600). Unsloth's offloaded checkpointing stays on.
- `GRAD_CKPT=1` (default from the evening of 2026-09-15; the runs in Tables 19.2 and 19.3 used `unsloth`): plain torch activation checkpointing instead of unsloth's CPU-offloaded variant. The bench measured the two at the same speed (section 19.2); the offloaded variant then died with CUDA out-of-memory and cuBLAS errors raised inside its backward at 11 GiB allocated, within 200 steps in three of the queue's packed runs (arm D seeds 1 and 2, Instruct C1) while the same runs completed under plain checkpointing (section 20.4). Same operation, different activation storage; the numbers are unaffected.
- `PACK=2048`: each micro-batch's sequences are concatenated into rows of at most 2,048 tokens with per-sequence position ids and unsloth's `packed_seq_lengths`, which selects the block-diagonal xformers kernel (no flash-attention on this card). Each row is forwarded and backed separately so memory is bounded by one row; the first token of every packed sequence carries no label, so the shifted loss never crosses a boundary. A probe with a train-mode LoRA model checked the mechanism: packed rows reproduce single-sequence log-probs to the same bf16 noise as padded batches (mean 0.025 nats per token), packing order changes nothing, and replacing one sequence moves its neighbours' log-probs by exactly zero.

**Table 19.2: training throughput over 60 steps (tokens per second, peak allocated GiB), 16 sequences per step**

| configuration | arm A (knowledge, 24-token texts) | arm C (mixed, 24 to 587 tokens) |
|---|---|---|
| before: 8 x 2, checkpointing, padded | 496 (8.1) | 858 (8.7) |
| 16 x 1, checkpointing, padded | 960 (8.1) | 766 (8.9) |
| 16 x 1, no checkpointing, padded | 1,334 (9.6) | out of memory |
| 16 x 1, no checkpointing, packed 768 | 1,251 (9.6) | 1,859 (19.1) |
| 16 x 1, no checkpointing, packed 2,048 | 1,271 (9.6) | 2,827 (17.9), fails in a full run |
| 16 x 1, no checkpointing, packed 4,096 | 1,349 (9.6) | 2,407 (22.6) |
| **16 x 1, checkpointing, packed 2,048 (new default)** | **1,017 (8.2)** | **2,133 (11.4)** |
| 32 x 1, no checkpointing, padded (32 sequences per step: a different recipe) | 2,174 (11.9) | out of memory |

Arm A gains 2x on the default (2.7x without checkpointing) and nothing from packing: sixteen 24-token texts are 400 tokens, and at that size a step is the fixed cost of launching a 36-layer forward and backward, about 0.3 seconds, whatever the layout; only more sequences per step would go faster (the 32-row line), which is a different recipe. Arm C gains 2.5x (3.3x without checkpointing), most of it from packing, because its padded batches were mostly padding: a micro-batch mixing 24-token texts with 200- to 600-token replay episodes pads every row to the longest. The loss-weighted M arms of step 9 run through the same path (2,239 tokens per second without checkpointing). A 4,096-token cap is slower than 2,048 and 3 GiB from the card's limit; 2,048 is the default, with 11 GiB of headroom under checkpointing.

### 19.3 The recipe's numbers do not move

Arm C was retrained on the new defaults with the section 8 seed (`RUN_TAG=fast`, four periodic points): 11.2 minutes of training at 1,863 tokens per second and 11.3 GiB, a 2.7-minute final evaluation, 15 minutes end to end where the section 15 rerun took 63. The recipe has two earlier seed-0 runs to compare with, the section 8 run and the section 15 rerun, and those two already disagree on 17.9% of their predictions (same seed, same code, same machine: bf16 non-determinism compounding over 800 steps); the fast run disagrees with the section 15 rerun on 19.8%. Level by level:

**Table 19.3: arm C, seed 0, three runs (accuracy %; the first two on the old path)**

| level | section 8 run | section 15 rerun | fast path |
|---|---|---|---|
| recall, trained format | 100 | 100 | 100 |
| recall, bare format | 20.6 | 19.4 | 15.0 |
| yes/no | 86.2 | 77.5 | 87.5 |
| pair | 73.8 | 80.0 | 73.8 |
| Timmy k=3 | 58.8 | 61.2 | 56.9 |
| k=4 | 51.2 | 53.8 | 43.8 |
| weakness | 52.5 | 60.0 | 52.5 |
| habitat | 34.4 | 29.4 | 31.2 |
| held-out species | 30.2 | 29.2 | 28.1 |
| ICL suite symbol | 79.2 | 78.1 | 79.7 |
| ICL suite natural | 88.0 | 87.5 | 89.1 |
| ARC-Easy | . | 59.0 | 64.0 |
| WikiText ppl | 23.18 | 22.80 | 22.09 |
| training minutes | 24.6 | 50.3 | 11.2 |
| evaluation minutes | 24.7 | 26.5 | 2.7 |

Every fast-path value lies between the two old-path values or within one half-width of them (the k=4 induction level, 43.8 against 51.2 and 53.8, is 1.3 half-widths below the lower one on 160 items). The general-text cost is 0.05 nats under the old runs' 0.84 and 0.86. Nothing here distinguishes the new path from a third run of the old one, which is the standard section 9.2 set for a code change: the recipe is the same, the samples are the same, only the arithmetic differs by bf16 batch shape.

### 19.4 What a run costs now

| | before | after |
|---|---|---|
| arm A, 800 steps | 17 min | about 7 |
| arm C, 800 steps | 30 min | about 12 |
| periodic point (subsample) | 1.6 min | about 0.7 |
| final evaluation | 26 min | 5 (9 with the section 10 extras) |
| one arm C run with four periodic points | 63 min | about 20 |

The remaining evaluation time is the with-context ladder (1,744 items whose prompts carry field-guide entries) and the ICL suite (prompts up to 650 tokens); both are now compute-bound at 64 rows per forward. The remaining training time for short-text arms is launch overhead per step, which only a larger batch or fewer steps would remove, and section 15 says the runs are already 200 steps too long for recall. Everything queued behind this section (steps 8, 9, 10) runs on the new defaults; runs before it are marked in the tracker by `micro=8, accum=2, grad_ckpt=unsloth, pack=0`.

**What the report should say.** Batching option rows across items cuts the evaluation from 26 to 9 minutes (5 without the scorer-study extras) with 1.9% of predictions flipping, the same-weights noise floor of section 9. One 16-sequence micro-batch packed into 2,048-token block-diagonal rows trains arm A 2x and arm C 2.5x faster at identical sequences per step (3.3x without activation checkpointing, which fits a bench and not a full run); the block-diagonal attention was verified directly. An arm C run with periodic points goes from about an hour to about twenty minutes.

## 20. Three seeds: the induction gain and the ICL cost are real, the manipulation and "sequential loses" gaps are not (STAT-1)

Date: 2026-09-15. Code: `SEED={1,2} PERIODIC=200 uv run python scripts/exp_curriculum.py {A,C,D}` (seed 0 is the section 15 rerun), `scripts/seeds_table.py` writes `reports/seeds_Qwen2.5-3B.md` (every metric, every gap). Data `results/curriculum_Qwen2.5-3B_{A,C,D}_s{1,2}_p200.json`; adapters `_s{1,2}_p200_lora`. The seed sets the LoRA initialisation, the sampler and the stream shuffles; the data, the item sets and the schedule are fixed. Seeds 1 and 2 ran on the section 19 fast path (D under plain torch checkpointing, see 20.4), seed 0 on the padded path; section 19.3 put those within the same-seed floor.

STAT-1 asked what the seed-to-seed noise is and which of section 8's gaps clear it. Section 8 compared one run per arm, and its headlines were: knowledge-only (A) gives recall and manipulation, the mixture (C) trades some manipulation for induction from the weights and keeps the ICL suite, and the sequential arm (D) loses 4 / 25 / 8 / 12 points against C on recall, yes/no, pair and real-name induction (section 11's paired table). Section 15 added ARC-Easy and the perplexity cost. With three seeds the rule is the one STAT-1 set: a gap counts if it exceeds twice the sd of the difference (the two arms' variances added, n = 3 each).

**Table 20.1: mean +- sd over seeds 0, 1, 2 (accuracy %; ppl is the section 16 slice; seed 0 re-scored with the section 19 scorer); base model, re-scored with the section 19 scorer, in the last column**

| measure | A | C | D | base |
|---|---|---|---|---|
| recall, trained fmt | 100.0 +- 0.0 | 100.0 +- 0.0 | 91.5 +- 3.5 | 13.1 |
| recall, bare | 21.2 +- 1.6 | 18.4 +- 1.3 | 44.2 +- 20.6 | 18.1 |
| yes/no (is-a) | 98.3 +- 2.9 | 80.0 +- 12.7 | 70.9 +- 8.3 | 42.5 |
| pair | 90.8 +- 1.9 | 78.7 +- 12.7 | 68.7 +- 16.0 | 51.2 |
| Timmy k=3 | 35.9 +- 3.6 | 63.8 +- 9.2 | 59.2 +- 7.0 | 35.6 |
| k=4 | 35.2 +- 6.9 | 57.9 +- 11.0 | 54.6 +- 7.0 | 27.5 |
| weakness | 41.5 +- 4.1 | 59.6 +- 11.9 | 55.8 +- 6.7 | 35.0 |
| held-out species | 26.0 +- 2.8 | 31.9 +- 3.2 | 36.1 +- 2.6 | 39.6 |
| ICL suite, symbol | 50.3 +- 2.9 | 79.0 +- 1.1 | 79.2 +- 1.8 | 60.4 |
| ICL suite, natural | 81.1 +- 1.1 | 87.5 +- 0.9 | 88.4 +- 1.1 | 84.9 |
| ARC-Easy (K) | 63.7 +- 2.1 | 64.0 +- 4.8 | 63.7 +- 2.4 | 73.5 |
| WikiText ppl | 39.1 +- 5.2 | 22.6 +- 0.2 | 21.9 +- 0.9 | 10.61 |

**Table 20.2: section 8's headline gaps against 2 sd of the difference**

| gap | measure | section 8 (one seed) | mean diff over 3 seeds | 2 sd | clears? |
|---|---|---|---|---|---|
| C - A | Timmy k=3 | +18.7 | +27.9 | 19.8 | **yes** |
| C - A | k=4 | +17.4 | +22.7 | 26.0 | no |
| C - A | weakness | +6.9 | +18.1 | 25.1 | no |
| C - A | ICL suite, symbol | +27.0 | +28.6 | 6.1 | **yes** |
| C - A | WikiText ppl | . | -16.5 | 10.4 | **yes** |
| C - A | yes/no | -10.0 | -18.3 | 26.0 | no |
| C - A | pair | -17.4 | -12.1 | 25.6 | no |
| D - C | recall, trained fmt | -3.8 | -8.5 | 6.9 | **yes** |
| D - C | Timmy k=3 | +1.9 | -4.6 | 23.2 | no |
| D - C | yes/no | -25.0 | -9.2 | 30.3 | no |
| D - C | pair | -7.6 | -10.0 | 40.8 | no |
| D - C | ICL suite, symbol | +3.1 | +0.2 | 4.2 | no |
| C - base | ARC-Easy | -14.5 | -9.5 | 9.6 | no |
| C - base | Timmy k=3 | +21.2 | +28.2 | 18.5 | yes |

### 20.1 How big the noise is

It depends on the arm and the level. Arm A is a stable recipe: sd 2 to 4 points on every level but k=4 (6.9). Arm C is not: its manipulation levels have sd 12.7 (yes/no 77.5 / 68.8 / 93.8; pair 81.2 / 65.0 / 90.0), and its three induction levels sd 9 to 12 (Timmy 61.9 / 55.6 / 73.8). Arm D sits between, with one level out of control: bare recall 58.8 / 20.6 / 53.1. The ICL suite, ARC-Easy and the perplexity slice have sd 1 to 4.5 across all three arms, so the general-ability measures are the reliable ones, and the 160-item ladder levels of the mixture arms are not: with p near 0.7 the binomial sd of a 160-item level is 3.6 points, and C's 12-point seed sd is three times that. The seed changes what the mixture teaches, not just which items it gets right.

The same-seed replicate column of `reports/seeds_Qwen2.5-3B.md` (the section 8 originals against the section 15 reruns, same seed, backend and code changed in between) moves by up to 8.7 points on C's manipulation and 11 on D's induction. A single seed 0 run's distance from its own replicate is of the same order as the seed sd, which is what section 9.2's floor said.

### 20.2 Which headlines survive

Three of section 8's claims clear 2 sd, and section 15's ARC-Easy loss holds across arms:

- The mixture gains induction from the weights over knowledge-only: Timmy k=3 +27.9 (2 sd 19.8). k=4 and weakness point the same way (+23, +18) and do not clear their 25-point bands; their means are larger than section 8's single-seed gaps, not smaller.
- The mixture keeps the ICL suite and knowledge-only destroys it: +28.6 on symbol labels (2 sd 6.1), the cleanest result in the project.
- Knowledge-only costs 16 more perplexity points on WikiText than the mixture (2 sd 10.4); both cost against the base model (C +12.0, 2 sd 0.4).
- Every trained arm loses about 9 ARC-Easy points against the base model: 63.7 / 64.0 / 63.7 against 73.5. Per arm the gap sits at the edge of its band (C -9.5, 2 sd 9.6); three arms agreeing to 0.3 points is what makes it a result. Section 15.3's -14.5 was a high draw.

Two of section 8's claims do not:

- "Knowledge-only manipulates better than the mixture": A 98.3 / 90.8 against C 80.0 / 78.7, differences of -18.3 and -12.1 with 2 sd 26.0 and 25.6. The direction held in all three seeds on yes/no (100 / 95 / 100 against 77.5 / 68.8 / 93.8) but the size is unknown to within +-13 points, because C's manipulation is where C's seed noise lives.
- "Sequential loses to interleaved": the 25 (yes/no) is -9.2 +- 15.2, the 8 (pair) is -10.0 +- 20.4, and induction (Timmy k=3) is -4.6 +- 11.6. Only the recall gap survives: D reaches 91.5 +- 3.5 in the trained format where A and C reach 100 in every run, -8.5 with 2 sd 6.9, and section 8.3's explanation (phase 2 trains at half the peak learning rate under one schedule, TRAIN-2) is still the open one. D matches C on the ICL suite (+0.2 +- 2.1) and on induction; the section 8 reading that staging costs the mixture's benefits was a single-seed reading.

### 20.3 What changes in how results are read

From here, an arm comparison on a ladder level of a mixture arm needs three seeds or a paired per-item test (section 11) before it is a claim; a 10-point gap on manipulation or induction between two single runs is inside the noise. General-ability measures (ICL suite means, ARC-Easy, WikiText) can be read from one run to about +-4. Sections 17 (Cg), 18 (P2) and 21 (M arms) compare single runs against arm C on ladder levels; their induction and manipulation gaps of 10 to 20 points should be read against Table 20.1's sd, and their ICL, ARC and perplexity numbers stand. The 2 sd rule is the one to keep; `seeds_table.py` restates every gap whenever a seed is added.

### 20.4 A note on the D runs

Both D seed runs died within 100 steps on the fast path under unsloth's offloaded gradient checkpointing (`use_gradient_checkpointing="unsloth"`: CUDA out-of-memory and cuBLAS internal / execution errors raised inside the checkpoint backward at 11 GiB allocated, with A, C and the M arms running 800 steps each on the same code; Instruct C1 died the same way in section 22's queue), and both completed under plain torch checkpointing (`GRAD_CKPT=1`), which the section 19 bench had measured at the same speed. Their configs record `grad_ckpt=True`. The default is now plain checkpointing; the numbers are the same operation either way, only the activation storage differs. The failure is not understood beyond the fact that arm D's packed rows vary in length more than the other arms' (256 to 2,000 tokens), which is where a state-machine bug in the offload buffers would show.

## 21. Mixture by loss weight: the nominal 45/40/15 never trained arm C, and imposing it removes the facts (TRAIN-1)

Date: 2026-09-15. Code: `PERIODIC=200 uv run python scripts/exp_curriculum.py {M0,M20,M40,M60}` on the section 19 fast path (packed rows, one 16-sequence micro-batch, seed 0); data `results/curriculum_Qwen2.5-3B_M*_p200.json`, per-item under `results/per_item/`; adapters `curriculum_Qwen2.5-3B_M*_p200_lora`. Comparison arm C is the section 19.3 fast-path rerun (`C_fast`) and its seed-2 rerun (section 20); both are the same recipe as section 8's arm C.

TRAIN-1 asked what arm C's mixture is in loss tokens rather than sequences. Section 8.1 answered the accounting half: 45% knowledge sequences are 84% of the loss-bearing tokens, because a knowledge text carries loss on every one of its ~25 tokens while an episode or a replay item carries it on 2 to 4 answer tokens. Step 9 tests the other half: does the recipe work *because* of that implicit weighting? The M arms keep arm C's sampling and change only the loss: each stream's mean token loss is weighted by its mixture fraction (`BY_LOSS`, `exp_curriculum.py`), so "K .45 / E .40 / R .15" means 45% of the gradient from knowledge, not 84%. Three additions from the step 9 design ride along: a self-teaching stream S (`universe.self_teaching`: completion, true/false and in-document multiple choice built from the training texts, answer-only loss, Self-Tuning 2406.06326), loss on every inferable demo label inside an episode rather than the query label alone (`ALL_ANSWER`, M20/M40/M60), and a sweep of the episode weight E in {0.2, 0.4, 0.6} at fixed R = 0.15.

**Table 21.1: the arms, by loss weight and by loss-bearing tokens (800 steps x 16 sequences, seed 0)**

| arm | loss weights K / S / E / R | knowledge weight (K+S) | loss-bearing tokens K / S / E / R | sequences K / S / E / R |
|---|---|---|---|---|
| C (pooled token mean) | implicit 84 / - / 12 / 4 | 84% | 141,683 / - / 19,582 / 7,144 | 5,845 / - / 5,034 / 1,921 |
| M0 | .45 / - / .40 / .15 | 45% | 141,683 / - / 19,582 / 7,144 (identical batches to C) | 5,845 / - / 5,034 / 1,921 |
| M20 | .43 / .22 / .20 / .15 | 65% | 135,479 / 7,010 / 18,290 / 7,434 | 5,585 / 2,698 / 2,534 / 1,983 |
| M40 | .30 / .15 / .40 / .15 | 45% | 93,652 / 4,983 / 36,331 / 7,322 | 3,856 / 1,912 / 5,078 / 1,954 |
| M60 | .17 / .08 / .60 / .15 | 25% | 52,391 / 2,662 / 54,612 / 7,412 | 2,165 / 1,035 / 7,608 / 1,992 |

M0 is the clean test: same seed, same sampler, so the same 12,800 sequences in the same order as C_fast (the `tok_*` and `lb_*` counters match to the token); only the per-token weights differ. Runs took 10 to 14 minutes each (1,740 to 1,890 tokens per second, 11.3 GiB).

**Table 21.2: from the weights (no context), accuracy %. Arm C as three runs: section 8 seed 0 slow path / fast path seed 0 / fast path seed 2**

| measure | C (3 runs) | M0 | M20 | M40 | M60 |
|---|---|---|---|---|---|
| L1 recall, trained fmt | 100 / 100 / 100 | **33.8** | 100 | **36.2** | **13.1** |
| L2 manipulation is-a | 77.5 / 87.5 / 93.8 | 45.0 | 58.8 | 45.0 | 45.0 |
| L2 manipulation pair | 80.0 / 73.8 / 90.0 | 50.0 | 50.0 | 50.0 | 50.0 |
| L3 induction, nonsense names | 61.2 / 56.9 / 73.8 | 31.2 | 46.9 | 31.9 | 35.0 |
| L3 induction, real names | 78.1 / 72.5 / 86.9 | 38.1 | 56.2 | 38.8 | 36.2 |
| L4 weakness | 60.0 / 52.5 / 71.2 | 35.0 | 42.5 | 32.5 | 32.5 |
| ICL suite, symbol labels | 78.1 / 79.7 / 80.2 | 78.6 | 79.2 | 79.7 | 81.2 |
| ARC-Easy (K) | 59.0 / 64.0 / 66.0 | 67.5 | 66.0 | 60.0 | 66.0 |
| WikiText ppl (base 10.61) | . / 22.1 / 22.6 | 14.2 | 16.5 | 16.9 | 15.1 |

Chance is 12.5 on recall, 50 on the L2 pair, 25 on the four-option items. The base model scores 13.1 / 42.5 / 51.2 / 35.6 / 38.1 / 35.0 on the first six rows, 60.4 on the ICL suite and 73.5 on ARC-Easy (re-scored with the section 19 scorer, `results/curriculum_Qwen2.5-3B_base.json`).

**Table 21.3: with the field guide in context, accuracy %**

| measure | base | C (3 runs) | M0 | M20 | M40 | M60 |
|---|---|---|---|---|---|---|
| L3 induction, nonsense names | 49.4 | 70.6 / 78.8 / 78.1 | **86.2** | 68.8 | 78.8 | 83.8 |
| L3 held-out species | 43.8 | 70.8 / 74.0 / 71.9 | **82.3** | 47.9 | 79.2 | 76.0 |
| L4 weakness | 46.9 | 67.5 / 70.6 / 71.2 | 76.9 | 63.8 | 74.4 | 75.6 |
| L8 reverse, easy | 95.6 | . / 80.6 / 82.5 | **90.6** | 88.8 | 78.1 | 88.1 |
| L8 reverse, hard | 88.1 | . / 79.4 / 79.4 | **90.6** | 82.5 | 76.2 | 84.4 |

### 21.1 The implicit 84% is the recipe

M0 has arm C's batches and arm C's sampler and does not learn the facts. Formatted recall is 33.8 at step 800 against arm C's 100 in all three runs, and the periodic curve shows it was not on its way: 10.0, 12.5, 20.0 at steps 200, 400, 600, where C_fast reads 25.0, 100, 100. Both manipulation levels sit at chance (45.0, 50.0) and induction from the weights is at or below the base model. Halving the knowledge stream's share of the gradient, from the 84% the pooled mean gives it to the 45% the mixture nominally says, is enough to keep 160 species out of the weights in 800 steps. Under Adam the absolute scale of the loss does not matter; what changed is the direction, now dominated by 19,582 episode-answer tokens and 7,144 replay tokens carrying 55% of the weight between them.

So the answer to TRAIN-1 is that arm C's mixture was never 45/40/15 in any sense that matters to the optimiser, and the number that made it work is the 84% it never declared. Every "mixture" comparison in sections 8 and 15 (C against A, Cn, D, E) was a comparison of pooled means over streams with very different label densities. The design table in section 8.1 now has to be read with Table 21.1's third column, not its first.

### 21.2 The E sweep at the loss level: no setting keeps the facts and gains anything

At E = .20 (M20) recall comes back to 100, but only because the self-teaching stream S lifts the knowledge weight to 65% (K + S); manipulation stays at 58.8 / 50.0 against C's 78 to 94 / 74 to 90, and induction from the weights is 10 to 20 points under C. At E = .40 (M40, knowledge 45%) and .60 (M60, 25%) recall is 36.2 and 13.1, and everything downstream of recall is at chance. The three points of the sweep are all worse than C on every from-the-weights measure, and the trend across them is monotone in the knowledge weight, not in E: recall_fmt 13 / 34 / 36 / 100 at knowledge weight 25 / 45 / 45 / 65%, then 100 at C's 84%.

What the extra episode weight buys is visible only with the field guide in context (Table 21.3): M0 is the best arm the project has produced on in-context induction (86.2 on nonsense names, 82.3 on held-out species; base 49.4 and 43.8), 8 to 12 points over arm C's runs, and it is the only trained arm that reads the backward question from context as well as the base model does (90.6 / 90.6 against the base's 95.6 / 88.1, where arm C's runs read 80 to 83 / 79). That is the skill the episodes teach, and giving them 40% of the gradient teaches more of it. It costs the facts, and in-context recall was already 100 for every arm, so the ceiling on the with-context ladder is the induction levels alone.

The self-teaching stream dissociates recall from use: M20 reaches 100 on the trained-format recall question and 58.8 / 50.0 on is-a and pair manipulation, a gap no other arm shows (arm A, knowledge texts alone, has 100 / 100 / 88.8 on the same three rows; arm C's manipulation tracks its recall). With 22% of the gradient on completion, true/false and in-document multiple-choice tasks that name the species and its attribute, the model learns to produce the attribute given the species in those forms; comparing two species or answering "is X a Y" stays at the base model's level. Recall needs less knowledge weight than manipulation does, and S buys the cheaper of the two. The all-answer loss (M20/M40/M60 carry loss on 2 to 3 times as many episode tokens as M0) shows no effect separable from the weight change.

### 21.3 What TRAIN-1 now says

Closed. Arm C is 84% knowledge by loss, and that weighting is load-bearing: the same batches at the nominal 45% do not memorise the facts in 800 steps (33.8 recall_fmt, manipulation at chance), and no loss-level episode weight in {.2, .4, .6} matches arm C from the weights. Any future mixture change should be stated and swept in loss weight, with the knowledge weight held at or above 0.8 unless the point is to trade the facts for in-context induction (M0's 86 / 82 / 91 with context is the reference for that trade). The pooled token mean stays the default; the `BY_LOSS` path stays in the script for stated-weight experiments. The section 17 follow-up (Cg with the general text taken out of the knowledge stream, K .40 / E .40 / R .15 / G .05 by sequence) is a sampler change, not a loss-weight change, and is unaffected; it remains queued.

## 22. Masked fine-tuning: one rendering learns the backward direction, costs manipulation and context reading, and paraphrases still win from the weights (TRAIN-5, EVAL-7)

Date: 2026-09-15. Code: `uv run python scripts/exp_curriculum.py {base,A1,A1m,A,C,C1,Cm} Qwen/Qwen2.5-3B-Instruct` on the section 19 fast path (seed 0); `encode_masked` and `universe.single_texts` in `exp_curriculum.py` / `universe.py`; the backward item set is `data/processed/reverse_v1.json` (`universe.reverse_items`, 320 items, sha 4e6f7dc244d9, section 9 rules). Data `results/curriculum_Qwen2.5-3B-Instruct_*.json`; adapters `curriculum_Qwen2.5-3B-Instruct_*_lora`.

Pan et al. (2510.09885) train a decoder on "here is the passage with 5 to 95% of its tokens masked; recover it", with the ordinary next-token loss on the reconstruction, and report forward and backward recall above 0.9 from a single rendering per fact where plain fine-tuning with paraphrases gives 0.95 forward and 0.04 backward. TRAIN-5 asks whether that removes the need for the 20 paraphrases arm A trains on and fixes the reversal curve. The setup here follows the paper's: Qwen2.5-3B-Instruct, LoRA r=64 on all linear layers, one text per trained species (136 texts of ~47 tokens carrying every attribute), 800 steps of 16 sequences (94 passes over each text). The masked sample is `Recover the original passage from the masked version.\nMasked: <text with a fraction t ~ U(0.05, 0.95) of its tokens replaced by <|fim_pad|>>\nOriginal: <text>`, loss on the original only; the plain sample is the text with full-sequence loss. Arms: A1 plain single text, A1m masked single text, A the 2,752 paraphrases (section 8's arm A on this model), C section 8's mixture, C1 the mixture with the single text in place of the paraphrases, Cm the mixture with the masked single text.

The backward probe is new. `L8_reverse_easy`: "Which creature is a {type}-type with an {diet} diet, found in {region}?", four species names, distractors of another type, so recalling the type of each option answers it (the merchant `reverse` construction, EVAL-7). `L8_reverse_hard`: distractors of the same type differing in diet or region, so the conjunction is needed. Every training text states attributes in the forward direction only. 160 items per level, chance 25.

**Table 22.1: from the weights (no context), accuracy %, Qwen2.5-3B-Instruct**

| measure | base | A1 plain, 1 text | A1m masked, 1 text | A 20 paraphrases | C mixture | C1 mixture, 1 text | Cm mixture, masked |
|---|---|---|---|---|---|---|---|
| L1 recall, bare | 20.0 | 38.8 | 20.0 | 20.6 | 24.4 | 45.0 | 45.6 |
| L1 recall, trained fmt | 11.9 | 65.6 | **87.5** | 100 | 100 | 54.4 | 36.2 |
| L2 manipulation is-a | 45.0 | 46.2 | 45.0 | 100 | 51.2 | 56.2 | 57.5 |
| L2 manipulation pair | 50.0 | 50.0 | 50.0 | 92.5 | 51.2 | 50.0 | 51.2 |
| L3 induction, nonsense names | 33.8 | 33.1 | 35.6 | 38.8 | 58.8 | 26.9 | 28.1 |
| L4 weakness | 33.1 | 33.1 | 38.8 | 44.4 | 53.8 | 30.0 | 34.4 |
| L8 reverse, easy | 19.4 | 21.2 | **60.6** | 87.5 | 35.0 | 21.9 | 24.4 |
| L8 reverse, hard | 30.0 | 20.0 | **48.1** | 28.8 | 21.2 | 23.8 | 26.2 |
| ICL suite, symbol labels | 64.6 | 60.4 | 51.0 | 54.2 | 78.1 | 83.3 | 80.2 |
| ICL suite, natural labels | 84.4 | 82.8 | 82.8 | 83.8 | 89.0 | 88.0 | 84.9 |
| ARC-Easy (K) | 70.0 | 66.0 | 72.5 | 67.0 | 66.5 | 80.0 | 79.0 |
| WikiText ppl | 11.0 | 21.1 | 18.0 | 38.0 | 27.7 | 14.7 | 13.5 |
| training minutes | - | 5.7 | 9.6 | 5.2 | 9.4 | 10.3 | 13.8 |

**Table 22.2: with the field guide in context, accuracy %**

| measure | base | A1 | A1m | A | C | C1 | Cm |
|---|---|---|---|---|---|---|---|
| L1 recall, trained fmt | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| L2 manipulation pair | 53.8 | 80.0 | 68.8 | 98.8 | 78.8 | 62.5 | 60.0 |
| L3 induction, nonsense names | 45.0 | 42.5 | 38.8 | 40.0 | 81.9 | 76.9 | 73.1 |
| L8 reverse, easy | 97.5 | 90.6 | 88.1 | 99.4 | 91.9 | 79.4 | 96.9 |
| L8 reverse, hard | 92.5 | 82.5 | **51.9** | 91.2 | 85.0 | 76.2 | 91.9 |

### 22.1 The backward direction arrives from one rendering

A1 and A1m see the same 136 sentences the same number of times. Plain fine-tuning on them gives forward recall of 65.6 in the trained format and nothing backward: 21.2 and 20.0 against a base of 19.4 and 30.0, the reversal curse as the paper describes it. Masking the same sentences gives 87.5 forward and 60.6 / 48.1 backward. The hard level, where the options share the type and the model has to place the diet and region conjunction on the right name, is 23 points over chance from texts that never mention any name after its attributes. This is the paper's direction on a 3B model at LoRA r=64 with a 320-item probe, weaker than its 0.9 (the paper trains to convergence and scores generation), and it is the first arm in this project with backward recall from the weights above the section 4 merchant result (42.5 with augmentation, 25.8 without).

Against paraphrases, the trade is explicit. Arm A's 2,752 renderings (20 per species, 5 minutes of training) give this model 100 forward recall, 100 / 92.5 on manipulation, and 87.5 on the easy backward level, because the paraphrase set includes renderings that name the type before the species, so type-to-name is a forward association for arm A. On the hard level, where diet and region have to be attached to the name, arm A is at 28.8 (chance 25) and A1m at 48.1. Twenty renderings buy everything the forward direction can buy; one masked rendering buys the one thing they do not, at 87.5 forward instead of 100 and with no manipulation (46.2 / 50.0, base 45.0 / 50.0). The paper's claim that masking removes the need for paraphrases holds for backward recall and not for use of the fact.

Two costs. The symbol-label ICL suite drops to 51.0 (base 64.6, plain 60.4): 94 passes over an instruction-shaped reconstruction task with a fixed prefix moves the instruction-tuned model's in-context behaviour more than the same passes over bare text. And Table 22.2's last row: with the field guide in the prompt, the base model answers the hard backward question at 92.5 and A1m at 51.9. The arm that learned the backward direction in its weights is the arm that no longer reads it from context, while A1 (82.5), C (85.0) and Cm (91.9) keep the skill. The masked model's weights answer the question one way and the context another, and the weights win about half the time. Nothing else in Table 22.2 moves like this (recall with context is 100 for every arm), so it is the backward association specifically, in the same items where the weights now hold an answer.

### 22.2 In the mixture, the single text is not enough passes

Cm replaces arm C's 2,752 paraphrases with the masked single text at the same 45% of sequences: 5,813 masked sequences over 136 texts is 43 passes each, and the reconstruction tokens are 91% of the loss-bearing tokens (279,109 of 306,000), the episodes 6%. Recall in the trained format is 36.2, backward 24.4 / 26.2, induction from the weights below C. The base model's general abilities are the best preserved of any trained arm (ARC-Easy 79.0, above the base's 70.0; WikiText 13.5 against C's 27.7; symbol ICL 80.2), which reads as an arm that has learned little rather than one that has learned without cost. C1, the same mixture with the single text unmasked, says it is the passes: 54.4 recall in the trained format (A1 reached 65.6 from 94 passes of the same text, arm A 100 from 20 paraphrases), backward at chance (21.9 / 23.8), induction from the weights at the base's level. Masking the text inside the mixture then costs recall rather than adding to it (Cm 36.2 against C1's 54.4), because the reconstruction task at 43 passes has not converged where the plain text has half-learned. One rendering per fact needs its passes, masked or not; at 45% of a 12,800-sequence budget it does not get them, and this is the arm that section 8's mixture becomes when the paraphrases are taken away. Both C1 and Cm are above the base model on ARC-Easy (80.0 and 79.0 against 70.0; arm C on this model 66.5) and near it on WikiText (14.7 and 13.5 against 11.0): the mixture without paraphrases barely moves the model, and the episodes and ICL replay alone seem to help the four-option format.

### 22.3 What TRAIN-5 and EVAL-7 now say

TRAIN-5: partly. Masked fine-tuning of one rendering does what the paper says on the backward direction (60.6 / 48.1 from 21.2 / 20.0, the plain text at chance) and improves forward recall over the plain text (87.5 against 65.6), so it removes the need for paraphrases for backward recall. It does not remove it for manipulation (46.2 / 50.0 against arm A's 100 / 92.5), it costs the symbol-label ICL suite 14 points against the base, it halves the model's ability to read the same backward fact from context (51.9 against 92.5), and inside the section 8 mixture at 43 passes it learns less than the plain text. The recipe that gives everything from the weights is still the paraphrases; masking is the cheaper route to the one direction paraphrases written forward do not cover, and the two should be tried together (masked paraphrases) before either replaces the other. Open: whether the in-context loss is a property of masked training or of any arm that holds a backward answer in its weights (arm A, which holds the easy direction, keeps 91.2 with context on the hard level). EVAL-7: the merchant `reverse` construction is category-level (section 4.3's caveat): its distractors differ in the category, so the easy level here reproduces it and the hard level is the entity-level test. Masked fine-tuning moves both, and the hard level is the one that matters.

## 23. Arm D controls: the decaying schedule was protecting the facts, and 10% knowledge replay makes sequential equal interleaved (TRAIN-2)

Date: 2026-09-16. Code: `SEED={0,1,2} PERIODIC=200 uv run python scripts/exp_curriculum.py {Dr,Dc,Dk}` (fast path, plain checkpointing; `SCHEDULE` in `exp_curriculum.py`); data `results/curriculum_Qwen2.5-3B_{Dr,Dc,Dk}{,_s1,_s2}_p200.json`; `scripts/seeds_table.py --arms A C D Dr Dc Dk --out seeds_d_controls_Qwen2.5-3B.md` writes the full table. Comparison arms are section 20's three seeds of C and D.

Section 8.3 flagged that arm D confounds staging with its learning-rate schedule: one 30-step warmup and linear decay to zero spans both phases, so the episode phase (steps 400 to 800) trains from half the peak rate down to nothing, and it sees no knowledge text at all. Section 20 then found that of "sequential loses", only the recall gap survives three seeds: D 91.5 +- 3.5 in the trained format against 100 in every A and C run (2 sd 6.9), while the manipulation and induction gaps are inside the seed noise. TRAIN-2 asks whether that residue is staging or the schedule. Three controls, each three seeds, everything else as arm D:

- Dr: the schedule restarts at the phase boundary (warmup 30, linear decay to zero at step 400, again at 800), so the episodes train at the same rates the knowledge did.
- Dc: the rate is held at the peak (1e-4) after the warmup for all 800 steps, so neither phase decays.
- Dk: the shared schedule of arm D, but phase 2 replays 10% knowledge texts (episodes .75, knowledge .10, replay .15).

The literature's prediction (SURVEY TRAIN-1/TRAIN-2) is that D stays behind C under any schedule and that the remedy is mixing or replay.

**Table 23.1: mean +- sd over seeds 0, 1, 2 (accuracy %; `reports/seeds_d_controls_Qwen2.5-3B.md` has every metric and every gap)**

| measure | C | D | Dr restart | Dc constant | Dk +10% K |
|---|---|---|---|---|---|
| recall, trained fmt | 100.0 +- 0.0 | 91.5 +- 3.5 | **28.1 +- 16.4** | **41.0 +- 18.8** | **99.8 +- 0.3** |
| recall, bare | 18.4 +- 1.3 | 44.2 +- 20.6 | 37.5 +- 1.7 | 42.1 +- 8.3 | 21.4 +- 1.0 |
| yes/no (is-a) | 80.0 +- 12.7 | 70.9 +- 8.3 | 67.1 +- 6.1 | 61.2 +- 6.3 | 78.3 +- 11.9 |
| pair | 78.7 +- 12.7 | 68.7 +- 16.0 | 50.4 +- 1.4 | 50.0 +- 4.3 | 75.0 +- 14.7 |
| Timmy k=3 | 63.8 +- 9.2 | 59.2 +- 7.0 | 37.7 +- 3.8 | 39.4 +- 7.1 | 58.7 +- 4.9 |
| k=4 | 57.9 +- 11.0 | 54.6 +- 7.0 | 32.5 +- 1.9 | 34.2 +- 6.7 | 60.8 +- 4.4 |
| weakness | 59.6 +- 11.9 | 55.8 +- 6.7 | 34.8 +- 2.5 | 41.3 +- 2.5 | 56.7 +- 5.7 |
| ICL suite, symbol | 79.0 +- 1.1 | 79.2 +- 1.8 | 80.1 +- 1.6 | 72.0 +- 6.0 | 80.2 +- 3.4 |
| ICL suite, natural | 87.5 +- 0.9 | 88.4 +- 1.1 | 88.4 +- 0.3 | 84.9 +- 2.3 | 87.0 +- 0.5 |
| ARC-Easy (K) | 64.0 +- 4.8 | 63.7 +- 2.4 | 69.0 +- 3.3 | 66.8 +- 0.8 | 62.0 +- 2.8 |
| WikiText ppl | 22.6 +- 0.2 | 21.9 +- 0.9 | 19.7 +- 1.4 | 21.4 +- 3.9 | 19.6 +- 0.6 |

Arm A (knowledge only, section 20) reads 100 / 98.3 / 90.8 on recall and manipulation and 35.9 / 35.2 / 41.5 on the three induction levels.

**Table 23.2: trained-format recall at the periodic points, per seed (steps 0 / 200 / 400 / 600 / 800; phase 2 starts at 400)**

| arm | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| D | 12.5 / 95.0 / 100 / 95.0 / 93.1 | 12.5 / 100 / 100 / 97.5 / 93.8 | 12.5 / 82.5 / 100 / 80.0 / 87.5 |
| Dr | 12.5 / 95.0 / 100 / **42.5** / 45.6 | 12.5 / 95.0 / 100 / **17.5** / 13.1 | 12.5 / 82.5 / 100 / **22.5** / 25.6 |
| Dc | 12.5 / 95.0 / 97.5 / **57.5** / 50.6 | 12.5 / 100 / 97.5 / **57.5** / 53.1 | 12.5 / 80.0 / 100 / **25.0** / 19.4 |
| Dk | 12.5 / 97.5 / 97.5 / 100 / 100 | 12.5 / 97.5 / 100 / 97.5 / 100 | 12.5 / 87.5 / 100 / 97.5 / 99.4 |

**Table 23.3: gaps against 2 sd of the difference (three seeds each side)**

| gap | recall fmt | yes/no | pair | Timmy k=3 | k=4 | ICL symbol | WikiText ppl |
|---|---|---|---|---|---|---|---|
| Dr - D | -63.4 (33.5) **clear** | -3.8 (20.7) | -18.3 (32.1) | -21.4 (16.0) **clear** | -22.1 (14.4) **clear** | +0.9 (4.8) | -2.3 (3.4) |
| Dc - D | -50.4 (38.2) **clear** | -9.6 (20.8) | -18.7 (33.1) | -19.8 (20.0) | -20.4 (19.3) **clear** | -7.1 (12.6) | -0.5 (8.0) |
| Dk - D | +8.3 (6.9) **clear** | +7.5 (29.0) | +6.3 (43.5) | -0.4 (17.2) | +6.2 (16.5) | +1.1 (7.7) | -2.3 (2.2) **clear** |
| Dk - C | -0.2 (0.7) | -1.7 (34.8) | -3.7 (38.9) | -5.0 (20.9) | +2.9 (23.7) | +1.3 (7.1) | -3.0 (1.3) **clear** |

### 23.1 Phase 2 at the full learning rate erases phase 1

Every D variant reaches 100 formatted recall at step 400, the end of the knowledge phase (Table 23.2). What happens next depends only on the learning rate the episodes train at. Under arm D's shared schedule, phase 2 starts at half the peak rate and decays to zero, and recall ends at 87.5 to 93.8. Restart the schedule (Dr) and 400 steps of episodes at the full rate, with no knowledge text in the batch, take recall to 13.1 / 25.6 / 45.6 by the end, most of it gone by step 600. Hold the rate constant (Dc) and it is 19.4 / 50.6 / 53.1. The facts are not being contradicted: the episode stream carries no statement about any species, only random labels over species names in demos. They are being overwritten by 6,400 sequences of answer-only loss on labels, which is what catastrophic forgetting of a LoRA adapter looks like at 1e-4.

Induction from the weights goes with the facts. Dr and Dc sit at 37.7 and 39.4 on Timmy k=3 and 32.5 / 34.2 on k=4, which is arm A's level (35.9 / 35.2), the level of a model that never trained on episodes at all. The 400 steps of episodes at the full rate taught nothing usable from the weights, because the induction items need the species facts that the same steps erased. With the field guide in context the four arms are alike (65 to 81 on Timmy), so the in-context skill the episodes teach is intact; it is the from-the-weights version that needs the facts under it.

Section 8.3's reading of arm D, "episodes were trained at a lower learning rate", was right about the mechanism and wrong about the sign. The lower rate did not hold the episodes back; it kept them from erasing phase 1. What the sequential arm lost relative to C (recall 91.5 against 100, a clear gap in section 20) is the mild version of what Dr and Dc show in full.

Two side effects of the forgetting arms are worth a line. Their WikiText perplexity and ARC-Easy are the best of any trained arm with episodes (Dr 19.7 ppl and 69.0 ARC against D's 21.9 and 63.7), which is the general-ability cost of section 15 and 16 partly reversing as the knowledge goes: Dr ends closer to the base model on general text because it ends closer to the base model on everything. And the bare-format recall of Dr (37.5 +- 1.7, chance 12.5) is well above arm C's 18.4 while its formatted recall is 28.1: the full-rate episode phase overwrote the trained answer format more completely than the association itself, which is the same format-versus-fact distinction section 18 drew for the distilled arm.

### 23.2 Ten percent knowledge replay in phase 2 makes sequential equal interleaved

Dk keeps arm D's schedule and its two phases and replaces 10% of phase 2's sequences (640 of 6,400) with knowledge texts. Formatted recall is 99.8 +- 0.3 against D's 91.5 +- 3.5 (+8.3, 2 sd 6.9, clear) and C's 100.0. On every other ladder level Dk is inside arm C's seed band and above arm D's mean: yes/no 78.3 against C's 80.0, pair 75.0 against 78.7, Timmy k=3 58.7 against 63.8, k=4 60.8 against 57.9, weakness 56.7 against 59.6, ICL suite 80.2 against 79.0, ARC-Easy 62.0 against 64.0. The one clear difference from C is in C's disfavour: WikiText perplexity 19.6 +- 0.6 against 22.6 +- 0.2 (-3.0, 2 sd 1.3), 0.14 nats per token less drift with the same facts, the same induction and the same ICL suite. The recall curve (Table 23.2) never dips in phase 2.

So "sequential loses" is answered as section 8.3 and the survey predicted: a staged run needs replay of the earlier stage, and with 10% of it the staging costs nothing that three seeds can see. What it does not give is any advantage over interleaving either, except the perplexity, which is the one measure where less knowledge exposure in the second half would be expected to help. The recipe stays interleaved (arm C) as the simpler of two equals; when staging is forced by the data pipeline (facts first, task later), 10% replay of the facts under a decaying schedule is the version to use.

### 23.3 What TRAIN-2 now says

Closed. The residual "sequential loses" gap of section 20 (recall 91.5 against 100) is a mild case of forgetting that arm D's decaying schedule was holding down: restarting the schedule per phase or holding the rate constant lets 400 steps of episodes without knowledge text erase most of the facts (formatted recall 28 and 41, induction back to arm A's level), while 10% knowledge replay in phase 2 under the original schedule matches arm C on every level within the seed noise and beats it on WikiText perplexity. Recommendation: interleave, or replay if you must stage; never restart the schedule on a phase that lacks the earlier phase's data.

## 24. Embedding block: on the LLM's own induction items a fine-tuned encoder wins from its weights; the bank-string transfer was the uncased tokenizer; new tokens lose everywhere (MODEL-3, MODEL-5, BASE-4, BASE-5, GRAPH-5)

*[Later: the heading's tokenizer explanation is withdrawn in 24.7: the cased EmbeddingGemma transfers at 77.1, so the mechanism is the encoder's robustness to case, not an uncased tokenizer.]*

Date: 2026-09-16. Code: `uv run python scripts/exp_embed_block.py {minilm,bge,qwen3} all` (new; the section 4 and 6 embedding experiments re-implemented on top of sentence-transformers so each encoder's own pooling is used, plus the baselines); data `results/embed_block_{minilm,bge,qwen3}.json`, per-item records `results/per_item/embed_block_<enc>.universe_{frozen,trained}.jsonl` on the frozen induction items; the pairing against the LLM arms is by item id on those files (the +/- counts and sign tests in 24.1 and 24.3).

Sections 4 and 6 ran every embedding experiment on all-MiniLM-L6-v2 (22M parameters, uncased WordPiece, mean pooling), scored prototypes on 300 random trials rather than on the ladder the LLM arms answer, and had no baseline below the fine-tuned centroid (BASE-4). The new-token conditions were given mean-of-subword or random initialisation and nothing else (BASE-5), and the 70.8% bank-string transfer was attributed to shared subwords without a cased tokenizer in the comparison (MODEL-3, MODEL-5). The graph memo added a text-initialised inductive KGE as the baseline that a relation scorer would add over prototypes (GRAPH-5). This step runs all of it on three encoders:

| encoder | parameters | pooling | tokenizer | note |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 22M, 384-d | mean | uncased WordPiece | the section 4 and 6 encoder |
| bge-base-en-v1.5 | 109M, 768-d | CLS | uncased WordPiece | the survey listed it as cased; its tokenizer lower-cases |
| Qwen3-Embedding-0.6B | 596M, 1024-d | last token | cased Qwen BPE | "Elrholm" (El / r / holm) and "ELRHOLM" (EL / RH / OL / M) share no token |

Three parts, each with the same seed and the same data as the original experiment. Universe: contrastive name-to-attribute-text training as in section 6.3 (8 epochs), then (a) the frozen ladder induction items scored by the query name's nearest demo name, each demo being one labelled card, so that every item has an LLM answer and an embedding answer on the same demos; (b) the section 6.3 protocol for continuity; (c) the 8-way type task with five baselines over ten random splits of the seen species (8 train names per type): centroid, logistic regression, and SetFit on the frozen and on the trained encoder. Merchant: section 4's zero-shot / fine-tune / new-token conditions plus the BASE-5 variants (N(mu, Sigma) initialisation, embedding-row warm-up, MOSAIC's joint MLM stage for the tied-embedding encoders, and for the cased encoder both case forms as tied tokens), scored as before on bare names, bank strings and descriptions, with a logistic-regression head on the name embeddings as the BASE-4 baseline. KGE: a DistMult scorer per relation over the encoder's entity embeddings, trained jointly on the training merchants' three relations and the held-out merchants' sells and located-in triples only, so the held-out category is a products-to-category bridge; against it, centroid and logistic regression on the same encoder and a relation-free contrastive control on the same triples.

**Table 24.1: the frozen induction items, embedding prototype (query's nearest demo card) against the LLM arms (accuracy %)**

| level | MiniLM frozen | MiniLM trained | bge-base frozen | bge-base trained | Qwen3-Emb frozen | Qwen3-Emb trained | LLM base | LLM A | LLM C | LLM C + field guide |
|---|---|---|---|---|---|---|---|---|---|---|
| Timmy k=3, nonsense labels (3-way) | 38.1 | 79.4 | 31.2 | 94.4 | 38.1 | 92.5 | 35.6 | 40.0 | 61.9 | 70.6 |
| real-name labels (3-way) | 36.2 | 76.2 | 32.5 | 93.8 | 32.5 | 91.9 | 38.1 | 76.2 | 77.5 | 73.8 |
| k=2 (2-way) | 55 | 86.9 | 45.6 | 96.2 | 53.8 | 93.1 | 53.1 | 55.6 | 76.2 | 81.9 |
| k=4 (4-way) | 24.4 | 62.5 | 26.9 | 86.2 | 27.5 | 91.2 | 27.5 | 41.9 | 51.9 | 60.6 |
| weakness (3-way) | 35 | 77.5 | 27.5 | 93.1 | 31.9 | 93.1 | 35.0 | 46.2 | 60.0 | 67.5 |
| habitat (3-way) | 31.2 | 53.1 | 26.9 | 72.5 | 35 | 80 | 31.2 | 30.6 | 29.4 | 65.6 |
| held-out species (3-way) | 41.7 | 39.6 | 46.9 | 25 | 31.2 | 39.6 | 39.6 | 22.9 | 29.2 | 72.9 |

**Table 24.2: section 6.3 protocol on the three encoders, trained (frozen in brackets)**

| measure | MiniLM | bge-base | Qwen3-Emb |
|---|---|---|---|
| canonical type text, 8-way | 99.3 [10.3] | 100 [11.8] | 100 [14] |
| synonym text, 8-way | 44.9 [13.2] | 61 [11.8] | 39.7 [12.5] |
| prototype type k=1 (3-way) | 79.3 [36] | 93 [32] | 94.3 [33.7] |
| prototype type k=3 | 90.3 [41.7] | 100 [32.7] | 99.3 [37] |
| prototype weakness k=3 | 90.7 [31.3] | 99.7 [33.3] | 99.7 [31] |
| prototype habitat k=1 | 46 [30.7] | 74.7 [32.3] | 78.7 [32.7] |
| prototype habitat k=3 | 60.7 [30.3] | 90.3 [30.7] | 95.7 [31] |
| prototype type k=1, held-out species | 34.3 [43.3] | 36 [44.7] | 39.3 [39.7] |

**Table 24.3: 8-way type of seen species from the name, 10 random splits (8 train names per type), mean +- sd**

| encoder | frozen centroid | frozen logreg | frozen SetFit | trained centroid | trained logreg | trained SetFit |
|---|---|---|---|---|---|---|
| MiniLM | 13.6 +- 3.1 | 13.2 +- 2.8 | 13.6 +- 2.9 | 83.7 +- 2.3 | 83.4 +- 2.6 | 76 +- 2.4 |
| bge-base | 11.4 +- 3.2 | 12.1 +- 3.5 | 13.6 +- 2.4 | 99.4 +- 0.7 | 99.7 +- 0.6 | 99 +- 1.3 |
| Qwen3-Emb | 12.6 +- 3.3 | 13.2 +- 3.1 | 12.1 +- 3.5 | 99.9 +- 0.4 | 100 +- 0 | 92.6 +- 3.6 |

**Table 24.4: merchant category (12-way, chance 8.3), nearest category text; logistic-regression head on names in the last two columns**

| encoder | condition | name train | bank train | desc train | name held-out | bank held-out | desc held-out | logreg bank train | logreg bank held-out | bank strings hitting a new token |
|---|---|---|---|---|---|---|---|---|---|---|
| MiniLM | zero_shot | 6.2 | 7.3 | 99.0 | 29.2 | 4.2 | 95.8 | 35.4 | 12.5 | - |
| MiniLM | ft_subword | 100.0 | 69.8 | 100.0 | 12.5 | 12.5 | 100.0 | 63.5 | 12.5 | - |
| MiniLM | ft_newtok_mean | 100.0 | 28.1 | 100.0 | 12.5 | 4.2 | 100.0 | 28.1 | 4.2 | 87.5 |
| MiniLM | ft_newtok_gauss | 100.0 | 30.2 | 100.0 | 0.0 | 8.3 | 100.0 | 32.3 | 12.5 | 87.5 |
| MiniLM | ft_newtok_warm | 100.0 | 55.2 | 100.0 | 16.7 | 8.3 | 100.0 | 58.3 | 12.5 | 87.5 |
| MiniLM | ft_newtok_mosaic | 99.0 | 24.0 | 100.0 | 12.5 | 12.5 | 100.0 | 26.0 | 8.3 | 87.5 |
| bge-base | zero_shot | 8.3 | 5.2 | 100.0 | 4.2 | 12.5 | 100.0 | 31.2 | 4.2 | - |
| bge-base | ft_subword | 100.0 | 84.4 | 100.0 | 4.2 | 0.0 | 100.0 | 84.4 | 0.0 | - |
| bge-base | ft_newtok_mean | 100.0 | 30.2 | 100.0 | 4.2 | 4.2 | 100.0 | 28.1 | 8.3 | 87.5 |
| bge-base | ft_newtok_gauss | 100.0 | 49.0 | 100.0 | 16.7 | 12.5 | 100.0 | 55.2 | 16.7 | 87.5 |
| bge-base | ft_newtok_warm | 100.0 | 76.0 | 100.0 | 20.8 | 16.7 | 100.0 | 78.1 | 16.7 | 87.5 |
| bge-base | ft_newtok_mosaic | 100.0 | 35.4 | 100.0 | 4.2 | 8.3 | 100.0 | 33.3 | 0.0 | 87.5 |
| Qwen3-Emb | zero_shot | 9.4 | 8.3 | 95.8 | 8.3 | 8.3 | 91.7 | 30.2 | 8.3 | - |
| Qwen3-Emb | ft_subword | 100.0 | 26.0 | 100.0 | 8.3 | 4.2 | 100.0 | 29.2 | 4.2 | - |
| Qwen3-Emb | ft_newtok_mean | 100.0 | 12.5 | 100.0 | 4.2 | 8.3 | 100.0 | 17.7 | 0.0 | 0.0 |
| Qwen3-Emb | ft_newtok_gauss | 100.0 | 8.3 | 100.0 | 8.3 | 8.3 | 100.0 | 7.3 | 4.2 | 0.0 |
| Qwen3-Emb | ft_newtok_warm | 100.0 | 6.2 | 100.0 | 8.3 | 8.3 | 100.0 | 7.3 | 12.5 | 0.0 |
| Qwen3-Emb | ft_newtok_tied | 100.0 | 13.5 | 100.0 | 4.2 | 8.3 | 100.0 | 14.6 | 8.3 | 87.5 |

**Table 24.5: GRAPH-5, held-out merchant category (12-way) after joint training on all triples but the held-out category triples**

| encoder | training | KGE scorer, name | KGE, bank | centroid, name | centroid, bank | logreg, name | logreg, bank | category text, name | category text, bank | (train merchants: KGE name / bank) |
|---|---|---|---|---|---|---|---|---|---|---|
| MiniLM | DistMult + encoder | 33.3 | 20.8 | 16.7 | 12.5 | 12.5 | 12.5 | 37.5 | 12.5 | 89.6 / 39.6 |
| MiniLM | pairs control (no relations) | - | - | 16.7 | 8.3 | 16.7 | 4.2 | 41.7 | 16.7 | - / - |
| bge-base | DistMult + encoder | 91.7 | 58.3 | 83.3 | 45.8 | 83.3 | 41.7 | 95.8 | 62.5 | 99.0 / 72.9 |
| bge-base | pairs control (no relations) | - | - | 83.3 | 37.5 | 83.3 | 37.5 | 87.5 | 54.2 | - / - |
| Qwen3-Emb | DistMult + encoder | 100.0 | 20.8 | 100.0 | 20.8 | 100.0 | 20.8 | 100.0 | 20.8 | 100.0 / 24.0 |
| Qwen3-Emb | pairs control (no relations) | - | - | 91.7 | 25.0 | 91.7 | 20.8 | 95.8 | 29.2 | - / - |

### 24.1 On the LLM's own items, a fine-tuned 22M encoder beats the 3B LLM's weights and matches its field guide

Table 24.1 is the comparison BASE-4 asked for. Each Timmy item gives k labelled cards and a query; the LLM reads them in a prompt, the encoder embeds the k card names and the query name and answers with the nearest card. Same 160 items per level, same demos, same query, paired. After the section 6.3 contrastive training (name to attribute text, 8 epochs, 20 seconds), MiniLM's prototype answers the nonsense-label Timmy items at 79.4 where arm C answers 61.9 from its weights (46 items the encoder gets right and the LLM wrong against 18 the other way, sign test p = 6e-4) and 70.6 with the field guide in context (37 against 23, p = 0.09). Weakness is 77.5 against 60.0 (49 / 21, p = 1e-3); k=4 is 62.5 against 51.9 (50 / 33, p = 0.08); k=2 is 86.9 against 76.2. Real-name labels, where arm C reads 77.5, are a tie (76.2; 28 / 30). Habitat, the latent partition that no LLM arm learns from its weights (29.4, chance 33), is 53.1 for the encoder from the weights (55 / 17, p = 8e-6) and still below the LLM reading it from the field guide (65.6; 23 / 43, p = 0.02). Held-out species are at chance for every from-the-weights scorer (39.6 / 29.2 / 22.9), as they must be, and 72.9 for the LLM with the guide, which is the one thing the encoder has no counterpart for: it cannot read a card it was not trained on.

So the prototype result of section 6.3 was not an artefact of its own protocol. On the LLM's items it holds at the same level (79 against the 85 of the 300-trial protocol, which draws three cards per label rather than one), and it is the best from-the-weights induction number in the project, from a model 140 times smaller than the LLM and 19 seconds of training against 11 minutes. What the encoder does not have is the rest of the ladder: no recall question, no yes/no or pair manipulation, no answer in words. It answers "which of these cards is this one most like", which is what the induction levels ask and nothing else does.

### 24.2 The baselines under the centroid: logistic regression ties it, SetFit is worse, and the frozen encoder is chance

Table 24.3 supplies what BASE-4 said was missing. On the 8-way type task from the bare name (8 train names per type, 9 held out, ten splits), the frozen MiniLM is at chance in every form (13.6 / 13.2 / 13.6 against 12.5), because an untrained species name is a random string to an untrained encoder. After the contrastive training, the centroid reads 83.7 +- 2.3 and a logistic-regression head on the same embeddings 83.4 +- 2.6: the classifier adds nothing to the nearest centroid, which is what a well-separated embedding looks like. SetFit on top of the trained encoder (20 same-type pairs per train name, one epoch) is worse, 76.0 +- 2.4: a second contrastive stage on 64 names moves an encoder that had already placed 136 names, and the head trained on the moved embeddings loses 8 points. SetFit is the recipe for an encoder that has not seen the entities; it is not a way to improve one that has.

### 24.3 A stronger encoder makes the prototype near-perfect, latent partitions included

The section 6.3 result scales with the encoder (Tables 24.1 and 24.2). bge-base-en-v1.5 (109M) and Qwen3-Embedding-0.6B, trained the same 8 epochs on the same 1,496 pairs, answer the LLM's Timmy items at 94.4 and 92.5 (arm C: 61.9 from the weights, 70.6 with the guide), k=4 at 86.2 and 91.2 (51.9 / 60.6), weakness at 93.1 (60.0 / 67.5), and the habitat partition, which the LLM never learns from its weights, at 72.5 and 80.0 against the LLM's 65.6 with the field guide in the prompt. Under the 300-trial protocol the type prototype is at 100 / 99.3 with three cards and the habitat prototype at 90.3 / 95.7, where MiniLM reads 90.3 and 60.7. The 8-way type task from the bare name is 99.4 +- 0.7 (bge) and 99.9 +- 0.4 (Qwen3) by centroid, with logistic regression tying it (99.7, 100) and SetFit again a step below (99.0, 92.6 +- 3.6). Held-out species stay at chance for all three (25.0 to 39.6), as they should. Two things do not scale: the synonym label (61.0 on bge, 39.7 on Qwen3, 44.9 on MiniLM), which is the one form that asks the encoder to know English rather than the universe, and the held-out card, which no from-the-weights method can read.

Paired against arm C on the same items, bge and Qwen3 get 54 and 55 Timmy items right that the LLM gets wrong, against 2 and 6 the other way; on habitat 77 and 85 against 8 and 4. These are not close. For the induction levels of the ladder the answer to MODEL-3 is that the embedding side was under-powered, not the LLM side over-rated: the strongest number in section 6 was the weakest encoder's.

### 24.4 The bank-string transfer is a property of uncased tokenizers, and new tokens lose on every encoder and every pooling

Table 24.4 restates section 4's merchant result on three tokenizers. With the original tokenizer (`ft_subword`), the two uncased WordPiece encoders carry the merchant's category from the mixed-case training sentences to the upper-case bank string it never saw at 69.8 (MiniLM, section 4's 70.8) and 84.4 (bge-base). The cased Qwen BPE encoder, trained on the same sentences to the same 100 on names and descriptions, reads the bank strings at 26.0. The mechanism is the one MODEL-4 and section 13 described for the LLM: "Elrholm" is El / r / holm and "ELRHOLM" is EL / RH / OL / M, no token in common, and the rest of the bank string (DEBIT CARD PURCHASE, STORE 4970, TUCSON AZ) is likewise a sequence of fragments the encoder never saw in training, where the uncased tokenizers lower-case it into the familiar ones. Section 4's "subwords carry knowledge across formats" is therefore a statement about uncased tokenizers; on a cased one the knowledge stays in the case it was taught in. The survey listed bge-base-en-v1.5 as cased; its tokenizer lower-cases, which is why it behaves like MiniLM here and why the casing test rests on Qwen3.

BASE-5 asked whether the new-token conditions were given a fair chance. Four initialisations and schedules on each encoder say the ranking does not change. Mean-of-subword initialisation gives 28.1 / 30.2 / 12.5 on the bank strings (MiniLM / bge / Qwen3, against 69.8 / 84.4 / 26.0 without new tokens); N(mu, Sigma) rows 30.2 / 49.0 / 8.3; an embedding-row-only warm-up before the full fine-tune 55.2 / 76.0 / 6.2; MOSAIC's joint MLM stage on the tied-embedding encoders 24.0 / 35.4. The warm-up is the one variant that matters, recovering most of the gap on the uncased encoders (14 and 8 points short of the original tokenizer), and it still loses. On Qwen3 the added token never fires on an upper-case string (0% of bank strings contain it), and when both case forms are added as tied tokens it fires on 87.5% of them and the accuracy is 13.5: the token identity was not the obstacle, the upper-case context around it is. MODEL-5's question, whether the failure was specific to mean pooling, has its answer in the same table: CLS pooling (bge) and last-token pooling (Qwen3) fail with new tokens the same way. The new-token verdict of section 4 stands on three encoders, three poolings and five initialisations, with the caveat that the row warm-up closes most of the gap and that MOSAIC's regime (thousands of occurrences per token) is not this one (14 to 20 texts per merchant).

The logistic-regression head (BASE-4) trained on the name and sentence embeddings and scored on the bank strings matches the nearest-category-text reading within a few points on every encoder and condition (63.5 against 69.8, 84.4 against 84.4, 29.2 against 26.0); it is not a way to get more out of the same embeddings. Held-out merchants are at chance from their names in every condition, as their names are random strings, and at 100 from their descriptions, which name their products.

### 24.5 A relation scorer adds nothing to an encoder that already bridges (GRAPH-5)

Table 24.5 is the text-initialised inductive KGE. The encoder embeds merchants, products, cities and category texts from their strings; a DistMult vector per relation scores (head, relation, tail) and the two are trained jointly on every triple of the 96 training merchants and on the sells and located-in triples of the 24 held-out ones, whose category is never stated. Because the category text lists the category's products, a held-out merchant's category is reachable by a two-hop bridge (merchant sells product, product is in the category text), and the question is whether the relation structure finds it better than the encoder's own similarity. It does not. On bge, the scorer reads the held-out category from the name at 91.7 and the plain nearest category text on the same jointly trained encoder at 95.8; on Qwen3, 100 and 100; on MiniLM, 33.3 and 37.5. The relation-free control, the same triples trained as (head text, tail text) contrastive pairs, is at 87.5 / 95.8 / 41.7 by the same nearest-text reading, and centroid and logistic regression over the training merchants' names are at 83.3 / 100 / 16.7. Whatever the bridge needs, the contrastive encoder does when it is trained on the merchant-to-product pairs; a per-relation bilinear form on top reorders nothing. From bank strings the numbers fall to 58.3 (bge) and 20.8 (Qwen3, the cased tokenizer again). GRAPH-5's premise, that SimKGC-style text initialisation beats structure-only KGE on sparse description-rich graphs, is not in question; what this shows is that with text initialisation the scorer is the part that can be dropped.

### 24.6 What the block says

- MODEL-3: the embedding model was the under-powered half. On the LLM's own induction items, bge-base and Qwen3-Embedding answer from their weights at 92 to 94 (Timmy), 86 to 91 (k=4) and 72 to 80 (the habitat partition the LLM never learns), against arm C's 62 / 52 / 29 from the weights and 71 / 61 / 66 with the field guide; MiniLM at 79 / 63 / 53 already beats the LLM's weights. For "which of these cards is this one most like", the encoder is the tool, and the size that matters is the encoder's, not the LLM's.
- MODEL-5 and BASE-5: the new-token failure is not mean pooling's, not the initialisation's and not the schedule's; a row warm-up recovers most but not all of it, and on a cased tokenizer the added token cannot reach the upper-case format at all.
- BASE-4: the frozen encoders are at chance on the universe (the names are strings), the trained centroid is matched by logistic regression and not improved by SetFit, and every number in section 6.3 reproduces on identical items with the LLM.
- GRAPH-5: no; the text-initialised encoder bridges products to category on its own.
- What does not transfer to the encoder: recall in words, manipulation, and reading a card it was never trained on (held-out species at chance everywhere but the LLM with the guide).

### 24.7 Addendum (2026-09-16, evening): EmbeddingGemma-300m and gte-modernbert-base, and a correction to 24.4

Two more encoders ran through the same block at the owner's request once the gated Google models could be fetched: google/embeddinggemma-300m (308M, mean pooling plus two dense layers, cased Gemma SentencePiece: "Elrholm" is El / r / holm and "ELRHOLM" is EL / RH / OL / M) and Alibaba-NLP/gte-modernbert-base (149M, CLS pooling, cased ModernBERT BPE with the same split). Data `results/embed_block_{egemma,gtemb}.json`, per-item files as before; every trained encoder of this block is now saved under `models/adapters/embed_<encoder>_<condition>` (the earlier three were re-run with saving on under the tag `saved`).

**Universe.** Both are at the level of bge-base and Qwen3-Embedding on the LLM's induction items, and above them on the latent partition: Timmy 91.9 / 93.1 (EmbeddingGemma / gte-modernbert), k=4 87.5 / 89.4, weakness 95.0 / 92.5, habitat 81.9 / 77.5 (bge 72.5, Qwen3 80.0; the LLM with the field guide 65.6), held-out species at chance (39.6 / 38.5). The 8-way type task from the bare name is 99.9 by centroid for both, logistic regression ties it, and SetFit is 88.9 (EmbeddingGemma) and 99.7 (gte-modernbert). Frozen, both are at chance everywhere, as the others were.

**Merchants, and the correction.** Section 24.4 concluded that the bank-string transfer was a property of uncased tokenizers, from two uncased encoders that transfer (69.8, 84.4) and one cased encoder that does not (Qwen3-Embedding, 26.0). The two cased encoders here split: gte-modernbert transfers at 21.9 with its original tokenizer, Qwen3's number, and EmbeddingGemma at 77.1, the uncased encoders' number, with a tokenizer that shares no piece between the two case forms. So the tokenizer's casing is not the mechanism. What the four encoders share is whether they place a word and its upper-case form near each other before any training: EmbeddingGemma was trained to (its retrieval training covers case and script variants), the uncased models get it from their tokenizer, and Qwen3-Embedding and gte-modernbert have neither. The statement that survives is that the transfer needs an encoder for which case is not a feature, by tokenizer or by training, and that a cased BPE alone is neither necessary nor sufficient for the failure. New tokens lose on both new encoders as on the others (mean init 32.3 / 14.6, warm-up 13.5 / 8.3; the tied case forms 62.5 / 22.9, where EmbeddingGemma's 62.5 is the only new-token condition on any encoder above half of its subword baseline), and the GRAPH-5 scorer again adds nothing over the encoder's own bridge (held-out category from the name 100 / 100 by the scorer and 100 / 100 by nearest category text; from the bank string 58.3 / 12.5, the same casing split).

**Reading.** For the prototype task, any modern encoder above 100M parameters fine-tuned for a few minutes is at 90 to 94 on the LLM's items; the choice between them is the merchant-string behaviour, where EmbeddingGemma is the only cased encoder that reads the upper-case format, and bge-base the best overall (84.4). The section 24.6 conclusions stand with 24.4's mechanism restated.

## 25. Retrieval: retrieved context beats the oracle, RAFT training adds a little, a neighbour list beats both, and the merchant record is found at 100% (BASE-1, REAL-2, GRAPH-7)

Date: 2026-09-16. Code: `ai_experiments.retrieval` (new: a fine-tuned MiniLM index over the field guide and over the merchant records, with the context builders), `RET=1 PERIODIC=200 uv run python scripts/exp_curriculum.py Cr` (arm C-RAFT), `RET=1 RUN_TAG=ret EVAL_ONLY=1 ADAPTER_NAME=... uv run python scripts/exp_curriculum.py {C,P2}` and `RET=1 RUN_TAG=ret ... base` (the retrieved and neighbour-list conditions on the existing adapters), `uv run python scripts/exp_retrieval_merchants.py` (REAL-2). Data `results/curriculum_Qwen2.5-3B_{Cr_p200,base_ret,C_ret,P2_ret}.json` with per-item files, `results/retrieval_merchants.json`, `results/retrieval_universe.json` (the universe retriever's recall).

Every with-context number so far was an oracle: the exact field-guide entry of every species in the item, or the merchant's own record, placed in the prompt (section 8's `ctx_frac = 0.5` episodes, the "+ field guide" columns, section 4's `incontext`). BASE-1 noted that RAFT was recommended twice and never run, and that oracle-only context is RAFT's worst configuration: a model trained on gold-only context can score below its no-context number when fed retrieved context with distractors. REAL-2 noted that recall@k of a bank string to its merchant record was never measured, so "retrieval" had never been tested end to end. GRAPH-7 asked whether the LLM uses entity-attribute-entity structure when the context is a neighbour list with no attribute words.

The retriever is all-MiniLM-L6-v2 fine-tuned as in section 24 (name to attribute text on the seen species, 8 epochs, 20 seconds), indexing every species' field-guide entry, held-out species included, plus the 2,752 training texts. Three context conditions on the frozen ladder: `oracle` (as before), `retrieved` (the top-3 documents per name in the item, right or wrong), `neighbours` (per name, three co-typed seen species and nothing else). Four models: the base model, arm C, the distilled arm P2, and the new arm Cr, which is arm C with its context episodes carrying RAFT context instead of the oracle entries: per name, the gold entry with probability 0.8 plus the two nearest non-gold documents, shuffled. On the merchant side, the fine-tuned MiniLM indexes the 120 canonical records; recall@1 and @5 from the bank string and from the bare name, then the untrained Qwen2.5-0.5B and -3B score the 12-way category with the top-1 record in the prompt, right or wrong, against the oracle record and no context.

**Table 25.1: the ladder under four contexts, accuracy % (none / oracle field guide / retrieved top-3 per name / neighbour list)**

| level | base | C | P2 | Cr |
|---|---|---|---|---|
| recall, trained fmt | 13.1 / 100 / 100 / - | 100 / 100 / 100 / - | 94.4 / 100 / 100 / - | 100 / 100 / 100 / - |
| yes/no | 42.5 / 100 / 100 / - | 77.5 / 98.8 / 100 / - | 58.8 / 100 / 100 / - | 96.2 / 98.8 / 100 / - |
| pair | 51.2 / 82.5 / 77.5 / - | 81.2 / 76.2 / 80 / - | 48.8 / 56.2 / 65 / - | 87.5 / 82.5 / 85 / - |
| Timmy k=3 | 35.6 / 49.4 / 41.9 / 36.9 | 61.9 / 70.6 / 78.1 / 87.5 | 33.1 / 46.9 / 44.4 / 40 | 68.8 / 69.4 / 85.6 / 93.1 |
| k=4 | 27.5 / 40.6 / 39.4 / 29.4 | 51.9 / 60.6 / 74.4 / 86.2 | 25.6 / 30.6 / 36.2 / 26.2 | 61.9 / 56.9 / 77.5 / 90.6 |
| weakness | 35 / 46.9 / 53.1 / 37.5 | 60 / 67.5 / 76.9 / 79.4 | 34.4 / 42.5 / 45 / 34.4 | 64.4 / 58.1 / 81.2 / 91.2 |
| habitat | 31.2 / 38.1 / 40.6 / 29.4 | 29.4 / 65.6 / 61.9 / 25 | 28.1 / 35 / 35 / 32.5 | 28.1 / 60.6 / 65.6 / 29.4 |
| held-out species | 39.6 / 43.8 / 38.5 / 40.6 | 29.2 / 72.9 / 64.6 / 81.2 | 36.5 / 40.6 / 51 / 37.5 | 31.2 / 69.8 / 62.5 / 81.2 |
| reverse easy | 23.1 / 95.6 / - / - | 30.6 / 80.6 / - / - | 20.6 / 93.1 / - / - | 34.4 / 81.2 / - / - |
| reverse hard | 20 / 88.1 / - / - | 19.4 / 74.4 / - / - | 18.1 / 86.2 / - / - | 21.9 / 81.9 / - / - |

**Table 25.2: arm Cr against arm C (seed 0, no context unless stated)**

| measure | C | Cr |
|---|---|---|
| recall, trained fmt | 100 | 100 |
| yes/no | 77.5 | 96.2 |
| pair | 81.2 | 87.5 |
| Timmy k=3 | 61.9 | 68.8 |
| weakness | 60 | 64.4 |
| ICL symbol | 78.6 | 78.2 |
| ICL natural | 87 | 87 |
| ARC-Easy | 58.5 | 60 |
| WikiText ppl | 22.787 | 22.777 |
| training minutes | 50.3 | 10.4 |
| loss tokens E | - | - |
| loss tokens Er | - | 19637 |

Cr periodic recall_fmt / Timmy: 0: 12.5 / 40.0, 200: 25.0 / 37.5, 400: 100.0 / 47.5, 600: 97.5 / 50.0

**Table 25.3: merchant record retrieval, recall@1 / @5 of the merchant's own record (120 records)**

| retriever | bank string, train | bank string, held-out | name, train | name, held-out |
|---|---|---|---|---|
| zero_shot | 78.1 / 96.9 | 87.5 / 95.8 | 99.0 / 100.0 | 95.8 / 100.0 |
| category_tuned | 55.2 / 74.0 | 29.2 / 41.7 | 89.6 / 100.0 | 29.2 / 54.2 |
| record_tuned | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |

ret1 gold hits over all items: 100.0

**Table 25.4: end-to-end category (12-way, chance 8.3), untrained base models, record-tuned retriever**

| model | context | bank string, train | bank string, held-out | clean name, train | clean name, held-out |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | none | 8.3 | 8.3 | 9.4 | 8.3 |
| Qwen2.5-0.5B | oracle | 8.3 | 12.5 | 67.7 | 62.5 |
| Qwen2.5-0.5B | ret1 | 8.3 | 12.5 | 67.7 | 62.5 |
| Qwen2.5-0.5B | ret3 | 8.3 | 8.3 | 24.0 | 25.0 |
| Qwen2.5-3B | none | 7.3 | 8.3 | 8.3 | 8.3 |
| Qwen2.5-3B | oracle | 81.2 | 75.0 | 89.6 | 91.7 |
| Qwen2.5-3B | ret1 | 81.2 | 75.0 | 89.6 | 91.7 |
| Qwen2.5-3B | ret3 | 46.9 | 41.7 | 70.8 | 62.5 |

### 25.1 Retrieved context beats the oracle entry, and no arm scores below its no-context number with distractors

Table 25.1's third figure in each cell is the ladder with the retriever's top-3 documents per name in the prompt, right or wrong. RAFT's warning was that a model trained on gold-only context can fall below its no-context score when the context is retrieved. Arm C, trained on gold-only context (section 8's `ctx_frac = 0.5` with the exact entries), does not: with retrieved context it is above its no-context number on every level and above its oracle number on induction (Timmy 78.1 against 70.6 oracle and 61.9 bare; k=4 74.4 against 60.6; weakness 76.9 against 67.5). The base model reads retrieved context about as well as the oracle (41.9 against 49.4 on Timmy, 53.1 against 46.9 on weakness). The retrieved context is not a degraded oracle here; it is a richer one. The top-3 for a seen species is three of its own paraphrased training texts (`results/retrieval_universe.json`: every top-3 document of every seen species is about it, and the compact entry itself ranks below the longer texts, top-1 for 0% of them), so the prompt states each fact three times in different words, and the trained model reads that better than the single entry it was trained on. The distractors RAFT worried about arrive mainly for the held-out species, whose names the retriever never saw: their entries are top-1 for 87.5% of them (the name appears verbatim in the entry, and nothing else in the index mentions it) and the other 12.5% get three documents about other species, and the held-out level with retrieved context is 64.6 for arm C against 72.9 with the oracle, the one place retrieval costs anything.

Arm Cr, arm C with RAFT context in its training episodes (per name the gold entry with probability 0.8 plus the two nearest non-gold documents, shuffled), trains in 10 minutes to the same from-the-weights numbers as C (recall 100, ICL 78.2 / 87.0 against 78.6 / 87.0, WikiText 22.78 against 22.79) with manipulation and induction a few points higher (96.2 / 87.5 / 68.8 against 77.5 / 81.2 / 61.9; single seeds, inside arm C's seed band of section 20). With retrieved context it is the best reader of the four: Timmy 85.6, k=4 77.5, weakness 81.2, against C's 78.1 / 74.4 / 76.9. That is the RAFT effect, 3 to 7 points on top of an arm that was not hurt by distractors in the first place. Its oracle numbers are C's (69.4 / 56.9 / 58.1 against 70.6 / 60.6 / 67.5), so training on distractors did not cost the clean case. Recommendation for BASE-1: train the context episodes on retrieved context when retrieval is what the deployed model will see; the gain is modest and the cost is none.

### 25.2 The distilled adapter cannot read context

P2 (section 18: the with-context ranking distilled into the bare format, 91.9 trained-format recall) is the arm that loses with context. With the oracle field guide in the prompt it answers Timmy at 46.9 and k=4 at 30.6 where the base model answers 49.4 and 40.6, and the pair manipulation at 56.2 against the base's 82.5; retrieved context does not repair it (44.4 / 36.2 / 65.0). Only recall and the yes/no question, the forms it was distilled on, are at 100 with context. Distilling the with-context answers into the bare prompt taught the model to answer without looking, and it keeps not looking when the answer is in front of it. Section 18 reported that the distilled arm stays inside its five question forms; this adds that it also stops using the context those forms came from. For any deployment where retrieval is available, P2 is the wrong arm.

### 25.3 A neighbour list with no attribute words is the best context for the trained model (GRAPH-7)

The fourth figure is GRAPH-7's probe: for each name in a Timmy item, three co-typed seen species and nothing else, no type, no weakness, no habitat word. The expectation was a number below the field-guide oracle, since the list only says "these belong together" and never what they are. For the base model the list is worth nothing (36.9 on Timmy, 29.4 on k=4, chance). For the episode-trained arms it is the best context of the four: C 87.5 on Timmy, 86.2 on k=4, 79.4 on weakness (its oracle numbers are 70.6 / 60.6 / 67.5) and 81.2 on held-out species (oracle 72.9); Cr 93.1 / 90.6 / 91.2 / 81.2. The trained model knows the species that share a type, so two demos whose lists overlap are the same type and the query's list places it: it is using the entity-attribute-entity structure the memo asked about, and doing so more reliably than it reads the attribute word off an entry. Habitat, the partition the lists do not encode, stays at chance under the list (25.0 / 29.4) where the oracle gives 65.6 / 60.6. So the answer to GRAPH-7 is yes, for the arms that learned the graph, and the practical reading is that a retriever returning *related entities* is worth more to this model than one returning the entities' descriptions, provided the model has been trained on the entities.

### 25.4 Merchant records: a retriever tuned for records is perfect at 120, and the oracle number is then the end-to-end number (REAL-2)

Table 25.3 is the number REAL-2 said was never measured. Over the 120 canonical merchant records, the zero-shot MiniLM already finds a bank string's own record at recall@1 78.1 (training merchants) and 87.5 (held-out), because the record states the merchant's name and the uncased tokenizer maps ELRHOLM and Elrholm to the same subwords (section 24.4). The encoder fine-tuned as in section 4, sentences and names toward the *category* text, is a worse record retriever (55.2 / 29.2): it was trained to make merchants of one category alike, which is the opposite of what record retrieval needs. Fine-tuned toward the *record* (the same anchors, the merchant's own fact as the positive), recall@1 is 100 on every split and every query form, held-out bank strings included, whose merchants the retriever never saw in training. With 120 records this is a solved problem for an uncased 22M encoder; the 10k-record version with Zipf frequencies and real-style truncations is PLAN row 22.

Table 25.4 is retrieval end to end. Because the record-tuned retriever's top-1 is the gold record for every item, `ret1` equals the oracle exactly: Qwen2.5-3B, untrained, reads the category off the retrieved record at 81.2 / 75.0 from the bank string (train / held-out) and 89.6 / 91.7 from the clean name, against 7 to 8 with no context. Two things in the table are not free. Qwen2.5-0.5B cannot use the record when the query is a bank string (8.3 with the oracle record in the prompt; section 4 had 14 with a different prompt), while it can from the clean name (67.7): the small model does not connect the upper-case string to the record even when the record is right there. And giving the 3B model the top-3 records instead of the top-1 halves its accuracy (46.9 / 41.7 on bank strings, 70.8 / 62.5 on names): two irrelevant records beside the right one are distractors it cannot ignore, which is the RAFT failure mode in its pure form, on an untrained model. The deployment reading is: retrieve one record, and make the retriever earn it.

### 25.5 What the step says

- BASE-1: run. Arm C, trained on oracle context only, does not fall below its no-context numbers with retrieved context; it gains (retrieved top-3 beats the oracle entry on induction, 78 against 71 on Timmy), because retrieval returns the paraphrased texts. Training on RAFT context (arm Cr) adds 3 to 7 points on the retrieved condition at no cost elsewhere; a second seed would make it a claim.
- REAL-2: measured. Record-tuned MiniLM: recall@1 100 from bank strings at 120 records (zero-shot 78 to 88, category-tuned 29 to 55); end to end equals the oracle on Qwen2.5-3B (81 / 75 on bank strings), top-3 halves it, the 0.5B model cannot read the record from a bank string.
- GRAPH-7: yes. A list of co-typed species per name, with no attribute words, is the best context for the episode-trained arms (Timmy 87.5 for C, 93.1 for Cr; oracle 70.6 / 69.4) and worthless for the base model (36.9): the trained model uses the entity-entity structure. It does nothing for the partition the list does not encode (habitat at chance).
- The distilled arm P2 reads context worse than the base model on every induction and manipulation level; do not deploy it with retrieval.

## 26. Scale: the mixture's induction and ICL gains hold at 1.5B and 7B, the manipulation trade vanishes at 7B, and the sequential arm forgets a third of the facts at 7B (MODEL-1)

*[Later: 'vanishes at 7B' is a 4-point gap from one seed against a 12.7-point seed sd on the manipulation levels (section 20).]*

Date: 2026-09-16. Code: `PERIODIC=200 uv run python scripts/exp_curriculum.py {base,A,C,D} Qwen/Qwen2.5-1.5B` and `LOAD_4BIT=1 PERIODIC=200 uv run python scripts/exp_curriculum.py {base,A,C,D} Qwen/Qwen2.5-7B` (QLoRA: NF4 base weights, the same LoRA r=64 on all linear layers, the same 800 x 16 sequences, seed 0, fast path); data `results/curriculum_Qwen2.5-{1.5B,7B}_{base,A_p200,C_p200,D_p200}.json` with per-item files and learning curves; the 3B column is the section 15 rerun (seed 0) with section 20's three-seed means beside it.

MODEL-1 asked why the 7B model was in none of the graphs. Every knowledge and curriculum result before this section is Qwen2.5-3B (or 0.5B in section 4), and the survey's reading was that label induction from symbol tuning is a scale phenomenon (never tested below 8B; at 8B it cost natural-label and benchmark points), so the 3B result could not say whether interleaving still wins, or is still needed, at another size. This section runs the four arms of section 8 at 1.5B and 7B with the same data, steps and hyperparameters, and reports the natural-label ICL column and the knowledge benchmark (ARC-Easy) separately, as the survey asked. The 7B runs use QLoRA, which is the only way a 7B model trains on this card; section 7 measured the 4-bit forward at about the bf16 3B's speed, and the adapters are the same shape.

**Table 26.1: the four arms at three scales, seed 0, from the weights (accuracy %; 3B three-seed mean +- sd in brackets where section 20 has it)**

| measure | 1.5B base | 1.5B A | 1.5B C | 1.5B D | 3B base | 3B A | 3B C | 3B D | 7B base | 7B A | 7B C | 7B D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| recall, trained fmt | 13.1 | 100 | 100 | 91.9 | 13.1 | 100 [100.0 +- 0.0] | 100 [100.0 +- 0.0] | 93.1 [91.5 +- 3.5] | 10.6 | 100 | 100 | 65.6 |
| recall, bare | 19.4 | 20.6 | 16.2 | 64.4 | 18.1 | 23.1 [21.2 +- 1.6] | 19.4 [18.4 +- 1.3] | 58.8 [44.2 +- 20.6] | 18.8 | 19.4 | 19.4 | 65.6 |
| yes/no | 51.2 | 85 | 55 | 55 | 42.5 | 100 [98.3 +- 2.9] | 77.5 [80.0 +- 12.7] | 63.8 [70.9 +- 8.3] | 45 | 97.5 | 93.8 | 88.8 |
| pair | 50 | 68.8 | 53.8 | 50 | 51.2 | 88.8 [90.8 +- 1.9] | 81.2 [78.7 +- 12.7] | 72.5 [68.7 +- 16.0] | 50 | 95 | 91.2 | 80 |
| Timmy k=3 | 35.6 | 29.4 | 55.6 | 48.8 | 35.6 | 40 [35.9 +- 3.6] | 61.9 [63.8 +- 9.2] | 53.1 [59.2 +- 7.0] | 33.1 | 55 | 73.8 | 71.2 |
| real-name labels | 33.1 | 56.9 | 56.9 | 56.2 | 38.1 | 76.2 | 77.5 | 66.2 | 35 | 80.6 | 95.6 | 89.4 |
| k=4 | 28.8 | 30.6 | 45 | 41.2 | 27.5 | 41.9 [35.2 +- 6.9] | 51.9 [57.9 +- 11.0] | 51.9 [54.6 +- 7.0] | 27.5 | 44.4 | 66.2 | 62.5 |
| weakness | 31.9 | 38.1 | 46.2 | 50 | 35 | 46.2 [41.5 +- 4.1] | 60 [59.6 +- 11.9] | 54.4 [55.8 +- 6.7] | 38.1 | 60 | 78.8 | 73.1 |
| habitat | 33.8 | 33.8 | 36.2 | 31.9 | 31.2 | 30.6 [32.3 +- 2.4] | 29.4 [33.1 +- 3.3] | 41.9 [39.8 +- 7.7] | 30.6 | 36.2 | 36.9 | 38.8 |
| held-out species | 40.6 | 29.2 | 32.3 | 36.5 | 39.6 | 22.9 [26.0 +- 2.8] | 29.2 [31.9 +- 3.2] | 36.5 [36.1 +- 2.6] | 46.9 | 30.2 | 33.3 | 31.2 |
| ICL symbol | 56.8 | 50 | 75 | 81.2 | 60.4 | 47.4 [50.3 +- 2.9] | 78.6 [79.0 +- 1.1] | 80.2 [79.2 +- 1.8] | 57.3 | 53.1 | 83.9 | 83.3 |
| ICL natural | 87.5 | 85.4 | 87.5 | 88 | 84.9 | 80.8 [81.1 +- 1.1] | 87 [87.5 +- 0.9] | 87.5 [88.4 +- 1.1] | 88 | 90.6 | 88 | 91.1 |
| ARC-Easy | 75.5 | 57 | 59.5 | 55.5 | 73.5 | 66 [63.7 +- 2.1] | 58.5 [64.0 +- 4.8] | 64.5 [63.7 +- 2.4] | 76.5 | 64.5 | 65.5 | 68.5 |
| WikiText ppl | 11.97 | 55.071 | 29.803 | 28.878 | 10.614 | 34.055 [39.1 +- 5.2] | 22.787 [22.6 +- 0.2] | 22.621 [21.9 +- 0.9] | 9.777 | 20.701 | 16.874 | 15.776 |
| reverse easy | 21.9 | 65 | 34.4 | 30 | 23.1 | 88.8 | 30.6 | 31.2 | 23.1 | 86.2 | 59.4 | 34.4 |
| training minutes | - | 5.3 | 6.8 | 7.9 | - | 17.2 | 50.3 | 28.3 | - | 10.2 | 20.8 | 22.4 |
| tokens/s | - | 1010 | 3077 | 2730 | - | 313 | 415 | 762 | - | 529 | 1003 | 965 |
| peak GiB | - | 4.97 | 8.09 | 8.09 | - | 8.06 | 8.71 | 8.7 | - | 11.51 | 13.62 | 13.87 |

**Table 26.2: with the field guide in context**

| measure | 1.5B base | 1.5B A | 1.5B C | 1.5B D | 3B base | 3B A | 3B C | 3B D | 7B base | 7B A | 7B C | 7B D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Timmy k=3 | 35.6 | 36.9 | 75.6 | 74.4 | 49.4 | 38.1 | 70.6 | 78.8 | 40.6 | 46.9 | 75.6 | 66.9 |
| k=4 | 28.8 | 28.8 | 66.2 | 74.4 | 40.6 | 40.6 | 60.6 | 71.9 | 28.8 | 31.9 | 70.6 | 62.5 |
| habitat | 33.8 | 31.9 | 69.4 | 67.5 | 38.1 | 30 | 65.6 | 75.6 | 35.6 | 33.1 | 71.9 | 78.8 |
| held-out species | 42.7 | 36.5 | 71.9 | 74 | 43.8 | 26 | 72.9 | 69.8 | 46.9 | 39.6 | 74 | 70.8 |
| pair | 75 | 77.5 | 56.2 | 50 | 82.5 | 91.2 | 76.2 | 73.8 | 52.5 | 90 | 73.8 | 73.8 |

**Table 26.3: the headline gaps per scale (C - A, D - C, C - base)**

| gap | measure | 1.5B | 3B | 7B |
|---|---|---|---|---|
| C - A | Timmy k=3 | +26.2 | +21.9 | +18.8 |
| C - A | yes/no | -30.0 | -22.5 | -3.7 |
| C - A | ICL symbol | +25.0 | +31.2 | +30.8 |
| C - A | ICL natural | +2.1 | +6.2 | -2.6 |
| C - A | ARC-Easy | +2.5 | -7.5 | +1.0 |
| C - A | WikiText ppl | -25.3 | -11.3 | -3.8 |
| D - C | Timmy k=3 | -6.8 | -8.8 | -2.6 |
| D - C | yes/no | +0.0 | -13.7 | -5.0 |
| D - C | ICL symbol | +6.2 | +1.6 | -0.6 |
| D - C | ICL natural | +0.5 | +0.5 | +3.1 |
| D - C | ARC-Easy | -4.0 | +6.0 | +3.0 |
| D - C | WikiText ppl | -0.9 | -0.2 | -1.1 |
| C - base | Timmy k=3 | +20.0 | +26.3 | +40.7 |
| C - base | yes/no | +3.8 | +35.0 | +48.8 |
| C - base | ICL symbol | +18.2 | +18.2 | +26.6 |
| C - base | ICL natural | +0.0 | +2.1 | +0.0 |
| C - base | ARC-Easy | -16.0 | -15.0 | -11.0 |
| C - base | WikiText ppl | +17.8 | +12.2 | +7.1 |
| A - base | Timmy k=3 | -6.2 | +4.4 | +21.9 |
| A - base | yes/no | +33.8 | +57.5 | +52.5 |
| A - base | ICL symbol | -6.8 | -13.0 | -4.2 |
| A - base | ICL natural | -2.1 | -4.1 | +2.6 |
| A - base | ARC-Easy | -18.5 | -7.5 | -12.0 |
| A - base | WikiText ppl | +43.1 | +23.4 | +10.9 |


### 26.1 What holds at every scale

Three of section 8's claims, the ones section 20 found clear at 3B, hold at 1.5B and at 7B with the same sign and about the same size (Tables 26.1 and 26.3):

- The mixture teaches induction from the weights and knowledge-only does not. Timmy k=3, C minus A: +26.2 at 1.5B, +21.9 at 3B, +18.8 at 7B; k=4 +14.4 / +10.0 / +21.8; weakness +8.1 / +13.8 / +18.8. Knowledge-only is at or near the base model's induction at 1.5B and 3B (29.4 / 40.0 against bases of 35.6 / 35.6) and above it at 7B (55.0 against 33.1), so the biggest model does some induction from the facts alone; the mixture still adds 19 points on top.
- The mixture keeps the symbol-label ICL suite and knowledge-only halves it: C minus A +25.0 / +31.2 / +30.8; A ends at 50.0 / 47.4 / 53.1 against bases of 56.8 / 60.4 / 57.3. The natural-label column, which the survey singled out as where symbol tuning cost at 8B, does not move at any scale: C is within 2 points of its base at 1.5B and 3B and at 88.0 against 88.0 at 7B.
- Every trained arm loses the knowledge benchmark and the base model's perplexity, and the size of the loss is the same at every scale: ARC-Easy -16 to -20 (1.5B), -7.5 to -15 (3B), -11 to -12 (7B) for arms A and C; WikiText +0.9 to +1.5 nats per token for A, +0.5 to +0.9 for C, with the mixture always the cheaper. QLoRA at 7B forgets no less than LoRA at 3B.

Trained-format recall is 100 for A and C at every scale and 91.9 / 93.1 / 65.6 for the sequential arm (26.3); bare-format recall stays at the base's level for A and C everywhere (the section 12 result), and held-out species stay at chance.

### 26.2 What changes with scale: manipulation, and the cost of the mixture

The one headline that does not hold across scales is arm A's manipulation advantage, and it changes in both directions. At 1.5B knowledge-only manipulates (yes/no 85.0, pair 68.8) and the mixture does not (55.0 / 53.8, chance 50 / 50): the small model cannot hold the facts and the induction skill at once, and the mixture at 45% knowledge by sequence loses the facts' usability while keeping their recall. At 3B the gap is 18 / 12 points (section 20: inside 2 sd). At 7B it is 3.7 / 3.8 (97.5 / 95.0 against 93.8 / 91.2): the model has room for both. The "trade" that section 8 described, manipulation for induction, is a small-model trade; at 7B the mixture arm is the best arm on every from-the-weights level but the two manipulation questions, and it loses those by less than the 160-item noise.

Two other scale-dependent numbers. Knowledge-only's perplexity cost falls with scale (WikiText 55.1 / 34.1 / 20.7 against bases of 12.0 / 10.6 / 9.8: +1.5 / +1.2 / +0.75 nats) while the mixture's is flat (+0.9 / +0.8 / +0.55), so at 7B the two arms are 0.2 nats apart where at 1.5B they were 0.6. And the backward direction: arm A's easy reverse lookup is 65.0 / 88.8 / 86.2 by scale, from the paraphrase set's type-before-name renderings (section 22), while the mixture reads 34.4 / 30.6 / 59.4: at 7B the mixture starts to recover the backward direction it loses at the smaller sizes.

### 26.3 The sequential arm at scale

The sequential arm is the one arm whose behaviour changes with scale in the direction section 23 predicted. Its recall curve is the same at every size: 95 to 100 at step 400 when the knowledge phase ends, then a slide through the episode phase. At 1.5B and 3B the slide stops at 91.9 and 93.1 (the shared schedule's decaying rate holds the forgetting to a few points, section 23). At 7B it does not: 100 at step 400, 65.0 at step 600, 65.6 at the end. The larger adapter (161M parameters, r=64 on 28 layers) overwrites more per step of episode-only training at the same learning rate, and the tail of the schedule is no longer enough to protect the facts. Everything else about D at 7B is good: induction 71.2 (C 73.8), ICL 83.3, the best ARC-Easy (68.5) and WikiText (15.8) of any trained arm, and the bare-format recall of 65.6 is the only bare-format number above the base in the table. The section 23 remedy, 10% knowledge replay in phase 2, is what a 7B staged run needs; without it, staging at 7B loses a third of the facts.

### 26.4 What the runs cost, and what MODEL-1 now says

Table 26.1's last rows: 1.5B trains an arm in 5 to 8 minutes at 1,000 to 3,100 tokens per second and evaluates in 1.5 minutes; 7B QLoRA trains in 10 to 21 minutes at 530 to 1,000 tokens per second and 11.5 to 13.6 GiB. Its evaluation is the cost: the first 7B mixture run scored the with-context ladder in 35 minutes at the 24 GB limit, where the driver spills to system memory instead of raising the out-of-memory error the scorer halves its batch on (section 7's warning), and the whole evaluation took 51 minutes; with a quarter of the forward budget for 4-bit models (the change is in the script) the D run evaluated in 6.4 minutes. A 7B arm is an hour, a 1.5B arm ten minutes; the 3B arm at 15 minutes remains the working size.

MODEL-1: the mixture recipe is not a 3B artefact. Interleaving still wins at 7B on induction (+19 on Timmy) and ICL (+31 on symbol labels), and it is still needed, since knowledge-only at 7B halves the ICL suite the same way it does at 3B. What 7B changes is the price: the manipulation gap between knowledge-only and the mixture, 18 points at 3B and 30 at 1.5B, is 4 points at 7B. The natural-label column is flat at every scale, which the survey's 8B symbol-tuning result had made the thing to watch, and the knowledge benchmark loses the same 10 to 15 points at every scale, which is the forgetting cost that no scale removes. Single seeds at 1.5B and 7B; the 3B seed sd (section 20) is the band to read them against.

## 27. Augmentation scaling: recall saturates at three renderings, manipulation comes from comparisons, backward recall from backward sentences, and variety costs general ability (DATA-5)

Date: 2026-09-16. Code: `KTEXTS=<spec> RUN_TAG=<spec> PERIODIC=200 uv run python scripts/exp_curriculum.py A` with `universe.knowledge_texts` (new: the descriptive templates cut to K, sentence-order permutations, reverse-direction statements, LLM-written texts), `scripts/gen_llm_texts.py` (Qwen2.5-3B-Instruct writes up to 28 texts per species; frozen as `data/processed/llm_texts_v1.json`); data `results/curriculum_Qwen2.5-3B_A_<spec>_p200.json` with per-item files and curves; the section 8 arm A rerun (`A_p200`) is the last point.

DATA-5 asked whether it is template diversity or simply the number of distinct token sequences that makes the facts extractable, and how it scales. Section 4 attributed a doubling to "augmentation" from one two-point comparison (one raw sentence against 14 templates); the survey found the "ten paraphrases" figure to be a chain of citations whose primary sources disagree (monotone gains to ten in one, a plateau at three per atomic fact in another, and Allen-Zhu and Li's numbers being about the *kind* of diversity: sentence-order permutation beats rewrites). This section holds everything fixed (arm A, knowledge texts only, 800 steps of 16 sequences, seed 0, Qwen2.5-3B) and changes the knowledge stream: the first 1, 3, 7 or 14 of the descriptive and question-form templates; then, on top of the 14, five sentence-order permutations of the six attribute sentences, four reverse-direction statements (attributes first, name last, never the L8 question form), or 16 to 28 sentences written by Qwen2.5-3B-Instruct from the field-guide facts (mean 24; a third of the generated lines were dropped for stating none of the species' attribute values literally, and the kept ones still garble a fact now and then, as generated augmentation does); and, added after the first seven runs, the 14 templates plus the section 8 stream's negative and comparative texts. At a fixed budget of 12,800 sequences, one template means 94 passes over each text and 14 templates means 7, so the sweep also trades repetition for variety. The readout is the ladder from the weights: recall in the trained format and bare, the two manipulation questions, induction, the backward L8 levels (section 22), and the general-ability costs.

**Table 27.1: arm A (knowledge only, 800 x 16 sequences, seed 0) under the knowledge-stream variants; distinct texts per species in the second row (base model in the last column)**

| measure | 1 template | 3 templates | 7 templates | 14 templates | 14 + 5 sentence-order permutations | 14 + 4 reverse-direction | 14 + 16 to 28 LLM-written | 14 + negatives and comparatives | section 8 arm A (14 + negatives + comparatives + lore) | base |
|---|---|---|---|---|---|---|---|---|---|---|
| distinct texts per species | 1 | 3 | 7 | 14 | 19 | 18 | 14 + 24 (16 to 28) | 20 | about 20 | - |
| passes per text (12,800 sequences) | 94 | 31 | 13 | 7 | 5 | 5 | - | 5 | - | - |
| sequences K | 12800 | 12800 | 12800 | 12800 | 12800 | 12800 | 12800 | 12800 | 12800 | - |
| recall, trained fmt | 73.8 | 100 | 97.5 | 100 | 100 | 100 | 100 | 100 | 100 | 13.1 |
| recall, bare | 23.1 | 53.1 | 64.4 | 21.9 | 20.6 | 18.1 | 20 | 21.9 | 23.1 | 18.1 |
| yes/no | 46.2 | 57.5 | 46.2 | 55 | 55 | 55 | 55 | 98.8 | 100 | 42.5 |
| pair | 50 | 58.8 | 50 | 50 | 50 | 50 | 50 | 93.8 | 88.8 | 51.2 |
| Timmy k=3 | 28.8 | 36.9 | 41.2 | 35.6 | 32.5 | 30.6 | 29.4 | 34.4 | 40 | 35.6 |
| weakness | 37.5 | 36.2 | 33.8 | 31.2 | 33.1 | 38.1 | 36.2 | 38.1 | 46.2 | 35 |
| reverse easy | 21.2 | 25.6 | 66.2 | 98.1 | 96.9 | 99.4 | 41.2 | 81.9 | 88.8 | 23.1 |
| reverse hard | 23.1 | 25.6 | 31.9 | 24.4 | 26.2 | 69.4 | 25.6 | 19.4 | 26.9 | 20 |
| novel choices | 11.2 | 11.2 | 14.4 | 14.4 | 13.8 | 11.9 | 13.1 | 13.1 | 12.5 | 11.2 |
| ICL symbol | 51.6 | 53.6 | 46.9 | 50.5 | 47.4 | 44.8 | 46.4 | 50.5 | 47.4 | 60.4 |
| ICL natural | 84.9 | 85.4 | 80.8 | 82.8 | 80.7 | 82.8 | 83.3 | 81.2 | 80.8 | 84.9 |
| ARC-Easy | 67.5 | 74 | 73 | 61 | 54.5 | 51.5 | 57.5 | 62 | 66 | 73.5 |
| WikiText ppl | 20.544 | 23.079 | 28.139 | 51.345 | 83.074 | 47.663 | 26.724 | 37.663 | 34.055 | 10.614 |
| training minutes | 7 | 7.1 | 7.2 | 7 | 7.2 | 7.1 | 7.1 | 7.1 | 17.2 | - |

**Table 27.2: with the field guide in context**

| measure | 1 template | 3 templates | 7 templates | 14 templates | 14 + 5 sentence-order permutations | 14 + 4 reverse-direction | 14 + 16 to 28 LLM-written | 14 + negatives and comparatives | section 8 arm A (14 + negatives + comparatives + lore) |
|---|---|---|---|---|---|---|---|---|---|
| pair | 53.8 | 80 | 62.5 | 50 | 50 | 50 | 95 | 95 | 91.2 |
| Timmy k=3 | 40.6 | 45 | 42.5 | 35.6 | 37.5 | 35.6 | 31.9 | 40 | 38.1 |
| reverse hard | 81.2 | 85 | 82.5 | 81.9 | 90 | 81.2 | 91.2 | 84.4 | 80.6 |

desc1 recall_fmt by step: 0: 12.5, 200: 55.0, 400: 67.5, 600: 67.5

desc3 recall_fmt by step: 0: 12.5, 200: 92.5, 400: 92.5, 600: 100.0

desc7 recall_fmt by step: 0: 12.5, 200: 77.5, 400: 100.0, 600: 100.0

desc14 recall_fmt by step: 0: 12.5, 200: 95.0, 400: 100.0, 600: 100.0

desc14perm recall_fmt by step: 0: 12.5, 200: 95.0, 400: 100.0, 600: 100.0

desc14rev recall_fmt by step: 0: 12.5, 200: 87.5, 400: 100.0, 600: 100.0

desc14llm recall_fmt by step: 0: 12.5, 200: 47.5, 400: 97.5, 600: 100.0

desc14cmp recall_fmt by step: 0: 12.5, 200: 90.0, 400: 100.0, 600: 100.0

full recall_fmt by step: 0: 12.5, 200: 100.0, 400: 100.0, 600: 100.0


### 27.1 Recall saturates at three templates; manipulation never comes from paraphrases

Trained-format recall (Table 27.1) is 73.8 with one template and 100 from three templates on (97.5 at seven), with the curve flat from step 400 in every variant but the first. The literature's plateau at about three renderings per fact holds here to the number. What does not come with any number of descriptive templates is the ability to use the fact: yes/no and pair manipulation are at chance for 1, 3, 7 and 14 templates (46 to 58 / 50 to 59) and stay at chance when the 14 are joined by sentence-order permutations, reverse-direction statements or up to 28 LLM-written sentences. The section 8 stream, which is the same 14 templates plus one negative ("Is N an X-type? No, N is a T-type, not X-type") and five two-species comparisons per species, reads 100 / 88.8. The ablation that isolates them, the 14 templates plus those six texts and nothing else (`desc14cmp`, 20 texts per species, 5 passes each), reads 98.8 / 93.8: the negatives and comparisons are the whole of the manipulation effect, and they also halve the perplexity cost of the 14 templates alone (37.7 against 51.3), being plain prose that dilutes the question-form share.

So DATA-5's question has two answers. For recall, distinct token sequences are what matter and three of them suffice; for manipulation, it is not diversity or count but *kind*: the model learns to compare two species and to reject a wrong type only from texts that compare and reject. Section 12 attributed manipulation to the episodes; this shows it is already in arm A's negatives and comparatives, which is where arm A's 98 / 91 (section 20) came from all along.

Bare-format recall behaves oddly and consistently: 53.1 and 64.4 at three and seven templates, then 21.9 at fourteen and 18 to 23 for every augmented variant and the section 8 stream (base 18.1). Templates 8 to 14 are the question-and-answer forms ("Question: What type is N?\nAnswer: N is a T-type."), whose answers begin with the name; once they are in the stream the bare question's cloze continuation (" T" after "Answer:") is displaced by the trained form. This is the section 12 format effect in reverse: a training format that matches the question and not the scored continuation costs the bare score.

### 27.2 Reverse-direction statements are the only augmentation that teaches the hard backward level

The easy backward level (L8, distractors of another type) follows the templates that put the type before the name: 21 to 26 at one to three templates (none do), 66.2 at seven (one does), 98.1 at fourteen (three do). The hard level (same type, the diet and region conjunction) is at chance for every variant but one: the four reverse-direction statements, which name the creature last after listing its attributes, take it from 24.4 to 69.4, with the easy level at 99.4 and everything else unchanged. Sentence-order permutation of the attribute sentences does nothing for either level (96.9 / 26.2); the LLM-written texts, which the model wrote mostly forward, dilute the type-before-name templates and pull the easy level down to 41.2. This is the survey's prediction (2510.09885, Figure 6; 2309.14402, Result 7) reproduced: the backward direction is bought with backward text and with nothing else, and section 22's masked fine-tuning result (48.1 on the hard level from one masked rendering) now has a cheaper competitor at 69.4 from four plain sentences.

### 27.3 More variety at a fixed budget costs more general ability

At 12,800 sequences the variants differ in how far they move the model off its distribution, and the direction is the opposite of what "augmentation protects the model" would suggest. WikiText perplexity (base 10.6): one template 20.5, three 23.1, seven 28.1, fourteen 51.3, fourteen plus permutations 83.1, plus reverse statements 47.7, plus LLM sentences 26.7. ARC-Easy (base 73.5): 67.5 / 74.0 / 73.0 at one to seven templates, then 61.0, 54.5, 51.5, 57.5. Seven forward templates keep ARC-Easy at the base and cost 1 nat on WikiText; the question-form templates and the attribute-list permutations, which are the least like natural prose, cost the most; the LLM-written prose costs the least of the 14-template variants (26.7 against 51.3) because it *is* natural prose. Symbol-label ICL is the exception: 45 to 54 for every variant against 60.4 for the base, knowledge-only damage that no text choice changes (the mixture's job, section 8). The recipe reading is that the knowledge stream should be few templates in natural prose plus the specific texts each skill needs (comparisons for manipulation, backward statements for backward recall), not many templates; and that a generated-prose stream is the cheapest way to add volume if volume is wanted.

### 27.4 What DATA-5 now says

Answered. Recall needs three distinct renderings per fact and no more; the "ten paraphrases" figure is not a recall requirement here. Manipulation comes from negative and comparative texts, not from paraphrase count or kind; backward recall comes from backward sentences and from nothing else; sentence-order permutation buys nothing on this ladder and costs the most perplexity; LLM-written sentences are the cheapest volume in general-ability terms and teach nothing the templates did not. Single seeds, arm A only, at one budget; the trade with repetition (94 passes at one template) is folded into the sweep.

## 28. Hyperparameters: rank, rate and steps are one axis, 1,600 steps is the recipe's missing parameter, MLP-only LoRA matches all-linear, and full fine-tuning found no working rate (TRAIN-4)

Date: 2026-09-16. Code: `LORA_R=<r> | LORA_TARGETS=mlp | FULL_FT=1` and the step and learning-rate arguments of `exp_curriculum.py` (new knobs; alpha stays 2r), `RUN_TAG=<variant> PERIODIC=200 uv run python scripts/exp_curriculum.py C ...`; data `results/curriculum_Qwen2.5-3B_C_<variant>_p200.json` with per-item files and curves; the baseline is the section 15 rerun of arm C (seed 0) with section 20's three-seed band beside it.

TRAIN-4 noted that rank 64, alpha 128, no dropout, no weight decay, betas (0.9, 0.95) and 800 steps were fixed across every arm, that the only sweeps had been learning rates, and that the one full fine-tuning attempt on 3B had run out of memory with fp32 master weights; and it asked whether LoRA is the right vehicle for injection at all. The survey's reading: Biderman et al. recommend rank 256 on all modules and the highest stable learning rate, and find the LoRA-to-full-FT gap does not close at any rank for continued-pretraining-style text; against that, three papers find rank 16 equal to much higher ranks or to full FT for extraction and prompt-format training, and Allen-Zhu and Li find target type matters more than rank. This section moves one thing at a time off arm C's recipe: rank 16 and 256; learning rate 5e-5 and 2e-4; 400 and 1,600 steps (the schedule rescaled to the run); LoRA on the MLP projections only; and full fine-tuning of all 3.09B weights in bf16 with bitsandbytes 8-bit AdamW on the same 800 x 16 sequences, at the adapter's rate (1e-4) and at 1e-5. Four of the runs died of an out-of-memory error in the first backward after a periodic evaluation (rank 256 and MLP-only at step 600, 1,600 steps at step 1,000, full fine-tuning at step 0) and were rerun with 1,024-token packed rows, the 1,600-step one with a periodic point every 400 steps; the allocator now garbage-collects around the periodic evaluation. Everything else is section 8's arm C on the fast path, seed 0.

**Table 28.1: arm C under the hyperparameter variants (seed 0; the baseline column carries section 20's three-seed mean +- sd in brackets)**

| measure | C (r64, lr 1e-4, 800, all) | r16 | r256 | lr 5e-5 | lr 2e-4 | 400 steps | 1,600 steps | MLP-only r64 | full FT, lr 1e-4 | full FT, lr 1e-5 |
|---|---|---|---|---|---|---|---|---|---|---|
| recall, trained fmt | 100 [100.0 +- 0.0] | 100 | 100 | 100 | 100 | 95.6 | 100 | 100 | 100 | 19.4 |
| recall, bare | 19.4 [18.4 +- 1.3] | 20.6 | 21.2 | 15.6 | 20.6 | 16.9 | 20 | 18.8 | 21.9 | 20 |
| yes/no | 77.5 [80.0 +- 12.7] | 55 | 78.8 | 73.8 | 96.2 | 46.2 | 97.5 | 81.2 | 50 | 45 |
| pair | 81.2 [78.7 +- 12.7] | 50 | 85 | 56.2 | 85 | 50 | 91.2 | 72.5 | 50 | 50 |
| Timmy k=3 | 61.9 [63.8 +- 9.2] | 43.1 | 67.5 | 44.4 | 63.8 | 39.4 | 81.2 | 61.9 | 40.6 | 32.5 |
| k=4 | 51.9 [57.9 +- 11.0] | 42.5 | 63.8 | 38.8 | 63.1 | 25.6 | 71.9 | 59.4 | 22.5 | 29.4 |
| weakness | 60 [59.6 +- 11.9] | 41.2 | 61.9 | 38.8 | 65 | 38.1 | 76.9 | 58.1 | 38.8 | 33.8 |
| habitat | 29.4 [33.1 +- 3.3] | 34.4 | 44.4 | 32.5 | 34.4 | 29.4 | 33.8 | 35 | 28.1 | 34.4 |
| reverse easy | 30.6 | 27.5 | 66.9 | 31.2 | 31.2 | 29.4 | 90.6 | 33.1 | 73.1 | 30.6 |
| ICL symbol | 78.6 [79.0 +- 1.1] | 75.5 | 76.5 | 79.2 | 79.2 | 76.6 | 78.1 | 77.1 | 59.9 | 58.9 |
| ICL natural | 87 [87.5 +- 0.9] | 89.1 | 84.4 | 90.6 | 84.3 | 87.5 | 85.4 | 86 | 69.8 | 84.9 |
| ARC-Easy | 58.5 [64.0 +- 4.8] | 63.5 | 61.5 | 70.5 | 54.5 | 59 | 65.5 | 67 | 45 | 72.5 |
| WikiText ppl | 22.787 [22.6 +- 0.2] | 19.033 | 31.107 | 17.919 | 28.648 | 20.612 | 20.949 | 20.933 | 94.414 | 11.67 |
| training minutes | 50.3 | 10.9 | 16 | 11.2 | 11.2 | 5.6 | 25.8 | 11.1 | 14.5 | 14.5 |
| tokens/s | 415 | 1919 | 1307 | 1868 | 1865 | 1876 | 1615 | 1881 | 1441 | 1442 |
| peak GiB | 8.71 | 10.59 | 15.24 | 11.6 | 11.6 | 11.6 | 9.84 | 9.41 | 19.66 | 19.66 |

**Table 28.2: with the field guide in context**

| measure | C (r64, lr 1e-4, 800, all) | r16 | r256 | lr 5e-5 | lr 2e-4 | 400 steps | 1,600 steps | MLP-only r64 | full FT, lr 1e-4 | full FT, lr 1e-5 |
|---|---|---|---|---|---|---|---|---|---|---|
| Timmy k=3 | 70.6 | 73.8 | 84.4 | 72.5 | 77.5 | 73.8 | 74.4 | 77.5 | 41.9 | 38.8 |
| held-out species | 72.9 | 68.8 | 82.3 | 65.6 | 63.5 | 71.9 | 72.9 | 72.9 | 33.3 | 37.5 |
| pair | 76.2 | 71.2 | 76.2 | 73.8 | 88.8 | 70 | 73.8 | 77.5 | 51.2 | 73.8 |

base recall_fmt / Timmy / ICL sym by step: 0: 12.5 / 42.5 / 63.5, 200: 10.0 / 47.5 / 66.7, 400: 92.5 / 35.0 / 65.6, 600: 100.0 / 45.0 / 72.9

r16 recall_fmt / Timmy / ICL sym by step: 0: 12.5 / 40.0 / 63.5, 200: 10.0 / 37.5 / 72.9, 400: 75.0 / 37.5 / 74.0, 600: 100.0 / 37.5 / 74.0

r256 recall_fmt / Timmy / ICL sym by step: 0: 12.5 / 40.0 / 63.5, 200: 77.5 / 47.5 / 67.7, 400: 97.5 / 37.5 / 72.9, 600: 100.0 / 55.0 / 74.0

lr5e-5 recall_fmt / Timmy / ICL sym by step: 0: 12.5 / 40.0 / 63.5, 200: 15.0 / 42.5 / 70.8, 400: 85.0 / 45.0 / 78.1, 600: 100.0 / 45.0 / 71.9

lr2e-4 recall_fmt / Timmy / ICL sym by step: 0: 12.5 / 40.0 / 63.5, 200: 65.0 / 47.5 / 65.6, 400: 100.0 / 45.0 / 67.7, 600: 100.0 / 45.0 / 77.1

s400 recall_fmt / Timmy / ICL sym by step: 0: 12.5 / 40.0 / 63.5, 200: 17.5 / 32.5 / 81.2

s1600 recall_fmt / Timmy / ICL sym by step: 0: 12.5 / 40.0 / 63.5, 400: 100.0 / 50.0 / 75.0, 800: 100.0 / 65.0 / 80.2, 1200: 100.0 / 82.5 / 78.1

mlp recall_fmt / Timmy / ICL sym by step: 0: 12.5 / 40.0 / 63.5, 200: 17.5 / 37.5 / 78.1, 400: 100.0 / 40.0 / 74.0, 600: 100.0 / 52.5 / 73.9

fullft recall_fmt / Timmy / ICL sym by step: 0: 12.5 / 40.0 / 63.5, 200: 100.0 / 37.5 / 43.8, 400: 100.0 / 30.0 / 47.9, 600: 100.0 / 45.0 / 55.2

fullft1e-5 recall_fmt / Timmy / ICL sym by step: 0: 12.5 / 40.0 / 63.5, 200: 10.0 / 27.5 / 61.5, 400: 7.5 / 22.5 / 60.4, 600: 20.0 / 22.5 / 62.5


### 28.1 One axis: how far the adapter moves buys use of the facts and pays in forgetting

Every variant stores the facts: trained-format recall is 100 for all of them but the 400-step run (95.6). What the variants differ in is whether the facts can be used, and the pattern is a single axis (Table 28.1). The two changes that move the model less, rank 16 and learning rate 5e-5, keep recall at 100 and drop manipulation to chance or near it (55.0 / 50.0 and 73.8 / 56.2 against the baseline's 77.5 / 81.2) and induction from the weights to the base model's level (43.1 and 44.4 on Timmy against 61.9; k=4 42.5 and 38.8 against 51.9). They also cost the least: WikiText 19.0 and 17.9 against 22.8, ARC-Easy 63.5 and 70.5 against 58.5. The change that moves the model more, learning rate 2e-4, is the mirror image: the best manipulation and induction of any arm C run in the project (96.2 / 85.0; Timmy 63.8, k=4 63.1, weakness 65.0) at the largest cost (WikiText 28.6, ARC-Easy 54.5, natural-label ICL 84.3 against 87.0). Rank 256 sits with the baseline on the ladder (78.8 / 85.0; 67.5 / 63.8 / 61.9) at a higher perplexity (31.1) and twice the memory (15.2 GiB; its first run at 2,048-token rows died at step 600, 28.2).

The step count is the same axis in time (Table 28.1 and the curves): at 400 steps the facts are in (95.6) and nothing else is (manipulation 46.2 / 50.0, induction 39.4 / 25.6 / 38.1, the base's numbers), which is what section 15's curves showed at the 400-step point of the 800-step run; at 1,600 steps (the same warmup and decay stretched over the run) every from-the-weights number is the best any arm C run has produced and the general-ability costs are *lower* than the baseline's: manipulation 97.5 / 91.2, Timmy 81.2, k=4 71.9, weakness 76.9 (baseline 61.9 / 51.9 / 60.0), ICL suite 78.1 / 85.4, ARC-Easy 65.5 (baseline 58.5), WikiText 20.9 (baseline 22.8). The curve says why: recall is at 100 by step 400 and stays there, while induction from the weights keeps climbing, 50.0 / 65.0 / 82.5 / 81.2 at steps 400 / 800 / 1,200 / 1,600, and the symbol-label ICL suite holds at 75 to 80 throughout. Section 15 read the 800-step curves as facts by step 200 and induction still rising at the end; it was. The recipe's 800 steps under-trained the mixture, and the second 800 steps, taken at a decaying rate, teach the use of the facts without moving the general-text distribution further, where the same movement compressed into 800 steps at 2e-4 costs 6 perplexity points and 4 ARC-Easy points. Longer at the same peak rate is the better direction than hotter.

Read against section 20's seed band (manipulation sd 12.7, induction 9 to 12 for arm C), the rank 16, learning-rate 5e-5 and 400-step drops of 20 to 30 points on manipulation and 18 to 26 on induction are outside it, and the rank-256 and learning-rate 2e-4 differences from the baseline are not, except 2e-4's yes/no (+18.7) and its general-ability costs (perplexity and ARC-Easy have sd 0.2 and 4.8). The survey's two camps both find their evidence here: rank 16 equals higher ranks for *recall* (Bornschein, prompt distillation) and does not for *manipulation and induction* (Biderman's "learns less" at low rank). What the sweep adds is that the learning rate and the step count sit on the same axis as the rank: any of the three, turned down, gives a model that recalls and cannot use, and turned up gives one that uses and forgets.

### 28.2 Where the adapter goes: MLP-only, and full fine-tuning

LoRA on the three MLP projections only (rank 64, 15% fewer trainable parameters than all seven, `mlp` in Table 28.1) is the baseline: recall 100, manipulation 81.2 / 72.5, induction 61.9 / 59.4 / 58.1, ICL 77.1 / 86.0, and a smaller cost (WikiText 20.9 against 22.8, ARC-Easy 67.0 against 58.5). The attention projections add nothing this ladder can see, which is Allen-Zhu and Li's finding that target type matters and their observation that the knowledge sits in the MLP blocks; it also says the section 8 recipe could drop the attention adapters and lose nothing but 3 GiB of optimiser state.

Full fine-tuning at the adapter's learning rate is not a comparison, it is a demonstration: all 3.09B weights in bf16 with 8-bit AdamW at 1e-4 (`full FT, lr 1e-4`) reach 100 recall by step 200 and nothing else, manipulation and induction at chance, the symbol-label ICL suite at 59.9 and the natural-label one at 69.8 (from 60.4 and 84.9), ARC-Easy 45.0, WikiText perplexity 94.4. At 1e-4 the whole model moves, and 800 steps of 12,800 synthetic sequences are enough to overwrite it. At 1e-5, the rate full fine-tuning is usually run at, the same 800 steps learn nothing: trained-format recall 19.4 (the curve reads 10.0 / 7.5 / 20.0 at steps 200 / 400 / 600), manipulation and induction at the base, and the model otherwise untouched (WikiText 11.7 against the base's 10.6, ARC-Easy 72.5, natural-label ICL 84.9). Between an order of magnitude that erases the model and one that does not reach it, the full fine-tune's working rate for 136 species in 800 steps was not found in two runs; 3e-5 is the next point if full fine-tuning is ever needed. Either way the cost of full fine-tuning is real: 19.7 GiB, a 5.8 GB checkpoint per run, two attempts to get one through the periodic evaluations, and no result on this ladder that the rank-64 adapter does not match.

### 28.3 What TRAIN-4 now says

*[Later: read with section 20, the 1,600-step gains on manipulation, induction and ARC-Easy are one seed each and inside the seed spread; only WikiText clears. The categoriser line (sections 38 to 47) trains all-linear rank 64 for 200 steps; PLAN row 47 tests longer training there.]*

Answered, as far as one seed per point allows. Rank, learning rate and step count are one axis on this ladder: turned down (rank 16, 5e-5, 400 steps) the adapter stores the facts and cannot use them, turned up (2e-4) it uses them and forgets more; 1,600 steps at the recipe's peak rate is the exception that improves both sides (manipulation 97.5 / 91.2, induction 81 / 72 / 77, perplexity and ARC-Easy better than at 800), so the recipe's step count was the parameter that was wrong. Rank 256 buys nothing over 64 and costs perplexity and memory; MLP-only LoRA matches all-linear at lower cost; full fine-tuning at the adapter's rate destroys the model and at 1e-5 learns nothing in 800 steps (recall 19.4), so its working rate lies between and was not found. LoRA is the right vehicle here, at rank 64 on the MLP projections, for twice the steps. Not run: WiSE-FT at other alphas and FineWeb replay in the merchant runs (the merchant pipeline is section 4's; general-text replay was done for the universe in section 17).

## 29. Encoders and entity count: a rank-64 adapter holds 1,000 species given the exposure, the masked-LM letter protocol learns nothing, and span prediction learns the facts at T5's rate and only the sentence at the decoders' (MODEL-2, REAL-3)

Date: 2026-09-16. Code: `UNIVERSE_N=125|625 ... uv run python scripts/exp_curriculum.py C` (the 1,000 and 5,000-species universes: `universe.build` draws three-part names when the two-part space of 864 cannot hold the universe, and `items` freezes their own ladder, held-out and probe sets as `*_n1000` / `*_n5000`); `scripts/exp_encoder.py` (new: Flan-T5-large trained by span prediction on the knowledge texts and prompt-to-answer on the episodes and replay, scored by decoder option log-probability and by generation); `scripts/exp_mlm.py` (new: ModernBERT-large trained with the masked-LM head, scored with the letter-at-mask protocol); `scripts/encoder_tables.py` prints Tables 29.2b and 29.3 from the result files. Data `results/curriculum_Qwen2.5-3B_C_n{1000,5000}{,x5}_p*.json`, `results/encoder_flan-t5-large_*.json` (the `_lr*` and `_lora*` files are the learning-rate and adapter follow-ups), `results/encoder_t5gemma-l-l-ul2_*.json`, `results/mlm_ModernBERT-large_*.json`, per-item files for each.

Two questions share this section because the literature ties them together. MODEL-2 asked whether an encoder trained with fill-in-the-blank should be an injection and extraction method: the same objective for both, native multiple choice at the mask, no format gap. The survey's cautions were that ModernBERT-Instruct answers through a single mask on single-token verbalisers and never tests injecting new facts, that WikiDYK's span-masked Flan-T5 beats causal models on memorisation only above about a thousand facts, and that encoder-only multi-mask generation fails. REAL-3 asked whether 100% recall survives 1,000 to 10,000 entities in one adapter; section 7 had extrapolated linearly from 136. The two meet at the entity count: the bidirectional advantage, if it exists, should appear where the decoder's capacity runs out.

Every answer on the ladder is multi-token under both encoders' tokenizers (type names, nonsense labels and species names are two to five pieces in ModernBERT's BPE and in T5's SentencePiece), which fixes the protocols. The encoder-decoder scores an option as a decoder target, exactly as the decoder arms score a continuation, and can also generate it. The encoder-only model cannot score a multi-token option at one mask; it gets the instruct recipe's protocol, the options listed with letters and the letter predicted at the mask, learned during training from the episodes, the replay and letter questions on the training facts (with 20% plain masked-LM examples as the regulariser). That is symbol scoring with listed options (section 10), not cloze, and the comparison with the decoder arms is between protocols as well as between models. T5Gemma, the modern encoder-decoder the owner asked to see beside Flan-T5, was fetched after the owner's Hub login; its tokenizer has no T5 sentinels, so its span examples use a placeholder word.

**Table 29.1: REAL-3, arm C (section 8 recipe, Qwen2.5-3B, seed 0) at 160, 1,000 and 5,000 species, 800 and 4,000 steps (accuracy %)**

| measure | 160 / 800 | 1,000 / 800 | 1,000 / 4,000 | 5,000 / 800 | 5,000 / 4,000 |
|---|---|---|---|---|---|
| recall, trained fmt | 100 | 51.2 | 100 | 14.4 | 21.9 |
| recall, bare | 19.4 | 17.5 | 16.9 | 13.8 | 13.1 |
| yes/no | 77.5 | 48.8 | 97.5 | 45 | 45 |
| pair | 81.2 | 45 | 86.2 | 48.8 | 48.8 |
| Timmy k=3 | 61.9 | 30.6 | 54.4 | 34.4 | 38.1 |
| k=4 | 51.9 | 30 | 41.9 | 22.5 | 22.5 |
| weakness | 60 | 36.9 | 52.5 | 30.6 | 38.1 |
| habitat | 29.4 | 36.2 | 34.4 | 39.4 | 31.2 |
| held-out species | 29.2 | 33.3 | 35.4 | 30.2 | 39.6 |
| ICL symbol | 78.6 | 77.6 | 74.5 | 82.8 | 74.5 |
| ICL natural | 87 | 84.9 | 80.2 | 84.4 | 80.7 |
| ARC-Easy | 58.5 | 58 | 63 | 57.5 | 60 |
| WikiText ppl | 22.787 | 20.664 | 41.287 | 26.054 | 30.581 |
| training minutes | 50.3 | 27 | 66.1 | 14.6 | 64.5 |
| peak GiB | 8.71 | 11.6 | 9.85 | 9.84 | 9.85 |

160 / 800 recall_fmt / Timmy by step: 0: 12.5 / 42.5, 200: 10.0 / 47.5, 400: 92.5 / 35.0, 600: 100.0 / 45.0

1,000 / 800 recall_fmt / Timmy by step: 0: 12.5 / 27.5, 200: 10.0 / 22.5, 400: 10.0 / 27.5, 600: 27.5 / 32.5

1,000 / 4,000 recall_fmt / Timmy by step: 0: 12.5 / 27.5, 800: 62.5 / 20.0, 1600: 100.0 / 45.0, 2400: 100.0 / 32.5, 3200: 100.0 / 50.0

5,000 / 800 recall_fmt / Timmy by step: 0: 7.5 / 32.5, 200: 10.0 / 35.0, 400: 12.5 / 37.5, 600: 15.0 / 35.0

5,000 / 4,000 recall_fmt / Timmy by step: 0: 7.5 / 32.5, 800: 15.0 / 25.0, 1600: 15.0 / 25.0, 2400: 10.0 / 25.0, 3200: 15.0 / 30.0

**Table 29.2: the encoder arms on the 160-species items (option log-probability; generation exact match in the gen rows), against the decoder arm C**

| measure | Flan-T5 base | Flan-T5 F2A (knowledge) | Flan-T5 F2 (mixture) | ModernBERT base (letters) | ModernBERT F (letters) | Qwen2.5-3B C |
|---|---|---|---|---|---|---|
| recall, trained fmt | 8.1 | 14.4 | 16.9 | 13.8 | 9.4 | 100 |
| recall, bare | 18.8 | 13.1 | 15.6 | 14.4 | 14.4 | 19.4 |
| yes/no | 45 | 46.2 | 45 | 55 | 55 | 77.5 |
| pair | 50 | 50 | 50 | 50 | 50 | 81.2 |
| Timmy k=3 | 35.6 | 29.4 | 29.4 | 32.5 | 36.9 | 61.9 |
| k=4 | 20 | 23.8 | 24.4 | 21.9 | 25 | 51.9 |
| weakness | 30.6 | 33.8 | 34.4 | 31.9 | 30 | 60 |
| habitat | 35 | 38.1 | 36.2 | 35 | 30 | 29.4 |
| held-out species | 36.5 | 27.1 | 36.5 | 25 | 42.7 | 29.2 |
| ICL symbol | 53.6 | 52.6 | 72.9 | 46.4 | 45.9 | 78.6 |
| ICL natural | 87 | 85.4 | 88 | 55.7 | 39.1 | 87 |
| ARC-Easy | 56 | 34.5 | 36.5 | 42.5 | 29 | 58.5 |
| training minutes | - | 9.8 | 16.8 | - | 7.2 | 50.3 |
| peak GiB | - | 7.46 | 9.25 | - | 7.97 | 8.71 |
| generation: recall fmt | 0 | 46.9 | 20 | - | - | - |
| generation: recall bare | 0 | 0 | 0 | - | - | - |
| generation: reverse hard | 0 | 0 | 0 | - | - | - |
| reverse easy | 21.9 | 31.2 | 26.2 | 28.8 | 25 | 30.6 |
| reverse hard | 28.1 | 23.8 | 25 | 21.9 | 26.2 | 19.4 |

**Table 29.2b: the encoder-decoders' fact levels in the span-prediction format they were trained in (option log-probability; generation in the gen rows)**

| measure | Flan-T5 base | Flan-T5 F2A (knowledge) | Flan-T5 F2 (mixture) | T5Gemma base | T5Gemma F2 full FT 1e-4 | T5Gemma F2 LoRA 1e-4 | T5Gemma F2 LoRA 3e-4 | Flan-T5 F2A LoRA 3e-4 |
|---|---|---|---|---|---|---|---|---|
| recall, trained fmt | 10 | 14.4 | 13.8 | 8.8 | 10 | 97.5 | 49.4 | 16.2 |
| recall, bare | 18.1 | 16.2 | 16.9 | 14.4 | 19.4 | 38.8 | 28.1 | 16.2 |
| yes/no | 43.8 | 45 | 45 | 55 | 55 | 45 | 48.8 | 45 |
| pair | 47.5 | 51.2 | 50 | 50 | 50 | 50 | 53.8 | 50 |
| reverse easy | 23.8 | 26.2 | 21.9 | 18.8 | 19.4 | 17.5 | 22.5 | 23.8 |
| reverse hard | 26.9 | 22.5 | 30.6 | 31.2 | 34.4 | 33.8 | 34.4 | 25.6 |
| generation: recall fmt | 0 | 43.8 | 21.2 | 0 | 0 | 95 | 43.8 | 18.8 |
| generation: recall bare | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| generation: reverse hard | 0 | 0 | 0 | 0 | 0 | 0.6 | 3.1 | 0.6 |

**Table 29.3: Flan-T5 at 1,000 and 5,000 species, the knowledge-only arm at T5's usual learning rates, and the rank-64 adapter runs on both encoder-decoders (span-format value in brackets)**

| measure | 160 / 800 | 1,000 / 800 | 1,000 / 4,000 | 5,000 / 4,000 | F2A 160 lr 3e-4 | F2A 160 lr 1e-3 | F2A 160 lr 1e-3 / 4,000 | F2A 160 LoRA 3e-4 | T5Gemma F2 LoRA 1e-4 | T5Gemma F2 LoRA 3e-4 |
|---|---|---|---|---|---|---|---|---|---|---|
| recall, trained fmt | 16.9 [13.8] | 12.5 [13.1] | 13.8 [13.8] | 11.2 [11.2] | 98.1 [100] | 100 [100] | 100 [100] | 11.9 [16.2] | 97.5 [97.5] | 48.8 [49.4] |
| recall, bare | 15.6 [16.9] | 12.5 [15] | 14.4 [15] | 15.6 [18.1] | 40 [33.1] | 29.4 [36.2] | 28.1 [26.2] | 9.4 [16.2] | 42.5 [38.8] | 28.1 [28.1] |
| generation: recall fmt | 20 [21.2] | 15 [13.1] | 24.4 [27.5] | 16.2 [17.5] | 100 [100] | 66.9 [100] | 92.5 [100] | 16.2 [18.8] | 93.1 [95] | 44.4 [43.8] |
| yes/no | 45 [45] | 48.8 [48.8] | 48.8 [52.5] | 43.8 [43.8] | 43.8 [55] | 53.8 [45] | 45 [45] | 45 [45] | 45 [45] | 55 [48.8] |
| Timmy k=3 | 29.4 | 30 | 29.4 | 29.4 | 30.6 | 33.1 | 36.2 | 29.4 | 35 | 36.2 |
| reverse hard | 25 [30.6] | - | - | - | 23.8 [20] | 20.6 [28.1] | 22.5 [28.8] | 24.4 [25.6] | 31.2 [33.8] | 28.8 [34.4] |
| ICL symbol | 72.9 | 71.3 | 80.8 | 83.4 | 51.6 | 40.6 | 44.2 | 46.4 | 43.2 | 43.8 |
| ICL natural | 88 | 88.5 | 89.6 | 87.5 | 75 | 52.6 | 45.8 | 78.1 | 43.8 | 39.6 |
| ARC-Easy | 36.5 | 35.5 | 33.5 | 32 | 30.5 | 29 | 22 | 31.5 | 39.5 | 30.5 |
| training minutes | - | 16.8 | 84.3 | 84.7 | 8.9 | 9.8 | 48 | 14.3 | 16.9 | 17.1 |


### 29.1 Entity count: a rank-64 adapter holds 1,000 species when it is shown them (REAL-3)

The recipe's budget is 12,800 sequences, 45% of them knowledge texts. At 160 species (2,752 texts) that is seven passes over each text; at 1,000 species (19,552 texts) it is a third of a pass, and at 5,000 (99,552 texts) a fifteenth. Table 29.1 separates capacity from exposure by running the same recipe for 800 and for 4,000 steps.

At 1,000 species and 800 steps the adapter has seen a third of the texts once, and trained-format recall is 51.2, manipulation at chance, induction at the base; the general-ability measures are where the 160-species run left them (ICL 77.6, ARC-Easy 58.0, WikiText 20.7). Nothing has gone wrong; the facts were not shown. At 4,000 steps (1.5 passes) recall is 100, reached by step 1,600 and held, manipulation 97.5 / 86.2 (the 160-species run's 77.5 / 81.2, with the section 28 effect of the longer run on top), induction 54.4 / 41.9 / 52.5 against 61.9 / 51.9 / 60.0, the ICL suite 74.5. So 1,000 species times six attributes fit in a rank-64 adapter on all seven projections with no sign of interference among three-part names that share syllables. What the larger universe costs is exposure, which is training time (66 minutes against 11 on the fast path), and general text: WikiText perplexity 41.3 against 22.8, because 4,000 steps over six times as much distinct synthetic text move the model further from its distribution than 1,600 steps over the small universe did (20.9 in section 28). The section 7 extrapolation, linear time in the entity count, is the right shape for time; it did not anticipate that perplexity scales with the distinct text too.

At 5,000 species the budget runs out before the capacity question can be asked. 4,000 steps are 0.3 passes over 99,552 texts, and trained-format recall is 21.9 (the curve reads 15 / 15 / 10 / 15 at steps 800 to 3,200: no take-off), manipulation and induction at chance, the general-ability numbers those of a lightly trained model (WikiText 30.6, ARC-Easy 60.0, ICL 74.5 / 80.7). The comparison that isolates interference is the 1,000-species run at the same 0.3 passes (800 steps: recall 51.2) against this one (21.9): at equal exposure per text the larger universe recalls less, which is the first sign of the capacity or name-collision effect REAL-3 asked about, but the design cannot separate it from the ladder's sampling (a third of the species have seen one rendering each, and which third differs). The run that would answer it, 5,000 species at 1.5 passes, is 20,000 steps, about five and a half hours on this card; it is the next REAL-3 experiment and was not run in this step. The 800-step run at 5,000 species (a fifteenth of a pass) recalls 14.4 in the trained format, the base model's number, with ICL 82.8 and WikiText 26.1: at that exposure the run is an expensive no-op, which is what the linear extrapolation predicts and the reason a 10k-record deployment needs either the budget scaled with the entities or the section 25 retriever in front of the model.

### 29.2 An encoder-only masked LM with the letter protocol learns nothing usable (MODEL-2, arm F)

ModernBERT-large (395M) trained for 800 steps on the mixture, its knowledge texts as masked LM with the attribute spans masked whole and 15% random tokens, its episodes and replay as letter questions, 20% of every batch a plain masked-LM example, scores at chance on every level of the ladder under the same letter protocol (Table 29.2): trained-format recall 9.4 (8-way chance 12.5), yes/no and pair 55.0 / 50.0, Timmy 36.9, k=4 25.0, held-out species 42.7. The untrained model under the same protocol is at chance too (13.8 / 55.0 / 50.0 / 32.5), so 800 steps of letter questions did not teach the protocol, and the ICL suite says what the training did instead: natural-label accuracy fell from 55.7 to 39.1. ModernBERT-Instruct learned to answer at a mask from 20M instruction examples; 5,000 letter questions on top of masked-LM injection are not that, and a single mask cannot score a multi-token answer, which every answer on this ladder is. The encoder-only route is closed at this size and budget; the reason is the protocol, not the injection, and the comparison stays a comparison of protocols.

### 29.3 Span prediction learns the facts at T5's learning rate and only the sentence at the decoders' (MODEL-2, arm F2)

Flan-T5-large (783M), trained by span prediction on the same knowledge texts (every attribute value masked with probability 0.5, 15% random spans, T5 sentinels) with the episodes and replay as prompt-to-answer, at the decoders' rate of 1e-4, recovers the symbol-label ICL suite the way the decoder mixture does (72.9 from a base of 53.6; the knowledge-only arm 52.6) and learns nothing the fact levels can see: trained-format recall 16.9 by option scoring and 20.0 by generation, bare recall 15.6, manipulation at chance, the backward levels at chance, and ARC-Easy down from 56.0 to 36.5 (Table 29.2). Scoring the fact levels in the span-prediction format the knowledge stream was trained in (the question with the sentinel where the answer goes, Table 29.2b) does not change it, and a direct look at the generations says what happened: asked for a species' type the model produces the trained sentence shape with the wrong type ("Syaott is a Voltrix-type." for a Verdane species), and its option scores are a fixed preference for two type names within two nats of each other across all eight options, the same ranking for every species. At 1,000 species (Table 29.3) the 1e-4 runs read the same at every budget (12.5 / 13.8 recall at 800 and 4,000 steps, generation 15 and 24 to 27) and at 5,000 species and 4,000 steps recall is 11.2.

The learning rate is the whole difference. The knowledge-only arm at T5's own fine-tuning rates (Table 29.3; full fine-tuning with 8-bit AdamW, 800 steps, 4.6 passes over the 2,752 texts) recalls the facts: at 3e-4 trained-format recall is 98.1 by option scoring and 100 by generation, against 14.4 / 46.9 at 1e-4, and the trained span format agrees (100). The same run reads 40.0 on bare recall, twice the decoder arm C's 19.4 on the question the training never showed, and nothing on label induction: Timmy k=3 30.6 against the base's 35.6 and the decoder mixture's 61.9 (with real type names as labels it reads 61.2 against the decoder's 77.5, the type vocabulary it learned rather than the induction skill). What it costs is the general-ability measures the knowledge-only decoder arm also loses: natural-label ICL 85.4 to 75.0, symbol ICL 51.6, ARC-Easy 30.5 (the 1e-4 run 34.5, the base 56.0). At 1e-3 recall is 100 as well, with generation in the trained format down to 66.9 (the model over-produces the sentence), natural ICL 52.6 and Timmy 33.1; 4,000 steps at 1e-3 keep recall at 100 and take natural ICL to 45.8 and ARC-Easy to 22.0. So 3e-4 for 800 steps is the Flan-T5 setting; the decoders' 1e-4 is an order of magnitude below the rate the T5 recipe fine-tunes at, and every 1e-4 Flan-T5 run in Tables 29.2 and 29.3 is an under-trained model, not evidence about the encoder-decoder's capacity. The 1,000 and 5,000-species Flan-T5 runs at 3e-4 (17, 84 and 85 minutes at 1e-4) are the runs that would put WikiDYK's claim, span-masked Flan-T5 memorising real facts better than 1B to 8B causal models above about a thousand facts, against the decoder's 100 at 1,000 species; they were not run in this step.

**T5Gemma** (large-large, UL2; 1.2B parameters, no T5 sentinels, so masked spans are a placeholder word and the target the spans in order) needs the opposite correction. Full fine-tuning of all its weights at 1e-4 with 8-bit AdamW damaged it (natural-label ICL 74.5 to 47.4, ARC-Easy 54.0 to 20.5) and taught it no fact (recall 12.5, generation 0), the failure the 3B decoder showed under full fine-tuning at that rate in section 28. The decoders' own recipe, a rank-64 adapter on every projection over a bf16 base (`LORA=64`), at 1e-4 on the mixture recalls 97.5 in the trained format by option scoring and 93.1 by generation (97.5 / 95.0 in the span format), bare recall 42.5, trained in 17 minutes at 4.8 GiB against the full fine-tune's 14 minutes at 11.7 GiB. It does not protect the rest of the model: natural-label ICL is 43.8 and symbol ICL 43.2 (base 74.5 / 50.0), ARC-Easy 39.5, Timmy 35.0 against the base's 33.8; the mixture's episodes, which hold the decoder's ICL suite at 78.6 and lift its induction to 61.9, do neither for T5Gemma under an adapter. At 3e-4 the adapter learns half the facts (recall 48.8, generation 44.4) and loses more (natural ICL 39.6). The adapter on Flan-T5 at 3e-4 learns nothing (recall 11.9, generation 16.2; natural ICL kept at 78.1): a rank-64 adapter at the rate that works for full fine-tuning of the same model is too small a step, and the adapter run at 1e-3 or above is the unrun follow-up. The two encoder-decoders therefore need different recipes (Flan-T5: full fine-tuning at 3e-4; T5Gemma: the adapter at 1e-4), and neither recipe keeps the general-ability measures where the decoder mixture keeps them.

### 29.4 What the step says

- REAL-3: capacity is not the limit at 1,000 species; exposure is. The section 8 recipe at 4,000 steps (1.5 passes over 19,552 texts) recalls 1,000 species at 100 with manipulation at 97.5 / 86.2, at the cost of an hour and a 41 WikiText perplexity; at 800 steps (0.3 passes) it recalls 51. At 5,000 species the budget runs out (22 at 0.3 passes, 14 at 0.07), and the same-exposure comparison (51 at 1,000 against 22 at 5,000) is the first sign of interference; 5,000 species at 1.5 passes (20,000 steps) is the run that decides it.
- MODEL-2, encoder-only: no. ModernBERT-large under the letter-at-mask protocol is at chance on every level after training and loses natural-label ICL; every answer on the ladder is multi-token, and the protocol that would fix that needs an instruction-tuning stage this budget does not contain.
- MODEL-2, encoder-decoder: yes for injection at the right rate, and the rate is the model's, not the decoders'. Flan-T5-large under full fine-tuning at 3e-4 recalls the 160 species at 98.1 / 100 (option scoring / generation) where 1e-4 learned the template only, with no label induction (Timmy at the base) and bare recall twice the decoder's; T5Gemma under a rank-64 adapter at 1e-4 recalls 97.5 / 93.1 where full fine-tuning at that rate destroyed it. Both lose the natural-label ICL suite and ARC-Easy in a way the decoder mixture does not (Flan-T5 75.0 / 30.5, T5Gemma 43.8 / 39.5), and the 1,000 and 5,000-species Flan-T5 runs at the working rate, the ones that would test WikiDYK's capacity claim against the decoder's 100 at 1,000 species, are unrun. On this ladder the encoder-decoder is a working injection vehicle and a worse general model after it; it is not a substitute for the decoder until that cost is understood.
- Everything trained here is saved under `models/adapters/` (the Flan-T5 checkpoints at each rate, the T5Gemma checkpoints and adapters, the Flan-T5 adapter, the ModernBERT weights, the EmbeddingGemma, gte-modernbert, MiniLM, bge and Qwen3-Embedding encoders of sections 24 and 24.7, and the merchant retrievers of section 25) and pushed to the DVC remote.

## 30. Knowledge editing: an editor writes one association per species perfectly, at no general-ability cost and in any phrasing, MEMIT needs a ridge on Qwen2.5-3B, and five facts per species collide on the key (BASE-2)

Date: 2026-09-17. Code: EasyEdit (github.com/zjunlp/EasyEdit, checked out beside this repo with its own environment; `ai_experiments` installed into it without dependencies), driver `EasyEdit/ai_exp_edit.py ALG type|all` (EasyEdit's `batch_edit` in one chunk with `sequential_edit=True`, so the edits are one closed-form update and the edited weights are kept; the library's default `edit` applies and undoes them one at a time) with `hparams/{MEMIT,AlphaEdit}/qwen2.5-3b.yaml` (derived from the repository's Llama-class settings: MLP down-projections of layers 4 to 8, `v_loss_layer` 35, 25 target-vector steps at 0.5, covariance from 20,000 WikiText samples with `mom2_update_weight` 15,000; `lm_head_module` is the embedding matrix because Qwen2.5-3B ties its output head to it); the edited model is saved as a full bf16 checkpoint under `models/adapters/edit_<alg>_<facts>_qwen2.5-3b` and scored with `scripts/exp_curriculum.py base <that directory>`, the untrained-model path, so the ladder, the probes, the ICL suite, ARC-Easy and WikiText are exactly the section 8 measures. `scripts/edit_tables.py` prints the table. Data `results/edit_<alg>_<facts>.json` (EasyEdit's own post-edit metrics and timing), `results/curriculum_edit_<alg>_<facts>_qwen2.5-3b_base.json` and per-item files.

BASE-2 asked how the dedicated editors compare on this ladder: 136 trained species with five attributes each is inside the range MEMIT and AlphaEdit edit in one batch on models of this size, and locality, the property arm A lacks, is what they are built to keep. The survey's expectation was efficacy and paraphrase above 90 with the general measures near the base, and nothing on manipulation, induction or the reverse levels, which every editor's portability numbers put in the single digits or low tens (2401.01286 Table 4; MQuAKE in 2410.02355). Two cautions shaped the design. MEMIT's keys are computed from the subject alone (2210.07229 Eqn 19), so five edits on one species share one key and the closed-form update averages their targets; the `type` runs (one edit per species, the type) are the clean test and the `all` runs (five edits per species: type, weakness, habitat, diet, region) the shared-key regime the survey warned about. And the edited layers were not re-located by causal tracing for Qwen2.5-3B; layers 4 to 8 are the repository's choice for Llama-3-8B and GPT-J, carried over, which is a limitation of this step and not a finding about the model. Edit requests are the ladder's own trained-format sentences with the subject at the front: "{N} is a" with target "{T}-type creature", "{N} is weak to" with "{W}-type attacks", "{N} lives in" with "{H} habitats", "{N} eats as an" with "{D}", "{N} is found in" with "{R}". The metric mapping is the one section 8 uses for the arms: EasyEdit's rewrite accuracy is the edit's own success criterion on its own prompt; trained-format recall is efficacy on the ladder's eight-way item, bare recall is paraphrase, the manipulation, induction and reverse levels are portability, and the ICL suite, ARC-Easy and WikiText perplexity are out-of-distribution locality.

**Table 30.1: knowledge editing on Qwen2.5-3B (EasyEdit MEMIT, MEMIT with a ridge on the solve, and AlphaEdit; layers 4-8, one batch) against the untrained base and the fine-tuned arms A and C on the 160-species items (accuracy %)**

| measure | base | A (knowledge) | C (mixture) | MEMIT type | MEMIT type, ridge 150 | MEMIT type, ridge 15 | AlphaEdit type | MEMIT all relation-first, ridge 15 | AlphaEdit all | MEMIT all record relation-first, ridge 15 |
|---|---|---|---|---|---|---|---|---|---|---|
| recall, trained fmt | 13.1 | 100 | 100 | 8.1 | 100 | 100 | 100 | 53.1 | 48.8 | 23.8 |
| recall, bare | 18.1 | 23.1 | 19.4 | 11.9 | 41.2 | 41.2 | 41.2 | 49.4 | 20.6 | 38.1 |
| yes/no | 42.5 | 100 | 77.5 | 47.5 | 61.2 | 55 | 57.5 | 55 | 42.5 | 57.5 |
| pair | 51.2 | 88.8 | 81.2 | 55 | 53.8 | 51.2 | 48.8 | 56.2 | 53.8 | 56.2 |
| Timmy k=3 | 35.6 | 40 | 61.9 | 31.9 | 33.8 | 35.6 | 30.6 | 36.2 | 31.2 | 36.2 |
| k=4 | 27.5 | 41.9 | 51.9 | 25.6 | 28.1 | 23.8 | 25 | 31.9 | 28.1 | 31.2 |
| weakness | 35 | 46.2 | 60 | 35.6 | 35 | 33.8 | 38.1 | 39.4 | 41.2 | 35.6 |
| habitat | 31.2 | 30.6 | 29.4 | 25.6 | 32.5 | 33.1 | 36.2 | 33.8 | 33.8 | 33.1 |
| held-out species | 39.6 | 22.9 | 29.2 | 37.5 | 30.2 | 34.4 | 37.5 | 33.3 | 35.4 | 36.5 |
| unseen recall | 16.7 | 12.5 | 12.5 | 12.5 | 12.5 | 8.3 | 8.3 | 12.5 | 16.7 | 12.5 |
| seen recall control | 16.7 | 29.2 | 20.8 | 16.7 | 95.8 | 95.8 | 91.7 | 50 | 8.3 | 12.5 |
| reverse easy | 23.1 | 88.8 | 30.6 | 25.6 | 23.8 | 23.1 | 23.8 | 22.5 | 18.8 | 26.9 |
| reverse hard | 20 | 26.9 | 19.4 | 25 | 24.4 | 26.2 | 24.4 | 25.6 | 22.5 | 25.6 |
| ICL symbol | 60.4 | 47.4 | 78.6 | 42.7 | 57.8 | 57.3 | 56.2 | 58.3 | 58.3 | 59.9 |
| ICL natural | 84.9 | 80.8 | 87 | 40.6 | 87 | 86 | 86 | 85.4 | 87.5 | 85.4 |
| ARC-Easy | 73.5 | 66 | 58.5 | 23 | 72.5 | 72 | 71.5 | 78.5 | 74.5 | 72 |
| WikiText ppl | 10.614 | 34.055 | 22.787 | 5.66187e+07 | 10.712 | 10.705 | 10.945 | 10.883 | 10.709 | 10.794 |
| recall, bare: type questions | 17.9 | 25.0 | 19.6 | 12.5 | 100.0 | 100.0 | 100.0 | 46.4 | 16.1 | 25.0 |
| recall, bare: weakness questions | 17.9 | 19.6 | 17.9 | 5.4 | 0.0 | 0.0 | 0.0 | 35.7 | 21.4 | 14.3 |
| recall, bare: habitat questions | 18.8 | 25.0 | 20.8 | 18.8 | 20.8 | 20.8 | 20.8 | 68.8 | 25.0 | 81.2 |
| edits | - | - | - | 136 | 136 | 136 | 136 | 680 | 680 | 680 |
| edit minutes | - | - | - | 22.6 | 26 | 25.9 | 34 | 119.5 | 126.9 | 112.2 |
| EasyEdit rewrite acc | - | - | - | 61.3 | 88.6 | 93.1 | 86.2 | 41.9 | 20.8 | 21.7 |
| EasyEdit rephrase acc | - | - | - | - | - | - | - | - | - | - |

### 30.1 MEMIT's closed-form update is ill-conditioned on Qwen2.5-3B and destroys the model

The MEMIT batch of 136 type edits (23 minutes, EasyEdit rewrite accuracy 61.3 on its own prompts) leaves a model that scores below the base on every level and has lost language: trained-format recall 8.1 (base 13.1), natural-label ICL 40.6 (84.9), ARC-Easy 23.0 (73.5), WikiText perplexity 5.7 x 10^7 (10.6) (Table 30.1, MEMIT type). The weights say why. The edited down-projections changed by 250,000 times their own norm at layer 4 (14,600 to 60,800 times at layers 5 to 7, 270 at layer 8; the largest single entry went from 0.7 to 212,000), so the update is not an edit but a replacement. MEMIT's update is R K^T (lambda C0 + K K^T)^-1 with C0 the uncentred covariance of the layer's keys over general text; the solve is done in double precision, so the problem is C0 itself. Collected over 1.37M WikiText tokens, the layer-4 covariance has a top eigenvalue of 128 and a median of 2 x 10^-9, with 6,045 of its 11,008 eigenvalues below 10^-6: Qwen2.5's MLP intermediate activations concentrate in a few massive dimensions and are close to zero in more than half of the rest, so lambda C0 does not regularise those directions at any lambda (15,000 times 2 x 10^-9 is 3 x 10^-5) and the inverse of a rank-136 K K^T takes over there. The four-edit smoke reached rewrite accuracy 100 with the same settings and was never scored for fluency, which is how the problem reached the full run. EasyEdit ships these hyperparameters for Qwen2.5-7B with a Wikipedia covariance over 100,000 samples; whether that spectrum is any fuller is not known here. The fix that stays inside MEMIT's derivation is a ridge on the solve (lambda C0 + K K^T + rho I), which is a `cov_ridge` hyperparameter added to the library for this step (`scripts/easyedit/easyeditor.patch`); AlphaEdit's solve carries an identity term by construction and projects the update onto the null space of C0, which on this spectrum is almost the whole space.

### 30.2 AlphaEdit inserts the type of every species with the general measures untouched, and nothing beyond the edited association

The same 136 edits through AlphaEdit (34 minutes; the null-space projections of the five layers computed once and cached) give the locality baseline BASE-2 asked for. Trained-format recall is 100, the number arms A and C reach after 800 steps. The bare-recall level mixes three questions, and split by attribute (the last rows of Table 30.1) it says something the pooled 41.2 hides: on the bare type question ("What type is X?", type names as options) the edited model scores 100 where the fine-tuned arms score 20 to 25, so the format gap that has followed every fine-tune since section 8 (100 in the trained sentence, chance on the plain question) does not exist for the editor. The edit is written at the subject's last token and fires on any prompt that ends with the species name; the seen-recall control, the same bare question on another sample of species, reads 91.7 (base 16.7, C 20.8). The other side of that mechanism is the weakness question ("What type is X weak to?", the same type-name options): 0.0, below the 12.5 chance, because the edit answers with the species' type wherever the word type and the name appear together. Habitat questions, which the edit does not touch, stay at the base (20.8). Everything the edit did not target is where the base left it: WikiText perplexity 10.9 against the base's 10.6 (arm C 22.8, A 34.1), ARC-Easy 71.5 against 73.5 (C 58.5), natural-label ICL 86.0 against 84.9, symbol-label ICL 56.2 against 60.4 (C's mixture lifts that to 78.6; an editor has no reason to). The edited weights moved by 61% of their norm at layer 4 and 22 to 27% at layers 5 to 8, with no entry larger than 0.32, so this is a large but bounded update, where MEMIT's was unbounded. And nothing was learned beyond the association itself: yes/no 57.5 and pair 48.8 (chance 50; A 100 / 88.8, C 77.5 / 81.2), Timmy k=3 30.6 and k=4 25.0 (base 35.6 / 27.5), weakness and habitat induction at the base, reverse easy 23.8 and hard 24.4 (A 88.8 on the easy form from its 14 renderings), held-out species 37.5. That is the survey's prediction to the letter (2401.01286 Table 4; 2410.02355 Table 5): efficacy and paraphrase above 90, out-of-distribution locality intact, portability nil. On this ladder the editor and the fine-tune are complements, not competitors: the fine-tune's mixture buys manipulation and induction at a perplexity cost of 12 points, and the editor buys exact, general-ability-free recall of exactly the stated association and nothing derived from it.

### 30.3 A ridge on the solve rescues MEMIT, and the two editors then agree

With rho I added to the inverse (`cov_ridge`; rho = 150, about one key's squared norm, and rho = 15), the same 136 edits give bounded updates (9 to 25% of the layer norms at rho = 150, largest entry 0.12) and a model that matches AlphaEdit on every measure (Table 30.1): trained-format recall 100, bare type question 100 (weakness 0.0, habitat 20.8, as for AlphaEdit), seen-recall control 95.8, WikiText perplexity 10.71 / 10.70 (base 10.61), ARC-Easy 72.5 / 72.0, natural-label ICL 87.0 / 86.0, symbol ICL 57.8 / 57.3 (base 60.4), yes/no 61.2 / 55.0, pair 53.8 / 51.2, Timmy at the base, reverse at chance. EasyEdit's own rewrite accuracy is 88.6 at rho = 150 and 93.1 at rho = 15 (AlphaEdit 86.2), and the two ridge values are indistinguishable on the ladder, so the result is not sensitive to rho within an order of magnitude; what matters is that the null directions of C0 are regularised at all. Editing takes 26 minutes for 136 facts against arm C's 9 on the fast path, most of it the per-edit target-vector optimisation (25 steps each), and the covariance statistics (20 minutes) and AlphaEdit's projections are one-time costs per model.

The picture BASE-2 asked for is therefore clean. On this ladder a locate-then-edit method writes the stated association into the weights perfectly (recall 100 in the trained sentence and 100 on the plain question, where the fine-tunes read 20 to 25, because the edit lives at the subject token and fires on any prompt ending in the name; it also fires on the wrong question about the same name, weakness 0.0) at no measurable cost to general text, reasoning or in-context learning, which arm A cannot do (perplexity 34, ARC 66) and arm C cannot do (perplexity 22.8, ARC 58.5). And it writes nothing else: no manipulation, no induction, no reverse, where the mixture buys 77.5 / 81.2 on manipulation and 61.9 on induction with its perplexity cost. The survey's prediction held on every line, including the caution that MEMIT's hyperparameters do not transfer across model families without checking the covariance; the ridge is the check.

### 30.4 Five facts per species: the shared key averages the edits away, and naming the relation first is the fix

The `all` run gives every species five edits with subject-first prompts ("{N} is weak to" and so on). MEMIT and AlphaEdit compute the key from the subject's last token, and with the subject at the front of every prompt the five keys of a species are identical, so the closed-form update maps one key to the mean of five target vectors. AlphaEdit's 680 edits (127 minutes, rewrite accuracy 20.8) do exactly that (Table 30.1, AlphaEdit all): type recall in the trained format drops to 48.8 (from 100 with one edit per species), the bare type question to 16.1 (the base's 17.9; weakness 21.4, habitat 25.0, so none of the five facts is retrievable in plain form), the seen-recall control to 8.3, while the general measures stay at the base (WikiText 10.71, ARC-Easy 74.5, natural ICL 87.5). The model is intact and holds a blur of each species' five facts, which is what averaging five targets predicts and what 2210.07229's Eqn 19 warned about. The remedy that needs no library change is to name the relation before the subject ("As a weakness, {N} is weak to"), so that the key at the subject's last token carries the relation through the causal context and the five edits of a species no longer collide; MEMIT with the ridge on those prompts ("As a weakness, {N} is weak to"; 120 minutes, rewrite accuracy 41.9) recovers part of every fact: on the bare questions type 46.4, weakness 35.7 and habitat 68.8 (base 18 to 19; the subject-first run 16 / 21 / 25), trained-format type recall 53.1, the seen-recall control 50.0, with the general measures still at the base (WikiText 10.88, ARC-Easy 78.5, natural ICL 85.4). Half a fix: the relation words shift the subject token's representation enough to separate the five keys partly, not enough to give each fact the 100 the single-fact edit had. The record formats of 30.5 take the other route.

### 30.5 Records: a relation line before the subject does less than a clause, and a key at the field token collides across species

The owner's suggestion was to write the facts as structured records, which is what a retailer or POI database is. Two record forms were tried (properties syntax; JSON's braces collide with EasyEdit's prompt formatting). The relation-first record ("relation: weakness / species: {N} / value:", key at the subject's last token as usual; 112 minutes, rewrite accuracy 21.7) recovers habitat (bare question 81.2) and little else (type 25.0, weakness 14.3, trained-format type recall 23.8), with the general measures at the base: a record line before the subject shifts the subject token's representation less than the clause "As a weakness," did (46 / 36 / 69), so the keys stay closer to shared. The subject-first record ("species: {N} / weakness:") with the key taken at the prompt's last token, the field's colon, needed the key path implemented in the library (`fact_token: last` raised in the shipped code; the patch is in `scripts/easyedit/easyeditor.patch`) and then failed in the two-species smoke: rewrite accuracy 6.7 on ten edits, where the subject-keyed type smoke reached 100 on four. The field token's representation is nearly the same for every species, so keying there trades the collision across relations for a collision across subjects. The full run was not made. What is left untried is the survey's own remedy, one target vector per species optimised jointly over its five prompts, which changes the library's target computation rather than the prompts.

### 30.6 What the step says

- BASE-2: the editors do what the literature says and nothing more. AlphaEdit, and MEMIT once its solve is regularised, write one association per species into Qwen2.5-3B in 26 to 34 minutes with trained-format recall 100 and the plain question also at 100, where every fine-tune in this report reads about 20 on the plain question; WikiText perplexity, ARC-Easy and the ICL suite stay at the base (arm C: perplexity 22.8, ARC 58.5). They give nothing derived from the association: verification and comparison at chance, induction at the base, reverse at chance, and the type edit answers the weakness question wrongly (0.0). Efficacy and paraphrase high, locality perfect, portability nil, as predicted (2401.01286, 2410.02355).
- MEMIT's hyperparameters do not transfer to Qwen2.5-3B without a check on the covariance: half its eigenvalues are below 10^-6, the closed-form update replaced layers 4 to 8 and the model lost language (perplexity 5.7 x 10^7). A ridge on the solve fixes it (`cov_ridge`, 15 or 150 indistinguishable). Two further library facts for anyone repeating this: `edit` applies and undoes edits one at a time, so the batch protocol is `batch_edit` in one chunk with `sequential_edit=True`; and `lm_head_module` must point at the embeddings because the model ties them.
- Several facts per subject are the open problem. With subject-first prompts the five edits of a species share a key and average (per-fact plain recall 16 to 25); relation-first clauses separate them partly (46 / 36 / 69); relation-first records less (25 / 14 / 81); keys at the field token collide across subjects instead. The joint target vector per subject is the untried fix.
- For the owner's fact-DB goal (REAL-5): an editor is a fast store of one association per entity that answers in any phrasing at zero cost to the model, and it is not a store of a record and not a source of anything computed from it. The record and the hop to a category come from retrieval or from the fine-tune's augmentation and training on the hop itself (row 33).
- Saved under `models/adapters/`: `edit_memit_type_qwen2.5-3b` (the destroyed model, kept as evidence), `edit_memit_type_ridge{150,15}_qwen2.5-3b`, `edit_alphaedit_{type,all}_qwen2.5-3b`, `edit_memit_allpre_ridge15_qwen2.5-3b`, `edit_memit_allyaml_ridge15_qwen2.5-3b`; pushed to DVC.

## 31. On-policy distillation from the pre-injection model returns every general measure to the base and erases the facts with them, unless the injection mixture stays in the loop: then it keeps the facts at arm C's level and beats general-text replay on every number (TRAIN-8)

*PLAN step 31. Code: `scripts/exp_onpolicy_distill.py` (the loop), `scripts/opd_tables.py` (the tables), `data/processed/opd_prompts_v1.json` (the frozen prompt pools). Results: `results/opd_<tag>.json` (curves), `results/curriculum_Qwen2.5-3B_C_opd_<tag>.json` and `results/curriculum_Qwen2.5-3B_{base,C}_rescore0917.json` (full ladder), adapters `models/adapters/curriculum_Qwen2.5-3B_C_opd_<tag>_lora`. Tracker experiment `onpolicy_distill`; the full-ladder scores are `curriculum_v2` runs with `EVAL_ONLY`.*

Thinking Machines' personalization experiment (`references/task_training_and_services.md` section 4) is our forgetting problem with a fix attached: midtraining Qwen3-8B on documents took IF-eval from 85 to 45, and on-policy distillation afterwards, with the original model as teacher on Tulu-3 prompts, brought it to 83 while the injected knowledge stayed (43 to 41). Arm C loses the same things (section 16: WikiText perplexity 10.6 to 22.8; section 15: ARC-Easy 73.5 to 58.5), and with a LoRA student the teacher costs nothing: it is the same weights with the adapter disabled, which reproduces the untrained base's perplexity to three decimals. The loop samples the student at temperature 1 on 64 prompts x 4 samples of up to 96 new tokens per step, scores every sampled prefix under both models, and takes one AdamW step on the adapter (lr 1e-4, 10-step warmup, linear decay over 120 steps, the injection recipe's shape) on the reverse KL. Both distributions are on this machine, so the loss is the exact per-position KL(student || teacher) over the whole vocabulary rather than the blog's sampled-token estimator (`KL=sample` implements that too; not run). The prompts are the first 40 words of FineWeb-Edu documents (run 1) or Tulu-3 user turns as "Question: ... Answer:" (run 2, the blog's setting); neither pool touches the species, the ladder, the ICL suite or the WikiText slice. A third pool (WikiText-2 train prefixes, TRAIN-7's replay source) was stopped at step 15 for time once the first two agreed. A subsample point (ladder[::4], ICL suite[::2], the 200 ARC-Easy items, the WikiText slice) is taken every 30 steps; the full ladder is scored on the saved adapter through `exp_curriculum.py` as for every other arm.

**Table 31.1: on-policy distillation (OPD) of arm C toward the adapter-off base on 120 steps of 64 prompts x 4 samples, full ladder (accuracy %; the base and arm C re-scored on 2026-09-18 reproduce their section 8 and 15 numbers exactly)**

| measure | base (sec. 8) | C (sec. 15) | Cg replay (sec. 17) | C + OPD FineWeb, exact KL | C + OPD Tulu, exact KL | C + OPD FineWeb + C replay |
|---|---|---|---|---|---|---|
| recall, trained fmt | 13.1 | 100 | 100 | 37.5 | 33.8 | 96.9 |
| recall, bare | 18.1 | 19.4 | 18.8 | 21.9 | 20.6 | 21.2 |
| yes/no | 42.5 | 77.5 | 73.8 | 46.2 | 45 | 77.5 |
| pair | 51.2 | 81.2 | 58.8 | 50 | 48.8 | 68.8 |
| Timmy k=3 | 35.6 | 61.9 | 47.5 | 38.1 | 36.2 | 61.9 |
| k=4 | 27.5 | 51.9 | 46.9 | 30.6 | 34.4 | 50.6 |
| weakness | 35 | 60 | 50 | 34.4 | 32.5 | 57.5 |
| habitat | 31.2 | 29.4 | 31.9 | 30.6 | 35 | 31.9 |
| held-out species | 39.6 | 29.2 | 30.2 | 35.4 | 32.3 | 27.1 |
| unseen recall | 16.7 | 12.5 | 12.5 | 12.5 | 12.5 | 12.5 |
| seen recall control | 16.7 | 20.8 | 20.8 | 20.8 | 25 | 20.8 |
| reverse hard | 20 | 19.4 | - | 20.6 | 25 | 26.2 |
| ICL symbol | 60.4 | 78.6 | 81.2 | 75 | 72.9 | 82.3 |
| ICL natural | 84.9 | 87 | 89.1 | 89.1 | 88.5 | 90.1 |
| ARC-Easy | 73.5 | 58.5 | 71 | 73 | 75.5 | 73.5 |
| WikiText ppl | 10.614 | 22.787 | 11.599 | 10.579 | 10.598 | 11.076 |
| repair minutes | - | - | - | 271.0 (199.7 sampling) | 478.4 (322.9 sampling) | 317.8 (218.0 sampling) |

**Table 31.2: the repairs' curves: exact or sampled-token reverse KL per completion token (mean over the step's samples), mean sample length, and the subsample points (ladder[::4], ICL suite[::2], the 200 ARC-Easy items, the WikiText slice)**

| run | step | KL | replay loss | mean len | recall fmt | yes/no | Timmy k=3 | ICL sym | ICL nat | ARC-Easy | WikiText ppl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FineWeb, exact KL | 0 | 1.5651 | - | 11.1 | 100 | 80 | 60 | 71.9 | 89.6 | 58 | 25.663 |
| FineWeb, exact KL | 30 | 0.0255 | - | 96 | 30 | 55 | 27.5 | 72.9 | 85.4 | 72 | 11.192 |
| FineWeb, exact KL | 60 | 0.0207 | - | 93.6 | 22.5 | 55 | 25 | 78.1 | 90.6 | 73 | 11.124 |
| FineWeb, exact KL | 90 | 0.0087 | - | 93.4 | 22.5 | 55 | 30 | 78.1 | 90.6 | 73 | 11.087 |
| FineWeb, exact KL | 120 | 0.007 | - | 94.1 | 22.5 | 55 | 30 | 78.1 | 90.6 | 73.5 | 11.088 |
| Tulu, exact KL | 0 | 1.427 | - | 15.6 | 100 | 80 | 60 | 71.9 | 89.6 | 58 | 25.663 |
| Tulu, exact KL | 30 | 0.0368 | - | 77.5 | 30 | 50 | 32.5 | 78.1 | 87.5 | 70.5 | 11.15 |
| Tulu, exact KL | 60 | 0.0145 | - | 75.9 | 25 | 50 | 35 | 75 | 87.5 | 72 | 11.084 |
| Tulu, exact KL | 90 | 0.0101 | - | 78.7 | 25 | 45 | 32.5 | 70.8 | 88.5 | 73 | 11.097 |
| Tulu, exact KL | 120 | 0.0105 | - | 82.6 | 25 | 45 | 35 | 74 | 88.5 | 72.5 | 11.105 |
| FineWeb + C replay | 0 | 1.5651 | 0.259 | 11.1 | 100 | 80 | 60 | 71.9 | 89.6 | 58 | 25.663 |
| FineWeb + C replay | 10 | 0.1578 | 0.5355 | 93.7 | 97.5 | 80 | 60 | 74 | 89.6 | 69.5 | 12.206 |
| FineWeb + C replay | 20 | 0.0668 | 0.3761 | 95.8 | 92.5 | 75 | 60 | 78.2 | 86.5 | 70 | 11.623 |
| FineWeb + C replay | 30 | 0.0498 | 0.3335 | 95.5 | 90 | 80 | 62.5 | 77.1 | 89.6 | 69.5 | 11.568 |
| FineWeb + C replay | 40 | 0.035 | 0.3846 | 94.7 | 100 | 85 | 60 | 76 | 86.5 | 66 | 11.604 |
| FineWeb + C replay | 50 | 0.0403 | 0.381 | 94.5 | 100 | 80 | 67.5 | 80.2 | 87.5 | 69.5 | 11.72 |
| FineWeb + C replay | 60 | 0.0338 | 0.3852 | 93.4 | 92.5 | 75 | 65 | 76 | 86.5 | 70 | 11.753 |
| FineWeb + C replay | 70 | 0.0367 | 0.3741 | 93.5 | 92.5 | 90 | 67.5 | 76 | 86.5 | 70.5 | 11.729 |
| FineWeb + C replay | 80 | 0.025 | 0.3547 | 91.8 | 97.5 | 75 | 62.5 | 74 | 83.3 | 72 | 11.651 |
| FineWeb + C replay | 90 | 0.0269 | 0.2908 | 93.9 | 97.5 | 75 | 57.5 | 74 | 85.4 | 71.5 | 11.577 |
| FineWeb + C replay | 100 | 0.0251 | 0.2897 | 93.8 | 100 | 70 | 52.5 | 82.3 | 85.4 | 70.5 | 11.561 |
| FineWeb + C replay | 110 | 0.0209 | 0.2916 | 94 | 100 | 75 | 60 | 81.2 | 87.5 | 72.5 | 11.565 |
| FineWeb + C replay | 120 | 0.0195 | 0.3637 | 93.7 | 100 | 70 | 60 | 84.4 | 85.4 | 72.5 | 11.553 |

### 31.1 The general measures come back to the base, completely, in thirty steps

Arm C's own samples are the first surprise: on FineWeb prefixes every one of the 256 samples ends within 96 tokens and the mean completion is 11 tokens, because the knowledge stream taught it 24-token sentences that end in end-of-text, and the teacher charges 1.57 nats per token for that on average. By step 10 the KL is 0.13 and the mean sample 95 tokens; by step 30 it is 0.026 and the subsample perplexity is already the teacher's (11.19 against 11.12 on the script's own forward path, which reads 0.05 nats above `exp_curriculum.py`'s 10.61 for the same weights and the same function, an unresolved path difference, so those numbers are only compared with each other; the full-ladder numbers below all come from `exp_curriculum.py`, whose re-score of the base and arm C on 2026-09-18 reproduced sections 8 and 15 to the last decimal); by step 120 the KL is 0.007 nats per token, the student is the teacher on its own samples. On the full ladder the repaired adapters read WikiText perplexity 10.58 and 10.60 against the base's 10.61 (arm C 22.8, arm Cg's replay 11.6), ARC-Easy 73.0 and 75.5 against 73.5 (C 58.5, Cg 71.0), natural-label ICL 89 (base 84.9, C 87.0). Item by item the repaired model agrees with the base on 92 to 97% of the ARC items and 96 to 99% of the yes/no and pair items: not "as good as the base" but the base's answers. The two prompt pools give the same result (16% of items flip between them, under the 17.9% between two same-seed runs of arm C in section 19); Tulu prompts leave more variance in sample length (68 of 256 still end early at step 120) and cost more time.

One general skill survives partly: symbol-label ICL, which the episodes raised from 60.4 to 78.6, reads 75.0 and 72.9 after the repair. The sampled prompts never exercise that format, so the KL never sees it; the same is true of the fact prompts, and those did not survive.

### 31.2 The facts go with it

Recall in the trained sentence falls from 100 to 37.5 (FineWeb prompts) and 33.8 (Tulu) on eight options, and the subsample curve shows it at 30 by step 30, the same step at which the perplexity arrives at the teacher's: the repair and the erasure are one process. Every from-the-weights gain of arm C returns to the base: yes/no 77.5 to 46.2 and 45.0 (base 42.5), pair 81.2 to 50.0 and 48.8 (51.2), Timmy k=3 61.9 to 38.1 and 36.2 (35.6), weakness 60 to 34.4 and 32.5 (35). What is left of the facts is a residue above chance: 37.5 on eight options where the base reads 13.1, and on those items the repaired model agrees with the base on fewer than 1% of its answers, so it is neither arm C nor the base there but a scrambled remnant.

### 31.3 The adapter is not erased, it is re-purposed

The obvious mechanism, that the KL to the adapter-off teacher drives the adapter to zero, is wrong. The repaired adapters' weight deltas (B A per module, summed over the 252 LoRA layers) have 1.007 and 0.983 times arm C's Frobenius norm, and a cosine of 0.82 with arm C's delta (per module type 0.76 for down_proj to 0.89 for q_proj). The repair rotated about a third of the adapter's mass into directions that cancel its effect on general prose, and the directions that held the facts went with it. The adapter is rank 64 across every linear layer and the facts are 160 associations written into it by 12,800 short sequences; nothing localises them, so a low-rank update that cancels the adapter's output distribution on ordinary text is not orthogonal to them. This is the same lesson as section 30 from the other side: the editors write one association into one layer's weights and the general measures do not move; the fine-tune spreads the facts over the whole adapter, and any general-purpose correction to that adapter moves them.

### 31.4 Keeping arm C's mixture in the loop keeps the facts, and the pair beats replay alone on every measure

The third run adds one 16-sequence micro-batch from arm C's own training mixture to every step (knowledge texts, episodes and ICL replay at .45 / .40 / .15, the ordinary token loss with weight 1, one sixteenth of the step's rows next to the 256 sampled ones), so the facts are rewritten while the KL term repairs. That is the whole difference, and it changes the result: recall in the trained sentence 96.9 (the plain repair 37.5), yes/no 77.5 (arm C's own 77.5), Timmy k=3 61.9 (C 61.9, arm Cg 47.5), weakness 57.5 (C 60, Cg 50), k=4 50.6 (C 51.9); the one from-the-weights loss is the pair level, 81.2 to 68.8, still ten points over Cg's 58.8. The general measures stay where the plain repair put them: WikiText perplexity 11.08, 0.043 nats over the base where Cg's replay left 0.088 and arm C 0.76; ARC-Easy 73.5, the base's number exactly (Cg 71.0, C 58.5); symbol-label ICL 82.3 and natural-label 90.1, the best of any arm in the report (base 60.4 and 84.9). Item by item the repaired adapter gives arm C's answer on 96.9% of the trained-sentence recall items and the base's answer on 89.5% of the ARC items. The curve (Table 31.2) shows the two terms settling against each other: the KL flattens at 0.02 to 0.035 nats per token instead of running to 0.007, the replay loss holds at 0.29 to 0.39 (arm C's own loss on that mixture is 0.26 at step 0), recall dips to 90 on the subsample at step 30 and is back at 100 from step 40, and ARC-Easy reaches 70 by step 20 and 72.5 by the end. The adapter geometry is the same story as 31.3 with a smaller angle: the delta has 1.07 times arm C's norm and a cosine of 0.91 with it (the plain repair 0.82); the KL still rotates the adapter, and the replay batch keeps re-writing the fact directions it rotates away.

### 31.5 What the step says

On-policy distillation from the pre-injection model does what the blog reports for the general measures, and better than replay did: perplexity, ARC-Easy and natural ICL return to the base exactly, where arm Cg's 5% general-text replay left perplexity at +0.09 nats and ARC at 71. It does not do what the blog reports for the knowledge. Thinking Machines kept 41 of 43 points of document knowledge through the repair; here 100 points of recall became 35. Three differences are candidates and this step cannot separate them: their knowledge sat in the full weights of an 8B model after midtraining, ours in a rank-64 adapter; their internal-QA measure is a reading-comprehension-style recall of documents, ours the exact sentence the injection trained; and their distillation stopped where the chat behaviour was recovered, while ours ran the KL to 0.007 nats, which is the student reproducing the teacher everywhere the samples reach. With the injection mixture kept in the loop the repair keeps what the blog kept and more: the facts at arm C's level, the general measures at the base, at the cost of one pair-level drop (81 to 69). Against the owner's goals (a categoriser that keeps the injected merchant facts and its general ability) this is the best injected adapter in the report, and it beats TRAIN-7's replay on every measure: perplexity +0.04 against +0.09 nats, ARC 73.5 against 71.0, Timmy 61.9 against 47.5, yes/no 77.5 against 73.8, pair 68.8 against 58.8. The price is time: arm Cg took 29 minutes of training, the plain repairs 271 and 478 minutes (200 and 323 of them sampling), the repair with replay 318 (218 sampling), and each run on top of the 50-minute injection it repairs. Whether 0.05 nats, 2.5 ARC points and 14 induction points over Cg are worth eleven times the compute is the owner's call; the arithmetic changes if the sampler does. The cost is the other honest number: on-policy distillation on a 3B model with Hugging Face generation is 4.5 to 8 hours per 2.8 million sampled tokens on this GPU, five to nine times the injection run it repairs, and a vLLM sampler or a rented H100 would be the first thing to change before running it again. *[Correction (section 44): this section's teacher was the 4-bit base and the student a bf16 adapter read on it, while Cg was bf16 throughout, so 'exactly' and the comparison with Cg carry that mismatch; the margins over Cg are single seeds.]*

Not run: the WikiText-train prompt pool (stopped at step 15), the sampled-token estimator (`KL=sample`), a lower learning rate or an early stop at the step where perplexity first reaches the teacher (step 30 in the plain runs, where recall had already fallen to 30), a sweep of the replay weight, and the repair on the 1,000 and 5,000-species adapters of section 29, where the general-ability loss was largest. Two engineering notes for whoever runs it next: the loop alternates a generate phase (256 KV caches) with a training phase (8 x L x V logits), and on this 24 GB card the two fragment each other's allocator reserve until the driver spills to system RAM (one attempt stalled at step 55 at five minutes per step); `torch.cuda.empty_cache()` between the phases fixed it. And unsloth's fused cross-entropy on the `labels=` path sizes its chunks from free GPU memory and raised "No or negligible GPU memory available" after the KL micro-batches; the replay loss is computed from the logits instead.

## 32. LoRA at ten times the rate: the decoder adapter has one usable rate and it is 1e-4 to 2e-4, 5e-4 stores the facts and nothing else, 1e-3 destroys the model; the Flan-T5 adapter at 1e-3 learns a third of the facts and loses natural ICL, so the section 28 and 29 verdicts are not rate artefacts (TRAIN-9)

*PLAN step 32. Code: `scripts/exp_curriculum.py C Qwen/Qwen2.5-3B 800 <lr>` with `RUN_TAG=lr5e-4|lr1e-3 PERIODIC=200`; `LORA=64 RUN_TAG=lora1e-3|lora3e-3 uv run python scripts/exp_encoder.py F2A google/flan-t5-large 800 <lr>`; `scripts/lr_tables.py` (the tables). Results `results/curriculum_Qwen2.5-3B_C_lr{5e-4,1e-3}_p200.json`, `results/encoder_flan-t5-large_F2A_lora{1e-3,3e-3}.json`; adapters under `models/adapters/`.*

Tinker's LoRA primer says an adapter wants about ten times the full-fine-tuning learning rate, independent of rank, and their SFT recipes use 1e-3 for LoRA against 1e-4 for full fine-tuning (`references/task_training_and_services.md` section 5). Every decoder adapter in this report trains at 1e-4, and section 28 stopped its sweep at 2e-4, which used the facts best and forgot most; the Flan-T5 adapter at 3e-4 learned nothing while full fine-tuning at the same rate recalled 98 (section 29.3). If the primer transfers, the adapter verdicts of sections 28 and 29 are rate artefacts. Four runs settle it: arm C at 5e-4 and 1e-3 (rank 64, alpha 128, 800 steps, the fast path, seed 0, the same 30-step warmup and linear decay as every arm), and the Flan-T5 knowledge-only adapter at 1e-3 and 3e-3.

**Table 32.1: arm C on Qwen2.5-3B (rank 64, 800 steps unless stated, seed 0) across learning rates; the recipe's 1e-4 is the full-fine-tuning rate, Tinker's LoRA rate is ten times that (accuracy %)**

| measure | base | 5e-5 | 1e-4 (recipe) | 2e-4 | 5e-4 | 1e-3 | 1e-4, 1,600 steps |
|---|---|---|---|---|---|---|---|
| recall, trained fmt | 13.1 | 100 | 100 | 100 | 100 | 17.5 | 100 |
| recall, bare | 18.1 | 15.6 | 15 | 20.6 | 21.9 | 17.5 | 20 |
| yes/no | 42.5 | 73.8 | 87.5 | 96.2 | 45 | 46.2 | 97.5 |
| pair | 51.2 | 56.2 | 73.8 | 85 | 50 | 42.5 | 91.2 |
| Timmy k=3 | 35.6 | 44.4 | 56.9 | 63.8 | 39.4 | 30.6 | 81.2 |
| k=4 | 27.5 | 38.8 | 43.8 | 63.1 | 35.6 | 25.6 | 71.9 |
| weakness | 35 | 38.8 | 52.5 | 65 | 44.4 | 39.4 | 76.9 |
| habitat | 31.2 | 32.5 | 31.2 | 34.4 | 31.9 | 32.5 | 33.8 |
| held-out species | 39.6 | 33.3 | 28.1 | 27.1 | 31.2 | 43.8 | 34.4 |
| seen recall control | 16.7 | 20.8 | 20.8 | 20.8 | 20.8 | 20.8 | 20.8 |
| reverse hard | 20 | 20.6 | 21.9 | 23.8 | 26.2 | 21.9 | 20 |
| ICL symbol | 60.4 | 79.2 | 79.7 | 79.2 | 68.8 | 43.2 | 78.1 |
| ICL natural | 84.9 | 90.6 | 89.1 | 84.3 | 75.5 | 45.9 | 85.4 |
| ARC-Easy | 73.5 | 70.5 | 64 | 54.5 | 50.5 | 28 | 65.5 |
| WikiText ppl | 10.614 | 17.919 | 22.087 | 28.648 | 55.253 | 1.53946e+06 | 20.949 |
| training minutes | - | 11.2 | 11.2 | 11.2 | 14.6 | 14.4 | 25.8 |

**Table 32.2: the Flan-T5 knowledge-only arm (F2A, span prediction) under full fine-tuning and the rank-64 adapter across learning rates, with the T5Gemma adapters for comparison (span-format value in brackets)**

| measure | Flan-T5 base | full FT 1e-4 | full FT 3e-4 | full FT 1e-3 | LoRA 3e-4 | LoRA 1e-3 | LoRA 3e-3 | T5Gemma LoRA 1e-4 (F2) | T5Gemma LoRA 3e-4 (F2) |
|---|---|---|---|---|---|---|---|---|---|
| recall, trained fmt | 8.1 [10] | 14.4 [14.4] | 98.1 [100] | 100 [100] | 11.9 [16.2] | 34.4 [35] | 16.2 [10.6] | 97.5 [97.5] | 48.8 [49.4] |
| recall, bare | 18.8 [18.1] | 13.1 [16.2] | 40 [33.1] | 29.4 [36.2] | 9.4 [16.2] | 13.1 [20.6] | 23.1 [23.1] | 42.5 [38.8] | 28.1 [28.1] |
| yes/no | 45 [43.8] | 46.2 [45] | 43.8 [55] | 53.8 [45] | 45 [45] | 46.2 [45] | 45 [45] | 45 [45] | 55 [48.8] |
| pair | 50 [47.5] | 50 [51.2] | 50 [52.5] | 53.8 [50] | 50 [50] | 51.2 [50] | 50 [50] | 50 [50] | 51.2 [53.8] |
| Timmy k=3 | 35.6 | 29.4 | 30.6 | 33.1 | 29.4 | 36.2 | 31.2 | 35 | 36.2 |
| reverse hard | 28.1 [26.9] | 23.8 [22.5] | 23.8 [20] | 20.6 [28.1] | 24.4 [25.6] | 26.2 [21.9] | 30.6 [24.4] | 31.2 [33.8] | 28.8 [34.4] |
| ICL symbol | 53.6 | 52.6 | 51.6 | 40.6 | 46.4 | 47.4 | 44.3 | 43.2 | 43.8 |
| ICL natural | 87 | 85.4 | 75 | 52.6 | 78.1 | 52.6 | 46.9 | 43.8 | 39.6 |
| ARC-Easy | 56 | 34.5 | 30.5 | 29 | 31.5 | 26.5 | 18 | 39.5 | 30.5 |
| training minutes | - | - | 8.9 | 9.8 | 14.3 | 15.5 | 15.1 | 16.9 | 17.1 |

### 32.1 The decoder adapter: 5e-4 stores the facts and nothing else, 1e-3 destroys the model

Section 28 read the rate as one axis with the step count: more rate, more use of the facts, more forgetting, with 2e-4 the best 800-step point (yes/no 96.2, Timmy 63.8, ARC 54.5). The axis does not continue. At 5e-4 recall in the trained sentence is still 100, and everything the adapter is supposed to do with the facts is gone: yes/no 45.0 and pair 50.0 (the base 42.5 / 51.2), Timmy k=3 39.4 (base 35.6), weakness 44.4; the ICL suite that the episodes raise in every other arm falls below the base on natural labels (75.5 against 84.9) and to 68.8 on symbols; ARC-Easy is 50.5 and WikiText perplexity 55 (the recipe's 22). The curve says how: ARC-Easy is at 40.5 by step 200 and stays there, perplexity is 38 at step 200 and 51 from step 400, and the facts take longer to finish arriving than at 1e-4 (trained-format recall 67.5 at step 200, 90 at 400, 100 at 600; the recipe reads 25 at step 200 and 100 from step 400), while the training loss is never lower than the recipe's at the same step (0.44 against 0.37 at step 200, 0.26 against 0.28 at the end). The higher rate does not move the adapter further along the section 28 axis; it makes the optimisation noisy enough that only the most repeated signal, the 2,752 knowledge sentences, gets written. At 1e-3 the run diverges: training loss 1.0 to 1.2 through the second half, WikiText perplexity 11,700 at step 200 and 1.5 million at the end, ARC-Easy 28 (chance for four options is 25), recall 17.5, ICL 43 / 46. The model is destroyed in the first 200 steps and never comes back.

So the primer's rule does not transfer as a number, and the reason is the parametrisation, not the model. Thinking Machines write the adapter as W + (alpha / r) B A with alpha fixed at 32, so at rank 64 their update is scaled by 0.5 and "the optimal learning rate is approximately independent of rank"; the ten-times rule is stated in that scaling. Our adapters use alpha = 2r, a scale of 2 at rank 64, four times theirs, and since B starts at zero the early change in W is proportional to the scale, so our 1e-4 moves the weights like their 4e-4 and section 28's 2e-4 like their 8e-4: the recipe already sits at about the primer's rate in effective terms, and 5e-4 here is twice it, 1e-3 four times. Read that way the two new runs agree with the post rather than contradict it (they also report that LoRA "pays a larger penalty in loss as batch size increases", and our 16-sequence batches are small, so that is not the mechanism here). The rate that section 28 found, 1e-4 to 2e-4 with the step count as the real lever (1,600 steps at 1e-4 beats every 800-step rate on every from-the-weights measure at equal forgetting to 2e-4), stands, and it is not a rate artefact.

### 32.2 The Flan-T5 adapter: ten times the rate learns a third of the facts and costs the model its natural-label ICL

Section 29.3 left the encoder-decoder adapters as an open contrast: Flan-T5's rank-64 adapter at 3e-4 learned nothing (trained-format recall 11.9) where full fine-tuning at the same rate recalled 98.1, and T5Gemma's adapter did best at 1e-4 and worse at 3e-4. At 1e-3 the Flan-T5 adapter learns a third of the facts (recall 34.4 by option scoring, 35 in the span format), so the 3e-4 result was partly a rate effect, and it pays for them at once: natural-label ICL falls from 87 to 52.6 (the 3e-4 adapter kept 78.1; the full fine-tune at 3e-4 kept 75), ARC-Easy to 26.5 (chance 25; base 56), the manipulation levels stay at chance. Full fine-tuning at 1e-3 recalls 100 and keeps natural ICL at 52.6 too, so at the same nominal rate the adapter reaches a third of full fine-tuning's recall for the same damage. At 3e-3 the adapter learns nothing again (recall 16.2, 10.6 in the span format) and the damage deepens (natural ICL 46.9, ARC-Easy 18, below chance), the decoder's 1e-3 shape: the adapter's window between learning nothing and destroying the model is the single point 1e-3, and it holds a third of the facts. Read with Table 32.1, the two families agree on the shape and disagree on the position: for the decoder the usable window is 1e-4 to 2e-4 (alpha = 2r) and everything above stores facts without use or destroys the model; for Flan-T5 under span prediction the adapter has no window at all, since the rate that begins to store the facts is already the rate that removes the general abilities, while full fine-tuning at 3e-4 gets both recall (98) and most of the ICL (75). The T5Gemma exception stands as section 29 left it: its adapter recalls 97.5 at 1e-4 and 48.8 at 3e-4.

### 32.3 What the step says

TRAIN-9 asked whether the adapter verdicts of sections 28 and 29 were rate artefacts. For the decoder, no: the recipe's 1e-4 with alpha = 2r is already about the primer's ten-times rate once the scaling is translated (their alpha 32 at rank 64 scales the update by 0.5, ours by 2), 2e-4 is the top of the usable window, 5e-4 stores the facts and returns every from-the-weights and general measure to the base or below it, and 1e-3 destroys the model in 200 steps. Section 28's answer, that the step count and not the rate is the lever the recipe was missing, stands. For Flan-T5, partly: the adapter at 1e-3 does learn (recall 12 to 34), but it damages the model as much as full fine-tuning at the same rate and learns a third as much, so full fine-tuning at 3e-4 remains the working recipe for that model and the rank-64 adapter is not the right vehicle for span-prediction injection into T5. At 3e-3 it learns nothing and destroys the model, as the decoder does at 1e-3. The four runs cost 59 minutes of training. Not run: rates between 2e-4 and 5e-4 for the decoder (the section 28 axis suggests nothing there beyond more forgetting), alpha 32 with a rescaled rate as the direct test of the parametrisation reading, and T5Gemma at 1e-3.

## 33. Item rebuilds: the type induction from the weights is real and identifiable, every other attribute is at chance, no arm ever applies the rule to an unshown group, the "weakness" level was the type level, and the ICL suite was not in-distribution-inflated (EVAL-4, EVAL-5)

*PLAN step 20. Code: `universe.induction_v2` and `universe.rule_agreement`, `icl_suite.suite2_items`, `items.generate_mmlu` / `freeze_v2` (frozen as `data/processed/induction2_v1.json`, `icl_suite2_v1.json`, `mmlu_v1.json`; `exp_curriculum.py` scores them in every run from this commit on), `scripts/exp_items_v2.py` (back-fills saved weights, results `results/items2_<tag>.json`, tracker experiment `items_v2`), `scripts/induct_strata.py` (Table 33.2), `scripts/items2_tables.py` (Table 33.1). No training.*

The reviewer's two evaluation objections (QUESTIONS.md EVAL-4 and EVAL-5) were that the induction items may not identify the rule they are meant to test, and that the ICL regression suite may be in-distribution for the trained arms. Both are about the instrument, so this step rebuilds the instrument and re-reads the saved weights through it, at no training cost: every number below is a saved adapter or checkpoint scored on new items.

**Table 33.1: the step 20 item sets on the saved weights (accuracy %; chance 33.3 on the k=3 induction levels, 25 on ARC-Easy and MMLU, the ICL suites' chance is the mean of 1/k). v1 rows are the same weights' original ladder scores**

| measure | base | A | B | C | D | Cg | C 2e-4 | C 1,600 steps | C + OPD | C + OPD + replay | AlphaEdit type |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 Timmy k=3 (type, 1 demo/group) | 35.6 | 40 | 33.1 | 61.9 | 53.1 | 47.5 | 63.8 | 81.2 | 38.1 | 61.9 | 30.6 |
| v1 habitat | 31.2 | 30.6 | 30.6 | 29.4 | 41.9 | 31.9 | 34.4 | 33.8 | 30.6 | 31.9 | 36.2 |
| v1 weakness (= type) | 35 | 46.2 | 33.8 | 60 | 54.4 | 50 | 65 | 76.9 | 34.4 | 57.5 | 38.1 |
| v2 type (2 demos/group, identifiable) | 41.2 | 36.2 | 35.6 | 66.2 | 66.9 | 54.4 | 68.8 | 82.5 | 37.5 | 67.5 | 35 |
| v2 habitat | 34.4 | 35 | 29.4 | 34.4 | 46.2 | 31.9 | 46.2 | 36.9 | 35 | 36.9 | 35.6 |
| v2 diet | 29.4 | 33.8 | 26.9 | 35 | 45 | 29.4 | 33.1 | 40 | 30 | 33.8 | 28.1 |
| v2 region | 31.2 | 34.4 | 33.1 | 36.9 | 45.6 | 35.6 | 32.5 | 39.4 | 31.9 | 35 | 31.2 |
| v2 type, unseen label | 0 | 0.6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.6 |
| v2 type + context | 45 | 46.2 | 81.9 | 66.9 | 83.8 | 65.6 | 75 | 76.2 | 65 | 82.5 | 38.8 |
| v2 habitat + context | 35 | 38.1 | 72.5 | 63.1 | 69.4 | 74.4 | 60.6 | 68.8 | 65 | 86.2 | 33.8 |
| v2 diet + context | 31.9 | 35.6 | 68.1 | 73.1 | 66.2 | 64.4 | 63.1 | 67.5 | 63.1 | 56.9 | 27.5 |
| v2 region + context | 37.5 | 33.8 | 71.9 | 66.2 | 70 | 66.2 | 61.2 | 72.5 | 52.5 | 56.2 | 34.4 |
| v2 type, unseen label + context | 0 | 1.2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| v1 ICL symbol | 54.7 | 50 | 72.4 | 75.5 | 80.2 | 81.2 | 77.1 | 80.2 | 75 | 83.9 | 56.2 |
| v2 ICL symbol (Q/A, rare words) | 56.8 | 51 | 71.9 | 76.5 | 81.2 | 80.7 | 75.5 | 74 | 66.2 | 79.7 | 56.2 |
| v1 ICL natural | 84.4 | 79.7 | 84.4 | 86.5 | 88 | 87 | 83.3 | 84.9 | 87.5 | 85.9 | 84.4 |
| v2 ICL natural (Q/A) | 83.4 | 84.4 | 87.5 | 86.5 | 88 | 90.6 | 85.4 | 84.3 | 89.6 | 92.7 | 83.3 |
| ARC-Easy | 71.5 | 62 | 74.5 | 57.5 | 62.5 | 70.5 | 53 | 66.5 | 73.5 | 72 | 71.5 |
| MMLU 5-shot | 50 | 49 | 52 | 41 | 41 | 48 | 42.5 | 46.5 | 53.5 | 49.5 | 50.5 |
| WikiText ppl | 10.614 | 34.055 | - | 22.787 | 22.621 | 11.599 | 28.648 | 20.949 | 10.579 | 11.076 | 10.945 |

**Table 33.2: v1 induction items by the competing rule (type items against the habitat rule, habitat and weakness items against the type rule): share of items per stratum and each arm's accuracy on it**

| level | stratum | items | base | A | C | D | C 1,600 steps | C + OPD + replay |
|---|---|---|---|---|---|---|---|---|
| L3_induct_type_nonsense | habitat-conflicting | 26 | 19.2 | 46.2 | 53.8 | 53.8 | 80.8 | 57.7 |
| L3_induct_type_nonsense | habitat-consistent | 12 | 33.3 | 33.3 | 75.0 | 41.7 | 91.7 | 83.3 |
| L3_induct_type_nonsense | habitat-silent | 122 | 39.3 | 39.3 | 62.3 | 54.1 | 80.3 | 60.7 |
| L3_induct_type_nonsense | all | 160 | 35.6 | 40.0 | 61.9 | 53.1 | 81.2 | 61.9 |
| L4_induct_habitat | type-conflicting | 24 | 41.7 | 37.5 | 12.5 | 25.0 | 12.5 | 25.0 |
| L4_induct_habitat | type-consistent | 17 | 5.9 | 23.5 | 47.1 | 47.1 | 47.1 | 41.2 |
| L4_induct_habitat | type-silent | 119 | 32.8 | 30.3 | 30.3 | 44.5 | 36.1 | 31.9 |
| L4_induct_habitat | all | 160 | 31.2 | 30.6 | 29.4 | 41.9 | 33.8 | 31.9 |
| L4_induct_weakness | habitat-conflicting | 22 | 40.9 | 40.9 | 45.5 | 40.9 | 77.3 | 54.5 |
| L4_induct_weakness | habitat-consistent | 18 | 44.4 | 50.0 | 55.6 | 61.1 | 94.4 | 72.2 |
| L4_induct_weakness | habitat-silent | 120 | 32.5 | 46.7 | 63.3 | 55.8 | 74.2 | 55.8 |
| L4_induct_weakness | all | 160 | 35.0 | 46.2 | 60.0 | 54.4 | 76.9 | 57.5 |

### 33.1 The v1 induction items: one demo per group cannot separate the rules, and "weakness" was never a separate rule

A v1 induction item (`universe.ladder`, section 8) shows one demo per group, so any attribute on which the three demos happen to differ is a rule consistent with the demos, and the item tests "group by the intended attribute" only when the competing rules point elsewhere. `rule_agreement` classifies each v1 item against its main competitor: for a type item, whether the habitat rule is silent (no demo shares the query's habitat), consistent (the habitat-sharing demo is the gold demo) or conflicting (it is another demo); for habitat and weakness items, the same against the type rule. Two things follow from the species table before any model is scored. First, 122 of the 160 type items are habitat-silent and 119 of the 160 habitat items type-silent: for three quarters of the items the competing rule offers no answer, so the item is decided by whichever rule the model has, and only the 38 type items and 41 habitat items where the rules conflict or coincide can tell them apart. Second, as the reviewer's DATA-1 (PLAN row 23) already noted, weakness is a bijection of type in this universe (`WEAKNESS = dict(zip(TYPE_LIST, TYPE_LIST[3:] + TYPE_LIST[:3]))`): two species share a weakness exactly when they share a type, so "group by weakness" and "group by type" are one partition and the L4 weakness level is a second draw of type items. The claim in sections 8 and 15 that induction "transfers to the weakness attribute, which no episode groups by" (54 against 45 at the base) is therefore not a transfer result: the episodes group by type on a quarter of their draws, and that is the rule the weakness items reward. The corrected reading is that the weakness level replicates the type level within noise (arm C 60.0 against 61.9, the 1,600-step run 76.9 against 81.2), and the real held-out attribute in v1 is habitat, where the arms sit at the base.

Table 33.2 scores the saved arms on the strata. On habitat items where the type rule conflicts with the habitat rule, arm C reads 12.5 (chance 33.3, 24 items) and the 1,600-step run 12.5; where the two rules coincide, 47.1 and 47.1; where type is silent, 30.3 and 36.1. Arm C is answering habitat items by type: it loses on exactly the items where type gives the wrong answer and wins where type gives the right one, and its pooled 29.4 is the average of the two. Arm D, the only arm above the base on pooled habitat (41.9), is at 25.0 on the conflicting stratum too; its gain sits in the type-silent items (44.5), where nothing in the item says which rule applies. On type items the picture is the mirror image: arm C reads 75.0 where habitat coincides with type and 53.8 where it conflicts, so some of the "type" induction is a habitat lean too, and the 1,600-step run at 80.8 on the conflicting stratum is the one arm that clearly holds the type rule against a competing one. The reviewer's reading, that "Timmy from the weights" is a learned default of grouping by type, is right for habitat and mostly right for the pooled number; it is not the whole story for the type items, where the conflicting stratum still separates the trained arms from the base (19.2).

### 33.2 The v2 induction items: type from the weights is real, the other attributes are at chance, and the unseen label is never chosen

`universe.induction_v2` builds items whose rule is identifiable by construction: each of the three groups has two demos that share the generating attribute and differ on every other one (type, habitat, diet, region; weakness follows type), so "group by b" mislabels a demo pair for every b other than the generating attribute, and the query is a seen species outside the demos. Four levels of 160 (type, habitat, diet, region), and a fifth, the unseen-label protocol of the ABFT paper (2505.14233): demos for two types only, three labels on offer, a query of a third type whose answer is the label no demo carries, which a model can only get by applying the rule ("this is like neither group") rather than by matching. Every item also has the field-guide entries of its seven species as a with-context condition. Each set was scored on the saved weights of the arms in Table 33.1 by `scripts/exp_items_v2.py` (a 1,536-token window, wider than the ladder's 768, which is why the v1 suite and ARC-Easy numbers in that table differ by a few points from the same weights' section 15 numbers; the comparisons below are within the table). *[Correction (section 44): these re-reads put bf16 adapters on the 4-bit base, which moves the section 33 means by 0 to 6 points; that, more than the window, is the difference from section 15.]*

The type induction survives the rebuild and comes out slightly stronger: arm C 66.2 (v1 61.9), arm D 66.9, C at 2e-4 68.8, the 1,600-step run 82.5, C after on-policy distillation with replay 67.5, against a base of 41.2 and the knowledge-only and episode-only arms at 36. The v1 headline was not an artefact of one demo per group; with six demos and a rule the demos pin down, the trained arms group by type from the weights. What the v1 items could not show, the v2 items do: on habitat, diet and region the same arms are at chance (arm C 34.4 / 35.0 / 36.9; the 1,600-step run 36.9 / 40.0 / 39.4; arm Cg 31.9 / 29.4 / 35.6), with one exception, arm D at 46.2 / 45.0 / 45.6, twelve points above chance on all three. The mixture arms learned one rule, type, which is the attribute the episodes group by most often and the one the knowledge texts state first; the sequential arm, whose second phase is episodes over all four attributes without knowledge text, is the only one that carries any of the other three from its weights. Section 33.1's stratification said arm C answers v1 habitat items by type; the v2 items, where the type rule is inconsistent with the demos by construction, say it has no habitat rule to fall back on and lands at chance.

The unseen-label level is zero for every model in the table, the base included: 0.0 on 160 items for ten of eleven arms, 0.6 for arm A. When the query's type is not among the demoed types, no arm chooses the label that no demo carries; every one of them copies a label from a demoed group. With the entries in context the number stays at zero. This is the cleanest statement so far of what "induction from the weights" is here: a learned mapping from the query's type to the label of the demo of the same type, applied when such a demo exists and never extended by elimination. The ABFT paper's unseen-label protocol was designed to separate copying from induction, and on this universe every arm is on the copying side of it.

With context the picture is the ICL one from sections 8 and 15: the base reads 45 / 35 / 32 / 38 across the four attributes and the trained arms 60 to 86, arm B (episodes only) 82 / 73 / 68 / 72 and arm D 84 / 69 / 66 / 70, so given the entries every attribute's rule is applied, and the on-policy-distilled adapter with replay is the best type-with-context arm at 82.5 / 86.2. The AlphaEdit checkpoint of section 30 is at the base on every v2 level (type 35.0, type with context 38.8): the editors write associations, not the skill.

### 33.3 The out-of-distribution ICL suite and MMLU: the preservation result stands, and MMLU tracks ARC-Easy

The reviewer's objection to the ICL suite was that its symbol labels come from the same generator as the training episodes' labels and its template is replay template 0, so "78 against 51" could be familiarity. `icl_suite.suite2_items` keeps the four held-out datasets and changes both: a "Q: ... / A:" frame with a `---` separator that no replay template uses, and symbol labels drawn from 50 uncommon English nouns (gimbal, tuffet, quince, ...) that the syllable-string, number and letter generator never produces. On it every arm reads within a few points of its v1 suite number scored in the same pass: arm C 76.5 against 75.5 on symbols and 86.5 against 86.5 on natural labels, arm D 81.2 / 80.2 and 88.0 / 88.0, arm Cg 80.7 / 81.2 and 90.6 / 87.0, arm B 71.9 / 72.4, the base 56.8 / 54.7. Two arms lose on the new frame: the 1,600-step run (74.0 against 80.2 on symbols) and the plain on-policy-distilled adapter (66.2 against 75.0), which spent 120 steps matching the base on FineWeb continuations; with replay in the loop it reads 79.7 / 83.9. The natural-label numbers are, if anything, higher on the new frame for every arm. The preservation result was not template or label familiarity: the replay stream protects the skill, not the string.

The 200-item, 5-shot MMLU slice (`items.generate_mmlu`; test questions of at most 60 words with choices of at most 10 words, five same-subject dev demonstrations, the ladder's cloze format) is the second forgetting proxy the plan asked for. It agrees with ARC-Easy on every arm: base 50.0 (ARC 71.5 in the same pass), arm C 41.0 (57.5), arm D 41.0 (62.5), C at 2e-4 42.5 (53.0), the 1,600-step run 46.5 (66.5), arm Cg 48.0 (70.5), C after on-policy distillation with replay 49.5 (72.0) and without it 53.5 (73.5), arm B 52.0 (74.5). The loss under injection is nine MMLU points against fourteen ARC points, general-text replay recovers most of it and on-policy distillation all of it, and the AlphaEdit checkpoint sits at the base (50.5). MMLU at 200 items has the same half-width as ARC at 200 (about seven points), so it confirms rather than sharpens; the WikiText slice of step 24 remains the instrument that separates arms.

### 33.4 What the step says

EVAL-4: the induction items were not identifiable and the weakness level was the type level, and both mattered for the reading of sections 8 and 15, but not for the headline. On items whose rule is pinned down by the demos, the trained arms still group by type from their weights (66 to 83 against a base of 41), which is the claim the report has made since section 8; what falls is everything around it: the "transfer to weakness" was a second measurement of type, the habitat level was answered by type where type gave an answer, and no arm has a habitat, diet or region rule in its weights except arm D, weakly. The unseen-label result draws the line: the learned skill maps a query to the label of the demo that shares its type and never extends to a group without a demo, on any arm, with or without context. For the owner's categoriser this is the finding to carry: a model fine-tuned on a user's labelled transactions will label a new merchant like the labelled merchant it most resembles on the one feature the training grouped by, and will not invent or select a category it has never seen attached to a similar example; unseen category names need the name in the prompt (the with-context rows) or a retrieval step, not more fine-tuning.

EVAL-5: answered in the suite's favour. The ICL preservation result holds on a template and label source the training never saw, MMLU tracks ARC-Easy, and the WikiText slice already replaced the coffee passage in step 24. From this commit every run scores the v2 induction levels (without and with context), the v2 suite and MMLU alongside the old ladder, so the next arms are read on both instruments at no extra cost beyond about a minute of evaluation. Not done: the inconsistent-label null of Wei et al. (labels shuffled so no rule fits, accuracy should be chance for any inducer), and identifiable items for the 1,000 and 5,000-species universes.

## 34. Morphology realism: the stem transfers to names with no trained part but the marker, at 60 to 100 as the marker's coverage goes from 0.3 to 1.0, and a word-initial stem works as well as a suffix (DATA-3)

*PLAN step 21 (DATA-3 only; the row's merchant questions DATA-4 and DATA-2 became row 36). Code: `universe.build(morph_p, morph_pos)`, `universe.PREFIX_MARKER`, `universe.probes_v2` (M2 levels), `items.freeze-morph` / `freeze-probes2` (frozen sets `*_v1_morph{30,50,90,100,70p}.json`, `probes2_v1*.json`), `MORPH_P` / `MORPH_POS` in `scripts/exp_curriculum.py`, `scripts/morph_tables.py` (the table). Results `results/curriculum_Qwen2.5-3B_E_m*_p200.json`, `..._E_rescore.json`, `..._base_m_probes2.json`, `..._base_m_m70p.json`, `..._base_probes2.json`; adapters `models/adapters/curriculum_Qwen2.5-3B_E_m*_p200_lora`.*

Section 8.4 reported that on a universe where 70% of names end in a suffix tied to their type, arm E classifies a never-trained name that carries the suffix at 93.8 in the trained answer format, and the reviewer raised three doubts (QUESTIONS.md DATA-3): the probe names recombine the same 36 prefixes and 24 suffixes as the trained names, so "never-trained name" meant "never-trained combination"; the marker reliability had two points, 0 and 0.7; and real stems (drug INN stems) are often word-initial. This step answers the three with the same recipe (arm E: knowledge .45, episodes .40, ICL replay .15, 800 steps, rank 64, the fast path, one seed) on six universes: suffix markers at 0.3, 0.5, 0.7, 0.9 and 1.0 of the names, and prefix markers at 0.7 (`PREFIX_MARKER`, eight stems at the front of the name, the suffix drawn from the plain pool). Every universe has its own species names, so each has its own frozen ladder, held-out induction and probe sets, and every run also scores the new `probes_v2`: names whose non-marker part is a prefix (or, in the prefix universe, a suffix) that no trained name contains, so the marker is the only familiar thing in the string. The section 8 adapter is re-scored with the new probes, and the untrained base on each universe is the floor.

**Table 34.1: arm E (the section 8 mixture on a morphology universe) across marker shares and marker positions, with the section 8 probes (never-trained names from trained name parts) and the step 21 probes (an unseen prefix or suffix beside the marker); chance 12.5 on every probe level (accuracy %)**

| measure | base, morph 0.7 (sec. 8) | base, morph 0.7, re-scored | E 0.7 (sec. 8) | E 0.7 (sec. 8), re-scored | E 0.3 | E 0.5 | E 0.7 fast path | E 0.9 | E 1.0 | base, prefix 0.7 | E prefix 0.7 | base, plain | C, plain (sec. 15) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| marked probe, trained fmt | 12.5 | 12.5 | 93.8 | 93.8 | 60.4 | 79.2 | 95.8 | 97.9 | 100 | 16.7 | 100 | 12.5 | 10.4 |
| plain probe, trained fmt | 10.4 | 12.5 | 14.6 | 16.7 | 14.6 | 14.6 | 14.6 | 16.7 | 18.8 | 16.7 | 8.3 | 14.6 | 14.6 |
| marked probe, bare | 12.5 | 12.5 | 12.5 | 12.5 | 12.5 | 12.5 | 18.8 | 12.5 | 12.5 | 18.8 | 14.6 | 12.5 | 12.5 |
| plain probe, bare | 10.4 | 10.4 | 12.5 | 12.5 | 12.5 | 12.5 | 12.5 | 12.5 | 12.5 | 14.6 | 12.5 | 14.6 | 12.5 |
| unseen-part marked probe, trained fmt | - | 8.3 | - | 95.8 | 56.2 | 81.2 | 95.8 | 100 | 100 | 18.8 | 97.9 | 8.3 | - |
| unseen-part plain probe, trained fmt | - | 10.4 | - | 14.6 | 12.5 | 16.7 | 16.7 | 12.5 | 16.7 | 12.5 | 8.3 | 10.4 | - |
| unseen-part marked probe, bare | - | 12.5 | - | 12.5 | 12.5 | 12.5 | 12.5 | 12.5 | 12.5 | 16.7 | 12.5 | 12.5 | - |
| unseen-part plain probe, bare | - | 12.5 | - | 12.5 | 12.5 | 12.5 | 14.6 | 12.5 | 14.6 | 12.5 | 12.5 | 12.5 | - |
| recall, trained fmt | 10 | 10.6 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 13.8 | 100 | 13.1 | 100 |
| recall, bare | 18.1 | 18.1 | 18.1 | 18.1 | 10.6 | 13.8 | 18.1 | 8.8 | 15 | 15.6 | 12.5 | 18.1 | 19.4 |
| yes/no | 41.2 | 42.5 | 96.2 | 96.2 | 95 | 97.5 | 96.2 | 97.5 | 97.5 | 42.5 | 96.2 | 42.5 | 77.5 |
| Timmy k=3 | 40.6 | 40 | 75 | 75.6 | 71.2 | 71.2 | 82.5 | 86.9 | 85.6 | 42.5 | 81.9 | 35.6 | 61.9 |
| k=4 | 38.8 | 37.5 | 76.9 | 75.6 | 69.4 | 73.8 | 79.4 | 78.1 | 91.9 | 40 | 81.2 | 27.5 | 51.9 |
| held-out species | 33.3 | 35.4 | 45.8 | 45.8 | 38.5 | 31.2 | 50 | 74 | 85.4 | 43.8 | 54.2 | 39.6 | 29.2 |
| weakness (= type) | 49.4 | 48.8 | 76.2 | 77.5 | 68.8 | 71.2 | 86.2 | 85.6 | 86.9 | 39.4 | 82.5 | 35 | 60 |
| habitat | 32.5 | 32.5 | 33.1 | 33.1 | 38.1 | 34.4 | 31.9 | 31.2 | 31.2 | 32.5 | 33.1 | 31.2 | 29.4 |
| ICL symbol | 60.4 | 60.4 | 81.8 | 82.3 | 75 | 76.5 | 78.1 | 77.1 | 77.1 | 60.4 | 74 | 60.4 | 78.6 |
| ICL natural | 84.4 | 84.9 | 84.4 | 84.4 | 87.5 | 87 | 88 | 86.5 | 88 | 84.9 | 88 | 84.9 | 87 |
| ARC-Easy | - | 73.5 | - | 64 | 67 | 55.5 | 67.5 | 61.5 | 61.5 | 73.5 | 63.5 | 73.5 | 58.5 |
| WikiText ppl | - | 10.614 | - | 23.893 | 24.629 | 25.626 | 24.957 | 29.466 | 22.855 | 10.614 | 24.738 | 10.614 | 22.787 |
| training minutes | - | - | 24.3 | - | 14.3 | 14.3 | 14.6 | 14.4 | 14.4 | - | 14.5 | - | 50.3 |

### 34.1 The stem transfers to genuinely novel names, and the effect is a monotone function of how many trained names carry the marker

The reviewer's first doubt does not survive the new probes. A marked probe whose prefix no trained name contains is classified in the trained answer format at 95.8 by the 0.7 run, the same number as the recombined-part probes of section 8 (95.8 on this run, 93.8 on the section 8 adapter), against 16.7 for the unseen-part plain control and 12.5 chance; at 0.9 and 1.0 coverage it is 100. The model is not matching a remembered prefix-suffix pair; it reads the suffix on a string it has never seen any part of except the suffix. The bare-option version of every probe stays at chance for every run (12.5 to 18.8), as the section 8 recall levels did: the skill is expressed in the trained sentence form, and a bare type name after "Answer:" does not elicit it.

The second doubt becomes a curve. One clarification first: in this universe a marker, when present, is always its type's (a name ending in `-orc` is Voltrix without exception), so `morph_p` is the *coverage* of the marker, the share of trained names that carry it, not its reliability; the reviewer's "slope against marker reliability" is a slope against coverage, and a reliability sweep (markers that sometimes lie) is a different experiment. Against coverage the marked-probe accuracy in the trained format is 60.4 / 79.2 / 95.8 / 97.9 / 100 at 0.3 / 0.5 / 0.7 / 0.9 / 1.0, and the unseen-part version 56.2 / 81.2 / 95.8 / 100 / 100: monotone, above chance by 44 points at the lowest coverage, at the ceiling from 0.9. The plain controls stay at 12.5 to 18.8 throughout. Recall of the trained species is 100 at every coverage (the facts do not depend on the stem), manipulation is 95 to 97.5 throughout, and the type induction from the weights rises with coverage (Timmy k=3 71.2 / 71.2 / 82.5 / 86.9 / 85.6; k=4 69.4 / 73.8 / 79.4 / 78.1 / 91.9) because on a marked universe the demos' and query's stems do part of the grouping. The clearest effect is on induction over held-out species without context, the level that is unknowable on the plain universe (section 15, arm C 29.2): 38.5 / 31.2 / 50.0 / 74.0 / 85.4. At full coverage every held-out species carries its type's stem, and the model places an entity it has never read about by its name alone at 85, which is the "a new drug name lands near its class" behaviour the owner asked for in section 8, now with its dependence on how consistently the class marks its members. The general measures show no trend in coverage (ICL 75 to 78 / 86.5 to 88, ARC-Easy 55.5 to 67.5, WikiText perplexity 22.9 to 29.5 with the 0.9 run highest); the five runs are five universes with different names and so different training text, and the perplexity spread between them (0.25 nats) is far above the 0.03-nat same-text replicate spread of section 16, so the universe's names move the perplexity cost more than the coverage does.

### 34.2 A stem at the front of the name works as well as one at the end

The third doubt was that real stems are often word-initial (the INN stems `-vir`, `-mab`, `-prazole` are suffixes, but `cef-`, `sulfa-` and most retailer variants, `AMZN Mktp`, `SQ *`, are prefixes) and that a suffix-only result might rest on the decoder's recency. On the prefix universe (eight stems at the front of 70% of the names, suffixes drawn from the plain pool, otherwise the same recipe) arm E classifies a marked probe in the trained format at 100 and a marked probe whose *suffix* no trained name contains at 97.9, against 8.3 for both plain controls; the rest of the ladder is the suffix run's within a few points (recall 100, yes/no 96.2, Timmy k=3 81.9 against 82.5, k=4 81.2 against 79.4, held-out species 54.2 against 50.0, ICL 74 / 88, ARC-Easy 63.5, perplexity 24.7). The untrained base reads 12.5 to 18.8 on every probe level on every universe (suffix 0.7: 12.5 / 8.3 on the marked and unseen-part marked probes; prefix 0.7: 16.7 / 18.8; the plain universe 12.5 / 8.3), so nothing in the stems is readable before training. The position of the stem does not matter to the mechanism; what matters is that the tokeniser exposes it, and both `Zev`-initial and `-orc`-final names split into a stem piece the adapter can attach the type to. The section 8 adapter, re-scored with the new probes on the current scorer, reproduces its 93.8 on the recombined-part probes and reads 95.8 on the unseen-part ones (plain controls 16.7 / 14.6), so the original result already held for genuinely novel names; the rest of its ladder re-scores to the section 8 numbers within a point (Timmy 75.6 against 75.0, k=4 75.6 against 76.9, held-out species 45.8 against 45.8). *[Note: `SQ *` marks the payment processor (Square), not a merchant family, and carries no category; section 35's normaliser strips it. The family-marker reading applies to `AMZN Mktp`-style prefixes only.]*

### 34.3 What the step says

DATA-3's three doubts, answered: the morphology transfer of section 8 is not a recombination effect (unseen-part probes read the same as recombined-part probes at every coverage), it is a monotone function of marker coverage (60 at 0.3, 79 at 0.5, 96 at 0.7, 98 to 100 above), and it does not depend on the stem being a suffix (prefix markers 100 / 97.9). The one correction is to the reviewer's word: `morph_p` is coverage, and a reliability sweep, where a marker sometimes points to the wrong class, has not been run; nor has the INN-stem-to-ATC-class replicate on real drug names, which needs a real name list and is the natural first item for row 36's realism work if the owner wants the pharmaceutical case as well as the merchant one. For the categoriser the reading is direct: a merchant family that shares a visible token (`AMZN`, `SQ *`, a franchise prefix) will be placed with its family by a fine-tuned model even when the rest of the string is new, and how reliably depends on how consistently the family's members carry the token in the training data, with about 60% of a perfect score at 30% coverage and the ceiling from 90%. Six runs, 87 minutes of training.

## 35. Merchant realism: a regex normaliser or six renderings per merchant close the bank-string gap for the embedding route, the normaliser alone lifts the 0.5B model from chance to its clean-name level, and the products-to-category bridge, not recall, is what product ambiguity breaks (DATA-4, DATA-2)

*PLAN step 36. Code: `merchants.renderings` / `rendering_texts` / `normalize` / `bank_hard_string` / `build_v2`, `scripts/exp_merchant_realism.py` (both routes, both sets, both conditions; per-item records under `results/per_item/merchant_realism.*.jsonl`), `scripts/merchant_realism_tables.py` (the tables). Results `results/merchant_realism.json`; tracker experiment `merchant_realism`.*

Section 4 left the merchant problem at a known place: contrastive fine-tuning of a small embedding model learns every trained merchant and carries 70.8% of them to the bank-statement string it never saw, while the 0.5B language model, under every training condition and even with the fact in the prompt, sits at chance on that string. The reviewer's DATA-4 asked why the obvious fix, training on noisy renderings and normalising the string at test time, was never tried; DATA-2 asked how much of the category result is recall of the products against the products-to-category bridge, and whether the bridge survives products that do not name their category. Both are answered here with section 4's two recipes unchanged (the embedding route: all-MiniLM-L6-v2, six contrastive epochs, 96 trained and 24 held-out merchants; the language-model route: Qwen2.5-0.5B, full fine-tuning at 1e-5 for 420 steps, all 120 merchants) on two merchant sets and two training conditions. The renderings (`merchants.rendering_texts`) are six card-statement strings per merchant drawn from twelve templates disjoint from the test template: processor prefixes (`TST*`, `PAYPAL *`, `PP*`, `CKCD`), the name cut to 8 or 10 characters, a vowel-dropped abbreviation (`ELRHLM`), store numbers, cities and dates, each tied to the merchant's name and products in a sentence. The normaliser (`merchants.normalize`) is fourteen prefix rules, four regexes for numbers, dates and store tokens, a city and state list, and title-casing; it removes noise and does not restore a truncated name. Two test strings per merchant: the held-out `bank_string` of section 4 (an intact name inside noise) and a new `bank_hard_string` with the name cut to eight characters (`CHKCARD FALVARRO 1032 OMAHA NE`), each scored raw and normalised. The v2 set (`merchants.build_v2`) keeps the 120 names and gives each category's product pool two products of the next category, so a product no longer names its category, and makes 20% of the merchants sell one product of another category.

**Table 35.1: the embedding route (all-MiniLM-L6-v2, contrastive, 96 trained and 24 held-out merchants): 12-way category by nearest category text (accuracy %; chance 8.3)**

| test string | v1 clean | v1 + renderings | v2 clean | v2 + renderings |
|---|---|---|---|---|
| name (trained) | 100 | 100 | 100 | 100 |
| bank string (trained) | 70.8 | 100 | 81.2 | 100 |
| bank string, normalised (trained) | 100 | 100 | 100 | 100 |
| truncated string (trained) | 44.8 | 61.5 | 42.7 | 57.3 |
| truncated, normalised (trained) | 92.7 | 91.7 | 95.8 | 96.9 |
| description (trained) | 100 | 100 | 100 | 100 |
| name (held out) | 12.5 | 8.3 | 0 | 0 |
| bank string (held out) | 4.2 | 8.3 | 8.3 | 0 |
| bank string, normalised (held out) | 12.5 | 8.3 | 0 | 0 |
| truncated, normalised (held out) | 12.5 | 8.3 | 8.3 | 4.2 |
| description (held out) | 100 | 100 | 83.3 | 79.2 |
| training minutes | 0.35 | 0.42 | 0.33 | 0.41 |

**Table 35.2: the language-model route (Qwen2.5-0.5B, full fine-tuning at 1e-5, 420 steps, all 120 merchants trained): section 4's formats plus the normalised and truncated bank strings, and the products-to-category bridge (accuracy %; chance 8.3 on 12-way, 25 on 4-way)**

| measure | v1 clean | v1 + renderings | v2 clean | v2 + renderings |
|---|---|---|---|---|
| clean_category (12-way) | 41.7 | 40.8 | 28.3 | 27.5 |
| bank_category | 12.5 | 18.3 | 9.2 | 9.2 |
| bank, normalised | 44.2 | 45 | 25.8 | 31.7 |
| truncated bank | 10.8 | 12.5 | 8.3 | 10 |
| truncated, normalised | 36.7 | 36.7 | 26.7 | 24.2 |
| sells (4-way) | 51.7 | 60.8 | 55 | 46.7 |
| reverse (4-way) | 42.5 | 30 | 43.3 | 29.2 |
| P(clean_category | sells correct) | 45.2 | 43.8 | 28.8 | 32.1 |
| merchants with sells correct | 62 | 73 | 66 | 56 |
| clean_category, single-category merchants | - | - | 31.2 | 30.2 |
| clean_category, multi-category merchants | - | - | 16.7 | 16.7 |
| sells, single / multi | - | - | 56.2 | 50 |
| sells, multi | - | - | 50 | 33.3 |
| perplexity, neutral English | 26.01 | 25.69 | 26.34 | 25.62 |
| training texts | 1680 | 2400 | 1680 | 2400 |
| training minutes | 2.4 | 2.5 | 2.4 | 2.4 |

### 35.1 The embedding route: renderings in training or a normaliser at test time each close the bank-string gap, and the truncated name needs both

The clean v1 column reproduces section 4.3.1 to the decimal (bank string 70.8 on the 96 trained merchants, name 100, held-out merchants at chance), so the two additions are read against a replicated baseline. Six card-statement renderings per merchant in the contrastive pairs take the never-seen bank string from 70.8 to 100. The normaliser alone, applied to the test string of the clean model, also takes it to 100: on section 4's test template the name is intact inside the noise, and once the prefix, store number, city and date are stripped the string *is* the name, which the clean model already maps at 100. The truncated string is where the two differ. Raw, the clean model reads 44.8 and the renderings model 61.5; normalised, 92.7 and 91.7. The normaliser does most of the work on that string too (it strips everything but the eight-character stem, and the WordPiece pieces of `FALVARRO` are the pieces of `Falvarro Llc`), and training on renderings adds 17 points to the raw number but nothing to the normalised one. The held-out merchants stay at chance in every cell (0 to 12.5 on 24 merchants), as in section 4: nothing about a name says what the store sells, and neither renderings nor normalisation changes that. On the v2 set the same pattern holds (81.2 to 100 with renderings, 100 normalised, truncated 42.7 / 57.3 raw and 95.8 / 96.9 normalised), with one difference that belongs to DATA-2: the *description* of a held-out merchant, which the untouched model and every v1 model map at 95.8 to 100, maps at 83.3 and 79.2 on v2, because two of a category's ten products now also belong to its neighbour and a fifth of the merchants sell one product of another category. The bridge from products to category is exact when the products are diagnostic and loses 17 to 21 points when they are not, on a model that never trained on those merchants at all.

### 35.2 The language-model route: the normaliser does what training could not, renderings add little, and ambiguity breaks the bridge

The clean v1 column reproduces section 4.3.2's best row to the decimal (clean category 41.7, bank string 12.5, sells 51.7, reverse 42.5, perplexity 26.0 at 1e-5), so again the additions are read against a replicated baseline. Section 4 found that no training condition, and not even the fact in the prompt, lifted the 0.5B model off chance on the bank string; the normaliser does: 12.5 to 44.2, which is the model's own clean-name number (41.7), and 36.7 on the truncated string against 10.8 raw. The knowledge was there and the string was the problem, exactly as the embedding route said in section 4.3.1 with its 70.8, and a hundred lines of regex fix it for a 0.5B decoder as they do for a 22M encoder. Training on the renderings is a different story for this route: bank string 12.5 to 18.3, normalised 44.2 to 45.0, truncated 10.8 to 12.5, so six noisy renderings per merchant in 2,400 training sentences teach a 0.5B model almost nothing about reading its own bank strings, where the same renderings took the embedding model from 70.8 to 100. The renderings are not free either: reverse lookup falls from 42.5 to 30.0 in both sets, because the added sentences run name-to-products only and dilute the backward statements (section 27's lesson about backward recall coming from backward sentences), while `sells` rises 51.7 to 60.8 on v1. General perplexity is unchanged (25.6 to 26.3).

DATA-2's question was how much of the category result is recall of the products and how much the bridge from products to category. On v1, P(category correct | the merchant's products were recalled in the 4-way `sells` item) is 45.2 against 41.7 unconditionally, on 62 merchants: knowing what the store sells raises the odds of naming its category by three points, so the failures are mostly failures of the bridge, not of recall. On v2, where two of every category's ten products belong to the neighbouring category too and a fifth of the merchants sell one product from elsewhere, the category numbers fall while recall holds: clean category 41.7 to 28.3 (renderings 27.5), P(category | sells) 45.2 to 28.8, while `sells` stays at 55.0 and reverse at 43.3. Inside v2 the multi-category merchants are the casualties: category 16.7 against 31.2 for the single-category ones, `sells` 50.0 against 56.2. The embedding route's held-out description number told the same story from the other side (100 to 79 to 83 on merchants it never trained on). A model that has recalled "sushi rolls, frozen vegetables and packaged snacks" has no more idea than chance which of two categories that is, and no amount of merchant training fixes it, because the ambiguity is in the world, not in the weights.

### 35.3 What the step says

DATA-4: the fix the reviewer asked about works, and which half of it matters depends on the model. For the embedding route, renderings in training and a normaliser at test time are each sufficient on section 4's bank strings (70.8 to 100 either way) and the normaliser carries the truncated strings too (44.8 to 92.7); for the 0.5B decoder only the normaliser works (12.5 to 44.2, its clean-name level), and renderings in training barely move it while costing backward recall. The string-alignment failure that section 4 reported for the decoder was real and is solved outside the model. DATA-2: the category result is mostly bridge, not recall (recalling the products lifts category accuracy by three points), and the bridge is what product ambiguity breaks: the v2 set costs 13 points of category on a model whose product recall is unchanged, and multi-category merchants read at half the single-category rate. For the owner's categoriser both halves point the same way: normalise the statement string before anything else (it is the cheapest and largest gain in this report), train the encoder route on renderings if the strings truncate, and expect the products-to-category step, not merchant recall, to be the residual error on real merchants, whose products are ambiguous by nature; a category-level signal (the user's own labels, row 35) rather than a product description is what closes that gap. Cost: eight training runs, 12 minutes in all.

Not run: renderings for the 3B decoder or the section 24.7 encoders (the recipe here is section 4's, chosen so the baselines replicate), a learned normaliser, and the v2 set with the user-labelled evaluation of row 35, where the ambiguity will matter most.

## 36. Real-use replicate: on a Zipf history with noisy strings the prototype recognises the merchants it was shown (85 to 90) and not the others (14 to 18), the category's name is the only thing that categorises an unseen real chain (38 to 56), an opaque unseen merchant is at chance for every method, label propagation loses to the prototype below ten labels, and amount and weekday features hurt (REAL-1, REAL-4, GRAPH-2, GRAPH-6)

*PLAN step 22. Code: `ai_experiments.transactions` (the history; frozen as `data/processed/transactions_v1.json`), `scripts/exp_realuse.py` (the grid), `scripts/realuse_tables.py` (the tables). Results `results/realuse.json`; tracker experiment `realuse`. No training.*

Every merchant result so far was on 120 opaque names with one clean bank string each and balanced categories, and every prototype result on 3-way balanced name-only trials; the reviewer's REAL-1 and REAL-4 said so. This step builds the history a categoriser actually sees and re-runs the prototype family on it, without training anything, so the numbers are what a frozen encoder gives a new user on day one. The history (`transactions.build_merchants` / `build_transactions`) has 240 merchants: 120 real chains a pretrained model has read about (Kroger, Starbucks, Shell, Home Depot, CVS, Delta, Petco, Verizon and so on, ten per section 4 category) and the 120 opaque merchants of section 4 (nobody has read about them), shuffled into one Zipf rank order (exponent 1.1) so that real and opaque merchants share the head and the tail; 3,000 transactions drawn by frequency, each a card-statement string from the section 4 test templates or the step 36 rendering templates (prefixes, truncations, abbreviations, store numbers, cities, dates), with a log-normal amount per category and a weekday. The result is what a real statement looks like: 24 head merchants make 2,048 of the 3,000 transactions, the tail of 144 merchants makes 343, the categories run from 784 transactions (Travel, because an airline landed at rank 1) to 80 (Groceries), and 19 merchants never appear at all.

The protocol is REAL-4's: a trial draws k labelled transactions per category (k = 1, 3, 10; a category with fewer gives what it has), and the classifier labels the other transactions 12-way; ten trials per cell. Four classifiers on unit-normalised embeddings of the string, raw or through the step 36 normaliser: the k-example centroid of section 6.3 and 24 (`proto`); the same with the amount band (seven bands) and weekday appended at weight 0.5 (`text+af`); label propagation over a 10-nearest-neighbour graph of all 3,000 strings, labelled and not (Zhou et al., alpha 0.9; GRAPH-2); and, for GRAPH-6, the category *name's* embedding as the class vector (`name`, which needs no labelled example) and its unit mean with the centroid (`mix`). Accuracy is broken down by merchant frequency bucket (head = the top 10% of merchants by count, torso = the next 30%, tail = the rest), by real against opaque merchant, and by whether the test transaction's merchant appears among the k labelled examples (seen) or not (unseen). Two frozen encoders: all-MiniLM-L6-v2 (22M) and bge-base-en-v1.5 (109M).

**Table 36.1: 12-way category accuracy on the 3,000-transaction history (frozen encoders; 10 trials; chance 8.3) by classifier and labelled transactions per category k; name = the category name's embedding alone, mix = its unit mean with the k-example centroid, lp = label propagation over the kNN graph of all 3,000 strings**

| encoder | strings | k | name (k=0 vector) | prototype | mix | label propagation |
|---|---|---|---|---|---|---|
| MiniLM | raw | 1 | 25.9 | 31 | 36.9 | 15.3 |
| MiniLM | raw | 3 | 25.9 | 40.5 | 45.8 | 23 |
| MiniLM | raw | 10 | 25.9 | 55.2 | 55.8 | 39.4 |
| MiniLM | norm | 1 | 38.9 | 41.7 | 48.3 | 21.2 |
| MiniLM | norm | 3 | 38.9 | 53 | 57.2 | 39.1 |
| MiniLM | norm | 10 | 38.9 | 68.8 | 70.4 | 67.1 |
| bge-base | raw | 1 | 27 | 32.4 | 38.2 | 18 |
| bge-base | raw | 3 | 27 | 41 | 48.4 | 24.5 |
| bge-base | raw | 10 | 27 | 56.3 | 57.2 | 41.2 |
| bge-base | norm | 1 | 47.5 | 43 | 51.8 | 22 |
| bge-base | norm | 3 | 47.5 | 53 | 61.7 | 41.6 |
| bge-base | norm | 10 | 47.5 | 67.4 | 70.4 | 68 |

**Table 36.2: the prototype and label-propagation classifiers by merchant frequency bucket (head = top 10% of merchants by count, torso = next 30%, tail = rest), known (real chain) vs opaque merchant, whether the test transaction's merchant is among the labelled examples, and the unseen merchants split by known vs opaque (accuracy %, normalised strings)**

| encoder | k | method | all | head | torso | tail | known | opaque | seen merchant | unseen merchant | unseen, known chain | unseen, opaque |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MiniLM | 1 | proto | 41.7 | 54.5 | 14.6 | 12.9 | 52.7 | 12.3 | 84.9 | 14.1 | 20.6 | 5.6 |
| MiniLM | 1 | lp | 21.2 | 25.4 | 11.9 | 12.3 | 24.7 | 11.8 | 44.3 | 6.7 | 7.1 | 6.1 |
| MiniLM | 3 | proto | 53 | 66.6 | 27.6 | 16.2 | 61.1 | 31.3 | 84.2 | 17.6 | 24.5 | 8.3 |
| MiniLM | 3 | lp | 39.1 | 47.4 | 23.1 | 16.8 | 43.5 | 27.2 | 67.6 | 7.4 | 8.5 | 6.1 |
| MiniLM | 10 | proto | 68.8 | 84.8 | 38 | 24 | 80 | 38.3 | 86.9 | 16.4 | 24.5 | 7.5 |
| MiniLM | 10 | lp | 67.1 | 80.2 | 42.9 | 28.2 | 75 | 45.7 | 86.4 | 11 | 16.3 | 5.6 |
| bge-base | 1 | proto | 43 | 55.5 | 16.9 | 14.5 | 54.1 | 13.2 | 89.6 | 13.2 | 18 | 6.7 |
| bge-base | 1 | lp | 22 | 25.9 | 13.2 | 14.1 | 25.5 | 12.6 | 45 | 7.5 | 8.2 | 6.5 |
| bge-base | 3 | proto | 53 | 65.7 | 29.1 | 19.2 | 61.5 | 30.3 | 87.1 | 14 | 19.3 | 6.7 |
| bge-base | 3 | lp | 41.6 | 49.6 | 25.4 | 22.2 | 46.6 | 28.1 | 68.3 | 11.6 | 14.4 | 7.7 |
| bge-base | 10 | proto | 67.4 | 81.6 | 40.8 | 26.6 | 78.9 | 36.3 | 85.3 | 15.8 | 24.8 | 6.3 |
| bge-base | 10 | lp | 68 | 80.5 | 45.2 | 31.2 | 75.9 | 46.6 | 86.7 | 14.1 | 20.6 | 7.4 |

**Table 36.3: amount band and weekday appended to the text embedding (weight 0.5), normalised strings (accuracy %)**

| encoder | k | prototype, text | prototype, text + amount + weekday | lp, text | lp, text + amount + weekday |
|---|---|---|---|---|---|
| MiniLM | 1 | 41.7 | 41.3 | 21.2 | 31 |
| MiniLM | 3 | 53 | 49.9 | 39.1 | 44.2 |
| MiniLM | 10 | 68.8 | 65.2 | 67.1 | 66.2 |
| bge-base | 1 | 43 | 40.3 | 22 | 31 |
| bge-base | 3 | 53 | 47.3 | 41.6 | 43.6 |
| bge-base | 10 | 67.4 | 60.1 | 68 | 60.4 |

MiniLM: 0.7 minutes for the grid.

bge-base: 0.7 minutes for the grid.

### 36.1 What a frozen encoder gives a new user: the prototype is a merchant memory, and the normaliser is worth ten points at every k

Pooled over the 3,000 transactions the numbers look like a working classifier: with ten labelled transactions per category the prototype reads 68.8 (MiniLM) and 67.4 (bge-base) 12-way, with three 53.0 on both, with one 41.7 and 43.0, all on normalised strings; the raw strings cost ten to thirteen points at every k (31.0 / 40.5 / 55.2 for MiniLM), so the step 36 normaliser is the first thing to apply and the largest single gain on this history too. The breakdown says what the pooled number is made of. On transactions whose merchant appears among the k labelled examples the prototype reads 84 to 90 at every k for both encoders; on transactions whose merchant does not, 13 to 18 (chance 8.3). The frequency buckets are the same fact seen through the Zipf distribution: the 24 head merchants, which make two thirds of the transactions and are nearly always among the labelled examples, read 55 / 67 / 85 at k = 1 / 3 / 10, the torso 15 / 28 / 38, the tail 13 / 16 / 24. A prototype built from a user's labelled transactions is a memory of the merchants in those transactions: it labels a new transaction from a labelled merchant almost perfectly, and it barely labels anything else. The k axis is the axis of how many merchants have been shown (with ten labels per category about 90 distinct merchants are seen and the head is covered), not of how well the classifier generalises.

### 36.2 The unseen merchant: the category's name reaches real chains through the encoder's own knowledge and nothing reaches an opaque one (GRAPH-6, REAL-1)

GRAPH-6 asked whether a category with zero examples can be classified from its name. It can, on the merchants the encoder has read about: the embedding of the bare category name (`Groceries`, `Gas & Auto`) as the class vector, with no labelled transaction at all, reads 38.9 (MiniLM) and 47.5 (bge-base) over the whole history, which beats the one-example prototype on bge-base (43.0) and is four points under it on MiniLM. On the unseen real chains, where the prototype reads 18 to 25, the name vector reads 38.3 to 42.4 (MiniLM) and 52.0 to 56.2 (bge-base): the encoder knows that `Kroger` is groceries and `Petco` is pets from pretraining, and the category name is the query that reaches that knowledge, where a centroid of other merchants' strings does not. The mix (the unit mean of the name vector and the k-example centroid) is the best classifier in the table at every k for both encoders: 48.3 / 57.2 / 70.4 on MiniLM and 51.8 / 61.7 / 70.4 on bge-base, six to nine points over the prototype at k = 1 and 3 and two to three at k = 10, where the centroid has the head covered and the name adds only its reach into the unseen. On the opaque merchants that no example covers, every method is at chance: prototype 5.6 to 8.3, label propagation 5.6 to 7.7, name vector 4.9 to 11.6, mix 5.3 to 7.3. REAL-1's known/unknown split is therefore not a nuance but the whole result: a real chain the model has read about is categorisable from the category's name alone at 40 to 56 with no example, an unknown merchant is categorisable only after an example of it is labelled, and no method in this table changes either statement.

### 36.3 Label propagation and the amount and weekday features (GRAPH-2, REAL-4)

GRAPH-2 expected transductive label propagation over the kNN graph of all 3,000 strings to beat the prototype when labels are few and to tie it at k = 5 or so, as it does on the few-shot image benchmarks. Here it is the reverse. With one or three labels per category label propagation reads 21 to 22 and 39 to 42 against the prototype's 42 to 43 and 53; at ten labels the two tie (67 to 68). The graph is real (repeated renderings of one merchant are near neighbours) but so is the Zipf distribution: with twelve labelled nodes in a 3,000-node graph, two thirds of whose nodes belong to 24 merchants, the propagated mass follows the head merchants' clusters and the small categories drown (the seen-merchant accuracy of label propagation at k = 1 is 44 against the prototype's 85), and it recovers only when enough labels sit inside each cluster. Label propagation does buy something the prototype does not, once it has labels: at k = 10 it reads 46 against 38 on the opaque merchants and 28 to 31 against 24 to 27 on the tail, because a labelled transaction of an opaque merchant propagates to that merchant's other renderings through the graph while the centroid dilutes it among the category's other merchants. That is the graph's real use on transaction data, and it is a k = 10 use.

REAL-4's second half was the transaction features. Appending the amount band (seven one-hot bands) and the weekday (seven) to the text embedding at weight 0.5 costs the prototype 0.4 to 7.3 points at every k on both encoders (MiniLM 41.7 / 53.0 / 68.8 to 41.3 / 49.9 / 65.2), and helps label propagation only at k = 1 and 3 (21 to 31, 39 to 44 on MiniLM), where the text graph was too sparse to propagate and any feature that groups transactions helps; at k = 10 it costs label propagation too (67.1 to 66.2 on MiniLM, 68.0 to 60.4 on bge-base). The categories' amount distributions overlap by construction (a log-normal per category with means from 18 to 270 dollars), and weekday carries nothing in this history, so the features add a noisy dimension to a centroid that was already a merchant memory. A learned weighting or a per-user feature selection might do better; concatenation at a fixed weight does not.

### 36.4 What the step says

The four questions of the row, in the owner's terms. REAL-1: on a realistic history the prototype family is a merchant memory: 85 to 90 on transactions of merchants the user has labelled, 14 to 18 on the rest, and the pooled 69 at ten labels per category is that mixture weighted by the Zipf head. The long tail is the unseen tail, and for it the encoder's pretrained knowledge of real chains is the only lever without an example: a category name reaches it at 38 to 56, an opaque merchant stays at chance. GRAPH-6: yes, a zero-example category is classifiable from its name, and mixing the name vector into the centroid is the best classifier at every k. GRAPH-2: label propagation is not the few-label method here; it loses to the prototype below ten labels and pays off only on opaque and tail merchants once their renderings carry a label. REAL-4: 12-way, imbalanced, on strings, the prototype protocol of sections 6.3 and 24 gives numbers a third to a half of the balanced 3-way ones, and amount and weekday features as fixed-weight one-hots hurt. For the categoriser: normalise the string, use the category name as part of every class vector so that new categories and real chains are covered from day one, treat the labelled history as the merchant memory it is, and expect nothing for a merchant the user has never labelled unless it is a chain the encoder knows or a fact database (rows 33 and 35) supplies it. Cost: 1.3 minutes of GPU for the whole grid.

Not run: a fine-tuned encoder on the user's own history (row 33), a learned feature weighting, the cluster-tightness attribute selector of REAL-4 (which needs the multi-attribute universe items, not this history), and real transaction exports.

## 37. REAL-6: the production-shaped evaluation set, and what the untrained models score on it: a 24-shot 3B model reads a quarter to a third of a user's items, the encoder prototype over the full history reads 55 to 60 of the labelled merchants and nothing of the rest, and the fact-DB record in the prompt lifts the instruct model to 58 with the unlabelled opaque merchants at 52 (REAL-6)

*PLAN step 35. Code: `ai_experiments.real6` (the set; frozen as `data/processed/real6_v1.json`, sha256 over the items), `scripts/exp_real6.py` (scoring: LLM option log-probability with and without the fact-DB record, encoder prototypes; bootstrap intervals and null bands per cell), `scripts/real6_tables.py`. Results `results/real6_<model>.json`, per-item records under `results/per_item/real6_*.jsonl`; tracker experiment `real6`. No training.*

The owner's goal (QUESTIONS.md REAL-5, REAL-6) is a categoriser that labels a user's bank transactions the way that user labelled similar ones before, on the user's own category names, including names the model has never seen and merchants the user has never labelled but a fact database knows. No evaluation in the report had that shape: the ladder has one universe and fixed labels, the merchant set has twelve fixed categories, and section 36's history has one user with the standard scheme. This step builds the yardstick and scores the untrained models on it, so that row 33's trained categorisers have something to beat.

The set (`real6.build`) has 20 synthetic users over the frozen transaction history of section 36 (240 merchants, real chains and opaque ones, Zipf frequencies, noisy strings). Each user has a scheme of 8 to 20 categories made from the 12 standard ones by merging (a user with 8 categories has folded some together, "Apparel & Eating out & Car") or splitting (a user with 20 has split "Restaurants" into "Coffee and snacks" and "Sit-down meals", with each merchant landing on one side arbitrarily, so the split is learnable only from that user's history), and each category carries one of three name types: the standard name ("Groceries"), a rename ("Food shopping", "Bills", "Gym") or a coined word with no meaning ("Zorbit", "Plenk"). Each user has a labelled history of 300 transactions (string, amount, weekday, the user's label) over about two thirds of their merchants, and test transactions in six cells: the merchant appears in the history (seen) or does not (unseen), crossed with the name type of its category; 1,179 items in all (212 / 238 / 109 seen and 240 / 240 / 140 unseen for standard / renamed / new). Every merchant has a fact-DB record (section 4's sentence for the opaque ones, a category-pool sentence for the real chains), so the unseen cells are the "merchant in the DB but not in the user's history" case of the owner's goal.

Two scoring routes, both without training. The LLM route is the ladder's option scoring: a 24-shot prompt of the user's own history (the categories listed, then 24 labelled transactions stratified over the categories, then the query), the user's category names as options, mean log-probability per token; and a second condition with the merchant's fact-DB record inserted before the query (`prompt_ctx`), the retrieval oracle that row 33 will try to earn. The encoder route is section 36's prototype: the class vector is the mean embedding of the user's normalised history strings per category, from the same 24 shots the LLM sees or from the full 300, and the name-plus-centroid mix. Every cell carries a 95% bootstrap interval and the section 11 null band (gold permuted within the cell, predictions fixed); chance is the mean of one over the user's category count, about 7.

**Table 37.1: the REAL-6 evaluation set (20 synthetic users, 8 to 20 categories each, 24-shot prompts of the user's own history) by cell: accuracy % [95% bootstrap interval], * = inside the null band; chance is the mean of 1/(categories per user), about 7**

| cell | items | Qwen2.5-3B base, 24-shot | + fact-DB record | Qwen2.5-3B-Instruct, 24-shot | Instruct + record | MiniLM prototype, 24 shots | MiniLM prototype, full history | MiniLM mix (name + history) | bge-base prototype, full history | bge-base mix |
|---|---|---|---|---|---|---|---|---|---|---|
| seen merchant, standard name | 212 | 34 [27.8, 40.1] | 67.9 [61.3, 74.1] | 43.4 [36.8, 50] | 78.3 [72.6, 84] | 30.2 [24.5, 36.3] | 53.3 [46.2, 59.9] | 58 [51.4, 64.6] | 60.4 [54.2, 67] | 64.2 [57.5, 70.3] |
| seen merchant, renamed | 238 | 30.7 [24.8, 36.1] | 50 [43.3, 56.3] | 34.5 [28.6, 40.3] | 54.6 [48.7, 60.9] | 31.9 [26.1, 37.4] | 56.3 [50, 62.2] | 51.3 [45.4, 58] | 58.8 [52.1, 64.7] | 56.7 [50.4, 63] |
| seen merchant, new word | 109 | 26.6 [19.3, 34.9] | 35.8 [27.5, 45] | 25.7 [17.4, 33.9] | 45 [35.8, 54.1] | 36.7 [28.4, 45] | 58.7 [49.5, 67.9] | 52.3 [43.1, 61.5] | 60.6 [52.3, 69.7] | 51.4 [41.3, 60.6] |
| unseen merchant, standard name | 240 | 25 [19.6, 30.4] | 55 [48.8, 61.3] | 33.8 [27.9, 39.6] | 70.4 [64.2, 76.7] | 15.8 [11.2, 20.4] | 12.5 [8.3, 16.7] | 27.9 [22.5, 33.3] | 20 [15.4, 24.6] | 36.7 [30.8, 42.5] |
| unseen merchant, renamed | 240 | 20.4 [15.4, 25.8] | 43.8 [37.5, 50] | 29.2 [23.8, 35] | 51.2 [44.6, 57.5] | 12.5 [8.8, 17.1] | 13.3 [8.8, 17.9] | 17.9 [12.9, 22.9] | 12.1 [8.3, 15.8] | 20 [14.6, 25] |
| unseen merchant, new word | 140 | 11.4 [6.4, 16.4] | 24.3 [17.9, 31.4] | 10.7 [5.7, 16.4] | 33.6 [25.7, 41.4] | 15 [9.3, 20.7] | 19.3 [12.9, 25.7] | 24.3 [17.9, 31.4] | 13.6 [8.6, 19.3] | 16.4 [10, 22.9] |
| seen merchant, all | 559 | 31.1 [27.5, 35.2] | 54 [49.7, 58.1] | 36.1 [32.2, 40.3] | 61.7 [57.6, 65.8] | 32.2 [28.3, 36.3] | 55.6 [51.9, 59.6] | 54 [50.3, 58.1] | 59.7 [55.6, 63.7] | 58.5 [54.4, 62.6] |
| unseen merchant, all | 620 | 20.2 [17.1, 23.2] | 43.7 [39.8, 47.7] | 26.8 [23.4, 30.3] | 54.7 [50.8, 58.5] | 14.4 [11.6, 17.3] | 14.4 [11.5, 17.1] | 23.2 [20, 26.6] | 15.5 [12.6, 18.5] | 25.6 [22.3, 29] |
| standard names, all | 452 | 29.2 [25, 33.2] | 61.1 [56.4, 65.5] | 38.3 [33.4, 42.7] | 74.1 [69.9, 78.1] | 22.6 [18.8, 26.1] | 31.6 [27.2, 35.8] | 42 [37.4, 46.7] | 38.9 [34.1, 43.1] | 49.6 [44.7, 54.2] |
| renamed, all | 478 | 25.5 [21.8, 29.5] | 46.9 [42.1, 51.3] | 31.8 [27.6, 36] | 52.9 [48.3, 57.7] | 22.2 [18.8, 25.7] | 34.7 [30.8, 39.1] | 34.5 [30.5, 38.9] | 35.4 [31.4, 39.7] | 38.3 [34.1, 43.3] |
| new words, all | 249 | 18.1 [13.7, 22.9] | 29.3 [24.1, 34.9] | 17.3 [12.9, 22.1] | 38.6 [32.1, 45] | 24.5 [19.3, 30.1] | 36.5 [30.9, 42.2] | 36.5 [30.5, 42.6] | 34.1 [28.9, 40.2] | 31.7 [26.5, 37.8] |
| all items | 1179 | 25.4 [23, 28.1] | 48.6 [45.7, 51.6] | 31.2 [28.7, 33.8] | 58 [55.1, 60.8] | 22.8 [20.4, 25.2] | 33.9 [31.3, 36.5] | 37.8 [34.9, 40.6] | 36.5 [33.6, 39] | 41.2 [38.3, 43.9] |

**Table 37.2: the same runs by whether the merchant is among the 24 prompt shots, elsewhere in the 300-row history, or absent from it (accuracy %; the encoder 'full history' columns use all 300 rows, so their first two groups differ only in the merchant's frequency)**

| group | items | Qwen2.5-3B base, 24-shot | + fact-DB record | Qwen2.5-3B-Instruct, 24-shot | Instruct + record | MiniLM prototype, 24 shots | MiniLM prototype, full history | MiniLM mix (name + history) | bge-base prototype, full history | bge-base mix |
|---|---|---|---|---|---|---|---|---|---|---|
| in the 24 shots | 160 | 58.8 | 71.9 | 60.6 | 78.1 | 88.1 | 87.5 | 85.0 | 86.2 | 85.6 |
| in the history, not the shots | 399 | 20.1 | 46.9 | 26.3 | 55.1 | 9.8 | 42.9 | 41.6 | 49.1 | 47.6 |
| not in the history | 620 | 20.2 | 43.7 | 26.8 | 54.7 | 14.4 | 14.4 | 23.2 | 15.5 | 25.6 |
| known chain, not in the history | 366 | 30.1 | 43.2 | 41.3 | 56.6 | 17.5 | 17.2 | 29.0 | 21.0 | 36.6 |
| opaque, not in the history | 254 | 5.9 | 44.5 | 5.9 | 52.0 | 9.8 | 10.2 | 15.0 | 7.5 | 9.8 |

### 37.1 The encoder prototypes: a merchant memory again, with the full history worth 25 points over 24 shots, and the category name reaching unseen merchants only when the name means something

With the same 24 labelled transactions the LLM sees, the MiniLM prototype reads 32.2 on seen merchants and 14.4 on unseen ones (chance 7.3); with the full 300-row history the seen cells rise to 53 to 59 (MiniLM) and 59 to 61 (bge-base) and the unseen cells stay at 12 to 20. The seen numbers are well under section 36's 85 to 90 for the same encoders, and the difference is the scheme: a user's categories merge and split the standard ones arbitrarily, so a centroid over "Apparel & Eating out & Car" averages three kinds of merchant and a "Coffee and snacks" centroid is one arbitrary half of the restaurants, and the noisy strings of a merchant's other renderings have to land nearer that centroid than the neighbouring one's. The three name types make no difference to the prototype (seen: 53 / 56 / 59 standard / renamed / new on MiniLM; 60 / 59 / 61 on bge-base), as they should not: a centroid never reads the name. Mixing the name vector into the centroid (section 36's best classifier) helps exactly where the name means something: unseen merchants with a standard name go from 12.5 to 27.9 (MiniLM) and 20.0 to 36.7 (bge-base), the renamed ones move 13 to 18 and 12 to 20, the coined words not at all (19 to 24 and 14 to 16, inside the noise), and the seen standard cell gains five points while the seen renamed and coined cells lose five to nine, because a name vector that means nothing (or means the standard category when the user's category is a split of it) pulls the centroid the wrong way. On this set the mix is a gain for standard-name categories and a cost for the others, and a production classifier would apply it per category by name type. Over everything, bge-base's mix reads 41.2 and its plain full-history prototype 36.5; MiniLM 37.8 and 33.9.

### 37.2 The language model: 24 shots of the user's history buy a quarter of the items, the fact-DB record buys half, and the coined names are where both routes are weakest

Qwen2.5-3B, untrained, with the categories listed and 24 labelled transactions of the user in the prompt, reads 25.4 over the 1,179 items (chance 7.3): 31.1 on seen merchants and 20.2 on unseen, 34 / 31 / 27 on seen merchants with standard / renamed / coined names and 25 / 20 / 11 on unseen ones. Table 37.2 says what "seen" means to a 24-shot prompt: when the query's merchant is among the 24 shots the model copies its label at 58.8 (the MiniLM prototype on the same 24 rows reads 88.1: a centroid matches a noisy string to its sibling rendering better than a 3B decoder reads a label off a demonstration), and when the merchant is elsewhere in the 300-row history the model is at 20.1, the same as for a merchant it has never seen, because it never saw those rows. On the unseen merchants the model's pretraining shows: real chains 30.1 against opaque 5.9 (chance), the same 30 that the encoder's category-name mix reaches by the same knowledge.

The fact-DB record before the query changes the picture more than anything in the row: 48.6 over all items, 54.0 on seen merchants, 43.7 on unseen ones, and on the opaque merchants the user never labelled, where nothing else in this report scores, 44.5. The record is a products sentence ("Elrholm is a store that sells canned goods, packaged snacks and frozen vegetables"), and the model bridges from products to the user's category when the category's name says what it contains (unseen merchants: standard 55.0, renamed 43.8) and much less when it does not (coined words 24.3; seen coined 35.8): the record can tell the model what the store sells, but which of the user's arbitrary words that maps to is only in the history, and a 24-shot prompt carries a fraction of it. Section 35's bridge result is visible here too: with a record in the prompt, the products-to-category step is the residual error. Qwen2.5-3B-Instruct, the model a production system would start from, is six points better than the base at 24 shots (31.2; seen 36.1, unseen 26.8, real chains it never saw 41.3 against the base's 30.1) and nine better with the record (58.0; seen 61.7, unseen 54.7, opaque unseen 52.0, unseen standard-name 70.4), the best column in the table and the only one above the full-history prototypes on the seen merchants. Its weakest cells are the same as everyone's: the coined names, 17.3 without the record and 38.6 with it.

### 37.3 What the step says

REAL-6 is the yardstick the owner's goal needs, and its first reading fixes the targets for row 33. Untrained, the best route on a user's own scheme is the encoder prototype over the full history for the merchants the user has labelled (53 to 61, against the 3B model's 31 with 24 shots), and the fact-DB record in the prompt for the merchants they have not (the 3B model at 43.7 unseen and 44.5 on opaque unseen, where every prototype is near chance): the two halves of the owner's goal are currently served by two different routes, and neither is above 50 on the coined-name categories, which only the user's history defines. Row 33's categoriser has to beat 55.6 / 59.7 on seen merchants (the full-history prototypes), 43.7 on unseen ones (the record oracle), and it has to do so on the renamed and coined names, where the standard-name shortcuts of both routes stop working. The instruct model with the record is the number to beat overall (58.0), the full-history prototypes on labelled merchants (55.6 / 59.7), and on the coined-name categories every untrained route is under 40. The set is frozen with its hash; every item carries the record, the user, the merchant, the name type and both prompts, so the row 33 arms (label SFT, contrastive prototypes, trace distillation, with and without the injected DB) score the same 1,179 items and pair against these columns. Cost of this step: about 1.5 GPU hours of scoring, no training.

Not done: real transaction exports (the owner's internal data would replace `transactions_v1` behind the same interface), more than 20 users, and the section 11 paired tests between columns (the per-item files allow them; the table reports intervals and null bands).

## 38. The production categoriser on REAL-6: label SFT lifts the 3B model to 57 and the tuned encoder to 76, other users' labels carry an unseen merchant to 80 to 86, the records injected into the weights reach the merchants only the DB knows once the exposure is section 8's (51 to 72, opaque 25 to 54) and not before, and the record in the prompt reaches them at 98, coined category names included (REAL-5)

*PLAN step 33. Code: `scripts/exp_categoriser.py` (training: label SFT LoRA on Qwen2.5-3B-Instruct; contrastive bge-base; each crossed with the fact DB as none / parametric / record in the prompt), `scripts/exp_real6.py` (scoring, with `CONDS`, fine-tuned encoder directories and `ENC_CTX`), `scripts/exp_items_v2.py` (ARC-Easy and MMLU of the adapters), `scripts/categoriser_tables.py`. Adapters `models/adapters/categoriser_*`; results `results/real6_categoriser_*.json`, `results/categoriser_*.json`, `results/items2_categoriser_*.json`; tracker experiments `categoriser`, `real6`, `items_v2`.*

Section 37 built the yardstick and read the untrained models on it; this step trains the categoriser the owner asked for and asks REAL-5's question: does an injected fact database improve it on the merchants the user never labelled, and does the gain survive the user's own category names? The scope is the two training routes that the report has evidence for and the three ways of giving them the database; the third training route of the question (teacher-trace distillation with the records as a tool, with an optional GRPO stage) is not run, for the reason section 31 measured: sampling from a 3B model with Hugging Face generation costs hours per run on this GPU, and the step is sized for one session.

**Training.** The LLM categoriser is a rank-64 adapter on Qwen2.5-3B-Instruct trained by label SFT in exactly the REAL-6 prompt format: for 150 rows of each user's 300-row history, the user's categories listed, 24 other rows of that history as shots, the row as the query and the user's label as the answer, loss on the label tokens only; 3,000 examples, 200 steps of 16 sequences at 1e-4 (about 1 epoch), one adapter for all 20 users, so the model learns to read a user's scheme and history from the prompt rather than one user's labels. The frozen test transactions are never queries in training (the seen cells' merchants are, as they should be: that is the user's history). The encoder categoriser is bge-base fine-tuned contrastively across users on (normalised history string, the user's category name) pairs, 6,000 pairs, three epochs, then scored as section 37's prototypes over each user's full history. One control was added after the first attempt showed why it is needed: a categoriser trained on 20 users' histories learns a merchant's category from *other* users' labels, so a merchant "unseen" by one user is usually seen in training, and the first tuned encoder read 81.5 on the unseen cells before any database was involved. A stratified quarter of the merchants (60, `real6.db_only_merchants`) is therefore held out of every training row, as query and as shot, for both routes; the frozen set is unchanged, and the unseen cells split in the tables into merchants other users labelled in training and merchants only the fact DB knows (123 of the 620 unseen items, 71 of them real chains), which is REAL-5's number. The database enters three ways: not at all; parametrically, as section 8's recipe on the 240 records (each record and three paraphrases, 960 texts, 30% of the LLM's training sequences with full-sequence loss; for the encoder, (record, standard category) pairs); or at inference, as the merchant's record before the query (the LLM trained and scored with it; the encoder with the record appended to the query string). Because 200 steps at 30% is about one pass over the 960 record texts, far below section 8's exposure (4.6 passes), the parametric arm was also run at 400 steps with half the sequences drawn from the records (about 3.3 passes).

**Table 38.1: the trained categorisers on the REAL-6 cells beside the untrained columns of section 37 (accuracy % [95% bootstrap interval], * = inside the null band; chance about 7)**

| cell | Instruct, 24-shot (sec. 37) | Instruct + record (sec. 37) | SFT, no DB | SFT + parametric DB | SFT + parametric DB, 400 steps at 50% | SFT + record in prompt | bge frozen, full history (sec. 37) | bge frozen, mix (sec. 37) | bge tuned, no DB | bge tuned, mix | bge tuned + record on query | bge tuned + parametric DB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seen merchant, standard name | 43.4 [36.8, 50] | 78.3 [72.6, 84] | 65.1 [58.5, 71.2] | 67.9 [61.3, 74.1] | 75 [68.9, 80.7] | 99.5 [98.6, 100] | 60.4 [54.2, 67] | 64.2 [57.5, 70.3] | 83 [77.8, 87.7] | 81.6 [76.4, 86.8] | 90.1 [86.3, 93.4] | 83.5 [78.3, 87.7] |
| seen merchant, renamed | 34.5 [28.6, 40.3] | 54.6 [48.7, 60.9] | 58 [52.5, 63.9] | 56.7 [50.8, 63] | 60.5 [54.2, 66.4] | 85.7 [81.1, 89.9] | 58.8 [52.1, 64.7] | 56.7 [50.4, 63] | 73.9 [68.5, 79] | 72.3 [66.8, 77.7] | 83.2 [78.6, 87.4] | 72.7 [67.2, 78.6] |
| seen merchant, new word | 25.7 [17.4, 33.9] | 45 [35.8, 54.1] | 56 [46.8, 65.1] | 55 [45, 64.2] | 64.2 [55, 73.4] | 90.8 [84.4, 96.3] | 60.6 [52.3, 69.7] | 51.4 [41.3, 60.6] | 88.1 [81.7, 93.6] | 83.5 [76.1, 89.9] | 91.7 [86.2, 96.3] | 83.5 [75.2, 90.8] |
| unseen merchant, standard name | 33.8 [27.9, 39.6] | 70.4 [64.2, 76.7] | 67.9 [61.7, 73.3] | 57.1 [50.8, 62.9] | 75.8 [70, 81.2] | 100 [100, 100] | 20 [15.4, 24.6] | 36.7 [30.8, 42.5] | 80 [74.6, 85.4] | 82.1 [77.1, 86.7] | 89.6 [85.4, 93.3] | 79.2 [73.3, 84.2] |
| unseen merchant, renamed | 29.2 [23.8, 35] | 51.2 [44.6, 57.5] | 47.1 [41.2, 53.8] | 45.8 [39.6, 52.1] | 59.6 [53.3, 65.4] | 75 [69.6, 80.4] | 12.1 [8.3, 15.8] | 20 [14.6, 25] | 60 [53.8, 66.7] | 56.2 [49.6, 62.1] | 66.7 [60.8, 72.5] | 52.1 [45.4, 58.3] |
| unseen merchant, new word | 10.7 [5.7, 16.4] | 33.6 [25.7, 41.4] | 42.9 [35, 51.4] | 40 [32.1, 49.3] | 60 [52.1, 68.6] | 92.1 [87.9, 96.4] | 13.6 [8.6, 19.3] | 16.4 [10, 22.9] | 77.9 [70.7, 84.3] | 77.9 [70.7, 84.3] | 87.9 [82.1, 92.9] | 77.1 [70, 84.3] |
| seen merchant, all | 36.1 [32.2, 40.3] | 61.7 [57.6, 65.8] | 60.3 [56.2, 64.4] | 60.6 [56.7, 64.9] | 66.7 [62.6, 70.7] | 91.9 [89.6, 94.1] | 59.7 [55.6, 63.7] | 58.5 [54.4, 62.6] | 80.1 [76.9, 83.2] | 78 [74.4, 81.6] | 87.5 [84.8, 90.5] | 78.9 [75.7, 82.1] |
| unseen merchant, all | 26.8 [23.4, 30.3] | 54.7 [50.8, 58.5] | 54.2 [50.3, 58.1] | 48.9 [44.7, 52.9] | 66 [62.3, 70] | 88.5 [86, 91.1] | 15.5 [12.6, 18.5] | 25.6 [22.3, 29] | 71.8 [68.1, 74.8] | 71.1 [67.6, 74.5] | 80.3 [77.3, 83.4] | 68.2 [64.7, 71.8] |
| standard names, all | 38.3 [33.4, 42.7] | 74.1 [69.9, 78.1] | 66.6 [61.9, 70.8] | 62.2 [57.7, 66.8] | 75.4 [71.7, 79.6] | 99.8 [99.3, 100] | 38.9 [34.1, 43.1] | 49.6 [44.7, 54.2] | 81.4 [77.9, 85] | 81.9 [78.3, 85.2] | 89.8 [86.9, 92.5] | 81.2 [77.7, 84.7] |
| renamed, all | 31.8 [27.6, 36] | 52.9 [48.3, 57.7] | 52.5 [47.9, 56.9] | 51.3 [46.7, 55.6] | 60 [55.6, 64.2] | 80.3 [76.6, 83.9] | 35.4 [31.4, 39.7] | 38.3 [34.1, 43.3] | 66.9 [62.6, 71.3] | 64.2 [59.8, 68.6] | 74.9 [71.3, 78.7] | 62.3 [57.9, 66.7] |
| new words, all | 17.3 [12.9, 22.1] | 38.6 [32.1, 45] | 48.6 [42.6, 54.6] | 46.6 [40.2, 52.6] | 61.8 [55.8, 67.9] | 91.6 [88, 94.8] | 34.1 [28.9, 40.2] | 31.7 [26.5, 37.8] | 82.3 [77.5, 86.7] | 80.3 [75.5, 84.7] | 89.6 [85.9, 93.2] | 79.9 [74.7, 84.7] |
| all items | 31.2 [28.7, 33.8] | 58 [55.1, 60.8] | 57.1 [54.3, 59.9] | 54.5 [51.7, 57.3] | 66.3 [63.9, 69] | 90.2 [88.4, 91.7] | 36.5 [33.6, 39] | 41.2 [38.3, 43.9] | 75.7 [73.2, 77.9] | 74.4 [71.9, 76.8] | 83.7 [81.6, 85.8] | 73.3 [70.8, 75.6] |

**Table 38.2: by whether the merchant is among the 24 prompt shots, elsewhere in the 300-row history, or absent from it, and for the absent ones whether another user's training rows carried it or only the fact DB knows it (accuracy %; the untrained section 37 columns did not train, so their split is a merchant subset only)**

| group | Instruct, 24-shot (sec. 37) | Instruct + record (sec. 37) | SFT, no DB | SFT + parametric DB | SFT + parametric DB, 400 steps at 50% | SFT + record in prompt | bge frozen, full history (sec. 37) | bge frozen, mix (sec. 37) | bge tuned, no DB | bge tuned, mix | bge tuned + record on query | bge tuned + parametric DB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| in the 24 shots | 60.6 | 78.1 | 96.2 | 96.9 | 88.8 | 96.2 | 86.2 | 85.6 | 96.2 | 95.0 | 97.5 | 95.6 |
| in the history, not the shots | 26.3 | 55.1 | 45.9 | 46.1 | 57.9 | 90.2 | 49.1 | 47.6 | 73.7 | 71.2 | 83.5 | 72.2 |
| not in the history | 26.8 | 54.7 | 54.2 | 48.9 | 66.0 | 88.5 | 15.5 | 25.6 | 71.8 | 71.1 | 80.3 | 68.2 |
| not in the history, labelled by other users in training | 26.0 | 56.1 | 54.9 | 47.7 | 64.4 | 86.3 | 15.1 | 25.2 | 80.1 | 80.1 | 82.7 | 75.5 |
| not in the history, DB-only (no user labelled it in training) | 30.1 | 48.8 | 51.2 | 53.7 | 72.4 | 97.6 | 17.1 | 27.6 | 38.2 | 35.0 | 70.7 | 39.0 |
| DB-only, known chain | 49.3 | 52.1 | 70.4 | 83.1 | 85.9 | 100.0 | 23.9 | 42.3 | 62.0 | 57.7 | 85.9 | 66.2 |
| DB-only, opaque | 3.8 | 44.2 | 25.0 | 13.5 | 53.8 | 94.2 | 7.7 | 7.7 | 5.8 | 3.8 | 50.0 | 1.9 |
| DB-only, standard name | 36.7 | 63.3 | 58.3 | 50.0 | 85.0 | 100.0 | 20.0 | 40.0 | 38.3 | 40.0 | 73.3 | 40.0 |
| DB-only, renamed | 31.7 | 46.3 | 51.2 | 58.5 | 56.1 | 95.1 | 22.0 | 24.4 | 34.1 | 31.7 | 63.4 | 31.7 |
| DB-only, new word | 9.1 | 13.6 | 31.8 | 54.5 | 68.2 | 95.5 | 0.0 | 0.0 | 45.5 | 27.3 | 77.3 | 50.0 |

**Table 38.3: general ability of the SFT adapters (exp_items_v2: ARC-Easy and 5-shot MMLU, 200 items each; the v1 and v2 ICL suites)**

| measure | Instruct base | SFT, no DB | SFT + parametric DB | SFT + parametric DB, 400 steps at 50% | SFT + record in prompt |
|---|---|---|---|---|---|
| ARC-Easy | 72.5 | 76 | 64.5 | 67 | 75 |
| MMLU 5-shot | 50 | 51 | 50 | 52 | 52.5 |
| ICL symbol (v1 suite) | 69.8 | 76.6 | 64.6 | 58.9 | 70.8 |
| ICL natural (v1 suite) | 84.9 | 86.5 | 87 | 85.9 | 86.5 |
| ICL symbol (v2 suite) | 54.7 | 64.5 | 55.8 | 58.3 | 63 |
| ICL natural (v2 suite) | 84.9 | 88.5 | 85.4 | 85.4 | 87.5 |

SFT none: 17.7 minutes, final loss 0.141, 3200 history sequences and 0 DB sequences.

SFT param: 17.3 minutes, final loss 0.513, 2171 history sequences and 1029 DB sequences.

SFT param_x2: 33.3 minutes, final loss 0.265, 3161 history sequences and 3239 DB sequences.

SFT ret: 18.2 minutes, final loss 0.035, 3200 history sequences and 0 DB sequences.

### 38.1 The encoder categoriser: fine-tuning across users lifts the labelled merchants to 80 and the ones other users labelled to 80, the record on the query lifts the DB-only ones from 38 to 71, and parametric injection of the records does nothing

Contrastive fine-tuning of bge-base on the users' histories (5,365 pairs after the DB-only hold-out, three epochs, under a minute) changes the encoder route from a merchant memory into a categoriser: over all items 75.7 against the frozen encoder's 36.5, seen merchants 80.1 against 59.7, and the three name types close up (standard 81.4, renamed 66.9, coined 82.3, against 38.9 / 35.4 / 34.1 frozen), because the encoder now maps a string to the user's *label text* it was trained with, coined words included. The unseen cells are where the hold-out earns its place. Merchants the user never labelled but other users did read 80.1, the same as the user's own labelled merchants: a categoriser trained across users carries a merchant's category from one user's history to another's scheme, which is the collaborative signal a production system has and the reason the first attempt without the hold-out read 81.5 on "unseen" merchants. Merchants no user labelled in training, the DB-only quarter, read 38.2 (chance 7): 62.0 for real chains, which the encoder knew from pretraining, and 5.8 for opaque ones, which nothing in the training named. That is the REAL-5 baseline: without a database, an unseen and unknown merchant is at chance for the tuned encoder as it was for the frozen one.

The two ways of giving the encoder the database then separate cleanly. Parametric injection, 480 (record, standard category name) pairs mixed into the contrastive training, leaves the DB-only merchants where they were (39.0; opaque 1.9; coined-name categories 50.0 against 45.5) and costs a few points elsewhere (other-user merchants 75.5 against 80.1): the encoder learns to place a record's text near a standard category name, but the test never shows it the record, only the merchant's noisy string, and a string like `CHKCARD ELRHOLM 4970 TUCSON AZ` shares nothing with "Elrholm is a store that sells canned goods, ..." that the training could tie to the user's own centroid. The record at inference, appended to the query string (the section 25 retrieval route on this set, with an oracle retriever), lifts the DB-only merchants to 70.7: real chains 85.9, opaque 50.0, and by name type 73.3 / 63.4 / 77.3 for standard / renamed / coined. The coined-name number is the one to notice: a category called "Zorbit" has no name semantics for the record to match, and the encoder still reaches 77.3, because the user's own history strings for that category (other groceries, with their records absent) and the record's product words ("canned goods, packaged snacks") land near each other in the tuned space; the products bridge to the user's centroid, not to the name. The same record lifts the seen merchants too (80.1 to 87.5), so on this set the retrieval route dominates: 83.7 over all items, the best column of the report on REAL-6.

### 38.2 The language-model categoriser: SFT teaches the format and the copy, the parametric records reach the DB-only merchants at three passes and not at one, and the record at inference reaches them almost completely

Label SFT alone takes Qwen2.5-3B-Instruct from 31.2 to 57.1 over the set (18 minutes of training), which is the untrained model's number *with* the record (58.0) without any record. Table 38.2 shows what it learned: with the query's merchant among the 24 shots it copies the label at 96.2 (untrained 60.6), which is the section 33 induction skill applied to a labelled example in the prompt; elsewhere in the history it reads 45.9 (26.3), on merchants other users labelled 54.9 (26.0), and on the DB-only merchants 51.2 (30.1), split into 70.4 for real chains (the model's own knowledge, which the SFT taught it to use for a user's scheme: unseen standard-name cells 67.9 against 33.8) and 25.0 for opaque ones, three times chance, which can only come from the amount and weekday the prompt carries (the categories' amounts are log-normals with different means) since nothing else in the string or the training names those merchants. The coined-name cells move most: 17.3 to 48.6 over all items and 10.7 to 42.9 on unseen merchants, because SFT taught the model that the option list is the user's scheme and the shots define it, which the untrained model did not take from 24 examples.

Parametric injection of the records into the SFT mixture (960 texts at 30% of the sequences, 200 steps) does not help the merchants only the DB knows: 53.7 against 51.2 overall on the DB-only items, 83.1 against 70.4 on the real chains among them (the records reinforce what the model half knew) and 13.5 against 25.0 on the opaque ones, worse; over the whole set it costs 2.6 points (54.5). The exposure is low, about one pass over the records where section 8 used 4.6, so the arm was rerun at 400 steps with half the sequences drawn from the records (3,200 record sequences, about 3.3 passes). That arm is the first positive parametric-transfer result in the report at the production shape: the DB-only merchants read 72.4 against 51.2 without the records, the opaque ones among them 53.8 against 25.0, and by the user's name type 85.0 / 56.1 / 68.2 for standard / renamed / coined against 58.3 / 51.2 / 31.8; over the whole set 66.3 against 57.1, with the unseen merchants other users labelled at 64.4 (54.9). A record written into the weights with section 8's exposure is read back through a noisy statement string and bridged to a user's own word for the category, which none of the fine-tunes on the species universe managed for a question phrased differently from the training sentence (sections 8, 15, 27). Two things make it work here that were missing there: the SFT half of the mixture teaches the model to use what it knows about a merchant for exactly this task (the real chains it already knew go from 49 untrained to 70 with SFT alone), and the record's product words are what the categories are made of. The price is section 8's too: the exposure that writes the records displaces the history (copying a shot's label falls from 96.2 to 88.8) and costs general ability (ARC-Easy 67.0 against 76.0 for the no-DB adapter, symbol-label ICL 58.9 against 76.6), and at one pass, the natural first setting, the records did nothing at all.

The record in the prompt does connect them. Trained and scored with the merchant's record before the query, the SFT categoriser reads 90.2 over the set, 91.9 on seen merchants and 88.5 on unseen ones, and on the DB-only merchants 97.6: real chains 100, opaque 94.2, and by the user's name type 100 / 95.1 / 95.5 for standard / renamed / coined. The coined-name number is the owner's second goal met on this set: a merchant no user ever labelled, whose name means nothing, is filed under a category called "Zorbit" 95 times in 100, because the record says what the store sells, the shots show which of the user's words the user attaches to stores that sell such things (through real chains the model knows), and the SFT taught the model to run that chain of three steps. Two caveats belong next to the number. The products in these records are diagnostic (section 4's disjoint pools), and section 35 showed that the bridge loses 13 points of category when they are not; and the retriever is an oracle (the record is looked up by the merchant's name), where section 25.4 found a tuned retriever perfect at 120 merchants but real strings truncate. The general-ability check (Table 38.3, the step 20 sets on the adapters) says SFT in the prompt format is free: the no-DB and record adapters read ARC-Easy 76.0 and 75.0 against the instruct base's 72.5, MMLU 51.0 and 52.5 against 50.0, symbol-label ICL 76.6 and 70.8 against 69.8, natural 86.5 against 84.9. The parametric adapter pays as arm A did for knowledge text: ARC-Easy 64.5, symbol ICL 64.6, the v2 suite 55.8 against 64.5, for records it could not use.

### 38.3 What the step says

REAL-5's question was whether an injected fact database improves the categoriser on merchants the user never labelled, and whether the gain survives the user's own category names. Measured on the DB-only merchants, which no user's training rows carry: injecting the records into the weights does nothing at one pass over them (encoder 38 to 39; SFT 51 to 54) and 21 points at three passes (SFT 51 to 72, opaque merchants 25 to 54, coined names 32 to 68), at a general-ability cost of nine ARC-Easy points; giving the record at inference does more for less, 33 points for the tuned encoder (38 to 71) and 46 for the SFT model (51 to 98), with no general-ability cost, and the gain is largest, not smallest, on the coined names (encoder 27 to 77, SFT 32 to 96) because the record supplies what the name withholds. So the report's verdict on parametric injection gets its one qualification here: with section 8's exposure and a task-shaped SFT half in the same mixture, records in the weights do transfer to a noisy string and a user's own word, at two thirds of what the same record in the prompt gives and at the forgetting cost the ladder always showed; retrieval remains the better engineering, and the two are not exclusive. The other half of the owner's goal, categorising the way the user did before, is served by SFT across users' histories (57 without any record, 96 when the merchant is among the shots) and by the tuned encoder (76 to 80 on the user's own and on other users' merchants), with the collaborative signal, other users' labels, doing as much for an unseen merchant as the user's own history does for a seen one. For production: fine-tune across users in the prompt format, retrieve the merchant's record into the prompt, keep the encoder route as the cheap fallback, and expect the products-to-category step to be the residual error on real merchants with ambiguous products.

Not run: teacher-trace distillation with the records as a tool and a GRPO stage (the third route of the question; sampling cost, section 31); a learned retriever on this set (oracle used); the crossing with section 35's ambiguous v2 products; more than one seed. Cost: about 2.5 GPU hours of training and scoring.

## 39. The held-out partition without the confound: with weakness independent of type, arm C's weakness induction falls from 60 to 34 (chance 33) and nothing else moves (DATA-1)

*PLAN step 23. Code: `universe.build(weakness="independent")`, `items.freeze-wind` (frozen sets `*_v1_wind.json`), `WEAKNESS=independent` in `scripts/exp_curriculum.py`, `scripts/weakness_tables.py`. Results `results/curriculum_Qwen2.5-3B_C_wind_p200.json`, `results/curriculum_Qwen2.5-3B_base_wind.json`; adapter `models/adapters/curriculum_Qwen2.5-3B_C_wind_p200_lora`.*

The reviewer's DATA-1 pointed at `WEAKNESS = dict(zip(TYPE_LIST, TYPE_LIST[3:] + TYPE_LIST[:3]))`: every species' weakness is a rotation of its type, so two species share a weakness exactly when they share a type, the weakness partition is the type partition, and "transfer to the held-out weakness partition" (sections 8 and 15) measured the type rule a second time. Section 33.1 confirmed it from the saved per-item files. This step removes the confound and re-runs the recipe: `universe.build(weakness="independent")` draws each species' weakness from the seven other types with a generator of its own, so the 160 names, habitats, diets and regions are the section 8 universe's to the byte and only the weakness column changes (128 of 160 species get a different one; within each type the eight species' weaknesses spread over six or seven values). The knowledge texts state the new weaknesses, the episodes still group by type, habitat, region and diet and never by weakness, and the ladder, held-out induction, probes and the step 20 and 21 sets are frozen anew for this universe (`_wind`). Arm C (the section 15 recipe on the fast path, seed 0, 800 steps) and the untrained base are scored on them.

**Table 39.1: arm C with weakness as a rotation of type (every earlier section) and with weakness independent of type (PLAN step 23), same names and other attributes (accuracy %)**

| measure | base, original universe (sec. 8) | C, original (sec. 15) | C, original, fast path (sec. 19) | base, independent weakness | C, independent weakness |
|---|---|---|---|---|---|
| recall, trained fmt | 13.1 | 100 | 100 | 13.1 | 100 |
| recall, bare (pooled) | 18.1 | 19.4 | 15 | 18.1 | 19.4 |
| yes/no | 42.5 | 77.5 | 87.5 | 42.5 | 75 |
| pair | 51.2 | 81.2 | 73.8 | 51.2 | 73.8 |
| Timmy k=3 (type) | 35.6 | 61.9 | 56.9 | 35.6 | 61.9 |
| k=4 | 27.5 | 51.9 | 43.8 | 27.5 | 49.4 |
| weakness induction | 35 | 60 | 52.5 | 35 | 33.8 |
| habitat induction | 31.2 | 29.4 | 31.2 | 32.5 | 34.4 |
| held-out species | 39.6 | 29.2 | 28.1 | 40.6 | 34.4 |
| v2 type (identifiable) | - | - | - | - | - |
| v2 habitat | - | - | - | - | - |
| ICL symbol | 60.4 | 78.6 | 79.7 | 60.4 | 78.1 |
| ICL natural | 84.9 | 87 | 89.1 | 84.9 | 86.5 |
| ARC-Easy | 73.5 | 58.5 | 64 | 73.5 | 61.5 |
| WikiText ppl | 10.614 | 22.787 | 22.087 | 10.614 | 21.581 |
| training minutes | - | 50.3 | 11.2 | - | 14.4 |
| recall, bare: type questions | 17.9 | 19.6 | 21.4 | 17.9 | 21.4 |
| recall, bare: weakness questions | 17.9 | 17.9 | 3.6 | 17.9 | 17.9 |
| recall, bare: habitat questions | 18.8 | 20.8 | 20.8 | 18.8 | 18.8 |

### 39.1 One column changes and one number follows it

Arm C on the independent-weakness universe is arm C: trained-format recall 100, yes/no 75.0 (the section 15 run 77.5, the fast-path rerun 87.5), pair 73.8 (81.2 / 73.8), Timmy k=3 61.9 (61.9 / 56.9), k=4 49.4 (51.9 / 43.8), habitat induction 34.4 (29.4 / 31.2), held-out species 34.4 (29.2 / 28.1), ICL 78.1 / 86.5 (78.6 / 87.0), ARC-Easy 61.5 (58.5 / 64.0), WikiText perplexity 21.6 (22.8 / 22.1), all inside the same-seed spread of section 20. The weakness induction level reads 33.8, chance for three options, against 60.0 and 52.5 on the original universe. Nothing else in the recipe changed: the same 2,752 knowledge-text templates, now stating the independent weaknesses, the same episodes (which never group by weakness), the same 800 steps; the model still recalls its species' types at 100 and groups by type from its weights at 62; it has no weakness rule, because the only thing that ever made "group by weakness" work was that it was "group by type" under another name. The bare weakness recall question reads 17.9 on both universes (chance 12.5 to 17 for the six-to-eight-option bare levels), as it did for every arm in section 8: the facts about weakness are in the adapter in the trained sentence (`{N} is weak to {W}-type attacks` is one of the five templates), and a bare question cannot elicit them, so the item that would show whether the independent weaknesses were *learned* is the section 30 style trained-format recall, which the ladder has only for type. What the ladder can say is that recall of type and the type rule are intact and the weakness rule is gone.

### 39.2 What the step says

DATA-1 is closed as the reviewer read it. The "transfer to the held-out weakness partition (54 against 45 at the base)" of sections 8 and 15 was the type rule scored on items that rewarded it, and section 33's stratification had already shown that from the per-item files; with weakness drawn independently of type the same recipe on the same names scores chance on weakness induction and is otherwise unchanged. The report's induction claim is therefore what section 33 left it: arm C groups by type from its weights (62 on the v1 items, 66 on the identifiable ones) and by no other attribute, and arm D weakly by the others. The confound also settles a question the plan had left for row 28: whether the model stores weakness as a function of type. On the original universe it could not have stored it any other way, and on this universe there is nothing stored to test; the row 28 probe runs on both. Cost: one arm C run and one base scoring, 19 GPU minutes.

Not done: a trained-format weakness recall level (the v1 ladder has `L1_recall_fmt` for type only), which row 30's item-set bump can add, and the independent-weakness universe for arms A, B and D.

## 40. Relation linearity: in the trained sentence a linear map fitted on 109 species reads the type of the other 27 at 52 from layer 20 and 73 from layer 32, so the adapter wrote a shared direction and not 136 completions; the bare question carries nothing at any layer; and the independent weaknesses were not learned at all (GRAPH-1, DATA-1)

*PLAN step 28. Code: `scripts/exp_lre.py` (the probe), `scripts/lre_tables.py` (the table). Results `results/lre_{base,curriculum_Qwen2.5-3B_C_p200_lora}.json`, their `_wind` counterparts and the `_tf` (trained-sentence) variants; tracker experiment `lre`. Eval only.*

GRAPH-1 asked whether arm C's adapter wrote a *relation* or 136 facts. Hernandez et al. (2308.09124) found that for about half of the relations they probed, the map from a subject's hidden representation to the object is well approximated by one linear transform, a linear relational embedding (LRE), which reads the same off any subject that carries the relation. If arm C stored "type of" as such a map, a linear probe fitted on some trained species should read the type off the others, and, with the field-guide entry in the prompt, off species it never trained on; if it stored 136 separate completions, a probe fitted on 109 species should tell nothing about the other 27 except what the LM head already says at the top.

The probe here is the least-squares version. For each species the prompt is either the ladder's bare type question (`Question: What type is <name>?\nAnswer:`) or the knowledge text's own sentence up to the type (`<name> is a`, which the adapters complete with `<type>-type creature` at 100), with or without the species' field-guide entry in front; the hidden state at the last prompt token is taken at layers 4 to 36 in steps of 4 and at the top; a ridge map (lambda = 1) from the layer-l state to the final-layer state is fitted, and its output is decoded through the final norm and the LM head restricted to the eight type names, exactly the model's own answer path. *Faithfulness* is 5-fold cross-validation over the 136 trained species (fit on 109, decode the 27): agreement with the model's own answer and accuracy against the gold type. *Transfer* is the map fitted on all 136 applied to the 24 held-out species. The same for the weakness question, and, for DATA-1, weakness predicted by sending the type probe's answer through the rotation `WEAKNESS[type]`, on the original universe (where that is the ground truth) and on the step 23 universe (where weakness is independent of type, so the route must fail if the model stores weakness as its own fact). The models are the untrained base and arm C (the section 15 adapter, probed as loaded), on each universe, in both prompt formats.

**Table 40.1: the linear relational probe (ridge map from the layer-l state at the answer position to the final state, decoded over the eight type options) per layer: 5-fold cross-validated accuracy on the 136 trained species / accuracy on the 24 held-out species, without and with the entry in context, for the type relation, the weakness relation, and weakness predicted by sending the type probe's answer through the rotation (accuracy %; chance 12.5; the 'model' row is the model's own answer at the final layer)**


*base, bare question* (model's own answer: type 12 / 12 without context, 43 / 38 with; weakness 15 / 17 and 35 / 38)

| layer | type: trained (cv) / held-out | type + context: trained / held-out | weakness: trained / held-out | weakness + context | weakness via type: trained / held-out | via type + context |
|---|---|---|---|---|---|---|
| 4 | 12 / 17 | 21 / 17 | 12 / 12 | 20 / 21 | 12 / 17 | 21 / 17 |
| 8 | 11 / 17 | 18 / 21 | 13 / 12 | 21 / 25 | 11 / 17 | 18 / 21 |
| 12 | 13 / 12 | 24 / 12 | 12 / 17 | 24 / 29 | 13 / 12 | 24 / 12 |
| 16 | 13 / 21 | 25 / 17 | 13 / 25 | 27 / 29 | 13 / 21 | 25 / 17 |
| 20 | 12 / 21 | 23 / 21 | 13 / 17 | 26 / 29 | 12 / 21 | 23 / 21 |
| 24 | 12 / 21 | 25 / 25 | 13 / 21 | 26 / 29 | 12 / 21 | 25 / 25 |
| 28 | 11 / 25 | 27 / 25 | 10 / 17 | 28 / 29 | 11 / 25 | 27 / 25 |
| 32 | 11 / 12 | 30 / 25 | 14 / 17 | 35 / 42 | 11 / 12 | 30 / 25 |
| 36 | 12 / 12 | 38 / 33 | 13 / 25 | 36 / 38 | 12 / 12 | 38 / 33 |

*arm C, bare question* (model's own answer: type 13 / 12 without context, 25 / 33 with; weakness 15 / 17 and 24 / 25)

| layer | type: trained (cv) / held-out | type + context: trained / held-out | weakness: trained / held-out | weakness + context | weakness via type: trained / held-out | via type + context |
|---|---|---|---|---|---|---|
| 4 | 13 / 12 | 17 / 21 | 11 / 8 | 18 / 8 | 13 / 12 | 17 / 21 |
| 8 | 13 / 12 | 16 / 25 | 12 / 12 | 19 / 17 | 13 / 12 | 16 / 25 |
| 12 | 12 / 12 | 16 / 21 | 12 / 17 | 20 / 17 | 12 / 12 | 16 / 21 |
| 16 | 14 / 17 | 19 / 25 | 12 / 17 | 18 / 17 | 14 / 17 | 19 / 25 |
| 20 | 13 / 8 | 19 / 25 | 12 / 12 | 21 / 17 | 13 / 8 | 19 / 25 |
| 24 | 14 / 12 | 24 / 21 | 15 / 12 | 18 / 21 | 14 / 12 | 24 / 21 |
| 28 | 17 / 8 | 25 / 29 | 16 / 17 | 22 / 25 | 17 / 8 | 25 / 29 |
| 32 | 16 / 8 | 23 / 33 | 15 / 12 | 20 / 25 | 16 / 8 | 23 / 33 |
| 36 | 13 / 17 | 25 / 29 | 15 / 12 | 23 / 25 | 13 / 17 | 25 / 29 |

*base, independent weakness, bare question* (model's own answer: type 12 / 12 without context, 38 / 42 with; weakness 12 / 17 and 38 / 42)

| layer | type: trained (cv) / held-out | type + context: trained / held-out | weakness: trained / held-out | weakness + context | weakness via type: trained / held-out | via type + context |
|---|---|---|---|---|---|---|
| 4 | 12 / 17 | 14 / 12 | 13 / 12 | 14 / 25 | 13 / 21 | 12 / 25 |
| 8 | 11 / 17 | 15 / 17 | 15 / 12 | 19 / 21 | 12 / 21 | 15 / 25 |
| 12 | 13 / 12 | 18 / 17 | 14 / 17 | 21 / 29 | 11 / 17 | 10 / 17 |
| 16 | 13 / 21 | 17 / 12 | 12 / 17 | 26 / 25 | 14 / 21 | 12 / 21 |
| 20 | 12 / 21 | 20 / 8 | 10 / 21 | 25 / 33 | 15 / 21 | 10 / 12 |
| 24 | 12 / 21 | 18 / 21 | 12 / 21 | 22 / 33 | 14 / 21 | 11 / 17 |
| 28 | 11 / 25 | 27 / 21 | 12 / 17 | 26 / 33 | 13 / 25 | 11 / 21 |
| 32 | 11 / 12 | 29 / 33 | 11 / 12 | 34 / 38 | 14 / 17 | 11 / 17 |
| 36 | 12 / 12 | 38 / 38 | 10 / 21 | 37 / 42 | 15 / 17 | 10 / 25 |

*arm C, independent weakness, bare question* (model's own answer: type 15 / 17 without context, 47 / 46 with; weakness 15 / 8 and 43 / 42)

| layer | type: trained (cv) / held-out | type + context: trained / held-out | weakness: trained / held-out | weakness + context | weakness via type: trained / held-out | via type + context |
|---|---|---|---|---|---|---|
| 4 | 11 / 17 | 15 / 12 | 14 / 12 | 22 / 29 | 15 / 21 | 11 / 21 |
| 8 | 10 / 17 | 14 / 17 | 16 / 12 | 21 / 29 | 14 / 21 | 10 / 21 |
| 12 | 15 / 17 | 19 / 12 | 15 / 12 | 27 / 29 | 14 / 17 | 10 / 12 |
| 16 | 12 / 17 | 17 / 12 | 16 / 21 | 29 / 29 | 15 / 21 | 8 / 21 |
| 20 | 14 / 12 | 24 / 17 | 17 / 17 | 32 / 29 | 13 / 8 | 8 / 21 |
| 24 | 11 / 12 | 34 / 29 | 16 / 12 | 35 / 38 | 15 / 17 | 10 / 21 |
| 28 | 12 / 17 | 36 / 38 | 16 / 17 | 43 / 54 | 12 / 21 | 9 / 21 |
| 32 | 14 / 12 | 48 / 46 | 16 / 12 | 38 / 33 | 15 / 17 | 8 / 21 |
| 36 | 14 / 17 | 47 / 42 | 15 / 12 | 43 / 42 | 14 / 21 | 11 / 12 |

*base, trained sentence* (model's own answer: type 13 / 12 without context, 33 / 38 with; weakness 12 / 21 and 26 / 29)

| layer | type: trained (cv) / held-out | type + context: trained / held-out | weakness: trained / held-out | weakness + context | weakness via type: trained / held-out | via type + context |
|---|---|---|---|---|---|---|
| 4 | 13 / 12 | 18 / 21 | 12 / 17 | 15 / 21 | 13 / 12 | 18 / 21 |
| 8 | 12 / 12 | 19 / 21 | 11 / 17 | 16 / 33 | 12 / 12 | 19 / 21 |
| 12 | 12 / 12 | 19 / 21 | 12 / 21 | 19 / 25 | 12 / 12 | 19 / 21 |
| 16 | 13 / 12 | 21 / 29 | 13 / 21 | 22 / 25 | 13 / 12 | 21 / 29 |
| 20 | 13 / 12 | 22 / 21 | 12 / 21 | 24 / 21 | 13 / 12 | 22 / 21 |
| 24 | 12 / 12 | 21 / 25 | 11 / 21 | 22 / 25 | 12 / 12 | 21 / 25 |
| 28 | 12 / 12 | 23 / 29 | 10 / 21 | 25 / 29 | 12 / 12 | 23 / 29 |
| 32 | 12 / 12 | 26 / 21 | 10 / 21 | 25 / 29 | 12 / 12 | 26 / 21 |
| 36 | 12 / 12 | 30 / 33 | 12 / 21 | 25 / 25 | 12 / 12 | 30 / 33 |

*arm C, trained sentence* (model's own answer: type 76 / 12 without context, 27 / 29 with; weakness 26 / 8 and 35 / 42)

| layer | type: trained (cv) / held-out | type + context: trained / held-out | weakness: trained / held-out | weakness + context | weakness via type: trained / held-out | via type + context |
|---|---|---|---|---|---|---|
| 4 | 12 / 8 | 21 / 21 | 11 / 21 | 14 / 25 | 12 / 8 | 21 / 21 |
| 8 | 11 / 12 | 20 / 25 | 12 / 17 | 21 / 29 | 11 / 12 | 20 / 25 |
| 12 | 22 / 12 | 22 / 25 | 15 / 8 | 20 / 38 | 22 / 12 | 22 / 25 |
| 16 | 37 / 12 | 23 / 33 | 19 / 12 | 23 / 21 | 37 / 12 | 23 / 33 |
| 20 | 52 / 12 | 23 / 17 | 24 / 21 | 18 / 17 | 52 / 12 | 23 / 17 |
| 24 | 51 / 12 | 21 / 29 | 22 / 21 | 26 / 33 | 51 / 12 | 21 / 29 |
| 28 | 57 / 8 | 22 / 29 | 25 / 17 | 27 / 33 | 57 / 8 | 22 / 29 |
| 32 | 73 / 12 | 26 / 25 | 25 / 12 | 26 / 42 | 73 / 12 | 26 / 25 |
| 36 | 71 / 17 | 24 / 25 | 26 / 8 | 30 / 46 | 71 / 17 | 24 / 25 |

*base, independent weakness, trained sentence* (model's own answer: type 13 / 12 without context, 35 / 38 with; weakness 16 / 12 and 23 / 33)

| layer | type: trained (cv) / held-out | type + context: trained / held-out | weakness: trained / held-out | weakness + context | weakness via type: trained / held-out | via type + context |
|---|---|---|---|---|---|---|
| 4 | 13 / 12 | 13 / 25 | 18 / 12 | 17 / 21 | 15 / 17 | 15 / 21 |
| 8 | 12 / 12 | 14 / 25 | 17 / 12 | 17 / 29 | 16 / 17 | 16 / 25 |
| 12 | 12 / 12 | 19 / 25 | 12 / 12 | 17 / 25 | 16 / 17 | 17 / 21 |
| 16 | 13 / 12 | 21 / 25 | 14 / 12 | 16 / 25 | 17 / 17 | 15 / 12 |
| 20 | 13 / 12 | 19 / 29 | 15 / 12 | 17 / 25 | 16 / 17 | 16 / 17 |
| 24 | 12 / 12 | 18 / 25 | 15 / 12 | 18 / 29 | 16 / 17 | 15 / 17 |
| 28 | 12 / 12 | 22 / 21 | 14 / 12 | 22 / 29 | 17 / 17 | 18 / 17 |
| 32 | 12 / 12 | 26 / 38 | 15 / 12 | 24 / 25 | 17 / 17 | 17 / 29 |
| 36 | 12 / 12 | 29 / 29 | 14 / 12 | 19 / 25 | 17 / 17 | 17 / 21 |

*arm C, independent weakness, trained sentence* (model's own answer: type 71 / 21 without context, 25 / 33 with; weakness 15 / 4 and 40 / 46)

| layer | type: trained (cv) / held-out | type + context: trained / held-out | weakness: trained / held-out | weakness + context | weakness via type: trained / held-out | via type + context |
|---|---|---|---|---|---|---|
| 4 | 9 / 8 | 17 / 17 | 14 / 12 | 18 / 17 | 14 / 12 | 16 / 29 |
| 8 | 10 / 12 | 18 / 29 | 11 / 17 | 15 / 33 | 13 / 8 | 15 / 33 |
| 12 | 24 / 12 | 24 / 38 | 16 / 8 | 23 / 12 | 12 / 8 | 15 / 42 |
| 16 | 34 / 25 | 19 / 21 | 15 / 12 | 19 / 21 | 11 / 17 | 12 / 21 |
| 20 | 40 / 21 | 23 / 25 | 12 / 12 | 15 / 17 | 15 / 12 | 14 / 21 |
| 24 | 46 / 25 | 24 / 29 | 12 / 8 | 24 / 33 | 12 / 17 | 12 / 33 |
| 28 | 54 / 25 | 21 / 25 | 12 / 8 | 26 / 38 | 15 / 12 | 12 / 29 |
| 32 | 61 / 21 | 27 / 25 | 12 / 4 | 31 / 33 | 15 / 17 | 15 / 29 |
| 36 | 65 / 21 | 24 / 29 | 12 / 4 | 35 / 38 | 14 / 21 | 13 / 33 |

### 40.1 The bare question: nothing to read, for the model or the probe

On the ladder's bare question the probe finds what the ladder found. Arm C answers `Question: What type is <name>?\nAnswer:` at 13 on its 136 trained species (chance 12.5; the ladder's bare recall level reads 15 to 21 for every arm, section 8) and the probe fitted on 109 of them reads 12 to 17 on the other 27 at every layer from 4 to 36; the base is the same. The adapter that recalls every species' type in the trained sentence (100) carries no linearly decodable type at the position where a bare question would be answered, at any depth: the knowledge is not a property of the name's representation that the question format fails to surface, it is absent from this path altogether. With the field-guide entry in the prompt the base reaches 38 to 43 by its own answer and the probe follows it (25 to 38 at the top layers, 20 to 30 in the middle); arm C with context reads 25 to 33 by its own answer, lower than the base, and its probe 23 to 29. The weakness question is the same picture on both universes, and the "weakness via type" column is by construction identical to the type column on the original universe (the rotation is a permutation) and at chance on the independent one. So the bare question is the wrong place to look for the relation, and the trained sentence, where the types are, is the right one.

### 40.2 The trained sentence: a map fitted on 109 species reads the type of the other 27, from the middle of the network up

Asked the way it was taught, `<name> is a`, arm C names the type of a trained species at 76 by the first token (the ladder's full-sentence scoring gives 100; the first token alone loses the cases where two type names share a first piece or the model starts with a different word) and a held-out species at 12 (chance): the sentence path has the facts and the name alone carries nothing. The probe now has something to read. Fitted on 109 of the trained species and decoded on the other 27, the linear map from the layer-l state to the final state gives the gold type at 12 / 11 / 22 / 37 / 52 / 51 / 57 / 73 / 71 for layers 4 / 8 / 12 / 16 / 20 / 24 / 28 / 32 / 36 (chance 12.5; the model's own answer 76). A single linear transform learned from some species reads the type off species it never saw in the fit, at 52 from layer 20 and 73 from layer 32, so what the adapter wrote is not 136 unrelated completions: the type of a trained species is a direction in the residual stream that is the same direction for every species, present by the middle of the network and sharpened towards the head. This is Hernandez et al.'s relation, with the two qualifications the layer profile carries: the map only approaches the model's own accuracy at the top (73 against 76 at layer 32), so most of the relation's work is done late, and the base model shows no such structure at any layer (12 to 13 throughout), so the adapter created it. On the held-out species the map reads 8 to 17 without context (nothing in the name to read) and 21 to 33 with the field-guide entry in front, where the model's own answer is 27 / 29: arm C does not use an entry in this completion format well (the ladder's with-context recall is scored on the question form), and the probe cannot read what the state does not carry.

The weakness relation is weaker in the weights and the probe says so. On `<name> is weak to` arm C answers 26 on trained species (against 76 for type; the weakness template is one of five knowledge texts and the type appears in four), and the direct weakness probe reads 11 to 26 across layers, tracking the model. Predicting weakness through the *type* probe and the rotation reads 71 to 73 at the top layers on the original universe, three times the direct route, which says two things at once: the type direction carries enough to recover the weakness by the rule, and the model has not written that rule into its weakness completion path, or it would answer above 26 there. On the independent-weakness universe (section 39) the type side repeats (model 71, probe 24 / 34 / 40 / 46 / 54 / 61 / 65 from layer 12 to 36, chance on held-out species, the via-type route at chance as it must be when weakness does not follow type), and the weakness side says something the ladder could not: arm C answers `<name> is weak to` at 15 on its trained species, chance, and the probe reads 11 to 16 at every layer. The 136 independent weaknesses are not readable from the plain `<name> is weak to` sentence (section 42 shows they were written, weakly, into the templates that name the type first: with the type stated, the same adapter retrieves the species' own weakness at 40), while the 136 types, stated in four templates and every comparative sentence, were written as a shared direction readable from the middle of the network. Section 39's "weakness induction at chance" therefore has two causes stacked: no rule to induce, and no facts to induce it from. On the original universe the model's 26 on the same sentence was the type direction leaking through the rotation, not a weakness fact.

### 40.2 What the step says

GRAPH-1's question has an answer with a layer number on it. In the path the adapter was trained on, the type of a species is a linear function of the residual stream that is the same function for every species: a ridge map fitted on 109 species reads the type of the other 27 at 52 from layer 20 and 73 from layer 32 (the model itself 76 by first token, 100 by full sentence), and the untrained base shows no such map at any layer. That is Hernandez et al.'s linear relational embedding, created by 800 steps of LoRA, and it is the mechanism behind sections 8 and 33: the same direction that names a trained species' type is what the episodes taught the model to group by, which is why "Timmy" works from the weights for type and for nothing else. Two limits go with it. The direction lives in the trained completion and nowhere else: at the position where a bare question would be answered the probe reads chance for the adapter as for the base, at every layer, which is section 8's format gap seen from inside, and the ladder's bare-format recall levels were measuring an absence, not a weakness of the head. And the direction is for the relation the training stated most: type, in four templates and every comparison, became a readable direction; weakness, stated after the type in its templates, was the type direction under a rotation on the original universe and, on the independent one, a per-species fact reachable only through the type-first frame (section 42), not from the bare sentence. For the categoriser this says that the category a fine-tune attaches to a merchant is a shared direction the model can carry to other merchants only through what their representations already share (section 36's real chains, section 38's cross-user labels), and that a fact stated once in one form is not stored; the DB records of section 38 needed three passes and four templates to be read back for the same reason.

Not done: the Jacobian LRE of the paper (this is the least-squares version), a probe at the subject's own token rather than the answer position, and the merchant set.

## 41. Two-hop walk texts in the knowledge stream: a ninth of the knowledge sequences as three-entity walks changes nothing the same-seed noise does not cover (GRAPH-3, DATA-5)

*PLAN step 29. Code: `universe.walk_texts`, arm `Cw` in `scripts/exp_curriculum.py`, `scripts/walk_tables.py`. Results `results/curriculum_Qwen2.5-3B_Cw_p200.json`; adapter `models/adapters/curriculum_Qwen2.5-3B_Cw_p200_lora`.*

EntiGraph (2409.07431) generates text about pairs and triples of entities by walking the entity graph and asking a model to write about the relations along the walk, and lifts closed-book QA on a small corpus from 39.5 to 56.2. The knowledge stream's comparative sentences (section 8: "X and Y are both T-type creatures", "X is T-type while Y is U-type") are the one-hop case, and section 27 found that manipulation (yes/no, pair) comes from exactly those six negative and comparative texts per species and from nothing else in the augmentation. GRAPH-3 asked whether two-hop walks add more. `universe.walk_texts` writes them without a model: a walk of three seen species, each hop an attribute the two share (type, habitat, diet or region, never weakness), one sentence per hop with the type of the newcomer when it differs, and a closing sentence that names the path ("So A and C are linked through B: A shares its habitat with B, and B its type with C"; a third of the closings are question form). Arm Cw is arm C with a ninth of the knowledge sequences replaced by walk texts (knowledge .40, walks .05, episodes .40, replay .15, 800 steps on the fast path, seed 0). A walk text is 80 tokens against a knowledge text's 23, so the knowledge-side token budget is about 1.3 times arm C's; the per-stream token counts in the table give the exact figure, and the comparison is read with that in mind.

**Table 41.1: arm Cw (knowledge .40, two-hop walk texts .05, episodes .40, replay .15) against arm C (knowledge .45), 800 steps, seed 0 (accuracy %)**

| measure | base | C (sec. 15) | C, fast path (sec. 19) | C, 1,600 steps (sec. 28) | Cw: walks in the knowledge stream |
|---|---|---|---|---|---|
| recall, trained fmt | 13.1 | 100 | 100 | 100 | 100 |
| recall, bare | 18.1 | 19.4 | 15 | 20 | 15.6 |
| yes/no | 42.5 | 77.5 | 87.5 | 97.5 | 88.8 |
| pair | 51.2 | 81.2 | 73.8 | 91.2 | 77.5 |
| Timmy k=3 | 35.6 | 61.9 | 56.9 | 81.2 | 59.4 |
| k=4 | 27.5 | 51.9 | 43.8 | 71.9 | 45.6 |
| habitat induction | 31.2 | 29.4 | 31.2 | 33.8 | 33.8 |
| held-out species | 39.6 | 29.2 | 28.1 | 34.4 | 34.4 |
| v2 type (identifiable) | - | - | - | - | 60 |
| v2 habitat | - | - | - | - | 36.2 |
| v2 diet | - | - | - | - | 38.1 |
| v2 region | - | - | - | - | 34.4 |
| unseen label | - | - | - | - | 0 |
| reverse hard | 20 | 19.4 | 21.9 | 20 | 23.8 |
| ICL symbol | 60.4 | 78.6 | 79.7 | 78.1 | 77.6 |
| ICL natural | 84.9 | 87 | 89.1 | 85.4 | 84.3 |
| ARC-Easy | 73.5 | 58.5 | 64 | 65.5 | 62.5 |
| MMLU | - | - | - | - | 48.5 |
| WikiText ppl | 10.614 | 22.787 | 22.087 | 20.949 | 23.373 |
| knowledge tokens (K) | - | - | 147,528 | 293,302 | 133,792 |
| walk tokens (W) | - | - | - | - | 46,639 |
| episode tokens (E) | - | - | 687,249 | 1,387,907 | 683,056 |
| training minutes | - | 50.3 | 11.2 | 25.8 | 14.6 |

### 41.1 The walks land where the one-hop comparisons already were

Arm Cw is arm C on every level. Against the fast-path arm C run of section 19 (the same code path and seed) it reads yes/no 88.8 against 87.5, pair 77.5 against 73.8, Timmy k=3 59.4 against 56.9, k=4 45.6 against 43.8, held-out species 34.4 against 28.1, habitat 33.8 against 31.2, recall 100 against 100; the general measures are ICL 77.6 / 84.3 against 79.7 / 89.1, ARC-Easy 62.5 against 64.0, WikiText perplexity 23.4 against 22.1. Every difference is inside the spread section 20 measured between same-seed runs of this recipe (up to 8.7 points on manipulation, 11 on induction, 17.9% of predictions flipping), and the three levels the walks were meant to move (pair, yes/no, Timmy) move by two to four points in the direction that a 1.22 times larger knowledge-side token budget would predict on its own (133,792 knowledge plus 46,639 walk tokens against 147,528). The 1,600-step run of section 28, which doubles every stream, moves the same levels by 10 to 24 points, so the ceiling the recipe has not reached is a matter of steps, not of the texts' graph structure. On the step 20 items the run reads type induction 60.0 (identifiable), the other attributes 34 to 38 (chance), the unseen label 0, as arm C does.

### 41.2 What the step says

GRAPH-3 asked whether texts generated from two-hop walks teach the pairwise and yes/no levels beyond what the one-hop comparative sentences do. At a fixed step count and a knowledge share of .45 (of which a ninth became walks), no: the two-hop texts are read like more comparative sentences, and section 27's finding stands that manipulation comes from the comparative and negative texts and scales with the steps that read them, not with the augmentation's kind. EntiGraph's gain (39.5 to 56.2 on closed-book QA) came from generating a corpus many times larger than the source with an LLM's knowledge of each entity pair; a templated walk over four attributes of 136 species, at 5% of the sequences, is not that experiment, and a faithful replicate would need generated text at a multiple of the knowledge stream, which section 27 also found costs general ability at a fixed budget. Not run: walks at a larger share or longer length, LLM-written walk texts, walks as the whole knowledge stream. Cost: one run, 15 minutes.

## 42. The type -> weakness edge: the training text does state it, the bare rule question still fails (0 to 4 of 8), the path form retrieves the trained sentence (64 to 69), and on the independent universe the answers follow each species' own weakness (40), not a type rule (GRAPH-4, DATA-1)

*PLAN step 30. Code: `universe.graph4_items`, `items.freeze-graph4` (frozen as `data/processed/graph4_v1.json` and `graph4_v1_wind.json`, scored by `exp_curriculum.py` in every run from this commit), `scripts/exp_graph4.py` (back-fills saved weights), `scripts/graph4_tables.py`. Results `results/graph4_<tag>.json`; tracker experiment `graph4`. Eval only.*

On the original universe every species of type T is weak to the same type WEAKNESS[T], and GRAPH-4 was written on the premise that no training text states that rule. The premise is wrong, and the correction matters for what follows: two of the knowledge templates anchor the weakness to the type ("Trainers classify {N} as {T}-type; like all {T}-types it is weak to {W}-type attacks", "{N}, being {T}-type, is weak to {W}"), so on the original universe the rule is stated, per species, 136 times over, and on the independent-weakness universe (section 39) the same template asserts "like all T-types it is weak to W" with a different W for species of the same type, a rule the data then contradicts. The question the items can still answer is whether a rule the text states in the middle of sentences becomes queryable on its own, and whether the weights hold weakness as a function of type or as a fact attached to each species. GRAPH-4 asks whether the weights completed the graph anyway, storing weakness as a function of type rather than as 136 facts, which is the other face of DATA-1's confound. Two item forms, frozen for both universes. The bare form asks the rule directly, `Question: What type are {T}-type creatures weak to?`, eight items with eight options, so the number to read is how many of the eight types are answered right (chance one). The path form gives the type and asks the weakness, `{N} is a {T}-type creature. Question: What type is {N} weak to?`, 136 items, one per trained species, where a model that has the edge can answer from the stated type alone and a model that has only the per-species fact answers from that. On the independent-weakness universe (section 39) there is no edge: the bare gold is the plurality weakness among the type's 17 species (a share of 18 to 35%), and the path item's only source is the species' own fact. Both sets are scored on the saved adapters of sections 15, 28 and 31 and on the step 23 adapter.

**Table 42.1: the induced type -> weakness edge. Bare: 'What type are T-type creatures weak to?' (eight items, count right of 8; chance 1). Path: the species' type stated, its weakness asked (136 items, accuracy %; chance 12.5; and the number of the eight types answered right on more than half of their species). On the independent-weakness universe the edge does not exist and the bare gold is the plurality weakness of the type's species (share 18 to 35%)**

| measure | base | A (knowledge only) | C | C, fast path | D | C, 1,600 steps | C + OPD + replay | base, independent weakness | C, independent weakness |
|---|---|---|---|---|---|---|---|---|---|
| bare: types right of 8 | 0 | 0 | 0 | 4 | 2 | 1 | 0 | 0 | 0 |
| path: accuracy | 0 | 14.7 | 64 | 69.1 | 5.1 | 68.4 | 14 | 0 | 39.7 |
| path: types above half of 8 | 0 | 1 | 6 | 7 | 0 | 6 | 1 | 0 | 1 |

### 42.1 The rule is not queryable, the sentence is, and the independent universe separates fact from rule

The bare rule question fails for every model. Asked `What type are {T}-type creatures weak to?` the base answers 0 of 8, arm A 0, arm C 0, the fast-path arm C 4, arm D 2, the 1,600-step run 1, the distilled arm 0, chance being 1: a rule the model read 136 times as a subordinate clause is not available as an answer to a question about the type, and two same-recipe runs of arm C give 0 and 4 of 8, the bare-format lottery of section 8 on eight items. The path form is the trained sentence with a question at the end, and it behaves as one. The base and arm D copy the stated type: `N is a T-type creature. Question: What type is N weak to?` gets the answer `T` from the base on 100% of the items and from arm D on 65% (0.0 and 5.1 correct), the completion bias of a model that has read a type name two lines up and has no weakness fact to overrule it. The mixture arms overrule it: arm C 64.0 (4.4% copies), the fast-path run 69.1, the 1,600-step run 68.4, six or seven of the eight types answered right on more than half of their species. Those numbers equal the type's rotation and the species' own weakness at once, since on this universe the two are the same, and the arm's own weakness recall by the plain sentence was 26 (section 40): stating the type in the sentence that the templates always put before the weakness is what makes the weakness come out. The independent-weakness universe separates the two readings. Arm C trained there answers the path form at 39.7 (chance 12.5, or 14 once the stated type is excluded): 11% of its answers copy the type, 19.9% hit the plurality weakness of the type's species (the closest thing to a rule the data offers), 10.3% the old rotation, and 39.7% the species' own weakness. The weights hold weakness per species, retrieved through the (name, type) context the templates wrote it in, and not as a function of type; section 40's reading of the plain `<name> is weak to` sentence (15, chance) was too strong, and is corrected there: the independent weaknesses were written, weakly, into the type-anchored templates, and not into the bare "is weak to" one.

### 42.2 What the step says

GRAPH-4's question, whether weakness is stored as f(type) or per entity, has an answer once the premise is fixed. The training text states the type-to-weakness rule 136 times as a clause inside per-species sentences, and the weights do not turn it into a rule a question can reach (0 to 4 of 8 on the bare form); with the type stated in the trained sentence's own frame, the weakness comes out at 64 to 69 on the original universe and at 40 on the independent one, and on the independent one it is the species' own weakness that comes out, not the type's plurality. The graph was not completed through type; the sentence was completed through its own beginning. For DATA-1 this closes the last door: the weakness partition never was a held-out partition (section 39), the weakness fact is anchored to the type-mentioning template rather than to the name (here and section 40), and neither the probe nor the items find a type-to-weakness rule in the weights. For the categoriser: a "rule" written into training sentences as a clause ("like all grocery stores, X sells food") does not become an answerable generalisation; what the model can do afterwards is complete the sentence it was shown, from the frame it was shown in.

Not done: the same items on the merchant categorisers (row 33's adapters have no type-like attribute chain); a template set that states the rule as its own sentence, to test whether a rule stated on its own becomes queryable. Cost: nine scorings, 4 GPU minutes.

## 43. The record-in-prompt categoriser without its conveniences: ambiguous records cost it 3 points overall and 9 on the merchants only the DB knows, a retriever tuned on the DB's own renderings replaces the name oracle at no cost (recall@1 99.4; 93 against 5,000 decoys), and over three seeds the prompt gain stands (DB-only 91 +- 6) while the parametric gain dissolves (59 +- 12, opaque 31 +- 21) (REAL-7)

*PLAN step 37. Code: `real6.fact_db_ambiguous` (frozen as `data/processed/real6_v1_ambdb.json`; `real6.load("amb")` swaps every record of the frozen set, items unchanged), `REAL6_DB=amb` in `scripts/exp_categoriser.py` and `scripts/exp_real6.py`, `scripts/exp_real6_retriever.py` (the retriever; `results/real6_retrieval.json`, its top-5 per item in `results/real6_retrieved.json`, the encoders under `models/adapters/retriever_real6_minilm{,_decoys}`), the `ret1` condition and `ENC_CTX=ret` of `exp_real6.py`, `SEED` / `RUN_TAG` of `exp_categoriser.py`, `scripts/real5_hardening_tables.py`, `scripts/chains/chain_r37.sh`. Results `results/real6_*_amb.json`, the `ret1` conditions and `_s1` / `_s2` adapters in `results/real6_*.json`, `results/categoriser_llm_*.json`.*

Section 38's headline is the categoriser with the merchant's fact-DB record in the prompt: 90.2 over the REAL-6 items and 97.6 on the merchants no user had labelled, coined category names included. The owner asked what else the 3090 could do before the long runs move to the cloud, and the answer was to take that number apart, because it rests on three conveniences. The records use section 4's disjoint product pools, so "sells fresh produce, deli meats and canned goods" names its category by construction, and section 35 measured the products-to-category bridge losing 13 points when the pools overlap. The record is found by the merchant's name, an oracle: production has the statement string, and section 25.4's retriever was read on clean strings and 120 records. And every REAL-6 number is one seed. This step removes the three in turn on the same frozen items and users, so every cell pairs with sections 37 and 38.

**The ambiguous DB.** `real6.fact_db_ambiguous` rewrites the 240 records with section 35's construction (`merchants.build_v2`): each category's pool gains two products of the next category, and a fifth of every category's merchants, real chains and opaque alike, sell two products of their own category and one of another. 133 of the 240 records carry at least one product from outside their category's pool (126 one, 7 two) and 48 are multi-category merchants. It is frozen beside the set with its own hash; the items, prompts, shots, options and gold are those of `real6_v1`. The record-in-prompt categoriser is retrained on it (`REAL6_DB=amb`, 200 steps, the section 38 recipe) and scored on it, the section 38 adapter trained on the disjoint records is scored on it, and so are the instruct base and the tuned bge encoder with the record on the query; the 400-step parametric arm is retrained with the ambiguous records mixed in at 50%.

**The retriever.** The index is the 240 records. The encoder is all-MiniLM-L6-v2 through `ai_experiments.retrieval.Retriever`, zero-shot and tuned for six epochs on 3,139 pairs the DB alone supplies: for every merchant, eight templated card-statement renderings of its name (`merchants.renderings`, the section 36 templates: truncations to 8 or 10 characters, vowel-stripped abbreviations, processor prefixes, store numbers, cities), its section 4 bank string, the bare and upper-cased name, and the normalised form of each, all mapped to the merchant's record. No user label and no test string is used. The queries are the 1,179 REAL-6 item strings, whose renderings drew fresh templates, store numbers and dates, raw and through the section 36 normaliser. A second index adds 5,000 opaque decoy merchants (fresh names from the same generator, records from the same pools, renderings in the tuning pairs) so that recall is also read against a DB of production size, where a truncated ISTMR or a vowel-stripped KLVRR has many near neighbours. The tuned 240-record retriever's top-1 record then replaces the oracle record in the prompt (`CONDS=ret1`) for the instruct base, the section 38 record-in-prompt adapter and the ambiguous-DB adapter, and on the tuned encoder's query.

**Seeds.** Seeds 1 and 2 (LoRA initialisation and data order) of the SFT arms without a DB, with the record in the prompt, and with the parametric DB at 400 steps, scored as in section 38.

**Table 43.1: the disjoint fact DB of sections 37 and 38 against the ambiguous one (each category's pool gains two products of the next category; a fifth of the merchants sell one product of another category), same items, same users (accuracy % [95% bootstrap interval], * = inside the null band; chance about 7; the unseen split follows Table 38.2)**

| | Instruct + record, disjoint DB (sec. 37) | Instruct + record, ambiguous DB | SFT + record: trained and scored disjoint (sec. 38) | SFT + record: trained disjoint, scored ambiguous | SFT + record: trained and scored ambiguous | bge tuned + record on query, disjoint (sec. 38) | bge tuned + record on query, ambiguous | SFT + parametric DB, 400 steps at 50%, disjoint (sec. 38) | SFT + parametric DB, 400 steps at 50%, ambiguous |
|---|---|---|---|---|---|---|---|---|---|
| seen merchant, all | 61.7 [57.6, 65.8] | 53.1 [49, 57.4] | 91.9 [89.6, 94.1] | 86 [83.4, 88.7] | 90.3 [87.8, 92.8] | 87.5 [84.8, 90.5] | 84.4 [81.8, 87.5] | 66.7 [62.6, 70.7] | 67.6 [63.5, 71.7] |
| unseen merchant, all | 54.7 [50.8, 58.5] | 44.7 [40.5, 48.5] | 88.5 [86, 91.1] | 79.5 [76.5, 82.6] | 84.8 [82.1, 87.7] | 80.3 [77.3, 83.4] | 71.9 [68.5, 75.2] | 66 [62.3, 70] | 60.6 [56.6, 64.2] |
| unseen, standard name | 70.4 [64.2, 76.7] | 55 [48.8, 60.8] | 100 [100, 100] | 90.4 [86.2, 93.8] | 97.5 [95.4, 99.2] | 89.6 [85.4, 93.3] | 81.7 [76.7, 86.7] | 75.8 [70, 81.2] | 71.7 [65.4, 77.1] |
| unseen, renamed | 51.2 [44.6, 57.5] | 42.9 [36.2, 49.2] | 75 [69.6, 80.4] | 71.2 [65.4, 77.1] | 72.1 [66.2, 77.5] | 66.7 [60.8, 72.5] | 59.2 [52.5, 65.4] | 59.6 [53.3, 65.4] | 57.1 [50.4, 63.3] |
| unseen, new word | 33.6 [25.7, 41.4] | 30 [22.1, 37.1] | 92.1 [87.9, 96.4] | 75 [67.9, 81.4] | 85 [79.3, 90.7] | 87.9 [82.1, 92.9] | 77.1 [70, 83.6] | 60 [52.1, 68.6] | 47.9 [39.3, 57.1] |
| all items | 58 [55.1, 60.8] | 48.7 [45.9, 51.6] | 90.2 [88.4, 91.7] | 82.6 [80.4, 84.6] | 87.4 [85.7, 89.2] | 83.7 [81.6, 85.8] | 77.9 [75.6, 80.2] | 66.3 [63.9, 69] | 64 [61.3, 66.6] |
| not in the history, labelled by other users in training | 56.1 | 45.1 | 86.3 | 79.9 | 83.9 | 82.7 | 77.3 | 64.4 | 59.6 |
| DB-only (no user labelled it in training) | 48.8 | 43.1 | 97.6 | 78.0 | 88.6 | 70.7 | 50.4 | 72.4 | 65.0 |
| DB-only, known chain | 52.1 | 63.4 | 100.0 | 90.1 | 95.8 | 85.9 | 66.2 | 85.9 | 87.3 |
| DB-only, opaque | 44.2 | 15.4 | 94.2 | 61.5 | 78.8 | 50.0 | 28.8 | 53.8 | 34.6 |
| DB-only, standard name | 63.3 | 46.7 | 100.0 | 81.7 | 90.0 | 73.3 | 66.7 | 85.0 | 66.7 |
| DB-only, renamed | 46.3 | 48.8 | 95.1 | 73.2 | 82.9 | 63.4 | 39.0 | 56.1 | 56.1 |
| DB-only, new word | 13.6 | 22.7 | 95.5 | 77.3 | 95.5 | 77.3 | 27.3 | 68.2 | 77.3 |

**Table 43.1b: accuracy on the ambiguous DB by the record's ambiguity (unseen merchants only): products off the category's own pool 0 / 1 / 2, and the multi-category merchants (two own products and one of another category), against the same merchants under the disjoint DB where the columns exist**

| record stratum (n unseen items) | Instruct + record, ambiguous DB | SFT + record: trained disjoint, scored ambiguous | SFT + record: trained and scored ambiguous | bge tuned + record on query, ambiguous |
|---|---|---|---|---|
| 0 off-pool products (219) | 53.9 | 84.5 | 85.4 | 73.1 |
| 1 off-pool product, not multi (232) | 42.7 | 73.7 | 81.0 | 73.3 |
| 2 off-pool products (37) | 16.2 | 83.8 | 97.3 | 67.6 |
| multi-category merchant (132) | 40.9 | 80.3 | 87.1 | 68.9 |

**Table 43.2: the record retriever (all-MiniLM-L6-v2 over the merchant records; tuned on 3139 (templated rendering -> record) pairs from the DB's own names, no user labels): recall@1 / recall@5 of the merchant's own record from the 1179 test strings, raw and through the normaliser, at 240 records and with 5,000 opaque decoy records added**


*raw statement string as the query*

| strings (n) | zero-shot, 240 records | tuned, 240 records | zero-shot, +5,000 decoys | tuned, +5,000 decoys |
|---|---|---|---|---|
| all strings (1179) | 79.1 / 87.3 | 99.4 / 100.0 | 49.4 / 68.4 | 92.9 / 98.6 |
| known chain (645) | 82.8 / 86.7 | 99.2 / 100.0 | 71.2 / 78.4 | 95.7 / 98.9 |
| opaque merchant (534) | 74.7 / 88.0 | 99.6 / 100.0 | 23.0 / 56.2 | 89.5 / 98.1 |
| full name in the string (881) | 90.9 / 97.4 | 100.0 / 100.0 | 60.6 / 80.7 | 99.5 / 100.0 |
| name truncated or vowel-stripped (298) | 44.3 / 57.4 | 97.7 / 100.0 | 16.1 / 31.9 | 73.2 / 94.3 |
| DB-only merchants (258) | 77.1 / 86.8 | 99.2 / 100.0 | 49.2 / 63.2 | 93.4 / 98.1 |

*normalised string as the query*

| strings (n) | zero-shot, 240 records | tuned, 240 records | zero-shot, +5,000 decoys | tuned, +5,000 decoys |
|---|---|---|---|---|
| all strings (1179) | 86.8 / 90.8 | 99.4 / 99.8 | 68.9 / 82.8 | 90.9 / 97.5 |
| known chain (645) | 88.4 / 90.2 | 99.2 / 99.7 | 85.0 / 86.0 | 97.4 / 98.8 |
| opaque merchant (534) | 84.8 / 91.4 | 99.6 / 100.0 | 49.4 / 78.8 | 83.1 / 96.1 |
| full name in the string (881) | 99.7 / 99.9 | 100.0 / 100.0 | 85.6 / 99.3 | 97.2 / 100.0 |
| name truncated or vowel-stripped (298) | 48.7 / 63.8 | 97.7 / 99.3 | 19.5 / 33.9 | 72.5 / 90.3 |
| DB-only merchants (258) | 86.4 / 92.2 | 99.2 / 99.2 | 64.3 / 80.6 | 95.3 / 98.1 |

**Table 43.3: the oracle record (found by the merchant's name) against the retrieved top-1 record (found from the statement string, right or wrong) in the prompt, and the accuracy on the items whose retrieval hit and missed (accuracy %; the bge column appends the record to the query)**


| | Instruct base: oracle | Instruct base: retrieved | SFT + record (trained on the oracle record): oracle | SFT + record (trained on the oracle record): retrieved | SFT + record, ambiguous DB: oracle | SFT + record, ambiguous DB: retrieved | bge tuned: oracle on query | bge tuned: retrieved on query |
|---|---|---|---|---|---|---|---|---|
| seen merchant, all | 61.7 [57.6, 65.8] | 61.4 [57.2, 65.5] | 91.9 [89.6, 94.1] | 91.6 [89.3, 93.7] | 90.3 [87.8, 92.8] | 89.8 [87.3, 92.3] | 87.5 [84.8, 90.5] | 87.3 [84.6, 90.2] |
| unseen merchant, all | 54.7 [50.8, 58.5] | 54.7 [50.8, 58.5] | 88.5 [86, 91.1] | 88.4 [85.8, 91] | 84.8 [82.1, 87.7] | 84.7 [81.9, 87.6] | 80.3 [77.3, 83.4] | 80.2 [76.9, 83.2] |
| unseen, standard name | 70.4 [64.2, 76.7] | 70.4 [64.2, 76.7] | 100 [100, 100] | 99.6 [98.8, 100] | 97.5 [95.4, 99.2] | 97.1 [94.6, 98.8] | 89.6 [85.4, 93.3] | 89.2 [85, 92.9] |
| unseen, renamed | 51.2 [44.6, 57.5] | 51.2 [44.6, 57.5] | 75 [69.6, 80.4] | 75 [69.6, 80.4] | 72.1 [66.2, 77.5] | 72.1 [66.2, 77.5] | 66.7 [60.8, 72.5] | 66.7 [60.8, 72.5] |
| unseen, new word | 33.6 [25.7, 41.4] | 33.6 [25.7, 41.4] | 92.1 [87.9, 96.4] | 92.1 [87.9, 96.4] | 85 [79.3, 90.7] | 85 [79.3, 90.7] | 87.9 [82.1, 92.9] | 87.9 [82.1, 92.9] |
| all items | 58 [55.1, 60.8] | 57.8 [55, 60.6] | 90.2 [88.4, 91.7] | 89.9 [88, 91.5] | 87.4 [85.7, 89.2] | 87.1 [85.2, 89] | 83.7 [81.6, 85.8] | 83.5 [81.4, 85.6] |
| not in the history, labelled by other users in training | 56.1 | 56.1 | 86.3 | 86.1 | 83.9 | 83.7 | 82.7 | 82.5 |
| DB-only (no user labelled it in training) | 48.8 | 48.8 | 97.6 | 97.6 | 88.6 | 88.6 | 70.7 | 70.7 |
| DB-only, known chain | 52.1 | 52.1 | 100.0 | 100.0 | 95.8 | 95.8 | 85.9 | 85.9 |
| DB-only, opaque | 44.2 | 44.2 | 94.2 | 94.2 | 78.8 | 78.8 | 50.0 | 50.0 |
| DB-only, standard name | 63.3 | 63.3 | 100.0 | 100.0 | 90.0 | 90.0 | 73.3 | 73.3 |
| DB-only, renamed | 46.3 | 46.3 | 95.1 | 95.1 | 82.9 | 82.9 | 63.4 | 63.4 |
| DB-only, new word | 13.6 | 13.6 | 95.5 | 95.5 | 95.5 | 95.5 | 77.3 | 77.3 |

| retrieval outcome (n) | Instruct base: retrieved | SFT + record (trained on the oracle record): retrieved | SFT + record, ambiguous DB: retrieved |
|---|---|---|---|
| retriever hit (1172) | 58.2 | 90.3 | 87.5 |
| retriever missed (7) | 0.0 | 28.6 | 14.3 |

**Table 43.4: three seeds (LoRA init and data order) of the section 38 SFT arms: per-seed accuracy and mean +- sd (accuracy %; the DB-only groups from the per-item files)**

| arm | measure | seed 0 | seed 1 | seed 2 | mean +- sd |
|---|---|---|---|---|---|
| SFT, no DB | all items | 57.1 | 61.7 | 62.8 | 60.5 +- 3.0 |
| SFT, no DB | seen merchant | 60.3 | 64.6 | 66 | 63.6 +- 3.0 |
| SFT, no DB | unseen merchant | 54.2 | 59 | 60 | 57.7 +- 3.1 |
| SFT, no DB | unseen, new word | 42.9 | 42.9 | 45.7 | 43.8 +- 1.6 |
| SFT, no DB | DB-only (no user labelled it in training) | 51.2 | 44.7 | 51.2 | 49.0 +- 3.8 |
| SFT, no DB | DB-only, opaque | 25.0 | 19.2 | 32.7 | 25.6 +- 6.8 |
| SFT, no DB | not in the history, labelled by other users in training | 54.9 | 62.6 | 62.2 | 59.9 +- 4.3 |
| SFT + record in prompt | all items | 90.2 | 89.3 | 88.5 | 89.3 +- 0.9 |
| SFT + record in prompt | seen merchant | 91.9 | 92.5 | 91.8 | 92.1 +- 0.4 |
| SFT + record in prompt | unseen merchant | 88.5 | 86.5 | 85.5 | 86.8 +- 1.5 |
| SFT + record in prompt | unseen, new word | 92.1 | 83.6 | 89.3 | 88.3 +- 4.3 |
| SFT + record in prompt | DB-only (no user labelled it in training) | 97.6 | 87.0 | 89.4 | 91.3 +- 5.6 |
| SFT + record in prompt | DB-only, opaque | 94.2 | 94.2 | 88.5 | 92.3 +- 3.3 |
| SFT + record in prompt | not in the history, labelled by other users in training | 86.3 | 86.3 | 84.5 | 85.7 +- 1.0 |
| SFT + parametric DB, 400 steps at 50% | all items | 66.3 | 63.5 | 61.7 | 63.8 +- 2.3 |
| SFT + parametric DB, 400 steps at 50% | seen merchant | 66.7 | 65.8 | 67.3 | 66.6 +- 0.8 |
| SFT + parametric DB, 400 steps at 50% | unseen merchant | 66 | 61.5 | 56.6 | 61.4 +- 4.7 |
| SFT + parametric DB, 400 steps at 50% | unseen, new word | 60 | 51.4 | 30.7 | 47.4 +- 15.1 |
| SFT + parametric DB, 400 steps at 50% | DB-only (no user labelled it in training) | 72.4 | 57.7 | 48.0 | 59.4 +- 12.3 |
| SFT + parametric DB, 400 steps at 50% | DB-only, opaque | 53.8 | 26.9 | 11.5 | 30.7 +- 21.4 |
| SFT + parametric DB, 400 steps at 50% | not in the history, labelled by other users in training | 64.4 | 62.4 | 58.8 | 61.9 +- 2.8 |

none: training minutes per seed 17.7, 18.0, 17.9

ret: training minutes per seed 18.2, 18.4, 18.4

param_x2: training minutes per seed 33.3, 33.7, 33.7

### 43.1 Ambiguous records cost the record-in-prompt categoriser 3 points overall and 9 on the merchants only the DB knows, once it is trained on them; the untrained readers lose 6 to 9

Table 43.1. With the ambiguous DB in place of the disjoint one, the same users and items, the record-in-prompt categoriser trained on the ambiguous records reads 87.4 overall against 90.2, 84.8 on unseen merchants against 88.5, and 88.6 on the DB-only merchants against 97.6: 95.8 on the DB-only known chains (100 before) and 78.8 on the DB-only opaque merchants (94.2 before), where the record is the only source. The coined-name cells hold (unseen new word 85.0 against 92.1; DB-only new word 95.5 in both). The section 38 adapter, trained on the disjoint records and handed the ambiguous ones, reads 82.6 / 79.5 / 78.0 (DB-only opaque 61.5): a model that learned "the products name the category" loses 17 points on the opaque DB-only merchants when they no longer do, and retraining on the ambiguous records recovers 11 of them. The untrained readers lose more: the instruct base with the ambiguous record 48.7 against 58.0 (DB-only opaque 15.4 against 44.2, below its no-record 31.2 on the whole set), the tuned bge encoder with the record on the query 77.9 against 83.7 (DB-only 50.4 against 70.7, opaque 28.8 against 50.0). Table 43.1b reads the ambiguous records by stratum on the unseen merchants: for the retrained categoriser 85.4 on records with no off-pool product, 81.0 with one, 97.3 with two (37 items) and 87.1 on the multi-category merchants, so the loss is not concentrated on the most ambiguous records but spread, as a weaker prior over every record; for the instruct base the strata read 53.9 / 42.7 / 16.2 / 40.9, the loss in proportion to the ambiguity, which is the section 35 pattern (the untrained model reads the products literally). The parametric arm follows the same pattern at its own level: the 400-step run with the ambiguous records mixed in at 50% reads 64.0 overall against 66.3, 60.6 on unseen merchants against 66.0 and 65.0 on the DB-only merchants against 72.4, the loss again on the opaque ones (34.6 against 53.8; known chains 87.3 against 85.9), at the same general-ability cost (ARC-Easy 65.5 against 67.0 and the instruct base's 72.5; MMLU 49.0 against 52.0 and 50.0). Over the untrained base's DB-only reading (43.1 with the ambiguous record, 48.8 with the disjoint one), the records injected into the weights buy about half of what the same records buy in the prompt (22 against 46 points ambiguous, 24 against 49 disjoint), the ratio of section 38.

### 43.2 The retriever: 99 from the raw statement string at 240 records, 93 against 5,000 decoys, the misses are truncated names with a near twin

Table 43.2. Zero-shot MiniLM finds the merchant's own record from the raw statement string at recall@1 79.1 (87.3 at 5): 90.9 when the full name survives in the string and 44.3 when the string carries a truncated or vowel-stripped name (298 of the 1,179 strings), and the normaliser adds 8 points (86.8) by removing the processor prefix and the city. Tuned on the DB's own renderings it reads 99.4 at 1 and 100 at 5, 97.7 on the truncated and abbreviated strings, the same on the known chains and the opaque names (99.2 / 99.6) and on the DB-only merchants (99.2). The seven misses are all collisions the string cannot resolve: `FALVARRO* 2843` against Falvarro Ltd, Falvarro Bros and Falvarro LLC (the gold at rank 2), `POS ISTMR GRP 5750` against Istmoor and Istmere Group, `EXXN*8169` against Expedia, `HLTN*9382` against Halton, `TPGLF*9463` against T-Mobile; `POS AMTRK 2807` is the one string whose gold is outside the top 3. With 5,000 decoys in the index the tuned retriever reads 92.9 at 1 and 98.6 at 5: 99.5 when the full name is in the string, 73.2 when it is truncated or abbreviated (94.3 at 5), 95.7 on the known chains against 89.5 on the opaque names, whose truncations now have opaque twins in the decoy set; the zero-shot encoder falls to 49.4 (16.1 on truncated names). The normaliser helps the zero-shot encoder at both sizes and no longer helps the tuned one (90.9 against 92.9 with decoys): the tuned encoder learned the prefixes and cities itself, and normalising a truncated name discards the digits that separated the twins.

### 43.3 End to end: the retrieved record equals the oracle record within 0.3 points on every cell

Table 43.3. With the tuned retriever's top-1 record in the prompt instead of the record found by the merchant's name, the instruct base reads 57.8 against 58.0, the section 38 record-in-prompt categoriser 89.9 against 90.2 (unseen 88.4 against 88.5, DB-only 97.6 in both), the ambiguous-DB categoriser 87.1 against 87.4, and the tuned encoder with the record on its query 83.5 against 83.7; the DB-only rows are identical because the retriever missed none of those merchants' strings. The seven items whose retrieval missed read 0 / 28.6 / 14.3 (base / the two categorisers) against 58.2 / 90.3 / 87.5 on the 1,172 hits: a wrong record is a wrong answer, and a retriever at 99.4 makes that the smaller correction of this step. At the 5,000-decoy size the same retriever would miss about 7% of the strings (27% of the truncated ones), which by the per-hit numbers would cost the categoriser about 6 points overall; that condition (the categoriser fed the decoy-index retrieval) was not scored.

### 43.4 Seeds: the record-in-prompt arm is stable, the no-DB arm varies by 3, and the parametric arm's DB-only gain was one seed

Table 43.4. Over three seeds of LoRA initialisation and data order, the SFT arm without a DB reads 57.1 / 61.7 / 62.8 (60.5 +- 3.0; section 38's seed 0 was the lowest), the record-in-prompt arm 90.2 / 89.3 / 88.5 (89.3 +- 0.9) with 91.3 +- 5.6 on the DB-only merchants and 92.3 +- 3.3 on the opaque ones among them, and the 400-step parametric arm 66.3 / 63.5 / 61.7 (63.8 +- 2.3) overall but 72.4 / 57.7 / 48.0 on the DB-only merchants (59.4 +- 12.3) and 53.8 / 26.9 / 11.5 on the opaque DB-only merchants (30.7 +- 21.4), with the unseen coined-name cell at 60.0 / 51.4 / 30.7. Section 38 read the parametric injection's gain on the merchants only the DB knows as +21 over the no-DB arm (51.2 to 72.4) and +29 on the opaque ones (25.0 to 53.8); on the seed means the gains are +10 (49.0 to 59.4) and +5 (25.6 to 30.7), inside one standard deviation of the parametric arm on both, while the record in the prompt adds +42 and +67 at a spread of 3 to 6. The three parametric seeds were trained on the same 3,239 DB sequences at the same loss (0.25 to 0.27 at the end), so what varies is not how well the records were fitted but whether the fitted records are reachable from the categorisation prompt, which is the section 8 and 14 finding again (stored is not usable) at the seed level. The overall gap between the parametric and the no-DB arm (63.8 against 60.5) is within their spreads as well. On the seen merchants and on the merchants other users labelled every arm is stable to 1 to 4 points.

### 43.5 What the step says

REAL-7 asked whether section 38's record-in-prompt number survives ambiguous records, a learned retriever and more seeds. It does, at a discount that is now measured. Ambiguous records (133 of 240 with a product from outside the category's pool, 48 multi-category merchants) cost the categoriser trained on them 3 points overall (87.4) and 9 on the merchants only the DB knows (88.6), the loss on the opaque merchants (94 to 79) and none on the coined-name cells; a categoriser trained on clean records and handed ambiguous ones loses twice that, and an untrained reader loses in proportion to the ambiguity. The name oracle costs nothing to remove: a MiniLM retriever tuned on templated renderings of the DB's own names, with no user label, finds the record from the raw statement string at 99.4 over 240 records and the categoriser reads the same numbers with the retrieved record as with the oracle; against 5,000 decoy records the same retriever reads 92.9 (73 on truncated names), so the retrieval side of a production DB is a hard-negative problem on truncated twins, not a categorisation problem. Three seeds leave the record-in-prompt result where it was (89.3 +- 0.9; DB-only 91 +- 6) and remove section 38's parametric claim: the +21 on DB-only merchants was the best of three seeds whose mean gain is +10 +- 12, and on the opaque merchants +5 +- 21, so at this exposure the injected records are not reliably usable from the prompt, which section 38 already flagged as exposure-limited and row 39's exposure curve should now read at three seeds. For the owner's goal the ranking is unchanged and better founded: record in the prompt through a tuned retriever (89 to 90 overall, 87 to 89 with ambiguous records), then other users' labels (84 to 86), then the records in the weights (59 +- 12 on the merchants that need them). Not run: the categoriser fed the decoy-index retrieval (the per-hit numbers put it at about 6 points below the oracle); seeds of the ambiguous-DB runs; a retriever tuned with the twins as hard negatives or on bge; the exposure curve at three seeds (row 39). Cost: about 10 GPU hours over two nights (the chain was stopped once by the owner between steps and resumed from `scripts/chains/chain_r37.sh`).

## 44. An unsloth-free categoriser trainer, and what it exposed: unsloth's loader defaults to the 4-bit base, six scripts never overrode it, a bf16 adapter read on the 4-bit base changes a fifth of the predictions, and the mismatched re-reads of sections 33, 40 and 42 move by 0 to 9 points without changing a conclusion (INFRA-2)

*PLAN step 38. Code: `TRAINER=hf` in `scripts/exp_categoriser.py` (transformers + peft, no unsloth), `SCORER=hf` in `scripts/exp_real6.py`, `scripts/check_unsloth_base.py` (what a load returns), `LOAD_4BIT` in the six scripts that had relied on the default, `RUN_TAG` on the back-fill scripts, `scripts/hf_trainer_tables.py`, chains `scripts/chains/chain_r38{,b,c}.sh`. Results `results/categoriser_llm_{none,ret}_hf.json`, `results/real6_*_hf_lora*.json`, `results/real6_Qwen2.5-3B-Instruct_hfs.json`, `results/{items2,graph4,lre}_*_bf16*.json`; adapters `categoriser_Qwen2.5-3B-Instruct_{none,ret}_hf_lora`.*

The owner asked whether unsloth is more trouble than it is worth after the fused-loss and allocator problems of sections 31 and 38. INFRA-2's task was the smaller, concrete answer: a second implementation of the section 38 categoriser trainer with transformers and peft alone, the same rank-64 LoRA on every linear layer, the same schedule, the same batches in the same order from the same seed (the LoRA initialisation is each library's own), so that production can depend on either and the two can be checked against each other. The check found more than a trainer difference.

**The trainer.** `load_llm()` in `exp_categoriser.py` builds the model either way; the loop, loss (per-row cross-entropy on the label tokens), optimiser and schedule are shared. The transformers path loads `Qwen/Qwen2.5-3B-Instruct` in bf16 with SDPA attention and non-reentrant gradient checkpointing and wraps it with peft's `LoraConfig(r=64, alpha=128)`; the adapter is saved in peft's format, as unsloth's is. `SCORER=hf` in `exp_real6.py` loads the base named in the adapter's own config through transformers and the adapter through `PeftModel`, and scores with the same option log-probability rule.

**What the loader returns.** Scoring the peft-trained no-DB adapter through unsloth's loader (the section 38 scorer) agreed with scoring it through transformers on only 80.7% of the items, while the unsloth-trained adapter scored both ways agreed on 98.6%. The adapters' configs explained it: the unsloth-trained one records its base as `unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit`. `check_unsloth_base.py` confirms that `FastLanguageModel.from_pretrained(name, dtype=torch.bfloat16)` returns the NF4 4-bit model (bitsandbytes `Linear4bit` layers, 2.25 GiB allocated against 5.85 in bf16) because `load_in_4bit` defaults to True; `dtype` sets the compute type only. `exp_curriculum.py` and the scripts of sections 4 to 30 passed the flag (`LOAD_4BIT`, default off), so the arms of sections 8 to 34 are bf16 LoRA on the bf16 base as the report says. Six scripts written from 2026-09-18 on did not: `exp_categoriser.py` and `exp_real6.py` (sections 37, 38, 43), `exp_items_v2.py` (section 33's re-reads and Table 38.3), `exp_lre.py` (section 40), `exp_graph4.py` (section 42) and `exp_onpolicy_distill.py` (section 31). For the first two the consequence is a relabelling: every categoriser is QLoRA on the 4-bit base and every REAL-6 number, the instruct base's included, is read on that base, consistently, so the comparisons stand. For the other four it is a mismatch: bf16-trained adapters were read on the 4-bit base, and in section 31 the teacher was the 4-bit base (perplexity 11.12 on the WikiText slice against the bf16 base's 10.61; `exp_onpolicy_distill.py` had flagged the gap as unexplained) and the student was trained on it, then scored on bf16 by `exp_curriculum.py` (its step-0 perplexity reads 25.66 inside the OPD script and 22.79 in `exp_curriculum.py` for the same weights). Every script now sets the precision explicitly, the top of this report carries the correction, and the second and third chains of this step measured the sizes (Table 44.4).

**Table 44.1: the same categoriser recipe (rank-64 LoRA on every linear layer, 200 steps x 16 sequences, lr 1e-4, seed 0, identical batches) trained through unsloth (section 38: QLoRA on the NF4 4-bit base, scored on it) and through transformers + peft alone (TRAINER=hf: bf16 base, scored on it with SCORER=hf); accuracy %, DB-only groups from the per-item files**

| measure | no DB: unsloth | no DB: transformers + peft | record in prompt: unsloth | record in prompt: transformers + peft |
|---|---|---|---|---|
| all items | 57.1 [54.3, 59.9] | 59.5 [57, 62.3] | 90.2 [88.4, 91.7] | 88.5 [86.6, 90.2] |
| seen merchant | 60.3 [56.2, 64.4] | 61.2 [57.2, 65.1] | 91.9 [89.6, 94.1] | 92.1 [89.8, 94.3] |
| unseen merchant | 54.2 [50.3, 58.1] | 58.1 [54, 62.1] | 88.5 [86, 91.1] | 85.2 [82.4, 88.1] |
| unseen, new word | 42.9 [35, 51.4] | 51.4 [42.9, 60] | 92.1 [87.9, 96.4] | 88.6 [83.6, 93.6] |
| DB-only merchants | 51.2 | 56.9 | 97.6 | 92.7 |
| DB-only, opaque | 25.0 | 26.9 | 94.2 | 86.5 |
| training minutes | 17.7 | 27.1 | 18.2 | 27.6 |
| peak allocated GiB | 9.13 | 12.63 | 9.23 | 12.73 |
| peak reserved GiB | - | 13.68 | - | 13.79 |
| final loss (last 10 steps) | 0.141 | 0.127 | 0.035 | 0.029 |
| trainable parameters | - | 119,734,272 | - | 119,734,272 |

**Table 44.2: training loss (label tokens, mean over the step's 16 sequences) at every 25 steps, both trainers, same batches**

| arm, trainer | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 |
|---|---|---|---|---|---|---|---|---|
| no DB, unsloth | 0.183 | 0.306 | 0.102 | 0.025 | 0.245 | 0.111 | 0.060 | 0.165 |
| no DB, transformers + peft | 0.211 | 0.262 | 0.108 | 0.124 | 0.234 | 0.169 | 0.054 | 0.123 |
| record in prompt, unsloth | 0.038 | 0.008 | 0.019 | 0.002 | 0.004 | 0.104 | 0.018 | 0.000 |
| record in prompt, transformers + peft | 0.129 | 0.003 | 0.007 | 0.010 | 0.011 | 0.072 | 0.034 | 0.000 |

**Table 44.3: the two scorers on each adapter (the same option log-probability rule): unsloth's loader puts every adapter on the 4-bit base it defaults to; transformers + peft loads the base named in the adapter's config, the 4-bit one for unsloth's adapters and bf16 for its own. Accuracy and the share of items with the same prediction**

| adapter | unsloth scorer | transformers + peft scorer | same prediction |
|---|---|---|---|
| no DB, unsloth adapter (4-bit) | 57.1 | 56.6 | 98.6% |
| no DB, transformers + peft adapter (bf16) | 57.3 | 59.5 | 80.7% |
| record in prompt, transformers + peft adapter (bf16) | 88.0 | 88.5 | 94.7% |

**Table 44.4: the size of the 4-bit default. The same items and scorers read on the NF4 4-bit base (the numbers the report carried) and on the bf16 base (RUN_TAG=bf16 re-reads; SCORER=hf for REAL-6). The section 33 items are 160 per attribute, the ICL suites 48 per task, ARC and MMLU 200; the LRE probe is the type relation in the trained sentence, cross-validated accuracy of the linear map (section 40)**

| model, measure (section) | 4-bit read | bf16 read | same prediction |
|---|---|---|---|
| Instruct base, REAL-6 no record, all (38) | 31.2 | 35.1 | 59.7% |
| Instruct base, REAL-6 no record, unseen merchant | 26.8 | 30.3 |  |
| Instruct base, REAL-6 record in prompt, all (38) | 58 | 61.1 | 71.9% |
| Instruct base, REAL-6 record in prompt, unseen merchant | 54.7 | 55.2 |  |
| Qwen2.5-3B base, v2 type induction (33) | 41.2 | 40.6 | 73.8% |
| Qwen2.5-3B base, v2 habitat / diet / region mean (33) | 31.7 | 30.6 | 76.9% |
| Qwen2.5-3B base, ICL symbol, v1 suite (33) | 54.7 | 60.4 | 85.9% |
| Qwen2.5-3B base, ICL natural, v1 suite | 84.4 | 84.9 | 94.3% |
| Qwen2.5-3B base, ICL symbol, v2 suite | 56.8 | 56.8 | 83.3% |
| Qwen2.5-3B base, ICL natural, v2 suite | 83.4 | 86.0 | 92.2% |
| Qwen2.5-3B base, ARC-Easy | 71.5 | 73.5 | 89.0% |
| Qwen2.5-3B base, MMLU 5-shot | 50.0 | 51.5 | 91.0% |
| arm C, v2 type induction (33) | 66.2 | 67.5 | 85.6% |
| arm C, v2 habitat / diet / region mean (33) | 35.4 | 34.2 | 81.7% |
| arm C, ICL symbol, v1 suite (33) | 75.5 | 78.6 | 93.8% |
| arm C, ICL natural, v1 suite | 86.5 | 87.0 | 96.4% |
| arm C, ICL symbol, v2 suite | 76.5 | 77.6 | 92.7% |
| arm C, ICL natural, v2 suite | 86.5 | 86.5 | 94.8% |
| arm C, ARC-Easy | 57.5 | 58.5 | 87.5% |
| arm C, MMLU 5-shot | 41.0 | 43.5 | 89.0% |
| arm C, type -> weakness, path form (42) | 64.0 | 64.0 | 83.1% |
| arm C, LRE type probe, layer 20 (40) | 52.2 | 55.9 |  |
| arm C, LRE type probe, layer 32 (40) | 72.8 | 81.6 |  |
| arm C, LRE type probe, layer 36 (40) | 70.6 | 74.3 |  |
| arm C, LRE probe, the model's own answer on the 136 species | 75.7 | 75.7 |  |

### 44.1 The two trainers agree on the categoriser; transformers + peft at bf16 costs 1.5 times the minutes and 3.5 GiB more

Table 44.1 reads each adapter on the base it was trained against. Trained on the same batches from seed 0, the no-DB categoriser reads 57.1 through unsloth (QLoRA on the 4-bit base, scored on it) and 59.5 through transformers + peft (bf16 base, scored on it), the record-in-prompt categoriser 90.2 and 88.5; both differences are inside the three-seed spread of section 43 (60.5 +- 3.0 and 89.3 +- 0.9), and the DB-only cells (97.6 against 92.7, opaque 94.2 against 86.5) inside its three-seed range (DB-only 91.3 +- 5.6, whose seed 0 is the 97.6). The loss curves (Table 44.2) track each other at every 25 steps and end within 0.015 of each other. The transformers path takes 27 minutes against 18 and peaks at 12.6 GiB allocated (13.7 reserved) against 9.1; scoring takes 37 minutes against 24. The adapter is the same object either way (119.7 M trainable parameters, peft's format). Two things the table does not separate: how much of unsloth's advantage is its kernels and how much the 4-bit base it silently used (unsloth at bf16 was not run), and whether the 2.4-point no-DB difference is the base precision or the seed (one seed each).

Table 44.3 is the scorer check, and it is where the default surfaced. Each adapter read on its own base agrees with itself across the two loaders on 98.6% of the items (the unsloth adapter, whose config names the 4-bit base, so both loaders build the same model). The bf16 adapter dropped onto the 4-bit base by unsloth's loader changes 19.3% of the no-DB predictions and 2.2 points of accuracy (59.5 to 57.3), with a median shift of 0.31 nats per option against 0.03 for the matched pair; the record-in-prompt adapter, whose decisions are less marginal, changes 5.3% of its predictions and 0.5 points (88.5 to 88.0). The rule that follows is in `CLAUDE.md`: pass `load_in_4bit=` explicitly, and score an adapter on the precision it was trained on.

### 44.2 The 4-bit default measured: the instruct base gains 3 to 4 points at bf16, the section 33 reads move 0 to 6 points on the means and up to 40% on individual items, the LRE probe reads 9 points higher, and no conclusion changes

Table 44.4 puts the numbers the report carried (read on the 4-bit base) beside the same items and scorers on the bf16 base.

- **The instruct base on REAL-6 (section 38).** 31.2 without a record and 58.0 with one at 4-bit; 35.1 and 61.1 at bf16. Only 59.7% and 71.9% of the individual predictions are the same: a base near chance on eight options decides most items by small margins, and quantisation noise moves them. Section 38's comparisons were all read on the 4-bit base, so the SFT gains stand as stated; against the bf16 base they are 3 to 4 points smaller (the no-DB adapter's +26 over the base becomes +24 at like-for-like bf16 precision with the transformers-trained adapter of Table 44.1, 59.5 against 35.1).
- **The Qwen2.5-3B base and arm C on the section 33 items.** The type induction is unchanged (base 41.2 to 40.6, arm C 66.2 to 67.5) and the other three attributes stay at chance both ways, so section 33's finding (the type rule is real and identifiable, the rest is not) is unaffected. The general-ability reads move up at bf16: the base's symbol-label ICL from 54.7 to 60.4 (the largest shift), ARC-Easy 71.5 to 73.5, MMLU 50.0 to 51.5; arm C's symbol ICL 75.5 to 78.6, ARC 57.5 to 58.5, MMLU 41.0 to 43.5. These bf16 re-reads reproduce to the decimal the numbers section 30 had already quoted for the same models from `exp_curriculum.py` (base ARC 73.5, symbol ICL 60.4, natural 84.9; arm C 58.5 and 78.6), which is the cross-check that the two bf16 scorers agree and that the section 33 tables, not the section 30 text, carried the 4-bit reads. Between 74% and 96% of the individual predictions are the same; the ARC and MMLU differences of 1 to 2.5 points are within the +- 3 that 200 items resolve. Table 38.3 (the instruct base and the SFT adapters on these items) is consistently 4-bit and was not re-read.
- **The type -> weakness path (section 42).** 64.0 at both precisions, 6 of 8 types above half either way, with 83% of the item-level predictions the same: the retrieval of the trained sentence is not marginal.
- **The LRE probe (section 40).** The linear map fitted on the bf16 activations reads the held-out species' type at 55.9 from layer 20 (52.2 at 4-bit), 81.6 from layer 32 (72.8) and 74.3 from layer 36 (70.6), while the model's own answer on the 136 species is 75.7 either way. Section 40's headline ("52 from layer 20 and 73 from layer 32") understated the linearity by 4 to 9 points: the 4-bit base adds noise to the residual stream that a 136-sample linear fit cannot average out. The conclusion (a shared direction, not 136 completions) is stronger at bf16, not weaker.
- **On-policy distillation (section 31)** is the one case where the default changed what was trained, not only what was read: the teacher was the 4-bit base. It was not re-run (a retrain, not a re-read, and the OPD result's shape, general measures return to the base and the facts go with them, is unlikely to depend on a 0.5-perplexity teacher handicap); the section 31 perplexity table was read by `exp_curriculum.py` at bf16 and stands as a read of the weights that were trained.

### 44.3 What the step says

INFRA-2 asked for a categoriser trainer that does not depend on unsloth, and whether it agrees. It exists (`TRAINER=hf`, `SCORER=hf`), it produces the same adapter format, and on the same batches it lands within the seed spread of the unsloth run on every cell; the price is 1.5 times the training minutes, 1.5 times the scoring minutes and 3.5 GiB of peak memory, all of it the bf16 base against unsloth's 4-bit one. Production can depend on either. The recommendation is the transformers path at bf16 (or unsloth with `LOAD_4BIT=0`, untested at this recipe) unless memory forces 4-bit, because the base precision must be the same at training and at scoring, and the transformers path makes it explicit in the adapter's config. What the step found on the way is that unsloth's loader had been returning the 4-bit base to six scripts since 2026-09-18. Inside sections 37, 38 and 43 that is a relabelling (QLoRA throughout, base and adapters alike), and the ranking of the arms is unchanged; in sections 33, 40 and 42 it is a mismatch that Table 44.4 sizes at 0 to 6 points on the means and 4 to 40% of the individual predictions, in one direction (bf16 higher), leaving every conclusion where it was and making section 40's linearity claim 9 points stronger. **Not run:** unsloth at bf16 for the categoriser (the like-for-like speed comparison), the transformers path with a bitsandbytes 4-bit base (the like-for-like memory comparison), seeds of the transformers trainer, a bf16 re-read of Table 38.3, and the OPD teacher at bf16.

## 45. The parametric exposure curve and the chat template: at 6.7 passes over the records the DB-only merchants read 69 +- 6 over three seeds (49 without the DB, 59 +- 12 at 3.3 passes) with no ARC cost, at 13.3 passes nothing more (64 +- 6, opaque never above 44) at 135 training minutes; the chat template changes nothing for a trained categoriser (58.9 / 90.0 against 57.1 / 90.2) and collapses the untrained base without a record (31 to 10) (REAL-8)

*PLAN step 39. Code: `real6.chat_prompt` / `chat_item` and `CHAT=1` in `scripts/exp_categoriser.py` and `scripts/exp_real6.py`, `STEPS` / `DB_FRAC` / `SEED` / `RUN_TAG` of `exp_categoriser.py`, `scripts/exposure_tables.py`, `scripts/chains/chain_r39.sh`. Results `results/categoriser_llm_param_x{4,8}_s{0,1,2}.json`, `real6_categoriser_Qwen2.5-3B-Instruct_param_x{4,8}_s*_lora.json`, `items2_*_param_x*_lora.json`, `real6_Qwen2.5-3B-Instruct_chat.json`, `real6_*_{none,ret}_chat_lora.json`, `categoriser_llm_{none,ret}_chat.json`; the adapters under `models/adapters/`. Everything is QLoRA on the 4-bit base and scored on it (section 44), so every cell sits beside Tables 38.1, 43.4 and 44.1.*

Section 38 injected the 240 fact-DB records into the categoriser's SFT mixture (the record and three paraphrases per merchant, 960 texts, full-sequence loss) and found that one pass over them did nothing, while 3.3 passes (400 steps with half the sequences drawn from the records) gave +21 on the merchants no user had labelled at a nine-point ARC-Easy cost. Section 43 then ran that arm at three seeds and the +21 became +10 +- 12: the section 38 adapter was the best seed. REAL-8 asked for the rest of the curve, 6.7 and 13.3 passes (800 and 1,600 steps at 50%), at three seeds each, and for the REAL-6 prompt through the instruct model's chat template, since production will use the template. The chain (`chain_r39.sh`) ran the chat block first (the instruct base with and without the record, then the no-DB and record-in-prompt categorisers trained and scored in the template), then the six exposure adapters, each scored on REAL-6 without a record and on the ARC / MMLU / ICL items of `exp_items_v2`: about 13 GPU hours, 68 minutes to train each 800-step adapter and 135 for each 1,600-step one.

**The chat template.** `real6.chat_prompt` puts everything up to the query line (the user's categories, the 24 shots, the query transaction and, with the record, the merchant's note) in the user turn, and opens the assistant turn with the `Category:` cue, so the option strings follow the cue exactly as in the plain format and the scorer is unchanged (`ai_experiments.scoring`, sum of option log-probs after the cue). With the cue left inside the user turn the instruct base opens a sentence ("Based on the transactions provided, ...") and bare option scoring is off its distribution; the prefill is what makes the two formats comparable. The two trained arms are trained in the template (the same 3,200 sequences, label loss only) and scored in it.

**Table 45.1: the parametric exposure curve. The SFT categoriser (rank-64 QLoRA on the 4-bit Qwen2.5-3B-Instruct, lr 1e-4, 16 sequences per step) with the 960 fact-DB texts (record and three paraphrases per merchant) mixed into its training sequences, by the number of passes over them; REAL-6 without a record, the ARC / MMLU / ICL items of exp_items_v2 (4-bit reads, as Table 38.3); mean +- sd over seeds, then seeds 0 / 1 / 2**

| measure | no DB (0 passes) | 200 steps at 30% (1.0 passes) | 400 steps at 50% (3.3 passes) | 800 steps at 50% (6.7 passes) | 1,600 steps at 50% (13.3 passes) |
|---|---|---|---|---|---|
| REAL-6, all items | 60.5 +- 3.0 (57.1 / 61.7 / 62.8) | 54.5 | 63.8 +- 2.3 (66.3 / 63.5 / 61.7) | 76.8 +- 2.2 (74.7 / 76.6 / 79.0) | 78.2 +- 0.3 (77.9 / 78.3 / 78.5) |
| REAL-6, unseen merchant | 57.7 +- 3.1 (54.2 / 59.0 / 60.0) | 48.9 | 61.4 +- 4.7 (66.0 / 61.5 / 56.6) | 75.7 +- 4.3 (71.0 / 76.8 / 79.4) | 75.9 +- 1.3 (74.4 / 76.5 / 76.9) |
| REAL-6, unseen, new word | 43.8 +- 1.6 (42.9 / 42.9 / 45.7) | 40.0 | 47.4 +- 15.1 (60.0 / 51.4 / 30.7) | 80.5 +- 4.3 (80.0 / 76.4 / 85.0) | 85.2 +- 2.9 (85.7 / 82.1 / 87.9) |
| DB-only merchants | 49.1 +- 3.8 (51.2 / 44.7 / 51.2) | 53.7 | 59.3 +- 12.3 (72.4 / 57.7 / 48.0) | 68.8 +- 5.8 (62.6 / 69.9 / 74.0) | 64.2 +- 5.7 (57.7 / 68.3 / 66.7) |
| DB-only, opaque | 25.6 +- 6.8 (25.0 / 19.2 / 32.7) | 13.5 | 30.8 +- 21.4 (53.8 / 26.9 / 11.5) | 44.2 +- 10.0 (38.5 / 38.5 / 55.8) | 34.0 +- 9.9 (23.1 / 42.3 / 36.5) |
| DB-only, known chain | 66.2 +- 3.7 (70.4 / 63.4 / 64.8) | 83.1 | 80.3 +- 5.6 (85.9 / 80.3 / 74.6) | 86.9 +- 6.4 (80.3 / 93.0 / 87.3) | 86.4 +- 2.9 (83.1 / 87.3 / 88.7) |
| ARC-Easy | 76.0 / - / - | 64.5 | 67.0 / - / - | 73.2 +- 2.3 (71.0 / 73.0 / 75.5) | 65.7 +- 8.3 (73.5 / 66.5 / 57.0) |
| MMLU 5-shot | 51.0 / - / - | 50.0 | 52.0 / - / - | 51.5 +- 0.5 (52.0 / 51.0 / 51.5) | 51.7 +- 0.6 (52.0 / 52.0 / 51.0) |
| ICL symbol (v2 suite) | 64.5 / - / - | 55.8 | 58.3 / - / - | 61.1 +- 2.0 (62.0 / 58.9 / 62.5) | 62.3 +- 1.7 (63.5 / 63.0 / 60.4) |
| ICL natural (v2 suite) | 88.5 / - / - | 85.4 | 85.4 / - / - | 85.3 +- 1.3 (86.5 / 83.9 / 85.4) | 84.0 +- 0.3 (83.9 / 84.4 / 83.8) |
| training minutes | 18 +- 0 (18 / 18 / 18) | 17 | 34 +- 0 (33 / 34 / 34) | 68 +- 0 (68 / 68 / 68) | 135 +- 0 (135 / 135 / 136) |
| final loss (last 10 steps) | 0.112 +- 0.025 (0.141 / 0.101 / 0.095) | 0.513 | 0.262 +- 0.015 (0.265 / 0.245 / 0.275) | 0.203 +- 0.007 (0.211 / 0.199 / 0.199) | 0.177 +- 0.016 (0.160 / 0.190 / 0.182) |

**Table 45.2: the REAL-6 prompt in the plain format (sections 38, 43) and through Qwen2.5-Instruct's chat template (everything up to the query line as the user turn, "Category:" opening the assistant turn; the categorisers trained and scored in it); accuracy %, 4-bit throughout; the last column is the share of items given the same prediction by the plain and chat readings**

| model | plain: all / unseen / DB-only / opaque | chat: all / unseen / DB-only / opaque | same prediction |
|---|---|---|---|
| Instruct base, no record | 31.2 / 26.8 / 30.1 / 3.8 | 10.0 / 7.7 / 8.1 / 1.9 | 12.1% |
| Instruct base + record | 58.0 / 54.7 / 48.8 / 44.2 | 63.2 / 59.2 / 62.6 / 50.0 | 66.8% |
| SFT, no DB | 57.1 / 54.2 / 51.2 / 25.0 | 58.9 / 57.7 / 53.7 / 25.0 | 72.7% |
| SFT + record in prompt | 90.2 / 88.5 / 97.6 / 94.2 | 90.0 / 86.9 / 98.4 / 96.2 | 92.3% |

Training (chat format): none: 19.1 min, final loss 0.099 (plain 0.141); ret: 19.5 min, final loss 0.05 (plain 0.035)

### 45.1 The exposure curve: the gain arrives between 3.3 and 6.7 passes, the seed spread halves, and the second doubling adds nothing the DB-only cells can see

Table 45.1 is the curve at five points, with the no-DB seeds of section 43 as the zero. On the merchants only the DB knows, the cells the parametric question is about, the means run 49.1 +- 3.8 (no DB), 53.7 (one pass), 59.3 +- 12.3 (3.3 passes), 68.8 +- 5.8 (6.7) and 64.2 +- 5.7 (13.3). The gain that section 43 found inside the noise at 3.3 passes is outside it at 6.7: +20 over the no-DB seeds, with all three seeds above every no-DB seed (62.6 / 69.9 / 74.0 against 51.2 / 44.7 / 51.2) and the spread halved. Doubling again does not add to it; the 13.3-pass seeds (57.7 / 68.3 / 66.7) sit inside the 6.7-pass range and their mean is 5 points lower. The split by merchant kind says where the ceiling is: the real chains among the DB-only merchants go from 66 without the DB to 87 at 6.7 passes and stay there (86), while the opaque ones, whose category is only in the record, go 26 / 14 / 31 +- 21 / 44 +- 10 / 34 +- 10. Two seeds at 6.7 passes read 38.5 on them and one 55.8; no seed at any exposure passes 56, against 94.2 with the same record in the prompt (section 38) and 91 +- 6 over section 43's seeds. Written into the weights with thirteen passes over four templates, the record of an opaque merchant is read back through a noisy statement string less than half the time; put in front of the query it is read nearly always. That is section 38's verdict with the curve completed: the exposure that section 8 needed (4.6 passes) is where the parametric gain becomes real here too, and more of it buys nothing.

The whole-set rows move more than the DB-only rows, and that is the part of the curve that is not about the records. All items go 63.8 +- 2.3 at 3.3 passes to 76.8 +- 2.2 at 6.7 and 78.2 +- 0.3 at 13.3; the unseen merchants with coined category names 47 +- 15 to 80 +- 4 to 85 +- 3. Those cells are mostly merchants other users labelled in training, and what doubles alongside the DB exposure is the history exposure: at 50% the 800-step arm sees 6,400 history sequences, two passes over the 3,200-example SFT set that the no-DB and 400-step arms see once, and the 1,600-step arm four. The no-DB arm was not run at 400, 800 or 1,600 steps, so the table cannot say how much of the 64-to-77 step is the second pass over the histories and how much the records; the DB-only cells, where the histories carry nothing about the merchant, are the clean reading, and there the step is 59 to 69. The training loss over the last ten steps tells the same story from the other side (0.26 at 3.3 passes, 0.20 at 6.7, 0.18 at 13.3, against 0.11 for the no-DB arm, whose sequences are all short labels): the mixture is still fitting the record texts at 13.3 passes while the REAL-6 reading of them has stopped moving.

**General ability.** Section 38's nine-point ARC cost was one seed at 3.3 passes (67.0 against 76.0 for the no-DB adapter, 71.5 for the instruct base at 4-bit, Table 44.4). At 6.7 passes the three seeds read 71.0 / 73.0 / 75.5 (73.2 +- 2.3), at the base's level and inside the +- 3 that 200 items resolve; at 13.3 passes 73.5 / 66.5 / 57.0 (65.7 +- 8.3), one seed keeping the base's score and one losing 14 points. MMLU does not move at any exposure (51 to 52 throughout, base 50.0). The v2 ICL suite reads 61 to 62 on symbol labels at both exposures against 64.5 for the no-DB adapter and 84 to 85 on natural labels against 88.5, a 3-to-4-point cost that does not grow with exposure. So the forgetting cost of parametric injection at this recipe is small and noisy rather than nine points: within a seed's spread at 6.7 passes, and at 13.3 passes a risk (one seed in three) rather than a certainty.

### 45.2 The chat template: nothing changes for a trained categoriser, the untrained base needs the record more than before

Table 45.2 reads the same items in the plain format and the template, 4-bit throughout. The two categorisers trained in the template land where their plain-format twins did: the no-DB adapter 58.9 against 57.1 (inside 60.5 +- 3.0 over seeds), the record-in-prompt adapter 90.0 against 90.2, with the DB-only merchants at 98.4 against 97.6 and the opaque ones 96.2 against 94.2. 73% and 92% of the individual predictions are the same, which is the same-seed agreement the adapters show between precisions (section 44) and between seeds (section 43): the format is not a variable once the model has been trained in it. The training runs are a minute longer (19 against 18; the template adds about 30 tokens a sequence) and end at the same loss.

The untrained instruct base is where the template matters. With the record in the prompt it reads 63.2 in the template against 58.0 plain, 5 points better, with the DB-only merchants at 62.6 against 48.8 and the opaque ones at 50.0 against 44.2: the model reads the note more readily as a user's message than as a line of a plain document. Without the record it reads 10.0 against 31.2, near chance (about 7), on 12% of the same predictions, and its predictions pile on two options (205 and 189 of the 1,179 items on one option position each; the plain format's predictions spread over every position). The template with an assistant prefill and no note is a shape the instruct tuning did not see, and the model's prior over the bare option strings after "Category:" is not the category prior the plain format elicited from the 24 shots. Section 37's 31.2 was already the weakest reading in the table, so nothing built on it changes; but a production prompt that runs the instruct base untrained, through the template, without a retrieved record, is below the frozen encoder's prototypes of section 37 (36.5 on the full history, 41.2 mixed) by 26 to 31 points, not 5 to 10.

### 45.3 What the step says

REAL-8 asked where the exposure curve goes and whether the chat template moves anything. The curve peaks at 6.7 passes: 69 +- 6 on the DB-only merchants over three seeds against 49 without the records and 59 +- 12 at 3.3 passes, a real gain at that exposure and at no general-ability cost the seeds can see, and the next doubling adds nothing (64 +- 6, ARC now a risk on one seed in three, 135 training minutes). On the merchants whose category is only in the record, the opaque ones, no exposure passes 56 while the record in the prompt reads 94 to 96 at a tenth of the training time. The recommendation of sections 38 and 43 stands with the numbers filled in: retrieve the record; if the weights must carry it, 800 steps at 50% is the setting, and the arm should be sized against a no-DB arm at the same step count before any of the whole-set gain is credited to the records. The chat template is the production format and can be used without a measured cost for a trained categoriser (58.9 / 90.0); the untrained instruct base should not be run in it without a record. **Not run:** the no-DB arm at 400, 800 and 1,600 steps (the history-exposure control that would split Table 45.1's whole-set step); the parametric arms in the chat template; bf16 at any exposure; the section 43 ambiguous DB on the 800-step arm.

## 46. FastFit, GLiClass and a logistic-regression head on the REAL-6 cells: without the record no per-user classifier reaches the cross-user ones (FastFit 49 from the history, the frozen centroid 36.5, the tuned cross-user centroid 75.7, SFT 57), and with the record on the statement a per-user FastFit reads 92.2 (SFT + record 90.2; 92.5% of the same predictions) and a logistic head on frozen bge 87.2, at a minute per user (BASE-6)

*PLAN step 40. Code: `scripts/exp_real6_fewshot.py` (routes `fastfit | gliclass base|train|<dir> | logreg`; `CTX=1` appends the merchant's fact-DB record to the statement in training and at test, `ENC=bge|mpnet`, `LOGREG_C`, `SMOKE=1`), `src/ai_experiments/real6_eval.py` (the per-cell summary and per-item records that `exp_real6.py` now imports), `scripts/real6_fewshot_tables.py`, `scripts/chains/chain_r40.sh`. Results `results/real6_{fastfit_bge,fastfit_mpnet,logreg_bge,logreg_bge_c10,logreg_bge_c100,gliclass,gliclass_ft}[_ctx].json` and their per-item files; the 120 per-user FastFit encoders under `models/adapters/fastfit_<enc>_<cond>[_ctx]/<user>/` and the two GLiClass fine-tunes under `models/adapters/gliclass_ft[_ctx]`, all in DVC. Packages added: `fast-fit` 1.2.1, `gliclass` 0.1.20, `scikit-learn`.*

The owner asked about IBM's FastFit (Yehudai and Bendel, arXiv 2404.12365): a few-shot classifier for many similar classes that fine-tunes a sentence encoder with a batch-contrastive loss between the examples and the class names and scores a query by token-level MaxSim against the label's text, "in seconds". REAL-6 is its advertised setting, per-user schemes of 8 to 20 similar categories with 24 shots, and it uses the one thing the centroid of sections 37 and 38 ignores, the words of the label. The scan of 2026-09-21 (`references/fewshot_scan_2026-09-21.md`) named GLiClass (Knowledgator, arXiv 2508.07662) as its nearest relative, a GLiNER-style encoder that prepends every label as a `<<LABEL>>` token and scores the label token's vector against the text, zero-shot, with a few-shot `<<EXAMPLE>>` mode and a fine-tuning script. Both go into the same cells as the encoders and the SFT categoriser, beside the cheapest possible head, a logistic regression on frozen bge embeddings.

**The arms.** Every arm is a classifier per user, fitted either on the 24 prompt shots or on the 300-row history (`shots` / `full`, the rows the prompt and the prototypes of section 37 use), and scored on the user's frozen items; the cells are those of Table 38.1 and the merchant groups those of Table 38.2. FastFit (`fastfit.modeling.FastFitTrainable`, README recipe: 40 epochs, batch 32, four repeats, Adafactor at 5e-5, no MLM, similarity loss over all tokens with the classification head at 0.1) on bge-base and on FastFit's own default paraphrase-mpnet-base-v2, statement and record truncated at 64 tokens (the README's 32 cuts the record); every user gets a fresh encoder from the pretrained weights. GLiClass (`gliclass-modern-base-v3.0`, 151M) untrained with the category names only and with the 24 shots as examples, then fine-tuned across the 20 users on their histories with the DB-only merchants held out (5,365 rows, three epochs, lr 1e-5, half the rows carrying the user's shots as examples), scored the same two ways; one model for all users. Logistic regression (scikit-learn, lbfgs) on frozen bge embeddings at C = 1, 10 and 100: the default C = 1 underfits unit-norm 768-dimensional features badly (12.9 / 27.1 from the shots / the history against 22.4 / 43.9 at C = 100), so the tables carry C = 100. Each arm runs plain and with the merchant's record appended to the statement in training and at test (`CTX=1`, the encoder counterpart of the record-in-prompt arm), which for a per-user classifier means the training rows carry records too. Three package repairs were needed and are in the script: gliclass 0.1.20 cannot train a single-label checkpoint as shipped (its collator drops the 0-d labels, its loss reshapes with `num_labels = -1`, and its Trainer catches the exception and skips every step while reporting a finished run with loss 0; the script stacks the labels and uses a masked cross-entropy over each row's real classes), `FastFitConfig` cannot be saved under transformers 5 without `has_no_defaults_at_init`, and `MPNetConfig` lacks the two attributes FastFit reads. The v3.0 tokenizer has `<<LABEL>>` and `<<SEP>>` but no `<<EXAMPLE>>`, so the untrained few-shot arm feeds the examples as plain text; the fine-tune adds the token.

**Table 46.1: few-shot classifiers on the REAL-6 cells without a record, one model per user from the 24 shots or the 300-row history (accuracy % [95% bootstrap interval], * = inside the null band; chance about 7)**

| cell | bge frozen centroid, history (sec. 37) | bge tuned centroid, history (sec. 38) | logreg on bge (C=100), 24 shots | logreg on bge (C=100), history | FastFit bge, 24 shots | FastFit bge, history | FastFit mpnet, 24 shots | FastFit mpnet, history | GLiClass, names only | GLiClass + 24 shots | GLiClass tuned, names only | GLiClass tuned + 24 shots | SFT, no DB (sec. 38) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seen merchant, standard name | 60.4 [54.2, 67] | 83 [77.8, 87.7] | 32.5 [26.4, 38.7] | 70.3 [64.2, 76.4] | 46.2 [39.6, 52.8] | 74.1 [67.9, 80.2] | 37.7 [31.6, 43.9] | 72.2 [66, 78.3] | 16 [11.3, 20.8] | 10.8 [6.6, 15.1] | 33.5 [26.9, 39.6] | 17.5 [12.3, 23.1] | 65.1 [58.5, 71.2] |
| seen merchant, renamed | 58.8 [52.1, 64.7] | 73.9 [68.5, 79] | 32.8 [26.9, 39.1] | 74.8 [68.9, 80.3] | 38.7 [32.8, 44.5] | 73.9 [68.5, 79.4] | 38.2 [32.4, 44.1] | 76.5 [70.6, 81.5] | 13 [8.8, 18.1] | 6.7* [3.4, 10.1] | 28.6 [23.1, 34] | 5.5* [2.9, 8.4] | 58 [52.5, 63.9] |
| seen merchant, new word | 60.6 [52.3, 69.7] | 88.1 [81.7, 93.6] | 43.1 [33.9, 52.3] | 75.2 [67, 83.5] | 41.3 [32.1, 50.5] | 74.3 [66.1, 82.6] | 38.5 [29.4, 47.7] | 78 [69.7, 85.3] | 0.9 [0, 2.8] | 12.8 [7.3, 19.3] | 34.9 [26.6, 44] | 23.9 [16.5, 31.2] | 56 [46.8, 65.1] |
| unseen merchant, standard name | 20 [15.4, 24.6] | 80 [74.6, 85.4] | 9.2* [5.4, 12.9] | 25.8 [20, 31.2] | 37.1 [30.4, 42.9] | 35.8 [29.6, 41.2] | 37.5 [31.2, 43.3] | 30.8 [25.4, 36.7] | 14.6 [10.4, 19.2] | 4.2* [1.7, 7.1] | 38.8 [32.5, 45] | 17.9 [13.3, 22.5] | 67.9 [61.7, 73.3] |
| unseen merchant, renamed | 12.1 [8.3, 15.8] | 60 [53.8, 66.7] | 10.4 [6.7, 14.2] | 9.2 [5.8, 12.5] | 12.1 [7.9, 16.2] | 19.6 [14.6, 24.6] | 24.2 [18.8, 29.6] | 24.2 [18.8, 29.6] | 18.8 [14.2, 24.2] | 5.8* [2.9, 8.8] | 34.6 [28.7, 40.8] | 10.4 [6.7, 14.6] | 47.1 [41.2, 53.8] |
| unseen merchant, new word | 13.6 [8.6, 19.3] | 77.9 [70.7, 84.3] | 16.4 [10.7, 22.9] | 17.1 [11.4, 23.6] | 11.4* [6.4, 17.1] | 22.1 [15.7, 29.3] | 12.1* [7.1, 17.9] | 17.9 [12.1, 24.3] | 0 [0, 0] | 7.9* [3.6, 12.9] | 37.1 [29.3, 45.7] | 18.6 [12.9, 25] | 42.9 [35, 51.4] |
| seen merchant, all | 59.7 [55.6, 63.7] | 80.1 [76.9, 83.2] | 34.7 [30.8, 38.6] | 73.2 [69.4, 76.9] | 42 [37.9, 46.5] | 74.1 [70.5, 77.6] | 38.1 [34.2, 42.2] | 75.1 [71.7, 78.7] | 11.8 [9.3, 14.7] | 9.5 [7, 12] | 31.7 [28.1, 35.6] | 13.6 [10.9, 16.8] | 60.3 [56.2, 64.4] |
| unseen merchant, all | 15.5 [12.6, 18.5] | 71.8 [68.1, 74.8] | 11.3 [9, 13.9] | 17.4 [14.7, 20.5] | 21.6 [18.4, 24.8] | 26.5 [23.1, 30] | 26.6 [22.9, 30.2] | 25.3 [22.1, 28.7] | 12.9 [10.5, 15.6] | 5.6* [3.9, 7.6] | 36.8 [32.7, 40.5] | 15.2 [12.1, 18.2] | 54.2 [50.3, 58.1] |
| standard names, all | 38.9 [34.1, 43.1] | 81.4 [77.9, 85] | 20.1 [16.4, 23.9] | 46.7 [42, 51.5] | 41.4 [36.9, 45.8] | 53.8 [48.9, 58.6] | 37.6 [33.4, 41.8] | 50.2 [45.6, 54.6] | 15.3 [11.9, 18.1] | 7.3* [5.1, 10] | 36.3 [31.6, 40.7] | 17.7 [13.9, 21.2] | 66.6 [61.9, 70.8] |
| renamed, all | 35.4 [31.4, 39.7] | 66.9 [62.6, 71.3] | 21.5 [18.2, 25.3] | 41.8 [37.4, 46.7] | 25.3 [21.3, 29.3] | 46.7 [42.3, 51.3] | 31.2 [27.2, 35.6] | 50.2 [45.8, 55] | 15.9 [12.6, 19.2] | 6.3* [4.2, 8.4] | 31.6 [27.6, 36] | 7.9 [5.6, 10.5] | 52.5 [47.9, 56.9] |
| new words, all | 34.1 [28.9, 40.2] | 82.3 [77.5, 86.7] | 28.1 [22.9, 34.1] | 42.6 [36.9, 48.6] | 24.5 [18.9, 29.7] | 45 [39, 51] | 23.7 [18.5, 29.3] | 44.2 [38.6, 50.6] | 0.4 [0, 1.2] | 10* [6.8, 14.5] | 36.1 [30.5, 41.8] | 20.9 [16.1, 26.1] | 48.6 [42.6, 54.6] |
| all items | 36.5 [33.6, 39] | 75.7 [73.2, 77.9] | 22.4 [20.2, 24.8] | 43.9 [41.1, 46.6] | 31.3 [28.7, 33.9] | 49 [46.1, 51.8] | 32.1 [29.3, 34.8] | 48.9 [46.1, 51.7] | 12.4 [10.4, 14.3] | 7.5* [6, 9] | 34.4 [31.8, 37.2] | 14.4 [12.5, 16.4] | 57.1 [54.3, 59.9] |

**Table 46.2: the same with the merchant's fact-DB record appended to the statement, in training and at test**

| cell | bge tuned centroid + record at test (sec. 38) | logreg on bge (C=100) + record, 24 shots | logreg on bge (C=100) + record, history | FastFit bge + record, 24 shots | FastFit bge + record, history | GLiClass + record, names only | GLiClass + record + 24 shots | GLiClass tuned + record, names only | GLiClass tuned + record + 24 shots | SFT + record in prompt (sec. 38) |
|---|---|---|---|---|---|---|---|---|---|---|
| seen merchant, standard name | 90.1 [86.3, 93.4] | 80.7 [75.5, 85.8] | 98.6 [96.7, 100] | 100 [100, 100] | 100 [100, 100] | 87.3 [82.5, 91] | 10.4 [6.6, 14.6] | 98.6 [96.7, 100] | 72.6 [66.5, 78.3] | 99.5 [98.6, 100] |
| seen merchant, renamed | 83.2 [78.6, 87.4] | 67.2 [61.3, 73.5] | 93.7 [90.3, 96.6] | 83.6 [78.6, 88.7] | 95 [92, 97.5] | 52.9 [46.6, 59.7] | 8.4* [5, 11.8] | 82.4 [77.3, 86.6] | 52.1 [45.4, 58] | 85.7 [81.1, 89.9] |
| seen merchant, new word | 91.7 [86.2, 96.3] | 78.9 [70.6, 87.2] | 97.2 [93.6, 100] | 89.9 [84.4, 95.4] | 99.1 [97.2, 100] | 0.9 [0, 2.8] | 0 [0, 0] | 87.2 [80.7, 92.7] | 70.6 [61.5, 78.9] | 90.8 [84.4, 96.3] |
| unseen merchant, standard name | 89.6 [85.4, 93.3] | 58.3 [52.1, 64.2] | 84.6 [80, 88.8] | 100 [100, 100] | 100 [100, 100] | 91.7 [87.9, 95] | 9.2* [5.8, 12.9] | 98.8 [97.1, 100] | 66.7 [60.4, 72.9] | 100 [100, 100] |
| unseen merchant, renamed | 66.7 [60.8, 72.5] | 51.2 [44.6, 57.9] | 67.5 [61.7, 73.3] | 75 [69.2, 80] | 67.9 [61.7, 73.8] | 55.4 [49.6, 62.1] | 11.7 [7.9, 16.2] | 68.8 [62.5, 74.6] | 41.2 [35, 47.5] | 75 [69.6, 80.4] |
| unseen merchant, new word | 87.9 [82.1, 92.9] | 65.7 [57.9, 73.6] | 89.3 [83.6, 94.3] | 92.9 [88.6, 97.1] | 98.6 [96.4, 100] | 0 [0, 0] | 0 [0, 0] | 78.6 [71.4, 85] | 54.3 [46.4, 62.1] | 92.1 [87.9, 96.4] |
| seen merchant, all | 87.5 [84.8, 90.5] | 74.6 [71.2, 78] | 96.2 [94.6, 97.7] | 91.1 [88.6, 93.4] | 97.7 [96.4, 98.7] | 55.8 [51.7, 60.1] | 7.5 [5.4, 9.8] | 89.4 [86.8, 91.9] | 63.5 [59.2, 67.4] | 91.9 [89.6, 94.1] |
| unseen merchant, all | 80.3 [77.3, 83.4] | 57.3 [53.2, 61] | 79 [75.8, 82.1] | 88.7 [86.3, 91.1] | 87.3 [84.7, 89.8] | 56.9 [53.4, 61] | 8.1 [6.1, 10.2] | 82.6 [79.4, 85.3] | 54 [50, 57.9] | 88.5 [86, 91.1] |
| standard names, all | 89.8 [86.9, 92.5] | 68.8 [64.2, 73] | 91.2 [88.3, 93.6] | 100 [100, 100] | 100 [100, 100] | 89.6 [86.9, 92.3] | 9.7 [7.3, 12.6] | 98.7 [97.6, 99.6] | 69.5 [65.5, 73.9] | 99.8 [99.3, 100] |
| renamed, all | 74.9 [71.3, 78.7] | 59.2 [55, 63.6] | 80.5 [76.8, 83.9] | 79.3 [75.7, 83.1] | 81.4 [77.6, 84.7] | 54.2 [50.2, 58.6] | 10 [7.3, 12.6] | 75.5 [71.5, 79.5] | 46.7 [42.1, 51.3] | 80.3 [76.6, 83.9] |
| new words, all | 89.6 [85.9, 93.2] | 71.5 [65.9, 77.5] | 92.8 [89.2, 96] | 91.6 [88.4, 94.8] | 98.8 [97.2, 100] | 0.4 [0, 1.2] | 0 [0, 0] | 82.3 [77.5, 86.7] | 61.4 [55.8, 67.1] | 91.6 [88, 94.8] |
| all items | 83.7 [81.6, 85.8] | 65.5 [62.8, 68.2] | 87.2 [85.2, 89.1] | 89.8 [88, 91.5] | 92.2 [90.7, 93.6] | 56.4 [53.3, 59.1] | 7.8 [6.3, 9.3] | 85.8 [83.7, 87.8] | 58.5 [55.7, 61.5] | 90.2 [88.4, 91.7] |


**Table 46.3a: the merchant-group split without a record (accuracy %; groups as in Table 38.2)**

| group | bge frozen centroid, history (sec. 37) | bge tuned centroid, history (sec. 38) | logreg on bge (C=100), 24 shots | logreg on bge (C=100), history | FastFit bge, 24 shots | FastFit bge, history | FastFit mpnet, 24 shots | FastFit mpnet, history | GLiClass, names only | GLiClass + 24 shots | GLiClass tuned, names only | GLiClass tuned + 24 shots | SFT, no DB (sec. 38) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| in the 24 shots | 86.2 | 96.2 | 87.5 | 93.1 | 89.4 | 91.9 | 85.6 | 93.8 | 10.6 | 11.2 | 38.8 | 20.0 | 96.2 |
| in the history, not the shots | 49.1 | 73.7 | 13.5 | 65.2 | 23.1 | 66.9 | 19.0 | 67.7 | 12.3 | 8.8 | 28.8 | 11.0 | 45.9 |
| not in the history | 15.5 | 71.8 | 11.3 | 17.4 | 21.6 | 26.5 | 26.6 | 25.3 | 12.9 | 5.6 | 36.8 | 15.2 | 54.2 |
| not in the history, labelled by other users in training | 15.1 | 80.1 | 12.7 | 18.1 | 20.9 | 26.2 | 26.6 | 24.5 | 12.5 | 5.8 | 41.4 | 17.9 | 54.9 |
| not in the history, DB-only (no user labelled it in training) | 17.1 | 38.2 | 5.7 | 14.6 | 24.4 | 27.6 | 26.8 | 28.5 | 14.6 | 4.9 | 17.9 | 4.1 | 51.2 |
| DB-only, known chain | 23.9 | 62.0 | 5.6 | 21.1 | 38.0 | 45.1 | 39.4 | 45.1 | 21.1 | 5.6 | 29.6 | 4.2 | 70.4 |
| DB-only, opaque | 7.7 | 5.8 | 5.8 | 5.8 | 5.8 | 3.8 | 9.6 | 5.8 | 5.8 | 3.8 | 1.9 | 3.8 | 25.0 |
| DB-only, standard name | 20.0 | 38.3 | 1.7 | 20.0 | 31.7 | 33.3 | 30.0 | 38.3 | 23.3 | 0.0 | 18.3 | 5.0 | 58.3 |
| DB-only, renamed | 22.0 | 34.1 | 9.8 | 14.6 | 24.4 | 34.1 | 31.7 | 26.8 | 9.8 | 7.3 | 22.0 | 0.0 | 51.2 |
| DB-only, new word | 0.0 | 45.5 | 9.1 | 0.0 | 4.5 | 0.0 | 9.1 | 4.5 | 0.0 | 13.6 | 9.1 | 9.1 | 31.8 |

**Table 46.3b: the merchant-group split with the record on the query (accuracy %)**

| group | bge tuned centroid + record at test (sec. 38) | logreg on bge (C=100) + record, 24 shots | logreg on bge (C=100) + record, history | FastFit bge + record, 24 shots | FastFit bge + record, history | GLiClass + record, names only | GLiClass + record + 24 shots | GLiClass tuned + record, names only | GLiClass tuned + record + 24 shots | SFT + record in prompt (sec. 38) |
|---|---|---|---|---|---|---|---|---|---|---|
| in the 24 shots | 97.5 | 100.0 | 100.0 | 100.0 | 100.0 | 48.8 | 8.8 | 88.1 | 72.5 | 96.2 |
| in the history, not the shots | 83.5 | 64.4 | 94.7 | 87.5 | 96.7 | 58.6 | 7.0 | 90.0 | 59.9 | 90.2 |
| not in the history | 80.3 | 57.3 | 79.0 | 88.7 | 87.3 | 56.9 | 8.1 | 82.6 | 54.0 | 88.5 |
| not in the history, labelled by other users in training | 82.7 | 55.9 | 75.7 | 88.3 | 85.3 | 54.7 | 7.2 | 80.7 | 54.1 | 86.3 |
| not in the history, DB-only (no user labelled it in training) | 70.7 | 62.6 | 92.7 | 90.2 | 95.1 | 65.9 | 11.4 | 90.2 | 53.7 | 97.6 |
| DB-only, known chain | 85.9 | 78.9 | 93.0 | 97.2 | 95.8 | 64.8 | 15.5 | 91.5 | 60.6 | 100.0 |
| DB-only, opaque | 50.0 | 40.4 | 92.3 | 80.8 | 94.2 | 67.3 | 5.8 | 88.5 | 44.2 | 94.2 |
| DB-only, standard name | 73.3 | 65.0 | 96.7 | 100.0 | 100.0 | 88.3 | 3.3 | 100.0 | 61.7 | 100.0 |
| DB-only, renamed | 63.4 | 53.7 | 85.4 | 78.0 | 85.4 | 68.3 | 29.3 | 78.0 | 51.2 | 95.1 |
| DB-only, new word | 77.3 | 72.7 | 95.5 | 86.4 | 100.0 | 0.0 | 0.0 | 86.4 | 36.4 | 95.5 |

**Table 46.4: training cost (minutes per user model on the 3090, mean of the 20 users; GLiClass fine-tune: one model over all users) and final training loss**

| model | rows per user | minutes per user | final loss (last epoch mean) |
|---|---|---|---|
| FastFit bge, 24 shots | 24 | 0.16 | 3.107 |
| FastFit bge, history | 300 | 1.3 | 3.808 |
| FastFit bge + record, 24 shots | 24 | 0.17 | 3.102 |
| FastFit bge + record, history | 300 | 1.35 | 3.804 |
| FastFit mpnet, 24 shots | 24 | 0.18 | 3.119 |
| FastFit mpnet, history | 300 | 1.45 | 3.858 |
| GLiClass fine-tune, all users | 5365 (total) | 4.8 (total) | 0.7559 |
| GLiClass fine-tune + record, all users | 5365 (total) | 8.0 (total) | 0.3273 |

logreg on bge, history: 0.56 minutes for all 20 users.

GLiClass, names only: 0.09 minutes for all 20 users.

GLiClass + 24 shots: 0.13 minutes for all 20 users.

### 46.1 Without the record: FastFit beats the per-user centroid, the label text helps on standard names only, and no per-user classifier comes near the cross-user ones

Table 46.1 puts the per-user classifiers between the frozen centroid (36.5 from the history, 23.5 from the shots in Table 37.1) and the two cross-user models (the tuned centroid at 75.7 and the SFT categoriser at 57.1). FastFit from the history reads 49.0 on bge and 48.9 on mpnet, 12 points over the frozen centroid; from the 24 shots 31.3 and 32.1, 8 points over it; the logistic head at C = 100 reads 43.9 and 22.4. All of that gain is on the seen merchants (FastFit 74.1 from the history against the centroid's 59.7; the merchant among the 24 shots 91.9 against 86.2, elsewhere in the history 66.9 against 49.1): a fitted classifier separates a user's own merchants better than an average of their embeddings. On the merchants the user never labelled every per-user method is between 12 and 27, against 71.8 for the tuned cross-user centroid and 54.2 for SFT, because a per-user model has nothing but the user's rows and the label's words to go on. The label's words are the one place FastFit differs from the centroid, and BASE-6's expectation for them holds exactly: on unseen merchants with standard names FastFit reads 35.8 (history) and 37.1 (shots) against the centroid's 20.0, on renamed ones 19.6 and 12.1 against 12.1, on coined names 22.1 and 11.4 against 13.6. Scoring against the label text is worth 16 points when the label is a category word and nothing when it is the user's own word, and 452 of the 1,179 items carry the user's own word. The merchants only the DB knows sit at 24 to 29 for FastFit, 5 to 10 on the opaque ones, chance for everything without a record.

GLiClass untrained reads 12.4 with the names alone (chance about 7) and 7.5 with the 24 shots as text: a zero-shot classifier built on label semantics has nothing to say about a bank string like "SQ *RIDGEVIEW MKT 0231" under a scheme of renamed categories, and with 600 tokens of untagged examples in front of a 15-token query it stops reading the query. Fine-tuned across users it reaches 34.4 with the names alone (unseen 36.8, coined names 36.1, so the fine-tune has taught it the merchants, not the words) and 14.4 with the examples; the examples hurt in every GLiClass arm, tuned or not, plain or with the record, and this was not investigated further.

### 46.2 With the record: a per-user FastFit matches the 3B categoriser, and a logistic head on the frozen encoder is five points behind it

Table 46.2 is the same classifiers with the record appended to the statement. FastFit on bge reads 92.2 from the history and 89.8 from the 24 shots; the section 38 headline, the SFT categoriser with the record in its prompt, reads 90.2. The two agree on 92.5% of the individual predictions (both right on 87.5, either on 94.8), so they are the same classifier in different bodies. By cell: unseen merchants 87.3 against 88.5, the DB-only merchants 95.1 against 97.6, the opaque ones among them 94.2 against 94.2, coined names 98.8 against 91.6, renamed 81.4 against 80.3, standard names 100 against 99.8. The 100 is not a literal match, the records contain their category's name in 9 of the 452 standard items; it is section 4's disjoint product pools read through a classifier fitted on 300 records of the same pools, the convenience section 43 priced at 3 points overall and 9 on the DB-only merchants for the SFT model, and FastFit was not run on the ambiguous DB. The logistic head on frozen bge at C = 100 reads 87.2 from the history (65.5 from the shots), DB-only 92.7, opaque 92.3, with no encoder training at all and 0.6 minutes for the 20 users; the tuned cross-user centroid with the record at test only (Table 38.1) reads 83.7. GLiClass untrained with the record reads 56.4, and its split is the purest reading of label semantics in the report: 89.6 on standard names, 54.2 on renamed, 0.4 on coined, a model that can carry "sells fresh produce, deli meats and canned goods" to Groceries and to nothing the user made up; fine-tuned, 85.8 with coined names at 82.3. Table 46.3b says where the record does its work: the DB-only merchants go from 24 to 90 (FastFit, shots), the opaque ones from 6 to 81, and from the history 95 and 94.

**Cost.** A FastFit encoder trains in 0.16 minutes per user from 24 shots and 1.3 from the history at 40 epochs (Table 46.4), with a full 110M-parameter encoder per user, 440 MB, which is the wrong shape for a million users and is what FastFit is (the package fine-tunes the whole encoder; a shared FastFit across users, or the frozen encoder with a per-user head, is the production shape, and the 87.2 head is 5 points from it). The GLiClass fine-tune takes 4.8 minutes for all users (8.0 with the records); the logistic heads 0.6 minutes for all users.

### 46.3 What the step says

BASE-6 asked whether FastFit beats the prototype classifier on the per-user categories. It beats the per-user centroid it was compared against (49 against 36.5 from the history, 31 against 23.5 from the shots), it does so on the seen merchants, and its distinctive move, scoring against the label's text, pays 16 points on standard category names and nothing on renamed or coined ones. It does not approach the cross-user models without a record (the tuned centroid at 75.7, SFT at 57.1): other users' labels are worth more than any per-user method, which is section 38's collaborative finding again. With the record on the statement the ordering inverts: a per-user FastFit reads 92.2, level with the 3B categoriser's 90.2 and agreeing with it on 92.5% of items, and a logistic head on the frozen encoder 87.2. The record is the thing, and once it is on the input the classifier can be a hundred times smaller. GLiClass adds the cleanest measurement of what label semantics alone can do (standard names 90, coined names 0 with the record, untrained) and a warning about its few-shot mode at this length. Recommendation for production unchanged in substance, sharper in shape: retrieve the record onto the statement; then a shared encoder with a per-user head, or a shared FastFit, before any per-user encoder; the SFT model's remaining edge is 2 points on the DB-only merchants at a thousand times the inference cost. **Not run:** FastFit and the logistic head on the ambiguous DB of section 43 (the disjoint pools flatter the record arms); a single FastFit shared across users (the shape that would compete with the tuned centroid without a record); seeds (one per arm; the per-user fits are 20 independent models each, so the all-items intervals are honest but the per-cell ones are single-seed); the GLiClass examples failure; SetFit, which section 24 already found no better than the centroid.

## 47. Retrieved shots: every rule's gain is the user's own label for the same merchant, which a lookup already gives at 98.6; on the items decided by the merchant's category the shot choice moves nothing at test and training with retrieved shots costs 20 to 30 points, because the adapter learns to copy (REAL-9)

*PLAN step 41. Code: `ai_experiments.real6_shots` (the rules `recent`, `nearest`, `transact`, `cluster`, and `render`, which rebuilds the REAL-6 prompt from any shot list in the builder's exact format), `SHOTS=<rule>` in `scripts/exp_categoriser.py` (training shots per query by the rule over the DB-only-free history, the query excluded from its own pool) and in `scripts/exp_real6.py` (test shots), `ai_experiments.real6_cells` (the corrected groups of QUESTIONS.md REAL-13), `scripts/retrieved_shots_tables.py`, `scripts/chains/chain_r41.sh` (log `logs/gpu41.log`, 28 steps, 12.7 hours). Results `results/real6_*_shots<rule>*.json`, `results/categoriser_llm_{none,ret}_shots<rule>.json`, per-item records with `merchant_in_shots` and `gold_in_shots`. QLoRA on the 4-bit base throughout, seed 0.*

The owner frames the categoriser as a recommender: the user's labelled history is the interaction log, and the choice of which rows go into the prompt is the retrieval step. Every REAL-6 prompt so far carried one frozen block of 24 shots per user, stratified over the categories and filled at random, and the SFT trainer drew 24 random rows per training query. This step chooses the 24 per query by four rules and runs them on the untrained instruct base, on the fixed-shot adapters of section 38 read with the rule's shots at test only, and on adapters trained with the rule's shots. `recent` is the 24 rows before the query in history order; REAL-6 histories are a frozen shuffle with no time, so here it is a random block, the control that separates "different shots" from "chosen shots" (row 43's set gives it a meaning). `nearest` is the 24 rows most similar to the query string under the row 37 MiniLM retriever. `transact` is TransAct V2's rule: the 8 most recent rows plus the nearest row of each of the user's categories in turn until the block is full. `cluster` is Pinner Progression's: k-means over the user's rows (k = the number of categories, 2 to 15) and greedy picks by similarity discounted 0.7 per shot already taken from the same cluster.

Midway through the chain, a review of the evaluation set (QUESTIONS.md REAL-13) found that REAL-6's cell names hide what decides an item. A user files every merchant under one label, so an item whose merchant is in the user's history is answered by a lookup (the label of the nearest history row under the same retriever reads 98.6 on those 444 items); 115 of the 559 items labelled "seen" lost every history row of their merchant to the 300-row cut; and of the remaining items, 509 are decided by the merchant's standard category (which the user's scheme renames or merges) and 103 fall in a category the user split in two with nothing to say which side. Table 47.5 reads the runs in those groups, and it is the table that says what the rules did. Tables 47.1 to 47.3 keep the original cells for comparison with sections 37 to 46.

**Table 47.5: the corrected cells (accuracy %; groups from `real6_cells`: 444 items whose merchant is in the user's history, 115 labelled seen whose merchant the 300-row cut removed, 509 determined by the merchant's standard category, 103 in a category the user split; 'hybrid' = the label of the nearest history row when its retriever cosine is at least 0.8, else the model; interval = users resampled)**

| arm | rule | in history | labelled seen, not in history | determined by category | split category | all [user interval] | hybrid, all |
|---|---|---|---|---|---|---|---|
| nearest-row lookup alone | - | 98.6 | 15.7 | 19.1 | 6.8 | 47.5 | - |
| Instruct base, no record | fixed | 37.6 | 30.4 | 28.5 | 20.4 | 31.2 [27.7, 34.7] | 47.7 |
| Instruct base, no record | recent | 41.0 | 33.0 | 31.0 | 14.6 | 33.4 [29.4, 37.7] | 48.3 |
| Instruct base, no record | nearest | 95.3 | 33.0 | 30.8 | 13.6 | 53.6 [49.8, 57.6] | 54.2 |
| Instruct base, no record | transact | 79.1 | 32.2 | 30.5 | 11.7 | 47.2 [42.4, 52.1] | 52.0 |
| Instruct base, no record | cluster | 93.9 | 34.8 | 35.4 | 19.4 | 55.9 [52.4, 59.3] | 56.7 |
| Instruct base + record | fixed | 62.2 | 60.0 | 60.1 | 31.1 | 58.0 [53.3, 62.2] | 68.0 |
| Instruct base + record | recent | 63.5 | 56.5 | 62.7 | 17.5 | 58.1 [51.3, 63.7] | 67.4 |
| Instruct base + record | nearest | 95.0 | 55.7 | 54.6 | 24.3 | 67.2 [63.7, 70.9] | 68.3 |
| Instruct base + record | transact | 86.5 | 56.5 | 63.1 | 29.1 | 68.0 [63.2, 72.6] | 71.6 |
| Instruct base + record | cluster | 90.3 | 60.0 | 60.1 | 24.3 | 68.2 [63.8, 72.3] | 71.0 |
| SFT no DB (fixed-shot adapter), rule shots at test | fixed | 65.3 | 40.9 | 59.3 | 27.2 | 57.1 [53.6, 60.3] | 65.3 |
| SFT no DB (fixed-shot adapter), rule shots at test | recent | 57.4 | 44.3 | 60.5 | 27.2 | 55.0 [51.1, 58.5] | 65.2 |
| SFT no DB (fixed-shot adapter), rule shots at test | nearest | 99.1 | 43.5 | 58.5 | 31.1 | 70.1 [66.9, 73.2] | 70.0 |
| SFT no DB (fixed-shot adapter), rule shots at test | transact | 98.9 | 46.1 | 59.5 | 34.0 | 70.9 [67.0, 74.4] | 70.8 |
| SFT no DB (fixed-shot adapter), rule shots at test | cluster | 99.3 | 44.3 | 60.3 | 26.2 | 70.6 [66.8, 74.3] | 70.5 |
| SFT + record (fixed-shot adapter), rule shots at test | fixed | 94.1 | 83.5 | 97.4 | 50.5 | 90.2 [86.0, 93.5] | 91.3 |
| SFT + record (fixed-shot adapter), rule shots at test | recent | 91.2 | 84.3 | 94.3 | 46.6 | 87.8 [82.6, 92.1] | 89.9 |
| SFT + record (fixed-shot adapter), rule shots at test | nearest | 99.1 | 82.6 | 93.3 | 54.4 | 90.8 [86.8, 94.1] | 90.7 |
| SFT + record (fixed-shot adapter), rule shots at test | transact | 98.6 | 82.6 | 97.4 | 60.2 | 93.0 [89.4, 95.7] | 93.0 |
| SFT + record (fixed-shot adapter), rule shots at test | cluster | 98.9 | 84.3 | 95.9 | 49.5 | 91.4 [87.2, 94.7] | 91.4 |
| SFT no DB, trained and read with the rule | fixed | 65.3 | 40.9 | 59.3 | 27.2 | 57.1 [53.6, 60.3] | 65.3 |
| SFT no DB, trained and read with the rule | recent | 61.5 | 47.0 | 56.0 | 13.6 | 53.5 [49.2, 57.2] | 63.4 |
| SFT no DB, trained and read with the rule | nearest | 98.6 | 33.9 | 36.1 | 14.6 | 57.5 [53.5, 62.1] | 57.5 |
| SFT no DB, trained and read with the rule | transact | 97.5 | 30.4 | 29.5 | 15.5 | 53.9 [50.4, 57.5] | 54.1 |
| SFT no DB, trained and read with the rule | cluster | 98.6 | 38.3 | 50.5 | 13.6 | 64.0 [59.2, 68.7] | 64.1 |
| SFT + record, trained and read with the rule | fixed | 94.1 | 83.5 | 97.4 | 50.5 | 90.2 [86.0, 93.5] | 91.3 |
| SFT + record, trained and read with the rule | recent | 92.1 | 85.2 | 89.8 | 49.5 | 86.4 [82.3, 90.2] | 88.0 |
| SFT + record, trained and read with the rule | nearest | 99.1 | 75.7 | 79.2 | 47.6 | 83.5 [79.4, 87.4] | 83.6 |
| SFT + record, trained and read with the rule | transact | 99.5 | 79.1 | 87.2 | 46.6 | 87.3 [83.9, 90.2] | 87.2 |
| SFT + record, trained and read with the rule | cluster | 99.3 | 78.3 | 80.6 | 47.6 | 84.2 [81.1, 87.0] | 84.2 |

**Table 47.1: the shot rules on the REAL-6 items (accuracy % [95% bootstrap interval]; the fixed column is the frozen stratified block of sections 37, 38, 43; 4-bit throughout; DB-only = unseen merchants no user labelled in training)**

| arm | cell | fixed | recent | nearest | transact | cluster |
|---|---|---|---|---|---|---|
| Instruct base, no record | all items | 31.2 [28.7, 33.8] | 33.4 [31, 36.1] | 53.6 [50.6, 56.5] | 47.2 [44.5, 50.1] | 55.9 [53, 58.8] |
| Instruct base, no record | seen merchant | 36.1 [32.2, 40.3] | 39.4 [35.4, 43.3] | 82.5 [79.4, 85.3] | 69.4 [65.7, 72.8] | 81.8 [78.7, 84.8] |
| Instruct base, no record | unseen merchant | 26.8 [23.4, 30.3] | 28.1 [24.5, 31.6] | 27.6 [24.2, 31.1] | 27.3 [23.9, 31.1] | 32.6 [29, 36.1] |
| Instruct base, no record | unseen, renamed | 29.2 [23.8, 35] | 22.5 [17.5, 28.3] | 25.4 [19.6, 31.2] | 26.7 [21.2, 32.5] | 31.7 [25.8, 38.3] |
| Instruct base, no record | unseen, new word | 10.7 [5.7, 16.4] | 8.6* [4.3, 13.6] | 11.4 [6.4, 17.1] | 10* [5.7, 15] | 23.6 [16.4, 30.7] |
| Instruct base, no record | DB-only merchants | 30.1 | 31.7 | 34.1 | 32.5 | 33.3 |
| Instruct base + record | all items | 58 [55.1, 60.8] | 58.1 [55.3, 60.8] | 67.2 [64.5, 69.7] | 68 [65.3, 70.7] | 68.2 [65.5, 70.9] |
| Instruct base + record | seen merchant | 61.7 [57.6, 65.8] | 62.1 [58.3, 66] | 86.9 [84.3, 89.4] | 80.3 [76.7, 83.4] | 84.1 [81, 86.9] |
| Instruct base + record | unseen merchant | 54.7 [50.8, 58.5] | 54.5 [50.3, 58.5] | 49.4 [45.3, 53.2] | 56.9 [52.9, 60.8] | 53.9 [49.8, 57.9] |
| Instruct base + record | unseen, renamed | 51.2 [44.6, 57.5] | 38.8 [32.5, 45] | 39.2 [32.9, 45.4] | 49.6 [42.5, 56.2] | 43.8 [37.1, 50] |
| Instruct base + record | unseen, new word | 33.6 [25.7, 41.4] | 35.7 [27.9, 43.6] | 23.6 [16.4, 30.7] | 43.6 [35.7, 52.1] | 38.6 [30.7, 47.9] |
| Instruct base + record | DB-only merchants | 48.8 | 53.7 | 55.3 | 57.7 | 56.9 |
| SFT no DB (fixed-shot adapter), rule shots at test | all items | 57.1 [54.3, 59.9] | 55 [52.3, 57.8] | 70.1 [67.4, 72.8] | 70.9 [68.3, 73.5] | 70.6 [67.9, 73.1] |
| SFT no DB (fixed-shot adapter), rule shots at test | seen merchant | 60.3 [56.2, 64.4] | 54.7 [50.8, 58.9] | 87.7 [85, 90.2] | 88 [85.2, 90.5] | 88 [85.3, 90.5] |
| SFT no DB (fixed-shot adapter), rule shots at test | unseen merchant | 54.2 [50.3, 58.1] | 55.2 [51.1, 59.4] | 54.2 [50.2, 58.2] | 55.5 [51.3, 59.4] | 54.8 [50.8, 58.7] |
| SFT no DB (fixed-shot adapter), rule shots at test | unseen, renamed | 47.1 [41.2, 53.8] | 49.6 [43.3, 55.8] | 50 [43.3, 56.2] | 51.7 [45.8, 58.3] | 47.9 [41.7, 54.2] |
| SFT no DB (fixed-shot adapter), rule shots at test | unseen, new word | 42.9 [35, 51.4] | 43.6 [35.7, 52.1] | 44.3 [36.4, 52.9] | 47.9 [40.7, 55.7] | 48.6 [40.7, 57.1] |
| SFT no DB (fixed-shot adapter), rule shots at test | DB-only merchants | 51.2 | 48.0 | 52.0 | 50.4 | 56.1 |
| SFT + record (fixed-shot adapter), rule shots at test | all items | 90.2 [88.4, 91.7] | 87.8 [85.8, 89.7] | 90.8 [89.1, 92.4] | 93 [91.3, 94.3] | 91.4 [89.7, 92.9] |
| SFT + record (fixed-shot adapter), rule shots at test | seen merchant | 91.9 [89.6, 94.1] | 89.8 [87.5, 92.3] | 95.7 [93.9, 97.1] | 95.3 [93.6, 96.8] | 95.9 [94.1, 97.3] |
| SFT + record (fixed-shot adapter), rule shots at test | unseen merchant | 88.5 [86, 91.1] | 86 [83.4, 88.7] | 86.3 [83.5, 88.9] | 90.8 [88.4, 93.1] | 87.4 [84.8, 90.2] |
| SFT + record (fixed-shot adapter), rule shots at test | unseen, renamed | 75 [69.6, 80.4] | 75 [69.6, 80.4] | 76.7 [71.7, 82.1] | 80.8 [75.4, 85.4] | 75 [69.6, 80.4] |
| SFT + record (fixed-shot adapter), rule shots at test | unseen, new word | 92.1 [87.9, 96.4] | 80.7 [74.3, 87.1] | 79.3 [72.9, 85.7] | 92.9 [88.6, 96.4] | 87.9 [82.1, 92.9] |
| SFT + record (fixed-shot adapter), rule shots at test | DB-only merchants | 97.6 | 96.7 | 96.7 | 96.7 | 95.9 |
| SFT no DB, trained and read with the rule | all items | 57.1 [54.3, 59.9] | 53.5 [50.9, 56.4] | 57.5 [54.6, 60.6] | 53.9 [51.1, 56.7] | 64 [61.2, 66.9] |
| SFT no DB, trained and read with the rule | seen merchant | 60.3 [56.2, 64.4] | 58.5 [54.9, 62.8] | 85.3 [82.3, 88.2] | 83.7 [80.9, 86.8] | 86.2 [83.4, 88.9] |
| SFT no DB, trained and read with the rule | unseen merchant | 54.2 [50.3, 58.1] | 49 [44.8, 52.9] | 32.4 [28.4, 35.6] | 27.1 [23.4, 30.8] | 44 [40.2, 48.1] |
| SFT no DB, trained and read with the rule | unseen, renamed | 47.1 [41.2, 53.8] | 44.6 [38.3, 50.8] | 29.6 [24.2, 35.4] | 25.8 [20.4, 31.7] | 35.8 [30, 42.1] |
| SFT no DB, trained and read with the rule | unseen, new word | 42.9 [35, 51.4] | 28.6 [21.4, 35.7] | 29.3 [22.1, 37.9] | 25 [18.6, 32.1] | 44.3 [35.7, 52.9] |
| SFT no DB, trained and read with the rule | DB-only merchants | 51.2 | 43.1 | 38.2 | 26.0 | 41.5 |
| SFT + record, trained and read with the rule | all items | 90.2 [88.4, 91.7] | 86.4 [84.3, 88.3] | 83.5 [81.3, 85.8] | 87.3 [85.3, 89.2] | 84.2 [81.8, 86.2] |
| SFT + record, trained and read with the rule | seen merchant | 91.9 [89.6, 94.1] | 90.7 [88.2, 93] | 94.3 [92.3, 96.1] | 95.3 [93.6, 97] | 95 [93.2, 96.6] |
| SFT + record, trained and read with the rule | unseen merchant | 88.5 [86, 91.1] | 82.6 [79.5, 85.5] | 73.9 [70.3, 77.6] | 80 [76.8, 82.9] | 74.5 [71, 78.1] |
| SFT + record, trained and read with the rule | unseen, renamed | 75 [69.6, 80.4] | 72.5 [67.1, 77.9] | 67.9 [62.1, 73.8] | 68.8 [62.1, 74.6] | 70.4 [64.6, 75.8] |
| SFT + record, trained and read with the rule | unseen, new word | 92.1 [87.9, 96.4] | 70 [62.9, 77.1] | 46.4 [38.6, 55] | 70 [62.1, 77.1] | 55 [46.4, 63.6] |
| SFT + record, trained and read with the rule | DB-only merchants | 97.6 | 91.1 | 75.6 | 77.2 | 75.6 |

**Table 47.2: each rule paired per item with the fixed block for the same arm (share of items with the same prediction; both right; either right; net gain in points)**

| arm | recent | nearest | transact | cluster |
|---|---|---|---|---|
| Instruct base, no record | same 43.8, both 21.0, either 43.6, +2.2 | same 37.1, both 24.8, either 60.1, +22.4 | same 41.1, both 24.0, either 54.5, +16.0 | same 38.7, both 25.7, either 61.4, +24.7 |
| Instruct base + record | same 58.3, both 43.9, either 72.2, +0.1 | same 55.4, both 45.9, either 79.3, +9.2 | same 58.6, both 48.7, either 77.4, +10.0 | same 59.8, both 48.4, either 77.8, +10.2 |
| SFT no DB (fixed-shot adapter), rule shots at test | same 70.3, both 48.1, either 64.0, -2.1 | same 66.4, both 52.7, either 74.5, +13.0 | same 69.0, both 53.9, either 74.0, +13.8 | same 66.1, both 52.5, either 75.1, +13.5 |
| SFT + record (fixed-shot adapter), rule shots at test | same 93.5, both 85.9, either 92.0, -2.4 | same 93.7, both 87.4, either 93.6, +0.6 | same 94.5, both 88.9, either 94.2, +2.8 | same 94.9, both 88.3, either 93.3, +1.3 |
| SFT no DB, trained and read with the rule | same 52.4, both 41.8, either 68.8, -3.6 | same 47.1, both 40.9, either 73.7, +0.4 | same 43.2, both 37.4, either 73.6, -3.1 | same 52.1, both 46.2, either 74.9, +7.0 |
| SFT + record, trained and read with the rule | same 88.5, both 82.7, either 93.9, -3.7 | same 83.5, both 79.1, either 94.6, -6.6 | same 85.7, both 81.8, either 95.6, -2.9 | same 82.4, both 78.8, either 95.6, -5.9 |

**Table 47.3: seen items by whether the query's merchant is among the 24 shots (accuracy %, share of items in the group), and all items by whether the gold label is among the shot labels**

| arm | group | fixed | recent | nearest | transact | cluster |
|---|---|---|---|---|---|---|
| Instruct base, no record | seen, merchant among the shots | 60.6 (14%) | 82.9 (10%) | 95.5 (38%) | 79.2 (38%) | 93.9 (38%) |
| Instruct base, no record | seen, merchant not among the shots | 26.3 (34%) | 27.1 (37%) | 32.8 (10%) | 31.9 (10%) | 34.8 (10%) |
| Instruct base, no record | gold label among the shot labels | 31.5 (99%) | 40.1 (67%) | 69.6 (69%) | 47.5 (99%) | 64.2 (82%) |
| Instruct base, no record | gold label not among the shot labels | 0.0 (1%) | 19.5 (33%) | 17.2 (31%) | 23.1 (1%) | 18.3 (18%) |
| Instruct base + record | seen, merchant among the shots | 78.1 (14%) | 90.2 (10%) | 95.3 (38%) | 86.7 (38%) | 90.3 (38%) |
| Instruct base + record | seen, merchant not among the shots | 55.1 (34%) | 54.1 (37%) | 55.2 (10%) | 56.0 (10%) | 60.0 (10%) |
| Instruct base + record | gold label among the shot labels | 58.5 (99%) | 66.3 (67%) | 80.7 (69%) | 68.6 (99%) | 75.6 (82%) |
| Instruct base + record | gold label not among the shot labels | 8.3 (1%) | 41.1 (33%) | 36.4 (31%) | 15.4 (1%) | 34.7 (18%) |
| SFT no DB (fixed-shot adapter), rule shots at test | seen, merchant among the shots | 96.2 (14%) | 96.7 (10%) | 99.1 (38%) | 98.9 (38%) | 99.3 (38%) |
| SFT no DB (fixed-shot adapter), rule shots at test | seen, merchant not among the shots | 45.9 (34%) | 42.9 (37%) | 44.0 (10%) | 46.6 (10%) | 44.3 (10%) |
| SFT no DB (fixed-shot adapter), rule shots at test | gold label among the shot labels | 57.2 (99%) | 61.6 (67%) | 81.4 (69%) | 71.2 (99%) | 78.3 (82%) |
| SFT no DB (fixed-shot adapter), rule shots at test | gold label not among the shot labels | 50.0 (1%) | 41.1 (33%) | 44.2 (31%) | 46.2 (1%) | 35.7 (18%) |
| SFT + record (fixed-shot adapter), rule shots at test | seen, merchant among the shots | 96.2 (14%) | 99.2 (10%) | 99.3 (38%) | 98.9 (38%) | 98.9 (38%) |
| SFT + record (fixed-shot adapter), rule shots at test | seen, merchant not among the shots | 90.2 (34%) | 87.2 (37%) | 81.9 (10%) | 81.9 (10%) | 84.3 (10%) |
| SFT + record (fixed-shot adapter), rule shots at test | gold label among the shot labels | 91.0 (99%) | 92.5 (67%) | 96.2 (69%) | 93.4 (99%) | 94.9 (82%) |
| SFT + record (fixed-shot adapter), rule shots at test | gold label not among the shot labels | 8.3 (1%) | 78.1 (33%) | 78.3 (31%) | 53.8 (1%) | 75.6 (18%) |
| SFT no DB, trained and read with the rule | seen, merchant among the shots | 96.2 (14%) | 96.7 (10%) | 98.6 (38%) | 97.5 (38%) | 98.6 (38%) |
| SFT no DB, trained and read with the rule | seen, merchant not among the shots | 45.9 (34%) | 47.7 (37%) | 34.5 (10%) | 31.0 (10%) | 38.3 (10%) |
| SFT no DB, trained and read with the rule | gold label among the shot labels | 57.2 (99%) | 60.0 (67%) | 72.4 (69%) | 54.4 (99%) | 72.3 (82%) |
| SFT no DB, trained and read with the rule | gold label not among the shot labels | 50.0 (1%) | 40.1 (33%) | 23.6 (31%) | 15.4 (1%) | 26.8 (18%) |
| SFT + record, trained and read with the rule | seen, merchant among the shots | 96.2 (14%) | 100.0 (10%) | 99.1 (38%) | 99.8 (38%) | 99.3 (38%) |
| SFT + record, trained and read with the rule | seen, merchant not among the shots | 90.2 (34%) | 88.1 (37%) | 75.9 (10%) | 78.4 (10%) | 78.3 (10%) |
| SFT + record, trained and read with the rule | gold label among the shot labels | 91.0 (99%) | 90.3 (67%) | 92.9 (69%) | 87.8 (99%) | 91.1 (82%) |
| SFT + record, trained and read with the rule | gold label not among the shot labels | 8.3 (1%) | 78.4 (33%) | 62.2 (31%) | 38.5 (1%) | 53.1 (18%) |

**Table 47.4: training cost of the rule-trained adapters (minutes, final loss over the last steps)**

| adapter | fixed | recent | nearest | transact | cluster |
|---|---|---|---|---|---|
| SFT none | 17.7 min, loss 0.141 | 18.4 min, loss 0.072 | 19.2 min, loss 0.015 | 19.2 min, loss 0.036 | 19.4 min, loss 0.027 |
| SFT ret | 18.2 min, loss 0.035 | 18.7 min, loss 0.015 | 19.5 min, loss 0.006 | 19.3 min, loss 0.001 | 25.1 min, loss 0.001 |

### 47.1 At test: a chosen block puts the merchant's own rows in the prompt, the model copies them, and nothing else moves

Read with the rule's shots, the fixed-shot adapters and the untrained base gain 9 to 25 points over the whole set: the no-DB adapter 57.1 to 70.1 / 70.9 / 70.6 (nearest / transact / cluster), the base 31.2 to 53.6 / 47.2 / 55.9, and the recent block, a different random 24, changes nothing (55.0, 33.4), so the gain is the choice and not the variety. Table 47.5 says where it is. In the in-history group the no-DB adapter goes from 65.3 to 99.1 and the base from 37.6 to 95.3: a similarity rule puts the merchant's own rows among the shots for all of these items (38% of all items, against 14% under the frozen block, Table 47.3) and the model copies the label, as section 38 found it does at 96 when a frozen block happens to contain one. On the items decided by the merchant's category the rules leave the no-DB adapter where it was (59.3 fixed; 58.5 to 60.5 under every rule) and the base within three points except for cluster (35.4 against 28.5), and the 115 truncated items and the 103 split ones move inside their noise. The copy is exactly the lookup: the nearest-row label alone reads 98.6 on the in-history group and 47.5 overall, the fixed-shot adapter with that lookup in front of it (the hybrid column) 65.3, and the adapters reading nearest shots are the same number with or without the lookup (70.1 and 70.0). A 3B model with retrieved shots is, on REAL-6, a merchant lookup plus the model it already was.

The record-in-prompt adapter is at the set's ceiling (97.4 on the category-determined items, a coin flip on the split ones), so the rules can only rearrange it: transact reads 93.0 against 90.2 (paired: 94.5% of the predictions the same, +2.8 net), from the in-history group (94.1 to 98.6) and the split items (50.5 to 60.2 on 103 items, inside a coin flip's noise); nearest reads 90.8 and loses 4 points on the category-determined items (93.3), and the reason is in Table 47.3: a pure similarity block leaves the gold label out of the shots for 31% of the items (cluster 18%, transact and the frozen block 1%). Section 33 found that no arm picks a label no demonstration carries; that holds for the untrained base (17 to 23 when the gold label is missing from the shots, 0 in the frozen block's 1%), while the record adapter picks it at 78 because the category list plus the record is enough once it has been trained on the format. TransAct's rule keeps the category coverage of the frozen block and adds the merchant's rows, which is why it is the best test-time block for both adapters.

### 47.2 Trained with the rule: the adapter learns to copy and loses the merchants it cannot copy

Trained with the rule's shots, the adapters are worse than the fixed-shot adapters read with the same shots on every rule: no DB 53.5 / 57.5 / 53.9 / 64.0 (recent / nearest / transact / cluster) against 55.0 / 70.1 / 70.9 / 70.6, record 86.4 / 83.5 / 87.3 / 84.2 against 87.8 / 90.8 / 93.0 / 91.4. The in-history group is at 98 to 99.5 either way; the loss is on everything the shots cannot answer. The category-determined items fall from 59.3 to 36.1 (nearest) and 29.5 (transact) without the DB, and from 97.4 to 79.2 / 87.2 / 80.6 with the record; the DB-only merchants, the owner's second goal, from 97.6 to 75.6 / 77.2 / 75.6 for the record adapter, the same size of loss that section 43's ambiguous records cost. Training loss says why: 0.001 to 0.015 at the end against 0.035 to 0.141 for the fixed-shot recipe. Every training query is a history row, and a merchant with several rows has its siblings in the pool, so under a similarity rule nearly every training target is sitting in the prompt with its label; the adapter learns that the answer is the label of the matching shot and stops reading the record and the category list. The random-block `recent` training costs 3.6 points against its own fixed-shot baseline, which is the expected cost of a different random block and not the copy. Row 46 withholds the query's merchant from its own training pool in 60% of the episodes (the share of test items whose merchant the history lacks), so the training distribution matches the test's mix of copyable and uncopyable items.

### 47.3 Cost

The rules cost one retriever pass over the user's 300 rows per user and one per query; the rule-trained adapters train in 18 to 25 minutes (fixed 18), and scoring a rule is the same 25 minutes as the fixed block. The chain ran 12.7 GPU hours for 20 scorings and 8 trainings.

### 47.4 What the step says

REAL-9 asked whether the fixed shots leave accuracy on the table and which shots a real history should supply. On REAL-6 the answer is narrower than the whole-set numbers suggest. The rules' gain (up to 25 points) is the user's own label for the same merchant, which a lookup on the history gives at 98.6 without a model; on everything the lookup cannot answer, no rule moves a trained categoriser at test, and training with retrieved shots teaches the adapter to copy and costs 20 to 30 points on the merchants it has not seen, the DB-only ones included. For a production design that means: put the merchant lookup, or a nearest-row block, in front of the model for merchants the user has labelled; at test use TransAct's block (recent plus the nearest per category), which keeps every category represented and gave the best record-in-prompt number (93.0 against 90.2, paired +2.8, one adapter); and do not train with retrieved shots unless the query's merchant is withheld from its own pool (row 46). What REAL-6 cannot say is whether shots beat a lookup when they disagree: its users never relabel, never file one merchant two ways and never split a category on anything observable, so the in-history group is a lookup by construction and the recent rule has no time to follow. Those are row 43's cells.

Not done: seeds (the trained adapters are one seed each; the test-only arms are paired on the same adapter and need none), a rule with more than 24 shots, the gold-label-aware block at training time. Cost: 12.7 GPU hours.

## 48. The REAL-6 audit: the seen cells are a lookup, the unseen cells are the merchant's standard category, the record-in-prompt categorisers sit at the set's ceiling of about 94, and the section 45 whole-set gain is mostly on items the DB cannot explain (REAL-13)

*PLAN step 45. Code: `ai_experiments.real6_cells` (the corrected groups, the nearest-row lookup cached in `results/real6_nn1.json`, the lookup-then-model hybrid, a user bootstrap), `real6_eval.summarize` (now also `_uci`, the interval with users resampled), `scripts/real6_audit_tables.py`. Every number is re-read from the saved per-item records of sections 37 to 47; nothing was rescored or retrained. CPU only.*

When the queue changed hands during row 41, a review of the evaluation set found four properties that the REAL-6 cell names hide (QUESTIONS.md REAL-13). First, `real6.build` chooses each history merchant's test rows before it cuts the shuffled history to 300 rows, so 115 of the 559 items labelled "seen" have no row of their merchant left in the history the model is given. Second, every user files every merchant under one label (0 of 1,026 user-merchant pairs carry two), so an item whose merchant is in the history is answered by looking the merchant up: the label of the history row nearest to the statement under the row 37 retriever is right on 98.6% of those 444 items, from the string alone. Third, every scheme is built from the twelve standard categories by merging, renaming and splitting, and a split assigns merchants to its halves at random; so of the 620 items whose merchant the history never had, 509 are decided by the merchant's standard category (the history shows which of the user's names that category became), 103 fall in a split category with nothing observable to say which half, and 8 in neither. Fourth, the intervals resample items, but the set samples 20 users and a user's items are not independent. Table 48.1 gives the groups, Table 48.2 re-reads every REAL-6 run of the report in them, and Table 48.3 the seeded arms.

**Table 48.1: REAL-6 in the corrected groups (`ai_experiments.real6_cells`); lookup = the label of the user's history row nearest to the statement under the row 37 retriever**

| group | items | of which DB-only merchants | labelled level | lookup accuracy | what decides the item |
|---|---|---|---|---|---|
| in history | 444 | 108 | seen | 98.6 | the user's own label for this merchant (one label per user-merchant pair) |
| labelled seen, not in history | 115 | 27 | seen | 15.7 | the merchant's category, as for an unseen merchant: the 300-row cut removed its rows |
| determined by category | 509 | 113 | unseen | 19.1 | the merchant's standard category, renamed or merged by the user's scheme |
| split category | 103 | 9 | unseen | 6.8 | which half of a split category the user put this merchant in: nothing observable |
| other (the history shows a different label for the category) | 8 | | unseen | 0.0 | |

**Table 48.2: every REAL-6 run of sections 37 to 47 in the corrected groups (accuracy %; DB-only = determined or split items whose merchant no user labelled in training; interval over items, then over users; sum = the option rule summing token log-probabilities instead of averaging them, LLM runs only; hybrid = the lookup when its cosine is at least 0.8, else the run)**

| sec. | run | in history | labelled seen, not in history | determined by category | split category | DB-only | all | item interval | user interval | sum rule, all | hybrid, all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 37 | Qwen2.5-3B base, 24 shots | 32.9 | 24.3 | 21.8 | 13.6 | 23.8 | 25.4 | [23.0, 28.1] | [21.4, 29.4] | 28.1 | 43.2 |
| 37 | Qwen2.5-3B base + record | 57.4 | 40.9 | 47.9 | 25.2 | 48.4 | 48.6 | [45.7, 51.6] | [42.4, 55.0] | 56.8 | 59.7 |
| 37 | Instruct, 24 shots | 37.6 | 30.4 | 28.5 | 20.4 | 30.3 | 31.2 | [28.7, 33.8] | [27.7, 34.7] | 32.1 | 47.7 |
| 37 | Instruct + record | 62.2 | 60.0 | 60.1 | 31.1 | 48.4 | 58.0 | [55.1, 60.8] | [53.3, 62.2] | 60.7 | 68.0 |
| 37 | MiniLM prototype, 24 shots | 37.4 | 12.2 | 15.9 | 7.8 | 13.9 | 22.8 | [20.4, 25.2] | [19.3, 26.2] | - | 39.0 |
| 37 | MiniLM prototype, full history | 66.2 | 14.8 | 16.7 | 3.9 | 16.4 | 33.9 | [31.3, 36.5] | [31.1, 37.1] | - | 40.6 |
| 37 | MiniLM mix | 62.2 | 22.6 | 27.3 | 2.9 | 26.2 | 37.8 | [34.9, 40.6] | [33.4, 42.3] | - | 45.9 |
| 37 | bge prototype, full history | 68.9 | 24.3 | 18.7 | 1.0 | 17.2 | 36.5 | [33.6, 39.0] | [33.5, 40.3] | - | 42.6 |
| 37 | bge mix | 65.3 | 32.2 | 30.1 | 5.8 | 27.9 | 41.2 | [38.3, 43.9] | [37.8, 44.6] | - | 48.6 |
| 38 | bge tuned across users, full history | 86.0 | 57.4 | 79.6 | 38.8 | 38.5 | 75.7 | [73.2, 77.9] | [71.5, 80.2] | - | 78.8 |
| 38 | bge tuned + record on the query | 91.7 | 71.3 | 89.2 | 42.7 | 71.3 | 83.7 | [81.6, 85.8] | [78.4, 88.7] | - | 85.4 |
| 38 | bge tuned + records in training | 85.1 | 54.8 | 78.8 | 21.4 | 39.3 | 73.3 | [70.8, 75.6] | [69.2, 77.1] | - | 76.5 |
| 38 | SFT no DB | 65.3 | 40.9 | 59.3 | 27.2 | 51.6 | 57.1 | [54.3, 59.9] | [53.6, 60.3] | 57.3 | 65.3 |
| 38 | SFT, records in the weights (1 pass) | 65.3 | 42.6 | 53.8 | 25.2 | 54.1 | 54.5 | [51.7, 57.3] | [50.9, 58.2] | 54.7 | 62.9 |
| 38 | SFT, records in the weights (3.3 passes) | 71.2 | 49.6 | 69.0 | 51.5 | 72.1 | 66.3 | [63.9, 69.0] | [62.1, 70.0] | 67.2 | 72.3 |
| 38 | SFT + record in prompt | 94.1 | 83.5 | 97.4 | 50.5 | 97.5 | 90.2 | [88.4, 91.7] | [86.0, 93.5] | 90.3 | 91.3 |
| 43 | SFT + record, ambiguous DB (trained on it) | 91.7 | 85.2 | 93.5 | 43.7 | 88.5 | 87.4 | [85.7, 89.2] | [83.1, 91.1] | 87.4 | 89.3 |
| 43 | SFT + record, retrieved top-1 | 93.7 | 83.5 | 97.2 | 50.5 | 97.5 | 89.9 | [88.0, 91.5] | [85.8, 93.2] | 90.1 | 91.1 |
| 44 | SFT no DB, transformers + peft (hf scorer) | 65.8 | 43.5 | 63.5 | 35.9 | 57.4 | 59.5 | [57.0, 62.3] | [56.6, 62.1] | 58.9 | 67.0 |
| 44 | SFT + record, transformers + peft (hf scorer) | 94.6 | 82.6 | 94.5 | 42.7 | 92.6 | 88.5 | [86.6, 90.2] | [84.5, 91.8] | 88.5 | 89.7 |
| 45 | Instruct, chat template | 13.3 | 9.6 | 9.2 | 1.0 | 8.2 | 10.0 | [8.4, 11.8] | [7.0, 13.0] | 10.2 | 34.0 |
| 45 | Instruct + record, chat template | 69.1 | 61.7 | 66.8 | 25.2 | 62.3 | 63.2 | [60.5, 66.0] | [58.0, 68.2] | 63.3 | 71.3 |
| 45 | SFT no DB, chat template | 65.5 | 40.0 | 61.9 | 35.9 | 54.1 | 58.9 | [56.2, 61.8] | [55.4, 62.1] | 58.8 | 67.3 |
| 45 | SFT + record, chat template | 95.5 | 85.2 | 92.9 | 59.2 | 98.4 | 90.0 | [88.3, 91.6] | [85.5, 93.8] | 89.5 | 90.9 |
| 46 | FastFit bge, 24 shots | 46.2 | 26.1 | 24.8 | 5.8 | 24.6 | 31.3 | [28.7, 33.9] | [27.8, 35.4] | - | 44.6 |
| 46 | FastFit bge, full history | 86.9 | 24.3 | 30.3 | 9.7 | 27.9 | 49.0 | [46.1, 51.8] | [46.0, 52.3] | - | 49.0 |
| 46 | FastFit bge + record, full history | 100.0 | 88.7 | 98.8 | 36.9 | 95.9 | 92.2 | [90.7, 93.6] | [88.0, 95.6] | - | 92.1 |
| 46 | logistic head, frozen bge, C=100 | 87.6 | 17.4 | 21.0 | 1.0 | 14.8 | 43.9 | [41.1, 46.6] | [40.5, 47.3] | - | 43.9 |
| 46 | logistic head + record, C=100 | 100.0 | 81.7 | 89.0 | 35.9 | 93.4 | 87.2 | [85.2, 89.1] | [82.7, 91.0] | - | 87.1 |
| 47 | SFT no DB, transact shots at test | 98.9 | 46.1 | 59.5 | 34.0 | 50.8 | 70.9 | [68.3, 73.5] | [67.0, 74.4] | 71.1 | 70.8 |
| 47 | SFT + record, transact shots at test | 98.6 | 82.6 | 97.4 | 60.2 | 96.7 | 93.0 | [91.3, 94.3] | [89.4, 95.7] | 92.7 | 93.0 |
| 47 | SFT + record, trained with transact shots | 99.5 | 79.1 | 87.2 | 46.6 | 77.0 | 87.3 | [85.3, 89.2] | [83.9, 90.2] | 87.2 | 87.2 |

**Table 48.3: the seeded arms in the corrected groups (mean +- sd over seeds 0, 1, 2; accuracy %)**

| arm | in history | labelled seen, not in history | determined by category | split category | DB-only | all |
|---|---|---|---|---|---|---|
| SFT no DB (sec. 43) | 68.5 +- 2.8 | 44.6 +- 3.9 | 62.7 +- 3.1 | 35.6 +- 7.4 | 49.5 +- 3.8 | 60.5 +- 3.0 |
| SFT + record (sec. 43) | 94.1 +- 0.6 | 84.3 +- 0.9 | 95.8 +- 1.4 | 45.3 +- 7.4 | 91.5 +- 5.5 | 89.3 +- 0.8 |
| records in the weights, 3.3 passes (sec. 43) | 70.9 +- 0.4 | 49.9 +- 2.2 | 64.9 +- 4.6 | 43.4 +- 7.2 | 59.3 +- 12.0 | 63.8 +- 2.3 |
| records in the weights, 6.7 passes (sec. 45) | 81.9 +- 0.9 | 62.6 +- 4.8 | 82.0 +- 3.5 | 44.0 +- 12.8 | 68.9 +- 5.4 | 76.8 +- 2.1 |
| records in the weights, 13.3 passes (sec. 45) | 84.1 +- 0.6 | 68.4 +- 2.7 | 85.1 +- 2.4 | 30.7 +- 5.9 | 64.5 +- 5.5 | 78.3 +- 0.3 |

### 48.1 The seen cells were a lookup, and the models were below it

Every "seen" number in sections 37 to 47 mixes 444 lookup items with 115 unseen ones. In the in-history group alone the nearest-row lookup reads 98.6; the untrained instruct base with its frozen 24 shots 37.6, the full-history prototypes of section 37 66 to 69 (not the 55.6 / 59.7 section 37.3 set as the target for labelled merchants), the SFT categoriser 65.3 without the record and 94.1 with it, the tuned bge encoder 86.0 and 91.7. Only the per-user classifiers with the record on the statement (section 46: FastFit 100, a logistic head on frozen bge 100) and the categorisers reading nearest or TransAct shots (section 47: 98.6 to 99.5) reach the lookup, and those shots are the lookup inside the prompt. The hybrid column puts the lookup in front wherever its cosine clears 0.8 (74% of the in-history items, none of the others): the no-record SFT goes from 57.1 to 65.3, the tuned bge from 75.7 to 78.8, the record SFT from 90.2 to 91.3. For merchants a user has labelled, a production system should look the label up and not ask a model, and REAL-6 cannot say whether a model should ever overrule the lookup, because its users never change their minds (row 43's set).

### 48.2 The unseen cells were the standard category, and the record-in-prompt arms are at the ceiling

On the 509 category-determined items the record-in-prompt categorisers read 95.8 +- 1.4 over three seeds (97.4 for seed 0), FastFit with the record 98.8, and on the 103 split items every run in the report is between 1 and 60, which is what a coin flip over two halves gives for trained models on 103 items. The attainable ceiling for the whole set is therefore about 94 (the lookup group at 99, the 624 category-determined and truncated items at 97 or so, the split ones at 50): the best systems of sections 46 and 47 (per-user FastFit with the record 92.2, the record SFT with TransAct shots 93.0) are at it, and no variant of the record arm can be ranked on this set any more. That includes the comparisons of sections 43 to 47 that differ by two to three points: the retrieved-record, hf-trainer and chat-template variants all sit inside the record arm's user interval ([86.0, 93.5]). The DB-only results, the owner's second goal, stand as reported: the DB-only merchants in the corrected groups read 97.5 with the record in the prompt, 91.5 +- 5.5 over seeds, and the no-DB arm 49.5 +- 3.8; they were already restricted to unseen levels, and only 9 of the 122 are in split categories.

### 48.3 The section 45 whole-set gain is mostly not the DB

Section 45 reported the whole set going from 60.5 (no DB) to 76.8 and 78.3 with the records in the weights at 6.7 and 13.3 passes, and noted that the history exposure doubled with the DB exposure. The corrected groups show where the gain is: the in-history items rise from 68.5 to 81.9 and 84.1 and the category-determined items from 62.7 to 82.0 and 85.1, while the DB-only merchants rise from 49.5 to 68.9 and 64.5. An in-history item's label is in the user's training rows, and a category-determined merchant is, for up to three quarters of those items (the 396 that are not DB-only), one other users labelled in training; the records cannot be what teaches either, and 800 or 1,600 steps over the users' histories (against 200 for the no-DB arm) can. So the no-DB categoriser is probably under-trained by a factor of four, which would be the cheapest gain the report has found for the production shape, and row 47 (the no-DB arm at 800 and 1,600 steps) moves up the queue from a control to a recipe question. The DB-only gain (+19 at 6.7 passes) is the part of section 45 that the records own.

### 48.4 Smaller corrections

The option rule: the untrained bases read 1 to 8 points higher when the options are scored by summed rather than mean token log-probability (the instruct base with the record 58.0 to 60.7, the base with the record 48.6 to 56.8), so section 37's untrained columns understate them; every trained categoriser moves by under a point. The user intervals are 1.3 to 2.2 times as wide as the item intervals (record SFT [88.4, 91.7] to [86.0, 93.5]); every REAL-6 table from here on carries the user interval, and differences under about four points on the whole set between separately trained adapters should be read as ties. Section 37.2's "elsewhere in the history" row (399 items at 20 to 26 for the untrained LLMs) contains the 115 truncated items; without them the untrained model is still at its unseen-merchant level on merchants that are in the history but not among the shots, so the reading (the model only uses what is in the prompt) stands.

### 48.5 What the step says

REAL-6 measured three things well: whether a categoriser reads the user's scheme at all (the coined names), whether a fact-DB record reaches merchants no user labelled (the DB-only merchants), and whether training across users carries a merchant's category to another user's scheme. It cannot measure the rest of the owner's task. The seen cells are a lookup, so the in-context route's advantage over a lookup table (following a user's relabelling, filing one merchant two ways, reasons the history shows but the merchant does not) is untested; and the unseen cells reduce to knowing the merchant's standard category, which the record gives, so the record-in-prompt arms are at the set's ceiling and further variants of them are ties. The queue follows from that: held-out users (row 42) still has something to measure on REAL-6 (the schemes are new even if the categories are not); row 43's set has to be built so that neither a lookup nor the standard category decides an item; the collaborative record (row 44) waits for that set; and the no-DB recipe question of 48.3 (row 47) goes first, because it may change every no-record number and costs a few GPU hours.

## 49. The no-DB categoriser was under-trained: four times the steps takes it from 60.5 to 73.3 (eight times: 76.7) with ARC-Easy and MMLU no lower, the gain on the merchants users labelled and none on the merchants only the DB knows; at matched history exposure the records in the weights still add 5 to 9 points overall and 12 to 20 on the DB-only merchants (REAL-8, REAL-13)

*PLAN step 47. Code: `scripts/chains/chain_r47.sh` (log `logs/gpu47.log`, 15 steps, 9.2 hours), `scripts/nodb_steps_tables.py`; no code change to the trainer (`STEPS`, `SEED`, `RUN_TAG`). Adapters `categoriser_Qwen2.5-3B-Instruct_none_st{400,800,1600}_s<k>_lora`, results `results/real6_*_none_st*_lora.json`, `results/items2_*_none_st*_lora.json`. QLoRA on the 4-bit base, scored on it, as sections 37 to 48.*

Section 48.3 found that the records-in-the-weights arms of section 45 gained 13 to 22 points on items the records cannot explain (merchants the user labelled, merchants other users labelled), and that those arms had trained for four and eight times the no-DB arm's 200 steps. This step trains the no-DB categoriser, unchanged otherwise, for 400 steps (one seed; the history exposure of section 45's 800-step arm, whose sequences were half records), 800 steps (three seeds; the exposure of the 1,600-step arm) and 1,600 steps (one seed), and reads them in section 48's groups and on the step 20 general measures.

**Table 49.1: the no-DB categoriser by training steps (16 sequences per step; accuracy %, mean +- sd where three seeds; groups as REPORT.md 48; DB-only = category-determined or split items whose merchant no user labelled in training; interval = users resampled, seed 0)**

| steps | passes over the 3,000 history episodes (150 per user) | in history | labelled seen, not in history | determined by category | split category | DB-only | all | seed 0 all [user interval] |
|---|---|---|---|---|---|---|---|---|
| 200 (3 seeds) | 1.1 | 68.5 +- 2.8 | 44.6 +- 3.9 | 62.7 +- 3.1 | 35.6 +- 7.4 | 49.5 +- 3.8 | 60.5 +- 3.0 | 57.1 [53.6, 60.3] |
| 400 (1 seed) | 2.1 | 75.0 | 58.3 | 71.5 | 27.2 | 49.2 | 67.4 | 67.4 [63.7, 70.9] |
| 800 (3 seeds) | 4.3 | 80.9 +- 0.8 | 58.0 +- 1.3 | 78.5 +- 3.8 | 33.7 +- 5.9 | 52.7 +- 2.1 | 73.3 +- 2.3 | 71.7 [67.5, 75.5] |
| 1,600 (1 seed) | 8.5 | 83.1 | 70.4 | 82.9 | 27.2 | 55.7 | 76.7 | 76.7 [72.6, 80.9] |

**Table 49.2: the records' own contribution, at matched history exposure (accuracy %, mean +- sd over seeds)**

| history exposure | arm | in history | labelled seen, not in history | determined by category | split category | DB-only | all |
|---|---|---|---|---|---|---|---|
| 6,400 history sequences | no DB, 400 steps | 75.0 | 58.3 | 71.5 | 27.2 | 49.2 | 67.4 |
| 6,400 history sequences | records at 50%, 800 steps (6.7 passes) | 81.9 +- 0.9 | 62.6 +- 4.8 | 82.0 +- 3.5 | 44.0 +- 12.8 | 68.9 +- 5.4 | 76.8 +- 2.1 |
| 12,800 history sequences | no DB, 800 steps | 80.9 +- 0.8 | 58.0 +- 1.3 | 78.5 +- 3.8 | 33.7 +- 5.9 | 52.7 +- 2.1 | 73.3 +- 2.3 |
| 12,800 history sequences | records at 50%, 1,600 steps (13.3 passes) | 84.1 +- 0.6 | 68.4 +- 2.7 | 85.1 +- 2.4 | 30.7 +- 5.9 | 64.5 +- 5.5 | 78.3 +- 0.3 |

**Table 49.3: general measures by training steps (exp_items_v2 on the same adapters; % ; mean over seeds where three)**

| steps | ARC-Easy | MMLU | ICL natural | ICL symbol | ICL v2 natural | ICL v2 symbol |
|---|---|---|---|---|---|---|
| 200 (1) | 76.0 | 51.0 | 86.5 | 76.6 | 88.5 | 64.5 |
| 400 (1) | 77.0 | 53.5 | 88.0 | 70.8 | 89.6 | 61.5 |
| 800 (3) | 78.2 | 52.8 | 86.3 | 72.9 | 88.2 | 66.1 |
| 1,600 (1) | 80.5 | 53.5 | 87.0 | 74.5 | 87.0 | 69.8 |

**Table 49.4: training cost (minutes, final loss; seed 0)**

| steps | minutes | final loss |
|---|---|---|
| 200 | 17.7 | 0.141 |
| 400 | 36.6 | 0.047 |
| 800 | 77.6 | 0.028 |
| 1,600 | 146.8 | 0.0 |

### 49.1 More steps teach the users' labels and other users' labels, not the merchants nobody labelled

The whole set goes 60.5 +- 3.0 (200 steps, three seeds) to 67.4 (400), 73.3 +- 2.3 (800, three seeds) and 76.7 (1,600), with the seed spread at 800 steps no wider than at 200 and the user intervals of 200 and 800 steps not overlapping ([53.6, 60.3] against [67.5, 75.5] for seed 0). The gain sits where the training data has the answer: on merchants in the user's history 68.5 to 80.9 and 83.1, on the category-determined merchants (three quarters of them labelled by other users in training) 62.7 to 78.5 and 82.9, and on the 115 truncated items, whose merchants other users labelled too, 44.6 to 58.0 and 70.4. The DB-only merchants, which no training row carries, move from 49.5 +- 3.8 to 52.7 +- 2.1 and 55.7, inside the seed spread: what the model knows about a merchant nobody labelled does not grow with more passes over other merchants. The split items stay at a coin flip or below. The curve is flattening (+6.9, +5.9, +3.4 per doubling on the whole set) and not yet flat at 8.5 passes over the history episodes. The general measures do not pay for it: ARC-Easy 76.0 at 200 steps and 78.2 (three seeds) at 800, 80.5 at 1,600 (instruct base 72.5, Table 38.3), MMLU 51 to 54, the ICL suite within its seed noise; training loss reaches 0.028 at 800 steps and rounds to zero at 1,600. So the 200-step recipe used for every categoriser in sections 38 to 48 stopped at about a quarter of what the no-DB arm can use, and the lookup-group number it was compared with (98.6) is partly a comparison with an under-trained model: at 1,600 steps the model alone reads 83.1 on merchants the user labelled.

### 49.2 The records' own share of section 45

At the same number of history sequences, the records in the weights still add to the whole set: 76.8 +- 2.1 against 67.4 at 6,400 history sequences, 78.3 +- 0.3 against 73.3 +- 2.3 at 12,800; and on the DB-only merchants, which only the records can reach, 68.9 +- 5.4 against 49.2 and 64.5 +- 5.5 against 52.7 +- 2.1. So about half of section 45's whole-set gain over the 200-step no-DB arm (60.5 to 76.8 and 78.3) was history exposure and half the records, and the DB-only gain (+12 to +20) is the records'. The parametric arm's DB-only number is still far under the record in the prompt (97.5, section 48), and section 45's recommendation stands.

### 49.3 What the step says

The no-DB categoriser was under-trained by a factor of four to eight: 800 steps (77 minutes) is worth 13 points on REAL-6 and 1,600 steps (147 minutes) 16, all of it on merchants some user labelled, with no general-ability cost. The record-in-prompt arm was not retrained here (it sits at the set's ceiling, section 48), and the DB-only merchants do not gain from steps, so the record in the prompt remains the route for them. Consequences for the queue: row 42's held-out-user adapters train at 800 steps (the three-seed point; 1,600 is four hours per adapter), and whether the extra steps are learning the users' schemes or memorising the twenty users' merchant labels is exactly what the held-out users will show, since a held-out user's labelled merchants were never in training. Not run: the record arm at 800 steps, a second and third seed at 400 and 1,600, and steps beyond 1,600.

## 50. Calibration and the auto-apply operating point: under the sum rule one temperature per arm calibrates every categoriser (ECE 3 to 4), the record arm is nearly calibrated as trained (T 1.4) and the no-record arm over-confident (T 2.8); at 95% precision the record categoriser can auto-apply 82% of a trained user's transactions and 60% of a new user's, the no-record one 56% and 48%; a foreign temperature ruins the probabilities and leaves the operating point intact; and the split categories are where confident errors concentrate (STAT-4)

*PLAN step 49. Code: `ai_experiments.calibration` (softmax over the user's categories, a bounded 1-D temperature fit after `../jqv/jqv/calibration.py`, ECE, Brier, the risk-coverage curve, AURC and threshold selection after `../kev/kev/metrics.py`), `scripts/calibration_tables.py`. CPU only, from the saved per-option log-probabilities of the per-item records; nothing rescored. Row 42's fold adapters enter as their plain folds; the rename-augmented folds are added to the same tables when row 42's chain finishes (section 51).*

REAL-11 asks for a correction rate, and a production categoriser needs a rule for when a label is applied without asking the user. That needs probabilities that mean what they say, or at least a confidence whose ranking holds across users. The Jev rebuilds the owner collected (`references/jqv_analysis.md` 2.4, `reflex_analysis.md` 2.3, `kev_analysis.md` 3) all found raw option distributions over-confident, one temperature fitted on the serving population fixing most of it, and the temperature depending on the task type (knowledge in the weights against evidence in the prompt), not the model. This step measures that on the categoriser arms. Protocol: per item the softmax over the user's categories of the option scores under the mean-per-token rule (what every REAL-6 table reports) and the sum rule; users in four folds (id mod 4, row 42's split); each fold's items are tempered with the temperature fitted by NLL on the other three folds and auto-applied against a threshold chosen on the other three folds' tempered confidences for 95% (and 90%) precision. No user's items fit their own temperature or threshold, and for row 42's fold adapters no user's items trained the adapter either. Intervals resample users.


**Table C.1 (mean rule): calibration and the auto-apply operating point, seed 0 or the merged folds (T = mean of the four leave-users-out fits; ECE in points over 10 bins; AURC in % risk; coverage = share of items auto-applied; oracle = threshold chosen on the same items, out-of-fold = threshold chosen on the other users, with the precision it realised; intervals resample users)**

| arm | T | accuracy | NLL raw / tempered | Brier raw / tempered | ECE raw | ECE tempered | AURC | coverage at 95%, oracle | at 95%, out-of-fold [interval] (precision) | at 90%, oracle | at 90%, out-of-fold (precision) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| instruct base, no record | 1.76 | 31.2 | 2.46 / 2.37 | 0.815 / 0.847 | 8.1 | 11.9 [9.7, 15.2] | 45.4 | 0.3 | 1.7 [0.7, 3.1] (80.0) | 5.5 | 5.9 (90.0) |
| instruct base + record | 0.83 | 58.0 | 1.49 / 1.48 | 0.605 / 0.586 | 15.0 | 8.9 [6.6, 13.1] | 22.9 | 1.8 | 2.2 [0.9, 3.7] (80.8) | 21.5 | 17.0 (89.0) |
| SFT no DB, 200 steps | 1.10 | 57.1 | 1.64 / 1.63 | 0.551 / 0.560 | 7.7 | 10.1 [8.1, 13.7] | 16.8 | 22.9 | 22.6 [18.1, 27.1] (94.8) | 44.5 | 43.6 (90.1) |
| SFT no DB, 800 steps | 1.43 | 71.7 | 1.24 / 1.15 | 0.387 / 0.392 | 4.8 | 8.3 [5.8, 12.2] | 8.2 | 51.4 | 51.3 [44.4, 56.7] (95.0) | 66.3 | 65.9 (90.0) |
| SFT + record | 0.62 | 90.2 | 0.39 / 0.32 | 0.177 / 0.160 | 10.9 | 3.2 [2.1, 7.2] | 2.7 | 74.9 | 79.3 [73.9, 84.9] (94.2) | 100.0 | 99.0 (90.7) |
| SFT + retrieved record | 0.66 | 89.9 | 0.41 / 0.35 | 0.181 / 0.164 | 10.8 | 3.1 [2.3, 6.7] | 2.8 | 75.8 | 79.6 [74.3, 84.9] (94.1) | 99.8 | 98.9 (90.5) |
| SFT no DB, 800 steps, user held out (row 42) | 1.45 | 70.1 | 1.21 / 1.14 | 0.408 / 0.410 | 5.2 | 6.0 [4.1, 9.5] | 9.2 | 43.3 | 45.2 [40.5, 51.0] (94.4) | 61.6 | 61.6 (89.8) |
| SFT + record, user held out (row 42) | 1.02 | 81.6 | 0.71 / 0.73 | 0.278 / 0.282 | 3.6 | 3.5 [2.6, 9.1] | 6.7 | 52.7 | 61.3 [53.9, 68.4] (92.8) | 78.5 | 73.6 (90.7) |

**Table C.1 (sum rule): calibration and the auto-apply operating point, seed 0 or the merged folds (T = mean of the four leave-users-out fits; ECE in points over 10 bins; AURC in % risk; coverage = share of items auto-applied; oracle = threshold chosen on the same items, out-of-fold = threshold chosen on the other users, with the precision it realised; intervals resample users)**

| arm | T | accuracy | NLL raw / tempered | Brier raw / tempered | ECE raw | ECE tempered | AURC | coverage at 95%, oracle | at 95%, out-of-fold [interval] (precision) | at 90%, oracle | at 90%, out-of-fold (precision) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| instruct base, no record | 2.38 | 32.1 | 2.67 / 2.24 | 0.843 / 0.805 | 19.8 | 9.5 [7.0, 12.2] | 43.8 | 5.3 | 4.7 [3.0, 6.5] (89.1) | 8.8 | 8.3 (90.8) |
| instruct base + record | 1.50 | 60.7 | 1.40 / 1.29 | 0.548 / 0.538 | 10.0 | 3.9 [2.7, 7.4] | 20.0 | 3.6 | 8.3 [4.2, 13.3] (89.8) | 25.3 | 24.3 (90.2) |
| SFT no DB, 200 steps | 1.80 | 57.3 | 1.57 / 1.38 | 0.555 / 0.522 | 14.3 | 3.0 [2.6, 7.2] | 15.9 | 30.8 | 31.6 [27.7, 35.8] (94.6) | 47.2 | 46.6 (89.8) |
| SFT no DB, 800 steps | 2.79 | 72.1 | 1.69 / 0.99 | 0.451 / 0.373 | 18.6 | 3.9 [3.2, 6.7] | 7.6 | 55.0 | 55.8 [51.2, 59.4] (95.0) | 66.4 | 66.8 (90.0) |
| SFT + record | 1.44 | 90.3 | 0.34 / 0.32 | 0.158 / 0.155 | 5.0 | 3.2 [1.9, 7.2] | 2.8 | 84.3 | 82.3 [77.0, 86.7] (95.4) | 100.0 | 99.9 (90.4) |
| SFT + retrieved record | 1.50 | 90.1 | 0.37 / 0.34 | 0.163 / 0.159 | 5.1 | 2.5 [1.9, 6.8] | 2.9 | 81.9 | 81.8 [76.5, 86.3] (95.2) | 100.0 | 99.8 (90.2) |
| SFT no DB, 800 steps, user held out (row 42) | 2.88 | 69.6 | 1.70 / 1.03 | 0.475 / 0.398 | 19.2 | 1.9 [1.9, 5.2] | 9.0 | 47.8 | 47.8 [43.0, 52.9] (94.5) | 62.9 | 63.7 (89.5) |
| SFT + record, user held out (row 42) | 2.20 | 81.3 | 0.92 / 0.62 | 0.310 / 0.279 | 12.6 | 4.7 [3.3, 9.3] | 6.2 | 57.5 | 60.4 [54.4, 67.7] (93.3) | 75.8 | 74.3 (90.3) |

**Table C.2: the seeded arms, mean +- sd over seeds 0 to 2 (mean rule)**

| arm | T | accuracy | ECE raw | ECE tempered | AURC | coverage at 95%, out-of-fold | its precision |
|---|---|---|---|---|---|---|---|
| SFT no DB, 200 steps | 0.98 +- 0.10 | 60.5 +- 3.0 | 10.9 +- 3.0 | 10.1 +- 1.6 | 14.7 +- 1.9 | 31.0 +- 8.0 | 94.7 +- 0.7 |
| SFT no DB, 800 steps | 1.34 +- 0.10 | 73.3 +- 2.3 | 4.3 +- 0.9 | 7.3 +- 1.2 | 7.3 +- 0.9 | 55.6 +- 3.7 | 94.8 +- 0.2 |
| SFT + record | 0.69 +- 0.07 | 89.3 +- 0.8 | 9.1 +- 1.6 | 2.6 +- 0.6 | 2.9 +- 0.2 | 80.9 +- 2.1 | 94.4 +- 0.4 |

**Table C.3: cross-arm transfer, seed 0, mean rule (each arm tempered by its own leave-users-out temperatures and by the other arm's, fitted on the same users; NLL and ECE tempered; out-of-fold coverage at 95% with its realised precision)**

| arm | temperature from | T | NLL | ECE | coverage at 95% (precision) |
|---|---|---|---|---|---|
| SFT no DB, 800 steps | SFT no DB, 800 steps (own) | 1.43 | 1.15 | 8.3 | 51.3 (95.0) |
| SFT no DB, 800 steps | SFT + record | 0.62 | 1.65 | 14.3 | 46.6 (95.5) |
| SFT + record | SFT + record (own) | 0.62 | 0.32 | 3.2 | 79.3 (94.2) |
| SFT + record | SFT no DB, 800 steps | 1.43 | 0.56 | 24.9 | 80.8 (94.8) |
| instruct base, no record | instruct base, no record (own) | 1.76 | 2.37 | 11.9 | 1.7 (80.0) |
| instruct base, no record | instruct base + record | 0.83 | 2.56 | 7.6 | 1.8 (81.0) |
| instruct base + record | instruct base + record (own) | 0.83 | 1.48 | 8.9 | 2.2 (80.8) |
| instruct base + record | instruct base, no record | 1.76 | 1.68 | 30.6 | 1.6 (78.9) |

**Table C.4: the 95% out-of-fold operating point by corrected group, seed 0 or merged folds, mean rule (accuracy; mean tempered confidence; share auto-applied; precision of the auto-applied)**

| arm | in history | labelled seen, not in history | determined by category | split category |
|---|---|---|---|---|
| instruct base, no record | 38 / 20 / 2 / 91 | 30 / 20 / 4 / 60 | 28 / 19 / 1 / 75 | 20 / 15 / 0 / - |
| instruct base + record | 62 / 51 / 2 / 100 | 60 / 49 / 3 / 67 | 60 / 50 / 2 / 100 | 31 / 45 / 5 / 20 |
| SFT no DB, 200 steps | 65 / 52 / 31 / 99 | 41 / 38 / 15 / 100 | 59 / 47 / 20 / 94 | 27 / 39 / 7 / 14 |
| SFT no DB, 800 steps | 80 / 72 / 65 / 97 | 58 / 56 / 37 / 93 | 75 / 64 / 50 / 97 | 33 / 50 / 14 / 21 |
| SFT + record | 94 / 93 / 86 / 97 | 83 / 88 / 71 / 96 | 97 / 92 / 83 / 100 | 50 / 79 / 48 / 16 |
| SFT + retrieved record | 94 / 92 / 86 / 97 | 83 / 86 / 71 / 98 | 97 / 91 / 83 / 100 | 50 / 77 / 49 / 16 |
| SFT no DB, 800 steps, user held out (row 42) | 73 / 70 / 57 / 95 | 61 / 57 / 36 / 95 | 75 / 64 / 45 / 95 | 39 / 50 / 6 / 50 |
| SFT + record, user held out (row 42) | 89 / 83 / 68 / 96 | 75 / 76 / 50 / 90 | 87 / 78 / 61 / 98 | 30 / 69 / 41 / 31 |

### 50.1 The sum rule is the one to calibrate, and the record arm needs almost no temperature

Under the mean rule a fitted temperature often makes ECE worse (the no-DB SFT at 800 steps 4.8 to 8.3, the untrained base 8.1 to 11.9) while lowering NLL, because the softmax of per-token averages is not a likelihood: a two-token and a six-token category name are put on one scale by the division, and no single temperature undoes that per item. Under the sum rule the softmax is the model's own probability of each name, and one temperature per arm brings every categoriser to ECE 2 to 4 (Table C.1, sum rule). The accuracies are the same under both rules (section 48.4), and the sum rule's auto-apply coverage at 95% is the same or higher at the same realised precision (+3 to +9 on the trained-user arms; -0.9 and +2.6 on the held-out folds), so the sum rule is the one to serve confidences from. The Jev series' prediction holds in direction: the record-in-prompt categoriser, which reads the category off evidence in the prompt, is nearly calibrated as trained (T 1.44, raw ECE 5.0, tempered 3.2), and the no-record one, which answers from its weights and the shots, is over-confident and more so the longer it trains (T 1.80 at 200 steps and 2.79 at 800; raw ECE 14.3 and 18.6, tempered 3.0 and 3.9). The untrained instruct base is over-confident without the record (T 2.38) and nearer calibrated with it (1.50).

### 50.2 The operating point: 82% of a trained user's transactions at 95% precision, 60% of a new user's

With the threshold chosen on other users (out-of-fold), the record categoriser auto-applies 82.3% of the items [77.0, 86.7] at a realised precision of 95.4% (sum rule; 80.9 +- 2.1 over three seeds under the mean rule), and at 90% precision it applies everything (99.9%, since its accuracy is 90). The retrieved record gives the same (81.8 at 95.2). The no-DB categoriser at 800 steps applies 55.8% at 95.0%, against 31.6% at 200 steps: row 47's longer training nearly doubles the share that needs no review. For users whose schemes the adapter never saw (row 42's folds) the shares fall to 60.4% (record, realised precision 93.3%, a little under target: the threshold chosen on other users' items transfers imperfectly when those items came from other fold adapters) and 47.8% (no DB, 94.5%). So the production number for a new user with the record in the prompt is about 60% auto-applied at 93 to 95% precision, and 40% sent for review. The untrained base cannot be operated this way at all (2 to 8% coverage), which is one more measure of what the label SFT buys.

### 50.3 A foreign temperature ruins the probabilities and leaves the operating point intact

The temperature is task-dependent, as the series found: the record arm tempered with the no-record arm's T (1.43 under the mean rule, against its own 0.62) goes from ECE 3.2 to 24.9 and NLL 0.32 to 0.56, and the no-record arm with the record arm's T from 8.3 to 14.3; the untrained base with and without the record behaves the same way (8.9 to 30.6). The auto-apply coverage at 95% precision barely moves (79.3 to 80.8, 51.3 to 46.6) because the threshold is chosen on the tempered confidences of the same arm and one scalar temperature preserves most of the confidence ranking. The practical rule: fit the auto-apply threshold per arm (per prompt layout) on held-out users, which is robust; fit a temperature per arm only where the probability itself is shown or consumed downstream, and never carry one across arms.

### 50.4 Where the confident errors are: the split categories

Table C.4 splits the 95% operating point by section 48's groups. On the category-determined items the record arm auto-applies 83% at 100% precision and on the merchants in the history 86% at 97%; on the 103 items in categories the user split with nothing to say which side, its accuracy is 50 (a coin flip, as section 48 found), its mean confidence 79, and it auto-applies 48% of them at 16% precision. The no-record arms do the same on a smaller scale (14% applied at 21%). So nearly all of the auto-apply errors on REAL-6 come from the items that have no answer in the evidence, and the model does not know that it does not know. A trained abstention option (row 52, REAL-14) is aimed at exactly these, and they are the cleanest test set for it: the uniform-target arm should put those items below the threshold without moving the others.

### 50.5 What the step says

STAT-4 asked whether the categoriser's probabilities are calibrated and at what confidence a label can be applied without review. Serve confidences from the sum rule; fit one temperature per arm on held-out users (record arm about 1.4, no-record arm about 2.8, bounded search); choose the auto-apply threshold per arm on held-out users, which transfers where the temperature does not. At 95% precision the record-in-prompt categoriser applies about 82% of a trained user's transactions and about 60% of a new user's, the 800-step no-record categoriser 56% and 48%. The residual errors at that operating point are concentrated in the categories the user split arbitrarily, where the model is confident and wrong; row 52 targets them, and row 43's set, with relabelling and per-user assignments, will be the harder test of the whole operating point.

Not done: per-user or per-category temperatures, a Brier or calibration term in the training loss (the Jev series' alternative to post-hoc scaling), the encoder arms (they save no per-option scores), and the rename-augmented folds (row 42, section 51).

## 51. Held-out users: the categoriser trained on other users' schemes reads standard names as well as ever and loses 10 to 25 points on the user's coined names, which it had been memorising; rename augmentation recovers most of it (coined +12 to +14), and the record-in-prompt categoriser on a new user is 82 without it and 87 with it (REAL-10)

*PLAN step 42. Code: `FOLD` and `RENAME` in `scripts/exp_categoriser.py`, `USERS` and the fold-from-name default in `scripts/exp_real6.py`, `scripts/chains/chain_r42.sh`, `scripts/heldout_users_tables.py`, `ai_experiments.real6_cells` (the shared table helpers). The users fall in four folds (user id mod 4, five users each); each fold adapter trains on the other fifteen users and is scored on its five, so all 1,179 items are read by an adapter that never saw their user. The no-DB arm at 800 steps (REPORT.md 49), the record arm at 200 (at the set's ceiling there, section 48), all on the 3090 (4-bit base). The rename-augmented folds ran under row 56's chain once the all-label loss had become the no-DB recipe (section 52), so the no-DB augmentation is measured on that recipe; the old recipe's rename folds were dropped (PLAN log, 2026-09-25).*

Every REAL-6 number until this step was on the twenty users whose histories trained the adapter. The production categoriser meets users whose schemes it never saw, so this is the number production needs.

**Table 51.1: trained on all 20 users against the user held out (accuracy %; held-out rows merge the four fold adapters, each scoring the five users it never trained on; interval = users resampled)**

| arm | all | standard | renamed | coined | in history | determined by category | split category | DB-only | interval (all) |
|---|---|---|---|---|---|---|---|---|---|
| SFT no DB, 800 steps, all 20 users trained | 71.7 | 79.4 | 63.2 | 73.9 | 80.0 | 75.4 | 33.0 | 52.5 | [67.5, 75.5] |
| SFT no DB, 800 steps, user held out | 70.1 | 82.1 | 61.9 | 63.9 | 73.4 | 75.0 | 38.8 | 59.0 | [66.1, 73.9] |
| SFT + record, 200 steps, all 20 users trained | 90.2 | 99.8 | 80.3 | 91.6 | 94.1 | 97.4 | 50.5 | 97.5 | [86.0, 93.5] |
| SFT + record, 200 steps, user held out | 81.6 | 99.6 | 72.4 | 66.7 | 89.2 | 86.6 | 30.1 | 86.9 | [78.6, 84.7] |

**Table 51.2: held out minus all-20 on the same items, points [user-resampled 95% interval]**

| arm | all | standard | renamed | coined | in history |
|---|---|---|---|---|---|
| SFT no DB | -1.6 [-5.1, +2.9] | +2.7 [-2.2, +7.6] | -1.3 [-6.7, +4.2] | -10.0 [-19.0, -0.4] | -6.5 [-10.5, -2.7] |
| SFT + record | -8.6 [-12.3, -4.8] | -0.2 [-1.1, +0.5] | -7.9 [-15.1, -1.7] | -24.9 [-39.0, -11.4] | -5.0 [-8.0, -1.9] |

**Table 51.3: rename augmentation on held-out users (accuracy %, then with minus without on the same items [interval])**

| recipe | augmentation | all | standard | renamed | coined |
|---|---|---|---|---|---|
| no DB, all-label 200 steps, held out | without | 74.5 | 85.4 | 64.9 | 73.1 |
| no DB, all-label 200 steps, held out | with | 77.6 | 86.5 | 65.3 | 85.1 |
| no DB, all-label 200 steps, held out | with minus without | +3.1 [+0.2, +6.0] | +1.1 [-2.0, +3.9] | +0.4 [-4.6, +5.5] | +12.0 [+5.1, +20.0] |
| SFT + record, 200 steps, held out | without | 81.6 | 99.6 | 72.4 | 66.7 |
| SFT + record, 200 steps, held out | with | 86.8 | 99.8 | 77.8 | 80.3 |
| SFT + record, 200 steps, held out | with minus without | +5.2 [+0.4, +9.8] | +0.2 [-0.5, +1.1] | +5.4 [-3.4, +14.4] | +13.7 [-1.0, +29.3] |

### 51.1 What transfers to a new user and what was memorised

On standard category names a held-out user is read as well as a trained one (no DB 82.1 against 79.4, record 99.6 against 99.8): the model knows what "Groceries" means and the user's shots only confirm it. On the user's own words it is not. The coined names fall 10.0 points without the DB and 24.9 with the record (Table 51.2), and the renamed ones 1 and 8: the categorisers had been learning the twenty users' private vocabularies ("Zorbit" is this user's groceries) from their training rows, not reading them off the shots, and a new user's vocabulary is only in the shots. The record arm loses most because it was the best at the memorised mapping (91.6 on coined names for trained users). Merchants the user labelled fall 5 to 7 points for the same reason: the trained adapter had also memorised which merchant each training user filed where, and for a new user only the 24 shots carry that. On the DB-only merchants, which no user labelled, the no-DB arm does not fall (59.0 against 52.5), and the record arm falls 97.5 to 86.9, the same coined-name loss: the record says what the store sells, and turning that into a new user's coined word is exactly the step that had been memorised. Overall: no DB 70.1 held out against 71.7 trained, the record arm 81.6 against 90.2.

### 51.2 Rename augmentation

Replacing each category name by a fresh coined word with probability 0.5 per training episode (consistently in the category list, the shots and the target) makes memorising a user's vocabulary useless and forces the mapping to be read from the shots. On held-out users it lifts the coined names by 12.0 [5.1, 20.0] points on the no-DB all-label recipe (73.1 to 85.1) and 13.7 [-1.0, 29.3] on the record arm (66.7 to 80.3), with standard names unchanged (+1.1, +0.2) and the whole set up 3.1 and 5.2 points. With it, the record-in-prompt categoriser reads a new user at 86.8 [82.6, 90.4], and the no-DB all-label one at 77.6.

### 51.3 What the step says

REAL-10 asked whether the categoriser holds for users whose schemes were never trained on. For standard names, yes. For a user's own coined and renamed names it held only because it had memorised the training users', and rename augmentation is the fix: it costs nothing on standard names and recovers most of the coined-name loss. Every categoriser from here trains with it and is scored on held-out users. The production numbers are the held-out ones: 86.8 with the record in the prompt, 77.6 without (both with augmentation); section 53's database episodes (without rename augmentation) read 89.9 without any record, and the two have not yet been combined.

Not done: rename augmentation at other rates, and a held-out-user set with more than twenty users (row 43).

## 52. Training efficiency: the categoriser's hours went into re-reading 24 shots to learn one label; putting the loss on every shot label gives in 100 steps (9 minutes) what 1,600 plain steps (147 minutes) gave, and on held-out users 4.4 points more than the 800-step recipe at a quarter of its time, at the cost of leaning less on what the model knew about chains; bf16 and the 4-bit base are the same model for this task (TRAIN-11)

*PLAN step 56 (the owner's request, 2026-09-23). Code: `ALL_LABELS` (the loss on every shot label, found through the tokenizer's offset mapping so training sees the scorer's tokenisation), `ANS_WEIGHT` and the throughput counters in `scripts/exp_categoriser.py`; `scripts/chains/chain_r56*.sh` on the 3090, `scripts/modal_jobs/r56_rest.json` on Modal; `scripts/train_efficiency_tables.py`. Tables 52.1 and 52.2 are the 3090 (4 x 4 sequences per step); 52.3 and 52.4 the H100 on Modal with one 16-sequence pass per step (section 54), where the local chain moved on 2026-09-25.*

One training step of the categoriser is 16 sequences of about 850 tokens (the user's category list, 24 labelled shots, the query) through the 3B model, at about 2,400 tokens a second on the 3090, close to section 7's ceiling for 3B: the card was busy. The loss fell on the answer's one to five tokens, so each 850-token pass taught one label, and row 47 had just shown that the no-DB categoriser needs 800 to 1,600 such steps. Two changes were tested: the loss on every shot's label as well (each predicted from the category list and the shots before it: 83 supervised tokens in 832 instead of about 3), and the bf16 base in place of the 4-bit one, which unsloth had chosen by default (section 44).

**Table 52.1: the no-DB categoriser on all 20 users, seed 0, 3090 (4 x 4 sequences per step; accuracy %; ARC-Easy, MMLU and symbol-label ICL from exp_items_v2)**

| recipe | steps | minutes | tokens/s | peak GiB | final loss | all | in history | determined by category | DB-only | ARC-Easy | MMLU | ICL symbol |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain, 4-bit | 200 | 17.7 | - | 9.13 | 0.141 | 57.1 | 65.3 | 59.3 | 51.6 | 76.0 | 51.0 | 76.6 |
| plain, 4-bit | 400 | 36.6 | - | 9.13 | 0.047 | 67.4 | 75.0 | 71.5 | 49.2 | 77.0 | 53.5 | 70.8 |
| plain, 4-bit | 800 | 77.6 | - | 9.13 | 0.028 | 71.7 | 80.0 | 75.4 | 52.5 | 75.0 | 52.5 | 75.0 |
| plain, 4-bit | 1600 | 146.8 | - | 9.13 | 0.0 | 76.7 | 83.1 | 82.9 | 55.7 | 80.5 | 53.5 | 74.5 |
| plain, bf16 | 200 | 17.9 | 2391 | 12.64 | 0.099 | 59.1 | 65.8 | 61.5 | 54.9 | 78.5 | 55.0 | 68.2 |
| plain, bf16 | 800 | 71.6 | 2394 | 12.64 | 0.039 | 73.1 | 78.6 | 79.8 | 57.4 | 81.5 | 54.0 | 68.2 |
| all-label, 4-bit | 100 | 9.1 | 2355 | 9.12 | 0.019 | 79.4 | 84.0 | 87.4 | 55.7 | 77.5 | 49.5 | 74.5 |
| all-label, 4-bit | 200 | 18.1 | 2369 | 9.13 | 0.002 | 80.4 | 86.0 | 86.4 | 45.9 | 78.0 | 52.0 | 75.0 |
| all-label, 4-bit | 400 | 36.3 | 2359 | 9.13 | 0.0 | 82.0 | 87.4 | 90.0 | 63.1 | 78.5 | 52.0 | 74.5 |
| all-label, bf16 | 200 | 17.7 | 2425 | 12.64 | 0.001 | 80.2 | 86.3 | 88.2 | 54.1 | 76.5 | 55.5 | 74.5 |

**Table 52.2: held-out users (row 42's four folds, 3090): all-label at 200 steps against the plain recipe at 800 steps (accuracy %; last row: all-label minus plain on the same items [user-resampled interval])**

| recipe | all | coined | in history | determined by category | DB-only known | DB-only opaque |
|---|---|---|---|---|---|---|
| plain, 800 steps (77 min) | 70.1 | 63.9 | 73.4 | 75.0 | 87.3 | 19.6 |
| all-label, 200 steps (18 min) | 74.5 | 73.1 | 78.6 | 80.4 | 70.4 | 3.9 |
| all-label minus plain | +4.4 [+0.9, +7.8] | +9.2 [-3.1, +20.2] | +5.2 [+1.8, +8.5] | +5.3 [-0.8, +11.3] | -16.9 [-32.9, +1.8] | -15.7 [-23.3, -6.2] |

**Table 52.3: held-out users (four folds, H100, one 16-sequence pass per step, 4-bit): the answer's share of the loss**

| loss | all | coined | DB-only known | DB-only opaque |
|---|---|---|---|---|
| all-label, token mean (answer ~3 of ~85 labelled tokens) | 73.4 | 70.3 | 67.6 | 11.8 |
| all-label, answer weighted to half of each sequence | 72.2 | 66.3 | 69.0 | 11.8 |
| weighted minus token mean | -1.2 [-3.1, +0.8] | -4.0 [-10.6, +2.9] | +1.4 [-7.7, +10.6] | +0.0 [-9.1, +6.8] |

**Table 52.4: 4-bit against bf16, all-label at 200 steps on all 20 users, three seeds each (H100, one 16-sequence pass per step; mean +- sd, then the three seeds)**

| base | REAL-6 | ARC-Easy | MMLU | ICL symbol | training minutes |
|---|---|---|---|---|---|
| 4-bit | 80.5 +- 1.0 (80.2, 79.6, 81.7) | 78.3 +- 2.0 (76.0, 79.5, 79.5) | 52.0 +- 0.5 (52.5, 51.5, 52.0) | 71.9 +- 0.6 (72.4, 71.3, 71.9) | 3.1, 3.0, 3.1 |
| bf16 | 80.0 +- 1.4 (79.6, 78.8, 81.6) | 74.8 +- 2.8 (75.0, 77.5, 72.0) | 55.8 +- 1.4 (55.0, 57.5, 55.0) | 72.6 +- 2.0 (73.5, 70.3, 74.0) | 3.0, 2.9, 2.9 |

### 52.1 The all-label loss: sixteen times less compute for the same trained-user accuracy

On the twenty training users the all-label loss reaches 79.4 in 100 steps (9.1 minutes), above the plain recipe's 76.7 at 1,600 steps (146.8 minutes), and 82.0 at 400; tokens per second are unchanged (the extra supervised tokens cost nothing, since the pass was already made), and ARC-Easy, MMLU and symbol-label ICL stay where the plain recipe put them. Three seeds on the H100 put its spread at 1.0 to 1.4 points (Table 52.4), tighter than the plain recipe's 2.3.

### 52.2 On held-out users the gain halves, and chain knowledge is the price

Scored on users it never trained on (row 42's folds), the all-label adapter at 200 steps reads 74.5 against the plain recipe's 70.1 at 800 steps: +4.4 [0.9, 7.8] at a quarter of the training time, with coined names +9.2 and merchants the user labelled +5.2. Against its trained-user 80.4 it loses 6 points where the plain recipe lost 1.6, so about half of the trained-user gain was faster memorisation of the training users (each sequence now supervises 24 of their merchant labels instead of one). And it gives up what the model knew: on the DB-only merchants that are real chains the plain recipe reads 87.3 and the all-label one 70.4 (-16.9 [-32.9, 1.8]); opaque ones fall from 19.6 to 3.9. With 25 answers per sequence to be read off the shots, the adapter learns to answer from the shots and stops consulting its own knowledge of the merchant. Weighting the final answer to half of each sequence's loss does not bring it back (Table 52.3: known chains 69.0 against 67.6, overall -1.2): the lost knowledge is not a matter of the answer's share of the gradient. Section 53 makes the question moot for merchants in a fact DB, where database episodes put known chains at 93 and opaque ones at 96.

### 52.3 bf16 against the 4-bit base

On the 3090 the bf16 base trains at the same speed (2,391 against about 2,370 tokens a second; unsloth's 4-bit kernels cost almost nothing) with 3.5 GiB more memory, and reads within two points of the 4-bit run at 200 and 800 steps. On the H100 with three seeds each, all-label at 200 steps reads 80.5 +- 1.0 on the 4-bit base and 80.0 +- 1.4 on bf16; the general measures move both ways (ARC-Easy 78.3 against 74.8, MMLU 52.0 against 55.8) and neither by much at three seeds. The categoriser does not care; the owner chose bf16 as the default for new runs (2026-09-26), which removes the quantisation step and the adapter-precision mismatch of section 44 at no cost on an 80 GB GPU.

### 52.4 What the step says

TRAIN-11 asked where the hours go and whether the same accuracy can come cheaper. They went into re-reading the prompt to learn one label per pass, not into precision. The all-label loss is the no-DB recipe from here (with rename augmentation, section 51): a quarter of the training time for +4.4 on held-out users, the ICL and general measures held. Its cost, less use of the model's own merchant knowledge, is the reason to put a fact DB into the weights as episodes (section 53) or in the prompt as a record, not a reason to train longer. The 3090 and the H100 give the same numbers (section 54); the H100 with one 16-sequence pass per step is six times faster.

Not done: the all-label loss for the record-in-prompt arm (its shots carry no record, so it would learn the record's use from one label in 25), the loss on a fraction of the shots, and packing (sequences are padded to the batch's longest).

## 53. The fact DB as supervised decisions: training episodes built from the database's records put the merchants no user labelled at 94 on held-out users with no record in the prompt (opaque ones 96, from 12), where the same records as prose reach 65; knowledge a model must use to decide is best taught as the decisions (REAL-15)

*PLAN step 57, from the owner's remark (2026-09-25) that measuring loss on more tokens might be a better way to impart a database. Code: `DBEP` (database episodes) and `DB_CAT` (category-bearing prose records) in `scripts/exp_categoriser.py`, `scripts/modal_jobs/r57.json`, `scripts/db_episodes_tables.py`. All arms on Modal (H100, 4-bit base, one 16-sequence pass per step), all-label loss, 200 steps, row 42's four held-out folds, scored without the record in the prompt.*

Every records-in-the-weights arm so far trained the record sentences as text: loss on every token of "Elrholm is a store that sells canned goods, packaged snacks and frozen vegetables", and three variants of it. That is the most "loss on more tokens" a record can get, and it reached the DB-only merchants at 59 +- 12 (3.3 passes) to 69 +- 6 (6.7 passes), opaque ones never above 44 (sections 43, 45). Section 52 had just shown that supervising the decisions the task makes (every shot's label) trains the task sixteen times faster. The same idea applied to the database: turn each record into categorisation decisions. A database episode is an ordinary training episode for a training user in which 8 of the 24 shots and the target are replaced by synthetic statement rows of fact-DB merchants (rendered by the generator, amounts from the category's distribution, no string equal to a REAL-6 test string), each labelled with that user's own name for the merchant's DB category; merchants whose category the user split are skipped, since the DB cannot say which half. Every DB merchant is covered, the 60 DB-only ones included, and their labels come from the database, never from a user. Half the episodes are database episodes (`DBEP=0.5`).

Two design choices matter for reading the result. First, the episodes use a category field, which REAL-6's product sentences do not state (real merchant and POI databases usually carry one); to give the prose arm the same information its records state the category too ("Elrholm is a Groceries store that sells ..."). Second, the scores are on held-out users only: on a training user, an episode supervises exactly the answer of the test item for that (user, merchant), which would measure memorisation.

**Table 53.1: the fact DB in the weights, four held-out folds, no record in the prompt at test (accuracy %; interval = users resampled)**

| arm | all | standard | renamed | coined | in history | determined by category | split category | DB-only | DB-only known | DB-only opaque | interval (all) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no DB | 73.4 | 83.8 | 65.1 | 70.3 | 78.4 | 78.4 | 38.8 | 44.3 | 67.6 | 11.8 | [68.6, 77.7] |
| database episodes | 89.9 | 99.3 | 80.3 | 91.2 | 92.1 | 95.3 | 59.2 | 94.3 | 93.0 | 96.1 | [86.9, 92.8] |
| prose records + category | 78.9 | 89.6 | 67.8 | 80.7 | 81.5 | 86.6 | 42.7 | 64.8 | 85.9 | 35.3 | [75.4, 82.1] |
| both | 88.0 | 99.3 | 76.4 | 89.6 | 90.1 | 95.1 | 48.5 | 91.8 | 95.8 | 86.3 | [84.1, 91.3] |

**Table 53.2: each arm minus no DB on the same items, points [user-resampled 95% interval]**

| arm | all | coined | determined by category | DB-only | DB-only known | DB-only opaque |
|---|---|---|---|---|---|---|
| database episodes | +16.5 [+12.6, +21.1] | +20.9 [+13.9, +29.4] | +16.9 [+10.4, +24.6] | +50.0 [+35.2, +64.4] | +25.4 [+11.5, +39.1] | +84.3 [+66.7, +96.6] |
| prose records + category | +5.5 [+2.6, +8.6] | +10.4 [+5.2, +16.7] | +8.3 [+3.8, +13.7] | +20.5 [+8.0, +33.1] | +18.3 [+5.8, +31.5] | +23.5 [+2.1, +44.2] |
| both | +14.6 [+10.7, +18.6] | +19.3 [+11.1, +28.1] | +16.7 [+10.1, +24.6] | +47.5 [+29.7, +64.4] | +28.2 [+14.8, +41.4] | +74.5 [+44.1, +96.2] |

**Table 53.3: DB-only merchants per fold (accuracy %, items in the fold)**

| arm | fold 0 | fold 1 | fold 2 | fold 3 |
|---|---|---|---|---|
| no DB | 46 (35) | 22 (36) | 62 (32) | 53 (19) |
| database episodes | 94 (35) | 97 (36) | 100 (32) | 79 (19) |
| prose records + category | 71 (35) | 58 (36) | 75 (32) | 47 (19) |
| both | 91 (35) | 94 (36) | 88 (32) | 95 (19) |

### 53.1 Decisions teach what prose does not

With the same information, the database as decisions puts the DB-only merchants at 94.3 on users the model never saw, +50.0 [35.2, 64.4] over no DB, and the opaque ones among them, which nothing but the database names, at 96.1 (+84.3); the prose records with the category reach 64.8 (+20.5) and 35.3 on the opaque ones. The effect holds in every fold (Table 53.3: 94, 97, 100, 79 against 46, 22, 62, 53). It is not only the DB-only merchants: every item gains (all +16.5, coined names +20.9, category-determined +16.9), because every database episode is also practice at mapping a category into a user's own scheme from the shots, with about 14,400 database-derived decisions per run (1,600 database episodes of 8 shots and a target). Prose added to the episodes does not help and costs the opaque merchants ten points (both: 86.3), the prose taking a quarter of the sequences from the episodes. The known chains, which the all-label recipe had taught the model to stop consulting (section 52.2), come back to 93.0.

This is the section 8 lesson (knowledge trained in one form stays in that form) turned to use: the categoriser has to use a merchant's category to decide, so the category is best taught as the decision, in the task's own format, and the prose record is the wrong form for it.

### 53.2 What it does not show yet

Three things stand between this and a production claim. The synthetic statement strings come from the same generator templates as the test strings (only exact matches were refused), so the reading of a real bank string is untested; row 43's held-out rendering family is the test. The database has 240 merchants and a production one millions: section 29 found interference at 5,000 species for this adapter, and how many merchants episodes can hold is row 59 (REAL-17). And the record-in-prompt categoriser has never been given the category field the episodes had, so "episodes against retrieval" is not yet a fair comparison; that is row 58 (REAL-16), with every arm on bf16 on the same GPU.

### 53.3 What the step says

REAL-15 asked whether the fact DB goes into the weights better as supervised decisions than as prose. It does, by a wide margin: 94 against 65 on the merchants only the database knows, 96 against 35 on the opaque ones, with no record in the prompt at test and on users the model never saw. For merchants in a fact DB, database episodes are the parametric route to use, and the no-record categoriser with them (89.9 overall) is above the record-in-prompt categoriser on held-out users without rename augmentation (81.6, section 51) and close to it with (86.8).

Not done: episodes with rename augmentation, other episode shares, the record in the prompt together with episodes, and the scale and rendering tests above.

## 54. Experiments on Modal: the repository's scripts run unchanged on an H100, one 16-sequence pass per step trains six times faster than the 3090, the numbers agree within run-to-run noise, and a train-and-score job costs under a dollar (INFRA-1)

*PLAN step 34, started 2026-09-25 when the owner provided Modal access (workspace `ynab`, shared; app `ai-experiments-training`, volumes `ai-exp-hf-cache` and `ai-exp-results`, names approved by the owner). Code: `scripts/modal_app.py`, job lists in `scripts/modal_jobs/`, `MICRO` in `scripts/exp_categoriser.py`, `scripts/modal_repro_tables.py`; Modal's agent skill in `.claude/skills/modal` (its bundled docs carry Modal's sample Docker Hub token, which GitHub push protection flags; replaced by an obvious placeholder).*

The image is the project's own environment: Python 3.12 and `uv sync --frozen` from `pyproject.toml` and `uv.lock` (torch 2.11 + cu128, unsloth 2026.9.4, the versions on the 3090), built once and cached. The code, the frozen item sets and the tracker's `runs.jsonl` are mounted at container start and copied into a writable tree, so a code change does not rebuild the image; model downloads persist in a volume. Each job runs any list of the repository's commands with any environment (the chain scripts' conventions unchanged), streams their output, and writes every file they created or changed (results, adapters, tracker rows) to its own directory in the results volume, from which `modal volume get` brings them back to a checkout, `just push-models` and a union of `runs.jsonl` by run id. A job list runs in parallel, at most eight containers at a time (the volume's commits contend beyond about five).

**Table 54.1: one run on two GPUs (row 56's all-label no-DB run, 200 steps, seed 0, 4-bit; interval = users resampled; paired with the 3090 run)**

| GPU, micro-batches | train minutes | tokens/s | peak GiB | final loss | REAL-6 all [interval] | same predictions as the 3090 | minus the 3090 [interval] |
|---|---|---|---|---|---|---|---|
| 3090, 4 x 4 | 18.1 | 2369 | 9.13 | 0.002 | 80.4 [76.3, 84.1] | 100.0 | - |
| H100, 4 x 4 | 4.9 | 8736 | 9.18 | 0.002 | 78.5 [73.9, 82.6] | 87.3 | -1.9 [-3.9, -0.1] |
| H100, 1 x 16 | 3.0 | 14091 | 25.18 | 0.002 | 80.1 [75.6, 83.8] | 86.8 | -0.3 [-2.5, +1.7] |

The three runs are the same code, data order and seed. The H100 with the 3090's four micro-batches of four trains in 4.9 minutes instead of 18.1 (3.7 times the tokens per second); one pass of 16 sequences per step, which fits in 25 GiB of the H100's 80, takes 3.0 minutes (5.9 times). The REAL-6 accuracies are within two points and the predictions agree on 87%, which is the same-seed run-to-run agreement the 3090 shows between separate runs (section 20); three seeds of the same recipe on the H100 later put its spread at 1.0 to 1.4 points (section 52.4). A job that trains and scores costs about $0.60 to $0.80 of H100 time; scoring, still batched for a 24 GB card, is now the larger half of it. Rows 57 and 58 ran as sixteen parallel jobs each in about twenty minutes, which would have been five hours each on the 3090.

INFRA-1 is answered for the categoriser line, and from 2026-09-25 all GPU work runs there (the owner's decision). The long runs the row was first written for (arm C at 5,000 species, Flan-T5 at 1,000 and 5,000 species) were not run: the species-universe questions they served have been overtaken by the categoriser's, and row 59 asks the capacity question on the merchant database instead.

## 55. Database episodes against the record in the prompt, with the same information: a tie (87.8 against 86.6 overall, 93.4 against 91.8 on the DB-only merchants); the category field is worth little to the record, and a database taught as decisions is as good as the database looked up at test (REAL-16)

*PLAN step 58. Code: `REC_CAT` in `scripts/exp_categoriser.py` and `scripts/exp_real6.py` (the record in the prompt, in training and at test, states the merchant's category: `real6.category_record`), `scripts/modal_jobs/r58.json`, `scripts/fair_record_tables.py`. Every arm on Modal on one setting: H100, bf16 (the default from 2026-09-26), one 16-sequence pass per step, 200 steps, row 42's four held-out folds.*

Section 53 found database episodes put the DB-only merchants at 94 on held-out users with no record at test, above the record-in-prompt categoriser's 87 (section 51), but the episodes were built from a category field the record in the prompt never carried, and the arms ran on different hardware and precision. This step gives the record the category ("Halvarro is a Home Improvement store that sells interior paint, garden hoses and plywood sheets") and runs all four arms alike.

**Table 55.1: database episodes against the record in the prompt at equal information (held-out users; accuracy %; interval = users resampled)**

| arm | all | standard | renamed | coined | in history | determined by category | split category | DB-only | DB-only known | DB-only opaque | interval (all) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no DB (all-label) | 76.8 | 87.2 | 66.5 | 77.9 | 80.0 | 83.5 | 43.7 | 56.6 | 83.1 | 19.6 | [73.3, 80.0] |
| database episodes (all-label), no record at test | 87.8 | 99.8 | 77.0 | 86.7 | 90.5 | 94.3 | 44.7 | 93.4 | 93.0 | 94.1 | [84.1, 91.0] |
| record in the prompt, products | 85.0 | 98.9 | 78.2 | 72.7 | 90.1 | 88.6 | 49.5 | 88.5 | 88.7 | 88.2 | [80.7, 88.6] |
| record in the prompt, with the category | 86.6 | 100.0 | 78.7 | 77.5 | 90.8 | 91.9 | 46.6 | 91.8 | 91.5 | 92.2 | [82.5, 90.4] |

**Table 55.2: paired differences on the same items, points [user-resampled 95% interval]**

| comparison | all | coined | determined by category | DB-only | DB-only opaque |
|---|---|---|---|---|---|
| database episodes (all-label), no record at test minus no DB (all-label) | +10.9 [+7.9, +13.8] | +8.8 [+2.2, +14.5] | +10.8 [+7.0, +14.7] | +36.9 [+28.1, +44.3] | +74.5 [+60.0, +91.5] |
| record in the prompt, with the category minus no DB (all-label) | +9.8 [+6.0, +13.1] | -0.4 [-15.0, +13.1] | +8.4 [+2.9, +14.3] | +35.2 [+26.5, +44.1] | +72.5 [+55.4, +93.9] |
| record in the prompt, with the category minus record in the prompt, products | +1.6 [-1.3, +4.4] | +4.8 [-9.4, +18.8] | +3.3 [-2.5, +9.4] | +3.3 [+0.7, +6.2] | +3.9 [+0.0, +9.8] |
| database episodes (all-label), no record at test minus record in the prompt, with the category | +1.2 [-2.7, +5.0] | +9.2 [-2.4, +20.7] | +2.4 [-1.5, +6.4] | +1.6 [-3.5, +6.5] | +2.0 [-7.5, +8.5] |

With the same information the two routes tie: database episodes, with nothing in the prompt about the merchant, read 87.8 [84.1, 91.0] on held-out users and 93.4 on the DB-only merchants; the record in the prompt with the category reads 86.6 [82.5, 90.4] and 91.8 (paired +1.2 [-2.7, 5.0] and +1.6 [-3.5, 6.5]). The category field adds little to the record (+1.6 overall, +3.3 on the DB-only merchants): the product sentence already names the category for these disjoint pools, and section 43's ambiguous records are where it would matter more. The episodes' clearer edge is on coined names (+9.2 [-2.4, 20.7]), which comes from the all-label loss the record arms do not use, not from the database. On bf16 the no-DB all-label arm reads 76.8 against 73.4 on the 4-bit base in section 53, inside the spread of the cross-precision runs of section 52.

REAL-16 asked whether database episodes beat the record in the prompt at equal information. They match it. So a merchant database can be served from the weights at no loss against retrieval, at this size: no lookup at serving time, and it still works when retrieval cannot find the record, at the cost of retraining when the database changes. Which to prefer then turns on the database's size and churn: row 59 measures how many merchants the weights can hold, and the combination (episodes in training, the record in the prompt when it is found) is the obvious next arm.

Not done: episodes with the record also in the prompt, the record arms with the all-label loss and rename augmentation, and the ambiguous database of section 43.

## 56. Batch size on the H100: 16 sequences per step at 1e-4 is already the efficient point; bigger batches match it only with the rate scaled, more data per step buys at most a point, and larger passes are not faster (TRAIN-12)

*PLAN step 60, from the owner's question whether batch sizes had been ablated on the H100. Code: `EFF_BATCH` in `scripts/exp_categoriser.py` (sequences per optimizer step; 16 in every earlier run), `scripts/modal_jobs/r60.json`, `scripts/batch_tables.py`. The all-label no-DB categoriser on bf16, all 20 users, one seed per setting against the three batch-16 seeds of section 52.4. [Q1]*

Only the layout had been compared before (four passes of 4 against one of 16 at the same 16 sequences per step, section 54). This step varies the effective batch itself, at equal samples (fewer steps for bigger batches, the rate at 1e-4 and scaled by the square root of the batch ratio) and at equal steps (two and four times the data), and the pass size for throughput.

**Table 56.1: batch size and pass size on the H100 (all-label no-DB categoriser, bf16, all 20 users; accuracy %)**

| setting | per step | per pass | steps | rate | sequences | minutes | tokens/s | peak GiB | final loss | all | in history | determined by category | coined | DB-only | ARC-Easy | MMLU | ICL symbol |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| batch 16, seed 0 | 16 | 16 | 200 | 1e-4 | 3,200 | 3.0 | 14052 | 28.69 | 0.001 | 79.6 | 85.6 | 87.4 | 88.0 | 54.1 | 75.0 | 55.0 | 73.5 |
| batch 16, seed 1 | 16 | 16 | 200 | 1e-4 | 3,200 | 2.9 | 14697 | 28.47 | 0.003 | 78.8 | 83.8 | 87.0 | 86.7 | 52.5 | 77.5 | 57.5 | 70.3 |
| batch 16, seed 2 | 16 | 16 | 200 | 1e-4 | 3,200 | 2.9 | 14637 | 28.57 | 0.002 | 81.6 | 86.9 | 89.0 | 88.8 | 57.4 | 72.0 | 55.0 | 74.0 |
| batch 16, passes of 8 | 16 | 8 | 200 | 1e-4 | 3,200 | 2.9 | 14761 | 18.0 | 0.002 | 80.7 | 86.7 | 88.4 | 87.6 | 54.9 | 77.5 | 56.5 | 74.5 |
| batch 32, equal samples | 32 | 32 | 100 | 1.41e-4 | 3,200 | 3.3 | 13033 | 50.96 | 0.004 | 79.5 | 84.9 | 87.2 | 87.1 | 51.6 | 73.5 | 53.5 | 72.4 |
| batch 32, equal samples, rate unscaled | 32 | 32 | 100 | 1e-4 | 3,200 | 3.3 | 13135 | 50.96 | 0.005 | 79.7 | 85.1 | 86.2 | 87.6 | 51.6 | 75.5 | 53.5 | 74.0 |
| batch 64, equal samples | 64 | 32 | 50 | 2e-4 | 3,200 | 3.3 | 13027 | 51.3 | 0.017 | 80.1 | 85.8 | 87.6 | 87.1 | 55.7 | 76.5 | 51.5 | 70.8 |
| batch 64, equal samples, rate unscaled | 64 | 32 | 50 | 1e-4 | 3,200 | 3.3 | 12905 | 51.3 | 0.07 | 72.3 | 78.4 | 77.0 | 73.1 | 52.5 | 72.5 | 54.5 | 70.8 |
| batch 32, equal steps | 32 | 32 | 200 | 1.41e-4 | 6,400 | 6.4 | 13321 | 50.96 | 0.0 | 81.6 | 87.4 | 89.8 | 90.4 | 60.7 | 77.5 | 54.0 | 72.9 |
| batch 64, equal steps | 64 | 32 | 200 | 2e-4 | 12,800 | 13.1 | 13105 | 51.4 | 0.0 | 80.9 | 87.2 | 89.8 | 92.4 | 58.2 | 80.0 | 54.0 | 72.9 |

The H100 is saturated at 8 to 16 sequences per pass: 14,761 tokens a second at 8, about 14,400 at 16 and 13,000 at 32, where each batch is padded to its longest sequence. At equal samples (3,200 sequences) a batch of 32 or 64 matches the batch-16 seeds (79.5 to 80.1 against 78.8 to 81.6) only when the rate is scaled; the batch of 64 at the unscaled rate stops at 72.3 with a final loss of 0.07, having had 50 steps of which 20 are warmup. At equal steps, twice and four times the data read 81.6 and 80.9, at the top of the seed range, for two and four times the compute. The general measures do not move. TRAIN-12's answer: keep 16 sequences per step at 1e-4 for 200 steps (three minutes on the H100); if a larger batch is ever wanted, scale the rate by the square root of the batch ratio and lengthen the warmup with it.

## 57. Novel merchants from a real POI database: with a readable name the categoriser files unknown businesses well (database episodes 88% on descriptive names, clean rendering), the bank-statement truncation costs every model without a record about 15 points, a real-database record fixes both at 85%, and the user's coined names remain the gap a strong reader closes (REAL-19)

*PLAN step 62, the owner's idea to build the novel merchants from Overture. Code: `scripts/build_novel_merchants.py` (frozen `data/processed/novel_merchants_v1.json`), `scripts/build_novel_merchants_variants.py` (`_full`, `_clean`), `ITEMS_SET` in `scripts/exp_real6.py`, `ADAPTERS_FROM` in `scripts/modal_app.py`, `scripts/modal_jobs/r62*.json`, `scripts/novel_merchants_tables.py`, `scripts/blind_ceiling.py` with its prompts and answers in `results/blind_opus/`. Row 58's bf16 fold adapters on their held-out users, and the untrained instruct model. [Q1, Q2, Q3]*

REAL-6 cannot ask whether a categoriser can infer an unknown merchant's kind from its name: its merchants all have records and its opaque names mean nothing. This set takes real US businesses from Overture's places (release 2026-09-23.1, downloaded 2026-09-26 with per-row licences, `data/external/overture_places_2026-09-23.1/`) that are in no database and no user history: 100 per standard category through an explicit map of Overture's taxonomy (ambiguous types such as furniture or convenience stores left out), each category split into 40 independents whose name contains a word for their kind ("Chappaqua Dentistry"), 40 whose name contains no category word of any category ("Bark N' Bubbles"), and 20 chains (a brand with 25 or more US locations); 1,198 items, each put to one REAL-6 user with that user's frozen 24 shots and labelled in their scheme, with a record built from Overture's category ("Chappaqua Dentistry is listed as a dental clinic") for the record arms. Overture's own category is the label, errors included (a liquor store tagged grocery_store; famous chains without a brand tag land among the plain names). At the owner's request the same items exist in three renderings that differ only in the query's statement string: the generator's (names cut to 8 or 10 characters or abbreviated: `THE GRMNG SHPP*8011`), the full name with bank noise (`DEBIT CARD PURCHASE THE GROOMING SHOPPE #0366 LODI NJ`) and the clean name first (`The Grooming Shoppe, Lodi NJ`).

**Table 57.1: the novel merchants (held-out users; top-1, top-3 and bits from the scorecard: options ranked by summed log-probability, bits after the leave-users-out temperature; the group columns are top-1 under the mean-per-token rule of every earlier REAL-6 table; the two rules differ by under a point)**

| arm | rendering | top-1 | top-3 | bits | descriptive | plain | chain | standard name | renamed | coined |
|---|---|---|---|---|---|---|---|---|---|---|
| untrained instruct, no record | A obscure | 39.6 | 57.6 | 2.93 | 46.2 | 29.8 | 37.8 | 42.6 | 44.8 | 13.1 |
| untrained instruct, no record | B full name + bank noise | 49.7 | 68.7 | 2.47 | 59.2 | 34.8 | 49.2 | 57.0 | 54.5 | 10.0 |
| untrained instruct, no record | C clean | 53.6 | 70.5 | 2.32 | 63.5 | 39.8 | 54.6 | 62.2 | 62.5 | 7.7 |
| untrained instruct + Overture record | A obscure | 69.3 | 84.7 | 1.46 | 65.6 | 71.7 | 73.5 | 83.7 | 72.3 | 30.8 |
| untrained instruct + Overture record | B full name + bank noise | 69.9 | 85.5 | 1.43 | 68.8 | 71.5 | 71.0 | 84.0 | 73.2 | 31.7 |
| untrained instruct + Overture record | C clean | 71.2 | 86.0 | 1.42 | 70.8 | 71.5 | 71.8 | 84.4 | 75.8 | 30.8 |
| SFT no DB | A obscure | 52.8 | 68.4 | 2.41 | 62.9 | 36.5 | 62.2 | 55.1 | 53.7 | 42.1 |
| SFT no DB | B full name + bank noise | 65.9 | 80.8 | 1.81 | 80.8 | 47.1 | 72.7 | 68.6 | 68.1 | 53.8 |
| SFT no DB | C clean | 67.8 | 80.9 | 1.73 | 82.3 | 50.2 | 74.8 | 70.9 | 70.5 | 55.7 |
| database episodes | A obscure | 57.2 | 73.8 | 2.10 | 67.1 | 42.9 | 66.0 | 59.3 | 59.0 | 48.4 |
| database episodes | B full name + bank noise | 71.0 | 83.6 | 1.51 | 85.6 | 53.5 | 79.8 | 75.9 | 73.4 | 57.9 |
| database episodes | C clean | 72.5 | 85.5 | 1.41 | 87.9 | 53.5 | 78.6 | 75.9 | 74.1 | 60.2 |
| record arm + Overture record | A obscure | 85.1 | 93.1 | 0.88 | 85.6 | 85.8 | 85.3 | 94.3 | 87.6 | 61.1 |
| record arm + Overture record | B full name + bank noise | 85.1 | 92.8 | 0.87 | 85.8 | 85.8 | 84.0 | 94.3 | 87.8 | 59.7 |
| record arm + Overture record | C clean | 84.0 | 92.5 | 0.87 | 85.6 | 84.8 | 83.2 | 94.7 | 86.7 | 57.5 |
| record + category arm + Overture record | A obscure | 84.8 | 93.7 | 0.81 | 84.4 | 85.2 | 84.5 | 93.3 | 84.5 | 64.7 |
| record + category arm + Overture record | B full name + bank noise | 84.6 | 94.0 | 0.80 | 83.1 | 85.2 | 84.5 | 93.0 | 83.8 | 64.3 |
| record + category arm + Overture record | C clean | 85.1 | 94.0 | 0.79 | 83.8 | 86.5 | 85.3 | 93.9 | 85.4 | 63.8 |

**Table 57.2: clean minus obscure on the same items, top-1 points [user-resampled interval]**

| arm | all | descriptive | plain | chain | coined |
|---|---|---|---|---|---|
| untrained instruct, no record | +14.3 [+9.2, +19.2] | +17.3 [+10.5, +23.8] | +10.0 [+4.2, +15.0] | +16.8 [+9.6, +24.7] | -5.4 [-10.4, -0.6] |
| untrained instruct + Overture record | +1.7 [-0.9, +4.3] | +5.2 [+1.5, +8.6] | -0.2 [-3.4, +2.7] | -1.7 [-4.7, +1.9] | +0.0 [-6.1, +4.7] |
| SFT no DB | +15.8 [+13.8, +18.4] | +19.4 [+15.5, +24.3] | +13.8 [+10.2, +17.0] | +12.6 [+8.3, +16.9] | +13.6 [+8.4, +19.6] |
| database episodes | +15.1 [+12.2, +18.8] | +20.8 [+16.7, +26.0] | +10.6 [+6.4, +14.5] | +12.6 [+7.6, +19.4] | +11.8 [+6.2, +17.3] |
| record arm + Overture record | -0.8 [-2.0, +0.3] | +0.0 [-1.4, +1.4] | -1.0 [-3.0, +0.6] | -2.1 [-3.9, -0.4] | -3.6 [-8.5, +0.0] |
| record + category arm + Overture record | +0.4 [-0.4, +1.4] | -0.6 [-1.7, +0.3] | +1.2 [+0.0, +2.6] | +0.8 [-1.0, +2.8] | -0.9 [-2.9, +1.1] |

**Table 57.3: descriptive names in the obscure rendering (n = 480; the category word survives in 65%; accuracy %)**

| arm | word survives the rendering (n=314) | word lost (n=166) | standard single category (n=203) | renamed single (n=135) | merged (n=47) | coined (n=95) | word survives + standard single (n=131) |
|---|---|---|---|---|---|---|---|
| untrained instruct, no record | 60.8 | 18.7 | 55.7 | 54.8 | 48.9 | 12.6 | 76.3 |
| SFT no DB | 79.6 | 31.3 | 66.5 | 68.9 | 48.9 | 53.7 | 85.5 |
| database episodes | 83.8 | 35.5 | 70.4 | 70.4 | 59.6 | 58.9 | 90.1 |
| record arm + Overture record | 85.4 | 86.1 | 94.1 | 93.3 | 76.6 | 61.1 | 93.9 |

**Table 57.4: a blind Opus 5.5 ceiling on 60 items (40 with coined category names), obscure (A) and clean (C) renderings (accuracy %)**

| reader | coined, all (n=40) | coined, descriptive (n=14) | coined, plain (n=14) | coined, chain (n=12) | controls (n=20) | all (n=60) |
|---|---|---|---|---|---|---|
| blind Opus 5.5, rendering A | 75 | 93 | 50 | 83 | 80 | 77 |
| blind Opus 5.5, rendering C | 88 | 100 | 71 | 92 | 90 | 88 |
| 3B no DB (obscure) | 42 | 57 | 29 | 42 | 65 | 50 |
| 3B database episodes (obscure) | 55 | 64 | 36 | 67 | 75 | 62 |
| 3B database episodes (clean) | 65 | 79 | 50 | 67 | 75 | 68 |

### 57.1 What the model infers from a name, and what the rendering takes away

With a readable name, the categoriser trained with database episodes files these never-seen businesses at 72.5 top-1 and 85.5 top-3 (clean): 87.9 on descriptive names, 78.6 on chains, 53.5 on names with no category word, which is still seven times chance. The untrained instruct model reads 53.6; label SFT without a database 67.8; the database episodes add 4.7 on top of that although none of these businesses is in their database (Table 57.1): what they teach is partly the general step from "what kind of store" to "which of this user's categories". The bank-statement truncation costs every model without a record 14 to 16 points (Table 57.2), almost all of it from truncation and abbreviation: the full name inside bank noise reads within two points of the clean name. On descriptive names the category word survives the generator's rendering 65% of the time, and where it is lost the models fall to 19 to 36 (Table 57.3); where it survives and the user's category has its standard name, the episode model reads 90.1. The remaining errors there are mostly ambiguous businesses or Overture's labels (a gas-station food mart tagged grocery_store, an electrical supplier tagged hardware_store).

### 57.2 A real-database record, and the coined names

Given the Overture category as a record, the record-in-prompt categoriser, trained on REAL-6's synthetic product records, reads 85% in every rendering and every name group (plain names 85.8): retrieval from a real POI database transfers directly, and makes the rendering irrelevant. The weak cell in every arm is the user's coined category name: 42 to 60 without a record, 58 to 65 with one. Two explanations were tested and failed: the coined category's examples in the prompt containing a recognisable chain does not raise accuracy (42 against 42 for the no-DB arm), and neither does whether the training users used the same coined word for another category (REAL-6's coined words come from a shared list of 24). A blind Opus 5.5 pass on 60 items (Table 57.4) reads the coined names at 88% from clean strings (100% on descriptive names; its reasons are the induction chain itself, "Macy's went to Tabbin") and at 75% from the generator's strings: the information is in the prompt, the 3B model reaches 55 to 65 of it. The untrained model's coined-name accuracy falls as the name becomes clearer (13 to 8): with a readable business name it prefers the category word that means the business over the user's invented word the examples point to.

### 57.3 What the step says

REAL-19 asked whether the categoriser can infer an unknown merchant's kind from its name and the user's examples, and whether baking a database in costs that. It can, well when the name is readable (88 on descriptive names, 79 on chains), and the baked-in database helps rather than costs (+4.7 clean and +4.4 obscure top-1 over the same recipe without it). The statement string matters as much as the model: truncated bank strings take 15 points that a record in the prompt gives back. The user's coined category names are the capacity gap between a 3B model and a strong reader, and row 64 takes that apart.

## 58. Capacity: database episodes hold 20,000 merchants as well as 240 when each gets about 30 training rows (DB-only merchants 92 to 96, opaque ones 90 to 94); a fixed budget spread thinner loses them, and the cost grows linearly (REAL-17)

*PLAN step 59. Code: `DB_EXTRA` (`merchants.build_extra`: generated opaque merchants in REAL-6's style, a standard category each, product records from the pools) and `DB_EPISODES` (database episodes as their own pool entries, targets cycling through the merchants) in `scripts/exp_categoriser.py`, `scripts/modal_jobs/r59.json`, `scripts/db_scale_tables.py`. H100, bf16, all-label, row 42's four held-out folds; REAL-6's 60 DB-only merchants are the measure. [Q2]*

A CPU check before launch found that the training pool holds 150 episodes per user, so more steps alone would repeat the same database rows; database episodes became their own pool entries with a set count. Fixed exposure gives every merchant 30 database rows (the 800 episodes of the 240-merchant run, scaled), the user episodes unchanged, 1.4 passes over the pool; fixed budget keeps 800 database episodes at every size.

**Table 58.1: database episodes as the database grows (held-out users; accuracy %; last column: DB-only minus no DB on the same items [user interval])**

| merchants in the DB | mode | DB rows per merchant | steps | train minutes | all | coined | determined by category | DB-only | DB-only known | DB-only opaque | DB-only minus no DB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none (section 55) | - | 0 | 200 | 2.9 | 76.8 | 77.9 | 83.5 | 56.6 | 83.1 | 19.6 | - |
| 240 | fixed exposure | 30.0 | 267 | 3.9 | 86.3 | 83.5 | 93.5 | 91.8 | 91.5 | 92.2 | +35.2 [+25.2, +44.7] |
| 1,000 | fixed exposure | 30.0 | 489 | 7.1 | 86.3 | 83.1 | 94.1 | 95.9 | 97.2 | 94.1 | +39.3 [+30.3, +48.6] |
| 1,000 | fixed budget | 7.2 | 267 | 4.0 | 83.1 | 76.3 | 89.6 | 82.8 | 93.0 | 68.6 | +26.2 [+18.3, +35.1] |
| 5,000 | fixed exposure | 30.0 | 1656 | 24.4 | 86.7 | 82.7 | 93.9 | 93.4 | 94.4 | 92.2 | +36.9 [+28.8, +45.0] |
| 5,000 | fixed budget | 1.4 | 267 | 4.0 | 79.6 | 77.5 | 86.6 | 69.7 | 87.3 | 45.1 | +13.1 [+4.9, +22.0] |
| 20,000 | fixed exposure | 30.0 | 6031 | 89.2 | 85.8 | 87.6 | 93.7 | 92.6 | 94.4 | 90.2 | +36.1 [+25.9, +46.7] |
| 20,000 | fixed budget | 0.4 | 267 | 4.0 | 77.4 | 75.1 | 83.3 | 56.6 | 80.3 | 23.5 | +0.0 [-10.6, +13.2] |

At a fixed exposure the DB-only merchants read 91.8, 95.9, 93.4 and 92.6 at 240, 1,000, 5,000 and 20,000 merchants, the opaque ones among them 92.2, 94.1, 92.2 and 90.2, and nothing else moves (all items 85.8 to 86.7, coined names 83 to 88): no interference up to 20,000 merchants for a rank-64 adapter on a 3B model. Section 29 found interference at 5,000 species for recalling several attributes per entity; filing a merchant into one of twelve categories is a far smaller fact. At a fixed budget the rows per merchant fall with the database (7.2, 1.4, 0.4) and so does the gain (82.8, 69.7, 56.6: nothing at 20,000). The cost is linear in the database: 20,000 merchants at 30 rows each is 6,031 steps, 89 minutes on one H100 (about $6). REAL-17's answer at this range: exposure, not capacity, is the constraint. Open: the minimum rows per merchant (between 7 and 30), the curve beyond 20,000 (a million merchants at this rate is about 75 H100-hours), and real merchants in place of generated ones (row 66, on Overture).

## 59. One scorecard for every run: on REAL-6 a system with no model (the user's own label for the merchant, else other users', else the user's most-used category) reads 87% top-1 on held-out users, level with the best categorisers; their value there is the top-3 (98 against 26) and everything a lookup cannot reach (EVAL-9)

*PLAN step 63. Code: `ai_experiments.scorecard` (top-1, top-3, MRR; bits and Brier after the leave-users-out temperature of section 50; coverage at 95% and 98% precision with the threshold from other users; the baseline ladder: uniform, the user's usage prior, their merchant lookup, other users' labels mapped into the user's scheme, the cascade lookup -> other users -> prior; skill as the share of the headroom over the best baseline), `scripts/scorecard_tables.py`. [Q1-3]*

The owner's research framing (QUESTIONS.md, research agenda) asks for metrics that match how a prediction is used: auto-file the extremely confident, suggest the rest. Top-1 against uniform chance matched neither.

**Table S.1: REAL-6 on held-out users, the scorecard (top-k in %; bits = -log2 p(gold) after the leave-users-out temperature; auto-file threshold chosen on other users; skill = share of the headroom over the best baseline; intervals resample users)**

| run | n | top-1 [interval] | top-3 [interval] | MRR | bits left [interval] | bits gained over the usage prior | auto-file at 98%: coverage (precision) | uniform top-1 / top-3 | usage prior top-1 / top-3 | lookup → other users → prior, top-1 | skill top-1 / top-3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SFT no DB, 800 steps (3090, 4-bit) | 1179 | 69.6 [65.7, 73.7] | 83.9 [81.0, 86.6] | 0.78 | 1.49 [1.29, 1.66] | +2.80 | 34.3 (96.8) | 7.3 / 21.9 | 8.6 / 25.7 | 87.0 | -134 / 78 |
| SFT no DB, all-label (H100, bf16) | 1179 | 76.7 [73.3, 79.8] | 90.2 [87.7, 92.4] | 0.84 | 1.22 [1.04, 1.41] | +3.07 | 7.9 (92.5) | 7.3 / 21.9 | 8.6 / 25.7 | 87.0 | -80 / 87 |
| database episodes (H100, bf16) | 1179 | 87.4 [83.5, 90.8] | 97.8 [95.5, 99.3] | 0.93 | 0.63 [0.45, 0.83] | +3.66 | 24.6 (91.7) | 7.3 / 21.9 | 8.6 / 25.7 | 87.0 | 3 / 97 |
| record in the prompt (H100, bf16) | 1179 | 83.3 [78.4, 87.6] | 96.6 [94.5, 98.4] | 0.90 | 0.74 [0.58, 0.92] | +3.55 | 50.8 (97.3) | 7.3 / 21.9 | 8.6 / 25.7 | 87.0 | -29 / 95 |
| record with category in the prompt (H100, bf16) | 1179 | 86.0 [81.8, 90.0] | 97.5 [95.8, 99.0] | 0.92 | 0.57 [0.44, 0.71] | +3.72 | 65.6 (97.9) | 7.3 / 21.9 | 8.6 / 25.7 | 87.0 | -8 / 97 |
| record in the prompt + rename augmentation (3090, 4-bit) | 1179 | 86.5 [83.3, 89.7] | 95.6 [93.2, 97.9] | 0.91 | 0.76 [0.57, 1.01] | +3.53 | 38.4 (97.1) | 7.3 / 21.9 | 8.6 / 25.7 | 87.0 | -4 / 94 |
| no DB, all-label + rename augmentation (3090, 4-bit) | 1179 | 78.0 [73.7, 82.1] | 89.5 [87.4, 91.7] | 0.85 | 1.12 [0.93, 1.32] | +3.17 | 52.5 (97.6) | 7.3 / 21.9 | 8.6 / 25.7 | 87.0 | -69 / 86 |

**Table S.2: the novel merchants (Overture, obscure renderings), the scorecard (no merchant lookup or other-user label exists for them)**

| run | n | top-1 [interval] | top-3 [interval] | MRR | bits left [interval] | bits gained over the usage prior | auto-file at 98%: coverage (precision) | uniform top-1 / top-3 | usage prior top-1 / top-3 | lookup → other users → prior, top-1 | skill top-1 / top-3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| untrained instruct, no record | 1198 | 39.6 [34.9, 43.3] | 57.6 [52.6, 61.2] | 0.53 | 2.93 [2.78, 3.15] | +1.20 | 0.1 (0.0) | 7.9 / 23.6 | 7.8 / 28.3 | 7.8 | 34 / 41 |
| untrained instruct + Overture record | 1198 | 69.3 [63.4, 75.4] | 84.7 [80.3, 88.8] | 0.79 | 1.46 [1.21, 1.74] | +2.68 | 14.4 (96.0) | 7.9 / 23.6 | 7.8 / 28.3 | 7.8 | 67 / 79 |
| SFT no DB | 1198 | 52.8 [46.7, 57.6] | 68.4 [62.7, 72.7] | 0.64 | 2.41 [2.23, 2.62] | +1.73 | 1.3 (86.7) | 7.9 / 23.6 | 7.8 / 28.3 | 7.8 | 49 / 56 |
| database episodes | 1198 | 57.2 [51.6, 61.4] | 73.8 [69.5, 77.4] | 0.69 | 2.10 [1.91, 2.35] | +2.03 | 3.6 (95.3) | 7.9 / 23.6 | 7.8 / 28.3 | 7.8 | 54 / 63 |
| record arm + Overture record | 1198 | 85.1 [78.5, 90.4] | 93.1 [90.3, 95.6] | 0.90 | 0.88 [0.63, 1.16] | +3.26 | 42.6 (96.7) | 7.9 / 23.6 | 7.8 / 28.3 | 7.8 | 84 / 90 |
| record + category arm + Overture record | 1198 | 84.8 [79.6, 89.1] | 93.7 [90.3, 96.1] | 0.90 | 0.81 [0.6, 1.04] | +3.33 | 48.5 (97.2) | 7.9 / 23.6 | 7.8 / 28.3 | 7.8 | 84 / 91 |

**Table S.3: what the auto-file set (98% precision) is made of, REAL-6 held-out users (share of auto-filed items per group, and the group's share of all items)**

| run | in history | labelled seen, not in history | determined by category | split category |
|---|---|---|---|---|
| SFT no DB, 800 steps (3090, 4-bit) | 51% of auto (38% of all) | 7% of auto (10% of all) | 40% of auto (43% of all) | 1% of auto (9% of all) |
| SFT no DB, all-label (H100, bf16) | 45% of auto (38% of all) | 13% of auto (10% of all) | 37% of auto (43% of all) | 5% of auto (9% of all) |
| database episodes (H100, bf16) | 38% of auto (38% of all) | 10% of auto (10% of all) | 44% of auto (43% of all) | 8% of auto (9% of all) |
| record in the prompt (H100, bf16) | 46% of auto (38% of all) | 10% of auto (10% of all) | 42% of auto (43% of all) | 2% of auto (9% of all) |
| record with category in the prompt (H100, bf16) | 43% of auto (38% of all) | 9% of auto (10% of all) | 44% of auto (43% of all) | 2% of auto (9% of all) |
| record in the prompt + rename augmentation (3090, 4-bit) | 48% of auto (38% of all) | 8% of auto (10% of all) | 42% of auto (43% of all) | 3% of auto (9% of all) |
| no DB, all-label + rename augmentation (3090, 4-bit) | 41% of auto (38% of all) | 8% of auto (10% of all) | 49% of auto (43% of all) | 1% of auto (9% of all) |

On REAL-6's held-out users the no-model cascade reads 87.0 top-1: the user's own past label for the merchant covers the merchants in their history, other users' labels cover almost every other merchant (each REAL-6 merchant is in someone's history, the DB-only ones included, since they are held out of training rows only), and the users label consistently. The best categorisers sit on it (database episodes 87.4, record with the category 86.0; skill 3 and -8), so REAL-6's top-1 mostly measures how well a model re-derives what a lookup gives free, as section 48 found for the seen cells. Their value is in the ranking (top-3 97 to 98 against 26 for the usage prior; skill 97) and in what a lookup cannot reach: on the novel merchants, where no lookup or other-user label exists, the cascade is the usage prior (7.8) and the categorisers read 53 to 85 (Table 57.1). Calibrated, the models leave 0.6 to 1.5 bits of uncertainty on REAL-6 (usage prior 4.3) and 0.8 to 2.9 on the novel merchants. Auto-filing at a 98% threshold chosen on other users realises 92 to 98% precision on new users, not 98: confidence thresholds drift between users, which supports the owner's rule of auto-filing a merchant the user has filed consistently before (the lookup is exact on those) and suggesting everything else. Every table from here reports this scorecard, skill over the cascade included; REAL-6's successor (row 43) and POI-1 (row 65) are where the lookup and collaborative signals stop answering.


## 60. Label induction: one or two examples of other businesses filed under a coined word teach every model what the word means (14B untrained 21 to 80% on v1, Opus 25 to 100% on v2), the fine-tuned 3B reads it like an untrained 14B, and the models copy the nearest example where a strong reader reasons from the category (REAL-20)

PLAN step 64. REAL-20 asks when a model infers what a meaningless category name means from the user's examples in the prompt. The
item set (`scripts/build_label_induction.py`, `data/processed/label_induction_v1.json`) holds 300 queries: real US businesses with
descriptive names from Overture, 25 for each of the twelve standard categories, rendered clean ("Name, City ST"). Each query is shown
under 12 conditions, which pair item by item. Every scheme holds the twelve standard categories. The query's category (gold) is
renamed to a fresh coined word ("Kofa"), and two other categories are coined too. The 24 labelled examples are other real businesses.

The base condition gives 2 gold examples, businesses of another kind in the gold category (the query is a pizzeria, the examples a
taqueria and a cafe). The other conditions vary one factor at a time:

- the number of gold examples: 0, 1, 4, 8;
- the examples' kind: the same kind as the query, or opaque generated names ("Oskpobury Group");
- the number of coined categories: 1 or 6;
- a decoy: one example of the query's own kind filed under another category, with and without the gold examples;
- a control that keeps the standard name.

The readers are:

- untrained Qwen2.5-Instruct at 3B, 7B and 14B;
- three 3B categorisers trained on REAL-6 users of fold 0 (all-label loss; with database episodes; with rename augmentation, the last
  trained for this step: `categoriser_Qwen2.5-3B-Instruct_none_h100bf16_f0_ren50_alllab_lora`);
- blind Opus 5.5 subagents in a Latin square: 60 queries, six conditions, six agents, and no agent sees a query twice
  (`results/blind_opus/`).

All models ran on Modal in bf16 (`scripts/modal_jobs/r64*.json`) and were scored by the sum rule; tables from
`scripts/label_induction_tables.py`.

**v1 is solvable by elimination.** Blind Opus scored 98 to 100% in every condition, including zero gold examples. Its reasons
say how: "No Groceries category; Kuvir is the unused coined label". The scheme leaves out exactly one standard category and puts one
coined word with no other examples in its place. v2 (`--empty 3`, `label_induction_v2.json`) is v1 item for item plus three coined
categories with no examples anywhere, as users have categories with no recent transactions. In v2 elimination leaves four words and
only the gold examples can decide. Blind Opus confirms this: 25% with no gold examples, chance among four.

**Table 60.1: v1 (12 options; solvable by elimination): top-1 % by condition, 300 queries per condition (blind Opus: its 60)**

| condition | Qwen2.5-3B-Instruct, untrained | Qwen2.5-7B-Instruct, untrained | Qwen2.5-14B-Instruct, untrained | 3B SFT no DB, all-label (fold 0) | 3B database episodes (fold 0) | 3B all-label + rename augmentation (fold 0) | blind Opus 5.5 (60 queries) |
|---|---|---|---|---|---|---|---|
| base | 50 | 73 | 87 | 73 | 81 | 85 | 100 |
| n_gold=0 | 7 | 11 | 21 | 22 | 11 | 26 | 100 |
| n_gold=1 | 44 | 69 | 80 | 65 | 73 | 78 | 100 |
| n_gold=4 | 62 | 77 | 88 | 80 | 86 | 87 | - |
| n_gold=8 | 72 | 83 | 89 | 84 | 89 | 90 | - |
| kind=same_kind | 83 | 93 | 96 | 90 | 93 | 95 | - |
| kind=opaque | 6 | 10 | 16 | 24 | 20 | 25 | 98 |
| n_coined=1 | 54 | 76 | 88 | 82 | 84 | 87 | - |
| n_coined=6 | 48 | 68 | 84 | 67 | 78 | 83 | - |
| decoy | 36 | 49 | 62 | 57 | 62 | 61 | 98 |
| decoy, n_gold=0 | 2 | 2 | 6 | 10 | 4 | 5 | - |
| control: standard name | 93 | 93 | 97 | 95 | 96 | 94 | 100 |


**Table 60.2: v2 (15 options: three empty coined categories): top-1 % by condition, 300 queries per condition (blind Opus: its 60)**

| condition | Qwen2.5-3B-Instruct, untrained | Qwen2.5-7B-Instruct, untrained | Qwen2.5-14B-Instruct, untrained | 3B SFT no DB, all-label (fold 0) | 3B database episodes (fold 0) | 3B all-label + rename augmentation (fold 0) | blind Opus 5.5 (60 queries) |
|---|---|---|---|---|---|---|---|
| base | 46 | 60 | 77 | 68 | 78 | 77 | 100 |
| n_gold=0 | 5 | 14 | 12 | 15 | 11 | 12 | 25 |
| n_gold=1 | 34 | 56 | 66 | 55 | 69 | 71 | 100 |
| n_gold=4 | 55 | 69 | 83 | 76 | 84 | 84 | - |
| n_gold=8 | 66 | 78 | 87 | 81 | 90 | 90 | - |
| kind=same_kind | 79 | 89 | 93 | 89 | 92 | 94 | - |
| kind=opaque | 4 | 7 | 5 | 15 | 15 | 10 | 3 |
| n_coined=1 | 47 | 62 | 78 | 69 | 80 | 78 | - |
| n_coined=6 | 40 | 59 | 74 | 63 | 75 | 79 | - |
| decoy | 30 | 45 | 58 | 51 | 60 | 60 | 97 |
| decoy, n_gold=0 | 1 | 2 | 3 | 7 | 5 | 4 | - |
| control: standard name | 92 | 93 | 96 | 94 | 94 | 94 | 100 |


**Table 60.3: v2 (15 options: three empty coined categories): the same on the blind reader's 60 queries only**

| condition | Qwen2.5-3B-Instruct, untrained | Qwen2.5-7B-Instruct, untrained | Qwen2.5-14B-Instruct, untrained | 3B SFT no DB, all-label (fold 0) | 3B database episodes (fold 0) | 3B all-label + rename augmentation (fold 0) | blind Opus 5.5 (60 queries) |
|---|---|---|---|---|---|---|---|
| base | 45 | 70 | 83 | 72 | 82 | 85 | 100 |
| n_gold=0 | 7 | 18 | 17 | 10 | 12 | 12 | 25 |
| n_gold=1 | 28 | 60 | 75 | 52 | 68 | 75 | 100 |
| n_gold=4 | 57 | 72 | 85 | 78 | 85 | 88 | - |
| n_gold=8 | 70 | 82 | 88 | 90 | 93 | 98 | - |
| kind=same_kind | 88 | 92 | 97 | 90 | 97 | 98 | - |
| kind=opaque | 2 | 8 | 7 | 12 | 15 | 8 | 3 |
| n_coined=1 | 48 | 73 | 83 | 72 | 82 | 80 | - |
| n_coined=6 | 43 | 70 | 82 | 72 | 82 | 82 | - |
| decoy | 33 | 53 | 63 | 62 | 72 | 68 | 97 |
| decoy, n_gold=0 | 2 | 3 | 5 | 8 | 5 | 2 | - |
| control: standard name | 95 | 92 | 95 | 97 | 93 | 97 | 100 |


**Table 60.4: the decoy condition, how often a wrong answer is the decoy's category (the one example of the query's own kind was filed there)**

| reader | v1 wrong | of which the decoy's category | v2 wrong | of which the decoy's category |
|---|---|---|---|---|
| Qwen2.5-3B-Instruct, untrained | 192 / 300 | 96 (50%) | 211 / 300 | 100 (47%) |
| Qwen2.5-14B-Instruct, untrained | 115 / 300 | 81 (70%) | 126 / 300 | 75 (60%) |
| 3B all-label + rename augmentation (fold 0) | 117 / 300 | 87 (74%) | 120 / 300 | 87 (73%) |

### 60.1 What the models do with a coined word

**The gold examples do the work.** On v2 the untrained 14B goes from 12% with no gold examples to 66 with one and 87 with eight; the
3B categorisers go from 11 to 15 with none to 55 to 71 with one and 81 to 90 with eight. Examples of the query's own kind give
89 to 94, close to the control that keeps the standard name (92 to 96).

**The meaning comes from the examples' names.** With opaque example names every reader stays near its no-example level (v2: 4 to
15% for the models, 3% for Opus). The models infer the word from what kind of businesses the examples are, not from the count of examples.

**Elimination is Opus's shortcut, not the models'.** On v1 with no gold examples the models read 7 to 26% where Opus reads 100.
Adding three empty coined categories (v2) costs the models 3 to 13 points at base (14B 87 to 77, the rename-trained 3B 85 to 77),
because each empty word is another plausible home for a business the examples do not explain. The number of coined categories
matters little otherwise (n_coined 1 against 6: 14B 78 against 74 on v2).

**Scale and training.** Untrained, 3B reads 46 at v2's base, 7B 60 and 14B 77. Fine-tuning the 3B on REAL-6 user episodes brings
it to 68 (all-label loss). Database episodes and rename augmentation bring it to 77 to 78, level with the untrained 14B in almost
every row. Rename augmentation was built for this: episodes where category names are replaced by coined words. Database episodes
help as much here, probably because they also teach the model to file a merchant it has never seen by its kind.

### 60.2 The decoy: copying the nearest example

One example of the query's own kind filed under another category drops every model by 15 to 19 points on v2 (base 77 to 58 for the
14B, 77 to 60 for the rename-trained 3B). Of the wrong answers, 60 to 74% are exactly the decoy's category (Table 60.4). Opus keeps
97%. The models weight the single most similar example over the two examples that define the category, where Opus reads the
category. With no gold examples and a decoy, all models fall to 1 to 7%.

This is how a similarity-driven reader behaves, and it matches section 48: the models are strong at "this merchant, or one like it,
went there". It is not always wrong for the product. A user who filed one pizzeria under "Date night" may want the next pizzeria
there too; gold here is the query's standard category, which is one reading of the user. What the item set settles is the
mechanism: the models copy the nearest example; they do not weigh it against the category the other examples define.

### 60.3 What the step says

For Q3 (inferring meaningless category names), a coined word is learned from one or two labelled examples of other businesses of
the same kind, by every model from 3B up. Training on user episodes, above all with coined names or database rows, moves a 3B to
where a 14B starts. The remaining gap to a strong reader (77 against 100 at two examples) is a gap in weighing the evidence. The
models follow the most similar example and are drawn to categories with no examples; Opus uses both examples and ignores a single
odd one. Two follow-ups:

- Train with decoys, and with empty categories, in the episodes (the rename-augmentation recipe plus deliberate inconsistencies).
- Test whether the 3B separates "same kind" from "same category" in its hidden states (row 67, probes).

POI-1 (row 65) now asks the same of real places at scale: schemes of 12 to 20 categories over Overture's 288 basic categories, with
places no user has filed.


## 61. POI-1: real places at 12 to 20 categories per user are hard for every reader (blind Opus 56%, the best 3B 59); a lookup by the place's kind answers most of what is answerable, the model adds 15 points on top (80%), and examples chosen by kind lift every model by 8 to 18 points (POI-1)

PLAN step 65. POI-1 moves the categoriser off REAL-6's synthetic merchants onto real places, and makes the categories harder. The
set is `scripts/build_poi1.py`, frozen as `data/processed/poi1_v1.json`, built from Overture places 2026-09-23.1 (every place keeps
its Overture id and per-row source licence).

- **Users.** 200 synthetic users over real US places.
- **Categories.** Each user groups 20 to 45 of Overture's basic categories into 12 to 20 of their own. Groups follow Overture's
  top-level taxonomy, split and merged at random, so some cross top levels. Each is named one of three ways: readable ("Health
  care"), merged from two members ("Surgery & primary care or general clinic"), or a coined word ("Gavir").
- **Histories.** Each user has 150 places, weighted Zipf over their categories, and no place is shared between users. The prompt
  shows a frozen block of 24 of them as labelled examples.
- **Test items.** 2,053 places that are in nobody's history, so no merchant lookup answers them. Half are of a basic category
  already in the user's history ("seen kind"); half are of a basic category in the scheme but not in the history ("unseen kind").
- **Users are consistent by construction:** every basic category maps to exactly one of the user's categories. So the **kind
  lookup** is exact whenever it applies. It is the user's label for other places of the same Overture basic category, and it is
  what a places database plus the user's history gives without any model.

**Readers:**

- untrained Qwen2.5-Instruct 3B, 7B and 14B;
- two REAL-6 categorisers as transfer;
- three 3B categorisers trained on POI-1's users with fold 0 held out (`POI=poi1_v1` in `exp_categoriser.py`; all-label loss, 200 or
  800 steps, one with rename augmentation);
- blind Opus 5.5 on 120 items (20 per level, `results/blind_opus/poi1_v1/`).

All model runs were on Modal in bf16 (`scripts/modal_jobs/r65*.json`); tables from `scripts/poi1_tables.py`, with the scorecard
extended to a set's own users (`ai_experiments.scorecard`, `users=`, `fold_of=`).

**Table 61.1: POI-1, fold 0's held-out users (50 users): the scorecard (temperature and auto-file thresholds fitted leave-users-out within the fold's users, grouped by (id // 4) mod 4; kind lookup = the user's label for another place of the same Overture basic category)**

| reader | n | top-1 [interval] | top-3 | MRR | bits left | auto-file at 98%: coverage (precision) | usage prior top-1 / top-3 | kind lookup: share, top-1 where it answers | kind lookup → other users → prior, top-1 | skill top-1 over it / top-3 | kind lookup, else the model: top-1 (the model's top-1 where the lookup has nothing) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct, untrained | 507 | 44.6 [39.5, 49.5] | 65.3 | 0.58 | 2.93 | 5.3 (96.3) | 7.5 / 23.3 | 59%, 100.0 | 65.3 | -60 / 55 | 74.6 (37.7) |
| Qwen2.5-7B-Instruct, untrained | 507 | 47.7 [42.2, 53.3] | 68.0 | 0.61 | 2.61 | 6.7 (94.1) | 7.5 / 23.3 | 59%, 100.0 | 65.3 | -51 / 58 | 74.2 (36.7) |
| Qwen2.5-14B-Instruct, untrained | 507 | 52.9 [47.8, 58.0] | 71.6 | 0.66 | 2.39 | 3.7 (94.7) | 7.5 / 23.3 | 59%, 100.0 | 65.3 | -36 / 63 | 77.9 (45.9) |
| 3B trained on REAL-6 (all-label), transfer | 507 | 46.5 [41.4, 51.2] | 64.5 | 0.59 | 2.79 | 1.6 (87.5) | 7.5 / 23.3 | 59%, 100.0 | 65.3 | -54 / 54 | 74.6 (37.7) |
| 3B trained on REAL-6 (all-label + rename), transfer | 507 | 37.9 [32.8, 42.9] | 60.9 | 0.53 | 3.00 | 4.7 (95.8) | 7.5 / 23.3 | 59%, 100.0 | 65.3 | -79 / 49 | 69.4 (25.1) |
| 3B trained on POI-1, 200 steps | 507 | 57.8 [53.4, 62.1] | 76.5 | 0.70 | 2.12 | 10.7 (96.3) | 7.5 / 23.3 | 59%, 100.0 | 65.3 | -22 / 69 | 80.3 (51.7) |
| 3B trained on POI-1, 800 steps | 507 | 55.4 [50.4, 60.1] | 73.6 | 0.68 | 2.27 | 4.7 (95.8) | 7.5 / 23.3 | 59%, 100.0 | 65.3 | -28 / 66 | 78.9 (48.3) |
| 3B trained on POI-1, 800 steps + rename | 507 | 59.2 [55.5, 63.2] | 78.5 | 0.71 | 1.94 | 16.4 (92.8) | 7.5 / 23.3 | 59%, 100.0 | 65.3 | -18 / 72 | 79.7 (50.2) |

**Table 61.2: top-1 % by level, fold 0 (seen / unseen = the place's Overture basic category is / is not in the user's history; standard / renamed / new = readable / merged / coined category name); blind Opus on its own sample (all users, 20 per level)**

| reader | R6_seen_new | R6_seen_renamed | R6_seen_standard | R6_unseen_new | R6_unseen_renamed | R6_unseen_standard |
|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct, untrained | 24 | 54 | 60 | 6 | 38 | 53 |
| Qwen2.5-7B-Instruct, untrained | 37 | 51 | 66 | 13 | 34 | 49 |
| Qwen2.5-14B-Instruct, untrained | 37 | 52 | 70 | 19 | 49 | 57 |
| 3B trained on REAL-6 (all-label), transfer | 37 | 55 | 59 | 12 | 45 | 47 |
| 3B trained on REAL-6 (all-label + rename), transfer | 42 | 52 | 46 | 15 | 32 | 27 |
| 3B trained on POI-1, 200 steps | 45 | 66 | 69 | 21 | 45 | 69 |
| 3B trained on POI-1, 800 steps | 32 | 68 | 71 | 12 | 47 | 67 |
| 3B trained on POI-1, 800 steps + rename | 56 | 60 | 72 | 35 | 53 | 56 |
| blind Opus 5.5 (sample) | 45 (20) | 65 (20) | 90 (20) | 25 (20) | 40 (20) | 70 (20) |

**Table 61.3: top-1 % by what the prompt's 24 shots show x the category name type (fold 0; blind Opus on its sample, n in brackets)**

| reader | kind in shots, standard | kind in shots, renamed | kind in shots, new | other kinds under gold, standard | other kinds under gold, renamed | other kinds under gold, new | gold not in shots, standard | gold not in shots, renamed | gold not in shots, new |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct, untrained | 61 | 67 | 33 | 56 | 34 | 7 | 47 | 50 | 0 |
| Qwen2.5-7B-Instruct, untrained | 72 | 64 | 47 | 52 | 28 | 18 | 47 | 58 | 0 |
| Qwen2.5-14B-Instruct, untrained | 73 | 64 | 47 | 61 | 38 | 21 | 55 | 75 | 5 |
| 3B trained on REAL-6 (all-label), transfer | 64 | 67 | 47 | 48 | 38 | 14 | 47 | 67 | 5 |
| 3B trained on REAL-6 (all-label + rename), transfer | 51 | 64 | 51 | 31 | 28 | 23 | 26 | 58 | 0 |
| 3B trained on POI-1, 200 steps | 68 | 82 | 55 | 69 | 44 | 27 | 71 | 42 | 5 |
| 3B trained on POI-1, 800 steps | 66 | 82 | 35 | 74 | 44 | 21 | 66 | 58 | 0 |
| 3B trained on POI-1, 800 steps + rename | 72 | 69 | 56 | 66 | 48 | 48 | 47 | 67 | 21 |
| blind Opus 5.5 (sample) | 91 (11) | 92 (12) | 57 (14) | 77 (26) | 29 (24) | 25 (20) | 67 (3) | 75 (4) | 17 (6) |

(fold 0 items per cell: kind in shots, standard: 106, kind in shots, renamed: 39, kind in shots, new: 55, other kinds under gold, standard: 121, other kinds under gold, renamed: 61, other kinds under gold, new: 56, gold not in shots, standard: 38, gold not in shots, renamed: 12, gold not in shots, new: 19)

**Table 61.4: kind-retrieved shots, top-1 % on fold 0's items whose Overture basic category is in the user's history (frozen 24 shots → up to six of them replaced by history places of the query's kind), by category name type**

| reader | standard (n=157) | renamed (n=65) | new (n=78) | all |
|---|---|---|---|---|
| Qwen2.5-3B-Instruct, untrained | 60 → 75 | 54 → 69 | 24 → 50 | 49 → 67 |
| Qwen2.5-14B-Instruct, untrained | 70 → 79 | 52 → 77 | 37 → 67 | 58 → 75 |
| 3B trained on REAL-6 (all-label + rename), transfer | 46 → 66 | 52 → 69 | 42 → 62 | 47 → 66 |
| 3B trained on POI-1, 800 steps | 71 → 77 | 68 → 74 | 32 → 45 | 60 → 68 |
| 3B trained on POI-1, 800 steps + rename | 72 → 83 | 60 → 82 | 56 → 76 | 65 → 81 |

On all 200 users the untrained and transfer readers read as on fold 0 (3B 43.9, 7B 47.9, 14B 51.9, REAL-6 all-label 45.1, REAL-6
rename 38.5).

### 61.1 How hard the task is, and what decides it

POI-1 is far harder than REAL-6. The best reader, the POI-trained 3B with rename augmentation, reaches 59% top-1, and blind Opus
reaches 56% on its sample. Opus gets 90% on seen kinds with readable names but 25% on unseen kinds with coined names. Much of the
set is underdetermined from the prompt. A coined or merged group whose members the 24 examples do not show cannot be decoded: when
the gold category's name is coined and only other kinds are filed under it, Opus reads 25%. When the query's kind is among the
examples, Opus reads 91 to 92% for readable and merged names.

The prompt's 24 examples, not the history, are what the model sees. The query's kind is among them in only 39% of items (66% of
seen-kind items).

### 61.2 Lookup first, model for the rest

The no-model cascade (kind lookup, then other users, then the usage prior) reads 65% on fold 0. The kind lookup answers 59% of
items, all correctly, and the usage prior handles the rest. Every model alone is below the cascade (skill -18 to -79). Combined,
with the kind lookup where it answers and the model elsewhere, the result is 69 to 80% (the POI-trained 3Bs 79 to 80, the untrained
14B 78). Where the lookup has nothing, the model is worth 25 to 52% against the prior's 7.5 (the POI-trained 3Bs 48 to 52).

This is the same division of labour section 59 found on REAL-6 with the merchant lookup. Here it is one level up, by the kind of
place, which needs a places database (Overture's category for the query). Real users are less consistent than POI-1's, so a real
kind lookup would be imperfect. The finding is the division of labour, not the 100%.

### 61.3 Scale, transfer and training

- **Scale.** Untrained, 3B, 7B and 14B read 44.6, 47.7 and 52.9.
- **Transfer.** The REAL-6 categorisers do not transfer: all-label 46.5, about the untrained 3B. The rename-augmented one does worse
  (37.9), losing most on readable names (unseen kind, standard: 27 against 53). Training on REAL-6's twelve-category schemes with
  coined words seems to teach it to distrust readable names.
- **Training on POI-1's own users.** This gives 55 to 59: fold 0's users are unseen, but their kinds of places and naming habits are
  not. 200 steps (57.8) do as well as 800 (55.4, intervals overlap). Rename augmentation helps most where it should, on coined
  names: seen kind 32 to 56, unseen kind 12 to 35, the latter above blind Opus's 25 on its sample. Its auto-file coverage at 98% is
  the highest (16.4%), but its realised precision is 92.8.

### 61.4 Examples chosen by kind

`scripts/build_poi1_variants.py kshots` (`poi1_v1_kshots.json`) swaps up to six of the 24 examples for history places of the
query's Overture basic category. This is retrieval by kind, which needs the places database. No training was redone. On fold 0's
items whose kind is in the history, it lifts every reader:

- untrained 3B, 49 to 67;
- untrained 14B, 58 to 75;
- POI-trained 3B with rename augmentation, 65 to 81, and on coined names 56 to 76.

Even so the models do not simply copy. With six examples of the query's kind in the prompt, all under the gold label, the 14B picks
it 88% of the time and the best 3B 92%. A lookup is exact there. So the order is: lookup by merchant, then lookup by kind, then the
model with kind-retrieved examples for kinds the user has never filed.

### 61.5 What the step says

On real places with realistic scheme sizes the models' value is where lookups stop: kinds of place the user has never filed, and
categories whose meaning must be read from a few examples.

- For Q1 (multiple-choice mechanics), the choice among 12 to 20 user categories is limited mostly by what the prompt shows, not by
  model size (3B to 14B: +8 points).
- For Q2 (knowledge injection), a places database helps twice without any training: as a lookup by kind, and as a way to choose
  examples.
- For Q3 (coined names), training with coined words on the right distribution of places is what moves coined names (24 to 56 on
  seen kinds for the 3B; 76 with kind-retrieved examples).

Open:

- all four folds for the POI-trained arms (fold 0 only here);
- the obscure rendering;
- inconsistent users (a kind filed under two categories), so the lookup is no longer exact;
- training with kind-retrieved examples;
- row 66 (facts or skill) uses POI-1's places as the injected database.


## 62. An encoder with one scored [MASK] per category (Laya's layout) matches the 3B on POI-1 (57 against 55 to 59, better top-3) at about a fortieth of the time per item, but trails it on REAL-6 by 15 points without the merchant record and 8 with it: what it lacks is the LLM's knowledge of what merchants are (MODEL-6)

PLAN step 53. MODEL-6 asks whether a small bidirectional encoder can do the categoriser's job in one forward pass. The layout is
Laya's (`references/laya_analysis.md`):

```
[CLS] instruction [SEP] [MASK] category 1 [MASK] category 2 ... [SEP] query, optional record, 24 shots as "statement -> category" [SEP]
```

Each [MASK]'s final hidden state goes through two fresh transformer layers and an MLP to one score, then a softmax over the options.
There is no vocabulary readout: the model is trained only to choose among the options it is given.

The model is ModernBERT-large (395M), either plain or starting from Laya's released checkpoint (`convaiinnovations/laya`,
Apache-2.0, its act head dropped). It is trained with plain cross-entropy on the gold option, not Laya's RL: section 3.2 of the memo
shows that the RL term is cross-entropy plus noise. Training runs across users with fold 0 held out (as the 3B's row 42 folds): 1,500
steps of 16 episodes, options shuffled per episode, 24 random history rows as shots, learning rate 3e-5 for the encoder and 1e-4 for
the head. It is scored on the held-out users' frozen items, shots and option order.

Code is `scripts/exp_encoder_mask.py`; jobs `scripts/modal_jobs/r53.json` (H100, about 5 minutes each); tables
`scripts/encmask_tables.py`; models `models/adapters/encmask_*` (DVC). Two engineering notes:

- Batches are padded to multiples of 128 tokens. A new sequence length per call cost about 50 times the forward pass (850 ms against
  14 ms for one item).
- torch.compile is off, as in Laya's own inference.

The 3B's time comes from its scoring runs: 2.0 minutes for 298 items, about 400 ms per item, with one option-scoring pass per item and
a scorer never tuned for latency. So "about 40 times faster" is the honest claim, not a benchmark.

**Table 62.1: REAL-6, fold 0's held-out users (5 users; calibration leave-users-out within them): the encoder against the 3B; top-1 by corrected group on the right**

| reader | n | top-1 [interval] | top-3 | bits left | auto-file at 98%: coverage (precision) | ms per item, batched / one at a time | in history | labelled seen, not in history | determined by category | split category |
|---|---|---|---|---|---|---|---|---|---|---|
| encoder: Laya checkpoint, untrained | 298 | 9.1 [4.8, 12.8] | 25.8 | 3.83 | 0.0 (nan) | 14 / 20 | 13 | 14 | 7 | 0 |
| encoder: ModernBERT-large, 1,500 steps | 298 | 53.4 [43.1, 63.5] | 72.5 | 2.34 | 15.1 (95.6) | 10 / 17 | 69 | 50 | 42 | 45 |
| encoder: Laya init, 1,500 steps | 298 | 59.4 [50.7, 66.1] | 73.2 | 2.21 | 18.1 (98.1) | 10 / 17 | 72 | 47 | 54 | 48 |
| encoder: Laya init, 4,000 steps | 298 | 59.1 [50.8, 65.2] | 74.2 | 2.19 | 8.4 (76.0) | 9 / 16 | 69 | 50 | 56 | 45 |
| 3B SFT no DB, all-label (H100 bf16) | 298 | 74.2 [63.8, 80.3] | 87.6 | 1.46 | 12.8 (84.2) | - | 77 | 61 | 81 | 52 |
| 3B database episodes (H100 bf16) | 298 | 88.3 [80.7, 94.8] | 96.3 | 0.69 | 31.5 (90.4) | - | 91 | 75 | 97 | 59 |
| encoder + record: ModernBERT-large | 298 | 75.2 [69.0, 82.3] | 90.3 | 1.30 | 36.2 (99.1) | 10 / 22 | 78 | 83 | 72 | 66 |
| encoder + record: Laya init | 298 | 76.5 [70.1, 83.7] | 95.0 | 1.25 | 52.7 (89.2) | 9 / 17 | 81 | 78 | 76 | 59 |
| 3B + record in the prompt (H100 bf16) | 298 | 82.9 [76.8, 90.1] | 96.0 | 0.77 | 20.8 (91.9) | - | 86 | 81 | 92 | 45 |
| 3B + record with category (H100 bf16) | 298 | 85.2 [77.9, 92.2] | 97.0 | 0.61 | 65.8 (98.0) | - | 88 | 72 | 98 | 34 |

**Table 62.2: POI-1, fold 0's held-out users (50 users), readers trained on POI-1's other users**

| reader | n | top-1 [interval] | top-3 | bits left | auto-file at 98%: coverage (precision) | ms per item, batched / one at a time |
|---|---|---|---|---|---|---|
| encoder: ModernBERT-large, 1,500 steps | 507 | 57.6 [53.4, 61.9] | 81.1 | 1.97 | 5.7 (93.1) | 7 / 36 |
| encoder: Laya init, 1,500 steps | 507 | 57.0 [52.4, 62.5] | 79.7 | 1.99 | 7.3 (94.6) | 7 / 40 |
| 3B, 200 steps | 507 | 57.8 [53.4, 62.1] | 76.5 | 2.12 | 10.7 (96.3) | - |
| 3B, 800 steps | 507 | 55.4 [50.4, 60.1] | 73.6 | 2.27 | 4.7 (95.8) | - |
| 3B, 800 steps + rename | 507 | 59.2 [55.5, 63.2] | 78.5 | 1.94 | 16.4 (92.8) | - |

### 62.1 What the encoder does and does not do

**Untrained, Laya's checkpoint is useless here (9%).** Its training (general decisions, 512 tokens) does not carry over to a user's
own categories with shots. Trained for 1,500 steps it reaches 59.4 (plain ModernBERT 53.4; the Laya start is worth a few points,
within the interval). 4,000 steps add nothing.

**On REAL-6 the gap is merchant knowledge.** Without the record the encoder trails the 3B by 15 points (59 against 74) and the
database-episode 3B by 29. The gap is largest where the merchant's standard category decides the answer: 42 to 56% against 81 to 97%.
The 3B knows from pre-training what "GOLD'S GYM" or "BARTELL DRUGS" is; the encoder has to learn it from the training rows. On
merchants in the user's history the two are closer (69 to 72 against 77).

With the record in the input the encoder reaches 75 to 77, 8 points under the 3B with the record (83 to 85). Its top-3 is 95.0
against 96 to 97, and it has the highest auto-file coverage in the table (52.7% of items at a threshold chosen for 98%). But that
threshold realises 89% precision on these five users: the calibration drift of section 59, sharper with five users to fit on.

**On POI-1 the encoder is level with the 3B.** It reaches 57.6 against 55.4 to 59.2, and its top-3 is better (81 against 74 to 79).
POI-1's places carry their kind in the name ("Northside Pediatrics"), and the task is reading the user's 24 examples, not recalling
what an obscure merchant sells. That is exactly what the layout is for, and it does it at 7 ms per item batched on an H100.

### 62.2 What the step says

For Q1 (multiple-choice mechanics), an encoder trained to choose among runtime-defined options, one scored marker each, does the
in-prompt part of the job as well as a 3B decoder: it reads the shots, which GLiClass could not (section 46). What it lacks is
world knowledge about merchants, which is where a decoder's pre-training pays.

For Q2 (knowledge injection), the encoder's weakness is fixed most cheaply by the record (+16 to 22 points). That points to a
production shape: a places or merchants database supplies the facts, and a small encoder reads them with the user's examples.

Open:

- all four folds (fold 0 has five REAL-6 users, so the intervals are wide);
- database episodes for the encoder (does it store merchant facts as the 3B did?);
- calibration objectives (soft targets, a bounded proper score beside the log score, temperature by option count), measured by bits,
  ECE and coverage at a realised 98%;
- pre-training the layout on general multiple-choice data before our task.


## 63. GLiClass and ModernBERT-Instruct through the same training: GLiClass-large works as well as Laya's layout (57 without the record, 74 with it, 58 on POI-1), and with the record it has the steadiest confidence of any encoder; ModernBERT-Instruct's single mask works with the record (73.5) but learns slowly without it (29 at 1,500 steps, 42 at 5,000), and its letters show position bias (MODEL-10)

PLAN step 69, on the owner's request to make GLiNER-type models and ModernBERT-Instruct work (2026-09-26). Both had failed here
before for reasons unrelated to their families:

- section 46's GLiClass was the 151M base, three epochs over 5,365 rows at 1e-5, examples in its `<<EXAMPLE>>` format on half the
  rows;
- section 29's ModernBERT was plain ModernBERT-large taught letters in 800 steps, never the instruction-tuned checkpoint.

Here both go through row 53's data, loop and scorer (`scripts/exp_encoder_mask.py`, `ARCH=gliclass|mbinstruct`): 1,500 steps of 16
episodes, the 24 shots as plain "statement -> category" lines on every episode, options shuffled, fold 0's users held out,
cross-entropy over the options only.

- **GLiClass modern-large v3.0.** The input is `<<LABEL>>name` per category, `<<SEP>>`, then the state. GLiClass's own label-token
  pooling and scorer give one logit per label; padded label slots are masked, which GLiClass does not do itself.
- **ModernBERT-Large-Instruct.** The model card's template, `QUESTION: <state> CHOICES: - A: name ... ANSWER: [unused0] [MASK]`,
  with the MLM head at the mask read over the options' IDs only.
- **Option IDs.** Letters carry prior meaning and a position preference (option-ID selection bias, Zheng et al. ICLR 2024;
  multiple-choice symbol binding, Robinson and Wingate ICLR 2023; the owner raised the same concern). One arm names the options
  `[unused1]`, `[unused2]`, ... instead: tokens with no prior meaning, learned in fine-tuning only.
- **Two follow-ups for ModernBERT-Instruct's weak no-record result:**
  - `MBI_SHOTLAB=1` writes each shot's label with its ID ("-> H: Grendo"), a test of whether binding the label to its ID is the
    obstacle;
  - 5,000 steps, a test of plain under-training.

Following the owner's steer that the 98% auto-file point is arbitrary, the tables now read confidence along the whole coverage
curve (precision on the most confident 25 / 50 / 75% of items, and AURC, the mean error over all coverages; lower is better). They
also add position bias: the total variation between where a model's picks sit in the option list and where the gold answers sit.

Jobs `scripts/modal_jobs/r69*.json`; tables `scripts/encmask_tables.py`.

**Table 63.1: REAL-6, fold 0's held-out users (5 users; calibration leave-users-out within them): the encoder against the 3B; top-1 by corrected group on the right**

| reader | n | top-1 [interval] | top-3 | bits left | precision at 25 / 50 / 75% coverage | AURC | position bias | ms per item, batched / one at a time | in history | labelled seen, not in history | determined by category | split category |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| encoder: Laya checkpoint, untrained | 298 | 9.1 [4.8, 12.8] | 25.8 | 3.83 | 11 / 7 / 9 | 0.888 | 0.65 | 14 / 20 | 13 | 14 | 7 | 0 |
| encoder: ModernBERT-large, 1,500 steps | 298 | 53.4 [43.1, 63.5] | 72.5 | 2.34 | 85 / 79 / 65 | 0.219 | 0.14 | 10 / 17 | 69 | 50 | 42 | 45 |
| encoder: Laya init, 1,500 steps | 298 | 59.4 [50.7, 66.1] | 73.2 | 2.21 | 92 / 81 / 70 | 0.190 | 0.15 | 10 / 17 | 72 | 47 | 54 | 48 |
| encoder: Laya init, 4,000 steps | 298 | 59.1 [50.8, 65.2] | 74.2 | 2.19 | 85 / 80 / 73 | 0.222 | 0.14 | 9 / 16 | 69 | 50 | 56 | 45 |
| GLiClass large, untrained | 298 | 7.7 [2.9, 12.7] | 28.2 | 3.87 | 12 / 10 / 10 | 0.888 | 0.66 | 13 / 27 | 6 | 6 | 10 | 10 |
| GLiClass large, 1,500 steps | 298 | 56.7 [45.5, 66.5] | 72.1 | 2.22 | 92 / 85 / 69 | 0.189 | 0.16 | 8 / 20 | 67 | 47 | 53 | 45 |
| ModernBERT-Instruct, letters, untrained | 298 | 10.4 [4.0, 15.8] | 27.9 | 3.82 | 16 / 10 / 10 | 0.880 | 0.81 | 11 / 14 | 12 | 8 | 11 | 3 |
| ModernBERT-Instruct, letters, 1,500 steps | 298 | 28.9 [23.8, 32.9] | 43.6 | 3.33 | 58 / 46 / 35 | 0.483 | 0.27 | 9 / 16 | 50 | 19 | 20 | 0 |
| ModernBERT-Instruct, unused-token IDs, 1,500 steps | 298 | 13.8 [6.9, 19.0] | 30.5 | 3.73 | 32 / 21 / 17 | 0.724 | 0.65 | 9 / 19 | 27 | 8 | 7 | 3 |
| ModernBERT-Instruct, letters, shot labels carry the letter | 298 | 27.2 [22.1, 30.8] | 43.0 | 3.36 | 57 / 44 / 34 | 0.493 | 0.32 | 9 / 16 | 48 | 22 | 18 | 0 |
| ModernBERT-Instruct, unused IDs, shot labels carry the ID | 298 | 19.1 [14.6, 23.9] | 39.3 | 3.75 | 35 / 26 / 21 | 0.713 | 0.45 | 8 / 18 | 36 | 6 | 14 | 0 |
| ModernBERT-Instruct, letters, 5,000 steps | 298 | 41.9 [36.3, 48.3] | 58.7 | 2.83 | 74 / 65 / 51 | 0.356 | 0.20 | 10 / 18 | 63 | 31 | 35 | 3 |
| 3B SFT no DB, all-label (H100 bf16) | 298 | 74.2 [63.8, 80.3] | 87.6 | 1.46 | 91 / 93 / 85 | 0.142 | 0.13 | - | 77 | 61 | 81 | 52 |
| 3B database episodes (H100 bf16) | 298 | 88.3 [80.7, 94.8] | 96.3 | 0.69 | 93 / 96 / 95 | 0.073 | 0.07 | - | 91 | 75 | 97 | 59 |
| encoder + record: ModernBERT-large | 298 | 75.2 [69.0, 82.3] | 90.3 | 1.30 | 99 / 90 / 88 | 0.092 | 0.09 | 10 / 22 | 78 | 83 | 72 | 66 |
| encoder + record: Laya init | 298 | 76.5 [70.1, 83.7] | 95.0 | 1.25 | 93 / 88 / 87 | 0.102 | 0.11 | 9 / 17 | 81 | 78 | 76 | 59 |
| GLiClass large + record | 298 | 74.2 [69.8, 82.3] | 85.2 | 1.26 | 100 / 97 / 90 | 0.062 | 0.13 | 6 / 14 | 75 | 78 | 73 | 66 |
| ModernBERT-Instruct + record | 298 | 73.5 [68.9, 79.8] | 90.6 | 1.27 | 95 / 94 / 87 | 0.087 | 0.13 | 12 / 19 | 76 | 78 | 72 | 62 |
| 3B + record in the prompt (H100 bf16) | 298 | 82.9 [76.8, 90.1] | 96.0 | 0.77 | 96 / 97 / 94 | 0.060 | 0.12 | - | 86 | 81 | 92 | 45 |
| 3B + record with category (H100 bf16) | 298 | 85.2 [77.9, 92.2] | 97.0 | 0.61 | 99 / 99 / 97 | 0.031 | 0.12 | - | 88 | 72 | 98 | 34 |

**Table 63.2: POI-1, fold 0's held-out users (50 users), readers trained on POI-1's other users**

| reader | n | top-1 [interval] | top-3 | bits left | precision at 25 / 50 / 75% coverage | AURC | position bias | ms per item, batched / one at a time |
|---|---|---|---|---|---|---|---|---|
| encoder: ModernBERT-large, 1,500 steps | 507 | 57.6 [53.4, 61.9] | 81.1 | 1.97 | 91 / 81 / 68 | 0.210 | 0.10 | 7 / 36 |
| encoder: Laya init, 1,500 steps | 507 | 57.0 [52.4, 62.5] | 79.7 | 1.99 | 94 / 80 / 70 | 0.195 | 0.06 | 7 / 40 |
| GLiClass large, 1,500 steps | 507 | 58.4 [54.7, 62.5] | 79.9 | 1.96 | 90 / 79 / 71 | 0.194 | 0.09 | 6 / 26 |
| 3B, 200 steps | 507 | 57.8 [53.4, 62.1] | 76.5 | 2.12 | 93 / 78 / 69 | 0.194 | 0.10 | - |
| 3B, 800 steps | 507 | 55.4 [50.4, 60.1] | 73.6 | 2.27 | 89 / 79 / 66 | 0.227 | 0.08 | - |
| 3B, 800 steps + rename | 507 | 59.2 [55.5, 63.2] | 78.5 | 1.94 | 98 / 83 / 72 | 0.172 | 0.08 | - |

### 63.1 What makes an encoder work here

**Every encoder is useless untrained** (GLiClass 7.7, ModernBERT-Instruct 10.4, Laya 9.1). The task, filing into one person's own
categories from their examples, is not in any of their training mixtures.

**Trained the same way, GLiClass-large and Laya's layout are equivalent.**

- Without the record: 56.7 and 59.4.
- With the record: 74.2 and 76.5.
- On POI-1: 58.4 and 57.0.

Both put a scored position next to each option's name. GLiClass's confidence is the steadiest of the encoders with the record:
100% precision on its most confident quarter, 97% on half, AURC 0.062, level with the 3B plus record (0.060).

So the earlier GLiClass failure (section 46) was the training, not the family: the base model, a fifth of the episodes, and shots
on half of them in a format the checkpoint had no token for.

**ModernBERT-Instruct's single mask works when the answer is a meaning match.** With the record it reads 73.5, with the best
encoder top-3 (90.6). Without the record it reaches 28.9 at 1,500 steps and 41.9 at 5,000, still climbing, far below the
per-option designs at 1,500 steps.

The binding hypothesis is rejected: labelling the shots with their letter changes nothing (27.2). The reading that fits is where
the decision is made. The per-option designs give every category its own position, which can attend to that category's name
and to the shots filed under it. The single mask must find the similar shot, carry its label, and map it to a letter, all
through one position and the vocabulary head. With the record, the record-to-name match is a direct semantic comparison, which
the instruction tuning already does.

**Letters do carry bias; unused IDs are worse.**

- ModernBERT-Instruct with letters has the highest position bias among the trained encoders (0.27 to 0.32, against 0.14 to 0.16
  for the per-option designs).
- Unused-token IDs, which carry no prior, fail (13.8, bias 0.65). 1,500 steps do not teach 26 fresh embeddings to act as pointers,
  so the model falls back on position.

The per-option designs avoid the question: no ID is ever predicted.

### 63.2 What the step says

The best encoder design for this task scores each option at its own position: Laya's layout or GLiClass, either one.

- **On POI-1** both match the 3B decoder (57 to 58 against 55 to 59) at 6 to 10 ms per item batched.
- **On REAL-6** the gap to the 3B stays where section 62 put it, merchant knowledge (the "determined by category" group: 53 to 56
  against 81 to 97). With the record the encoders reach 74 to 77 against 85.

**Hypotheses for the remaining gap:**

- **Merchant facts.** Database episodes, which lifted the 3B from 74 to 88 on fold 0, would do the same for an encoder. Test: the
  encoder with the 3B's database episodes.
- **Model size.** A larger per-option encoder (GLiClass-large is 400M; no larger modern encoder is public) or distillation from
  the 3B's distributions (soft targets, row 68) closes part of it.
- **ModernBERT-Instruct's single mask** reaches the per-option designs only with far more training; it is not the design to
  pursue.

Models `models/adapters/enc{gli,mbi}_*` (DVC).


## 64. Results as a share of the maximum achievable score, and the scale test: on REAL-6 the best decoders sit at 95 to 96% of the ceiling (94.0 overall) and 7B / 14B add nothing over the 3B; on POI-1 a 14B gains one point over the 3B (60.2 against 59.2) where choosing the shots by kind gains nine; the limit on POI-1 is what the prompt shows, not model size (MODEL-11)

PLAN step 70, and the owner's request (2026-09-26) to express results as a percentage of the maximum theoretical score.

**The ceilings** (`ai_experiments.ceiling`, `scripts/ceiling_tables.py`). Each item gets the best probability of a correct top-1 that
any reader could have from what is observable; a set's ceiling is the mean over its items, so it applies to any subset, and "% of
ceiling" is top-1 over the ceiling on the same items.

- **REAL-6, exact.** An ideal reader knows the merchant's true standard category and the user's exact scheme, but not the
  per-merchant coin flip of a split category: 1 in history, else 1 / (the number of the user's categories holding the merchant's
  standard category).

  The coin-flip assumption was checked. On all 103 split items the four-fold runs score 40 to 47% (no run beats 50; the
  fold-0-only shares above 100 in Table 64.2's split column are 29 items of noise).
- **Label induction, exact per condition.** 1 when a readable example of the gold word is in the prompt; else 1 / (the coined words
  no readable example explains).
- **POI-1, exact only on seen kinds.** 100 when the place's Overture basic category is in the user's history, since a perfect
  places database plus the history decides it. A never-filed kind has no computable ceiling. A rule that ignored the category
  names (a guess among the groups sharing the item's top level) was beaten by every trained reader, because a readable name
  ("Health care") tells a reader where a new clinic goes. So unseen kinds are read as a bracket: the best reader below, 100 above.

**The scale test** (row 70) trains the three best 3B recipes at Qwen2.5-7B and 14B, fold 0, bf16 on Modal (`LLM_BASE` in
`exp_categoriser.py`, `scripts/modal_jobs/r70.json`; MICRO 8 for the 14B):

- database episodes with all-label loss, 200 steps;
- the record stating the category, 200 steps;
- POI-1 all-label with rename augmentation, 800 steps.

**Table 64.1: REAL-6's ceiling by corrected group (an ideal reader: the merchant's true category and the user's exact scheme, not the coin flip of a split)**

| group | items (all users) | ceiling, all users | items (fold 0) | ceiling, fold 0 |
|---|---|---|---|---|
| in history | 444 | 100.0 | 106 | 100.0 |
| labelled seen, not in history | 115 | 90.9 | 36 | 88.9 |
| determined by category | 509 | 99.1 | 123 | 99.2 |
| split category | 103 | 50.0 | 29 | 50.0 |
| other | 8 | 50.0 | 4 | 50.0 |
| all | 1179 | 94.0 | 298 | 92.8 |

**Table 64.2: REAL-6, fold 0's held-out users: top-1 as a share of the ceiling on the same items**

| run | n | top-1 | ceiling on the same items | % of ceiling | headroom left (points) | in history: % of ceiling | labelled seen, not in history: % of ceiling | determined by category: % of ceiling | split category: % of ceiling |
|---|---|---|---|---|---|---|---|---|---|
| 3B SFT no DB, all-label | 298 | 74.2 | 92.8 | 79.9 | 18.6 | 77 | 69 | 82 | 103 |
| 3B database episodes | 298 | 88.3 | 92.8 | 95.1 | 4.5 | 91 | 84 | 98 | 117 |
| 7B database episodes | 298 | 87.2 | 92.8 | 94.0 | 5.5 | 90 | 97 | 98 | 69 |
| 14B database episodes | 298 | 88.9 | 92.8 | 95.8 | 3.9 | 91 | 94 | 97 | 124 |
| 3B + record with category | 298 | 85.2 | 92.8 | 91.9 | 7.6 | 88 | 81 | 99 | 69 |
| 7B + record with category | 298 | 82.2 | 92.8 | 88.6 | 10.6 | 86 | 91 | 89 | 90 |
| 14B + record with category | 298 | 88.6 | 92.8 | 95.5 | 4.2 | 93 | 94 | 97 | 90 |
| encoder (Laya layout), no record | 298 | 59.1 | 92.8 | 63.7 | 33.7 | 72 | 53 | 54 | 97 |
| GLiClass-large + record | 298 | 74.2 | 92.8 | 79.9 | 18.6 | 75 | 88 | 74 | 131 |
| encoder (Laya layout) + record | 298 | 76.2 | 92.8 | 82.1 | 16.6 | 80 | 88 | 76 | 117 |

**Table 64.3: POI-1, fold 0. Seen kinds (the place's Overture basic category is in the user's history) have an exact ceiling of 100 (a perfect places database plus the history); unseen kinds have none, so they are read as a bracket: the best reader below, 100 above**

(fold 0: 300 seen-kind items, 207 unseen-kind; blind Opus 5.5 read 56 on its sample)

| run | seen kind: top-1 = % of ceiling | seen kind, coined name | unseen kind: top-1 | all items: top-1 |
|---|---|---|---|---|
| untrained 14B | 57.7 | 37.2 | 45.9 | 52.9 |
| 3B, all-label + rename | 65.3 | 56.4 | 50.2 | 59.2 |
| 7B, all-label + rename | 63.0 | 57.7 | 51.2 | 58.2 |
| 14B, all-label + rename | 66.0 | 66.7 | 51.7 | 60.2 |
| GLiClass-large | 65.3 | 53.8 | 48.3 | 58.4 |
| 3B + kind-retrieved shots | 81.0 | 75.6 | 50.7 | 68.6 |

(unseen kinds: the ceiling lies between 51.7, the best reader here, and 100)

**Table 64.4 label_induction_v2: ceiling and % of ceiling by condition**

| condition | ceiling | 3B: top-1 / % of ceiling | 14B: top-1 / % of ceiling | 3B rename-trained: top-1 / % of ceiling |
|---|---|---|---|---|
| base | 100 | 46 / 46 | 77 / 77 | 77 / 77 |
| n_gold=0 | 25 | 5 / 19 | 12 / 48 | 12 / 49 |
| n_gold=1 | 100 | 34 / 34 | 66 / 66 | 71 / 71 |
| n_gold=4 | 100 | 55 / 55 | 83 / 83 | 84 / 84 |
| n_gold=8 | 100 | 66 / 66 | 87 / 87 | 90 / 90 |
| kind=same_kind | 100 | 79 / 79 | 93 / 93 | 94 / 94 |
| kind=opaque | 25 | 4 / 16 | 5 / 21 | 10 / 40 |
| n_coined=1 | 100 | 47 / 47 | 78 / 78 | 78 / 78 |
| n_coined=6 | 100 | 40 / 40 | 74 / 74 | 79 / 79 |
| decoy | 100 | 30 / 30 | 58 / 58 | 60 / 60 |
| decoy, n_gold=0 | 25 | 1 / 5 | 3 / 13 | 4 / 15 |
| control: standard name | 100 | 92 / 92 | 96 / 96 | 94 / 94 |

### 64.1 Where each set stands

**REAL-6 is effectively solved at 3B.** The best runs sit at 95 to 96% of the ceiling:

- 3B database episodes: 88.3, 95.1%;
- 14B database episodes: 88.9, 95.8%;
- 14B record with category: 88.6, 95.5%.

That leaves about 4 points. Most of it is on merchants in the user's history (91% of ceiling): the lookup is exact there (section
48), so a lookup in front of the model (section 59's cascade) takes those points. The 7B is not better than the 3B (87.2, 82.2).
On five users the differences between 3B, 7B and 14B are within noise.

**POI-1 is limited by what the prompt shows, not by model size.**

- On seen kinds, where the ceiling is 100, the trained 3B, 7B and 14B read 63 to 66%, and the 14B gains one point overall (60.2
  against 59.2).
- Choosing up to six of the 24 shots by the place's kind lifts the 3B to 81% of ceiling on seen kinds (68.6 overall) with no
  retraining. A kind lookup reaches 100% there by construction.
- The 14B's one clear gain is coined names on seen kinds (57 to 67): larger models read a coined word from its examples better,
  as section 60 found.
- On unseen kinds every trained reader is at 48 to 52. That is the bracket's floor; blind Opus read 25 to 70 across unseen-kind
  levels.

**Label induction** is where the gap to the ceiling is widest for the models. At two gold examples the best reach 77% of a ceiling
Opus reaches, and with a decoy they fall to 58 to 60% of it (section 60).

### 64.2 Hypotheses for what is left

1. **REAL-6's last 4 points are the lookup.** A model given the user's own label for a merchant in their history (the merchant
   lookup in front, or retrieved shots of the same merchant) should reach about 99% of ceiling on the in-history group. Test: the
   cascade lookup, then the model, read against the ceiling.
2. **POI-1's seen kinds are a retrieval problem.** Kind-retrieved shots reach 81% of ceiling, a lookup 100. Training with
   kind-retrieved shots (the model learns to trust the same-kind examples), or putting the kind in the prompt as a record, should
   close most of the remaining 19 points. Test: train with the `poi1_v1_kshots` layout and with an Overture record.
3. **POI-1's unseen kinds need the category names' meaning.** 14B does not beat 3B there (51.7 against 50.2), so it is not raw
   capacity. Candidates:
   - a description of each user category from its filed places ("Gavir: counselling, psychology"), put in the prompt;
   - the examples' kinds as records.
4. **Label induction's gap is the decoy and one-example cases** (section 60): train with decoys and empty categories.

Models from rows 69 and 70 are in DVC (`models/adapters/categoriser_Qwen2.5-{7B,14B}-Instruct_*`, `enc{gli,mbi}_*`).
