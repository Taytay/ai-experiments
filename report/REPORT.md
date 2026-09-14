# Fine-tuning LLMs and embedding models for merchant knowledge: frameworks, vocabulary, and knowledge injection

Date: 2026-09-12 to 2026-09-14. Hardware: RTX 3090 (24 GB), driver 591.86, Windows 11, torch 2.11 + cu128.
Sections 4 and 6 ran with transformers + torch only (Smart App Control blocked triton at the time; see NOTES.md); sections 7 and 8 use unsloth.
Supporting docs: [frameworks.md](frameworks.md), [lit_review.md](lit_review.md). Code: `../experiments/`. Raw numbers: `../results/`. Every run is tracked with config + git commit in `../evals/` (see `../evals/LEADERBOARD.md`).

## 1. Executive summary

- **Frameworks.** For one 24 GB GPU, Unsloth is the consensus choice (fastest, lowest VRAM, native Windows, now covers embedding models via `FastSentenceTransformer`). Axolotl is the pick for multi-GPU nodes with YAML-driven reproducibility. TRL is the substrate both wrap and the right layer if you need a custom loss. torchtune is unmaintained since July 2025; do not start on it. For RL at scale, verl. For embedding models specifically, the trainer is sentence-transformers' `SentenceTransformerTrainer` whether or not Unsloth is wrapping it.
- **New vocabulary.** Almost never worth it for merchant names. Subword tokenization already handles them; expansion requires continued pretraining and can hurt at small token budgets. If you must add tokens, initialize inside the existing embedding distribution (mean-of-subwords or Hewitt's N(mu, Sigma) sampling), never random, and train afterwards. Our experiments went further: added merchant tokens actively **hurt** both models. For the embedding model they collapsed transfer to unseen bank-statement strings from 70.8% to 23-26% (section 4.3.1); for the LLM they cut knowledge extraction from 46.7% to 31.7% at identical data and steps. Post-hoc aliasing of an uppercase token onto the trained mixed-case embedding did not rescue the bank format either. Fix the strings with normalization, not the tokenizer.
- **User-invented labels and analogy (section 6).** Prompts like "Timmy labeled his Blaxorc 'FooFoo'... how will he label his Radsup?" are a scale phenomenon: with the facts in context a 3B model reaches 49 to 52% on a 3-way task (chance 33) and 0.5B never leaves chance. Knowledge injected into weights by LoRA was fully recalled (100% in the trained format) but did not power that in-context analogy, and fine-tuning eroded the ability. For an embedding model, defining a user's label as the centroid of their labeled examples gave 85% (type) and 63% (a latent attribute the labels never name) with three examples and no retraining. Recommendation: prototypes for user categories, retrieval-in-context for LLM analogy, a 3B to 8B model.
- **Teach the task as the injection (section 8).** A six-arm sweep on Qwen2.5-3B settled the inject-then-task question. Symbol-tuning episodes generated from the database (few-shot items with fresh random labels and a varied grouping attribute) raise in-context label induction by about 30 points on every test, including a partition and species never trained on, and lift few-shot classification with random labels on public datasets from 60 to 76 where declarative text alone lowered it to 51. Interleaving those episodes with the knowledge text in one run gives 100% recall *and* the Timmy task from the weights at 59% (base 36, knowledge-only 40, chance 33), with transfer to the held-out partition (54%). Sequential staging matches the induction number but loses 3 points of recall, 21 of yes/no manipulation and 9 of pairwise reasoning. A 15% generic replay slice keeps the skill portable (78 vs 71 on public data) and holds perplexity at 15 instead of 20. On a universe where 70% of names carry a type suffix, the interleaved model types never-seen names from their suffix at 94% (chance 12.5) while neutral names stay at chance: the drug-stem mechanism, measured.
- **Performance (section 7).** On this RTX 3090, unsloth LoRA trains 0.5B at 17.7k tokens/s (2.2x plain transformers, 3x less memory), 3B at 2.7k tokens/s (69% MFU) and 7B QLoRA at 1.3k tokens/s. Small models on short facts are overhead-bound (13 to 17% MFU) and want sequence packing more than a faster GPU; 3B and up are compute-bound and scale with TFLOPS. The card handles ~1.5B for full fine-tuning, ~8B for LoRA, ~30B for QLoRA. A 6-attribute entity costs ~1,800 training tokens with a diverse recipe, so a 10k-entity database trains in 17 minutes on 0.5B or about 2 to 4 hours on 3B to 7B. Watch for Windows WDDM system-memory fallback near 24 GB, which slows training ~3x instead of failing.
- **Knowledge injection.** Dumping the merchant database as one sentence per store into the model does not produce usable knowledge, even when memorized. Paraphrase and QA augmentation of the same facts (the Physics-of-LMs / EntiGraph recipe) is what makes knowledge extractable in new task formats. At a sane learning rate (1e-5 here) augmented full fine-tuning doubled category-inference accuracy over the raw dump (41.7% vs 21.7%, retrieval ceiling 69%) with little forgetting; at 5x that rate it destroyed general ability (perplexity 16 to 800) and LoRA or WiSE-FT weight averaging were the rescue. Raw statement strings defeated every method including retrieval at this model size, so normalize merchant strings before the model sees them. Retrieval (fact in context) remains the strongest and cheapest baseline; the right production design is RAG over the merchant DB plus augmented fine-tuning for the head of the distribution, with RAFT-style training so the model uses retrieved records well.

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
| **incontext** (fact in prompt, RAG upper bound) | **69.2** | 14.2 | **100.0** | **100.0** | 16.15 |
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
4. **For parametric knowledge that transfers**, build an augmented corpus: 10+ paraphrases per merchant, QA in both directions, statement-style mentions, product-to-category chains. Full fine-tune (or high-rank LoRA on all modules) at low LR with replay of general data; average with the base checkpoint (WiSE-FT) if perplexity drift matters.
5. **Train the model to use retrieval** (RAFT): fine-tune on prompts containing retrieved records plus distractors, targets that cite the record. This is what makes knowledge improve *other* tasks (budget Q&A, anomaly explanations, merchant normalization) rather than just the classifier.
6. **Evaluate on held-out merchants and formats**, and track hallucination on merchants the model was never taught.

## 6. Fictional universe: user-invented labels and analogy over injected knowledge

Added 2026-09-13. Question: teach a model a completely custom taxonomy (a Pokemon-style creature universe with fictional type names), then answer prompts like *"Timmy labeled his Blaxorc 'FooFoo' and his FrodRock 'blammo'. How will he label his Radsup?"* where both the entities and the labels are novel. Code: `experiments/universe.py`, `exp_universe_ladder.py`, `exp_universe_embed.py`. Tracked runs: `evals/LEADERBOARD.md`, experiments `universe_ladder` and `universe_embed`.

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

Conditions: base, base + field-guide entries in context (RAG ceiling), LoRA r=64 on all linear layers (600 steps, batch 16, bf16), LoRA + context.

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

1. **Define user labels by examples, not names.** The prototype approach (a label = the centroid of the transactions a user put under it) handles synonyms, typos, and idiosyncratic categories with zero retraining and already reaches 63 to 85% on 3-way tasks at MiniLM scale. Use a stronger encoder (bge, EmbeddingGemma, Qwen3-Embedding) trained on transaction-to-merchant-record pairs and this should be the production classifier for personal categories.
2. **For LLM-side analogy prompts, keep the knowledge in context.** Retrieve the merchant records for the merchants named in the prompt; the model then does the mapping. Use a 3B+ model (7B preferable) since the ability to follow arbitrary label mappings from few examples is absent at 0.5B.
3. **Parametric injection is for coverage, not for reasoning, unless the task is part of the injection.** Fine-tune so the model recognizes merchants when retrieval misses, at a low learning rate with replay, and verify manipulation-format accuracy (yes/no, comparisons) rather than bare recall. Section 8 shows that mixing few-shot episodes into the same run changes this picture: the injected knowledge then does power in-weights analogy (59% vs 40% here).
4. **Inferring the user's grouping rule is the open problem.** Habitat-tracking labels were the hardest case for both model types. A practical fix is to compute prototypes over several attribute-specific embeddings (type, merchant, amount band, time) and pick the space in which the user's examples cluster most tightly.

## 7. Performance: throughput, memory, bottlenecks, scaling

Measured 2026-09-13 with `experiments/bench_throughput.py` (each config in its own process, 12 timed steps after 3 warm-up steps, synthetic token batches so tokenization is excluded). GPU: RTX 3090, 24 GB, driver 591.86, WDDM. Dense bf16 tensor-core peak assumed 71 TFLOPS for MFU; the 7B forward pass reached 60 TFLOPS, which confirms that peak is the right reference. Tracked as `bench_throughput`; raw numbers in `results/bench_throughput.json`.

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

Date: 2026-09-13/14. Model: Qwen2.5-3B, unsloth LoRA r=64 alpha=128 on all linear layers, lr 1e-4, 800 steps of 16 sequences (micro-batch 8, accumulation 2), max length 768, seed 0. Code: `experiments/exp_curriculum.py`, `experiments/icl_suite.py`, episode and probe generators in `experiments/universe.py`. Tracker experiment `curriculum_v2`; raw numbers in `results/curriculum_Qwen2.5-3B_<arm>.json`; table printer `experiments/curriculum_summary.py`.

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
2. **Interleaving gives both.** Arm C recalls 100% of facts in the trained format, does the Timmy task *from its own weights* at 59.4% (base 35.6, knowledge-only 40.0), reaches 52.5% at k=4 against 25 chance, and transfers the skill to the held-out weakness partition without context (53.8, base 32.5). With context it keeps most of arm B's gain (76.2). On the public ICL suite it scores 78.1 with symbol labels and 88.0 with natural labels, both above base. General perplexity rises from 8.5 to 14.9, which is within the user's stated tolerance.
3. **Sequential matches interleaved on induction but erodes the knowledge.** Arm D lands at 61.2 on the Timmy task without context, the same as C and Cn, and posts the best public-suite score (81.2). But its recall slips to 96.9, yes/no membership falls to 63.8 (C: 85.0), pairwise same-type to 65.0 (C: 73.8), and induction with real type names as labels to 58.1 (C: 71.2). Even recall *with the facts in context* drops to 89.4 while every other arm is at 100: the second phase pulled the model's answer format away from the first. Two stages means the last stage wins; interleaving holds both in place. This settles the user's question in favour of one mixed run.
4. **Replay buys portability, not the effect itself.** Without replay (Cn) the universe metrics are the same or slightly better (Timmy 60.6, weakness 60.6) but the public-suite score falls from 78.1 to 70.8 and perplexity rises from 14.9 to 20.4. Fifteen percent generic episodes is what keeps the induction skill general rather than universe-shaped.
5. **Habitat induction stays hard for every arm** (best 45.0 without context, D). Habitat is an independently random attribute with six values, so the task needs both per-species recall of a second attribute and the induction step. Type and weakness are eight-way and shared across 17 species each, which gives many more training exposures per value. This is the same "latent grouping" difficulty seen in section 6.3 for the embedding model.

### 8.4 Morphology: a name stem becomes a feature

On the morphology universe 70% of species names end in one of eight suffixes tied to their type (for example `-orc` for Voltrix), the rest use neutral suffixes; held-out species and never-trained probe names follow the same rule. Two observations before training: the base model already exploits shared suffixes in context (Timmy k=4 38.1 vs 26.9 on the plain universe; weakness 50.0 vs 32.5), so the suffix is a usable surface cue for a 3B model with no fine-tuning at all.

After interleaved training on that universe (arm E), a **never-trained name that ends in its type's suffix is classified correctly 93.8% of the time** in the trained answer format, against 12.5% chance. Never-trained names with a neutral suffix stay at chance (14.6%), so the model is reading the stem, not guessing better in general. The same probes on arm C, trained on the plain universe where those suffixes are spread randomly across types, score at or below chance (4.2% and 6.2%): the suffix only becomes a feature when the training data makes it predictive, exactly the condition drug stems and processor prefixes satisfy in real data. The bare-format probe rows in table 8.3 sit at chance for every arm for the format reason noted in section 6.4, which is why the trained-format rows were added.

The effect shows up in the induction tasks as well. Arm E does the Timmy task from its weights at 75.0% (C: 59.4) and at 76.2% for k=4 (C: 52.5), and induction over held-out species without any context rises to 46.9% (C: 31.2): with a 70% reliable stem the model can place an unseen creature by its name alone, which is the "new drug name lands near its class" behaviour the user asked for. Recall (100%), public-suite score (81.8) and perplexity (15.6) are unchanged relative to C, so the morphology signal costs nothing elsewhere.

### 8.5 Cost

Each trained arm took 21 to 25 minutes of training at 837 to 1,171 tokens/s and 8.7 GiB peak (episodes average roughly 250 tokens, so the run is far from the card's limit), plus about 7 minutes of evaluation over 4,400 scored items. Arm A on short knowledge texts ran at 498 tokens/s, launch-overhead-bound as in section 7.2; packing would roughly triple that. The whole eight-arm sweep was about three and a half hours of GPU time including two restarts caused by evaluation-time memory handling (fused loss refusing to run with a full allocator cache; full-vocabulary float logits for long prompts), both fixed in the scorer.

### 8.6 Recommendation, revised

Train one run whose batches mix augmented knowledge text (about half), symbol-tuning episodes generated from the same database with random labels and varied grouping attributes (about 40%), and 10 to 20% generic few-shot replay. Do not stage it. Hold out one attribute and a slice of entities from the episodes and use them, plus a public few-shot suite with random labels, as the regression metric instead of perplexity. If the entity names carry any morphology (drug stems, retailer name variants, processor prefixes), make sure the same rendering variety appears in both the knowledge text and the episodes, because that is what turns the shared subword pieces into a feature the model uses without context. Next steps that this sweep did not run: the same arms on 7B, a real-world replicate with drug names and ATC classes, and the merchant database with statement-style noisy renderings in place of the creature universe.
