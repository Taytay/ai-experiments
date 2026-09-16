# Fine-tuning LLMs and embedding models for merchant knowledge: frameworks, vocabulary, and knowledge injection

Date: 2026-09-12 to 2026-09-14. Hardware: RTX 3090 (24 GB), driver 591.86, Windows 11, torch 2.11 + cu128.
Sections 4 and 6 ran with transformers + torch only (Smart App Control blocked triton at the time; see NOTES.md); sections 7 and 8 use unsloth.
Supporting docs: [frameworks.md](frameworks.md), [lit_review.md](lit_review.md). Code: `../scripts/`. Raw numbers: `../results/`. Every run is tracked with config + git commit in `../evals/` (see `../evals/LEADERBOARD.md`).

## 1. Executive summary

- **Frameworks.** For one 24 GB GPU, Unsloth is the consensus choice (fastest, lowest VRAM, native Windows, now covers embedding models via `FastSentenceTransformer`). Axolotl is the pick for multi-GPU nodes with YAML-driven reproducibility. TRL is the substrate both wrap and the right layer if you need a custom loss. torchtune is unmaintained since July 2025; do not start on it. For RL at scale, verl. For embedding models specifically, the trainer is sentence-transformers' `SentenceTransformerTrainer` whether or not Unsloth is wrapping it.
- **New vocabulary.** Almost never worth it for merchant names. Subword tokenization already handles them; expansion requires continued pretraining and can hurt at small token budgets. If you must add tokens, initialize inside the existing embedding distribution (mean-of-subwords or Hewitt's N(mu, Sigma) sampling), never random, and train afterwards. Our experiments went further: added merchant tokens actively **hurt** both models. For the embedding model they collapsed transfer to unseen bank-statement strings from 70.8% to 23-26% (section 4.3.1); for the LLM they cut knowledge extraction from 46.7% to 31.7% at identical data and steps. Post-hoc aliasing of an uppercase token onto the trained mixed-case embedding did not rescue the bank format either. Fix the strings with normalization, not the tokenizer.
- **User-invented labels and analogy (section 6).** Prompts like "Timmy labeled his Blaxorc 'FooFoo'... how will he label his Radsup?" are a scale phenomenon: with the facts in context a 3B model reaches 49 to 52% on a 3-way task (chance 33) and 0.5B never leaves chance. Knowledge injected into weights by LoRA was fully recalled (100% in the trained format) but did not power that in-context analogy, and fine-tuning eroded the ability. For an embedding model, defining a user's label as the centroid of their labeled examples gave 85% (type) and 63% (a latent attribute the labels never name) with three examples and no retraining. Recommendation: prototypes for user categories, retrieval-in-context for LLM analogy, a 3B to 8B model.
- **Teach the task as the injection (section 8).** A six-arm sweep on Qwen2.5-3B settled the inject-then-task question. Symbol-tuning episodes generated from the database (few-shot items with fresh random labels and a varied grouping attribute) raise in-context label induction by about 30 points on every test, including a partition and species never trained on, and lift few-shot classification with random labels on public datasets from 60 to 76 where declarative text alone lowered it to 51. Interleaving those episodes with the knowledge text in one run gives 100% recall *and* the Timmy task from the weights at 59% (base 36, knowledge-only 40, chance 33), with transfer to the weakness attribute (54%), which training episodes never group by; weakness is a fixed function of type in this universe, though, so this is a type replicate and not evidence of latent-partition transfer (see 6.2 and DATA-1). Sequential staging matches the induction number but loses 3 points of recall, 21 of yes/no manipulation and 9 of pairwise reasoning. A 15% generic replay slice keeps the skill portable (78 vs 71 on public data) and holds perplexity at 15 instead of 20. On a universe where 70% of names carry a type suffix, the interleaved model types never-seen names from their suffix at 94% (chance 12.5) while neutral names stay at chance: the drug-stem mechanism, measured.
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
4. **For parametric knowledge that transfers**, build an augmented corpus: 10+ paraphrases per merchant, QA in both directions, statement-style mentions, product-to-category chains. Full fine-tune (or high-rank LoRA on all modules) at low LR with replay of general data; average with the base checkpoint (WiSE-FT) if perplexity drift matters.
5. **Train the model to use retrieval** (RAFT): fine-tune on prompts containing retrieved records plus distractors, targets that cite the record. This is what makes knowledge improve *other* tasks (budget Q&A, anomaly explanations, merchant normalization) rather than just the classifier.
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

1. **Define user labels by examples, not names.** The prototype approach (a label = the centroid of the transactions a user put under it) handles synonyms, typos, and idiosyncratic categories with zero retraining and already reaches 63 to 85% on 3-way tasks at MiniLM scale. Use a stronger encoder (bge, EmbeddingGemma, Qwen3-Embedding) trained on transaction-to-merchant-record pairs and this should be the production classifier for personal categories.
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
2. **Interleaving gives both.** Arm C recalls 100% of facts in the trained format, does the Timmy task *from its own weights* at 59.4% (base 35.6, knowledge-only 40.0), reaches 52.5% at k=4 against 25 chance, and transfers the skill to the weakness attribute, which no episode groups by, without context (53.8, base 32.5; a type replicate in this universe, see 6.2). With context it keeps most of arm B's gain (76.2). On the public ICL suite it scores 78.1 with symbol labels and 88.0 with natural labels, both above base. General perplexity rises from 8.5 to 14.9, which is within the user's stated tolerance.
3. **Sequential matches interleaved on induction but erodes the knowledge** (with the schedule confound described in 8.1). Arm D lands at 61.2 on the Timmy task without context, the same as C and Cn, and posts the best public-suite score (81.2). But its recall slips to 96.9, yes/no membership falls to 63.8 (C: 85.0), pairwise same-type to 65.0 (C: 73.8), and induction with real type names as labels to 58.1 (C: 71.2). Even recall *with the facts in context* drops to 89.4 while every other arm is at 100: the second phase pulled the model's answer format away from the first. Two stages means the last stage wins; interleaving holds both in place. This settles the user's question in favour of one mixed run.
4. **Replay buys portability, not the effect itself.** Without replay (Cn) the universe metrics are the same or slightly better (Timmy 60.6, weakness 60.6) but the public-suite score falls from 78.1 to 70.8 and perplexity rises from 14.9 to 20.4. Fifteen percent generic episodes is what keeps the induction skill general rather than universe-shaped.
5. **Habitat induction stays hard for every arm** (best 45.0 without context, D). Habitat is an independently random attribute with six values, so the task needs both per-species recall of a second attribute and the induction step. Type and weakness are eight-way and shared across 17 species each, which gives many more training exposures per value. This is the same "latent grouping" difficulty seen in section 6.3 for the embedding model.

### 8.4 Morphology: a name stem becomes a feature

On the morphology universe 70% of species names end in one of eight suffixes tied to their type (for example `-orc` for Voltrix), the rest use neutral suffixes; held-out species and never-trained probe names follow the same rule. Two observations before training: the base model already exploits shared suffixes in context (Timmy k=4 38.1 vs 26.9 on the plain universe; weakness 50.0 vs 32.5), so the suffix is a usable surface cue for a 3B model with no fine-tuning at all.

After interleaved training on that universe (arm E), a **never-trained name that ends in its type's suffix is classified correctly 93.8% of the time** in the trained answer format, against 12.5% chance. Never-trained names with a neutral suffix stay at chance (14.6%), so the model is reading the stem, not guessing better in general. The same probes on arm C, trained on the plain universe where those suffixes are spread randomly across types, score at or below chance (4.2% and 6.2%): the suffix only becomes a feature when the training data makes it predictive, exactly the condition drug stems and processor prefixes satisfy in real data. The bare-format probe rows in table 8.3 sit at chance for every arm for the format reason noted in section 6.4, which is why the trained-format rows were added.

The effect shows up in the induction tasks as well. Arm E does the Timmy task from its weights at 75.0% (C: 59.4) and at 76.2% for k=4 (C: 52.5), and induction over held-out species without any context rises to 46.9% (C: 31.2): with a 70% reliable stem the model can place an unseen creature by its name alone, which is the "new drug name lands near its class" behaviour the user asked for. Recall (100%), public-suite score (81.8) and perplexity (15.6) are unchanged relative to C, so the morphology signal costs nothing elsewhere.

### 8.5 Cost

Each trained arm took 21 to 25 minutes of training at 837 to 1,171 tokens/s and 8.7 GiB peak (episodes average roughly 250 tokens, so the run is far from the card's limit), plus about 7 minutes of evaluation over 4,400 scored items. Arm A on short knowledge texts ran at 498 tokens/s, launch-overhead-bound as in section 7.2; packing would roughly triple that. The whole eight-arm sweep was about three and a half hours of GPU time including two restarts caused by evaluation-time memory handling (fused loss refusing to run with a full allocator cache; full-vocabulary float logits for long prompts), both fixed in the scorer.

### 8.6 Recommendation, revised

Train one run whose batches mix augmented knowledge text (about half), symbol-tuning episodes generated from the same database with random labels and varied grouping attributes (about 40%), and 10 to 20% generic few-shot replay. Do not stage it. Hold out one attribute and a slice of entities from the episodes and use them, plus a public few-shot suite with random labels, as the regression metric instead of perplexity. If the entity names carry any morphology (drug stems, retailer name variants, processor prefixes), make sure the same rendering variety appears in both the knowledge text and the episodes, because that is what turns the shared subword pieces into a feature the model uses without context. Next steps that this sweep did not run: the same arms on 7B, a real-world replicate with drug names and ATC classes, and the merchant database with statement-style noisy renderings in place of the creature universe. Since 2026-09-14 the ordered work queue lives in `PLAN.md` at the repo root; the questions it refers to are defined in `reports/QUESTIONS.md` and grounded in a 34-paper survey in `references/SURVEY.md`. Results of queue steps are appended below this section, one numbered subsection per step with its ID in the heading.

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
| recall, bare format | 19.4 | 18.8 | -0.3 [-1.9, 1.2] | | |
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
