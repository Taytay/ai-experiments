# Fine-tuning LLMs and embedding models for merchant knowledge: frameworks, vocabulary, and knowledge injection

Date: 2026-09-12. Hardware: RTX 3090 (24 GB), driver 591.86, Windows 11, torch 2.11 + cu128.
Experiments run with transformers + torch only (Smart App Control blocks triton/pyarrow; see NOTES.md).
Supporting docs: [frameworks.md](frameworks.md), [lit_review.md](lit_review.md). Code: `../experiments/`. Raw numbers: `../results/`.

## 1. Executive summary

- **Frameworks.** For one 24 GB GPU, Unsloth is the consensus choice (fastest, lowest VRAM, native Windows, now covers embedding models via `FastSentenceTransformer`). Axolotl is the pick for multi-GPU nodes with YAML-driven reproducibility. TRL is the substrate both wrap and the right layer if you need a custom loss. torchtune is unmaintained since July 2025; do not start on it. For RL at scale, verl. For embedding models specifically, the trainer is sentence-transformers' `SentenceTransformerTrainer` whether or not Unsloth is wrapping it.
- **New vocabulary.** Almost never worth it for merchant names. Subword tokenization already handles them; expansion requires continued pretraining and can hurt at small token budgets. If you must add tokens, initialize inside the existing embedding distribution (mean-of-subwords or Hewitt's N(mu, Sigma) sampling), never random, and train afterwards. Our experiments reproduce this: added tokens gave no gain over subwords in the embedding model and a small loss for the LLM. The one place a tokenizer intervention paid off was **aliasing**: copying a trained mixed-case merchant embedding into a token for the bank-statement uppercase form (see 4.3).
- **Knowledge injection.** Dumping the merchant database as one sentence per store into the model does not produce usable knowledge, even when memorized. Paraphrase and QA augmentation of the same facts (the Physics-of-LMs / EntiGraph recipe) is what makes knowledge extractable in new task formats. Full fine-tuning beats LoRA at absorbing new facts; LoRA forgets less. Retrieval (fact in context) remains the strongest and cheapest baseline; the right production design is RAG over the merchant DB plus augmented fine-tuning for the head of the distribution, with RAFT-style training so the model uses retrieved records well.

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

**Embedding models.** Same mechanics on the underlying transformer; the pooling layer is unaffected. Contrastive fine-tuning after adding tokens is mandatory because the new rows start with no meaning. Our test (4.2) shows it buys nothing over subwords for MiniLM.

**Cheap tricks that do work.**
- *Text normalization* before tokenizing: title-case the merchant, strip store numbers, expand `SQ *`, `TST*`, `POS DEBIT`. This closes much of the mixed-case vs uppercase gap for free.
- *Token aliasing*: after training on clean names, add the uppercase/statement form as a token and copy the trained embedding into it. Tested in 4.3.
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

RESULTS_PLACEHOLDER

## 5. Recommended recipe for the personal-finance use case

1. **Normalize first.** Canonicalize statement strings (case, store numbers, processor prefixes) before anything touches a model. Cheapest win available.
2. **Retrieval is the backbone.** Embed the merchant DB (name, aliases, what they sell, category) and retrieve top-k records into the classifier's context. Long-tail merchants will never be reliably parametric.
3. **Fine-tune the embedding model** on (statement string, merchant record) and (merchant record, category) pairs with hard negatives so retrieval itself is good. Use sentence-transformers' trainer; Unsloth for speed once triton loads. Do not add vocabulary.
4. **For parametric knowledge that transfers**, build an augmented corpus: 10+ paraphrases per merchant, QA in both directions, statement-style mentions, product-to-category chains. Full fine-tune (or high-rank LoRA on all modules) at low LR with replay of general data; average with the base checkpoint (WiSE-FT) if perplexity drift matters.
5. **Train the model to use retrieval** (RAFT): fine-tune on prompts containing retrieved records plus distractors, targets that cite the record. This is what makes knowledge improve *other* tasks (budget Q&A, anomaly explanations, merchant normalization) rather than just the classifier.
6. **Evaluate on held-out merchants and formats**, and track hallucination on merchants the model was never taught.
