# Synthetic Continued Pretraining (EntiGraph)

- arXiv 2409.07431v2 (3 Oct 2024) - https://arxiv.org/abs/2409.07431
- Zitong Yang, Neil Band, Shuangping Li, Emmanuel Candès, Tatsunori Hashimoto (Stanford)
- Venue: preprint (arXiv v2); later ICLR 2025
- Source: docs/papers/2409.07431/paper.txt

## One-paragraph summary
The paper argues that continued pretraining (CPT) on a small corpus fails because each fact appears in too few surface forms, and proposes "synthetic CPT": use a prompted LM to expand the small corpus into a large, *diverse* one, then CPT on that. Their instantiation, EntiGraph, extracts salient entities per document and prompts gpt-4-turbo to write text about every entity pair and many triples in the context of the source document. On 265 QuALITY books (1.3M tokens) they generate 455M synthetic tokens, CPT Llama 3 8B Base, and get closed-book QA accuracy rising log-linearly with synthetic token count from 39.49% to 56.22% (Fig. 2), beating CPT on raw text (which drops below base) and CPT on simple paraphrases (which plateaus early). The parametric knowledge also composes with RAG (Table 3) and survives instruction tuning. A toy graph model explains the scaling as "rearranging" knowledge into the deductive closure, with a mixture-of-exponential curve that fits the data (Fig. 4).

## Problem
Learning facts from a corpus where each fact appears once. Prior work (Allen-Zhu and Li; Berglund et al. reversal curse) shows models need hundreds to thousands of diverse representations per fact. Raw CPT on 1.3M tokens is 10,000x smaller than any prior CPT corpus (Table 1) and hurts: Raw CPT scores 38.15% vs 39.49% for the base model (Fig. 2).

## Method
1. Entity extraction prompt: summarize the document and list all salient entities as JSON (App. G.1).
2. Relation analysis prompt: for each entity pair (and a subset of triples), rephrase the document emphasizing each entity, then discuss their interaction, citing the document title (App. G.1). Diversity is thus "externalized" to the combinatorial structure of the entity graph rather than left to sampling temperature.
3. CPT Llama 3 8B Base: 2 epochs on 455M tokens, context 2048, batch 16, peak LR 5e-6 cosine, full-parameter FSDP, with 10% RedPajama replay per batch (App. C). 8xH100 at 6090 tok/s, about 41 h.
4. Baselines: Raw CPT (4 epochs, 0.1 replay, jointly tuned) and Rephrase CPT (three fixed easy/medium/hard rephrase prompts at temperature 1.0, stopped at 38M tokens; App. G.2). App. D adds a task-specific "QA SFT" baseline (28M tokens of generated QA pairs).
5. Evaluation: 4,609 QuALITY MCQs contextualized with title and author; 5-shot CoT; 64 samples per question, a random valid parse is taken (App. H.1). Closed-book summarization is scored by GPT-4 claim decomposition (false vs salient claims, Fig. 3).

## Experiments and results
- Scaling: EntiGraph CPT is log-linear in synthetic tokens up to 455M (Fig. 2); Rephrase CPT scales much more slowly and was stopped at 38M with a clear gap. The fitted curve (App. F.1) is y = 64.55 - 13.84(0.9989)^x - 8.47(0.8961)^x - 3.93(0.0546)^x, i.e. an asymptote near 64.5%.
- Closed-book endpoint: 56.22% vs GPT-4 closed-book 51.30% and GPT-3.5 44.81% (Fig. 2).
- Open-book: EntiGraph CPT + RAG 62.60% vs Llama 3 8B Base + RAG 60.35% at Recall@8 = 99.63% (Table 3); oracle GPT-4 86.09%. RAG alone adds 20.86 points to base; EntiGraph alone adds 16.73, i.e. over 80% of the RAG gain without test-time documents.
- QA SFT (App. D, Fig. 7) rises sharply and reaches similar accuracy at 28M tokens, but cost over $5k to generate because QA pairs are short; EntiGraph is cheaper per token.
- Summarization: EntiGraph Instruct produces more salient claims with far fewer false claims than Raw Instruct (Fig. 3); at token-matched 29M, EntiGraph still has fewer false claims than Rephrase (Fig. 8).
- Instruction tuning on UltraChat lifts AlpacaEval win rate 0% to 6.25% (App. C), and the model answers explicit, implicit and cross-article questions (Table 2/5).
- Theory (Sec. 6): memorization on an Erdos-Renyi relation graph; BFS-based augmentation; Theorem 1 bounds; three regimes (linear, log-linear, plateau) with plateau set by the fraction of reachable pairs.

## Limitations
- Needs a strong external generator (gpt-4-turbo); the authors admit self-bootstrapping is untested and hallucination risk grows on harder content (Sec. 7.1).
- One corpus, one model size, one seed; MCQ with a permissive 64-sample parsing scheme.
- 455M tokens for 1.3M source tokens (350x) is a heavy compute-for-data trade; the closed-book gain is still 6 points short of RAG.
- Rephrase baseline was cut at 38M tokens, so its asymptote is not measured.
- The toy model assumes pure memorization and Erdos-Renyi structure.

## Relevance to this workspace
- DATA-5 directly. Our "augmented" arm is 14 fixed templates, which is the Rephrase baseline in miniature; EntiGraph predicts it saturates early. The scaling curve says the doubling we saw from one to 14 templates is the linear regime, and further templated paraphrases will plateau. What kept scaling was multi-entity relation text. For merchants: prompt a local model to write texts relating (merchant, product), (merchant, neighbourhood), (product, category), and triples, so that the products-to-category bridge (DATA-2) and reverse relations (EVAL-7, reversal curse) get their own sentences. Run the proposed 1/3/7/14 template sweep plus 14 and 28 LLM-generated relation texts per merchant and plot `clean_category` against distinct texts.
- LIT-1: the paper is already cited; the missing piece is its Raw CPT < base result and the 10% replay it uses to avoid it. Our merchant runs have no general-text replay (TRAIN-4).
- REAL-2 and BASE-1: Table 3 predicts a fine-tuned model plus real retrieval beats base plus retrieval, so the end-to-end retrieval experiment should be run on arm C, not only on base.
- REAL-3: the 350x token multiplier is a warning for 10k entities; EntiGraph's per-book token counts (Fig. 6) scale with entity count squared.
- Cost on one 24 GB GPU: generation is the main cost. With 136 species or 120 merchants and roughly 5 attributes each, pairs plus a few triples give about 1,500 to 3,000 texts of 200 to 300 tokens, under 1M tokens; a batched Qwen2.5-7B-Instruct on the RTX 3090 produces that in tens of minutes, or use an API. Training an extra 1M tokens on 3B LoRA is about 20 min at the measured 851 tok/s; the full sweep is one afternoon.

## Key references worth following up
- Akyürek et al. 2024, Deductive closure training (ACL Findings)
- Maini et al. 2024, Rephrasing the web (ACL), arXiv 2401.16380
- Mecklenburg et al. 2024, arXiv 2404.00213
- Ovadia et al. 2024, arXiv 2312.05934
- Berglund et al. 2023, reversal curse, arXiv 2309.12288
- Snell et al. 2022, arXiv 2209.15189 (context distillation, cited as the alternative to long context)
