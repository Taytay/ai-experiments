# Knowledge-Instruct: Effective Continual Pre-training from Limited Data using Instructions

- arXiv 2504.05571 v1, 8 Apr 2025 - https://arxiv.org/abs/2504.05571
- Oded Ovadia*, Meni Brief*, Rachel Lemberg, Eitam Sheetrit (Microsoft Industry AI)
- Venue: preprint; code and Companies dataset at github.com/meniData1/knowledge-instruct
- Source: references/papers/2504.05571/paper.txt

## One-paragraph summary
Knowledge-Instruct converts a small corpus into an information-dense instruction-tuning set: an extractor LLM (GPT-4o-mini by default) pulls out entities, then iteratively extracts every fact about each entity, rewrites each fact so the entity is named explicitly, deduplicates, paraphrases each fact k times, and wraps each paraphrase in one of 25 rule-based prompts such as "Tell me a fact about {entity}". The instruct model is then fine-tuned on these pairs together with general SFT data in a single stage. On a new fully fictional Companies benchmark, long-tail PopQA and MultiHop-RAG, it beats raw CPT, Rephrase CPT (100 rephrasings per document) and EntiGraph-style Synthetic CPT by wide margins, forgets less on general benchmarks, and improves rather than degrades answering with oracle context.

## Problem
Facts in small corpora appear once, so CPT cannot internalise them; CPT on an instruct model also breaks its chat behaviour and needs a second SFT stage; EntiGraph needs very many tokens plus instruction tuning (Sec. 2). The design principles are entities as anchors, full coverage, explicit context, repetition through paraphrase, and preserving the corpus's fact distribution (Sec. 3.1).

## Method
Six stages (Sec. 3.2): entity extraction with follow-up "what did you miss" prompts; fact extraction per entity, also iterated; contextualisation so each fact names the entity; deduplication; k paraphrases per fact; template conversion (App. C lists the 25 templates). Models: Llama-3.1-8B-Instruct and Phi-4-14B. Training (App. A): LR 1e-5, batch 4, context 2048; the CPT baselines get a follow-up SFT stage on 10,000 OpenOrca instructions at LR 1e-6. Baselines: CPT, Rephrase CPT (100 rephrasings per document from 10 prompts, App. D), Synthetic CPT (EntiGraph code, pairs/triples). Evaluation: open-ended questions, GPT-4o judge with strict scoring (Sec. 4.2). Companies has 23 fictional companies with a GPT-4o-written profile, catalog and financial report, 690 questions.

## Experiments and results
- Table 1, Llama-3.1-8B-Instruct (Companies / PopQA / MultiHop-RAG / MultiHop+Oracle / avg): Base 0.0 / 13.2 / 22.4 / 70.7 / 26.6; CPT 4.2 / 40.5 / 45.0 / 66.1 / 38.9; Rephrase CPT 17.7 / 49.2 / 41.2 / 66.7 / 43.7; Synthetic CPT 53.5 / 73.4 / 50.6 / 69.0 / 61.6; Knowledge-Instruct 81.8 / 76.8 / 56.5 / 80.0 / 73.6. Phi-4: Base 1.2 / 5.0 / 26.5 / 74.0; CPT 9.4 / 31.8 / 33.4 / 75.3; Rephrase CPT 13.3 / 32.7 / 36.7 / 75.6; Knowledge-Instruct 86.2 / 80.1 / 60.9 / 79.6 (avg 76.7). GPT-4o base 4.8 / 62.4 / 45.6 / 82.6.
- Forgetting (Table 2, Llama trained on MultiHop-RAG): average over nine benchmarks 63.9 for Knowledge-Instruct vs 61.5 Synthetic CPT, 60.4 CPT, 60.6 Rephrase CPT, 61.0 base, 63.6 base+Orca SFT. GSM8K 75.7 vs base 77.6, CPT 65.5; MMLU-Pro 35.3 vs base 33.6, CPT 22.6.
- CPT vs SFT (Fig. 3): Synthetic CPT before its SFT stage is best on Companies directly but underperforms on the easier oracle version; Knowledge-Instruct keeps both. The authors' "regularization hypothesis" is that mixing general SFT data with the knowledge data in one stage is what preserves reasoning (Sec. 5.3).
- Paraphrase count (Fig. 4, Companies, Llama): accuracy rises with the number of paraphrases per fact and "begins to plateau around 3 paraphrases"; five are called sufficient (Sec. 5.4).
- Extractor ablation (Fig. 2): Llama extracts the most facts but repeats itself, which acts as free paraphrasing; GPT-4o-mini is the cost/quality choice.
- Coverage (Table 3): with the concatenated extracted facts as context, GPT-4o scores 97.1 (PopQA) and 95.5 (Companies) vs 99.0 and 98.4 with full documents.

## Limitations
Only 8B and 14B instruct models, full fine-tuning, no LoRA. Companies text is GPT-4o-generated, which favours a GPT-4o-mini extractor and a GPT-4o judge. The paraphrase curve is a single dataset/model figure with no numbers in text. CPT baselines were run at an LR (1e-6 for their SFT stage) chosen to limit forgetting, not tuned for learning. Nothing is said about how many facts or entities the recipe scales to, or about seed variance.

## Relevance to this workspace
- DATA-5: the saturation point here is per atomic fact, about 3 paraphrases with 5 sufficient (Fig. 4), lower than the "10 per document" figure from Mecklenburg/Ovadia 2023. Our 14-20 texts per entity, each covering several attributes, already exceed this; the 1/3/7/14 sweep proposed in DATA-5 should show the plateau by 3-7.
- TRAIN-1 / TRAIN-2: Knowledge-Instruct's central claim is that one-stage mixing of knowledge instructions with general instructions beats a two-stage CPT-then-SFT pipeline, which is the same shape as our arm C (interleaved) vs arm D (sequential). Their Table 2 GSM8K and MMLU-Pro drops for two-stage CPT are the analogue of our ICL-suite loss.
- BASE-3 / EVAL-1: their trained format is "Tell me a fact about {entity} -> fact", i.e. the entity-conditioned cloze that our `L1_recall_fmt` already scores at 100. Their oracle result (Knowledge-Instruct raises MultiHop+Oracle from 70.7 to 80.0 while CPT lowers it to 66.1) predicts that our with-context ceiling should rise, not fall, under an instruction-style knowledge stream.
- EVAL-3: they refuse multiple choice outright to avoid guessing (Sec. 4.1) and use a strict LLM judge on generation.
- Cost on one 24 GB GPU: our universe facts are already atomic and entity-named, so only the template conversion, 3-5 paraphrases and a general-SFT mixture (10k OpenOrca) are needed; training cost equals arm C.

## Key references worth following up
Ovadia et al. 2312.05934; Yang et al. EntiGraph 2409.07431; Maini et al. 2401.16380; Brief et al. "cocktail effect" 2410.01109; Kandpal et al. long-tail (ICML 2023); Allen-Zhu & Li 2309.14402; Tang & Yang MultiHop-RAG 2401.15391.
