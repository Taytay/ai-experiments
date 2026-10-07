# Papers read for the knowledge-injection survey (2026-09-14)

Each folder holds `paper.txt` (pdftotext extraction), `summary.md`, `meta.json`, and where available the TeX source. The synthesis is in `references/SURVEY.md`; the questions it answers are in `reports/QUESTIONS.md`.

## Thread 1: knowledge storage, extraction, manipulation, FT vs RAG
- [2309.14316](2309.14316/summary.md) - Physics of LMs 3.1, Knowledge Storage and Extraction (2024) - one rendering per fact is memorized but not extractable; rewrites plus sentence permutation give 96.6%, mixed QA+doc training works even unaugmented.
- [2309.14402](2309.14402/summary.md) - Physics of LMs 3.2, Knowledge Manipulation (2024) - functions of stored facts need CoT at train and test; inverse search ~0% without reversed text.
- [2405.05904](2405.05904/summary.md) - Does Fine-Tuning on New Knowledge Encourage Hallucinations? (2024) - unknown facts fit slowly and then damage known facts linearly; early-stop, MaybeKnown mixing, IDK labels.
- [2312.05934](2312.05934/summary.md) - Fine-Tuning or Retrieval? (2024) - RAG beats unsupervised FT on 7B; FT helps only with paraphrases, monotone to ten, no saturation shown.

## Thread 1b: recipes that make injected knowledge extractable
- [2409.07431](2409.07431/summary.md) - Synthetic continued pretraining, EntiGraph (2024) - paraphrases plateau; entity-relation text scales log-linearly to 455M tokens; composes with RAG.
- [2406.06326](2406.06326/summary.md) - Self-Tuning (2025) - template-derived cloze/MCQ/NLI tasks over the documents, staged with QA review: EM 3.6 to 31.5, matching open-book.
- [2412.14964](2412.14964/summary.md) - Knowledge injection via self-distillation (2025) - KL to the with-context teacher beats SFT by 7-10 points, matches RAG closed-book on Qwen2.5-3B.
- [2209.15189](2209.15189/summary.md) - Learning by Distilling Context (2022) - the general teacher-template / student-template recipe; students cheat on input distribution unless labelings are mixed.
- [2403.10131](2403.10131/summary.md) - RAFT (2024) - train with gold plus 1-3 distractors, gold present 40-100%; golden-only training is worst.

## Thread 1c: recent injection methods, LoRA vs full FT
- [2510.09885](2510.09885/summary.md) - Masked fine-tuning for knowledge injection (2026) - "recover the masked passage" training gives forward and backward recall without paraphrases on 3B-8B Instruct.
- [2603.22213](2603.22213/summary.md) - SPA (2026) - seven learning-strategy prompts scaled to 15M-455M tokens beat EntiGraph/SEAL/Active Reading; a scaling recipe, not a paraphrase method.
- [2504.05571](2504.05571/summary.md) - Knowledge-Instruct (2025) - atomic entity-named facts, 3-5 paraphrases, one-stage mix with general SFT: 81.8% vs 17.7% for 100x rephrase CPT.
- [2404.00213](2404.00213/summary.md) - Injecting New Knowledge via SFT (2024) - token-scaled QA saturates 5x to 10x from uneven fact coverage; fact-scaled sets keep improving.
- [2405.09673](2405.09673/summary.md) - LoRA Learns Less and Forgets Less (TMLR 2024) - r=256 all-modules closes IFT but not CPT gaps; forgetting ordered by rank and duration.

## Thread 2: multiple-choice scoring validity
- [2104.08315](2104.08315/summary.md) - Surface Form Competition (EMNLP 2021) - PMI_DC = log P(option|prompt) - log P(option|domain premise); beats mean-per-token at every GPT-2/GPT-3 size.
- [2402.01781](2402.01781/summary.md) - When Benchmarks are Targets (ACL 2024) - symbol vs cloze vs hybrid scoring; hybrid recommended; small models least rank-stable.
- [2406.08446](2406.08446/summary.md) - OLMES (NAACL 2025) - cloze is the only informative format for ~1B base models; per-task normalisation rule (pmi / char / none).
- [2607.12767](2607.12767/summary.md) - Accuracy under Length Bias (ICML 2026) - mean-per-token over-corrects toward long options; Bayesian sum-minus-b*length fixes it with no extra passes.
- [2502.14127](2502.14127/summary.md) - Which of These Best Describes MCQ Evaluation (ACL 2025) - constructed-response conversion, item rubrics, choices-only baselines, IRT.
- [2309.03882](2309.03882/summary.md) - LLMs Are Not Robust Multiple Choice Selectors, PriDe (ICLR 2024) - RStd bias metric; label-free prior estimation and division.

## Thread 3: symbol tuning and fine-tuning for in-context learning
- [2305.08298](2305.08298/summary.md) - Symbol tuning (EMNLP 2023) - +9 to +16 points without natural labels on 8B-540B; 8B loses 6-7 points with natural labels; 22 datasets, 30k symbols.
- [2505.14233](2505.14233/summary.md) - Mechanistic Fine-tuning for ICL, ABFT (2025) - attention-only loss on induction heads, W_Q/W_K, 512 prompts, 32 steps; works from 812M.
- [2512.19879](2512.19879/summary.md) - Fine-Tuned In-Context Learners (2025) - fine-tune on k-shot prompts with loss on all answers; Gemma-2 2B matches 540B symbol tuning.

## Thread 4: masked-LM encoders
- [2502.03793](2502.03793/summary.md) - It's All in the [MASK], ModernBERT-Large-Instruct (2025) - single-mask ATP with [unused0] anchor, 80/20 dummy mix; 43.06 MMLU vs Qwen2-0.5B 33.7.
- [2505.12306](2505.12306/summary.md) - Bidirectional LMs are Better Knowledge Memorizers, WikiDYK (2025) - Flan-T5-770M span prediction beats 1-8B causal LMs above ~1,000 facts; scored by generation.
- [2412.13663](2412.13663/summary.md) - ModernBERT (2024) - 8k context, 83 unused tokens, 30% masking, fine-tuning LRs 1e-5 to 8e-5.
- [2510.16797](2510.16797/summary.md) - MOSAIC (2026) - new-token rows learn only with a joint domain-token-restricted MLM loss (alpha 0.3) plus contrastive, then contrastive-only.

## Thread 5: knowledge editing
- [2210.07229](2210.07229/summary.md) - MEMIT (ICLR 2023) - 10k batch edits on 6B/20B; keys depend on subject only; reverse relations out of scope.
- [2410.02355](2410.02355/summary.md) - AlphaEdit (ICLR 2025) - null-space projection preserves general capability through 3k sequential edits on 8B; small models still degrade.
- [2401.01286](2401.01286/summary.md) - Comprehensive Study of Knowledge Editing, KnowEdit/EasyEdit (2024) - single-layer masked fine-tune (FT-M) equals or beats MEMIT on insertion; portability weak everywhere.
- [2511.05852](2511.05852/summary.md) - Can Fine-Tuning Erase Edits? (KDD 2026) - later LoRA removes 10-45 points of edit efficacy; AlphaEdit most fragile.

## Thread 6: tokenizer robustness, embedding models, few-shot embedding classifiers
- [2406.11687](2406.11687/summary.md) - Tokenization Falling Short (2024) - case and typos change token ids; scale does not fix it; BPE-dropout p=0.2 helps structure probes.
- [2506.05176](2506.05176/summary.md) - Qwen3 Embedding (2025) - 0.6B/4B/8B, last-token pooling, instruction-aware queries, false-negative mask; cased Qwen BPE.
- [2209.11055](2209.11055/summary.md) - SetFit (2022) - label-contrastive pairs then logistic head; 62.3 at 8/class vs 43.0 fine-tuning; no centroid ablation.

## Thread 7: Pinterest recommender systems (read 2026-10-05 for the history-aware encoders, PLAN rows 194-208)
- [2306.00248](2306.00248/summary.md) - TransAct (KDD 2023) - 100 recent actions with action-type embeddings and the candidate concatenated onto every row (early fusion), 2-layer transformer, first-10 plus max-pool read-out; the bare transformer is only +1.56 of the +9.40.
- [2506.02267](2506.02267/summary.md) - TransAct V2 (2025) - candidate-anchored nearest neighbours from a 10^4-action lifelong history plus the most recent r; impression ("seen, not chosen") negatives beat in-batch for the next-action auxiliary loss (w=0.01).
- [2205.04507](2205.04507/summary.md) - PinnerFormer (KDD 2022) - dense all-action objective (any positive in the next 28 days from random positions) keeps a daily batch user embedding near realtime; logQ correction more than doubles in-batch recall.
- [2007.03634](2007.03634/summary.md) - PinnerSage (KDD 2020) - Ward clusters over fixed item embeddings, medoids and time-decayed importance (lambda 0.01); the category-level analogue of our MaxSim reader.
- [2504.17811](2504.17811/summary.md) - OmniSage (KDD 2025) - RWR top-k neighbours per type into a 1-layer transformer, trained with pair, feature and sequence contrastive tasks; graph lifts recall 0.44 to 0.61; random negatives +31-44%.
- [2404.16260](2404.16260/summary.md) - OmniSearchSage (WWW 2024) - crowd-written board titles (+17%) and engaged queries (+7%) as item text: the encoder-side crowd line; hash n-gram tower; logQ sampled softmax.
- [2507.12704](2507.12704/summary.md) - PinFM (2025) - pretrain a causal transformer on activity sequences (next / multi / future-token InfoNCE), fine-tune with the candidate appended (early fusion +3.8% vs pooled +1.9%); frozen gives ~0.
- [2606.00422](2606.00422/summary.md) - UniPinRec (RecSys 2026) - one transformer for dense generative retrieval and ranking; masked action labels in the history plus a joint loss beat the production ranker +14.7%; shared history KV ~3x cheaper ranking; argues against semantic IDs.
- [2508.16793](2508.16793/summary.md) - Bootstrapping Conditional Retrieval (RecSys 2024) - the item's own attribute as a condition in the user tower, trained on plain engagement pairs; 82.8% vs 20.3% condition match.
- [2506.23060](2506.23060/summary.md) - Multi-Embedding Retrieval (KDD 2025) - implicit interest clusters (farthest-point init, single-assignment routing) plus explicit conditions logged at engagement; 3.2% overlap, each helps a different user segment.
- [2606.00324](2606.00324/summary.md) - LLMs Need Encoders for Semantic IDs Too / PrefixMem (2026) - hashed prefix memory beats a 336M transformer for deep SID levels (memory, not compute); useless below 3 levels, so not for our small category codes.

## Thread 8: late interaction and decision models (read 2026-10-05 for PLAN rows 209-210: base encoders, a late-interaction decision model)
- [2609.25859](2609.25859/summary.md) - BELXTR (preprint 2026) - per-name candidates with an MML loss and a learnable temperature matter most in multi-vector entity linking: no temperature -9 to -15 recall@1, concatenating an entity's names into one document -2.8 to -21.2; gains concentrated on near-identical names (NLM-Gene 82.29 vs 57.93 single-vector).
- [2304.01982](2304.01982/summary.md) - XTR (NeurIPS 2023) - in-batch top-k token alignment lets candidates be scored from retrieved tokens with missing similarities imputed by the k'-th score (MRR@10 37.4 vs 22.6 without); multi-vector is weak on long queries (ArguAna 40.7 vs 51.1 single-vector).
- [2605.24938](2605.24938/summary.md) - SMART (preprint 2026) - a single-vector embedder's final-layer token states support MaxSim for free (+0.3 to +2.5 on MMEB-V2); the hybrid pooled + MaxSim objective beats pooled-only by +6.5 and late-only by +0.8; no inference-only gain on classification.
- [2511.16106](2511.16106/summary.md) - Weighted Chamfer (AAAI 2026, inferred) - per-token weights on MaxSim terms from IDF or a convex ranking loss on a frozen ColBERTv2: +1.28% / +3.66% relative BEIR rerank Recall@10.
- [2511.07969](2511.07969/summary.md) - Unified Work Embeddings (preprint 2026) - soft late interaction (softmax over target tokens, tau_a 0.1) 39.45 MAP vs hard MaxSim 37.63, which fell below mean pooling (38.00), on short labour-market texts; symmetric per-graph InfoNCE +1.77; zero-shot over unseen label spaces (+8.9 MAP on O*NET).
- [2406.17968](2406.17968/summary.md) - LITE (preprint 2024) - a separable row-then-column MLP over the token similarity matrix beats sum-of-max (MS MARCO MRR@10 .393 vs .383), but distillation from a cross-encoder matters more (NQ ColBERT .690 -> .756).
- [2606.22807](2606.22807/summary.md) - KaLM-Reranker-V1 FBNL (preprint 2026) - cached, mean-pooled passage states read by decoder cross-attention: the 0.27B Nano scores 58.54 BEIR at 1/8 the cost of gte-reranker-base (56.77); training at several pooling ratios is essential (53.58 -> 38.57 without); no MaxSim baseline.
- [2510.14880](2510.14880/summary.md) - mxbai-edge-colbert-v0 (preprint 2025) - 17M/32M Ettin ColBERTs: projection 48-96 flat (0.597-0.599), 32 drops (0.577); residual 2-layer head +1.3; Ettin needs ~1.8x the lr; a saturated teacher distils 3 points worse; answerai-colbert-small-v1 still best on short text (NanoBEIR 0.6545).
- [2602.16609](2602.16609/summary.md) - ColBERT-Zero (LightOn, 2026) - contrastive phases run in the multi-vector setting beat a KD step on a dense model (BEIR 55.43 vs 54.09); supervised contrastive + KD in ColBERT gets 99.4% of it for a tenth of the compute; keep the base's prompts (dropping them costs). Read 2026-10-06 for rows 222, 209.

## Thread 10: recommender graphs and two-tower models (owner, 2026-10-07)
- [2411.19513](2411.19513/summary.md) - ContextGNN (2024) - pair-wise scores for items in the user's local subgraph, two-tower beyond it, a learned per-user fusion; our known vs first-time payee split
- [2412.17245](2412.17245/summary.md) - GraphHash (2024) - Louvain clusters of the user-item graph as hash buckets; for us, behaviour clusters of payees across real users
- [2605.05238](2605.05238/summary.md) - DG-SA-GNN (2026) - four dynamic user-similarity graphs, MovieLens-100K only; weak evidence
- [2501.01073](2501.01073/summary.md) - G2PT (2025) - autoregressive graph generation; not applicable to categorisation
