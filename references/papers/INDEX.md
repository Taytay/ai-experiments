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
