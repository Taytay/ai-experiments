# SPA: A Simple but Tough-to-Beat Baseline for Knowledge Injection

- arXiv 2603.22213 v2, 9 Aug 2026 (dated August 11, 2026 on the page) - https://arxiv.org/abs/2603.22213
- Kexian Tang*, Jiani Wang*, Shaowen Wang, Kaifeng Lyu (Tsinghua IIIS)
- Venue: preprint; code at github.com/Tangkexian/SPA
- Source: references/papers/2603.22213/paper.txt

## One-paragraph summary
SPA (Scaling Prompt-engineered Augmentation) is a single-stage synthetic-data recipe for corpus-level knowledge injection: seven fixed, human-written prompt templates derived from learning-science strategies (key concepts, mind map, implications, critical-thinking QA, case study, two-reader discussion, teacher-style explanation) are applied repeatedly to each source document with a generator LLM, each prompt contributing an equal share of tokens, and the target model is then continually pretrained on the resulting corpus with no replay. In token-matched comparisons on SQuAD, QuALITY and MultiHop-RAG, SPA matches or beats SEAL, PaST, EntiGraph, SoG and Active Reading, with the gap widening as the synthetic corpus scales to hundreds of millions of tokens. The analysis attributes SEAL's saturation to diversity collapse of an RL-trained generator and argues that a good one-stage prompt set beats multi-stage pipelines.

## Problem
Domain corpora are small and non-redundant, so direct CPT overfits surface forms. Prior remedies are either RL-trained generators (SEAL, PaST) that need downstream rewards or multi-stage pipelines (EntiGraph, SoG, Active Reading) with fragile intermediate steps (Sec. 3.2). The authors want a task-agnostic baseline that is simple and scales.

## Method
Given corpus D, generate D~ with M = 7 prompts, each producing |D~|/M tokens, then CPT (Sec. 3.3). Prompts have Instruct and Base variants (App. A). Settings (App. B): 2 epochs, context 2048, weight decay 0, 3% linear warmup then linear decay; SQuAD peak LR 5e-5 and batch 64 at large scale (LR swept in [4e-5, 7e-5] at small scale); QuALITY LR 3e-5, batch 64; MultiHop-RAG LR in [4e-5, 6e-5], batch 8. Full-parameter training is implied (EntiGraph codebase reproduction, Table 7). Benchmarks: SQuAD (200 passages, 974 questions, Qwen2.5-7B as both generator and target, up to 3200 samples per passage = 120M tokens, ~4000x the corpus); QuALITY (265 passages, Llama-3-8B, gpt-oss-120b generator, 455M tokens ≈ 350x, 5-shot CoT MC eval); MultiHop-RAG (609 articles, 15M tokens, GPT-4o-mini generator, Qwen2.5-7B and Llama-3-8B). Judging uses GPT-4.1 on the first paragraph (SQuAD) or first sentence (MultiHop-RAG) of the output (App. C).

## Experiments and results
- Table 1 (largest scale): SQuAD base 31.31, Rephrase 86.86, QA 89.63, SEAL 74.23, Active Reading 90.25, SPA 91.27; the with-passage ceiling for Qwen2.5-7B is 91.38 (Sec. 5.1). QuALITY: base 39.27, QA 52.33, EntiGraph 56.22, Active Reading 51.75, SPA 57.03. MultiHop-RAG Qwen2.5-7B: base 60.91, EntiGraph 85.42, Active Reading 83.33, SPA 86.64; Llama-3-8B: base 73.16, EntiGraph 84.31, Active Reading 84.44, SPA 88.36.
- Scaling (Fig. 2, Sec. 4-5.1): SEAL with 5 augmentations per passage gives 58.2%; 27 gives 70.74%; SPA trails SEAL at the smallest budgets but overtakes it at ~27 samples per passage and keeps improving to 120M tokens.
- Generator strength (Table 2, 27M tokens QuALITY): QA baseline 52.99 with GPT-4-Turbo vs 47.47 with gpt-oss-120b; SPA with gpt-oss-120b 52.26; with GPT-4o, SPA 55.49 vs QA 51.31.
- Diversity (Table 3/8): SEAL is far more repetitive than all others on compression ratio, self-repetition, self-BLEU and POS-compression; SPA is comparable to Active Reading.
- Prompt ablation (Table 4, 22M tokens SQuAD): single prompts score 78.95 (key concepts) to 85.52 (implications) vs 87.68 for the full set; removing the weakest prompt costs 1.62 points (Table 5). A subset tuned on SQuAD reaches 88.19 but transfers worse to QuALITY (51.51 vs 52.26).
- Forgetting (Tables 9-10): after 11M SQuAD tokens the general-benchmark average is 61.10 vs base 60.96; after 455M QuALITY tokens it drops to 54.33 vs base 58.56 (EntiGraph 54.24, Active Reading 53.27). No replay was used.

## Limitations
Only 7B-8B models; token budgets are 15M-455M, i.e. hundreds to thousands of times the corpus, which is the regime where SPA wins, and the paper concedes it does not lead at small budgets. Data-generation cost is external (API or a 120B MoE). Evaluation is LLM-judged generation on base models. Hyperparameters were re-tuned per scale for small budgets, complicating the curves. Forgetting is measured only on four MC benchmarks and no replay mitigation is studied.

## Relevance to this workspace
- DATA-5: SPA is the "LLM-generated texts" arm, but it is a scaling method, not a paraphrase method: seven genre transformations, each carrying the facts in a different discourse form, multiplied to 350-4000x tokens. The paper's own Rephrase and QA baselines are within 1-5 points of SPA on SQuAD, so at our scale (14-20 texts per entity) the expected win over templated paraphrases is small; the lever is form diversity and token volume together.
- LIT-1: adds the 2025-2026 synthetic-CPT lineage (SEAL, PaST, Active Reading, SoG, GraphGen) missing from the report.
- TRAIN-4 / EVAL-5: SPA's forgetting without replay (58.56 to 54.33) supports adding general-text replay to `ft_aug`.
- REAL-3: their SQuAD result reaches the in-context ceiling for 200 passages at 120M tokens; useful as an upper bound on what parametric injection can do per entity.
- Cost on one 24 GB GPU: generation with a local Qwen2.5-7B (their SQuAD generator) for 136 species or 120 merchants x 7 prompts x ~10 samples x ~300 tokens is ~3M tokens, roughly an hour with a batched server; training 3M tokens on 3B LoRA at 851 tok/s is about an hour. A 30-50x-token SPA arm is therefore an afternoon; the 4000x regime is out of reach.

## Key references worth following up
EntiGraph / Yang et al. 2409.07431; Active Reading (Oguz et al., ICLR 2026); SEAL (Zweiger et al., NeurIPS 2025); PaST 2601.11258; GraphGen 2505.20416; Maini et al. Rephrasing the Web 2401.16380; Abonizio et al. 2508.06178; Gu et al., data mixing phase transitions (NeurIPS 2025); Allen-Zhu & Li Part 3.3.
