# Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs

- arXiv 2312.05934, v3 (30 Jan 2024) - https://arxiv.org/abs/2312.05934
- Oded Ovadia, Menachem Brief, Moshik Mishaeli, Oren Elisha (Microsoft, Israel)
- Venue: preprint
- Source: references/papers/2312.05934/paper.txt

## One-paragraph summary
The paper compares unsupervised fine-tuning (continued next-token training on Wikipedia chunks) with RAG (bge-large-en embeddings, FAISS, top-K chunks prepended) for injecting knowledge into Llama2-7B, Mistral-7B and Orca2-7B. On four MMLU subjects plus prehistory (topics the models partly know) and on a new 910-question current-events set (August-November 2023, after the models' cutoffs), RAG beats fine-tuning in every configuration, and fine-tuning plus RAG is not reliably better than RAG alone (Tables 1, 2). Plain fine-tuning on new information barely helps and degrades Llama2 (0.353 to 0.219); fine-tuning on GPT-4 paraphrases of the same chunks improves monotonically with the number of paraphrases (Fig. 4). The authors conclude RAG is the more reliable injector and hypothesize that new knowledge must be repeated in numerous forms to be learned.

## Problem
Given a text corpus B_Q relevant to a question set Q, which transformation of the model (fine-tuning on B_Q or retrieval from B_Q at inference) most raises the multiple-choice knowledge score L_{M,Q}? The paper also separates "previously seen" knowledge (MMLU topics) from "entirely new" knowledge (current events).

## Method
- Knowledge bases: Wikipedia articles per topic, cleaned with wikiextractor, cut into 256-token chunks with BOS/EOS markers.
- Fine-tuning: unsupervised causal LM training, lr 1e-6 to 5e-5 (searched), up to 5 epochs, batch 64, 4xA100; validation on 240 held-out paraphrased chunks per task.
- RAG: bge-large-en, dot-product top-K, K in {0..5}; 0-shot and 5-shot.
- Current events: GPT-4 writes four specific questions per chunk, keeps the two most specific, manual verification; 910 questions. Paraphrases: GPT-4, ten per chunk, prompt in Appendix B.
- Evaluation: LM-Evaluation-Harness; each option appended to the question and scored by log-likelihood; argmax is the prediction (Eq. 4).

## Experiments and results
- Table 1 (MMLU, log-likelihood accuracy): e.g. Mistral-7B anatomy 0-shot base 0.556, +RAG 0.681, FT 0.570, FT+RAG 0.659; Orca2 astronomy 0-shot base 0.645, +RAG 0.750, FT 0.651, FT+RAG 0.750; Llama2 college chemistry 0-shot 0.310, 0.380, 0.390, 0.390. RAG improves over base everywhere; FT improves in most cases but less; FT as RAG generator helps only sometimes. Fig. 2 averages the relative gains.
- Table 2 (current events): Mistral base 0.481, +RAG 0.875, FT-reg 0.504, FT-par 0.588, FT-reg+RAG 0.810, FT-par+RAG 0.830. Llama2 0.353, 0.585, 0.219, 0.392, 0.326, 0.520. Orca2 0.456, 0.876, 0.511, 0.566, 0.820, 0.826. Base models exceed 0.25 through reasoning and partial prior knowledge (Appendix C examples).
- Fig. 4: accuracy on current events is a monotonically increasing function of the number of paraphrases for all three models.
- Fig. 3: training loss drops sharply at each epoch boundary, the signature of memorization (Tirumala et al.).
- Table 3 (Appendix A): no stable best K; anatomy prefers K=2, elsewhere the best-to-worst K gap can be large; the authors call K an unstable hyperparameter.

## Limitations
Multiple-choice log-likelihood scoring only, no generation; unsupervised FT only, no instruction-format or QA-format fine-tuning, no LoRA; three 7B models; Wikipedia-only sources; the current-events RAG advantage is inflated by a one-to-one chunk-to-question construction (Section 5). Paraphrase counts are reported only up to ten with no saturation analysis; hyperparameters are acknowledged as decisive.

## Relevance to this workspace
- DATA-5 / LIT-1: this is the paper behind "expose the model to numerous variations". It shows monotone gain up to ten GPT-4 paraphrases (Fig. 4) and does *not* show saturation at ten; the saturation claim in LIT-1 needs a different source or our own sweep. Their paraphrases are free LLM rewrites of whole chunks, not templates, so the proposed 14 + 28 LLM-generated arm is the direct replicate.
- BASE-1 / REAL-2 / REPORT-3: the "RAG ceiling" in our report is their current-events setting (verbatim fact in context) and should be named "oracle context"; their end-to-end RAG with a real retriever and unstable K is what REAL-2 asks for. Their FT+RAG rows are the template for arm C-RAFT's evaluation columns (with context, without, and FT vs base as generator).
- TRAIN-4: unsupervised FT on new facts with lr up to 5e-5 and 5 epochs degraded Llama2 on the target task (0.353 to 0.219), i.e. forgetting inside the domain, not just outside; a general-text replay control belongs in `ft_aug`.
- EVAL-1 / EVAL-2: their scorer is the lm-eval-harness log-likelihood of option text appended to the question, without the option list shown, the same family as our cloze scorer; it makes our numbers comparable with theirs but inherits the same length prior, which they do not correct.
- MODEL-3: the retriever is bge-large-en, one of the candidates listed for the embedding upgrade.

## Key references worth following up
- 2309.12288 Berglund et al., reversal curse (motivates paraphrase-as-repetition)
- 2205.10770 Tirumala et al., memorization without overfitting
- 2308.08747 Luo et al., catastrophic forgetting in continual fine-tuning
- 2305.11206 LIMA (fine-tuning does not add knowledge)
- 2310.07521 Wang et al., factuality survey (taxonomy of factual errors)
- Kandpal et al. ICML 2023, long-tail knowledge (no arXiv id in bibliography)
