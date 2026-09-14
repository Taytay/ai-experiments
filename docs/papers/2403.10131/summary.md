# RAFT: Adapting Language Model to Domain Specific RAG

- arXiv 2403.10131v2 (5 Jun 2024) - https://arxiv.org/abs/2403.10131
- Tianjun Zhang, Shishir G. Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica, Joseph E. Gonzalez (UC Berkeley)
- Venue: preprint, under review (as marked in the text)
- Source: docs/papers/2403.10131/paper.txt

## One-paragraph summary
RAFT is a fine-tuning recipe for the "domain-specific open-book exam": the documents are known in advance, and at test time a retriever will supply top-k chunks, some irrelevant. Each training example is a question, a set of documents and a chain-of-thought answer that quotes the relevant passage verbatim. For a fraction P of examples the golden document is present with k-1 distractors; for 1-P only distractors are present, forcing memorization. On PubMed, HotpotQA and three Gorilla API benchmarks with LLaMA2-7B, RAFT beats domain-specific SFT with and without RAG and GPT-3.5 + RAG (Table 1). The chain-of-thought answers add up to 15 points (Table 2), the optimal P is 40 to 100% depending on the dataset (Fig. 5), and training with one to three distractors beats golden-only training and is robust to varying k at test time (Fig. 6).

## Problem
RAG alone does not use the chance to study the fixed corpus; SFT alone either ignores documents at test time or is not trained to cope with imperfect retrieval. Plain LLaMA2-7B with RAG scores near zero on HotpotQA and the API benchmarks because of style mismatch (Table 1).

## Method
- Training data (Sec. 3): P fraction: Q + D* + D1..Dk -> A*; (1-P) fraction: Q + D1..Dk -> A*. Standard SFT on this.
- A* is a GPT-4-1106-generated reasoning chain with ##begin_quote## citations and a final ##Answer (Fig. 3); Gorilla data already contains reasoning.
- Default: one golden and four distractor documents in training; at test time the golden document with four distractors (Sec. 4.4), or top-3 retrieved chunks for the Fig. 6 study; RAFT is retriever-agnostic.
- Baselines: LLaMA2-7B-chat zero-shot, with RAG, domain-specific fine-tuning (DSF), DSF + RAG, GPT-3.5 + RAG.

## Experiments and results
- Table 1 (PubMed / HotPot / HuggingFace / Torch Hub / TensorFlow): GPT-3.5 + RAG 71.60 / 41.5 / 29.08 / 60.21 / 65.59; LLaMA2-7B 56.5 / 0.54 / 0.22 / 0 / 0; LLaMA2-7B + RAG 58.8 / 0.03 / 26.43 / 8.60 / 43.06; DSF 59.7 / 6.38 / 61.06 / 84.94 / 86.56; DSF + RAG 71.6 / 4.41 / 42.59 / 82.80 / 60.29; RAFT 73.30 / 35.28 / 74.00 / 84.95 / 86.86.
- Note that DSF + RAG is worse than DSF alone on HuggingFace, Torch Hub and TensorFlow: a fine-tuned model that never saw context during training can be hurt by retrieved context (Sec. 4.1).
- CoT ablation (Table 2): without CoT 68.30 / 25.62 / 59.07 / 86.56 / 83.21; gains of 9.66 on HotpotQA and 14.93 on HuggingFace; slight loss on Torch Hub.
- Golden fraction (Fig. 5, NQ / TriviaQA / HotpotQA): best P is 40%, 60% or 100% depending on dataset; P = 80% is better than 100% in the text's summary.
- Distractors (Fig. 6, test with top-3): golden-only training is worst; D* + 3D best for NQ, D* + 1D best for HotpotQA; distractor training makes accuracy stable as test-time k varies from 2 to 10.
- Qualitative (Fig. 4): DSF answers a film title instead of the screenwriter; RAFT quotes the passage and answers correctly.

## Limitations
- LLaMA2-7B only; GPT-4 writes the reasoning chains, so part of the gain may be distillation of GPT-4 style.
- No seeds or confidence intervals; metrics per dataset are not spelled out; PubMed is yes/no.
- Baseline zeros on HotpotQA and API tasks are style failures, which inflates relative gains.
- P and the distractor count are swept separately, not jointly, and on different datasets from the main table.
- The retriever used for Fig. 6 is not described; closed-book performance of RAFT models is not reported.

## Relevance to this workspace
- BASE-1: the arm to run is C-RAFT: contexts of one gold field-guide entry (or merchant record) plus two distractors retrieved by the fine-tuned MiniLM from noisy renderings, P about 0.6, answer-only loss, order randomized; evaluate three ways: no context, oracle context, retrieved top-3 (right or wrong). RAFT's Fig. 6 says our current ctx_frac = 0.5 without distractors is the worst training configuration for a model that will later see retrieved context.
- REAL-2 and REPORT-3: the DSF + RAG < DSF rows are the concrete prediction that arm C fed real retrieved context could score *below* its no-context number unless trained with distractors; report oracle and retrieved separately, and rename the ceiling "oracle context" as REPORT-3 suggests.
- EVAL-3: the CoT-with-citation gain (Table 2) is only available if we add a generation-based eval; under cloze scoring the RAFT answer format cannot be used, so start with short answers and add citations once EVAL-3 exists.
- MODEL-4: retrieved contexts contain the clean merchant name, so RAFT-style training also tests whether the model can bridge `ELRHOLM` to `Elrholm` when the record is in the prompt.
- Cost on one 24 GB GPU: prompts grow from about 30 tokens to 200 to 300 tokens with three entries; at answer-only loss the label tokens are unchanged, so throughput per step falls by roughly the sequence-length ratio. Expect about 1.5x to 2x arm C, roughly 40 to 50 min on 3B for 800 steps, plus 19 s for the MiniLM retriever and a few minutes to build contexts.

## Key references worth following up
- Lin et al. 2023, RA-DIT, arXiv 2310.01352
- Shi et al. 2023, LLMs can be easily distracted by irrelevant context (ICML)
- Liu et al. 2023, Lost in the middle, arXiv 2307.03172
- Wang et al. 2023, InstructRetro, arXiv 2310.07713
- Patil et al. 2023, Gorilla, arXiv 2305.15334
- Asai et al. 2023, Self-RAG, arXiv 2310.11511
