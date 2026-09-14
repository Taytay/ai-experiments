# It's All in The [MASK]: Simple Instruction-Tuning Enables BERT-like Masked Language Models As Generative Classifiers

- arXiv 2502.03793v2, 10 Feb 2025 - https://arxiv.org/abs/2502.03793
- Benjamin Clavié, Nathan Cooper, Benjamin Warner (Answer.AI)
- Venue: preprint ("submitted to Natural Language Processing")
- Source: docs/papers/2502.03793/paper.txt

> Extraction caveat: Table 2 is column-shifted in the plain-text extraction; decoder baselines were reconstructed by column counting and cross-checked against Table 4 and the prose.

## One-paragraph summary
The paper turns ModernBERT-Large (395 M) into a zero-shot classifier and multiple-choice answerer using nothing but its pretrained MLM head. Every task is reframed as a cloze with exactly one `[MASK]`, preceded by an untrained anchor token `[unused0]`; training data is a filtered 20 M-example subset of FLAN restricted to single-token answers. The surprising ingredient is that replacing 20 % of the training examples with "dummy" MLM examples (30 % random masking, every masked position labelled with the `[MASK]` id itself, i.e. a meaningless constant label) beats both pure answer-token prediction and a 20 % mix of real MLM (Table 1). ModernBERT-Large-Instruct reaches 43.06 MMLU zero-shot, above similarly sized decoders and at 93 % of Llama-3.2-1B; fine-tuned through the MLM head it matches or slightly beats a classification head on seven NLU tasks (Table 3). The effect is specific to a modern backbone: the identical recipe on RoBERTa-Large or GTE-en-MLM-Large is far weaker (Table 4).

## Problem
Encoder models dominate production classification but need a task head and fine-tuning, so they are weak zero-shot; existing MLM-head methods (PET, UniMC) need heavy prompt engineering, custom attention masks, or conversion to autoregressive decoding. The authors ask whether a modern encoder can be instruction-tuned as simply as an LLM.

## Method
- **Answer Token Prediction (ATP)** (Sec. 2.2): mask a single token, the verbalizer of the label or answer. A verbalizer is a single token that stands for the whole answer; when no meaningful single token exists, semantically empty verbalizers ("A", "B", "C", "D") are used, with the options listed in the prompt. Multi-mask answers are deliberately avoided because MLMs fill all masks in one pass and "adding multiple [MASK] tokens could bias the model to consistently predict the longest possible answer" (Sec. 2.3); training the model to ignore surplus masks is left to future work.
- **Data** (Sec. 2.3, Fig. 2): FLAN-2022, 396 M examples of which 120 M are single-token; MMLU/BBH and the evaluation classification sets are filtered out, large datasets are down-sampled, leaving 20 M examples. FLAN's own templating diversity is reused; the only added template is `[unused0] [MASK]` before the answer.
- **Objective mix** (Sec. 2.4): 80 % ATP / 20 % dummy MLM. A labelling bug made the "MLM" 20 % predict `[MASK]` for every masked position; this variant beat correct MLM (Table 1: ATP only 37.07 MMLU / 13.15 MMLU-Pro; 0.8 ATP + 0.2 MLM 41.83 / 14.74; 0.8 ATP + 0.2 dummy 43.06 / 17.16), replicated over seeds, hypothesised to be dropout-like label regularisation. Footnote 1: 15-20 % ratios similar, 10 % or 25 % worse. Footnote 7: dummy examples help the instruct model's generalisation but not task-specific fine-tuning.
- **Inference template** (Sec. 2.5.2): instructions, text, option list, `Answer: [unused0] [MASK]`. The paper does not report learning rate, steps, or batch size for the FLAN phase.

## Experiments and results
- **Zero-shot** (Table 2; MMLU, MMLU-Pro, ADEv2, NeurIPS Impact Statement, One Stop English). ModernBERT-Large-Instruct: MMLU 43.06, MMLU-Pro 17.16, ADEv2 53.31, NIS 85.53 (all four confirmed by Table 4), OSE 20.62. By column reconstruction the decoder MMLU baselines are SmolLM2-360M 35.8, Qwen2-0.5B 33.7, Llama-3.2-1B 45.83, SmolLM2-1.7B 48.44, Qwen2.5-1.5B 59.67. Note the sub-1B decoder baseline is Qwen2-0.5B, not Qwen2.5-0.5B as QUESTIONS.md states. UniMC (RoBERTa-based, custom attention mask) beats it on MMLU-Pro and OSE; it beats every zero-shot baseline on ADEv2 and NIS.
- **Fine-tuned** (Table 3): MLM-head 84.67 average vs classification head 84.39; gains on fine-grained tasks (SST-5 61.13 vs 59.28, MNLI 91.03 vs 90.8), small loss on SST-2 (96.22 vs 97.1). Sweep: epochs {1,2,3}, LR {2e-5, 3e-5, 5e-5}.
- **Backbone ablation** (Table 4): same recipe on GTE-en-MLM-Large gives MMLU 36.69 / ADEv2 20.26; on RoBERTa-Large 33.11 / 16.24; averages 47.8 vs 35.44 vs 26.44.

## Limitations
Single-token outputs only; no few-shot or in-context evaluation ("we leave this to future work"), and the cited Samuel 2024 result says MLM in-context learning only appears at 900 M+ parameters; FLAN is dated; no 1B+ encoder existed to test scaling; instruct-phase hyperparameters unreported; MMLU-Pro is scored with listed options and letter verbalizers, which is symbol scoring with its known selection bias.

## Relevance to this workspace
- **MODEL-2 (arm F).** This is the recipe. Multi-token answers are handled by listing options and predicting a letter at one mask, never by multiple masks. For our 8-way type questions the type names must be single ModernBERT BPE tokens (with or without leading space, matching the training template) or be replaced by letters. The `[unused0]` anchor and 20 % dummy mix should be copied verbatim. Fine-tuning LRs 2e-5 to 5e-5, 1-3 epochs.
- **EVAL-1.** Mask-position scoring gives a proper distribution over verbalizers, so PMI calibration is one extra forward pass with the entity removed. Letter verbalizers would reintroduce the position bias that our current cloze scoring avoids.
- **MODEL-3.** The 43.06 vs 33.7 MMLU comparison is the evidence that a 0.4 B encoder stores and retrieves world knowledge at least as well as a 0.5 B decoder, but the paper never tests injecting new facts.
- **Label induction (Timmy).** Genuinely open: the paper does not test ICL, and the cited evidence says it is size-gated for MLMs.

## Key references worth following up
2406.04823 (Samuel, BERTs are generative in-context learners); 2412.13663 (ModernBERT); 2210.11416 (Flan-T5 scaling, from memory); 2301.13688 (FLAN collection, from memory); UniMC (Yang et al., EMNLP 2022; 2210.08590 from memory); PET 2001.07676 and 2009.07118 (from memory); 2406.01574 (MMLU-Pro).
