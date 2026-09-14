# Physics of Language Models: Part 3.1, Knowledge Storage and Extraction

- arXiv 2309.14316, v3 (16 Jul 2024; v1 Sep 2023) - https://arxiv.org/abs/2309.14316
- Zeyuan Allen-Zhu (Meta / FAIR), Yuanzhi Li (MBZUAI)
- Venue: ICML 2024 (as cited in the companion paper's bibliography); v3 adds Llama experiments
- Source: docs/papers/2309.14316/paper.txt

## One-paragraph summary
Using a fully synthetic biography corpus (bioS: 100k people, six attributes, ~50 templates per sentence; bioR: Llama-30B-written near-real biographies), the authors pretrain small GPT2/Llama models from scratch and ask whether QA fine-tuning on half the people generalizes to QA about the other half. It does not: with one biography per person, the model memorizes the text token-by-token (99%+ next-token accuracy) yet QA accuracy on unseen people is near zero regardless of model size (up to 682M), passes (1350), LoRA rank or full fine-tuning (Fig. 2). Knowledge augmentation in the *pretraining* data (multiple rewrites, sentence permutation, full-name repetition) raises out-of-distribution QA accuracy from 9.7% to 96.6% (Fig. 3). Linear probes explain why: augmentation makes the model store attributes almost linearly in the hidden state of the name tokens (Q-probing, Fig. 7) and early in the sentence (P-probing, Fig. 5); without augmentation the attribute is tied to preceding attributes rather than to the name. Mixed training (QA data interleaved with biographies) sidesteps the problem (86.6% on unaugmented bioS, Fig. 1). Augmenting only a "celebrity" subset lifts unaugmented "minority" people from 4.4% to 86.8% (Fig. 8). MLM/BERT-style pretraining fails to store multi-word attributes extractably (Fig. 9).

## Problem
Does a model answer factual questions by extracting knowledge from the source text it saw, or because it saw similar questions? Internet-trained models cannot separate the two. The paper builds a controlled setting where the model sees biographies of everyone but QA for only half, and measures OOD QA generation accuracy (exact match, beam 4) on the other half.

## Method
- Data: bioS single (one six-sentence entry, fixed order), 15 augmentations: `multiM` (M rewrites with resampled templates), `permuteP` (P random sentence orders), `fullname` (pronouns replaced by the full name), and combinations; bioR (Llama-generated), `multiM`, `fullname`.
- Models: GPT2 with rotary embeddings, 124M (bioS) / 302M (bioR) / 682M; downsized Llama; GBERT (bidirectional GPT2 with whole-word MLM) for Section 7.
- Regimes: (a) BIO pretrain (80k steps, batch 96, 512 ctx) then QA finetune (full or LoRA on q/v rank 2-32 plus embedding rank 0-128, 50k steps); (b) mixed training with QA ratio QAr (default 0.8, i.e. 8:2 QA:BIO tokens).
- Probes: P-probing (linear head plus rank-2 embedding update, predicting each attribute from the token before each attribute) and Q-probing (linear head on the last hidden state of a name-only input, rank-16 embedding update).

## Experiments and results
- Result 1 (Fig. 1): mixed training reaches 86.6% OOD QA on bioS single and 77.7% on bioR single; the model first learns knowledge from QA of Ptrain then aligns it with BIO. Higher QAr helps, especially on less augmented data (Fig. 10: bioS single from 24.5% at QAr=0.1 to 87.1% at 0.6).
- Result 2 (Fig. 2): BIO pretrain + QA finetune on bioS single: at most ~10% test accuracy for every LoRA/full-FT setting, even at 682M and 1350 passes; in-distribution QA training accuracy 99%. Birth date reaches 33% because it always comes first after the name (Fig. 3).
- Result 3 (Fig. 3, Fig. 7 left): bioS single 9.7%; +fullname 48.9%; +permute1 4.4% (hurts); +permute5 70.0%; multi5 alone 41.0%; multi2+permute 96.1%; multi5+permute 96.6%. More augmentation, higher accuracy. French translation gives about 40% (footnote 15).
- Result 4 (Fig. 5): on bioS single, P-probing accuracy for company name is ~2% until the token right before it; on multi5+permute it is ~100% from the first position. bioS couple (Fig. 6) shows attributes get chained to a preceding correlated attribute.
- Result 5 (Fig. 7): Q-probing accuracy tracks QA finetune accuracy row by row; knowledge is near-linearly stored on the name only when augmented.
- Result 6 (Fig. 8, Fig. 17): adding a 100k celebrity group with multi5+permute lifts minority (single+permute1) QA from 4.4% to 86.8%; bioR single 10.0% to 76.3%; bioR single+fullname+CEL 82.2%. WikiBook text instead of celebrities does not help (7.3%). Format mismatch between celebrity and minority data weakens the effect.
- Result 7 (Fig. 9): GBERT extracts only single/independent-word attributes (birth date, major); multi-word attributes fail even with augmentation and 2x training.
- Fig. 11: LoRA beats full FT for QA finetune; q/v rank barely matters, embedding rank 128 helps with the BIO-to-QA distribution shift.

## Limitations
Models trained from scratch on synthetic data, 124M-682M parameters; no pretrained-LLM continued pretraining (the authors cite Jiang et al. 2402.12847 as confirming on Llama-7B). Generation-only evaluation with exact match. Augmentation counts are coarse (1, 2, 5). Mixed training is "studying for the test"; the fraction of QA-format data needed is explored only through QAr.

## Relevance to this workspace
- EVAL-1 / EVAL-3: the 20.6 vs 100.0 bare-vs-trained-format gap is exactly "memorized but not extractable". Their fix is not more fine-tuning on the same text but (a) more diverse rewrites in the knowledge stream and (b) mixed training with QA-format items for a subset of entities. Concrete arm: add bare-format `Question: What type is X? Answer: Y` for 50% of species to the interleaved mix; score bare recall on the other 50%. Their eval is free generation with exact match, which we lack.
- DATA-5: the diversity that matters is template *and order* diversity; multi5 alone gives 41% while multi2+permute gives 96.1%. Our 14-20 templates are per-entity rewrites without sentence-order permutation across attributes; add permuted multi-attribute sentences. Their data give no saturation point at 10 rewrites; that claim in LIT-1 is not from this paper.
- TRAIN-1 / TRAIN-2: QAr=0.8 by tokens is their best mixed ratio, and OOD accuracy rises with the QA share (Fig. 10). This supports a token-weighted E sweep, and predicts interleaved arms beat sequential ones for reasons beyond the LR schedule.
- TRAIN-4: LoRA beat full FT for extraction; a large embedding-rank update matters more than q/v rank. Try LoRA on `embed_tokens`/`lm_head`.
- MODEL-2: Result 7 is a warning. An MLM encoder stored single-word attributes (type names) but not multi-word ones (categories like `Pharmacy & Health`). Arm F should expect success on type and failure on multi-token labels unless verbalizers are single tokens.
- BASE-3 / diagnostics: Q-probing (linear probe on the name's hidden state) would tell whether arm C's bare-format failure is storage or extraction; cheap to add.
- DATA-1 / EVAL-4: bioS couple shows correlated attributes are chained to each other, not the name; our weakness=rotation-of-type will be stored as a type function.
- REAL-3: Part 3.3 (2404.05405) is the capacity scaling reference for 1k-10k entities.

## Key references worth following up
- 2404.05405 Part 3.3 capacity scaling laws
- 2402.12847 Jiang et al., instruction-tuned LMs are better knowledge learners (confirms on Llama-7B)
- 2309.14402 Part 3.2 (companion)
- 2309.00667 Berglund et al., Taken out of context
- 2104.08696 knowledge neurons; 2012.14913 FFN key-value memories (probing lineage)
