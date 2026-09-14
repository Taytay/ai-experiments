# Physics of Language Models: Part 3.2, Knowledge Manipulation

- arXiv 2309.14402, v2 (16 Jul 2024; v1 Sep 2023) - https://arxiv.org/abs/2309.14402
- Zeyuan Allen-Zhu (Meta / FAIR), Yuanzhi Li (MBZUAI)
- Venue: preprint in the extracted text (v2 adds Llama/Mistral and larger data)
- Source: references/papers/2309.14402/paper.txt

## One-paragraph summary
Starting from models that extract biography attributes near-perfectly (pretrained on bioS multi5+permute), the paper asks whether they can *use* stored knowledge for four operations: partial/dual retrieval, classification ("born in an even month?"), comparison ("born later than B?"), and inverse search ("who was born on ...?"). Retrieval works, but partial retrieval of later tokens (birth year) is poor. Classification and comparison fail without chain-of-thought (CoT) at both training and inference, even with 25k-2.5M fine-tuning samples for tasks with tiny theoretical sample complexity; including CoT in training does not help non-CoT inference, and fine-tuning for extraction first does not help manipulation. Inverse search is essentially 0% under every training regime, data augmentation, model size and data size, unless the pretraining data already presents attributes before the name. GPT-3.5/4 and Llama-3 show the same pattern on real data (Figs. 2, 5, 7, 9).

## Problem
Distinguish true knowledge manipulation (the model combines "A's attribute is T" with "T has property f") from memorized equivalents in the training set. Synthetic biographies make the distinction controllable: the manipulation QA appears for Ptrain, the test asks about Ptest whose biographies were seen but never questioned.

## Method
- Base models: the Part 3.1 checkpoints pretrained on bioS multi5+permute (100% birth date extraction, 98% major); either BIO-pretrained only or additionally QA-finetuned for single extraction.
- LoRA fine-tuning (rank 8/16 q/v, rank 128 embedding, 50k steps, batch 48) on the manipulation QAs for Ptrain; OOD generation accuracy on Ptest. Full FT was worse and dropped (footnote 10).
- "Train with hint": with probability 0.5 the answer is preceded by the relevant attribute values (CoT); test with and without hint.
- Inverse search: 10 tasks (birth date to first/full name, ..., all attributes to name), both QA finetune and BIO+QA mixed training (QAr 0.5/0.8), plus four "reverse" datasets that move the name later in the entry.
- Scale checks: GPT2/Llama/Mistral up to 5.5x GPT2-small on bioS 10x-50x (N up to 5M) with full augmentation (Figs. 10d, 13, 14).

## Experiments and results
- Result 1 (Fig. 3 middle): dual retrieval is near-perfect on multi5+permute; on multi5+fullname (fixed order) asking company city before company name drops accuracy sharply because city is determined by name and always follows it.
- Result 2 (Fig. 3 left): birth year retrieval is poor even when the full date is perfect; the model cannot skip to later tokens. GPT2 tokenizes years as one token, Llama/Mistral as four (Remark B.1), which changes partial-retrieval numbers.
- Result 3 (Fig. 4/11): even-month classification without hint: 60.4% at 2.5k, 75.9% at 10k, 95.3% at 50k training people (chance 50%). Month ranking: 85.6% at 50k. Major ranking (100 majors, chance ~50.5%): 52.2% at 25k, 53.9% at 50k; on Mistral 5.5x with 2.5M samples 69.7% (Fig. 13). Major subtraction stays at 1.1-1.2% (chance 1%) through 50k and 1.2% at 2.5M.
- Result 4: trained with hint, tested without hint: even-month 10k goes 75.9% to 80.3%; with hint at test 94.2%, tracking hint accuracy 91.0% (footnote 14 gives the composition formula).
- Result 5: BIO-pretrained vs QA-finetuned base differ by about a point everywhere in Fig. 4.
- Result 6 (Fig. 5): GPT-4 on 4,779 WikiBio celebrities: birth date extraction 99%, even-month classification 50.7% correct, birth-order comparison 52.3% (born 1900-1910), 71.1% (1900-1950), 81.6% (all pairs).
- Result 7 (Fig. 6, 14): inverse search test accuracy near 0 for all 16 augmented datasets, both regimes, all sizes; only the `reverse` datasets give non-trivial accuracy.
- Result 8 (Fig. 7): GPT-4 next-sentence in Pride and Prejudice 65.9% vs previous-sentence 0.8%; WikiBio inverse 42% vs forward 99%; Chinese idiom first character 17.6% vs last character 90.6%.
- Result 9: mitigations are CoT data, RAG, reversal rewriting, line numbers.

## Limitations
Synthetic, from-scratch models; the extraction-then-manipulation pipeline uses LoRA only; manipulation functions are simple (parity, ranking, subtraction) and the negative results are about sample efficiency rather than impossibility (large samples do eventually work for parity). GPT-4 results are illustrative and contamination cannot be excluded.

## Relevance to this workspace
- L1 vs L2/L3 ladder: L1 recall is the easy case; any level that is a *function* of a stored attribute (category from recalled products, "which species share a weakness", comparisons) is knowledge manipulation and, per Results 3-5, will not be learned from few examples without CoT-style training data. Result 5 predicts that raising L1 recall (e.g. more templates) will not move manipulation levels.
- DATA-2: the products-to-category bridge is a classification of a stored attribute. The paper predicts the bridge works only if the category is stored directly (it is, in our templates) or the model states products first; test with a CoT-format episode variant ("Products: ...; Category: ...").
- EVAL-7 and the `reverse` task: inverse search is ~0 unless reversed text is in training. Our reversal-curse result is expected; the remedy is reverse-rendered knowledge texts ("Canned goods are sold by X"), which is a DATA-5 augmentation, not a training-recipe change.
- EVAL-2 / MODEL-4: Result 2 shows later tokens of a multi-token attribute are conditionally near-deterministic; that is the mechanism behind mean-per-token length bias, and tokenizer differences (single vs multi-token) change the numbers.
- TRAIN-1: mixed training with QAr 0.5-0.8 is their strongest regime and still fails inverse search; no mixing ratio fixes a missing data direction.
- EVAL-4: in-context induction ("Timmy") is in-context reasoning, which the authors explicitly exempt from these negative results; the ladder should keep in-weights manipulation and in-context induction as separate axes.

## Key references worth following up
- 2309.12288 Berglund et al., the reversal curse
- 2403.13799 Golovneva et al., reverse training to nurse the reversal curse
- 2403.00758 Guo et al., semantic-aware permutation training
- 2404.19737 Gloeckle et al., multi-token prediction (partial retrieval)
- 2404.05405 Part 3.3 capacity scaling laws
- 2404.15758 Pfau et al., hidden computation with filler tokens
