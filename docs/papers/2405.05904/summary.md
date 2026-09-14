# Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?

- arXiv 2405.05904, v3 (1 Oct 2024) - https://arxiv.org/abs/2405.05904
- Zorik Gekhman, Gal Yona, Roee Aharoni, Matan Eyal, Amir Feder, Roi Reichart, Jonathan Herzig (Technion, Google Research)
- Venue: not printed in the extracted text (camera-ready with reviewer acknowledgments; EMNLP 2024 per the reader's recollection)
- Source: docs/papers/2405.05904/paper.txt

## One-paragraph summary
The paper tests the conjecture that supervised fine-tuning on facts the model does not already know teaches it to hallucinate. It introduces SliCK, a four-way categorization of (question, answer) pairs by how reliably PaLM 2-S base produces the answer under 10 random 4-shot prompts (greedy plus 16 samples at T=0.5): HighlyKnown, MaybeKnown, WeaklyKnown, Unknown. Fine-tuning datasets of fixed size (6,142 closed-book ENTITYQUESTIONS items) are built with 0-100% Unknown examples. Higher Unknown share monotonically lowers test accuracy on known facts (Fig. 3a); Unknown examples are fitted far slower than Known ones (Fig. 1), are neutral at the early-stopping point but harmful at convergence (Fig. 3b), and a linear model with R^2 0.86 shows each fitted Unknown example hurts about as much as a fitted Known example helps (Table 1). Fine-tuning purely on HighlyKnown is not optimal; MaybeKnown examples give the best overall accuracy (Table 2). Relabeling Unknown examples as "I don't know" removes the overfitting penalty (Table 3).

## Problem
When SFT data contains facts outside the model's pretraining knowledge, does the model learn them, ignore them, or learn to answer ungrounded? The difficulty is deciding per example whether the fact is known; prior work used P(True) or fake facts.

## Method
- Model: PaLM 2-S base, full fine-tuning, lr 1e-5, batch 128, dropout 0.05, 50 epochs with per-epoch dev evaluation; EARLY_STOP is the best dev epoch (5-10 epochs), CONVERGENCE is 50 epochs.
- Data: ENTITYQUESTIONS, 12 randomly sampled relations for train/dev/test, 7 disjoint relations for an OOD test set; no subject/object overlap between train and test; examples with multiple correct answers removed.
- SliCK: PCorrect(T=0) is the fraction of 10 greedy answers matching (exact match); PCorrect(T>0) the fraction of 160 samples. Unknown means never correct. Train split is 24% HighlyKnown, 23% MaybeKnown, 17% WeaklyKnown, 36% Unknown. Annotation cost: 170 inference calls per example, over 15M total.
- Variants: X% Unknown with relation distribution held fixed; ablation D_Known drops the Unknown items; single-category datasets D_HighlyKnown etc.; D_IDK relabels Unknown answers as "I don't know".
- Significance: 100 random test splits, paired t-tests (Appendix J).

## Experiments and results
- Fig. 3a: test accuracy falls with %Unknown at every duration; EARLY_STOP is best, CONVERGENCE worst, and the gap between them widens with %Unknown.
- Fig. 3b: at EARLY_STOP, D and D_Known are nearly identical (Unknown neutral); at CONVERGENCE D underperforms D_Known by an amount proportional to the Unknown ratio.
- Fig. 1/4: Known examples are fitted within the first epochs; Unknown ones only late; at EARLY_STOP the model has fitted most Known and few Unknown examples.
- Table 1: Accuracy = 36.9 + 7.3 N_kn/|D| - 8.3 N_unk/|D| (R^2 0.86); OOD relations: 36.2 + 3.2 - 3.0 (R^2 0.95). Fine-tuning on Unknown "Where is E1 located?" hurts "Who founded E2?".
- Table 2 (EARLY_STOP / CONVERGENCE, full test set): D_HighlyKnown 40.5/40.0; D_MaybeKnown 43.6/43.2 (best); D_WeaklyKnown 39.2/35.4; D_Unknown 37.5/25.8; D_Natural 43.5/41.8. On the MaybeKnown test subset D_MaybeKnown raises 60.1 to 69.9 with HighlyKnown unchanged (98.7 to 98.4). Accuracy on Unknown test items is at most 3.2%, confirming the label.
- Table 3: D (50% Unknown) drops 43.0 to 38.8 from EARLY_STOP to CONVERGENCE with 100% answered; D_IDK holds 61.8 at both, answering 58.7% then 55.6%.
- Fig. 5: SliCK Unknown items have lower post-FT accuracy than P(True)-thresholded items at equal coverage; fewer than 10 prompts weakens the category.
- Appendix E: lr 1e-4 and 1e-6 give the same conclusions (Fig. 6).

## Limitations
One model, one size, closed-book short-answer QA only; full fine-tuning only (LoRA left to future work, with Biderman et al. cited); no mixing with other instruction tasks; real facts only (Appendix F argues fake facts confound "new" with "update"). The hallucination measure is a test-accuracy drop on known facts, which merges hallucination and forgetting (footnote 17).

## Relevance to this workspace
- TRAIN-3 / TRAIN-4: our knowledge is 100% Unknown by construction (fictional entities). The paper predicts slow fitting followed by late overfitting that damages pre-existing knowledge; a fixed 800 steps with one final eval cannot see this. Evaluate every 200 steps on a held-out *known-facts* set (real-world QA the base model gets right, plus the ICL suite) and pick the stopping point per arm.
- TRAIN-1 / mixing: the literature's ratio advice is about Known vs Unknown, not K/E/R streams. Adding MaybeKnown-style examples (facts the base model half-knows: real merchants, common categories) to the mix is their best-performing recipe and maps onto general-text or known-fact replay, which is absent from every merchant run.
- EVAL-5: their forgetting proxy is closed-book accuracy on facts known before fine-tuning, with OOD relations. Build the analogue: 200 SliCK-labelled real-entity questions scored before and after each arm, with 100-split paired t-tests as in Appendix J (also STAT-2).
- EVAL-3: their whole measurement is greedy generation with exact match; a generation eval is a prerequisite for comparing with this line of work.
- REAL-1: SliCK is the tool for splitting real merchants into known vs unknown buckets before fine-tuning; accuracy by bucket is the natural report.
- BASE-1 / abstention: the IDK relabeling result suggests teaching "unknown merchant" as a label for entities outside the trained set, which also addresses hallucination on the long tail.
- MODEL-1: single-model limitation; our 0.5B vs 3B vs 7B sweep would be a direct contribution here.

## Key references worth following up
- 2403.05612 Kang et al., unfamiliar fine-tuning examples control how LMs hallucinate
- 2402.12847 Jiang et al., instruction-tuned LMs are better knowledge learners
- 2405.01525 Lin et al., FLAME factuality-aware alignment
- 2311.09677 R-tuning (refusing unknown questions)
- 2207.05221 Kadavath et al., LMs (mostly) know what they know (P(True), P(IK))
- Ghosal, Hashimoto, Raghunathan ICML 2024, understanding finetuning for factual knowledge extraction (no arXiv id given)
