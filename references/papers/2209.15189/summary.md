# Learning by Distilling Context

- arXiv 2209.15189v1 (30 Sep 2022) - https://arxiv.org/abs/2209.15189
- Charlie Snell, Dan Klein, Ruiqi Zhong (UC Berkeley)
- Venue: preprint
- Source: references/papers/2209.15189/paper.txt

## One-paragraph summary
This is the general framework behind prompt distillation: sample raw task inputs x, let the *same* model answer under a rich teacher template (instructions, examples, explanations, scratch-pad), extract the final answer, and fine-tune the model to produce that answer under a minimal student template. Training is driven by the difference in what teacher and student see, not by different weights. The paper shows the method internalizes three signals: abstract instructions and explanations (Natural-Instructions-V2 with an 11B TK-Instruct teacher, Rouge-L 9.0 -> 34.7 against a teacher at 43.4), concrete in-context examples (SPIDER text-to-SQL with Incoder-6.7B, beating gradient descent on the same examples by 8.6 to 9.0 points, Table 2), and step-by-step reasoning (8-digit addition, 0% -> 95% direct answers, Table 3). Simultaneous distillation lets the student absorb more examples than the context window holds, and sequential distillation can overwrite earlier updates. An appendix applies it to fact editing on Counterfact (Table 5).

## Problem
Gains from context tokens (instructions, examples, scratch-pads) vanish when the context is removed and cost inference compute; the goal is to move them into the weights, by analogy to human practice turning working-memory knowledge into habit.

## Method
- Four components (Sec. 2.1): input distribution D (rule-generated, unlabeled pool, or few-shot sampled from the model), teacher template, student template, answer extractor f.
- Objective (Eq. 1): maximize E_{x~D} E_{y~P_teacher(.|T_teacher(x))} log P_student(f(y) | T_student(x)) with teacher weights frozen. Implementation minimizes token-level KL, approximated by an empirical distribution of 100 sampled tokens per position to save memory (Sec. 2.4).
- Variants (Sec. 2.3): simultaneous (sum of losses over K teacher templates), sequential (chain of students), recursive (student becomes the next teacher, Choi et al.).
- Defaults (App. B.3): 4,096 distillation examples, 1 epoch, batch 16, AdamW, LR 1e-5 (TK-Instruct) or 1e-4 (Incoder), 32 TPU-v3 cores.

## Experiments and results
- H1, instructions (Sec. 3.1): student template = raw input only; teacher has task description, 2 positive and 2 negative examples. Rouge-L 9.0 -> 34.7 (teacher 43.4) averaged over 10 NI-V2 tasks; 11.1x fewer inference tokens.
- H2, explanations: the distillation margin correlates with the in-context margin of explanations, r = 0.75, p = 0.01 (Fig. 5).
- H3, task-id association and overwriting (Table 1, Table 6): teacher 81; pre-distill 49 correct / 48 wrong-id; "naive" per-task input distributions give 68 / 61 (the student cheats by keying on the input distribution); "mixed" input distributions give 70 / 16, and re-shuffling ids overwrites the old mapping.
- H4, SPIDER (Table 2): teacher 27.7 / 28.2 (4 / 8 examples); pre-distill 0.3; post-distill 22.1 / 27.9; direct gradient descent on the same examples 13.4 / 18.9.
- H5, beyond the context window: simultaneous distillation of 8 examples where only 4 fit gives 16.2 +/- 0.6 vs 14.22 +/- 0.8 for 4-shot in-context.
- H6, scratch-pad on T5-small (Table 3): teacher with scratch-pad 93; student direct answers 0 -> 95; transfer-learning baseline 72; multi-task 61; 8x fewer inference tokens.
- H7, transfer: TK-Instruct direct addition 1% -> 17% while NI Rouge-L stays 57 -> 58 (10k distillation examples mixed with 65,536 NI examples); synthetic word problems 17% -> 30%.
- App. A.2, Counterfact fact editing (Table 5, score / magnitude): teacher paraphrase 73 / 29, neighborhood 58 / 8; post-distill 79 / 28 and 48 / -2; ROME 89 / 33 and 74 / 4; MEND 65 / 12 and 38 / -12. Paraphrase generalization is recovered; locality (neighborhood) is weak because the teacher itself is weak there.
- App. A.1: distilling a "positive ending" instruction raises sentiment of generations without loss of coherence or entropy (Table 4).

## Limitations
- 2022-era models (T5-11B, Incoder-6.7B, T5-small); no decoder-only chat models.
- KL approximated with 100 samples; no full-vocabulary distillation.
- Student is bounded by the teacher; explanations only help where they help the teacher.
- Fact editing is 34 facts with weak locality; no scaling to hundreds of entities.
- Single runs, few error bars; mixing ratios for forgetting (10k vs 65k) chosen without ablation.

## Relevance to this workspace
- BASE-3: this is the cheapest instantiation. Teacher template = field-guide entry plus question; student template = question; f = identity; targets can be sampled tokens (hard) or top-k logits (soft). Hard-label context distillation is essentially our existing pipeline with answers taken from the with-context model, so it can be added with no new loss code; soft-label KL is the upgrade (see 2412.14964 for why soft beats hard).
- EVAL-4: the task-id experiment is a warning that applies to label induction. With "naive" per-task input distributions the student keyed on the input distribution rather than the task index; only mixed inputs prevented it. Our one-demo-per-group episodes let a model key on "type" rather than the demonstrated rule; build episodes where the same inputs appear under different labelings.
- BASE-2: context-distilled edits matched ROME on paraphrase but not on neighborhood locality, which predicts that distillation will not automatically fix arm A's forgetting; combine with replay.
- TRAIN-2: sequential distillation overwrites earlier updates by design, which is a feature for updating merchant facts and a hazard for staged curricula.
- Cost on one 24 GB GPU: hard-label version is the same cost as an episode arm (about 25 min on 3B for 1.25M tokens); generating teacher answers for a few thousand prompts at short context is minutes. Soft-label version is the 2412.14964 cost (about 2x).

## Key references worth following up
- Askell et al. 2021, arXiv 2112.00861
- Choi et al. 2022, arXiv 2206.11349
- Meng et al. 2022, ROME, arXiv 2202.05262
- Mitchell et al. 2021, MEND, arXiv 2110.11309
- Nye et al. 2021, scratchpads, arXiv 2112.00114
- Wang et al. 2021, zero-label learning, arXiv 2109.09193
