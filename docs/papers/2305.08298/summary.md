# Symbol tuning improves in-context learning in language models

- arXiv 2305.08298v2 (30 Dec 2023; v1 May 2023) - https://arxiv.org/abs/2305.08298
- Jerry Wei, Le Hou, Andrew Lampinen, Xiangning Chen, Da Huang, Yi Tay, Xinyun Chen, Yifeng Lu, Denny Zhou, Tengyu Ma, Quoc V. Le (Google, Stanford)
- Venue: EMNLP 2023 (cited as such by the two 2025 papers; the arXiv text carries no venue line)
- Source: docs/papers/2305.08298/paper.txt

## One-paragraph summary
Symbol tuning fine-tunes an instruction-tuned LM on few-shot classification prompts in which the instruction is removed and the natural-language labels are replaced by arbitrary symbols (integers, letter strings, random words), so the only way to solve the prompt is to read the input-label mapping in the exemplars. Applied to Flan-PaLM 8B/62B/62B-cont/540B with 22 datasets and about 30k symbols, it raises accuracy on 11 unseen classification tasks by +5.5 to +15.8 points in the settings without natural labels, improves BIG-Bench list-function and Turing-concept tasks, and restores the ability to follow flipped labels that instruction tuning had removed. Only 1k-2k steps are needed and no instruction data has to be mixed in for models 62B and up; the 8B model, however, loses 6-7 points in settings where natural labels are available, and the paper attributes this to over-fitting to "all labels are arbitrary".

## Problem
Instruction-tuned LMs are not forced to use in-context exemplars because the task is already stated by the instruction and by meaningful label words; they ignore random or flipped labels (Min et al. 2022b; Wei et al. 2023). The goal is a cheap fine-tuning procedure that makes models actually learn input-label mappings from context.

## Method
Section 3. Training prompts are built from 22 HuggingFace classification datasets (NLI, sentiment, paraphrase, commonsense, topic, coreference, misc.; Table 4, 291,693 examples, capped at 25k per dataset), 2-10 exemplars per class, one of 10 input-label templates (Appendix C.2), labels remapped to one of ~30k symbols (1-4 digit integers, 1-3 letter strings, MIT 10k word list; Figure 3). Evaluation uses 11 datasets absent from both symbol tuning and Flan (Table 5, at most 100 examples each, 704 total), k=4 exemplars per class, and a disjoint pool of ~270k symbols (5-digit integers, 3-4 letter strings, 100k word list). Four ICL settings cross instruction present/absent with relevant labels present/absent (Figure 4). Tuning: Flan-PaLM starting points, batch 32, Adafactor, lr 3e-3 (8B, 62B) or 1e-3 (540B), input/target 2048/512 with packing, dropout 0.05-0.1, 4k steps for 8B/62B and 1k for 540B (Table 7). Loss is on the target only (inputs and labels separated by EOS).

## Experiments and results
- Unseen ICL tasks (Table 1, average of 11 tasks, chance 42.4). 8B: 63.9 to 57.6 (-6.3) with instruction+relevant labels, 61.6 to 54.3 (-7.3) relevant labels only, 42.4 to 58.2 (+15.8) instruction only, 44.2 to 52.8 (+8.6) neither. 62B: +1.2, +0.8, +14.4, +9.8. 62B-cont: +1.6, +4.2, +15.5, +11.1. 540B: +2.2, +1.4, +9.3, +5.5. Symbol-tuned 8B (58.2) beats untuned 62B (57.0) when relevant labels are absent.
- Algorithmic reasoning (Figure 5, Table 8): list functions +18.2 (8B, 19.2 to 37.4), +11.1 (62B), +15.5 (62B-c), +3.6 (540B); Turing concepts +15.3 (8B, 62B), +14.1 (62B-c), +4.7 (540B).
- Flipped labels (Figure 6, Table 2): improvements of +26.5 (8B), +33.7 (62B), +34.0 (540B) over Flan-PaLM; symbol-tuned accuracies 53.0/57.5/62.3/54.7 for 8B/62B/62B-c/540B, i.e. only around chance (50), which the authors flag.
- Steps (Figure 7): most change within the first 1k-2k steps; 540B degrades after 1k steps.
- Mixing instruction data (Figures 8, 9): even 16% symbol data gives most of the ICL gain; more symbol data does not change ICL settings but monotonically improves flipped-label following; the paper recommends 100% symbol data for "large-enough" models and notes 8B still drops on relevant-label settings (footnote 4). A.4 confirms no instruction data was mixed in the main runs.
- Number of datasets (Figure 10): more datasets help, 62B more than 8B; with 1-2 datasets performance rises without relevant labels but falls with them.
- Label space (Figure 15): 30k > 3k > 300 > 30 labels, with the largest effect on 8B. Label type (Table 3): words best for flipped labels; for 8B words are worst when relevant labels are available. Randomized (inconsistent) labels give no gain (Figure 16).
- Exemplars per class (Figure 14): the largest advantage is at one exemplar per class, where Flan-PaLM is below chance.
- Benchmarks (Figure 11, Tables 13-21): MMLU 5-shot 49.5 to 47.5 (8B), 59.8 to 58.6 (62B), 65.3 to 64.9 (62B-c), 73.0 to 72.8 (540B); BBH direct 36.2 to 33.0 (8B), BBH CoT 30.5 to 17.7 (8B outlier), larger models within about 1 point. Zero-shot MMLU changes at most 1.7 points.
- Continued instruction tuning for the same steps does not reproduce the gains (Table 2).

## Limitations
Only Flan-PaLM (closed, TPU-scale); the smallest model is 8B, so nothing below that; classification only (no CoT or open generation); flipped-label accuracy still near chance; no perplexity or knowledge-retention metric beyond MMLU/BBH; no seeds or confidence intervals; single label-symbol scheme; evaluation datasets are small (50-100 items).

## Relevance to this workspace
- MODEL-1. 8B is the paper's small regime: symbol tuning helped only when natural labels were absent and cost 6-7 points otherwise, with small MMLU/BBH drops. Nothing below 8B was tested, so our 3B arm B result (ICL-suite symbol 60.4 to 75.5 and natural 84.4 to 87.0 without regression) is not covered by the paper; the natural-label column is the one to watch on 7B.
- Task diversity. They used 22 datasets, 7 task types and 30k symbols; Figure 10 shows that one or two tasks improves symbol settings while hurting natural-label settings, which is our arm Cn pattern (symbol suite 70.8 vs 78.1 with replay). Our single database with 4 grouping attributes plus 4 replay datasets is on the low end; the label-space ablation says a large pseudo-label pool matters most for small models.
- TRAIN-1/TRAIN-2. Mixing ratio was swept by example weight, not loss tokens; 16% symbol data suffices for ICL, but flipped-label following scales with the fraction. Symbol tuning is itself a second stage after instruction tuning with a fresh schedule, and 8B still lost priors; the remedy tested is mixing, not schedule changes. Continued instruction tuning alone gives nothing (A.1).
- EVAL-4. The randomized-label control (A.8) is the right null for "is the Timmy gain induction or format familiarity": episodes with inconsistent labels should give no gain. The k=1 result (Figure 14) warns that one-demo-per-group items mostly test "arbitrary symbols are labels".
- EVAL-5. Their OOD protocol (disjoint symbol pools, datasets absent from all tuning, 10 templates shared, four instruction/label settings) is the template our suite should copy.

## Key references worth following up
2110.15943 (MetaICL), 2202.12837 (Min et al., role of demonstrations), 2303.03846 (larger LMs do ICL differently), 2007.05549 (meta-augmentation), 2211.15661 and 2212.07677 (ICL as gradient descent), 2210.11416 (Flan-PaLM).
