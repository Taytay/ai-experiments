# Fine-Tuned In-Context Learners for Efficient Adaptation

- arXiv 2512.19879v1 (22 Dec 2025) - https://arxiv.org/abs/2512.19879
- Jorg Bornschein, Clare Lyle, Yazhe Li, Amal Rannen-Triki, Xu Owen He, Razvan Pascanu (Google DeepMind; Microsoft AI; MakerMaker AI)
- Venue: preprint (ICLR-style formatting with a reproducibility section; no venue stated)
- Source: docs/papers/2512.19879/paper.txt

## One-paragraph summary
The paper asks which adaptation method wins for a single concrete downstream task with 3-150 labelled examples: in-context learning with a frozen model (ICL-Only), fine-tuning on x-to-y pairs (FT-Only), or fine-tuning on k-shot prompts and also prompting with k examples at test time (ICL+FT). ICL+FT matches ICL-Only at 3-10 examples and then scales like fine-tuning, consistently matching or beating both baselines on 23 BBH tasks, the 11 symbol-tuning evaluation tasks, Parity-20 and FLoRes translation, across Gemma-2 2B/9B/27B and Qwen-3 0.6B-4B. Hyperparameters are chosen prequentially: each new example is first scored and then trained on, so all data is used for both validation and training in one run.

## Problem
ICL is sample efficient but plateaus; fine-tuning scales but is weak with few examples and needs held-out data for hyperparameter selection, which is exactly what is scarce. The authors want one method with both properties and a data-efficient selection protocol.

## Method
Sections 3-4, Algorithm 1. For example i, sample min(K, i) context examples from the already-seen data, compute the next-step loss on y_i (the prequential metric), then take E gradient steps on k-shot sequences built from seen data. The training sequence is "Next example:" separated k-shot text with loss on every response y_i in the sequence, context and target alike (Appendix B.1), so K also acts like a batch size. Gemma-2 2B/9B/27B, Adafactor, lr in {1e-4, 2e-4, 3e-4}, epochs in {1, 2, 5, 10, 15}, 5 seeds, 2 SEM error bars; classification uses MAP over answers, generation uses sampled continuations with regex parsing. Qwen-3 (0.6B, 1.7B, 4B; 30B as ICL-Only reference) with AdamW in Appendix D.

## Experiments and results
- BBH, 23 tasks (Table 1, Figures 2-3). After 30 examples: FT-Only 27.3/52.8/62.6, ICL-Only 37.2/57.1/64.6, ICL+FT 55.3/68.7/72.3 for 2B/9B/27B. After about 150 examples: FT-Only 50.7/69.7/72.6, ICL-Only 37.6/56.6/64.2, ICL+FT 67.5/78.7/81.8. Gemini 1.5 Pro ICL-Only is 72.8 and 76.6. ICL+FT on a model beats ICL-Only on a model three times larger, or needs 3-10x fewer examples.
- 11 NLP tasks from Wei et al. 2023 (Table 2, 50-100 examples): FT-Only 77.7/81.2/82.6, ICL-Only 74.9/82.1/82.5, ICL+FT 84.3/85.1/86.4; Wei et al. report 84.4 for symbol-tuned Flan-PaLM-540B, so Gemma-2 2B with per-task ICL+FT reaches that number.
- Parity-20 (Figure 4): Gemma-2 27B ICL-Only under 2% even at 300 shots; FT-Only reaches 100% at about 100 examples; ICL+FT at about 30.
- FLoRes (Figure 4, C.6): marginal gains into Kurdish/Bemba; both fine-tuning variants slightly hurt translation into English.
- Hyperparameter robustness (Table 3/5): switching from per-dataset prequential selection to one global setting costs FT-Only 1.7-9.5 points but ICL+FT -0.3 to 2.7. Prequential vs i.i.d. training (Table 4/6, 9B): FT-Only 52.8 vs 53.2 (30 examples) and 69.7 vs 68.1 (150); ICL+FT 68.8 vs 68.9 and 78.7 vs 78.7.
- Number of in-context examples (Figure 5, C.3): the 0-to-1 step gives most of the gain; plateau or decline beyond about five.
- Flipped labels on BBH Navigate (Figure 15, C.4): ICL-Only struggles; ICL+FT is best; adding an explicit "answers are flipped" instruction at train and test time lifts both fine-tuning methods (Figure 5).
- LoRA (rank 16 on MLP and KV projections, Adam; C.8): FT-Only 52.8 to 59.1 (30) and 69.7 to 70.2 (150); ICL+FT 68.8 to 67.6 and 78.7 to 78.9, i.e. no material change.
- Qwen-3 (Appendix D): 1.7B ICL+FT reaches 58% on Sports Understanding at N=100 vs about 47% for 30B ICL-Only.

## Limitations
Per-task fine-tuning, not a general ICL skill; no cross-task or regression metric is reported (forgetting, perplexity); test sets of 100 examples; classification scored by MAP and generation by regex; the prequential metric cannot be compared to held-out accuracy directly; translation into English got worse; instruction-tuned prompting is minimal, so numbers are not comparable to tuned prompts.

## Relevance to this workspace
- BASE-4 and the missing "fine-tune on the target task" baseline. The paper defines exactly the two baselines we lack for the Timmy task: FT-Only on x-to-y pairs and ICL+FT on k-shot episodes of the real task, with a prequential hyperparameter loop that fits our 160-item levels. It also gives a per-item next-step metric that answers STAT-2 without a held-out split.
- TRAIN-1. Their loss covers every answer in the k-shot sequence, not only the final one. Our episodes place loss on 2-4 final tokens per ~250-token sequence, which is the root of the "85% K by gradient weight" concern; adopting all-answer loss multiplies the E-stream signal by about k+1 at zero sequence cost.
- MODEL-1. ICL+FT works from 0.6B upward, and the 2B model's gain over ICL-Only is the largest of the three sizes (Table 1), so small models benefit most from being trained in the prompt format they are evaluated in; this argues that the 7B run should be cheap to justify but may show a smaller delta.
- TRAIN-2 and TRAIN-4. Global hyperparameters are safe for ICL+FT but not for FT-Only, and LoRA rank 16 is equivalent to full fine-tuning here; both support keeping one shared recipe across arms and trying lower ranks.
- EVAL-5. The Wei et al. 11-task suite (SUBJ, TEH, stance, ADEC, OR, SOT, TOS, TC) is reused here with numbers per model size, giving us an external reference point if we adopt those datasets as the disjoint suite.
- Symbol tuning link. The 2B ICL+FT number (84.3) equalling 540B symbol tuning (84.4) shows per-task training in prompt format substitutes for scale, which is the same lesson as our arm B versus base 3B.

## Key references worth following up
2110.15943 (MetaICL), Chen et al. 2022 in-context tuning (ACL 2022), 2507.04221 (context tuning for in-context optimization), 2404.02060 (long-context LLMs struggle with long ICL), Kossen et al. 2023 (ICLR 2024, label relationships; arXiv id likely 2307.12375, verify), Zhu, Panigrahi, Arora "context-enhanced learning" (no id in the bibliography).
