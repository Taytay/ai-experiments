# A Comprehensive Study of Knowledge Editing for Large Language Models

- arXiv 2401.01286v5 (17 Nov 2024) - https://arxiv.org/abs/2401.01286
- Ningyu Zhang, Yunzhi Yao, Bozhong Tian, Peng Wang, Shumin Deng, Mengru Wang, Zekun Xi, Shengyu Mao, Jintian Zhang, Yuansheng Ni, Siyuan Cheng, Ziwen Xu, Xin Xu, Jia-Chen Gu, Yong Jiang, Pengjun Xie, Fei Huang, Lei Liang, Zhiqiang Zhang, Xiaowei Zhu, Jun Zhou, Huajun Chen (Zhejiang University, NUS, UCLA, Ant Group, Alibaba)
- Venue: preprint ("ongoing work")
- Source: references/papers/2401.01286/paper.txt

## One-paragraph summary
A survey plus benchmark. It defines knowledge insertion, modification and erasure (Section 3.2), classifies editors into external-knowledge (IKE, SERAC, MeLLo), merged-representation (T-Patcher, GRACE, LoRA, MELO, REMEDI) and intrinsic weight editing (KE, MEND, MALMEN, ROME, MEMIT, PMET, AlphaEdit) (Table 2), releases the KnowEdit benchmark (six datasets, Table 3) and the EasyEdit toolkit, and evaluates eight methods on Llama2-7b-chat with four criteria: edit success, portability, locality, fluency (Section 3.5, Table 4). The headline finding is that a masked-loss fine-tune of a single causal-traced FFN layer (FT-M) is the best overall method, that every method scores poorly on portability, and that ROME/MEMIT have worse locality than MEND, AdaLoRA or FT-M on this benchmark.

## Problem
Editors had been compared on inconsistent datasets and metrics; the field lacked a unified taxonomy, a standard benchmark covering insertion/modification/erasure, and evidence on whether edits transfer to reasoning and leave general abilities intact.

## Method
- Benchmark: WikiData_recent (post-July-2022 triples, insertion; 570 train / 1,266 test), ZsRE (10,000 / 1,230), WikiBio hallucination correction (592 / 1,392), WikiData_counterfact (1,455 / 885), ConvSent sentiment (14,390 / 800), Sanitation erasure (80 / 80).
- Metrics: Edit Succ = argmax generation equals target (Eqn 10); Portability = alias, compositionality/reasoning, logical generalization including the reversed relation; Locality = answer unchanged on in-distribution neighbours (forgetfulness, relation specificity) and on out-of-distribution general benchmarks (Eqn 11); Fluency = bi/tri-gram entropy.
- Methods: SERAC, ICE ("Imagine that ..." prompt), AdaLoRA, MEND, ROME, MEMIT, FT-L (ROME's single-layer FT maximising target probability at once), FT-M (same layer, standard cross-entropy on the masked target). All run through EasyEdit with greedy decoding.

## Experiments and results
- Table 4, WikiData_recent (insertion) columns SERAC / ICE / AdaLoRA / MEND / ROME / MEMIT / FT-L / FT-M: Edit Succ 98.68 / 60.74 / 100.00 / 95.75 / 97.18 / 97.05 / 55.75 / 100.00; Portability 63.52 / 36.93 / 64.69 / 55.88 / 55.25 / 56.37 / 40.86 / 65.44; Locality 100.00 / 33.34 / 56.42 / 94.76 / 54.77 / 52.15 / 43.70 / 64.33.
- ZsRE: MEMIT Edit Succ 95.37, Portability 52.67, Locality 48.32; FT-M 99.98 / 60.31 / 89.78; MEND locality 92.79.
- WikiData_counterfact: MEMIT 98.05 / 58.56 / 46.62; FT-M 100.00 / 74.36 / 76.76.
- ConvSent edit success below 65 for all; Sanitation: ROME 85.00 success but locality 50.31.
- General tasks after 5 sequential edits (Table 5, CommonsenseQA, PIQA, TriviaQA, XSum, MMLU, AGIEval): all edited models within about one point of the unedited Llama2-Chat except FT-L on TriviaQA, 45.39 to 34.60.
- Continual editing (Figure 4): FT-L, ROME and AdaLoRA all collapse at 1,000 sequential edits; AdaLoRA is stable to about 100.
- Analysis: ROME/MEMIT updates concentrate on a few columns of the value matrix (Figure 6); MEMIT raises Hit@50 of the target in the edited columns from 59.7% to 70.2% (Figure 7); dominant error type is partial token replacement, and 47.3% of FT-L errors are fact-irrelevant words (Figure 5). Causal tracing localises the entity rather than the fact; RSim below 0.6 beyond five layers (Section 5.2, Figure 8-9), echoing Hase et al. that edit success is unrelated to where facts are stored.

## Limitations
One base model (Llama2-7b-chat); Table 4 numbers were revised after bug fixes, and versions differ; AlphaEdit is listed in Table 2 but not benchmarked; no LoRA-after-edit study; sequential editing tested only on three methods; the paper's own Section 7 concedes it is unclear whether edits that shift output distributions "truly constitute successful or useful edits".

## Relevance to this workspace
Informs BASE-2, TRAIN-4, EVAL-5, EVAL-7, BASE-3, LIT-1.
- New subjects: WikiData_recent is by construction knowledge insertion of facts the model never saw; MEMIT reaches 97.05 success but only 52.15 locality and 56.37 portability, while FT-M and AdaLoRA reach 100 with locality 64.33 / 56.42. So on the one benchmark closest to our task, a localised fine-tune matches or beats MEMIT.
- Metric mapping: edit success = L1 trained-format recall; portability's "alias" = our morphology / bank-string variants, "compositionality" = L2-L4 manipulation, "logical generalization" includes the reversed relation = our reverse tasks; in-distribution locality (relation specificity, forgetfulness) is a test we do not have and should add for the five attributes per species; out-of-distribution locality = ICL suite, MMLU, WikiText (EVAL-5). Fluency = EVAL-3 free generation.
- TRAIN-4: FT-M (single mid-layer FFN, masked cross-entropy) is a cheap arm between our full LoRA and MEMIT and would test whether ICL damage comes from touching attention and all layers.
- Reverse tasks: BIRD (bidirectional editing) is the only editor aimed at the reversal curse (Section 3.3.3).
- Qwen: not tested; EasyEdit is the intended vehicle for BASE-2, but Qwen2.5 hyperparameters and layer choices must be checked in the repository, the paper does not cover them.
- Compute: not reported per method.

## Key references worth following up
2305.13172 (Yao et al., editing LLMs: problems, methods, opportunities; source of portability/locality sets), 2307.12976 (Cohen et al., ripple effects, WikiData_recent source), 2308.09124 (linearity of relation decoding), 2008.09036 (Heinzerling and Inui, LM as KB: entity representations and storage capacity, relevant to REAL-3), 2305.09144 (retentive or forgetful memorisation), 2310.10322 (BIRD, reversal curse editing; verify id), 2310.19704 (Mazzia et al. survey), 2312.05934 (Ovadia et al.).
