# LoRA Learns Less and Forgets Less

- arXiv 2405.09673 v2, 20 Sep 2024 - https://arxiv.org/abs/2405.09673
- Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, John P. Cunningham (Columbia; Databricks Mosaic Research)
- Venue: TMLR (08/2024)
- Source: references/papers/2405.09673/paper.txt

## One-paragraph summary
On Llama-2-7B in code and math, across continued pretraining (up to 20B tokens) and instruction fine-tuning (73M-103M tokens), LoRA at ranks 16-64 substantially underperforms full fine-tuning; rank 256 targeting all modules closes the gap in IFT but not in CPT. In exchange, LoRA forgets less of HellaSwag/ARC/WinoGrande, ordered by rank, more than weight decay or dropout achieve, and keeps generations more diverse. SVD shows full-FT weight deltas are 10-100x higher rank than typical LoRA. The paper closes with recipe advice: use LoRA for IFT rather than CPT, rank 256 on all modules, alpha = 2r, and sweep the LR in [1e-5, 5e-4] taking the highest stable value.

## Problem
Whether LoRA matches full FT is contested, with earlier claims based on easy tasks. The authors ask under which conditions LoRA approaches full FT on hard target domains and how much it mitigates forgetting (Sec. 1).

## Method
Datasets (Table 1): StarCoder-Python (20B tokens) and OpenWebMath (14.7B) for CPT; Magicoder-Evol-Instruct-110K and MetaMathQA for IFT. Learning metrics: HumanEval pass@1 and GSM8K; forgetting: mean of HellaSwag, ARC-Challenge, WinoGrande (Sec. 3). Each condition trains full FT and LoRA r in {16, 64, 256} on all attention and MLP projections with alpha = 2r and lora_dropout 0.05, after an exhaustive LR sweep per method (App. A, B). Optimizer is decoupled LionW; LRs: code CPT 1e-5 both; math CPT 1e-5 full, 4e-5 LoRA; code IFT 2e-4 (r=16, 64), 1e-4 (r=256); math IFT 1e-5 full, 1e-4 LoRA (5e-5 at r=256). 32xH100.

## Experiments and results
- Code CPT (Fig. 1A, Table S1): LoRA r=256 peaks at HumanEval 0.224 at 20B tokens, roughly full FT at 4B (0.218); full FT reaches 0.263. Code IFT (Table S5): at epoch 4, r=16 0.358, r=64 0.417, r=256 0.498 vs full FT 0.497 at epoch 8. Math CPT (Table S3): r=256 0.203 vs full FT 0.293. Math IFT (Table S7): r=256 0.634 at epoch 8, r=64 0.624 at epoch 4, full FT 0.642.
- Forgetting (Fig. 2, Tables S2/S4/S6/S8): IFT forgets more than CPT and code more than math; forgetting grows with duration and is ordered by rank. Code CPT at 20B: full 0.545 vs r=256 0.617; code IFT: full 0.414 vs r=64 0.509; math CPT: 0.613 vs 0.616; math IFT at epoch 16: 0.559 vs 0.567.
- Tradeoff (Fig. 3): LoRA offers better learning/forgetting Pareto in code, full FT in math; rank is a knob. LoRA r=256 learns as much as full FT with weight decay or dropout while forgetting much less (Fig. 4). Full FT collapses generation diversity; LoRA sits between base and full (Fig. 5).
- Tülu-v2-mix (App. C): LoRA even at r=16 matches full FT on MT-bench, GSM8K and MMLU; at 6 epochs full FT forgets most.
- SVD (Fig. 6, S7): the full-FT delta needs rank >1500 of 4096 to explain 90% variance; rank grows with tokens; MLP deltas are higher rank than attention; first and last layers are lower rank.
- Hyperparameters (Sec. 4.7, Fig. 7, S1, S3): alpha = 2r matters most at high rank; targeting MLP or All beats Attention alone, with MLP driving gains; LoRA needs LRs an order of magnitude above full FT (best 5e-4 code, 2e-4 math IFT) and is more LR-sensitive; LoRA is ~15% slower per token and saves ~40% peak memory at small batch (App. I). Table S14: 7B with Adam needs ~112 GB for states in fp32 vs 15.12 GB with LoRA and bf16 frozen weights.

## Limitations
Single 7B model; code and math only, not factual recall; forgetting measured by three MC benchmarks; LionW rather than AdamW in most runs; SVD only for CPT; the scale question (Sec. 6) is left open; no rank above 256.

## Relevance to this workspace
- TRAIN-4: our r=64, alpha=128 already satisfies alpha = 2r and all-projection targeting; the missing arms are r=256 and full FT on 3B. Knowledge text with full-sequence loss is CPT-like, where the paper says the gap does not close at any rank, so expect the r=256 arm to help recall and the 3B full-FT arm to help most. Their Table S14 arithmetic (16 bytes per parameter with fp32 Adam) is why 3B full FT OOMed; bf16 weights with 8-bit AdamW brings it to roughly 6 bytes per parameter (~18 GB before activations), feasible at 512 tokens with gradient checkpointing. LR sweep 5e-5 to 2e-4 matches their recommended band; they would add 5e-4.
- TRAIN-3: forgetting worsens monotonically with duration in every panel, which supports the every-200-steps evaluation.
- EVAL-5: IFT-style data (our episodes, answer-only loss) forgets more than CPT-style data in their results; the ICL-suite regression in our arms is in line with that.
- STAT-1: they train one model per condition; no seeds, so no guidance on noise.
- MODEL-1: the LR band and rank advice are from 7B; the 0.5B full-FT vs 3B LoRA comparison in our sweep confounds method with scale, and this paper cannot resolve it.
- Cost on one 24 GB GPU: r=256 on all projections of Qwen2.5-3B is on the order of 0.4-0.5B trainable parameters (reader's estimate), about 8-9 GB of adapter, gradient and fp32 Adam state on top of 6 GB of bf16 weights; fits with short sequences. Runtime ~20-25% above arm C.

## Key references worth following up
Kalajdzievski rsLoRA 2312.03732; Kalajdzievski forgetting scaling laws 2401.05605; Ivison et al. Tülu 2 2311.10702; Zhuo et al. Astraios 2401.00788; Hayou et al. LoRA+ 2402.12354; Liu et al. DoRA 2402.09353; Zhang et al. "When scaling meets LLM finetuning" 2402.17193; Zeng & Lee expressive power of LoRA 2310.17513.
