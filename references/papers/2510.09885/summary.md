# Diffusion-Inspired Masked Fine-Tuning for Knowledge Injection in Autoregressive LLMs

- arXiv 2510.09885 v6, 9 Jun 2026 (first version Oct 2025) - https://arxiv.org/abs/2510.09885
- Xu Pan, Ely Hahami, Jingxuan Fan, Ziqian Xie, Haim Sompolinsky (Harvard; UTHealth Houston; Hebrew University)
- Venue: preprint
- Source: references/papers/2510.09885/paper.txt

## One-paragraph summary
The paper asks why autoregressive LLMs (arLLMs) need paraphrase augmentation to turn fine-tuned documents into answerable knowledge, and why they suffer the reversal curse, while masked-diffusion LLMs (dLLMs) do not. It shows that LLaDA-8B learns forward and backward QA from raw documents without paraphrases, then transfers the mechanism to ordinary decoder models by "masked fine-tuning": the training example becomes a chat turn in which the user shows the document with a random fraction of tokens replaced by a mask token and asks for the recovered passage, and the assistant emits the original document (loss on the assistant tokens only). With this objective, Llama-3.1-8B, Llama-3.2-3B, Qwen2.5-7B and Qwen3-4B reach 0.9+ forward and backward accuracy on three datasets with no paraphrases, beat knowledge-editing baselines (which score near zero), and the same objective gives small gains on a 1.2M-sample knowledge corpus (GPQA-diamond) and on GSM8K/MATH SFT.

## Problem
Knowledge injection by fine-tuning on documents fails to generalise to QA unless the documents are paraphrased many times, and it fails almost completely on questions whose cue-to-answer order is reversed relative to the training text (Sec. 1-2). Paraphrasing costs LLM calls, and reordered-sequence augmentation breaks grammar. The authors hypothesise (H1) that dLLMs avoid both problems and (H2) that the demasking objective, not the architecture or decoding, is what matters.

## Method
Three datasets: NameDescription (60 fictitious name/description statements, 30 paraphrases each), Biography (100 six-sentence bios, 5 paraphrases), and a new Wiki set (94 articles about 2025 events, 10 same-order and 10 permute-order GPT-o3-mini paraphrases). Evaluation is open-ended generation scored by ROUGE-1 against the gold answer, with forward and backward question types (Sec. 3, App. A.3). Baseline arLLM fine-tuning wraps the document in a generic chat instruction ("Tell me a fact."). Masked fine-tuning samples a mask ratio t ~ U(0.05, 0.95) per batch, replaces tokens with a reserved token (Llama id 128013; Qwen uses "[]", id 1294), and trains with the standard AR loss on the assistant reconstruction (Eq. 2, Fig. 2). All runs are full-parameter fine-tuning on 4xH100, batch 64, Adam with wd 0.1, betas (0.9, 0.95), 2% warmup, LR 5e-6 for arLLMs, 1e-5 for the dLLM, 3e-6 to 5e-6 for masked variants; bf16 mixed precision gave about 30% better accuracy than the alternative (App. A.4). Reported numbers are the best checkpoint by macro-averaged forward/backward accuracy.

## Experiments and results
- Paraphrase dependence (Table 1, mean over four arLLMs): NameDescription forward 0.264 without paraphrases vs 0.954 with; backward stays at 0.029 / 0.031. Biography forward 0.153 vs 0.975, backward 0.002 either way. Wiki forward 0.464 vs 0.674, backward 0.309 vs 0.391. Permute-order paraphrases are the only augmentation that lifts backward accuracy (Fig. 6).
- dLLM (Table 2): LLaDA without paraphrases reaches 0.869/0.852 (NameDescription fwd/bwd), 0.892/0.696 (Biography), 0.908/0.778 (Wiki); paraphrases add only a few points.
- Masked arLLM (Fig. 3, Table 5): masked Llama-3.2-3B without paraphrases scores N2D fwd/bwd 0.887/0.932, D2N 0.992/0.933, Biography 0.967/0.738, Wiki 0.970/0.908, versus plain Llama-3B with paraphrases at 0.951/0.040, 0.967/0.025, 0.988/0.001, 0.622/0.334. Masked Qwen3-4B without paraphrases: 0.928/0.902, 0.950/0.967, 0.944/0.623, 0.907/0.870.
- Control (Fig. 14): replacing the masked document in the prompt with random tokens collapses accuracy to the naive-fine-tuning level, so the gain is not generic input augmentation.
- Knowledge editing (Table 6): DocTER+ROME, UnKE and AnyEdit are near zero everywhere (e.g., ROME Llama 8B NameDescription fwd 0.019).
- Mask ratio (Fig. 4): fixed t = 0.5 or 0.75 matches random t; t = 0 fails completely.
- Scale (Table 3): Qwen3-4B on Webscale-RL, GPQA-diamond: base 0.242, CPT AR 0.278, CPT masked 0.293, SFT AR 0.359, SFT masked 0.389. Math (Table 4): Llama 3B GSM8K 0.686 base / 0.686 SFT / 0.735 masked; Qwen 4B 0.591 / 0.776 / 0.789.
- Cost (Table 10, 8B Wiki): 199.3 vs 91.9 TFLOPs per step and 51.1 vs 37.9 GB peak memory, but convergence rate more than 2x faster, so total compute is comparable. Two-shot evaluation (Table 9) and chat-template ablation (Table 8, mean effect +0.038) leave the conclusions unchanged.

## Limitations
Only Instruct checkpoints of 3B-8B are used; there is no 0.5B result, no base-model result and no LoRA result. Checkpoints are selected on the test metric. Datasets are tiny (60-100 documents). Forgetting on general benchmarks is not measured ("minor degradation" is asserted in Sec. 5). ROUGE-1 is a lenient correctness measure. Seed variance is visible (Fig. 13). The method presupposes an instruction-following model that can execute "recover the masked passage".

## Relevance to this workspace
- LIT-1: this is the masked-FT paper already flagged; it is now read.
- DATA-5: the paper attributes "QA capability saturates at around 10 paraphrases per sample" to Ovadia 2023 and Mecklenburg 2024 (Sec. 2.1), and offers masked FT as a route that needs no paraphrases at all. A masked-FT arm at 1 template per entity is the cleanest test of "diversity vs distinct sequences".
- EVAL-7 / reversal curse: our `reverse` merchant task is a backward question. Plain AR training with same-order templates should stay near zero on genuinely backward items; masked FT predicts a large jump.
- TRAIN-4: full FT with LR 5e-6, wd 0.1, bf16 mixed precision; note their 30% mixed-precision effect when we retry 3B full FT.
- EVAL-3: their evaluation is generation-based; ours has none.
- Cost on one 24 GB GPU: a masked prompt roughly doubles each ~25-token knowledge text plus a fixed instruction (~70 tokens), so arm C's 1.25M tokens becomes ~2.5-3M, about 50-60 min on 3B LoRA at 851 tok/s, trivial on 0.5B full FT. Requires Qwen2.5-*-Instruct checkpoints (MODEL-1 extension).

## Key references worth following up
Berglund et al. 2309.12288; Golovneva et al. reverse training 2403.13799; Ovadia et al. 2312.05934; Mecklenburg et al. 2404.00213; Kitouni et al. factorization curse (NeurIPS 2024); Prabhudesai et al. 2507.15857; Lampinen et al. 2505.00661; Zhao et al. 2503.05919; Pan et al. 2504.21239.
