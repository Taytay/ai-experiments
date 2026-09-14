# Bidirectional LMs are Better Knowledge Memorizers? A Benchmark for Real-world Knowledge Injection (WikiDYK)

- arXiv 2505.12306v1, 18 May 2025 - https://arxiv.org/abs/2505.12306
- Yuwei Zhang, Wenhao Yu, Shangbin Feng, Yifan Zhu, Letian Peng, Jayanth Srinivasa, Gaowen Liu, Jingbo Shang (UCSD, Tencent AI Lab, UW, Cisco)
- Venue: preprint
- Source: docs/papers/2505.12306/paper.txt

> Extraction caveat: the BiLM rows of Table 4 are column-shifted in the plain-text extraction; reconstructed by column counting and cross-checked against Tables 5 and 11.

## One-paragraph summary
WikiDYK is a knowledge-injection benchmark built from 12,290 Wikipedia "Did You Know" facts (Jan 2022 to Apr 2025) with 77,180 generated questions in five types: reliability (recall the bolded entity), generality (recall another slot of the same fact), paraphrase, portability (multi-hop with pretrained knowledge), and locality (pretrained knowledge retained). Continued pre-training on the raw facts with next-token prediction injects essentially nothing (< 2 % match on most CLMs, Table 4) and damages locality; synthetic QA and span prediction help CLMs; but Flan-T5-220M/770M trained with T5 span prediction memorise far more (Flan-T5-770M 46.09 % reliability match on the full set vs 16.09 for the best CLM, Llama-3.1-8B with span prediction). The authors attribute the gap to bidirectional attention rather than the objective, and propose an ensemble of small BiLMs routed by a scope classifier as an external memory for an LLM, raising Flan-T5-220M reliability by 29.1 points.

## Problem
Existing injection benchmarks use noisy Wikipedia snapshots and ambiguous synthetic questions; the field lacks a curated, continuously updated test ground, and whether architecture (causal vs bidirectional) matters for memorisation has not been tested.

## Method
- **Data** (Sec. 3): about 10 expert-curated facts per day; each converted by GPT-4o/4.1/o3-mini prompts (Appendix B) into the five question types; static models score near chance on the injected facts (Table 3: Qwen2.5-7B 0.86 reliability match) while RAG with top-3 articles reaches 20-32.
- **Injection objectives** (Sec. 4): NTP on raw facts; synthetic QA (gpt-4.1-mini generates all possible QAs per fact); span prediction (T5 span corruption, span lengths 1-5, exhaustive enumeration of masked variants); for CLMs the same span task is expressed as a "Predict the masked words" prompt. Upsampling s = 1000 exposures per fact; full fine-tuning below 3 B, LoRA r=32 above; BiLMs are Flan-T5-220M/770M, T5-v1.1-large, RoBERTa-large (10 masks appended at test time).
- **Evaluation format**: free generation after `{question}\nAnswer:`, scored by case-sensitive substring match and token F1 (Appendix A). Not multiple choice.
- **Ensemble** (Sec. 4.4): GMM semantic or temporal clustering, one Flan-T5 per cluster, DeBERTa-v3-large scope classifier, fallback to Llama-3.1-8B.

## Experiments and results
- **Table 4 (full 12,290 facts, s=1000)**, reliability match: NTP 0.45-1.49 for all CLMs; QA 2.29-19.08 (Qwen2.5-1.5B best); SP 1.06-16.09 (Llama-3.1-8B best). Flan-T5-220M 10.06, Flan-T5-770M 46.09; with ensemble 39.16 and 52.82. Paraphrase for Flan-T5-770M 33.25 (ensemble 40.02); portability stays below 7 for everyone. Locality collapses for NTP (Llama-2-7b down 25.62 points).
- **Scaling the number of facts** (Fig. 3, Tables 7-9): at 100 facts CLMs with span prediction are competitive or better (Gemma-3-1B SP 75.00 reliability vs Flan-T5-770M 78.00, Flan-T5-220M 56.00); at 1,000 facts Gemma-3-1B SP falls to 50.00, Llama-3.2-1B SP to 33.00, while Flan-T5-770M holds 73.00; at 3,500 facts 25.00 / 17.00 vs 66.00. RoBERTa-large at 1,000 facts scores 3.00 (Table 8); T5-v1.1-large 52.00.
- **Upsampling** (Fig. 4): gains saturate around 6,000 exposures; locality drops up to 10 points beyond 3,000.
- **Ensemble ablation** (Table 5): semantic clustering improves with more clusters (10 clusters 39.16), temporal clustering fails because of the classifier, not the injection.

## Limitations
No theory; no controlled pretraining of a CLM and a BiLM on the same data (Limitation section); the BiLMs are instruction-tuned encoder-decoders (Flan-T5), not encoder-only MLMs, and the one encoder-only model tested (RoBERTa-large) fails under multi-mask generation; free-generation scoring penalises models that know the answer but format it differently; portability barely moves for anyone.

## Relevance to this workspace
- **MODEL-2.** Strong evidence that span-masked bidirectional training stores facts more reliably than causal fine-tuning, at 220 M-770 M vs 1 B-8 B, on real text, evaluated by generation. Two cautions: (a) the advantage only opens up above ~1,000 facts; at our scale (136 species, 120 merchants) span-prediction CLMs did as well, so arm F's expected win is on capacity (REAL-3), not on the current ladder; (b) the winners are encoder-decoders that emit multi-token answers, which sidesteps ModernBERT-Instruct's single-token constraint. A Flan-T5-large span-prediction arm is a cheap second encoder arm with published precedent.
- **EVAL-1 / EVAL-3.** Their reliability vs paraphrase gap (46 vs 33 for Flan-T5-770M) is the same format-sensitivity we see (100 vs 20.6); the paper measures it by generation and substring match, which is exactly the missing EVAL-3 check.
- **Extractability.** Recall of the injected entity is achievable; multi-hop use of it (portability) is not, matching Allen-Zhu & Li and our L5 results.
- **BASE-2 / REAL-3.** Locality degrades with upsampling; their scope-classifier ensemble is a modular alternative to editing methods.

## Key references worth following up
2309.12288 (reversal curse); 2310.10322 (bidirectional model editing untying the reversal curse); 2405.14862 (Bitune); 2504.00472 (Memorizing is not enough); 2411.07175 (continual memorization of factoids); 2404.00213 (Mecklenburg, SFT injection); 2312.05934 (Ovadia); 2305.01651 (Onoe, new entities from descriptions); 2305.14795 (MQuAKE); 2502.10708 (injection survey).
