# Literature Summary: Vocabulary Expansion and Knowledge Injection for LLMs

> Superseded on 2026-09-14 by `references/SURVEY.md` (34 papers read in full, mapped to question
> IDs). Kept for the vocabulary-expansion mechanics in Topic 1, which the survey does not repeat.

Compiled 2026-09-12 by a research subagent (web search + paper fetches). Framing: teach a model a large merchant/store database (what each sells) so that transaction categorization improves and the knowledge transfers to other tasks.

---

## Topic 1: Adding new vocabulary/tokens to an existing model

**Mechanics.** In HF Transformers you call `tokenizer.add_tokens([...])` and then `model.resize_token_embeddings(len(tokenizer))`; new rows are randomly initialized by default ([HF docs](https://huggingface.co/docs/transformers/en/main_classes/tokenizer), [walkthrough](https://www.depends-on-the-definition.com/how-to-add-new-tokens-to-huggingface-transformers/)). For sentence-transformers the same applies to the underlying `Transformer` module; the pooling/normalization layers are unaffected, and you then need contrastive fine-tuning so the new rows become meaningful ([sbert modules docs](https://sbert.net/docs/package_reference/sentence_transformer/modules.html)). No peer-reviewed paper was found specifically on adding tokens to sentence-embedding models; the practice is documented only in blogs/docs.

**Initialization strategies (roughly in order of sophistication):**
- *Random*: Hewitt shows why this is dangerous: a near-zero embedding yields logit 0, which can dominate a softmax whose other logits are large and negative, so the model assigns high probability to the new token regardless of context ([Hewitt 2021](https://www.cs.columbia.edu/~johnhew/vocab-expansion.html)).
- *Hewitt 2021*: sample new embeddings from N(mu, Sigma) where mu, Sigma are the mean and covariance of existing embeddings; this bounds the KL divergence between pre- and post-expansion models by log(1 + 1/n) ([Hewitt 2021](https://www.cs.columbia.edu/~johnhew/vocab-expansion.html), [code](https://github.com/john-hewitt/embed-init)).
- *Mean-of-subword-embeddings*: initialize the new token as the mean of the embeddings of the pieces the old tokenizer would have split it into; widely used baseline ([Medium tutorial](https://medium.com/everyday-ai/add-extra-new-tokens-to-pre-trained-llm-tokenizer-05b0a89f058f)).
- *WECHSEL* (Minixhofer et al., NAACL 2022): initialize target-tokenizer embeddings using aligned static multilingual word vectors so they are semantically close to source tokens; transferred RoBERTa/GPT-2 to new languages with up to 64x less compute than training from scratch ([arXiv](https://arxiv.org/abs/2112.06598v2), [ACL](https://aclanthology.org/2022.naacl-main.293/)).
- *FOCUS* (Dobler & de Melo, EMNLP 2023): represent each new token as a sparsemax-weighted combination of overlapping tokens selected by similarity in an auxiliary static embedding space; beats random and prior methods on LM loss ([arXiv](https://arxiv.org/abs/2305.14481)).
- *OFA* (Liu et al., NAACL 2024): injects alignment from multilingual static vectors and optionally factorizes the embedding matrix to cut parameters ([arXiv](https://arxiv.org/abs/2311.08849)).
- *ZeTT* (Minixhofer et al., NeurIPS 2024): a hypernetwork trained over randomly sampled tokenizers predicts embeddings for any new tokenizer; remaining gap closed with <1B tokens of continued training ([arXiv](https://arxiv.org/abs/2405.07883)).
- *Token Distillation* (Dobler, Elliott, de Melo 2025): distill the representation the model produces under the *original* tokenization into the new token's embedding; reported to beat strong baselines ([arXiv](https://arxiv.org/pdf/2505.20133)).
- Sobering comparison: Mundra et al. (CoNLL 2024) found that simple multivariate (Hewitt-style) initialization performs on par with advanced cross-lingual methods, and that initializing inside the convex hull of existing embeddings is what matters ([arXiv](https://arxiv.org/abs/2407.05841)).

**Continued pretraining is required.** Every method above is a warm start; Chinese-LLaMA added 20k Chinese tokens and then did secondary pretraining plus instruction tuning ([Cui et al. 2023](https://arxiv.org/abs/2304.08177)). ZeTT and FOCUS both report that continued training closes the remaining gap.

**When expansion helps vs hurts.**
- Helps when the base tokenizer is grossly inefficient (e.g., 3-4 byte tokens per Chinese character) ([Cui et al.](https://arxiv.org/abs/2304.08177)).
- Dagan et al. (ICML 2024) find that swapping to a specialized tokenizer pays off in speed/context size but only when fine-tuning on more than ~50B tokens ([arXiv](https://arxiv.org/abs/2402.01035)).
- Zhao et al. (LLaMA Beyond English) found vocab extension *hurt*: 0.5B tokens of further pretraining on the original vocab beat an extended vocab pretrained on 30B tokens; they conclude extension "might not be a suitable choice for small-scale incremental pretraining in the order of tens of billions" of tokens ([arXiv](https://arxiv.org/html/2401.01055)).
- Tao et al. (NeurIPS 2024) show compute-optimal vocab size grows with model size, but this is a *from-scratch* scaling law, not evidence for post-hoc expansion ([arXiv](https://arxiv.org/abs/2407.13623)).

**Practitioner takeaway for merchant names.** Merchant strings like "SQ *BLUE BOTTLE" are already tokenizable as subwords; nothing in the literature suggests you need new tokens for them. Expansion only makes sense for a small set of extremely frequent atomic symbols (e.g., category IDs) and even then requires meaningful continued training. Default: do not expand; teach the *meaning* of merchants via data (Topic 2) and/or retrieval.

---

## Topic 2: Injecting knowledge so it generalizes

**Knowledge must be stored in an extractable form.** Allen-Zhu & Li (Physics of LMs 3.1) show that a model trained on one biography per person can memorize yet achieve 0% QA extraction "regardless of subsequent instruction fine-tuning"; extraction requires knowledge augmentation (paraphrase, sentence shuffling, translation) during pretraining, and they recommend mixing instruction-style data into pretraining rather than only after ([arXiv](https://arxiv.org/abs/2309.14316)). The reversal curse compounds this: training on "A is B" does not yield "B is A" ([Berglund et al., ICLR 2024](https://arxiv.org/pdf/2309.12288)). For merchants: include both directions ("Blue Bottle sells coffee" and "a coffee shop in Oakland is Blue Bottle") and many surface forms of each merchant string.

**Plain fine-tuning underperforms RAG.** Ovadia et al. find RAG consistently beats unsupervised fine-tuning for both existing and new knowledge, and that "numerous variations of the same fact" help fine-tuning but do not close the gap ([arXiv](https://arxiv.org/abs/2312.05934)). Soudani et al. report RAG wins "by a large margin particularly for least popular factual knowledge", exactly the long-tail merchant regime ([arXiv](https://arxiv.org/abs/2403.01432)). Gekhman et al. (EMNLP 2024) show unknown facts are learned slower and, once learned, "linearly increase the model's tendency to hallucinate" ([arXiv](https://arxiv.org/abs/2405.05904)).

**Synthetic augmentation makes fine-tuning work.**
- *EntiGraph* (Yang et al. 2024): extract entities and generate text connecting sampled entity pairs, expanding 1.3M source tokens to 455M; closed-book QuALITY accuracy rose from 39.5% (base Llama-3-8B) to 56.2%, with log-linear scaling in synthetic tokens, while a Rephrase baseline scaled poorly and raw CPT was below base. Gains compound with RAG (62.6% vs 60.4% for base+RAG) ([arXiv](https://arxiv.org/html/2409.07431)). This is the most direct template for a merchant graph (merchant-category-location-product relations).
- *Instruction Pre-Training* (Cheng et al., EMNLP 2024): augment raw corpora with synthesized instruction-response pairs; in continual pretraining Llama3-8B became comparable to Llama3-70B ([arXiv](https://arxiv.org/abs/2406.14491)).
- *Self-Tuning* (Zhang et al. 2024): self-supervised memorization/comprehension/self-reflection tasks over documents; ~20% EM gain on extraction ([arXiv](https://arxiv.org/abs/2406.06326)).
- *Active Reading* (Meta 2025): models generate their own study strategies; 8B experts reached 66% on Wikipedia SimpleQA (+313% relative vs vanilla FT) and 26% on FinanceBench (+160%) ([arXiv](https://arxiv.org/abs/2508.09494)).
- *Prompt (context) distillation* (Kujanpaa et al. 2024): a teacher with the document in context generates targets for a student without it; outperforms SFT and can surpass RAG ([arXiv](https://arxiv.org/abs/2412.14964)).
- *SEAL* (2025): RL-trained self-edits for test-time weight updates; 47% knowledge incorporation, beating GPT-4.1 synthetic data in single-passage settings ([alphaXiv](https://www.alphaxiv.org/overview/2506.10943v2)).

**Full FT vs LoRA and forgetting.** Biderman et al. (TMLR 2024): LoRA underperforms full FT on the target domain but forgets less than full FT and beats weight decay/dropout as a regularizer ([arXiv](https://arxiv.org/abs/2405.09673)). Shuttleworth et al. show LoRA introduces "intruder dimensions" that accumulate across sequential tasks ([arXiv](https://arxiv.org/pdf/2410.21228)). Mitigations: LR re-warm/re-decay plus replay of pretraining-like data matches full retraining at 405M-10B scale ([Ibrahim et al. 2024](https://arxiv.org/abs/2403.08763)); WiSE-FT weight interpolation between base and fine-tuned models preserves OOD robustness ([Wortsman et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.pdf)). Memory layers (Meta, Dec 2024) add sparse key-value capacity with gains "especially pronounced for factual tasks" ([arXiv](https://arxiv.org/abs/2412.09764)); Sparse Memory Finetuning (2025) learned new facts with only an 11% NaturalQuestions drop vs 89% for full FT and 71% for LoRA ([arXiv](https://arxiv.org/abs/2510.15103)).

**Hybrid: fine-tune to use retrieval.** RAFT trains the model to answer with retrieved documents while ignoring distractors and quoting relevant spans, improving domain RAG on PubMed/HotpotQA/Gorilla ([arXiv](https://arxiv.org/abs/2403.10131)).

**Recommendation for the merchant use case.** (1) Keep the vocabulary. (2) Use RAG over the merchant DB as the primary knowledge source; long-tail merchants are where parametric injection is weakest. (3) If you want parametric transfer, run EntiGraph/Active-Reading-style synthetic continued pretraining with bidirectional, multi-surface-form facts plus mixed-in instruction data and replay, at low LR; consider WiSE-FT averaging. (4) Layer RAFT-style training so the model uses retrieved merchant records well. (5) Evaluate hallucination on held-out unknown merchants per Gekhman.

Unverified: no paper was located on token addition specific to sentence-transformers; the Chinese-LLaMA paper reports its own success but no ablation against no-expansion, so the "helps vs hurts" evidence rests on Zhao et al. and Dagan et al.
