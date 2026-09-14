# Can Fine-Tuning Erase Edits? On the Fragile Coexistence of Knowledge Editing and Fine-tuning

- arXiv 2511.05852v4 (27 Jun 2026) - https://arxiv.org/abs/2511.05852
- Yinjie Cheng, Paul Youssef, Christin Seifert, Jörg Schlötterer, Zhixue Zhao (University of Sheffield, Marburg University)
- Venue: accepted KDD '26 (Jeju, August 2026)
- Source: docs/papers/2511.05852/paper.txt

## One-paragraph summary
The first systematic study of what happens to knowledge edits when the edited model is subsequently fine-tuned. Across 254 configurations (GPT2-XL, GPT-J-6B, Llama2-7B, Llama3.1-8B-Instruct; MEMIT, AlphaEdit, MEND; 100 / 1,000 / 10,000 edits on zsRE and CounterFact; LoRA, DoRA, full FT on Commonsense reasoning and HotpotQA), edits generally decay substantially: average relative efficacy loss is 38.10% under full FT, 28.71% under LoRA, 29.88% under DoRA (Table 4). AlphaEdit's edits are the most fragile, MEND's are near-useless to begin with, GPT-J is the most robust model, and the paper introduces the Edit Flip Ratio (EFR) to count individually successful edits that become unsuccessful.

## Problem
KE and FT are studied in isolation although real pipelines chain them. If FT erases edits, editing must be redone after every adaptation; if edits persist, malicious or backdoor edits propagate silently (Section 1-2).

## Method
Four model groups: base, FT-only, KE-only, KE-then-FT. KE metrics ES, PS, NS as in ROME/MEMIT (Appendix A); EFR = |edits successful before FT and failing after| / |successful before| (Eqn 1). FT uses the LoRA/DoRA recipes of Hu et al. and Liu et al. Activation analysis: layer-wise magnitude of activation change (Eqn 2), cosine directional similarity between KE and FT displacement (Eqn 6), UMAP of activations (Section 5).

## Experiments and results
- Table 2 (ES, zsRE, MEMIT): GPT-J 10,000 edits 96.63 no-FT, 66.52 after LoRA, 67.83 DoRA, 89.38 full FT; GPT2-XL 10,000: 62.61 to 20.34 (LoRA); Llama2 1,000: 51.38 to 46.52 (LoRA) and 10.22 (full FT). AlphaEdit Llama2 1,000 zsRE: 93.23 to 50.45 after LoRA, the largest drop in the study. On CF (a laxer pairwise-probability criterion) drops are smaller, e.g. GPT-J MEMIT 10,000: 99.10 to 94.34.
- Table 4 relative decrease: GPT-J average 15.93%, GPT2-XL 40.84%, Llama2 39.92%; full FT is benign on GPT-J (4.51%) but catastrophic on Llama2 (74.41%).
- Table 5 EFR on GPT-J: MEMIT zsRE 10,000 under LoRA 15.60%; AlphaEdit zsRE 10,000 under LoRA 25.27% (the abstract's number); CF EFR mostly under 8% except 10,000-edit rows.
- Fig. 2 / Section 4.1: LoRA removes 9.73 points of MEMIT ES on Llama2 at 100 edits versus 35.37 for AlphaEdit; the gap widens to 33.50 points at 10,000. Explanation offered: the null space is orthogonal to the knowledge used by the FT loss, so FT overwrites it without penalty.
- Knowledge-rich FT data hurts more: HotpotQA FT produces larger decay than Commonsense, up to 14.31 ES points (GPT-J, 100 zsRE, MEMIT; Table 9).
- Qualitative (Table 6): stable edits have frequent target tokens (English, Islam, piano); erased edits have rare targets (Lecanorales); "emergent" edits appear after FT because FT removes conflicting edits from the batch.
- Selective FT (Table 7, Llama2, AlphaEdit, CF): FT of only edited layers cuts ES from 96 to 66 at 100 edits (all layers: 98), but costs downstream accuracy (65.43 vs 80.61); FT of only non-edited layers also erases edits (98 to 72), contrary to the hypothesis that it would preserve them.
- KE mildly reduces later FT effectiveness; MEMIT preserves FT performance better than AlphaEdit (Table 8).
- Activation analysis (Figure 3-7): FT shifts activations far more than KE across all layers, in a direction nearly orthogonal to the edit direction; erased edits had smaller activation change during KE than stable edits.
- DeepSeek was dropped: with causal-traced layers, ES fell from 20.17% at 1,000 edits to 0.4% at 10,000 (Appendix B.4).

## Limitations
FT datasets are unrelated to the edited facts; the interesting case for us (FT on text that mentions the edited subjects) is not run. Only decay is measured, no fix is tested; the two mitigations (activation-shift regularisation in edited layers, projecting FT updates onto the orthogonal complement of edited parameters) are proposals. Absolute KE metrics are acknowledged to be unreliable (they cite 2601.17343) and only relative changes are trusted. Multi-seed runs cover FT only.

## Relevance to this workspace
Informs BASE-2, TRAIN-4, EVAL-5, TRAIN-2.
- Directly answers "do edits survive later LoRA fine-tuning": no, not reliably; expect 10-45 point efficacy drops and 15-25% EFR after a LoRA pass on unrelated data, worse on knowledge-heavy data and worse for AlphaEdit than MEMIT. Any BASE-2 design that stacks MEMIT facts and then LoRA-trains episodes (the interleaved curriculum) must measure decay; the safer order is LoRA episodes first, MEMIT facts last, or MEMIT as a facts-only arm.
- Model family matters: GPT-J (large MLP, 16384 x 4096) is robust, Llama-style models are fragile, and DeepSeek failed to edit at all with default layer choices. Qwen2.5-3B is Llama-like; treat causal-tracing layer selection and lambda as tunable, not defaults.
- Token frequency: stable edits use frequent target tokens. Our type labels ("Spark", "Frost") are common words, favourable; nonsense labels and multi-token invented targets fall in the fragile class.
- Metric idea for STAT-2: EFR is a per-item flip metric; our per-item predictions should be saved so the same paired analysis can be done between arms.
- Qwen: not tested. Compute: not reported.

## Key references worth following up
2510.00625 (Is model editing built on sand? keyword substitution rather than semantic change), 2601.17343 (are we evaluating edit locality properly?), 2605.28839 (hidden facts after editing), 2407.07791 (spread of manipulated knowledge in multi-agent systems), 2403.13355 (BadEdit; verify id), 2405.16720 (large-scale knowledge washing; verify id).
