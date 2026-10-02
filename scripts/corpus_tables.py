"""Tables for PLAN step 154 (MODEL-24): readers on strands-decider's own evaluation sets (eval_corpus_slot.py), against strands v19
(re-scored here with its own evaluate_checkpoint, and its published numbers). Accuracy by set and task; the trained arm as the mean
[range] of its seeds.
usage: uv run python scripts/corpus_tables.py
"""
import glob
import json

import numpy as np

READERS = [("Qwen3.5-2B-Base, untrained (decider readout)", ["results/corpus_eval_slot_Qwen3.5-2B-Base.json"]),
           ("strands v19 (re-scored here)", ["results/corpus_eval_strands_strands-decider-2B-hobson-v19.json"]),
           ("decider's method on strands' corpus, from Qwen3.5-2B-Base", sorted(glob.glob("results/corpus_eval_slot_corpusslot_qwen3.5-2b-base_*.json"))),
           ("decider-2B as published (untrained here)", ["results/corpus_eval_slot_decider-2b.json"]),
           ("decider-4B as published (untrained here)", ["results/corpus_eval_slot_decider-4b.json"])]
PUBLISHED = {"held-out short tasks": 0.641, "ContractNLI": 0.872, "MuSiQue": 0.884, "BoardgameQA": 0.822, "HotpotQA (held out)": 0.717,
             "adequacy, HelpSteer2": 0.722, "adequacy, generated": 0.798}
COLS = [("held-out short tasks", "holdout_v5_norule", "_all"), ("emotion", "holdout_v5_norule", "emotion"),
        ("massive_intent", "holdout_v5_norule", "massive_intent"), ("sarcasm", "holdout_v5_norule", "sarcasm"),
        ("hate_severity", "holdout_v5_norule", "hate_severity"),
        ("ContractNLI", "multistep_v14_eval", "contractnli"), ("MuSiQue", "multistep_v14_eval", "musique"),
        ("BoardgameQA", "multistep_v14_eval", "boardgame"), ("HotpotQA (held out)", "multistep_v14_eval", "hotpotqa"),
        ("generated v16", "generated_v16_eval", "_all"), ("generated v18", "generated_v18_eval", "_all"),
        ("adequacy, HelpSteer2", "adequacy_hs2_eval", "_all"), ("adequacy, generated", "adequacy_gen_eval", "_all")]


def cell(vals):
    v = np.array(vals) * 100
    return "–" if not len(v) else f"{v.mean():.1f}" if len(v) == 1 else f"{v.mean():.1f} [{v.min():.1f}, {v.max():.1f}]"


if __name__ == "__main__":
    print("| reader | " + " | ".join(c[0] for c in COLS) + " |"); print("|---|" + "---|" * len(COLS))
    print("| strands v19, published | " + " | ".join(f"{100 * PUBLISHED[c[0]]:.1f}" if c[0] in PUBLISHED else "–" for c in COLS) + " |")
    for name, files in READERS:
        docs = [json.load(open(f))["sets"] for f in files]
        print(f"| {name} ({len(files)}) | " + " | ".join(cell([d[s][t]["acc"] for d in docs if t in d.get(s, {})]) for _, s, t in COLS) + " |")
    print("\nScoring minutes per set (one H100; decider readout in this repo's loop, BATCH 8; strands through its evaluate_checkpoint, batch 16):\n")
    print("| reader | " + " | ".join(s for s in dict.fromkeys(c[1] for c in COLS)) + " |"); print("|---|" + "---|" * 6)
    for name, files in READERS:
        d = json.load(open(files[0]))["sets"]
        print(f"| {name} | " + " | ".join(f"{d[s]['_minutes']:.2f}" for s in dict.fromkeys(c[1] for c in COLS)) + " |")
    for f in sorted(glob.glob("results/corpusslot_ft_*.json")):
        r = json.load(open(f)); print(f"\n{f}: {r['train_minutes']} min, {r['tokens']:,} tokens, {r['steps']} steps, final loss {r['final_loss']}")
