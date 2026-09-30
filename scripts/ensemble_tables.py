"""Tables for PLAN step 131 (owner, 2026-09-30: "if we just need another opinion, would an untrained 4B help? an ensemble of smaller
models?"). CPU only, over saved blind_v1 scores. The base is REPORT 126's pair: the encoder (Ettin-1B, three seeds) x decider-4B trained
with the other-users line (three seeds, reading blind_v1_others). Each candidate third reader X is added two ways, with the per-item rule
(REPORT 118): "pair -> X" (what the pair does not auto-file is re-decided on the pair x X product) and "all three" (every transaction
on the three-model product). Also X alone, how often X's top answer differs from the pair's where the pair is wrong (its independence),
and a four-reader ensemble. Relative compute per transaction beside each (decider-4B = 1).
usage: uv run python scripts/ensemble_tables.py
"""
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments import scorecard as S
from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
from decision_tables import fmt, folds_calibrate, load, product  # noqa: E402

R = "results/per_item/real6_"
REC = "ren50_dbep50_mislead_v1_alt10s_lk10_ov10"
DEC = R + "dm_decider_decider_decider-4b_none_h100bf16st800*_emp20_f0_" + REC + "_oth50_aux100_labrand255_laylabelled_shots_ev10soft_lora_blind_v1_others_labrand255_laylabelled_shots.noctx.jsonl"
ENC = R + f"encmask_ettin1b_st3000_h100fresh*_f0_decnone_h100fresh*_emp20_f0_{REC}_odb20_short15_kinds20_ev10soft_blind_v1.noctx.jsonl"
D4 = R + "dm_decider_decider_decider-4b_none_h100bf16st800{s}_emp20_f0_" + REC + "_aux100_labrand255{lay}_ev10soft_lora_blind_v1_labrand255{lay}.noctx.jsonl"
CANDS = [  # (name, file, relative compute per transaction)
    ("decider-4B untrained (letters)", R + "dm_decider_decider-4b_blind_v1.noctx.jsonl", 1.0),
    ("decider-4B untrained (random A-Z labels)", R + "dm_decider_decider-4b_blind_v1_labrand26.noctx.jsonl", 1.0),
    ("decider-4B recipe, decider's own layout (other prompt)", D4.format(s="", lay=""), 1.0),
    ("decider-4B recipe, YAML history", D4.format(s="", lay="_layyaml_labelled"), 2.0),
    ("decider-4B recipe, TSV table", D4.format(s="", lay="_laytable_labelled"), 1.2),
    ("decider-4B recipe, labelled rows, seed 2 (same prompt)", D4.format(s="s2", lay="_laylabelled_shots"), 1.0),
    ("decider-35B-A3B untrained", R + "dm_decider_decider-35b-a3b_blind_v1.noctx.jsonl", 1.5),
    ("decider-35B-A3B recipe (row 125)", R + "dm_decider_decider_decider-35b-a3b_none_h200bf16st800_emp20_f0_" + REC + "_aux100_labrand255_laylabelled_shots_ev10soft_lora_blind_v1_labrand255_laylabelled_shots.noctx.jsonl", 1.5),
]
WS = (5, 10, 20)


def effort(P, items, auto, W):
    ids = sorted(P); ok = [int(np.argmax(P[i])) == items[i]["answer"] for i in ids]
    t3 = [items[i]["answer"] in np.argsort(-P[i])[:3] for i in ids]; a = [auto[i] for i in ids]
    return S.effort(a, ok, t3, wrong=W), 100 * np.mean(a)


def rule(P, W):
    return {i: bool(S.decide(p.max(), np.sort(p)[-3:].sum(), W)) for i, p in P.items()}


def systems(pairs, xs, items):
    """{W: ([pair -> X efforts], [all-three efforts])} over the seed pairs; xs: the extra readers (one or more)."""
    out = {w: ([], []) for w in WS}
    for e, d in pairs:
        P2 = product([e, d], items); Pn = product([e, d] + xs, items)
        for w in WS:
            a2, an = rule(P2, w), rule(Pn, w); ids = [i for i in P2 if i in Pn]
            esc = {i: (P2[i] if a2[i] else Pn[i]) for i in ids}; ae = {i: a2[i] or an[i] for i in ids}
            out[w][0].append(effort(esc, items, ae, w)[0]); out[w][1].append(effort(Pn, items, an, w)[0])
    return out


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in json.loads((PROCESSED / "blind_v1.json").read_text())["items"]}
    encs = [load(f) for f in sorted(glob.glob(ENC))]; decs = [load(f) for f in sorted(glob.glob(DEC))]
    pairs = [(e, d) for e in encs for d in decs]
    base = {w: [] for w in WS}; wrong_ids = []
    for e, d in pairs:
        P2 = product([e, d], items)
        for w in WS:
            base[w].append(effort(P2, items, rule(P2, w), w)[0])
        wrong_ids.append({i for i in P2 if int(np.argmax(P2[i])) != items[i]["answer"]})
    print(f"**Table EN.1: blind_v1, effort per transaction under the per-item rule (mean [range] over {len(pairs)} seed pairs of the encoder x decider with the other-users line)**\n")
    print("| third reader X | compute (4B = 1) | X alone: top-1 / effort W=10 | X right where the pair is wrong % | pair -> X: W = 5 / 10 / 20 | all three: W = 5 / 10 / 20 |")
    print("|---|---|---|---|---|---|")
    print("| (none: the pair) | 0 | – | – | " + " / ".join(fmt(base[w], 3) for w in WS) + " | – |")
    loaded = {}
    for name, f, cost in CANDS:
        if not Path(f).exists():
            print(f"| {name} | – | missing | | | |"); continue
        x = load(f); loaded[name] = x; Px = folds_calibrate(x, items)
        top1 = 100 * np.mean([int(np.argmax(Px[i])) == items[i]["answer"] for i in Px])
        alone = effort(Px, items, rule(Px, 10), 10)[0]
        rescue = np.mean([100 * np.mean([int(np.argmax(Px[i])) == items[i]["answer"] for i in w if i in Px]) for w in wrong_ids])
        s = systems(pairs, [x], items)
        print(f"| {name} | {cost} | {top1:.1f} / {alone:.3f} | {rescue:.1f} | " + " / ".join(fmt(s[w][0], 3) for w in WS) + " | " + " / ".join(fmt(s[w][1], 3) for w in WS) + " |")
    for combo in (("decider-4B untrained (letters)", "decider-35B-A3B untrained"),
                  ("decider-4B untrained (letters)", "decider-4B recipe, decider's own layout (other prompt)"),
                  ("decider-4B untrained (letters)", "decider-4B recipe, decider's own layout (other prompt)", "decider-35B-A3B untrained")):
        s = systems(pairs, [loaded[n] for n in combo], items)
        cost = sum(c for n, _, c in CANDS if n in combo)
        print(f"| + " + " + ".join(combo) + f" | {cost} | – | – | " + " / ".join(fmt(s[w][0], 3) for w in WS) + " | " + " / ".join(fmt(s[w][1], 3) for w in WS) + " |")
