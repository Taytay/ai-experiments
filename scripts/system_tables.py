"""Tables for PLAN step 110: the system on blind_v1 over three seeds of each part: the encoder (Ettin-1B, the encoder recipe with KINDS 0.2,
REPORT 111) auto-files at its own calibrated 98% threshold, decider-4B (the recipe, or the recipe + KINDS 0.1) reads the rest. Each pair
of seeds is one system; the table gives the mean and range over the 3 x 3 pairs, and each part alone. `scorecard.effort` (wrong = 5).
blind_v1 has no payee identities on its history rows, so the owner's history rule is not in front here (REPORT 98 measures it on
REAL-7).
usage: uv run python scripts/system_tables.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments import real6_cells as RC
from ai_experiments import scorecard as S
from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
from effort_tables import calibrated  # noqa: E402

REC = "ren50_dbep50_mislead_v1_alt10s_lk10_ov10"
SEEDS = ("", "s1", "s2")
DEC = {"decider-4B, recipe": "real6_dm_decider_decider_decider-4b_none_h100bf16st800{s}_emp20_f0_" + REC + "_aux100_labrand255_ev10soft_lora_blind_v1_labrand255.noctx.jsonl",
       "decider-4B, recipe + kinds 0.1": "real6_dm_decider_decider_decider-4b_none_h100bf16st800{s}_emp20_f0_" + REC + "_kinds10_aux100_labrand255_ev10soft_lora_blind_v1_labrand255.noctx.jsonl"}
ENC = "real6_encmask_ettin1b_st3000_h100fresh{s}_f0_decnone_h100fresh{s}_emp20_f0_" + REC + "_odb20_short15_kinds20_ev10soft_blind_v1.noctx.jsonl"


def run(items, parts):
    """parts: list of calibrated dicts, earliest first; each auto-files what clears its threshold, the last one also suggests."""
    ids = sorted(set.intersection(*(set(p) for p in parts))); a, ok, t3, t1 = [], [], [], []
    for i in ids:
        m = next((p[i] for p in parts[:-1] if p[i]["a98"]), parts[-1][i]); y = items[i]["answer"]; top = int(np.argmax(m["p"]))
        a.append(bool(m["a98"])); ok.append(top == y); t3.append(y in np.argsort(-m["p"])[:3]); t1.append(top == y)
    prec = 100 * np.mean([o for x, o in zip(a, ok) if x]) if any(a) else float("nan")
    return S.effort(a, ok, t3), 100 * np.mean(a), prec, 100 * np.mean(t1), 100 * np.mean([not parts[0][i]["a98"] for i in ids]) if len(parts) > 1 else float("nan")


def fmt(v, d=1):
    v = np.array(v); return f"{v.mean():.{d}f} [{v.min():.{d}f}, {v.max():.{d}f}]" if len(v) > 1 else f"{v.mean():.{d}f}"


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in json.loads((PROCESSED / "blind_v1.json").read_text())["items"]}
    load = lambda f: calibrated(RC.load_recs(f), items) if RC.load_recs(f) else None  # noqa: E731
    encs = [e for e in (load(ENC.format(s=s)) for s in SEEDS) if e]
    print(f"**Table SY.1: blind_v1, the system over seeds (encoder seeds: {len(encs)})**\n")
    print("| system | runs | effort | auto-filed % | precision % | top-1 | decider reads % |"); print("|---|---|---|---|---|---|---|")
    rows = [("encoder alone", [run(items, [e]) for e in encs])]
    for lab, pat in DEC.items():
        decs = [d for d in (load(pat.format(s=s)) for s in SEEDS) if d]
        rows.append((f"{lab} alone", [run(items, [d]) for d in decs]))
        rows.append((f"encoder, then {lab}", [run(items, [e, d]) for e in encs for d in decs]))
    for lab, v in rows:
        if v:
            v = np.array(v)
            print(f"| {lab} | {len(v)} | {fmt(v[:, 0], 3)} | {fmt(v[:, 1])} | {fmt(v[:, 2])} | {fmt(v[:, 3])} | {fmt(v[:, 4], 0) if not np.isnan(v[:, 4]).all() else "–"} |")
