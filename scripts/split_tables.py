"""Tables for PLAN step 104 (part: the encoder / decoder split, CPU): on blind_v1, routing each transaction to the encoder (Ettin-1B with
the full encoder recipe, REPORT 104) or to decider-4B (row 89), read by user effort (`scorecard.effort`: auto-filed right 0, wrong 5, top-3
suggestion 1, search 3) and by the share of transactions the decoder must read (its cost is about three times the encoder's).

Routes: each model alone; by the prompt (the payee's rows in the slice -> encoder, else decoder: `in_shots`); encoder first (the
encoder auto-files what clears its calibrated 98% threshold, the decoder reads the rest). Auto-filing: each model's own leave-fold-out
temperature and 98% threshold (effort_tables.calibrated).
usage: uv run python scripts/split_tables.py
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
DEC = f"real6_dm_decider_decider_decider-4b_none_h100bf16st800{{s}}_f0_{REC}_aux100_labrand26_ev10soft_lora_blind_v1_labrand26.noctx.jsonl"
ENC = f"real6_encmask_ettin1b_st3000_h100fresh_f0_decnone_h100fresh_f0_{REC}_ev10soft_blind_v1.noctx.jsonl"

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in json.loads((PROCESSED / "blind_v1.json").read_text())["items"]}
    enc = calibrated(RC.load_recs(ENC), items)
    print("**Table SP.1: blind_v1, routing between Ettin-1B (encoder recipe) and decider-4B (row 89), by user effort (wrong auto-file = 5)**\n")
    print("| route | decider seed | read by the decoder | top-1 | auto-filed (precision) | searches | effort |"); print("|---|---|---|---|---|---|---|")
    for s in ("", "s1"):
        dec = calibrated(RC.load_recs(DEC.format(s=s)), items)
        ids = sorted(set(enc) & set(dec))
        routes = {"decoder alone": {i: "d" for i in ids}, "encoder alone": {i: "e" for i in ids},
                  "payee in the slice -> encoder, else decoder": {i: "e" if items[i]["in_shots"] else "d" for i in ids},
                  "encoder auto-files at its 98%, decoder reads the rest": {i: "e" if enc[i]["a98"] else "d" for i in ids}}
        for name, rt in routes.items():
            a, ok, t3, t1 = [], [], [], []
            for i in ids:
                m = enc[i] if rt[i] == "e" else dec[i]; y = items[i]["answer"]; top = int(np.argmax(m["p"]))
                a.append(bool(m["a98"])); ok.append(top == y); t3.append(y in np.argsort(-m["p"])[:3]); t1.append(top == y)
            n_dec = 100 * np.mean([rt[i] == "d" for i in ids])
            prec = 100 * np.mean([o for x, o in zip(a, ok) if x]); srch = 100 * np.mean([not x and not t for x, t in zip(a, t3)])
            print(f"| {name} | {s or 's0'} | {n_dec:.0f}% | {100 * np.mean(t1):.1f} | {100 * np.mean(a):.1f} ({prec:.1f}) | {srch:.1f} | {S.effort(a, ok, t3):.3f} |")
