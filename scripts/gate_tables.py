"""PLAN step 135: for payees seen before, let the model override YNAB's suggestion only when p(model top) - p(YNAB's) > delta; right-first % on
all seen payees / on transactions filed differently from the rule (2a) / on the usual ones. usage: uv run python scripts/gate_tables.py
"""
import glob, json, sys, warnings, numpy as np
sys.path.insert(0, "scripts")
from decision_tables import folds_calibrate, load, product
from ai_experiments.paths import PROCESSED
warnings.filterwarnings("ignore")
R = "results/per_item/real6_"
for name in ("bulk", "blind", "blind_v2"):
    if name == "bulk":
        items = {i["id"]: i for i in json.load(open(PROCESSED / "blind_bulk_v1.json"))["items"]}
        rule = json.load(open(PROCESSED / "blind_bulk_v1_ynabrule.json")); ph = json.load(open(PROCESSED / "blind_bulk_v1_payeehist.json"))
        d = sorted(glob.glob(R + "dm_decider_decider_decider-4b_none_h100bf16st800*_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_aux100_labrand255_laylabelled_shots_ev10soft_lora_blind_bulk_v1_labrand255_laylabelled_shots.noctx.jsonl"))
        e = sorted(glob.glob(R + "encmask_ettin1b_st3000_h100fresh*_f0_decnone_h100fresh*_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_odb20_short15_kinds20_ev10soft_blind_bulk_v1.noctx.jsonl"))
    elif name == "blind_v2":
        items = {i["id"]: i for i in json.load(open(PROCESSED / "blind_v2_others.json"))["items"]}
        rule = json.load(open(PROCESSED / "blind_v2_ynabrule.json")); ph = json.load(open(PROCESSED / "blind_v2_payeehist.json"))
        d = sorted(glob.glob(R + "dm_decider_decider_decider-4b_none_h100bf16st800*_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_oth50_aux100_labrand255_laylabelled_shots_ev10soft_lora_blind_v2_others_labrand255_laylabelled_shots.noctx.jsonl"))
        e = sorted(glob.glob(R + "encmask_ettin1b_st3000_h100fresh*_f0_decnone_h100fresh*_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_odb20_short15_kinds20_ev10soft_blind_v2.noctx.jsonl"))
    else:
        from ensemble_tables import DEC, ENC
        items = {i["id"]: i for i in json.load(open(PROCESSED / "blind_v1_others.json"))["items"]}
        rule = json.load(open(PROCESSED / "blind_v1_ynabrule.json")); ph = json.load(open(PROCESSED / "blind_v1_payeehist.json"))
        d, e = sorted(glob.glob(DEC)), sorted(glob.glob(ENC))
    P = product([load(e[0]), load(d[0])], items)
    ids = sorted(P); seen = [i for i in ids if rule[i] >= 0]
    print(f"== {name}: encoder x decider (first seeds), payees seen before; right first % overall-seen / on 2a / on control")
    for delta in (None, 0.0, 0.2, 0.4, 0.6, 0.8, 1.01):
        first = {}
        for i in seen:
            p = P[i]; top = int(np.argmax(p)); r = rule[i]
            if delta is None: pick = top
            else: pick = top if (top != r and p[top] - p[r] > delta) else r
            first[i] = pick == items[i]["answer"]
        a = [i for i in seen if items[i]["answer"] != rule[i]]; c = [i for i in seen if items[i]["answer"] == rule[i]]
        lab = "model first always" if delta is None else ("YNAB always" if delta > 1 else f"model overrides YNAB when p(top) - p(YNAB's) > {delta}")
        print(f"  {lab}: {100*np.mean([first[i] for i in seen]):.1f} / {100*np.mean([first[i] for i in a]):.1f} / {100*np.mean([first[i] for i in c]):.1f}")
