"""Tables for PLAN steps 99 (MODEL-17, the encoder rematch) and 101 (MODEL-18, newer small decoders): every reader trained on the final
recipe's episodes against decider-4B's final recipe (row 89, seed 0), fold 0's held-out users on REAL-6 and the REAL-6 behaviour sets,
all 200 users on REAL-7.

  R.1  top-1 on REAL-6 fold 0, novel names (fold 0), REAL-7 v1 (the 24 latest rows) and v2 all_kind (row 98's slice), ms per item
  R.2  decoders: the trained behaviours on fold 0 (misleading names in the DB, the user's override, alternation rules, confident on
       evidence-free and 60 / 40 items)
usage: uv run python scripts/rematch_tables.py
"""
import json
import warnings

import numpy as np

from ai_experiments import real6 as R6
from ai_experiments import real6_cells as RC
from ai_experiments.paths import PROCESSED, ROOT

REC = "ren50_dbep50_mislead_v1_alt10s_lk10_ov10"
DEC = f"decnone_h100_f0_{REC}"
DEC2 = f"decnone_h100fresh_f0_{REC}"
FINAL = f"decider_decider-4b_none_h100bf16st800_f0_{REC}_aux100_labrand26_ev10soft_lora"


def dm(adapter):
    return lambda s: f"real6_dm_decider_{adapter}_{s or 'real6'}_labrand26.noctx.jsonl"


def enc(name):
    return lambda s: f"real6_{name}{'_' + s if s else ''}.noctx.jsonl"


DEC3 = f"decnone_h100freshs1_f0_{REC}"
ENCODERS = [("ModernBERT-large 395M, REAL-6 episodes (row 53)", [enc("encmask_mbert_st1500_f0")], "encmask_mbert_st1500_f0"),
            ("Laya 395M, REAL-6 episodes (row 53)", [enc("encmask_laya_st1500_f0")], "encmask_laya_st1500_f0"),
            ("GLiClass-large, REAL-6 episodes (row 69)", [enc("encgli_large_st1500_f0")], "encgli_large_st1500_f0"),
            ("ModernBERT-large 395M, recipe episodes, one draw cycled (memorised)", [enc(f"encmask_mbert_st3000_h100_f0_{DEC}")], f"encmask_mbert_st3000_h100_f0_{DEC}"),
            ("Ettin-encoder 1B, recipe episodes, one draw cycled (memorised)", [enc(f"encmask_ettin1b_st3000_h100_f0_{DEC}")], f"encmask_ettin1b_st3000_h100_f0_{DEC}"),
            ("ModernBERT-large 395M, recipe episodes", [enc(f"encmask_mbert_st3000_h100fresh_f0_{DEC2}")], f"encmask_mbert_st3000_h100fresh_f0_{DEC2}"),
            ("Laya 395M, recipe episodes", [enc(f"encmask_laya_st3000_{t}_f0_{d}") for t, d in (("h100fresh", DEC2), ("h100freshs1", DEC3))], f"encmask_laya_st3000_h100fresh_f0_{DEC2}"),
            ("GLiClass-large 400M, recipe episodes", [enc(f"encgli_large_st3000_{t}_f0_{d}") for t, d in (("h100fresh", DEC2), ("h100freshs1", DEC3))], f"encgli_large_st3000_h100fresh_f0_{DEC2}"),
            ("Ettin-encoder 1B, recipe episodes", [enc(f"encmask_ettin1b_st3000_{t}_f0_{d}") for t, d in (("h100fresh", DEC2), ("h100freshs1", DEC3))], f"encmask_ettin1b_st3000_h100fresh_f0_{DEC2}"),
            ("EuroBERT 2.1B, recipe episodes", [enc(f"encmask_eurobert_st6000_h100fresh_f0_{DEC2}")], f"encmask_eurobert_st6000_h100fresh_f0_{DEC2}"),
            ("GLiClass-large 400M, recipe episodes + dated REAL-7 slices", [enc(f"encgli_large_st3000_h100fresh_f0_{DEC2}_r7all_kind")], f"encgli_large_st3000_h100fresh_f0_{DEC2}_r7all_kind")]
DECODERS = [("decider-4B, final recipe (row 89)", [dm(FINAL), dm(FINAL.replace("st800_", "st800s1_"))], None),
            ("decider-4B, + dated REAL-7 slices (row 98)", [dm(FINAL.replace("_ov10_", "_ov10_r7all_kind_")), dm(FINAL.replace("_ov10_", "_ov10_r7all_kind_").replace("st800_", "st800s1_"))], None)] + [
    (f"{m}, final recipe", [dm(f"slot_{m}_none_h100bf16st800_f0_{REC}_aux100_labrand26_ev10soft_lora")], None)
    for m in ("MiniCPM5-2B", "granite-4.0-micro", "SmolLM3-3B", "Phi-4-mini-instruct")]
SETS = [("", "REAL-6", True), ("real6_v1_novel", "novel names", True), ("real7_v1", "REAL-7 v1", False), ("real7_v2_all_kind", "REAL-7 v2 all_kind", False)]


def sm(z):
    z = np.asarray(z, float); e = np.exp(z - z.max()); return e / e.sum()


def items_of(s):
    return {i["id"]: i for i in (json.loads((PROCESSED / f"{s}.json").read_text())["items"] if s else R6.load()["items"])}


def top1(recs, items, fold0):
    ids = [i for i in recs if i in items and (not fold0 or items[i]["user"] % 4 == 0)]
    return 100 * np.mean([int(np.argmax(recs[i]["sum_lp"])) == items[i]["answer"] for i in ids]) if ids else None, len(ids)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    its = {s: items_of(s) for s, _, _ in SETS}
    print("**Table R.1: readers trained on the final recipe's episodes (REAL-6 sets: fold 0's held-out users; REAL-7: all 200 new users)**\n")
    print("| reader | " + " | ".join(lab for _, lab, _ in SETS) + " | ms per item (batched / one at a time) |"); print("|---|" + "---|" * (len(SETS) + 1))
    for lab, fs, res in ENCODERS + DECODERS:
        cells = []
        for s, _, fold0 in SETS:
            v = [top1(r, its[s], fold0)[0] for r in (RC.load_recs(f(s)) for f in fs) if r]
            cells.append("–" if not v else f"{np.mean(v):.1f}" + (f" [{min(v):.1f}, {max(v):.1f}]" if len(v) > 1 else ""))
        ms = "–"
        if res and (ROOT / "results" / f"{res}.json").exists():
            d = json.loads((ROOT / "results" / f"{res}.json").read_text()); d = d.get("noctx", d)
            ms = f"{d.get('ms_per_item_batched', '–')} / {d.get('ms_per_item_single', '–')}"
        if lab.startswith("decider-4B, final"):
            ms = "32 / 75 (r99c timing, warm)"
        if any(c != "–" for c in cells):
            print(f"| {lab} | " + " | ".join(cells) + f" | {ms} |")
    beh = {s: items_of(s) for s in ("mislead_v1", "override_v1", "alternation_v1")}
    print("\n**Table R.2: the trained behaviours, fold 0 (first seed)**\n")
    print("| reader | misleading, in DB | override | alternation stores / restaurants | evidence-free: p >= 0.9 | 60 / 40 splits: p >= 0.9 |"); print("|---|---|---|---|---|---|")
    for lab, fs, _ in ENCODERS[5:] + DECODERS:
        f = fs[0]
        ml, ov, al = RC.load_recs(f("mislead_v1")), RC.load_recs(f("override_v1")), RC.load_recs(f("alternation_v1"))
        if not (ml and ov and al):
            continue
        mi, oi, ai = beh["mislead_v1"], beh["override_v1"], beh["alternation_v1"]
        ok = lambda r, it, i: int(np.argmax(r[i]["sum_lp"])) == it[i]["answer"]  # noqa: E731
        a = np.mean([ok(ml, mi, i) for i in ml if mi[i]["mkind"] == "misleading" and mi[i]["in_db"] and mi[i]["cond"] == "none"])
        b = np.mean([ok(ov, oi, i) for i in ov if oi[i]["cond"] == "override"])
        c1 = np.mean([ok(al, ai, i) for i in al if ai[i]["scen"] == "observable" and ai[i]["kind"] == "store"])
        c2 = np.mean([ok(al, ai, i) for i in al if ai[i]["scen"] == "observable" and ai[i]["kind"] == "restaurant"])
        d = np.mean([sm(ml[i]["sum_lp"]).max() >= 0.9 for i in ml if mi[i]["mkind"] == "neutral" and not mi[i]["in_db"] and mi[i]["cond"] == "none"])
        e = np.mean([sm(al[i]["sum_lp"]).max() >= 0.9 for i in al if ai[i]["scen"] == "unobservable"])
        print(f"| {lab} | {100 * a:.1f} | {100 * b:.1f} | {100 * c1:.1f} / {100 * c2:.1f} | {100 * d:.1f} | {100 * e:.1f} |")
