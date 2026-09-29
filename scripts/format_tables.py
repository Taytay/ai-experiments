"""Tables for PLAN steps 111 to 114 (owner, 2026-09-29): prompt formats, one change at a time against the recipes (decider-4B: row 89 +
rand255 + empty categories; Ettin-1B: the encoder recipe with kinds), each arm trained and read in its own format (`oneslot.build_layout`,
exp_encoder_mask.py ENC_LAYOUT / DOW_FIRST, exp_categoriser.py NSHOTS).

  FM.1  per arm (mean over seeds): REAL-6 fold 0, novel names, misleading names in the DB, the day-of-week alternation rule (restaurants),
        real businesses in no DB (descriptive), blind_v1 top-1, calibrated auto-filing and effort alone, and effort in the system (the other
        model at its recipe: encoder first, decider for the rest); the 48-row arms also on blind_v1_b48
usage: uv run python scripts/format_tables.py
"""
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments import scorecard as S

sys.path.insert(0, str(Path(__file__).parent))
import rematch_tables as R  # noqa: E402
from effort_tables import calibrated  # noqa: E402

REC = R.REC
PI = "results/per_item/"
DEC = "real6_dm_decider_decider_decider-4b_none_h100bf16st800{s}_emp20_f0_" + REC + "{n}_aux100_labrand255{lay}_ev10soft_lora_{set}_labrand255{lay}.noctx.jsonl"
ENC = "real6_encmask_ettin1b_st{steps}_h100fresh{s}_f0_decnone_h100fresh{s}_emp20_f0_" + REC + "_odb20_short15_kinds20{n}_ev10soft{lay}_{set}.noctx.jsonl"
DEC_ARMS = [("decider, recipe (decider's layout)", "", ""), ("labelled list, ends 'Category: ('", "", "_laylabelled"),
            ("labelled list + 'Category: (AE) Pets' rows", "", "_laylabelled_shots"), ("weekday next to the date", "", "_dow"),
            ("48 rows", "_sh48", ""), ("TSV table (Options / 'Answer: (')", "", "_laytable"), ("TSV table, labelled cells, open ' (' cell", "", "_laytable_labelled"),
            ("JSON lines, labelled", "", "_layjson_labelled"), ("YAML list, labelled", "", "_layyaml_labelled"), ("TypeScript, labelled", "", "_layts_labelled"),
            ("TypeScript enum, Categories[\"GU\"]", "", "_layts_enum"), ("TypeScript enum + name comments", "", "_layts_enum_names")]
ENC_ARMS = [("encoder, recipe", "", ""), ("labelled options and rows", "", "_laylabelled_shots"), ("weekday next to the date", "", "_dow"),
            ("48 rows", "_sh48", ""), ("TSV table", "", "_laytable"), ("JSON lines, labelled", "", "_layjson_labelled"),
            ("YAML list, labelled", "", "_layyaml_labelled"), ("TypeScript, labelled (batch 8 x 6,000 steps)", "", "_layts_labelled")]
SEEDS = ("", "s1", "s2")


def files(pat, st, **kw):
    kw.setdefault("steps", "6000" if kw.get("lay") == "_layts_labelled" and pat.startswith("real6_encmask") else "3000")
    return [f for s in SEEDS for f in glob.glob(PI + pat.format(s=s, set=st, **kw))]


def load(f):
    return {r["id"]: r for r in map(json.loads, open(f))}


def fmt(v, d=1):
    v = [x for x in v if x is not None and not np.isnan(x)]
    return "–" if not v else f"{np.mean(v):.{d}f}" + (f" [{min(v):.{d}f}, {max(v):.{d}f}]" if len(v) > 1 else "")


def effort(items, parts):
    ids = sorted(set.intersection(*(set(p) for p in parts))); a, ok, t3 = [], [], []
    for i in ids:
        m = next((p[i] for p in parts[:-1] if p[i]["a98"]), parts[-1][i]); y = items[i]["answer"]
        a.append(bool(m["a98"])); ok.append(int(np.argmax(m["p"])) == y); t3.append(y in np.argsort(-m["p"])[:3])
    return S.effort(a, ok, t3)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    its = {s: R.items_of(s) for s in ("", "real6_v1_novel", "mislead_v1", "alternation_v1", "novel_merchants_v1", "blind_v1", "blind_v1_b48")}
    ok = lambda r, it, i: int(np.argmax(r[i]["sum_lp"])) == it[i]["answer"]  # noqa: E731
    base = {"dec": [calibrated(load(f), its["blind_v1"]) for f in files(DEC, "blind_v1", n="", lay="")],
            "enc": [calibrated(load(f), its["blind_v1"]) for f in files(ENC, "blind_v1", n="", lay="", steps="3000")]}
    for kind, pat, arms in (("dec", DEC, DEC_ARMS), ("enc", ENC, ENC_ARMS)):
        print(f"\n**Table FM.1{'a' if kind == 'dec' else 'b'}: {'decider-4B' if kind == 'dec' else 'Ettin-1B'} (mean [range] over seeds)**\n")
        print("| arm | seeds | REAL-6 | novel names | misleading | day rule (restaurants) | real businesses (descriptive) | blind_v1 | blind auto-filed | effort alone | effort in the system | blind_v1_b48 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for lab, n, lay in arms:
            c = {k: [] for k in ("r6", "nv", "m", "dr", "nm", "b", "ba", "ea", "es", "b48")}
            st = lambda s: s if kind == "dec" else ("" if s == "real6" else s)  # noqa: E731
            for f in files(pat, "real6" if kind == "dec" else "", n=n, lay=lay) if kind == "dec" else []:
                c["r6"].append(R.top1(load(f), its[""], True)[0])
            if kind == "enc":
                c["r6"] = [R.top1(load(f), its[""], True)[0] for f in glob.glob(PI + pat.format(s="*", set="", n=n, lay=lay, steps="6000" if lay == "_layts_labelled" else "3000").replace("_.noctx", ".noctx"))]
            c["nv"] = [R.top1(load(f), its["real6_v1_novel"], True)[0] for f in files(pat, "real6_v1_novel", n=n, lay=lay)]
            for f in files(pat, "mislead_v1", n=n, lay=lay):
                r = load(f); mi = its["mislead_v1"]; c["m"].append(100 * np.mean([ok(r, mi, i) for i in r if mi[i]["mkind"] == "misleading" and mi[i]["in_db"] and mi[i]["cond"] == "none"]))
            for f in files(pat, "alternation_v1", n=n, lay=lay):
                r = load(f); ai = its["alternation_v1"]; c["dr"].append(100 * np.mean([ok(r, ai, i) for i in r if ai[i]["scen"] == "observable" and ai[i]["kind"] == "restaurant"]))
            for f in files(pat, "novel_merchants_v1", n=n, lay=lay):
                r = load(f); nm = its["novel_merchants_v1"]; c["nm"].append(100 * np.mean([ok(r, nm, i) for i in r if nm[i]["nm_group"] == "descriptive" and nm[i]["user"] % 4 == 0]))
            for f in files(pat, "blind_v1", n=n, lay=lay):
                r = load(f); bl = its["blind_v1"]; cal = calibrated(r, bl)
                c["b"].append(100 * np.mean([ok(r, bl, i) for i in r])); c["ba"].append(100 * np.mean([cal[i]["a98"] for i in cal]))
                c["ea"].append(effort(bl, [cal]))
                c["es"] += [effort(bl, [e, cal]) for e in base["enc"]] if kind == "dec" else [effort(bl, [cal, d]) for d in base["dec"]]
            for f in files(pat, "blind_v1_b48", n=n, lay=lay):
                c["b48"].append(100 * np.mean([ok(load(f), its["blind_v1_b48"], i) for i in load(f)]))
            nseeds = len(c["b"])
            print(f"| {lab} | {nseeds} | {fmt(c['r6'])} | {fmt(c['nv'])} | {fmt(c['m'])} | {fmt(c['dr'])} | {fmt(c['nm'])} | {fmt(c['b'])} | {fmt(c['ba'])} | {fmt(c['ea'], 3)} | {fmt(c['es'], 3)} | {fmt(c['b48'])} |")
    print("\nThe recipe read on blind_v1_b48 without training on 48 rows: " + fmt([100 * np.mean([ok(load(f), its["blind_v1_b48"], i) for i in load(f)]) for f in files(DEC, "blind_v1_b48", n="", lay="")]))
