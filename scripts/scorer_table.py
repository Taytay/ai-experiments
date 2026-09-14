"""Every arm under every scoring rule, from the saved per-item records (PLAN step 2; EVAL-1, EVAL-2, EVAL-6).

usage: uv run python scripts/scorer_table.py [Qwen2.5-3B] [--arms base A B ...] [--per-item DIR] [--out-root DIR]

Reads results/per_item/curriculum_<tag>_<arm>.<condition>.jsonl for the curriculum arms and
results/per_item/rescore_universe_*.jsonl for the section 6 adapters, re-scores every item under
each rule in ai_experiments.scorers (no model needed) and writes

  results/scorers_<tag>.json    accuracies per arm x condition x rule x level, RStd, histograms, fitted b
  reports/scorers_<tag>.md      the tables: one per rule and condition, the cells that move beyond the
                                item-count noise, the question-free (UNC) artifact check, constant
                                predictors, RStd

and prints the digest that goes into REPORT.md.
"""
import argparse
import json
from collections import defaultdict

from ai_experiments.paths import ROOT
from ai_experiments import scorers as SC

ARMS = ["base", "A", "B", "C", "Cn", "D", "base_m", "E"]
LABEL = {"base": "base", "A": "A know", "B": "B epis", "C": "C inter+R", "Cn": "Cn inter", "D": "D seq",
         "base_m": "base(m)", "E": "E morph",
         "universe_Qwen2.5-0.5B_lr0.0001": "U-0.5B (sec 6)", "universe_Qwen2.5-3B_lr0.0001": "U-3B (sec 6)",
         "universe_Qwen2.5-3B_lr0.0001_unsloth": "U-3B unsloth (sec 6)"}
LEVELS = ["L1_recall", "L1_recall_fmt", "L2_manip_isa", "L2_manip_pair", "L3_induct_type_nonsense",
          "L3_induct_type_realnames", "L3_induct_type_k2", "L3_induct_type_k4", "L4_induct_weakness",
          "L4_induct_habitat", "L5_novel_choices", "L6_unseen_recall", "L6_seen_recall_ctrl", "L3_induct_heldout",
          "M_probe_marked", "M_probe_plain", "M_probe_marked_fmt", "M_probe_plain_fmt",
          "ICL_symbol_banking77", "ICL_symbol_dbpedia", "ICL_symbol_sst2", "ICL_symbol_subj",
          "ICL_natural_banking77", "ICL_natural_dbpedia", "ICL_natural_sst2", "ICL_natural_subj"]
MEANS = {"ICL_symbol_mean": [l for l in LEVELS if l.startswith("ICL_symbol")],
         "ICL_natural_mean": [l for l in LEVELS if l.startswith("ICL_natural")]}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("tag", nargs="?", default="Qwen2.5-3B")
ap.add_argument("--arms", nargs="*", default=ARMS)
ap.add_argument("--per-item", default=str(ROOT / "results" / "per_item"))
ap.add_argument("--out-root", default=str(ROOT))
a = ap.parse_args()
PER_ITEM, OUT_ROOT = __import__("pathlib").Path(a.per_item), __import__("pathlib").Path(a.out_root)


# ------------------------------------------------------------------ load
def load_arm(stem: str, base_like: bool):
    """-> {"noctx": records, "ctx": records} or None if the files are missing."""
    conds = ("base", "base_ctx") if base_like else ("trained", "trained_ctx")
    paths = [PER_ITEM / f"{stem}.{c}.jsonl" for c in conds]
    if not all(p.exists() for p in paths):
        return None
    return {"noctx": SC.read_records(paths[0]), "ctx": SC.read_records(paths[1])}


data = {}
for arm in a.arms:
    d = load_arm(f"curriculum_{a.tag}_{arm}", base_like=arm.startswith("base"))
    if d:
        data[arm] = d
for p in sorted(PER_ITEM.glob("rescore_universe_*.trained.jsonl")):
    name = p.name[len("rescore_"):-len(".trained.jsonl")]
    d = load_arm(f"rescore_{name}", base_like=False)
    if d:
        data[name] = d
if not data:
    raise SystemExit(f"no per-item files under {PER_ITEM}")
arms = list(data)
print(f"{len(arms)} arms: {', '.join(arms)}", flush=True)

# ------------------------------------------------------------------ score
out = {"tag": a.tag, "rules": {k: SC.SCORERS[k].__doc__ or k for k in SC.SCORERS}, "arms": {}}
levels_seen = []
for arm in arms:
    out["arms"][arm] = {}
    for cond, recs in data[arm].items():
        params = SC.params_for(recs)
        acc = {s: SC.accuracy(recs, s, params) for s in SC.SCORERS}
        for s in acc:  # ICL means per rule, like the runs report them
            for m, parts in MEANS.items():
                have = [acc[s][l] for l in parts if l in acc[s]]
                if have:
                    acc[s][m] = round(sum(have) / len(have), 1)
        out["arms"][arm][cond] = dict(
            acc=acc, n=SC.counts(recs), chance=SC.chance(recs), bayes_b={k: round(v, 3) for k, v in params["bayes_b"].items()},
            rstd={s: SC.rstd(recs, s, params) for s in ("mean", "pmi_dc", "hybrid", "mcf")},
            hist={lv: dict(sorted(c.items())) for lv, c in SC.histogram(recs, "mean", params).items()})
        for lv in acc["mean"]:
            if lv not in levels_seen:
                levels_seen.append(lv)
levels = [l for l in LEVELS + list(MEANS) if l in levels_seen] + [l for l in levels_seen if l not in LEVELS and l not in MEANS]


# ------------------------------------------------------------------ analyses
def cell(arm, cond, rule, lv):
    return out["arms"][arm][cond]["acc"][rule].get(lv)


movers = []  # (cond, arm, level, mean, lo_rule, lo, hi_rule, hi, halfwidth)
for cond in ("noctx", "ctx"):
    for arm in arms:
        if cond not in out["arms"][arm]:
            continue
        n_of = out["arms"][arm][cond]["n"]
        for lv in levels:
            vals = {r: cell(arm, cond, r, lv) for r in SC.REAL if cell(arm, cond, r, lv) is not None}
            if len(vals) < 2 or lv in MEANS:
                continue
            lo_r, hi_r = min(vals, key=vals.get), max(vals, key=vals.get)
            hw = SC.halfwidth(vals["mean"], n_of[lv])
            if vals[hi_r] - vals[lo_r] > hw:
                movers.append((cond, arm, lv, vals["mean"], lo_r, vals[lo_r], hi_r, vals[hi_r], hw))

artifacts = []  # question-free (unc) beats chance beyond noise
constant = []   # one option predicted for >= 80% of a level's items under the mean rule
for arm in arms:
    for cond, d in out["arms"][arm].items():
        for lv in levels:
            if lv in MEANS or lv not in d["n"]:
                continue
            u, ch, n = d["acc"]["unc"][lv], d["chance"][lv], d["n"][lv]
            if u - ch > SC.halfwidth(ch, n):
                artifacts.append((cond, arm, lv, u, ch, n))
            top_idx, top = max(d["hist"][lv].items(), key=lambda kv: kv[1])
            if top / n >= 0.8 and len(d["hist"][lv]) >= 1 and n >= 8:
                constant.append((cond, arm, lv, top_idx, round(100 * top / n), d["acc"]["mean"][lv]))
out["movers"] = [dict(cond=c, arm=a_, level=l, mean=m, low_rule=lr, low=lo, high_rule=hr, high=hi, halfwidth=hw)
                 for c, a_, l, m, lr, lo, hr, hi, hw in movers]
out["unc_artifacts"] = [dict(cond=c, arm=a_, level=l, unc=u, chance=ch, n=n) for c, a_, l, u, ch, n in artifacts]
out["constant_predictors"] = [dict(cond=c, arm=a_, level=l, option_index=i, share=s, mean_acc=m) for c, a_, l, i, s, m in constant]


# ------------------------------------------------------------------ write
def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join("" if v is None else (f"{v:g}" if isinstance(v, (int, float)) else str(v)) for v in r) + " |" for r in rows])


def lab(arm):
    return LABEL.get(arm, arm)


md = [f"# Scoring rules on the saved per-item scores: {a.tag}", "",
      "Generated by `scripts/scorer_table.py` from `results/per_item/`. The runs' reported numbers are the `mean` rule. "
      "Every other column re-scores the same forward passes; nothing here needed a model. Rule definitions: "
      "`src/ai_experiments/scorers.py`.", ""]
md += ["## Cells that move", "",
       f"A cell moves when the spread across the real rules ({', '.join(SC.REAL)}) exceeds the 95% binomial half-width "
       "for that level's item count (so seed noise is not the reason). `unc` is excluded: it is a diagnostic.", ""]
md.append(table(["condition", "arm", "level", "mean (reported)", "lowest rule", "low", "highest rule", "high", "half-width"],
                [[c, lab(a_), l, m, lr, lo, hr, hi, hw] for c, a_, l, m, lr, lo, hr, hi, hw in movers]) if movers else "none")
md += ["", "## Question-free baseline above chance (UNC artifact check)", "",
       "`unc` scores the options after the cue line alone; the question is never seen. A level where that beats chance "
       "by more than the half-width can be answered partly from the option strings.", ""]
md.append(table(["condition", "arm", "level", "unc", "chance", "n"], [[c, lab(a_), l, u, ch, n] for c, a_, l, u, ch, n in artifacts]) if artifacts else "none")
md += ["", "## Constant predictors under the mean rule", "",
       "Levels where one option index takes at least 80% of the predictions.", ""]
md.append(table(["condition", "arm", "level", "option index", "share %", "accuracy"], [[c, lab(a_), l, i, s, m] for c, a_, l, i, s, m in constant]) if constant else "none")
for cond in ("noctx", "ctx"):
    md += ["", f"## {'Without context' if cond == 'noctx' else 'With field-guide context'}: one table per rule", ""]
    for rule in SC.SCORERS:
        arms_c = [x for x in arms if cond in out["arms"][x]]
        if not arms_c:
            continue
        md += [f"### `{rule}`" + (" (diagnostic)" if rule in SC.DIAGNOSTIC else ""), ""]
        rows = [[lv] + [cell(x, cond, rule, lv) for x in arms_c] for lv in levels
                if any(cell(x, cond, rule, lv) is not None for x in arms_c)]
        md.append(table(["level"] + [lab(x) for x in arms_c], rows)); md.append("")
md += ["## RStd (recall standard deviation across option indices, percentage points; mean rule, no context)", ""]
arms_n = [x for x in arms if "noctx" in out["arms"][x]]
md.append(table(["level"] + [lab(x) for x in arms_n],
                [[lv] + [out["arms"][x]["noctx"]["rstd"]["mean"].get(lv) for x in arms_n] for lv in levels if lv not in MEANS]))
md += ["", "## RStd under `mcf` (letter position bias; no context)", ""]
md.append(table(["level"] + [lab(x) for x in arms_n],
                [[lv] + [out["arms"][x]["noctx"]["rstd"]["mcf"].get(lv) for x in arms_n] for lv in levels if lv not in MEANS]))
md += ["", "## Fitted length slope b per level (`bayes` rule; log-prob per option token; no context)", ""]
md.append(table(["level"] + [lab(x) for x in arms_n],
                [[lv] + [out["arms"][x]["noctx"]["bayes_b"].get(lv) for x in arms_n] for lv in levels if lv not in MEANS]))
md += ["", "## Predicted-option histograms (mean rule, no context)", "", "`index:count` per level.", ""]
md.append(table(["level"] + [lab(x) for x in arms_n],
                [[lv] + [" ".join(f"{i}:{c}" for i, c in out["arms"][x]["noctx"]["hist"].get(lv, {}).items()) for x in arms_n]
                 for lv in levels if lv not in MEANS]))

(OUT_ROOT / "results").mkdir(parents=True, exist_ok=True); (OUT_ROOT / "reports").mkdir(parents=True, exist_ok=True)
json_path, md_path = OUT_ROOT / "results" / f"scorers_{a.tag}.json", OUT_ROOT / "reports" / f"scorers_{a.tag}.md"
json_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

# ------------------------------------------------------------------ digest
print(f"wrote {json_path.relative_to(OUT_ROOT)} and {md_path.relative_to(OUT_ROOT)}")
print(f"{len(movers)} cells move beyond noise, {len(artifacts)} levels answerable question-free, {len(constant)} constant predictors")
if movers:
    by_rule = defaultdict(int)
    for c, a_, l, m, lr, lo, hr, hi, hw in movers:
        by_rule[hr] += 1
    print("  highest-scoring rule among movers:", dict(by_rule))
for c, a_, l, m, lr, lo, hr, hi, hw in movers:
    print(f"  {c:5s} {lab(a_):16s} {l:28s} mean {m:5.1f}  {lr}={lo:5.1f} .. {hr}={hi:5.1f}  (±{hw})")
for c, a_, l, u, ch, n in artifacts:
    print(f"  UNC>{'chance':6s} {c:5s} {lab(a_):16s} {l:28s} unc {u:5.1f} vs chance {ch:5.1f} (n={n})")
for c, a_, l, i, s, m in constant:
    print(f"  constant {c:5s} {lab(a_):16s} {l:28s} option {i} {s}% of predictions, acc {m}")
