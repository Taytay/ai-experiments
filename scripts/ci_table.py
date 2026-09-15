"""Confidence intervals, empirical null bands and paired tests from the per-item records (PLAN step 4).

usage: uv run python scripts/ci_table.py [Qwen2.5-3B] [--rule mean] [--per-item DIR] [--out-root DIR] [--draws 1000]

For every arm and condition: per-level accuracy under one rule (default `mean`, the reported one),
its 95% bootstrap CI, and the 95% null band from permuting gold indices over the saved predictions
(EVAL-6: is the level above what a gold-blind predictor scores on these items?). For the headline
comparisons of REPORT.md section 8 (each trained arm against its base, C against A, Cn, D; E against
base_m), a paired table: difference, bootstrap CI, flip counts and exact McNemar p on the shared item
ids. Arms on different universes (E, base_m vs the rest) are never paired.

  results/ci_<tag>.json    everything below as data
  reports/ci_<tag>.md      the tables
"""
import argparse
import json
from pathlib import Path

from ai_experiments.paths import ROOT
from ai_experiments import scorers as SC
from ai_experiments import stats as ST

ARMS = ["base", "A", "B", "C", "Cn", "D", "base_m", "E", "P", "P2"]
LABEL = {"base": "base", "A": "A know", "B": "B epis", "C": "C inter+R", "Cn": "Cn inter", "D": "D seq",
         "base_m": "base(m)", "E": "E morph", "P": "P distill", "P2": "P2 distill-opt"}
PAIRS = [("base", "A"), ("base", "B"), ("base", "C"), ("base", "Cn"), ("base", "D"), ("A", "C"), ("Cn", "C"), ("D", "C"),
         ("base_m", "E"), ("base", "P"), ("C", "P"), ("P", "P2"), ("C", "P2"), ("A", "P2")]
LEVELS = ["L1_recall", "L1_recall_fmt", "L2_manip_isa", "L2_manip_pair", "L3_induct_type_nonsense",
          "L3_induct_type_realnames", "L3_induct_type_k2", "L3_induct_type_k4", "L4_induct_weakness",
          "L4_induct_habitat", "L5_novel_choices", "L6_unseen_recall", "L6_seen_recall_ctrl", "L3_induct_heldout",
          "M_probe_marked", "M_probe_plain", "M_probe_marked_fmt", "M_probe_plain_fmt",
          "ICL_symbol_banking77", "ICL_symbol_dbpedia", "ICL_symbol_sst2", "ICL_symbol_subj",
          "ICL_natural_banking77", "ICL_natural_dbpedia", "ICL_natural_sst2", "ICL_natural_subj", "ICL_symbol_all", "ICL_natural_all"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("tag", nargs="?", default="Qwen2.5-3B")
ap.add_argument("--rule", default="mean", choices=list(SC.SCORERS))
ap.add_argument("--arms", nargs="*", default=ARMS)
ap.add_argument("--per-item", default=str(ROOT / "results" / "per_item"))
ap.add_argument("--out-root", default=str(ROOT))
ap.add_argument("--draws", type=int, default=1000)
a = ap.parse_args()
PER_ITEM, OUT_ROOT = Path(a.per_item), Path(a.out_root)


def load_arm(arm):
    conds = ("base", "base_ctx") if arm.startswith("base") else ("trained", "trained_ctx")
    paths = [PER_ITEM / f"curriculum_{a.tag}_{arm}.{c}.jsonl" for c in conds]
    if not all(p.exists() for p in paths):
        return None
    return {"noctx": SC.read_records(paths[0]), "ctx": SC.read_records(paths[1])}


data = {arm: d for arm in a.arms if (d := load_arm(arm))}
if not data:
    raise SystemExit(f"no per-item files under {PER_ITEM}")


def with_pooled(recs):
    """Add copies of the ICL suite items relabelled ICL_symbol_all / ICL_natural_all (192 items each), so the
    suite means the report quotes get an interval and a paired test of their own."""
    pooled = [dict(r, level=r["level"].rsplit("_", 1)[0] + "_all", id=r["id"] + "|all") for r in recs if r["level"].startswith("ICL_")]
    return recs + pooled


data = {arm: {c: with_pooled(r) for c, r in d.items()} for arm, d in data.items()}
params = {arm: {c: SC.params_for(r) for c, r in d.items()} for arm, d in data.items()}
arms = list(data)
print(f"{len(arms)} arms: {', '.join(arms)}; rule {a.rule}; {a.draws} draws", flush=True)

out = {"tag": a.tag, "rule": a.rule, "draws": a.draws, "arms": {}, "paired": {}}
levels_seen = []
for arm in arms:
    out["arms"][arm] = {}
    for cond, recs in data[arm].items():
        p = params[arm][cond]
        acc = SC.accuracy(recs, a.rule, p)
        ci = ST.bootstrap_ci(recs, a.rule, p, n_boot=a.draws)
        null = ST.null_band(recs, a.rule, p, n_perm=a.draws)
        out["arms"][arm][cond] = {lv: dict(acc=acc[lv], ci=ci[lv], null=null[lv], n=n, chance=SC.chance(recs)[lv],
                                           above_null=acc[lv] > null[lv][1])
                                  for lv, n in SC.counts(recs).items()}
        for lv in acc:
            if lv not in levels_seen:
                levels_seen.append(lv)
    print(f"  {arm}: done", flush=True)
levels = [l for l in LEVELS if l in levels_seen] + [l for l in levels_seen if l not in LEVELS]

for x, y in PAIRS:
    if x in data and y in data:
        for cond in ("noctx", "ctx"):
            out["paired"][f"{x}->{y}|{cond}"] = ST.paired(data[x][cond], data[y][cond], a.rule, params[x][cond], params[y][cond], n_boot=a.draws)


def table(headers, rows):
    fmt = lambda v: "" if v is None else (f"{v:g}" if isinstance(v, (int, float)) else str(v))
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows])


def lab(arm):
    return LABEL.get(arm, arm)


md = [f"# Confidence intervals and null bands: {a.tag} (rule `{a.rule}`)", "",
      f"Generated by `scripts/ci_table.py` from `results/per_item/`, {a.draws} draws. Each cell is `acc [ci_lo, ci_hi] | null_lo-null_hi`: "
      "the 95% bootstrap CI over items, then the 95% band of a gold-blind predictor (gold indices permuted over the same "
      "predictions). A bold accuracy is above its null band. Item counts and chance (mean 1/k) follow each table.", ""]
for cond in ("noctx", "ctx"):
    arms_c = [x for x in arms if cond in out["arms"][x]]
    md += [f"## {'Without context' if cond == 'noctx' else 'With field-guide context'}", ""]
    rows = []
    for lv in levels:
        row = [lv]
        for x in arms_c:
            c = out["arms"][x][cond].get(lv)
            if c is None:
                row.append("")
            else:
                acc = f"**{c['acc']:g}**" if c["above_null"] else f"{c['acc']:g}"
                row.append(f"{acc} [{c['ci'][0]:g}, {c['ci'][1]:g}] \\| {c['null'][0]:g}-{c['null'][1]:g}")
        rows.append(row)
    first = next(iter(arms_c))
    md.append(table(["level"] + [lab(x) for x in arms_c], rows)); md.append("")
    md += ["n and chance per level: " + ", ".join(f"{lv} {out['arms'][first][cond][lv]['n']}/{out['arms'][first][cond][lv]['chance']:g}"
                                                    for lv in levels if lv in out["arms"][first][cond]), ""]
md += ["## Paired comparisons on shared items", "",
       "`diff` is B minus A in points with its bootstrap CI; `flips` are items only A got right / only B got right; "
       "`p` is the exact two-sided McNemar test on those discordant items. Arms on different universes are not paired.", ""]
for key, res in out["paired"].items():
    xy, cond = key.split("|")
    x, y = xy.split("->")
    md += [f"### {lab(x)} -> {lab(y)} ({'no context' if cond == 'noctx' else 'with context'})", ""]
    md.append(table(["level", "n", lab(x), lab(y), "diff", "95% CI", "flips A/B", "McNemar p"],
                    [[lv, r["n"], r["acc_a"], r["acc_b"], r["diff"], f"[{r['ci'][0]:g}, {r['ci'][1]:g}]", f"{r['only_a']}/{r['only_b']}", r["p"]]
                     for lv in levels if (r := res.get(lv))]))
    md.append("")

(OUT_ROOT / "results").mkdir(parents=True, exist_ok=True); (OUT_ROOT / "reports").mkdir(parents=True, exist_ok=True)
jp, mp = OUT_ROOT / "results" / f"ci_{a.tag}.json", OUT_ROOT / "reports" / f"ci_{a.tag}.md"
jp.write_text(json.dumps(out, indent=1), encoding="utf-8"); mp.write_text("\n".join(md) + "\n", encoding="utf-8")
print(f"wrote {jp.relative_to(OUT_ROOT)} and {mp.relative_to(OUT_ROOT)}")

# digest: which reported cells sit inside their null band, and which headline differences are not significant
inside = [(arm, cond, lv, c["acc"], c["null"]) for arm in arms for cond in out["arms"][arm] for lv, c in out["arms"][arm][cond].items() if not c["above_null"]]
print(f"{len(inside)} arm x condition x level cells are inside their null band (not above a gold-blind predictor):")
for arm, cond, lv, acc, null in inside:
    print(f"  {cond:5s} {lab(arm):10s} {lv:28s} {acc:5.1f} in null {null[0]}-{null[1]}")
ns = [(k, lv, r) for k, res in out["paired"].items() for lv, r in res.items() if r["p"] >= 0.05 and abs(r["diff"]) >= 5]
print(f"{len(ns)} paired differences of 5+ points with McNemar p >= 0.05:")
for k, lv, r in ns:
    print(f"  {k:22s} {lv:28s} diff {r['diff']:+6.1f} CI [{r['ci'][0]}, {r['ci'][1]}] flips {r['only_a']}/{r['only_b']} p={r['p']}")
