"""Constructed-response results next to the cloze rules, with three-way agreement (PLAN step 3, EVAL-3).

usage: uv run python scripts/gen_table.py [Qwen2.5-3B] [--per-item DIR] [--out-root DIR]

Reads the raw generations in results/per_item/gen_<stem>.<cond>.jsonl (scripts/gen_eval.py), matches
them to the frozen items' options HERE (so the matching rule can change without a GPU), joins the
cloze records results/per_item/<stem>.<cond>.jsonl for the same items, and reports per level and arm:

  exact       the generation's first line, normalised, equals the gold option text
  fuzzy       the option the generation names: an option whose text contains the generation or is
              contained in it (unique, 4+ characters), else the highest difflib ratio at 0.6 or more;
              nothing matched counts as unanswered (pred -1)
  listed      options listed in the prompt: the letter the model wrote, else the same fuzzy match
  mean, pmi_dc  the two cloze rules on the same items
  agree3      share of items where fuzzy generation, mean and pmi_dc pick the same option
  gen=mean, gen=pmi   pairwise agreement of the fuzzy generation with each rule

Writes results/gen_table_<tag>.json and reports/gen_<tag>.md and prints the digest. gen_eval.py's
own first-pass fields (exact, fuzzy_pred, listed_pred) stay in the per-item files but are not used.
"""
import argparse
import difflib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from ai_experiments.paths import ROOT
from ai_experiments import items as I
from ai_experiments import merchants as M
from ai_experiments import scorers as SC
from ai_experiments.scoring import LETTERS

ARMS = ["base", "A", "B", "C", "Cn", "D", "base_m", "E", "P"]
LABEL = {"base": "base", "A": "A know", "B": "B epis", "C": "C inter+R", "Cn": "Cn inter", "D": "D seq",
         "base_m": "base(m)", "E": "E morph", "P": "P distill",
         "universe_Qwen2.5-0.5B_lr0.0001": "U-0.5B (sec 6)", "universe_Qwen2.5-3B_lr0.0001": "U-3B (sec 6)",
         "universe_Qwen2.5-3B_lr0.0001_unsloth": "U-3B unsloth (sec 6)", "merchants_Qwen2.5-0.5B": "0.5B merchants (sec 4)"}
LEVELS = ["L1_recall", "L1_recall_fmt", "L3_induct_type_nonsense", "L3_induct_type_realnames", "L3_induct_type_k2",
          "L3_induct_type_k4", "L3_induct_heldout", "clean_category", "bank_category", "sells"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("tag", nargs="?", default="Qwen2.5-3B")
ap.add_argument("--per-item", default=str(ROOT / "results" / "per_item"))
ap.add_argument("--out-root", default=str(ROOT))
a = ap.parse_args()
PER_ITEM, OUT_ROOT = Path(a.per_item), Path(a.out_root)


# ------------------------------------------------------------------ matching (CPU, from the raw text)
def norm(s: str) -> str:
    s = " ".join(s.strip().split()).lower()
    return re.sub(r"[\s\.\,\;\:\!\?\"']+$", "", s)


def first_line(gen: str) -> str:
    return gen.strip().split("\n")[0] if gen and gen.strip() else ""


def fuzzy(gen: str, options: list[str]) -> tuple[int, float]:
    g = norm(first_line(gen))
    if not g:
        return -1, 0.0
    opts = [norm(o) for o in options]
    contains = [i for i, o in enumerate(opts) if o and (o in g or (len(g) >= 4 and g in o))]
    if len(contains) == 1:
        return contains[0], 1.0
    if len(contains) > 1:  # several options share the fragment: take the longest overlap
        best = max(contains, key=lambda i: len(opts[i]) if opts[i] in g else len(g))
        return best, 0.9
    ratios = [difflib.SequenceMatcher(None, g, o).ratio() for o in opts]
    best = max(range(len(opts)), key=ratios.__getitem__)
    return (best if ratios[best] >= 0.6 else -1), round(ratios[best], 3)


def listed(gen: str, options: list[str]) -> int:
    line = first_line(gen)
    m = re.match(r"^\s*\(?([A-Z])(?:[\.\):,\s]|$)", line)
    if m and m.group(1) in LETTERS[:len(options)]:
        return LETTERS.index(m.group(1))
    return fuzzy(gen, options)[0]


# ------------------------------------------------------------------ items per stem
_frozen = {}


def items_for(stem: str) -> dict[str, dict]:
    if stem.startswith("merchants_"):
        ms = M.build(); its = [it for it in M.eval_items(ms) if it["task"] in ("clean_category", "bank_category", "sells")]
        seen = Counter(); out = {}
        for it in its:
            out[f"{it['task']}:{seen[it['task']]:03d}"] = it; seen[it["task"]] += 1
        return out
    morph = stem.endswith(("_E", "_base_m"))
    if morph not in _frozen:
        _frozen[morph] = {it["id"]: it for it in I.load_all(morph=morph).ladder}
    return _frozen[morph]


# ------------------------------------------------------------------ load and score
runs = []
for p in sorted(PER_ITEM.glob("gen_*.jsonl")):
    if "_smoke" in p.name:
        continue
    stem, cond = p.name[len("gen_"):-len(".jsonl")].rsplit(".", 1)
    runs.append((stem, cond, p))
if not runs:
    raise SystemExit(f"no gen_*.jsonl under {PER_ITEM}")


def arm_label(stem):
    if stem.startswith(f"curriculum_{a.tag}_"):
        return LABEL.get(stem[len(f"curriculum_{a.tag}_"):], stem)
    return LABEL.get(stem.removeprefix("rescore_"), stem)


out, rows = {}, []
for stem, cond, p in runs:
    gen = {r["id"]: r for r in SC.read_records(p)}
    items = items_for(stem)
    cloze_path = PER_ITEM / f"{stem}.{cond}.jsonl"
    cloze = {r["id"]: r for r in SC.read_records(cloze_path)} if cloze_path.exists() else {}
    params = SC.params_for(list(cloze.values())) if cloze else None
    per = defaultdict(Counter)
    for i, g in gen.items():
        it = items[i]; opts = it["options"]; gold = it["answer"]
        ex = norm(first_line(g["gen"])) == norm(opts[gold])
        fz, _ = fuzzy(g["gen"], opts)
        li = listed(g["listed_gen"], opts)
        c = per[g["level"]]
        c["n"] += 1; c["exact"] += ex; c["fuzzy"] += fz == gold; c["listed"] += li == gold; c["unanswered"] += fz < 0
        if i in cloze:
            pm, pp = SC.predict(cloze[i], "mean", params), SC.predict(cloze[i], "pmi_dc", params)
            c["n_cloze"] += 1; c["mean"] += pm == gold; c["pmi"] += pp == gold
            c["agree3"] += fz == pm == pp; c["gen=mean"] += fz == pm; c["gen=pmi"] += fz == pp
    out[f"{stem}|{cond}"] = {}
    for lv, c in per.items():
        n, nc = c["n"], c["n_cloze"] or 1
        d = dict(n=n, exact=round(100 * c["exact"] / n, 1), fuzzy=round(100 * c["fuzzy"] / n, 1), listed=round(100 * c["listed"] / n, 1),
                 unanswered=round(100 * c["unanswered"] / n, 1))
        if c["n_cloze"]:
            d.update(mean=round(100 * c["mean"] / nc, 1), pmi_dc=round(100 * c["pmi"] / nc, 1), agree3=round(100 * c["agree3"] / nc, 1),
                     gen_eq_mean=round(100 * c["gen=mean"] / nc, 1), gen_eq_pmi=round(100 * c["gen=pmi"] / nc, 1))
        out[f"{stem}|{cond}"][lv] = d
        rows.append((cond, arm_label(stem), lv, d))

levels = [l for l in LEVELS if any(r[2] == l for r in rows)] + sorted({r[2] for r in rows} - set(LEVELS))


def table(headers, rs):
    fmt = lambda v: "" if v is None else (f"{v:g}" if isinstance(v, (int, float)) else str(v))
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rs])


md = [f"# Constructed-response evaluation: {a.tag}", "",
      "Generated by `scripts/gen_table.py` from the raw generations in `results/per_item/gen_*.jsonl` (greedy decoding, 32 tokens, "
      "`scripts/gen_eval.py`) and the cloze records on the same items. `exact`: first line equals the gold option; `fuzzy`: the option the "
      "generation names (containment either way, else difflib ratio at 0.6 or more; unanswered below); `listed`: options listed in the prompt, "
      "letter or text parsed. `mean` and `pmi_dc` are the cloze rules; `agree3` is the share of items where fuzzy generation, mean and "
      "pmi_dc pick the same option.", ""]
for cond in sorted({r[0] for r in rows}, key=lambda c: ("ctx" in c, "incontext" in c, c)):
    md += [f"## condition `{cond}`", ""]
    md.append(table(["arm", "level", "n", "exact", "fuzzy", "listed", "unanswered", "mean", "pmi_dc", "agree3", "gen=mean", "gen=pmi"],
                    [[arm, lv, d["n"], d["exact"], d["fuzzy"], d["listed"], d["unanswered"], d.get("mean"), d.get("pmi_dc"),
                      d.get("agree3"), d.get("gen_eq_mean"), d.get("gen_eq_pmi")]
                     for lv in levels for c, arm, l, d in rows if c == cond and l == lv]))
    md.append("")

(OUT_ROOT / "results").mkdir(parents=True, exist_ok=True); (OUT_ROOT / "reports").mkdir(parents=True, exist_ok=True)
jp, mp = OUT_ROOT / "results" / f"gen_table_{a.tag}.json", OUT_ROOT / "reports" / f"gen_{a.tag}.md"
jp.write_text(json.dumps(out, indent=1), encoding="utf-8"); mp.write_text("\n".join(md) + "\n", encoding="utf-8")
print(f"wrote {jp.relative_to(OUT_ROOT)} and {mp.relative_to(OUT_ROOT)}")
for cond, arm, lv, d in rows:
    extra = f" | mean {d['mean']:5.1f} pmi {d['pmi_dc']:5.1f} | agree3 {d['agree3']:5.1f}" if "mean" in d else ""
    print(f"  {cond:12s} {arm:22s} {lv:26s} exact {d['exact']:5.1f} fuzzy {d['fuzzy']:5.1f} listed {d['listed']:5.1f} unanswered {d['unanswered']:5.1f}{extra}")
