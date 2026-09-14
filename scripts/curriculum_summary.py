"""Side-by-side table of curriculum v2 arms from results/curriculum_<model>_<arm>.json.

usage: uv run python scripts/curriculum_summary.py [Qwen2.5-3B] [--md]
"""
import json
import sys
from pathlib import Path
from ai_experiments.paths import ROOT

tag = next((a for a in sys.argv[1:] if not a.startswith("--")), "Qwen2.5-3B")
md = "--md" in sys.argv
R = ROOT / "results"
ARMS = ["base", "A", "B", "C", "Cn", "D", "base_m", "E"]
LABEL = {"base": "base", "A": "A know", "B": "B epis", "C": "C inter+R", "Cn": "Cn inter", "D": "D seq",
         "base_m": "base(m)", "E": "E morph"}
ROWS = [  # (metric, source condition-key)
    ("L1_recall_fmt", "noctx"), ("L2_manip_isa", "noctx"), ("L2_manip_pair", "noctx"),
    ("L3_induct_type_nonsense", "noctx"), ("L3_induct_type_k2", "noctx"), ("L3_induct_type_k4", "noctx"),
    ("L3_induct_type_realnames", "noctx"), ("L4_induct_weakness", "noctx"), ("L4_induct_habitat", "noctx"),
    ("L5_novel_choices", "noctx"), ("L3_induct_heldout", "noctx"), ("L6_unseen_recall", "noctx"),
    ("M_probe_marked", "noctx"), ("M_probe_plain", "noctx"),
    ("ICL_symbol_mean", "noctx"), ("ICL_natural_mean", "noctx"), ("L7_ppl_general", "noctx"),
    ("L3_induct_type_nonsense", "ctx"), ("L4_induct_weakness", "ctx"), ("L3_induct_heldout", "ctx"),
    ("L1_recall_fmt", "ctx"),
    ("train_minutes", "noctx"), ("tokens_per_s", "noctx"), ("peak_alloc_GiB", "noctx"),
]

data = {}
for arm in ARMS:
    p = R / f"curriculum_{tag}_{arm}.json"
    if p.exists():
        d = json.loads(p.read_text())
        data[arm] = {"noctx": d.get("trained", d.get("base", {})), "ctx": d.get("trained_ctx", d.get("base_ctx", {}))}
arms = [a for a in ARMS if a in data]
if not arms:
    sys.exit(f"no results for {tag}")


def cell(v):
    return "" if v is None else (f"{v:g}" if isinstance(v, (int, float)) else str(v))


hdr = ["metric"] + [LABEL[a] for a in arms]
rows = []
for m, cond in ROWS:
    name = m + (" (+ctx)" if cond == "ctx" else "")
    rows.append([name] + [cell(data[a][cond].get(m)) for a in arms])
if md:
    print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join(r) + " |")
else:
    w = [max(len(x) for x in col) for col in zip(hdr, *rows)]
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
    for r in rows:
        print("  ".join(x.rjust(w[i]) if i else x.ljust(w[i]) for i, x in enumerate(r)))
