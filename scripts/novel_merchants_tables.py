"""Tables for REPORT.md section 57 (PLAN step 62, REAL-19): the novel-merchant set (1,198 real US places from Overture, in no database
and no history), three renderings of the same items, row 58's bf16 fold adapters on their held-out users and the untrained model.
  57.1  per arm and rendering: the scorecard's top-1 / top-3 / bits and top-1 by name group and by the user's category name type
  57.2  paired: clean minus obscure per arm [user interval]
  57.3  descriptive names by whether the category word survives the obscure rendering, and by the user's category (standard / renamed /
        merged / coined)
  57.4  the blind Opus ceiling (scripts/blind_ceiling.py)
usage: uv run python scripts/novel_merchants_tables.py
"""
import json
import re
import subprocess
import sys

import numpy as np

from ai_experiments import real6 as R6
from ai_experiments import real6_cells as RC
from ai_experiments import scorecard as S
from ai_experiments.paths import PROCESSED, ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from build_novel_merchants import WORDS  # noqa: E402

CAT = "real6_categoriser_Qwen2.5-3B-Instruct"
V = [("A obscure", ""), ("B full name + bank noise", "_full"), ("C clean", "_clean")]
ARMS = [("untrained instruct, no record", "real6_Qwen2.5-3B-Instruct_novel_merchants_v1{v}.noctx.jsonl"),
        ("untrained instruct + Overture record", "real6_Qwen2.5-3B-Instruct_novel_merchants_v1{v}.ctx.jsonl"),
        ("SFT no DB", f"{CAT}_none_h100bf16_f?_alllab_lora_novel_merchants_v1{{v}}.noctx.jsonl"),
        ("database episodes", f"{CAT}_none_h100bf16_f?_alllab_dbep50_lora_novel_merchants_v1{{v}}.noctx.jsonl"),
        ("record arm + Overture record", f"{CAT}_ret_h100bf16_f?_lora_novel_merchants_v1{{v}}.ctx.jsonl"),
        ("record + category arm + Overture record", f"{CAT}_ret_h100bf16_f?_reccat_lora_novel_merchants_v1{{v}}.ctx.jsonl")]


def items(v):
    return {i["id"]: i for i in json.loads((PROCESSED / f"novel_merchants_v1{v}.json").read_text())["items"]}


def acc(rs, its, f):
    x = [r["correct"] for i, r in rs.items() if f(its[i])]
    return f"{100 * np.mean(x):.1f}" if x else "-"


def t1():
    print("**Table 57.1: the novel merchants (held-out users; top-1, top-3 and bits from the scorecard: options ranked by summed log-probability, bits after the leave-users-out temperature; the group columns are top-1 under the mean-per-token rule of every earlier REAL-6 table; the two rules differ by under a point)**\n")
    print("| arm | rendering | top-1 | top-3 | bits | descriptive | plain | chain | standard name | renamed | coined |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for label, pat in ARMS:
        for vl, v in V:
            its = items(v); rs = RC.load_recs(pat.format(v=v))
            if not rs:
                continue
            sc = S.scorecard(rs, its)
            print(f"| {label} | {vl} | {sc['top1']:.1f} | {sc['top3']:.1f} | {sc['bits']:.2f} | " + " | ".join(acc(rs, its, f) for f in (
                lambda x: x["nm_group"] == "descriptive", lambda x: x["nm_group"] == "plain", lambda x: x["nm_group"] == "chain",
                lambda x: x["name_type"] == "standard", lambda x: x["name_type"] == "renamed", lambda x: x["name_type"] == "new")) + " |")


def t2():
    print("\n**Table 57.2: clean minus obscure on the same items, top-1 points [user-resampled interval]**\n")
    print("| arm | all | descriptive | plain | chain | coined |")
    print("|---|---|---|---|---|---|")
    its = items(""); iu = {i: x["user"] for i, x in its.items()}
    for label, pat in ARMS:
        A, C = RC.load_recs(pat.format(v="")), RC.load_recs(pat.format(v="_clean"))
        cells = []
        for f in (lambda x: True, lambda x: x["nm_group"] == "descriptive", lambda x: x["nm_group"] == "plain", lambda x: x["nm_group"] == "chain", lambda x: x["name_type"] == "new"):
            ids = [i for i in A if f(its[i])]; us = sorted({iu[i] for i in ids}); by = {u: [C[i]["correct"] - A[i]["correct"] for i in ids if iu[i] == u] for u in us}
            rng = np.random.default_rng(0); b = [100 * np.mean(np.concatenate([by[u] for u in rng.choice(us, len(us))])) for _ in range(1000)]
            cells.append(f"{100 * np.mean([C[i]['correct'] - A[i]['correct'] for i in ids]):+.1f} [{np.percentile(b, 2.5):+.1f}, {np.percentile(b, 97.5):+.1f}]")
        print(f"| {label} | " + " | ".join(cells) + " |")


def t3():
    its = items(""); users = {u["user"]: u for u in R6.load()["users"]}
    desc = [i for i, x in its.items() if x["nm_group"] == "descriptive"]
    surv = lambda x: bool(re.search(rf"\b({WORDS[x['std']]})\b", x["text"], re.I))  # noqa: E731
    merged = lambda x: len(users[x["user"]]["categories"][x["answer"]]["standard"]) > 1  # noqa: E731
    groups = [("word survives the rendering", surv), ("word lost", lambda x: not surv(x)), ("standard single category", lambda x: x["name_type"] == "standard" and not merged(x)),
              ("renamed single", lambda x: x["name_type"] == "renamed" and not merged(x)), ("merged", lambda x: merged(x) and x["name_type"] == "renamed"),
              ("coined", lambda x: x["name_type"] == "new"), ("word survives + standard single", lambda x: surv(x) and x["name_type"] == "standard" and not merged(x))]
    print(f"\n**Table 57.3: descriptive names in the obscure rendering (n = {len(desc)}; the category word survives in {100 * np.mean([surv(its[i]) for i in desc]):.0f}%; accuracy %)**\n")
    print("| arm | " + " | ".join(f"{g} (n={sum(f(its[i]) for i in desc)})" for g, f in groups) + " |")
    print("|---|" + "---|" * len(groups))
    for label, pat in [ARMS[0], ARMS[2], ARMS[3], ARMS[4]]:
        rs = RC.load_recs(pat.format(v=""))
        print(f"| {label} | " + " | ".join(f"{100 * np.mean([rs[i]['correct'] for i in desc if f(its[i])]):.1f}" for _, f in groups) + " |")


if __name__ == "__main__":
    t1(); t2(); t3()
    print("\n**Table 57.4: a blind Opus 5.5 ceiling on 60 items (40 with coined category names), obscure (A) and clean (C) renderings (accuracy %)**\n")
    sys.stdout.flush(); subprocess.run([sys.executable, str(ROOT / "scripts" / "blind_ceiling.py"), "score", "novel_merchants_v1"], check=True)
