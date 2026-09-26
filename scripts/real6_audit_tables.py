"""Tables for the REAL-6 audit (PLAN step 45, QUESTIONS.md REAL-13): every REAL-6 run of sections 37 to 47 re-read in the corrected
groups of `ai_experiments.real6_cells`, from the saved per-item records (no rescoring).

  48.1  the groups themselves: items, the nearest-row lookup, and what decides each group
  48.2  one row per run: in history / labelled seen, not in it / determined by category / split / DB-only (determined or split
        items whose merchant no user labelled in training) / all items with the item and the user-resampled interval / the sum
        rule (LLM runs) / the lookup-then-model hybrid
  48.3  the seeded arms (sections 43 and 45) as mean +- sd over three seeds in the same groups
usage: uv run python scripts/real6_audit_tables.py            prints markdown
"""
import json

import numpy as np

from ai_experiments import real6 as R6
from ai_experiments import real6_cells as RC
from ai_experiments.paths import ROOT

P = ROOT / "results" / "per_item"
CAT = "categoriser_Qwen2.5-3B-Instruct"
GROUPS = RC.KINDS[:4]
# (section, label, per-item stem and condition)
RUNS = [
    (37, "Qwen2.5-3B base, 24 shots", "Qwen2.5-3B.noctx"), (37, "Qwen2.5-3B base + record", "Qwen2.5-3B.ctx"),
    (37, "Instruct, 24 shots", "Qwen2.5-3B-Instruct.noctx"), (37, "Instruct + record", "Qwen2.5-3B-Instruct.ctx"),
    (37, "MiniLM prototype, 24 shots", "minilm.shots"), (37, "MiniLM prototype, full history", "minilm.full"), (37, "MiniLM mix", "minilm.mix"),
    (37, "bge prototype, full history", "bge.full"), (37, "bge mix", "bge.mix"),
    (38, "bge tuned across users, full history", "categoriser_bge_none.full"), (38, "bge tuned + record on the query", "categoriser_bge_none_ctx.full"),
    (38, "bge tuned + records in training", "categoriser_bge_param.full"),
    (38, "SFT no DB", f"{CAT}_none_lora.noctx"), (38, "SFT, records in the weights (1 pass)", f"{CAT}_param_lora.noctx"),
    (38, "SFT, records in the weights (3.3 passes)", f"{CAT}_param_x2_lora.noctx"), (38, "SFT + record in prompt", f"{CAT}_ret_lora.ctx"),
    (43, "SFT + record, ambiguous DB (trained on it)", f"{CAT}_ret_amb_lora_amb.ctx"), (43, "SFT + record, retrieved top-1", f"{CAT}_ret_lora.ret1"),
    (44, "SFT no DB, transformers + peft (hf scorer)", f"{CAT}_none_hf_lora_hfs.noctx"), (44, "SFT + record, transformers + peft (hf scorer)", f"{CAT}_ret_hf_lora_hfs.ctx"),
    (45, "Instruct, chat template", "Qwen2.5-3B-Instruct_chat.noctx"), (45, "Instruct + record, chat template", "Qwen2.5-3B-Instruct_chat.ctx"),
    (45, "SFT no DB, chat template", f"{CAT}_none_chat_lora.noctx"), (45, "SFT + record, chat template", f"{CAT}_ret_chat_lora.ctx"),
    (46, "FastFit bge, 24 shots", "fastfit_bge.shots"), (46, "FastFit bge, full history", "fastfit_bge.full"),
    (46, "FastFit bge + record, full history", "fastfit_bge_ctx.full"), (46, "logistic head, frozen bge, C=100", "logreg_bge_c100.full"),
    (46, "logistic head + record, C=100", "logreg_bge_c100_ctx.full"),
    (47, "SFT no DB, transact shots at test", f"{CAT}_none_lora_shotstransact.noctx"), (47, "SFT + record, transact shots at test", f"{CAT}_ret_lora_shotstransact.ctx"),
    (47, "SFT + record, trained with transact shots", f"{CAT}_ret_shotstransact_lora.ctx"),
]
SEEDED = [("SFT no DB (sec. 43)", [f"{CAT}_none_lora", f"{CAT}_none_s1_lora", f"{CAT}_none_s2_lora"], "noctx"),
          ("SFT + record (sec. 43)", [f"{CAT}_ret_lora", f"{CAT}_ret_s1_lora", f"{CAT}_ret_s2_lora"], "ctx"),
          ("records in the weights, 3.3 passes (sec. 43)", [f"{CAT}_param_x2_lora", f"{CAT}_param_x2_s1_lora", f"{CAT}_param_x2_s2_lora"], "noctx"),
          ("records in the weights, 6.7 passes (sec. 45)", [f"{CAT}_param_x4_s{s}_lora" for s in range(3)], "noctx"),
          ("records in the weights, 13.3 passes (sec. 45)", [f"{CAT}_param_x8_s{s}_lora" for s in range(3)], "noctx")]


def load(stem):
    p = P / f"real6_{stem}.jsonl"
    return {r["id"]: r for r in map(json.loads, open(p))} if p.exists() else {}


def pct(x):
    return f"{100 * sum(x) / len(x):.1f}" if x else "-"


DB = R6.db_only_merchants()
ITEMS = {it["id"]: it for it in R6.load()["items"]}


def is_db_only(i):
    return RC.kind(i) in GROUPS[2:] and ITEMS[i]["merchant"] in DB


def group_accs(rs, key="correct"):
    out = [pct([r[key] for i, r in rs.items() if RC.kind(i) == g]) for g in GROUPS]
    out.append(pct([r[key] for i, r in rs.items() if is_db_only(i)]))
    return out


def item_ci(c, n_boot=1000):
    x = np.array(list(c.values()), float); g = np.random.default_rng(1)
    b = [100 * x[g.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return np.percentile(b, [2.5, 97.5]).round(1)


def t1():
    nn = RC.nn1(); k = RC.kinds()
    print("**Table 48.1: REAL-6 in the corrected groups (`ai_experiments.real6_cells`); lookup = the label of the user's history row nearest to the "
          "statement under the row 37 retriever**\n")
    print("| group | items | of which DB-only merchants | labelled level | lookup accuracy | what decides the item |")
    print("|---|---|---|---|---|---|")
    why = {GROUPS[0]: "the user's own label for this merchant (one label per user-merchant pair)",
           GROUPS[1]: "the merchant's category, as for an unseen merchant: the 300-row cut removed its rows",
           GROUPS[2]: "the merchant's standard category, renamed or merged by the user's scheme",
           GROUPS[3]: "which half of a split category the user put this merchant in: nothing observable"}
    for g in GROUPS:
        ids = [i for i in k if k[i] == g]
        lv = sorted({ITEMS[i]["level"].rsplit("_", 1)[0].replace("R6_", "") for i in ids})
        print(f"| {g} | {len(ids)} | {sum(ITEMS[i]['merchant'] in DB for i in ids)} | {', '.join(lv)} | {pct([nn[i][1] for i in ids])} | {why[g]} |")
    other = [i for i in k if k[i] == RC.KINDS[4]]
    print(f"| other (the history shows a different label for the category) | {len(other)} | | unseen | {pct([nn[i][1] for i in other])} | |")


def t2():
    print("\n**Table 48.2: every REAL-6 run of sections 37 to 47 in the corrected groups (accuracy %; DB-only = determined or split items whose "
          "merchant no user labelled in training; interval over items, then over users; sum = the option rule summing token log-probabilities "
          "instead of averaging them, LLM runs only; hybrid = the lookup when its cosine is at least 0.8, else the run)**\n")
    print("| sec. | run | " + " | ".join(GROUPS) + " | DB-only | all | item interval | user interval | sum rule, all | hybrid, all |")
    print("|---|---|" + "---|" * (len(GROUPS) + 6))
    for sec, label, stem in RUNS:
        rs = load(stem)
        if not rs:
            print(f"| {sec} | {label} | missing: {stem} |"); continue
        c = {i: r["correct"] for i, r in rs.items()}
        ic, uc = item_ci(c), RC.user_ci(c)
        if "sum_lp" in next(iter(rs.values())):
            s = [int(np.argmax(r["sum_lp"]) == r["answer"]) for r in rs.values()]; sm = pct(s)
        else:
            sm = "-"
        h = RC.hybrid(rs)
        print(f"| {sec} | {label} | " + " | ".join(group_accs(rs)) + f" | {pct(list(c.values()))} | [{ic[0]}, {ic[1]}] | [{uc[0]}, {uc[1]}] | {sm} | {pct(list(h.values()))} |")


def t3():
    print("\n**Table 48.3: the seeded arms in the corrected groups (mean +- sd over seeds 0, 1, 2; accuracy %)**\n")
    print("| arm | " + " | ".join(GROUPS) + " | DB-only | all |")
    print("|---|" + "---|" * (len(GROUPS) + 2))
    for label, stems, cond in SEEDED:
        runs = [load(f"{s}.{cond}") for s in stems]
        if not all(runs):
            print(f"| {label} | missing |"); continue
        cols = []
        for j in range(len(GROUPS) + 2):
            v = []
            for rs in runs:
                if j < len(GROUPS):
                    x = [r["correct"] for i, r in rs.items() if RC.kind(i) == GROUPS[j]]
                elif j == len(GROUPS):
                    x = [r["correct"] for i, r in rs.items() if is_db_only(i)]
                else:
                    x = [r["correct"] for r in rs.values()]
                v.append(100 * sum(x) / len(x))
            cols.append(f"{np.mean(v):.1f} +- {np.std(v, ddof=1):.1f}")
        print(f"| {label} | " + " | ".join(cols) + " |")


if __name__ == "__main__":
    t1(); t2(); t3()
