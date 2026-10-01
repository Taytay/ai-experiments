"""Tables for PLAN step 52 (REAL-14): does a trained "not listed here" option flag the questions a categoriser cannot answer better than a
confidence threshold does? decider-4B one slot + shot-label loss + database episodes (row 80) without the option in training, and with
decider's augmentation (exp_decider_finetune.py ABSTAIN / ABSTAIN_SWAP), read on real6_v1_abst (build_real6_abstain.py): every fold-0 item
with the option added (full) and with its gold category hidden (nogold, where the option is the answer).

  A.1  per model: top-1 on full items over the real options (the option's score dropped), and with it (choosing it counts as wrong);
       how often the option is chosen on full items (false alarms) and on nogold items (recall); AUROC for telling nogold from full
       items by P(option) and by 1 - the top real option's probability (the threshold route, open to every model); REAL-6 top-1
       without the option (the ordinary set)
usage: uv run python scripts/abstain_tables.py
"""
import json
import warnings

import numpy as np

from ai_experiments import real6 as R6
from ai_experiments import real6_cells as RC
from ai_experiments.paths import PROCESSED

F = "real6_dm_decider_decider_decider-4b_none_h100bf16st800_f0_dbep50_aux100{a}_lora_{s}.noctx.jsonl"
MODELS = [("decider-4B, no abstain training (row 80)", ""), ("decider-4B, abstain 0.1, hidden gold 0.25 (decider's)", "_abst10sw25"),
          ("decider-4B, abstain 0.2, hidden gold 0.5", "_abst20sw50")]


def softmax(z):
    z = np.asarray(z, float); z = z - z.max(); e = np.exp(z); return e / e.sum()


def auroc(pos, neg):
    """P(a random positive scores above a random negative), ties half."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in json.loads((PROCESSED / "real6_v1_abst.json").read_text())["items"]}
    r6 = {i["id"]: i for i in R6.load()["items"]}
    print("**Table A.1: REAL-6 fold 0 with a \"not listed here\" option: full items (gold present) and nogold items (gold category hidden; "
          "the option is right)**\n")
    print("| model | n full / nogold | top-1, real options | top-1 with the option | option chosen, full | option chosen, nogold (recall) "
          "| AUROC nogold vs full: P(option) | AUROC: 1 - top real p | REAL-6 top-1 (no option) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for lab, a in MODELS:
        recs = {i: r for i, r in RC.load_recs(F.format(a=a, s="real6_v1_abst")).items() if i in items and items[i]["user"] % 4 == 0}
        if not recs:
            continue
        full = [i for i in recs if items[i]["abst_case"] == "full"]; nog = [i for i in recs if items[i]["abst_case"] == "nogold"]
        z = {i: np.asarray(recs[i]["sum_lp"], float) for i in recs}
        p = {i: softmax(z[i]) for i in recs}; pr = {i: softmax(z[i][:-1]) for i in recs}  # with the option; the real options only
        top_real = np.mean([int(np.argmax(z[i][:-1])) == items[i]["answer"] for i in full])
        top_with = np.mean([int(np.argmax(z[i])) == items[i]["answer"] for i in full])
        fa = np.mean([int(np.argmax(z[i])) == len(z[i]) - 1 for i in full]); rec = np.mean([int(np.argmax(z[i])) == len(z[i]) - 1 for i in nog])
        au_opt = auroc([p[i][-1] for i in nog], [p[i][-1] for i in full])
        au_thr = auroc([1 - pr[i].max() for i in nog], [1 - pr[i].max() for i in full])
        plain = {i: r for i, r in RC.load_recs(F.format(a=a, s="real6")).items() if i in r6 and r6[i]["user"] % 4 == 0}
        t1 = f"{100 * np.mean([int(np.argmax(r['sum_lp'])) == r['answer'] for r in plain.values()]):.1f}" if plain else "–"
        print(f"| {lab} | {len(full)} / {len(nog)} | {100 * top_real:.1f} | {100 * top_with:.1f} | {100 * fa:.1f}% | {100 * rec:.1f}% | "
              f"{au_opt:.3f} | {au_thr:.3f} | {t1} |")
