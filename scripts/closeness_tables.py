"""PLAN step 138 (owner, 2026-09-30: "when our model gives a suggested answer, do we know what other answers it was close to giving and
how close?"). On blind_v1's first-time payees (no earlier filing of the payee), the three-model system's calibrated distribution: where
the right category ranks in the full list, its probability against the top one's, the best reader's ceiling on the same items (`best`:
the rule-based ideal reader; -1 = no basis in the prompt), by what kind of first-time transaction it is; and examples.
usage: uv run python scripts/closeness_tables.py
"""
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
from decision_tables import load, product  # noqa: E402
from ensemble_tables import DEC, ENC  # noqa: E402

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in json.loads((PROCESSED / "blind_v1_others.json").read_text())["items"]}
    ph = json.loads((PROCESSED / "blind_v1_payeehist.json").read_text())
    big = load("results/per_item/real6_dm_decider_decider-35b-a3b_blind_v1.noctx.jsonl")
    P = product([load(sorted(glob.glob(ENC))[0]), load(sorted(glob.glob(DEC))[0]), big], items)
    first = [i for i in sorted(P) if not ph[i]]
    groups = {"all first-time payees": first}
    for w in sorted({items[i]["why"] for i in first}, key=lambda w: -sum(items[i]["why"] == w for i in first)):
        sel = [i for i in first if items[i]["why"] == w]
        if len(sel) >= 10:
            groups[w] = sel
    print("**Table CL.1: first-time payees, encoder x decider x 35B: where the right category ranks, and the ideal reader's ceiling**\n")
    print("| group | n | rank 1 | 2 | 3 | 4-5 | 6-10 | >10 | median categories | ideal reader right % | no basis in the prompt % | p(top) when wrong | p(right) when wrong | p(right) / p(top) when wrong |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for g, sel in groups.items():
        ranks = np.array([int(np.where(np.argsort(-P[i]) == items[i]["answer"])[0][0]) + 1 for i in sel])
        wrong = [i for i, r in zip(sel, ranks) if r > 1]
        pt = np.mean([P[i].max() for i in wrong]) if wrong else np.nan; pr = np.mean([P[i][items[i]["answer"]] for i in wrong]) if wrong else np.nan
        ratio = np.median([P[i][items[i]["answer"]] / P[i].max() for i in wrong]) if wrong else np.nan
        b = [items[i]["best"] for i in sel]
        cells = [100 * np.mean(ranks == 1), 100 * np.mean(ranks == 2), 100 * np.mean(ranks == 3), 100 * np.mean((ranks >= 4) & (ranks <= 5)),
                 100 * np.mean((ranks >= 6) & (ranks <= 10)), 100 * np.mean(ranks > 10)]
        print(f"| {g} | {len(sel)} | " + " | ".join(f"{c:.0f}%" for c in cells) + f" | {np.median([len(items[i]['options']) for i in sel]):.0f} | "
              f"{100 * np.mean([x == items[i]['answer'] for x, i in zip(b, sel)]):.0f}% | {100 * np.mean([x == -1 for x in b]):.0f}% | {pt:.2f} | {pr:.2f} | {ratio:.2f} |")
    print("\n**Examples (first-time payees the system got wrong): the top five with probabilities; the right one marked**\n")
    rng = np.random.default_rng(3)
    wrong = [i for i in first if int(np.argmax(P[i])) != items[i]["answer"]]
    for i in rng.choice(wrong, 8, replace=False):
        it = items[i]; opts = [o.strip() for o in it["options"]]; order = np.argsort(-P[i])
        top = ", ".join(f"{'**' if c == it['answer'] else ''}{opts[c]} {100 * P[i][c]:.0f}%{'**' if c == it['answer'] else ''}" for c in order[:5])
        r = int(np.where(order == it["answer"])[0][0]) + 1
        print(f"- `{it['text'][:48]}` ({it['why']}, {len(opts)} categories): {top}" + ("" if r <= 5 else f"; right one **{opts[it['answer']]}** at rank {r} ({100 * P[i][it['answer']]:.1f}%)"))
