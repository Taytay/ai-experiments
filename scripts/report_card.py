"""PLAN step 133 (owner, 2026-09-30: the rank score is hard to read; do not penalise plausible suggestions: a user may file Amazon under
four categories, so suggesting the others is right). A report card in plain percentages, and the rank score as "work saved".

Plausible categories for a transaction, known when its prompt is built (as production would know them): the categories this user filed
the same payee under in the prompt's rows (payee matched by `oneslot.payee_key`), and the categories among the user's own that other users
file the payee under (blind_v1_others' line). Plausible categories are shown freely (model order, at most five) and never penalised;
any other category is shown only if the model is confident enough (`scorecard.suggest`) and costs LAM when wrong.
Rank score as before (right at rank r: r - 1; not shown 10; wrong auto-file W; accepting a suggestion CONFIRM); "work saved" = 1 - score / 10
(no model: every transaction a search, 10).
usage: uv run python scripts/report_card.py [W] [CONFIRM] [LAM]
"""
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments import oneslot
from ai_experiments import scorecard as S
from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
from decision_tables import folds_calibrate, load, product  # noqa: E402
from ensemble_tables import DEC, ENC  # noqa: E402

BIG = "results/per_item/real6_dm_decider_decider-35b-a3b_blind_v1.noctx.jsonl"
_ARGS = [a for a in sys.argv[1:4] if a.replace(".", "", 1).isdigit()] if Path(sys.argv[0]).name == "report_card.py" else []
W, CONFIRM, LAM = (float(x) for x in (_ARGS + ["20", "0.5", "1"][len(_ARGS):]))


def plausible_sets(items):
    out = {}
    for i, it in items.items():
        opts = [o.strip() for o in it["options"]]; idx = {o: k for k, o in enumerate(opts)}
        lines = it["prompt"].split("\n"); q = [l for l in lines if l.startswith("Transaction: ")][-1][len("Transaction: "):]
        key = oneslot.payee_key(q); ps = set()
        for a, b in zip(lines, lines[1:]):
            if a.startswith("Transaction: ") and b.startswith("Category: ") and a != "Transaction: " + q and key and oneslot.payee_key(a[len("Transaction: "):]) == key:
                c = b[len("Category: "):].strip()
                if c in idx:
                    ps.add(idx[c])
        for l in lines:
            if l.startswith("Other users file this payee as: "):
                for part in l[len("Other users file this payee as: "):].split("), "):
                    name = part.rsplit(" (", 1)[0].strip()
                    if name in idx:
                        ps.add(idx[name])
        out[i] = ps
    return out


def card(P, items, plaus, use_plaus=True):
    rows = []
    for i, p in P.items():
        g = items[i]["answer"]; pl = plaus[i] if use_plaus else None
        cost, auto, shown = S.rank_effort(p, g, W, lam=LAM, confirm=CONFIRM, plausible=pl)
        wrong_shown = [c for c in shown if c != g]
        rows.append(dict(cost=cost, auto_ok=auto and shown[0] == g, auto_bad=auto and shown[0] != g, first=not auto and shown[0] == g,
                         later=not auto and g in shown[1:], search=not auto and g not in shown, k=len(shown) if not auto else np.nan,
                         implaus=sum(c not in plaus[i] for c in wrong_shown) if not auto else 0))
    m = {k: np.nanmean([r[k] for r in rows]) for k in rows[0]}
    return m


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in json.loads((PROCESSED / "blind_v1_others.json").read_text())["items"]}
    plaus = plausible_sets(items)
    encs = [load(f) for f in sorted(glob.glob(ENC))]; decs = [load(f) for f in sorted(glob.glob(DEC))]; big = load(BIG)
    systems = {"decider-4B": [folds_calibrate(d, items) for d in decs],
               "encoder + decider": [product([e, d], items) for e in encs for d in decs],
               "encoder + decider + 35B": [product([e, d, big], items) for e in encs for d in decs]}
    n_pl = np.mean([len(v) for v in plaus.values()])
    print(f"**Report card, blind_v1 (1,500 transactions; W = {W:g}, confirm {CONFIRM:g}, clutter {LAM:g} per implausible wrong suggestion; "
          f"plausible categories per transaction: {n_pl:.1f} on average)**\n")
    cols = [("auto_ok", "filed automatically, right"), ("auto_bad", "filed automatically, WRONG"), ("first", "suggested, right one 1st"),
            ("later", "suggested, right one 2nd-5th"), ("search", "not suggested: user searches")]
    print("| system | plausible free | " + " | ".join(c for _, c in cols) + " | suggestions shown | implausible shown per 100 | work saved vs no model |")
    print("|---|---|" + "---|" * len(cols) + "---|---|---|")
    for n, Ps in systems.items():
        for use in (True, False):
            ms = [card(P, items, plaus, use) for P in Ps]
            g = lambda k: np.mean([m[k] for m in ms])  # noqa: E731
            print(f"| {n} | {'yes' if use else 'no'} | " + " | ".join(f"{100 * g(k):.1f}%" for k, _ in cols)
                  + f" | {g('k'):.1f} | {100 * g('implaus'):.1f} | {100 * (1 - g('cost') / 10):.0f}% |")
