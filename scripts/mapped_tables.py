"""Tables for PLAN step 150: on blind_v2, decider-4B (other-users line, three seeds, no new training) reading the raw other-users line (other
users' own category names, REPORT 123) against the profile mapped onto the user's categories (scripts/payee_profiles.py); the no-model
suggestion from the mapped profile; and each with the untrained 35B for first-time payees (REPORT 139). Top-1, first-time payees
(right first / top 3), payees seen before, new users, person-to-person, the user's own patterns, effort at W = 10.
usage: uv run python scripts/mapped_tables.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
from blind2_tables import readers  # noqa: E402
from decision_tables import evaluate, folds_calibrate, load, product  # noqa: E402

R = "results/per_item/real6_dm_decider_decider_decider-4b_none_h100bf16st800{s}_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_oth50_aux100_labrand255_laylabelled_shots_ev10soft_lora_blind_v2_{st}_labrand255_laylabelled_shots.noctx.jsonl"
P2P = {"p2p", "ambiguous_p2p", "p2p_anon"}
HABIT = {"changed_mind", "weekday_split", "idiosyncratic", "named_category", "for_whom", "couple_split"}

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in json.loads((PROCESSED / "blind_v2_mapped.json").read_text())["items"]}
    ph = json.loads((PROCESSED / "blind_v2_payeehist.json").read_text())
    big = readers(2)["decider-35B untrained"][0]
    first = [i for i in items if not ph[i]]; seen = [i for i in items if ph[i]]
    newu = [i for i in items if items[i]["why"] == "new_user"]; p2p = [i for i in items if items[i]["why"] in P2P]
    habit = [i for i in items if items[i]["why"] in HABIT]
    res = {}
    for s in ("", "s1", "s2"):
        raw, mapped = load(R.format(s=s, st="others")), load(R.format(s=s, st="mapped"))
        for n, rec in (("raw other-users line (REPORT 123)", raw), ("mapped profile line", mapped)):
            Pd = folds_calibrate(rec, items); res.setdefault(n, []).append(Pd)
            Pb = product([rec, big], items)
            res.setdefault(n + " + 35B for first-time payees", []).append({i: Pb[i] if not ph[i] else Pd[i] for i in Pd if i in Pb})
    print("| system | top-1 | first-time payees: 1st / top 3 | payees seen before | new users | person-to-person | own patterns | effort W=10 |")
    print("|---|---|---|---|---|---|---|---|")
    for n, Ps in res.items():
        rows = []
        for P in Ps:
            rank = {i: int(np.where(np.argsort(-p) == items[i]["answer"])[0][0]) + 1 for i, p in P.items()}
            a = lambda sel: 100 * np.mean([rank[i] == 1 for i in sel if i in rank])  # noqa: E731
            rows.append([a(list(rank)), a(first), 100 * np.mean([rank[i] <= 3 for i in first if i in rank]), a(seen), a(newu), a(p2p), a(habit),
                         evaluate(P, items, 10, "per-item")[2]])
        m = np.mean(rows, 0)
        print(f"| {n} | {m[0]:.1f} | {m[1]:.1f} / {m[2]:.1f} | {m[3]:.1f} | {m[4]:.1f} | {m[5]:.1f} | {m[6]:.1f} | {m[7]:.3f} |")
    opts = {i: [o.strip() for o in items[i]["options"]] for i in items}
    nm = lambda sel, k: 100 * np.mean([opts[i][items[i]["answer"]] in items[i]["mapped_rank"][:k] for i in sel])  # noqa: E731
    print(f"| no-model suggestion from the mapped profile | {nm(list(items), 1):.1f} | {nm(first, 1):.1f} / {nm(first, 3):.1f} | {nm(seen, 1):.1f} | {nm(newu, 1):.1f} | "
          f"{nm(p2p, 1):.1f} | {nm(habit, 1):.1f} | – |")
