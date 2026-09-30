"""PLAN step 149 (routing): kind lines only where the payee is new to the user. Per item, decider-4B (other-users line, no kind training;
three seeds) reads the prompt with a kind line (generator's kinds, or the query kind inferred from name and filings) for first-time payees
and the plain prompt for payees seen before; against no kind lines and against REPORT 139's 35B routing. Saved readings only.
usage: uv run python scripts/kindroute_tables.py
"""
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
from blind2_tables import readers  # noqa: E402
from decision_tables import evaluate, folds_calibrate, load, product  # noqa: E402

R = "results/per_item/real6_dm_decider_decider_decider-4b_none_h100bf16st800{s}_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_oth50_aux100_labrand255_laylabelled_shots_ev10soft_lora_blind_v{v}_{st}_labrand255_laylabelled_shots.noctx.jsonl"

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("| set | system | top-1 | first-time payees: 1st / top 3 | payees seen before | effort W=10 |"); print("|---|---|---|---|---|---|")
    for v in (1, 2):
        items = {i["id"]: i for i in json.loads((PROCESSED / f"blind_v{v}_others.json").read_text())["items"]}
        ph = json.loads((PROCESSED / f"blind_v{v}_payeehist.json").read_text())
        big = readers(v)["decider-35B untrained"][0]
        first = [i for i in items if not ph[i]]; seen = [i for i in items if ph[i]]
        res = {}
        for s in ("", "s1", "s2"):
            plain = load(R.format(s=s, v=v, st="others")); kinds = load(R.format(s=s, v=v, st="kinds")); inf = load(R.format(s=s, v=v, st="kindsinf"))
            systems = {"no kind lines": plain,
                       "kind lines for first-time payees (database kinds)": {i: kinds[i] if not ph[i] else plain[i] for i in plain},
                       "kind lines for first-time payees (inferred kinds)": {i: inf[i] if not ph[i] else plain[i] for i in plain}}
            for n, rec in systems.items():
                P = folds_calibrate(rec, items)
                res.setdefault(n, []).append(P)
            Pd = folds_calibrate(plain, items); Pb = product([plain, big], items)
            res.setdefault("35B for first-time payees (REPORT 139)", []).append({i: Pb[i] if not ph[i] else Pd[i] for i in Pd if i in Pb})
            Pk = folds_calibrate({i: inf[i] if not ph[i] else plain[i] for i in plain}, items); Pkb = product([{i: inf[i] if not ph[i] else plain[i] for i in plain}, big], items)
            res.setdefault("inferred kind lines + 35B for first-time payees", []).append({i: Pkb[i] if not ph[i] else Pk[i] for i in Pk if i in Pkb})
        for n, Ps in res.items():
            rows = []
            for P in Ps:
                rank = {i: int(np.where(np.argsort(-p) == items[i]["answer"])[0][0]) + 1 for i, p in P.items()}
                a = lambda sel: 100 * np.mean([rank[i] == 1 for i in sel if i in rank])  # noqa: E731
                rows.append([a(list(rank)), a(first), 100 * np.mean([rank[i] <= 3 for i in first if i in rank]), a(seen), evaluate(P, items, 10, "per-item")[2]])
            m = np.mean(rows, 0)
            print(f"| blind_v{v} | {n} | {m[0]:.1f} | {m[1]:.1f} / {m[2]:.1f} | {m[3]:.1f} | {m[4]:.3f} |")
