"""Tables for PLAN step 149: "Kind:" lines under every history row and the query (canonical payee kinds). decider-4B with the other-users
line: without kinds (REPORT 123, three seeds); that adapter reading kind lines zero-shot; trained with kind lines (KIND_LINES=1, 15% unknown,
two seeds) reading prompts with the generator's kinds, with the query's kind inferred (step 148: the 35B from name and other users'
filings), and without kind lines. blind_v1 and blind_v2: top-1; first-time payees (right first / top 3); new users; person-to-person
payees; the user's own patterns; payees seen before; effort under the per-item rule at W = 10.
usage: uv run python scripts/kindline_tables.py
"""
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
from decision_tables import evaluate, folds_calibrate, load  # noqa: E402

R = "results/per_item/real6_dm_decider_decider_decider-4b_none_h100bf16st800*_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_oth50"
ARMS = [("no kind lines (REPORT 123)", "", "others"), ("kind lines read zero-shot", "", "kinds"),
        ("kind lines read zero-shot, query kind inferred", "", "kindsinf"),
        ("trained with kind lines, generator kinds", "_kind", "kinds"), ("trained with kind lines, query kind inferred", "_kind", "kindsinf"),
        ("trained with kind lines, prompts without them", "_kind", "others")]
P2P = {"ambiguous_p2p", "p2p", "p2p_anon"}
HABIT = {"idiosyncratic", "changed_mind", "named_category", "weekday_split", "for_whom", "couple_split"}

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("| set | arm | seeds | top-1 | first-time payees: 1st / top 3 | new users | person-to-person | own patterns | payees seen before | effort W=10 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for v in (1, 2):
        items = {i["id"]: i for i in json.loads((PROCESSED / f"blind_v{v}_others.json").read_text())["items"]}
        ph = json.loads((PROCESSED / f"blind_v{v}_payeehist.json").read_text())
        first = [i for i in items if not ph[i]]; seen = [i for i in items if ph[i]]
        newu = [i for i in items if items[i]["why"] in ("short_history", "new_user")]
        p2p = [i for i in items if items[i]["why"] in P2P]; habit = [i for i in items if items[i]["why"] in HABIT]
        for name, arm, st in ARMS:
            fs = sorted(glob.glob(f"{R}{arm}_aux100_labrand255_laylabelled_shots_ev10soft_lora_blind_v{v}_{st}_labrand255_laylabelled_shots.noctx.jsonl"))
            if not fs:
                continue
            rows = []
            for f in fs:
                rec = load(f)
                rank = {i: int(np.where(np.argsort(-x) == items[i]["answer"])[0][0]) + 1 for i, x in rec.items() if i in items}
                a = lambda sel: 100 * np.mean([rank[i] == 1 for i in sel if i in rank])  # noqa: E731
                rows.append([a(list(rank)), a(first), 100 * np.mean([rank[i] <= 3 for i in first if i in rank]), a(newu), a(p2p), a(habit), a(seen),
                             evaluate(folds_calibrate(rec, items), items, 10, "per-item")[2]])
            m = np.mean(rows, 0)
            print(f"| blind_v{v} | {name} | {len(fs)} | {m[0]:.1f} | {m[1]:.1f} / {m[2]:.1f} | {m[3]:.1f} | {m[4]:.1f} | {m[5]:.1f} | {m[6]:.1f} | {m[7]:.3f} |")
