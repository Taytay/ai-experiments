"""Table for PLAN steps 105 to 108: user effort on blind_v1 for each decider-4B recipe of recipe_tables.RECIPES, alone and behind the
encoder (split_tables.ENC auto-files at its own calibrated 98% threshold, decider reads the rest); `scorecard.effort` (wrong = 5).
usage: uv run python scripts/recipe_effort_tables.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments import real6_cells as RC
from ai_experiments import scorecard as S
from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
from effort_tables import calibrated  # noqa: E402
from recipe_tables import A, RECIPES  # noqa: E402
from split_tables import ENC  # noqa: E402

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in json.loads((PROCESSED / "blind_v1.json").read_text())["items"]}
    enc = calibrated(RC.load_recs(ENC), items)
    print("| recipe | seeds | effort alone | auto-filed alone | effort, encoder first | auto-filed, encoder first |"); print("|---|---|---|---|---|---|")
    for lab, kw, seeds in RECIPES:
        v = []
        for sd in seeds:
            r = RC.load_recs(f"real6_dm_decider_{A.format(s=sd, **kw)}_blind_v1_lab{kw['lab']}.noctx.jsonl")
            if not r:
                continue
            dec = calibrated(r, items); ids = sorted(set(dec) & set(enc)); row = []
            for route in ("d", "ed"):
                a, ok, t3 = [], [], []
                for i in ids:
                    m = enc[i] if (route == "ed" and enc[i]["a98"]) else dec[i]; y = items[i]["answer"]
                    a.append(bool(m["a98"])); ok.append(int(np.argmax(m["p"])) == y); t3.append(y in np.argsort(-m["p"])[:3])
                row += [S.effort(a, ok, t3), 100 * np.mean(a)]
            v.append(row)
        if v:
            v = np.array(v); f = lambda k, d: f"{v[:, k].mean():.{d}f}" + (f" [{v[:, k].min():.{d}f}, {v[:, k].max():.{d}f}]" if len(v) > 1 else "")  # noqa: E731
            print(f"| {lab} | {len(v)} | {f(0, 3)} | {f(1, 1)} | {f(2, 3)} | {f(3, 1)} |")
