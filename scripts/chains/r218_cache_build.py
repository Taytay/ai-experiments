"""Row 218 (2026-10-06): the caches the late-interaction jobs read, built once on Modal (`SAVE_DATA=1`; later jobs copy them with
DATA_FROM=<this job's tag>): v4 training households 0-999 and held-out 100000-100049 (two_tower.households, data/interim/hh_cache) with
their neighbour lists (li_decider._nbrs, data/interim/li_nb), and blind_v2's 250 budgets with theirs. Keys are computed inside the
image, so every job of the same image finds them."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import blind_budgets  # noqa: E402
import li_decider as L  # noqa: E402
from two_tower import households  # noqa: E402

t, cache = time.time(), {}
for name, bs in (("train", households("train", range(1000))), ("test", households("test", range(100000, 100050))), ("blind2", blind_budgets.budgets())):
    for n, b in enumerate(bs):
        L.prepared(b, None, cache, full=False)
        if n % 100 == 0:
            cache.clear(); print(f"{name} {n} ({time.time() - t:.0f}s)", flush=True)
print(f"done ({time.time() - t:.0f}s)")
