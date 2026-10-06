"""Row 226 (2026-10-06): the caches the v5 late-interaction jobs read, built once on Modal (`SAVE_DATA=1`; later jobs copy them with
DATA_FROM=<this job's tag>), as scripts/chains/r218_cache_build.py did for v4: v5 training households 0-(TRAIN_N-1) and held-out
100000-100049 (RS_V5=1 in the job's env; two_tower.households, data/interim/hh_cache) with their neighbour lists (li_decider._nbrs,
data/interim/li_nb). v4's held-out households are rebuilt too (RS_V5=0 per call is not possible inside one process, so only v5 here;
their neighbour lists are content-keyed and come from r218-cache)."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import li_decider as L  # noqa: E402
from two_tower import households  # noqa: E402

assert os.environ.get("RS_V5") == "1"
t, cache = time.time(), {}
for name, bs in (("train", households("train", range(int(os.environ.get("TRAIN_N", "200"))))), ("test", households("test", range(100000, 100050)))):
    for n, b in enumerate(bs):
        L.prepared(b, None, cache, full=False)
        if n % 50 == 0:
            cache.clear(); print(f"{name} {n} ({time.time() - t:.0f}s)", flush=True)
print(f"done ({time.time() - t:.0f}s)")
