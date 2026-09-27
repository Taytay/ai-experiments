"""Download kev (github.com/jaredpalmer/kev, Apache-2.0) at a pinned commit as a source archive and print its directory, for
`PYTHONPATH=$(python scripts/fetch_kev.py) python -m kev.train ...` (row 80): its package pins torch < 2.9, whose Triton gives wrong
Gated DeltaNet gradients on Hopper (fla #640), so it runs from source inside this repo's Qwen3.5 overlay instead of being installed.
usage: uv run python scripts/fetch_kev.py [SHA]
"""
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

sha = sys.argv[1] if len(sys.argv) > 1 else "5920c5f"
dst = Path("/tmp") / f"kev-{sha}"
if not dst.exists():
    data = urllib.request.urlopen(f"https://github.com/jaredpalmer/kev/archive/{sha}.tar.gz").read()
    tarfile.open(fileobj=io.BytesIO(data)).extractall("/tmp")
    next(p for p in Path("/tmp").glob(f"kev-{sha}*") if p != dst).rename(dst)
print(dst)
