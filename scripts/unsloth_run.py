# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "ai-experiments",
#   "unsloth @ git+https://github.com/unslothai/unsloth",
#   "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo",
#   "bitsandbytes",
#   "sentence-transformers>=6.1.0",
# ]
# [tool.uv]
# override-dependencies = ["transformers==5.19.0", "torch==2.13.0", "torchvision==0.28.0", "sentence-transformers>=6.1.0"]
# [tool.uv.sources]
# ai-experiments = { path = "..", editable = true }
# ///
"""Row 238 (owner, 2026-10-07: "That article I sent recommended unsloth to fine tune it"; "Are you sure that uv can't be forced to install a
given version?"): run a repo script under unsloth with EmbeddingGemma 2. unsloth (release and main, 2026-10-07) caps transformers at
5.17 and EmbeddingGemma 2 needs 5.18+; `uv run --with` and UV_OVERRIDE cannot lift the cap, but a script's own [tool.uv]
override-dependencies can. This project is a path dependency, so its code and data are the checkout's.
usage: uv run --script scripts/unsloth_run.py scripts/li_decider.py train    (env as the target script reads it; LOADER=unsloth)
"""
import unsloth  # noqa: F401  first, before anything imports transformers, so its patches all apply
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
target = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(target, run_name="__main__")
