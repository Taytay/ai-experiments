"""Repo-relative locations. The package is installed editable, so this file sits inside the
checkout and ROOT is the checkout, whatever the working directory is."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # src/ai_experiments/paths.py -> repo root
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
ADAPTERS = MODELS / "adapters"
EVALS = ROOT / "evals"
