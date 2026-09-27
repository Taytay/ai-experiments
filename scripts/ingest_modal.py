"""Bring Modal jobs' outputs into the checkout (CLAUDE.md, "GPU work: Modal"): for each tag already downloaded to modal_out/<tag>/
(`modal volume get ai-exp-results <tag> modal_out/`), copy results/ into results/, union its evals/runs.jsonl rows into ours by run_id
(sorted), and copy its adapters into models/adapters/. Then: `just push-models`, commit the .dvc files, `just drop-all`.
usage: uv run python scripts/ingest_modal.py <tag> [<tag> ...]
"""
import json
import shutil
import sys

from ai_experiments.paths import ROOT

if __name__ == "__main__":
    runs_path = ROOT / "evals" / "runs.jsonl"
    rows = {json.loads(ln)["run_id"]: ln.rstrip("\n") for ln in runs_path.read_text().splitlines() if ln.strip()}
    n0 = len(rows)
    for tag in sys.argv[1:]:
        src = ROOT / "modal_out" / tag
        assert src.is_dir(), f"{src} missing: modal volume get ai-exp-results {tag} modal_out/"
        for f in (src / "results").rglob("*") if (src / "results").exists() else []:
            if f.is_file():
                dst = ROOT / "results" / f.relative_to(src / "results"); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dst)
        if (src / "evals" / "runs.jsonl").exists():
            for ln in (src / "evals" / "runs.jsonl").read_text().splitlines():
                if ln.strip():
                    rows.setdefault(json.loads(ln)["run_id"], ln)
        for a in (src / "models" / "adapters").iterdir() if (src / "models" / "adapters").exists() else []:
            if a.is_dir():
                shutil.copytree(a, ROOT / "models" / "adapters" / a.name, dirs_exist_ok=True); print("adapter", a.name)
    runs_path.write_text("".join(rows[k] + "\n" for k in sorted(rows)))
    print(f"runs.jsonl: {n0} -> {len(rows)} rows")
