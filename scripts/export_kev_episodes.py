"""The categoriser's training episodes as kev labelled requests (PLAN step 80, MODEL-16), for `python -m kev.train --data`: each
episode's prompt without its "Category:" cue is the state, one choice question (exp_decision_models.QUESTION) with the category names as
criteria, the gold name as label. Built by exp_decider_finetune.episodes() under the same env as the categoriser arm it is compared
with (DBEP, FOLD, RENAME, POI_*, ...), so users, layouts and augmentations match.
usage: DBEP=0.5 FOLD=0 uv run python scripts/export_kev_episodes.py OUT.jsonl
"""
import json
import os
import sys

from ai_experiments.paths import ROOT

out = sys.argv[1]
os.environ.setdefault("MODEL", "Mapika/decider-2b")  # only names the (unused) decider run; the episodes do not depend on it
sys.argv = [sys.argv[0]]
_src = (ROOT / "scripts" / "exp_decider_finetune.py").read_text().split("\ndef main():")[0]
G = {"__name__": "decider_episodes", "__file__": str(ROOT / "scripts" / "exp_decider_finetune.py")}
exec(compile(_src, "exp_decider_finetune.py", "exec"), G)
n = 0
with open(out, "w") as f:
    for state, opts, gold in G["episodes"]():
        q = {"type": "choice", "instructions": G["QUESTION"], "criteria": {o: None for o in opts}, "label": opts[gold]}
        f.write(json.dumps({"state": state, "questions": {"category": q}}, ensure_ascii=False) + "\n"); n += 1
print(f"{n} kev records -> {out} (episodes {G['SFX']})")
