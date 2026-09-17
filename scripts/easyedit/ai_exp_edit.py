"""Knowledge-editing arm for the ai-experiments universe (PLAN step 19, BASE-2): MEMIT or AlphaEdit on Qwen2.5-3B, one batch of
edits (`batch_edit`, a single chunk; `edit` would apply and undo them one at a time), the edited model saved as a full checkpoint so `scripts/exp_curriculum.py base <dir>` scores it on the frozen items.

Runs from the EasyEdit checkout with its own environment (the ai_experiments package is installed there without its deps):
  cd /home/taytay/projects/Taytay/EasyEdit && .venv/bin/python ai_exp_edit.py ALG FACTS [--n N]
    ALG    MEMIT | AlphaEdit        (hparams/<ALG>/qwen2.5-3b.yaml, derived from the 7B settings: layers 4-8, v_loss_layer 35, WikiText covariance)
    FACTS  type | all               one edit per trained species (its type) or five (type, weakness, habitat, diet, region)
    --n    edit only the first N species (smoke)
Edit requests are (prompt with the subject, subject, target): "{N} is a" -> "{T}-type creature", "{N} is weak to" -> "{W}-type attacks",
"{N} lives in" -> "{H} habitats", "{N} eats as an" -> "{D}", "{N} is found in" -> "{R}". Output: the edited model under
/home/taytay/projects/Taytay/ai-experiments/models/adapters/edit_<alg>_<facts>_qwen2.5-3b (bf16), and results/edit_<alg>_<facts>.json with
EasyEdit's own rewrite / rephrase metrics and the timing.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

from easyeditor import BaseEditor, MEMITHyperParams, AlphaEditHyperParams
from ai_experiments import universe as U

ALG = sys.argv[1] if len(sys.argv) > 1 else "MEMIT"
FACTS = sys.argv[2] if len(sys.argv) > 2 else "type"
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else None
ROOT = Path("/home/taytay/projects/Taytay/ai-experiments")
TAG = f"_ridge{os.environ['COV_RIDGE']}" if os.environ.get("COV_RIDGE") else ""
OUT_DIR = (Path(os.environ["SMOKE_DIR"]) if N and os.environ.get("SMOKE_DIR") else ROOT / "models" / "adapters") / f"edit_{ALG.lower()}_{FACTS}{TAG}_qwen2.5-3b{'_smoke' if N else ''}"
OUT_JSON = ROOT / "results" / f"edit_{ALG.lower()}_{FACTS}{TAG}{'_smoke' if N else ''}.json"

species = [s for s in U.build() if not s["heldout"]]
if N:
    species = species[:N]
TEMPLATES = {"type": ("{N} is a", "{T}-type creature"), "weakness": ("{N} is weak to", "{W}-type attacks"), "habitat": ("{N} lives in", "{H} habitats"),
             "diet": ("{N} eats as an", "{D}"), "region": ("{N} is found in", "{R}")}
attrs = ["type"] if FACTS == "type" else list(TEMPLATES)
prompts, targets, subjects = [], [], []
for s in species:
    f = dict(N=s["name"], T=s["type"], W=s["weakness"], H=s["habitat"], D=s["diet"], R=s["region"])
    for a in attrs:
        p, t = TEMPLATES[a]
        prompts.append(p.format(**f)); targets.append(t.format(**f)); subjects.append(s["name"])
print(f"{ALG} on Qwen2.5-3B: {len(prompts)} edits ({len(species)} species x {len(attrs)} facts), e.g. {prompts[0]!r} -> {targets[0]!r}", flush=True)

HP = {"MEMIT": MEMITHyperParams, "AlphaEdit": AlphaEditHyperParams}[ALG]
hparams = HP.from_hparams(f"./hparams/{ALG}/qwen2.5-3b")
if os.environ.get("COV_RIDGE"):  # MEMIT: ridge on the closed-form solve (see memit_main.py); the run's names carry it
    hparams.cov_ridge = float(os.environ["COV_RIDGE"])
hparams.batch_size = len(prompts)  # one chunk: every edit in a single MEMIT / AlphaEdit batch (the plan's protocol); sequential_edit=True keeps the edited weights
editor = BaseEditor.from_hparams(hparams)
t0 = time.time()
metrics, edited_model, _ = editor.batch_edit(prompts=prompts, target_new=targets, subject=subjects, ground_truth=None, sequential_edit=True, verbose=False)
el = time.time() - t0
summary = {}
for m in metrics:
    post = m.get("post", {})
    for k, v in post.items():
        if isinstance(v, (int, float)):
            summary.setdefault(k, []).append(float(v))
        elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
            summary.setdefault(k, []).append(float(sum(v) / len(v)))
summary = {k: round(100 * sum(v) / len(v), 1) for k, v in summary.items()}
print(f"edited in {el / 60:.1f} min; EasyEdit post-edit means: {summary}", flush=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
edited_model.to(torch.bfloat16).save_pretrained(OUT_DIR); editor.tok.save_pretrained(OUT_DIR)
OUT_JSON.parent.mkdir(exist_ok=True)
OUT_JSON.write_text(json.dumps(dict(alg=ALG, facts=FACTS, n_edits=len(prompts), n_species=len(species), edit_minutes=round(el / 60, 1),
                                    hparams={k: v for k, v in vars(hparams).items() if isinstance(v, (int, float, str, bool, list))},
                                    easyedit_post=summary, model_dir=str(OUT_DIR.relative_to(ROOT) if OUT_DIR.is_relative_to(ROOT) else OUT_DIR)), indent=2))
print(f"saved {OUT_DIR} and {OUT_JSON}")
