"""Tables for PLAN steps 79 and 80 (MODEL-16): one answer slot over all options (decider's layout and letter-slot loss) against this
repo's per-option loss and scorer, on the same episodes, and the published decision models trained on REAL-6. Fold 0's held-out users.

  S.1  REAL-6 (database episodes): Qwen3.5-2B per-option (row 78), Qwen3.5-2B one slot (200 / 800 steps; with the shot labels' token
       loss, AUX_LM), decider-2B one slot, kev-4B fine-tuned (its pointer head), with the zero-shot decision models for reference
  S.2  POI-1 (row 73's episodes, descriptions + kind lines + kind-retrieved examples): the same comparison
  S.3  scoring time per item, from the jobs' logs
Scorecard columns as open_models_tables.py.
usage: uv run python scripts/one_slot_tables.py
"""
import json
import sys
import warnings
from pathlib import Path

from ai_experiments import real6 as R6
from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
import open_models_tables as O  # noqa: E402

Q35 = "real6_categoriser_Qwen3.5-2B"
HEAD = "| reader | n | top-1 [interval] | top-3 | bits left | auto-file at 98%: coverage (precision) |"

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    r6 = {i["id"]: i for i in R6.load("v1")["items"]}
    print("**Table S.1: REAL-6, fold 0's held-out users, database episodes (DBEP 0.5): one answer slot against the per-option loss**\n")
    print(HEAD); print("|---|---|---|---|---|---|")
    for lab, f in [("Qwen3.5-2B, per-option loss + all-label (row 78), 200 steps", f"{Q35}_none_h100bf16_hf_f0_alllab_dbep50_lora_hfs.noctx.jsonl"),
                   ("Qwen3.5-2B, one slot, 200 steps", "real6_dm_decider_slot_Qwen3.5-2B_none_h100bf16_f0_dbep50_lora_real6.noctx.jsonl"),
                   ("Qwen3.5-2B, one slot, 800 steps", "real6_dm_decider_slot_Qwen3.5-2B_none_h100bf16st800_f0_dbep50_lora_real6.noctx.jsonl"),
                   ("Qwen3.5-2B, one slot + shot-label loss, 200 steps", "real6_dm_decider_slot_Qwen3.5-2B_none_h100bf16_f0_dbep50_aux100_lora_real6.noctx.jsonl"),
                   ("Qwen3.5-2B, one slot + shot-label loss, 800 steps", "real6_dm_decider_slot_Qwen3.5-2B_none_h100bf16st800_f0_dbep50_aux100_lora_real6.noctx.jsonl"),
                   ("decider-2B, one slot, 200 steps", "real6_dm_decider_decider_decider-2b_none_h100bf16_f0_dbep50_lora_real6.noctx.jsonl"),
                   ("decider-2B, one slot, 800 steps", "real6_dm_decider_decider_decider-2b_none_h100bf16st800_f0_dbep50_lora_real6.noctx.jsonl"),
                   ("kev-4B fine-tuned (its trainer, from the released checkpoint)", "real6_dm_kev_kev_kev-4b_none_h100bf16_f0_dbep50_real6.noctx.jsonl"),
                   ("decider-2B zero-shot", "real6_dm_decider_decider-2b_real6.noctx.jsonl"),
                   ("kev-4B zero-shot", "real6_dm_kev_kev-4b_real6.noctx.jsonl")]:
        r = O.card_row(lab, O.load(f), r6)
        if r:
            print(r)
    doc = json.loads((PROCESSED / "poi1_v1.json").read_text())
    poi = {i["id"]: i for i in doc["items"]}; users = {u["user"]: u for u in doc["users"]}
    print("\n**Table S.2: POI-1, fold 0's held-out users, row 73's episodes, layout: descriptions + kind lines + kind-retrieved examples**\n")
    print(HEAD + " seen-kind top-1 | unseen-kind top-1 |"); print("|---|---|---|---|---|---|---|---|")
    for lab, f in [("Qwen3.5-2B, per-option loss + all-label (row 78)", f"{Q35}_poi1_v1_none_h100bf16_hf_pkind_pdesc_uns50_f0_ren50_alllab_lora_poi1_v1_desc_kshots_kinds_hfs.noctx.jsonl"),
                   ("Qwen3.5-2B, one slot", "real6_dm_decider_slot_Qwen3.5-2B_poi1_v1_none_h100bf16_pkind_pdesc_uns50_f0_ren50_lora_poi1_v1_desc_kshots_kinds.noctx.jsonl"),
                   ("decider-2B, one slot (row 77)", "real6_dm_decider_decider_decider-2b_poi1_v1_none_h100bf16_pkind_pdesc_uns50_f0_ren50_lora_poi1_v1_desc_kshots_kinds.noctx.jsonl")]:
        recs = O.load(f)
        ex = f" {O.top1(recs, poi, lambda i: i['seen']):.1f} | {O.top1(recs, poi, lambda i: not i['seen']):.1f} |"
        r = O.card_row(lab, recs, poi, users=users, extra=ex)
        if r:
            print(r)
