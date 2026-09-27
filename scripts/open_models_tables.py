"""Tables for PLAN steps 77 (MODEL-14: the decision models Ollaya serves) and 78 (MODEL-15: Apache-2.0 Qwen3.5 in place of the
research-licensed Qwen2.5-3B), fold 0's held-out users throughout.

  O.1  REAL-6: untrained readers (Qwen2.5-3B, Qwen3.5-2B / 4B, decider, Decision-1.0, kev, Von) without and with the merchant record,
       and trained ones (database episodes on each Qwen; the [MASK] encoder from ModernBERT, Laya and Von at 200 / 1,500 steps)
  O.2  POI-1: the same readers on the plain layout, kind lines + kind-retrieved examples, and descriptions on top; trained: the row 73
       recipe on each Qwen and on decider-2B (its own letter readout)
  O.3  label induction v2: untrained readers
  O.4  scoring time per item on one H100, from the jobs' logs
Scorecard columns: top-1 [user-bootstrap interval], top-3, bits left, auto-file coverage at 98% (realised precision), temperature and
thresholds fitted leave-users-out within fold 0 ((id // 4) mod 4).
usage: uv run python scripts/open_models_tables.py
"""
import json
import os
import warnings

import numpy as np

from ai_experiments import real6 as R6
from ai_experiments import real6_cells as RC
from ai_experiments import scorecard as S
from ai_experiments.paths import PROCESSED, ROOT

Q25 = "real6_categoriser_Qwen2.5-3B-Instruct"
DM = [("decider-2b", "decider_decider-2b"), ("decider-4b", "decider_decider-4b"), ("Decision-1.0 Sol-2B", "decision_Decision-1.0-Sol-2B"),
      ("Decision-1.0 Nox-4B", "decision_Decision-1.0-Nox-4B"), ("kev-4b", "kev_kev-4b"), ("Von 1.2", "von_von")]
F0 = lambda u: (u // 4) % 4  # noqa: E731


def load(name):
    return RC.load_recs(name) if (ROOT / "results" / "per_item" / name).exists() else {}


def card_row(label, recs, items, users=None, extra=""):
    recs = {i: r for i, r in recs.items() if i in items and items[i]["user"] % 4 == 0}
    if not recs:
        return None
    sc = S.scorecard(recs, items, users=users, fold_of=F0)
    return (f"| {label} | {sc['n']} | {sc['top1']:.1f} [{sc['top1_ci'][0]}, {sc['top1_ci'][1]}] | {sc['top3']:.1f} | {sc['bits']:.2f} | "
            f"{sc['cov98']:.1f} ({sc['prec98']:.1f}) |{extra}")


def top1(recs, items, sel=lambda i: True):
    v = [int(np.argmax(r["sum_lp"])) == r["answer"] for i, r in recs.items() if i in items and items[i]["user"] % 4 == 0 and sel(items[i])]
    return 100 * np.mean(v) if v else float("nan")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    r6 = {i["id"]: i for i in R6.load("v1")["items"]}
    head = "| reader | n | top-1 [interval] | top-3 | bits left | auto-file at 98%: coverage (precision) |"
    print("**Table O.1: REAL-6, fold 0's held-out users: untrained readers without / with the merchant record, and trained readers**\n")
    print(head); print("|---|---|---|---|---|---|")
    rows = [("Qwen2.5-3B-Instruct, untrained", "real6_Qwen2.5-3B-Instruct.noctx.jsonl", "real6_Qwen2.5-3B-Instruct.ctx.jsonl"),
            ("Qwen3.5-2B, untrained", "real6_Qwen3.5-2B_hfs.noctx.jsonl", "real6_Qwen3.5-2B_hfs.ctx.jsonl"),
            ("Qwen3.5-4B, untrained", "real6_Qwen3.5-4B_hfs.noctx.jsonl", "real6_Qwen3.5-4B_hfs.ctx.jsonl")]
    rows += [(f"{lab}, zero-shot", f"real6_dm_{t}_real6.noctx.jsonl", f"real6_dm_{t}_real6.ctx.jsonl") for lab, t in DM]
    for lab, a, b in rows:
        for cond, f in (("no record", a), ("record", b)):
            r = card_row(f"{lab}, {cond}", load(f), r6)
            if r:
                print(r)
    for lab, f in [("Qwen2.5-3B, database episodes (row 70)", f"{Q25}_none_h100bf16_f0_alllab_dbep50_lora.noctx.jsonl"),
                   ("Qwen3.5-2B, database episodes", "real6_categoriser_Qwen3.5-2B_none_h100bf16_hf_f0_alllab_dbep50_lora_hfs.noctx.jsonl"),
                   ("Qwen3.5-4B, database episodes", "real6_categoriser_Qwen3.5-4B_none_h100bf16_hf_f0_alllab_dbep50_lora_hfs.noctx.jsonl")] + \
                  [(f"[MASK] encoder from {s}, {n} steps", f"real6_encmask_{k}_st{n}_f0.noctx.jsonl") for k, s in (("mbert", "ModernBERT"), ("laya", "Laya"), ("von", "Von"))
                   for n in (200, 1500)]:
        r = card_row(lab, load(f), r6)
        if r:
            print(r)

    doc = json.loads((PROCESSED / "poi1_v1.json").read_text())
    poi = {i["id"]: i for i in doc["items"]}; users = {u["user"]: u for u in doc["users"]}
    print("\n**Table O.2: POI-1, fold 0's held-out users, by prompt layout (plain; kind lines + kind-retrieved examples; the same with the "
          "category descriptions of row 73), with seen / unseen kind top-1**\n")
    print(head.replace(" |\n", " |") + " seen-kind top-1 | unseen-kind top-1 |"); print("|---|---|---|---|---|---|---|---|")
    L = {"plain": "poi1_v1", "kinds + kshots": "poi1_v1_kshots_kinds", "desc + kinds + kshots": "poi1_v1_desc_kshots_kinds"}
    readers = [("Qwen2.5-3B-Instruct, untrained", lambda s: f"real6_Qwen2.5-3B-Instruct_{s}.noctx.jsonl"),
               ("Qwen3.5-2B, untrained", lambda s: f"real6_Qwen3.5-2B_{s}_hfs.noctx.jsonl"),
               ("Qwen3.5-4B, untrained", lambda s: f"real6_Qwen3.5-4B_{s}_hfs.noctx.jsonl")] + \
              [(f"{lab}, zero-shot", (lambda t: lambda s: f"real6_dm_{t}_{s}.noctx.jsonl")(t)) for lab, t in DM] + \
              [("Qwen2.5-3B trained, row 73 recipe", lambda s: f"{Q25}_poi1_v1_none_h100bf16_pkind_pdesc_uns50_f0_ren50_alllab_lora_{s}.noctx.jsonl"),
               ("Qwen3.5-2B trained, row 73 recipe", lambda s: f"real6_categoriser_Qwen3.5-2B_poi1_v1_none_h100bf16_hf_pkind_pdesc_uns50_f0_ren50_alllab_lora_{s}_hfs.noctx.jsonl"),
               ("Qwen3.5-4B trained, row 73 recipe", lambda s: f"real6_categoriser_Qwen3.5-4B_poi1_v1_none_h100bf16_hf_pkind_pdesc_uns50_f0_ren50_alllab_lora_{s}_hfs.noctx.jsonl"),
               ("decider-2b trained, row 73 recipe (its letter readout)", lambda s: f"real6_dm_decider_decider_decider-2b_poi1_v1_none_h100bf16_pkind_pdesc_uns50_f0_ren50_lora_{s}.noctx.jsonl")]
    for lab, fn in readers:
        for lay, s in L.items():
            recs = load(fn(s))
            ex = f" {top1(recs, poi, lambda i: i['seen']):.1f} | {top1(recs, poi, lambda i: not i['seen']):.1f} |"
            r = card_row(f"{lab}, {lay}", recs, poi, users=users, extra=ex)
            if r:
                print(r)
    for k, s in (("mbert", "ModernBERT"), ("laya", "Laya"), ("von", "Von")):
        for n in (200, 1500):
            recs = load(f"real6_encmask_{k}_poi1_v1_st{n}_f0.noctx.jsonl")
            ex = f" {top1(recs, poi, lambda i: i['seen']):.1f} | {top1(recs, poi, lambda i: not i['seen']):.1f} |"
            r = card_row(f"[MASK] encoder from {s}, {n} steps (its own layout)", recs, poi, users=users, extra=ex)
            if r:
                print(r)

    li = {i["id"]: i for i in json.loads((PROCESSED / "label_induction_v2.json").read_text())["items"]}
    print("\n**Table O.3: label induction v2, fold 0's users (900 items), top-1 %, untrained readers**\n")
    print("| reader | top-1 |"); print("|---|---|")
    for lab, f in [("Qwen2.5-3B-Instruct", "real6_Qwen2.5-3B-Instruct_label_induction_v2.noctx.jsonl"), ("Qwen3.5-2B", "real6_Qwen3.5-2B_label_induction_v2_hfs.noctx.jsonl"),
                   ("Qwen3.5-4B", "real6_Qwen3.5-4B_label_induction_v2_hfs.noctx.jsonl")] + [(lab, f"real6_dm_{t}_label_induction_v2.noctx.jsonl") for lab, t in DM]:
        recs = load(f)
        if recs:
            print(f"| {lab} | {top1(recs, li):.1f} |")

    print("\n**Table O.4: scoring time per POI-1 item on one H100 (12 to 20 options after a ~900-token prompt), from the jobs' logs; "
          "wrappers as written for this study (no CUDA graphs or compilation; kev in float32, one item at a time; Von one item at a time)**\n")
    print("| reader | ms per item | how the options are read |"); print("|---|---|---|")
    for lab, ms, how in [("decider-2b, Decision-1.0 Sol-2B", "~25", "one pass per question, all options"),
                         ("decider-4b, Decision-1.0 Nox-4B, Von 1.2", "35 to 60", "one pass per question, all options"),
                         ("kev-4b", "~200", "one pass per question; float32, unbatched"),
                         ("[MASK] encoder (row 53)", "7 to 10", "one pass, batched"),
                         ("Qwen2.5-3B, the repo's option scorer", "~225", "one row per option: the prompt re-read 12 to 20 times"),
                         ("Qwen2.5-14B, the repo's option scorer", "~600", "one row per option")]:
        print(f"| {lab} | {ms} | {how} |")
