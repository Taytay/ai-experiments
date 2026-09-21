"""Tables for REPORT.md section 46 (PLAN step 40, BASE-6): FastFit, GLiClass and a logistic-regression head on the REAL-6 cells, beside the
bge centroid and the SFT categoriser of sections 37 and 38.

  46.1  the cells without a record: bge frozen and tuned centroids, logistic regression from the shots / the history, FastFit (bge and
        mpnet encoders) from the shots / the history, GLiClass with the names only / with the 24 shots as examples, untrained and fine-tuned
        across users, and the no-DB SFT categoriser
  46.2  the same with the merchant's record on the query (training and test), beside the tuned encoder with the record at test and the
        record-in-prompt SFT categoriser
  46.3  the merchant-group split (in the shots, elsewhere in the history, other users' rows, DB-only) for both column sets
  46.4  training cost per user and final loss
usage: uv run python scripts/real6_fewshot_tables.py            prints markdown; '-' where a run is missing
"""
import json
import sys
from pathlib import Path

from ai_experiments.paths import ROOT

sys.path.insert(0, str(Path(__file__).parent))
import categoriser_tables as CT  # noqa: E402

R = ROOT / "results"
PLAIN = [("bge frozen centroid, history (sec. 37)", "bge", "full"), ("bge tuned centroid, history (sec. 38)", "categoriser_bge_none", "full"),
         ("logreg on bge, 24 shots", "logreg_bge", "shots"), ("logreg on bge, history", "logreg_bge", "full"),
         ("FastFit bge, 24 shots", "fastfit_bge", "shots"), ("FastFit bge, history", "fastfit_bge", "full"),
         ("FastFit mpnet, 24 shots", "fastfit_mpnet", "shots"), ("FastFit mpnet, history", "fastfit_mpnet", "full"),
         ("GLiClass, names only", "gliclass", "zs"), ("GLiClass + 24 shots", "gliclass", "ex"),
         ("GLiClass tuned, names only", "gliclass_ft", "zs"), ("GLiClass tuned + 24 shots", "gliclass_ft", "ex"),
         ("SFT, no DB (sec. 38)", "categoriser_Qwen2.5-3B-Instruct_none_lora", "noctx")]
CTX = [("bge tuned centroid + record at test (sec. 38)", "categoriser_bge_none_ctx", "full"),
       ("logreg on bge + record, 24 shots", "logreg_bge_ctx", "shots"), ("logreg on bge + record, history", "logreg_bge_ctx", "full"),
       ("FastFit bge + record, 24 shots", "fastfit_bge_ctx", "shots"), ("FastFit bge + record, history", "fastfit_bge_ctx", "full"),
       ("GLiClass + record, names only", "gliclass_ctx", "zs"), ("GLiClass + record + 24 shots", "gliclass_ctx", "ex"),
       ("GLiClass tuned + record, names only", "gliclass_ft_ctx", "zs"), ("GLiClass tuned + record + 24 shots", "gliclass_ft_ctx", "ex"),
       ("SFT + record in prompt (sec. 38)", "categoriser_Qwen2.5-3B-Instruct_ret_lora", "ctx")]


def table(n, title, cols):
    print(f"**Table 46.{n}: {title}**\n")
    print("| cell | " + " | ".join(c[0] for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for label, key in CT.ROWS:
        print(f"| {label} | " + " | ".join(CT.cell(CT.load(t), c, key) for _, t, c in cols) + " |")
    print()


def cost():
    print("**Table 46.4: training cost (minutes per user model on the 3090, mean of the 20 users; GLiClass fine-tune: one model over all users) and final training loss**\n")
    print("| model | rows per user | minutes per user | final loss (last epoch mean) |\n|---|---|---|---|")
    for label, tag, cond, rows in (("FastFit bge, 24 shots", "fastfit_bge", "shots", 24), ("FastFit bge, history", "fastfit_bge", "full", 300),
                                   ("FastFit bge + record, 24 shots", "fastfit_bge_ctx", "shots", 24), ("FastFit bge + record, history", "fastfit_bge_ctx", "full", 300),
                                   ("FastFit mpnet, 24 shots", "fastfit_mpnet", "shots", 24), ("FastFit mpnet, history", "fastfit_mpnet", "full", 300)):
        d = CT.load(tag).get(cond, {})
        print(f"| {label} | {rows} | {d.get('train_minutes_per_user', '-')} | {d.get('final_loss_mean', '-')} |")
    for label, tag in (("GLiClass fine-tune, all users", "gliclass_ft"), ("GLiClass fine-tune + record, all users", "gliclass_ft_ctx")):
        d = CT.load(tag).get("train", {})
        print(f"| {label} | {d.get('n_rows', '-')} (total) | {d.get('train_minutes', '-')} (total) | {d.get('final_loss', '-')} |")
    for label, tag, cond in (("logreg on bge, history", "logreg_bge", "full"), ("GLiClass, names only", "gliclass", "zs"), ("GLiClass + 24 shots", "gliclass", "ex")):
        d = CT.load(tag).get(cond, {})
        if d:
            print(f"\n{label}: {d.get('minutes')} minutes for all 20 users.")


def main():
    table(1, "few-shot classifiers on the REAL-6 cells without a record, one model per user from the 24 shots or the 300-row history "
             "(accuracy % [95% bootstrap interval], * = inside the null band; chance about 7)", PLAIN)
    table(2, "the same with the merchant's fact-DB record appended to the statement, in training and at test", CTX)
    CT.in_shots(PLAIN, "Table 46.3a: the merchant-group split without a record (accuracy %; groups as in Table 38.2)")
    CT.in_shots(CTX, "Table 46.3b: the merchant-group split with the record on the query (accuracy %)")
    print()
    cost()


if __name__ == "__main__":
    main()
