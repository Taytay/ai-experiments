"""Few-shot classifiers that score against the label text, on the REAL-6 cells (PLAN row 40, BASE-6): IBM's FastFit and Knowledgator's
GLiClass beside a logistic-regression head, one model per user from the 24 shots the LLM sees or from the user's whole history.

  fastfit   FastFit (Yehudai and Bendel 2024, arXiv 2404.12365; `fastfit.modeling`, its HF trainer replaced by a plain loop because the
            package's trainer imports datasets.load_metric, gone since datasets 3): a sentence encoder trained per user with the batch-
            contrastive SupCon loss over statements and their label texts (token-level MaxSim similarity, num_repeats copies of the
            batch) plus the small classification head at 0.1, the README's recipe (40 epochs, batch 32, num_repeats 4, Adafactor at 5e-5);
            every query is then scored by MaxSim against the user's label texts (inference_direction "doc"). ENC=bge (the section 37/38
            encoder) or mpnet (FastFit's default, paraphrase-mpnet-base-v2). One saved model per user and condition.
  gliclass  GLiClass (Stepanov et al. 2025, arXiv 2508.07662; knowledgator/gliclass-modern-base-v3.0, 151M): the user's label names and the
            statement in one sequence, one score per label; `zs` = the names only (untrained on this data), `ex` = the user's 24 shots as
            <<EXAMPLE>> demonstrations in the sequence. `gliclass train` fine-tunes it across users on the histories (the DB-only merchants
            held out, as exp_categoriser does; half the rows with the user's shots as examples) into models/adapters/gliclass_ft[_ctx];
            `gliclass gliclass_ft` scores that.
  logreg    logistic regression on the frozen bge-base embeddings per user (the section 24 head), the cheap reference beside the centroid.
CTX=1 appends the merchant's fact-DB record to every statement, in training and at test (section 38's encoder had it at test only,
because that encoder was trained across users; here each user's rows all carry a record, as they would in production).

usage: uv run python scripts/exp_real6_fewshot.py fastfit                         CONDS=shots,full ENC=bge|mpnet EPOCHS=40 CTX=0|1
       uv run python scripts/exp_real6_fewshot.py gliclass [base|train|<dir>]      CONDS=zs,ex CTX=0|1 (train: EPOCHS=3)
       uv run python scripts/exp_real6_fewshot.py logreg                          CONDS=shots,full CTX=0|1
       SMOKE=1: every 10th item, the first 2 users, 2 epochs, no tracker, models under models/smoke
outputs: results/real6_<tag>.json and results/per_item/real6_<tag>.<cond>.jsonl in exp_real6.py's shape (tags fastfit_<enc>, gliclass,
         gliclass_ft, logreg_bge, each + _ctx); tracker experiment "real6"; models/adapters/fastfit_<enc>_<cond>[_ctx]/<user>
"""
import json
import math
import os
import random
import sys
import time

import numpy as np

from ai_experiments import merchants as M
from ai_experiments import real6 as R6
from ai_experiments.evals.tracker import Run
from ai_experiments.paths import ROOT
from ai_experiments.real6_eval import summarize, write_recs

ROUTE = sys.argv[1] if len(sys.argv) > 1 else "fastfit"
WHAT = sys.argv[2] if len(sys.argv) > 2 else "base"
assert ROUTE in ("fastfit", "gliclass", "logreg")
SMOKE = bool(os.environ.get("SMOKE"))
CTX = bool(int(os.environ.get("CTX", "0")))
SEED = int(os.environ.get("SEED", "0"))
ENC = os.environ.get("ENC", "bge")
ENCODERS = {"bge": "BAAI/bge-base-en-v1.5", "mpnet": "sentence-transformers/paraphrase-mpnet-base-v2"}
GLICLASS = "knowledgator/gliclass-modern-base-v3.0"
CONDS = os.environ.get("CONDS", "zs,ex" if ROUTE == "gliclass" else "shots,full").split(",")
EPOCHS = int(os.environ.get("EPOCHS", "3" if ROUTE == "gliclass" else "40"))
if SMOKE:
    EPOCHS = min(EPOCHS, 2)
BS, REPEATS, LR, MAXLEN = 32, 4, 5e-5, 64  # FastFit README: batch 32, num_repeats 4, Adafactor at the default 5e-5; 64 tokens hold a statement plus record (max 33)
DOC = R6.load()
DB_ONLY = R6.db_only_merchants()
USERS = DOC["users"][:2] if SMOKE else DOC["users"]
ITEMS = [it for it in (DOC["items"][::10] if SMOKE else DOC["items"]) if it["user"] in {u["user"] for u in USERS}]
MODELS = ROOT / "models" / ("smoke" if SMOKE else "adapters")
tag = {"fastfit": f"fastfit_{ENC}", "logreg": "logreg_bge", "gliclass": "gliclass" if WHAT in ("base", "train") else WHAT}[ROUTE]
tag += "_ctx" if CTX and not tag.endswith("_ctx") else ""  # a fine-tuned model dir already carries _ctx
random.seed(SEED); np.random.seed(SEED)


def qtext(text, merchant):
    """The encoder's input for a statement: the normalised bank string, plus the merchant's fact-DB record under CTX."""
    return M.normalize(text) + (f" {DOC['fact_db'][merchant]}" if CTX else "")


def rows_of(u, cond):
    """The user's training rows: the 24 shots of the LLM prompt (`shots`, `ex`) or the whole 300-row history (`full`)."""
    if cond in ("shots", "ex"):
        shots = set(u["shots"]); return [h for h in u["history"] if h["text"] in shots]
    return u["history"]


def record(it, pred, uid):
    return dict(id=it["id"], level=it["level"], answer=it["answer"], pred=int(pred), correct=bool(pred == it["answer"]), user=uid, merchant=it["merchant"], known=it["known"], options=it["options"])


def finish(run, results, cond, recs, t0, extra=None):
    results[cond] = summarize(recs); results[cond]["minutes"] = round((time.time() - t0) / 60, 2)
    results[cond].update(extra or {})
    run.log({k: v for k, v in results[cond].items() if isinstance(v, (int, float))}, condition=cond)
    run.artifact(write_recs(tag, cond, recs, SMOKE))


# ----------------------------------------------------------------------------------------------------------------------------- FastFit
def fastfit_user(u, rows, tok, out_dir):
    """Train one FastFit model on the user's rows and return (scores over the user's categories for each of the user's items, minutes)."""
    import torch
    from transformers import AutoConfig
    from transformers.optimization import Adafactor, get_linear_schedule_with_warmup
    from fastfit.modeling import FastFitConfig, FastFitTrainable
    names = [c["name"] for c in u["categories"]]
    texts = [qtext(h["text"], h["merchant"]) for h in rows]; y = torch.tensor([names.index(h["label"]) for h in rows])
    cfg = FastFitConfig.from_encoder_config(AutoConfig.from_pretrained(ENCODERS[ENC]), clf_dim=len(names), num_repeats=REPEATS, clf_factor=0.1, sim_factor=1.0,
                                            mask_prob=0.0, mlm_factor=0.0, rep_tokens="all", inference_type="sim", inference_direction="doc", clf_level="cls")
    type(cfg).has_no_defaults_at_init = True  # transformers 5 instantiates the config class bare on save; FastFitConfig() asserts
    torch.manual_seed(SEED)
    model = FastFitTrainable.from_encoder_pretrained(ENCODERS[ENC], config=cfg).cuda()
    q = tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
    d = tok([names[i] for i in y.tolist()], padding=True, truncation=True, max_length=16, return_tensors="pt")
    steps_per_epoch = math.ceil(len(rows) / BS); total = EPOCHS * steps_per_epoch
    opt = Adafactor(model.parameters(), lr=LR, scale_parameter=False, relative_step=False, warmup_init=False)
    sched = get_linear_schedule_with_warmup(opt, 0, total)
    model.train(); t0 = time.time(); losses = []
    order = torch.arange(len(rows))
    for ep in range(EPOCHS):
        order = order[torch.randperm(len(rows))]
        for i in range(0, len(rows), BS):
            b = order[i:i + BS]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(query_input_ids=q["input_ids"][b].cuda(), query_attention_mask=q["attention_mask"][b].cuda(),
                            doc_input_ids=d["input_ids"][b].cuda(), doc_attention_mask=d["attention_mask"][b].cuda(), labels=y[b].cuda())
            out.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad()
            losses.append(out.loss.item())
    minutes = (time.time() - t0) / 60
    model.eval()
    labs = tok(names, padding=True, truncation=True, max_length=16, return_tensors="pt")
    model.set_documetns((labs["input_ids"], labs["attention_mask"]))
    its = [it for it in ITEMS if it["user"] == u["user"]]
    qi = tok([qtext(it["text"], it["merchant"]) for it in its], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
    scores = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(its), 64):
            scores.append(model.inference_forward(qi["input_ids"][i:i + 64].cuda(), qi["attention_mask"][i:i + 64].cuda()).float().cpu())
    model.save_pretrained(out_dir, safe_serialization=True)
    del model, opt; torch.cuda.empty_cache()
    return its, torch.cat(scores).numpy(), minutes, float(np.mean(losses[-steps_per_epoch:]))


def run_fastfit(run):
    import datasets
    datasets.load_metric = lambda *a, **k: None  # fast-fit 1.2.1's trainer module imports it at import time; the model classes do not use it
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ENCODERS[ENC])
    results = {}
    for cond in CONDS:
        t0 = time.time(); recs = []; minutes = []; losses = []
        for u in USERS:
            out_dir = MODELS / f"fastfit_{ENC}_{cond}{'_ctx' if CTX else ''}" / str(u["user"])
            its, S, m, loss = fastfit_user(u, rows_of(u, cond), tok, out_dir)
            minutes.append(m); losses.append(loss)
            for it, s in zip(its, S):
                recs.append(record(it, int(s.argmax()), u["user"]))
            print(f"    {cond} {u['user']}: {len(rows_of(u, cond))} rows, {len(u['categories'])} categories, {m:.1f} min, loss {loss:.3f}, acc {100 * np.mean([r['correct'] for r in recs if r['user'] == u['user']]):.1f}", flush=True)
        finish(run, results, cond, recs, t0, dict(train_minutes_per_user=round(float(np.mean(minutes)), 2), final_loss_mean=round(float(np.mean(losses)), 3), epochs=EPOCHS))
    return results


# ---------------------------------------------------------------------------------------------------------------------------- GLiClass
def gliclass_examples(u, exclude_text=None):
    return [dict(text=qtext(h["text"], h["merchant"]), labels=[h["label"]]) for h in rows_of(u, "shots") if h["text"] != exclude_text]


def run_gliclass(run):
    import torch
    from transformers import AutoTokenizer
    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    src = GLICLASS if WHAT == "base" else str(MODELS / WHAT)
    model = GLiClassModel.from_pretrained(src); tokz = AutoTokenizer.from_pretrained(src, add_prefix_space=True)
    ids = {t: tokz.convert_tokens_to_ids(t) for t in ("<<LABEL>>", "<<SEP>>", "<<EXAMPLE>>")}
    print(f"    {src}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters, special tokens {ids}, unk {tokz.unk_token_id}, problem_type {model.config.problem_type}", flush=True)
    pipe = ZeroShotClassificationPipeline(model, tokz, classification_type="single-label", device="cuda:0", progress_bar=False, max_classes=25, max_length=1024)
    results = {}
    for cond in CONDS:
        t0 = time.time(); recs = []
        for u in USERS:
            names = [c["name"] for c in u["categories"]]
            its = [it for it in ITEMS if it["user"] == u["user"]]
            texts = [qtext(it["text"], it["merchant"]) for it in its]
            ex = [gliclass_examples(u)] * len(texts) if cond == "ex" else None
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = pipe(texts, names, batch_size=16, examples=ex)
            for it, res in zip(its, out):
                best = max(res, key=lambda r: r["score"]) if res else None
                recs.append(record(it, names.index(best["label"]) if best and best["label"] in names else -1, u["user"]))
            print(f"    {cond} {u['user']}: acc {100 * np.mean([r['correct'] for r in recs if r['user'] == u['user']]):.1f}", flush=True)
        finish(run, results, cond, recs, t0)
    return results


def train_gliclass(run):
    """Fine-tune GLiClass across users on the histories (DB-only merchants held out); half the rows carry the user's shots as examples."""
    import torch
    from transformers import AutoTokenizer
    from gliclass import GLiClassModel
    from gliclass.data_processing import AugmentationConfig, DataCollatorWithPadding, GLiClassDataset
    from gliclass.training import Trainer, TrainingArguments
    out_dir = MODELS / f"gliclass_ft{'_ctx' if CTX else ''}"
    model = GLiClassModel.from_pretrained(GLICLASS); tokz = AutoTokenizer.from_pretrained(GLICLASS, add_prefix_space=True)
    new = [t for t in ("<<LABEL>>", "<<SEP>>", "<<EXAMPLE>>") if tokz.convert_tokens_to_ids(t) in (None, tokz.unk_token_id)]
    if new:
        tokz.add_tokens(new, special_tokens=True); model.resize_token_embeddings(len(tokz)); print(f"    added tokens {new}")
    # gliclass 0.1.20 cannot train a single-label checkpoint as shipped: its collator drops 0-d label tensors (the batch gets an empty
    # list) and its single-label loss reshapes the logits with self.num_labels = -1 ("only one dimension can be inferred"; the Trainer
    # swallows the error and skips every step). Two shims: stack the labels, and a masked cross-entropy over each row's real classes.
    from gliclass import model as GM
    import torch.nn.functional as F

    def ce_loss(self, logits, labels, classes_embedding=None, classes_embedding_mask=None):
        if classes_embedding_mask is not None:
            logits = logits.masked_fill(classes_embedding_mask == 0, -1e4)
        loss = F.cross_entropy(logits.float(), labels.view(-1).long())
        if self.config.contrastive_loss_coef > 0 and classes_embedding is not None:
            loss = loss + GM.sequence_contrastive_loss(classes_embedding, classes_embedding_mask) * self.config.contrastive_loss_coef
        return loss

    class Collator(DataCollatorWithPadding):
        def __call__(self, batch):
            out = super().__call__(batch)
            if isinstance(out.get("labels"), list) and isinstance(batch[0]["labels"], torch.Tensor) and batch[0]["labels"].dim() == 0:
                out["labels"] = torch.stack([b["labels"] for b in batch])
            return out

    assert model.config.problem_type == "single_label_classification", model.config.problem_type
    GM.GLiClassBaseModel.get_loss = ce_loss
    rng = random.Random(SEED); data = []
    for u in USERS:
        names = [c["name"] for c in u["categories"]]
        for h in u["history"]:
            if h["merchant"] in DB_ONLY:
                continue
            ex = dict(text=qtext(h["text"], h["merchant"]), all_labels=list(names), true_labels=[h["label"]])
            if rng.random() < 0.5:
                ex["examples"] = gliclass_examples(u, exclude_text=h["text"])
            data.append(ex)
    rng.shuffle(data)
    if SMOKE:
        data = data[:64]
    ds = GLiClassDataset(data, tokz, AugmentationConfig(enabled=False), {}, 1024, model.config.problem_type, model.config.architecture_type, model.config.prompt_first)
    args = TrainingArguments(output_dir=str(ROOT / "models" / "smoke" / "gliclass_tmp"), learning_rate=1e-5, weight_decay=0.01, others_lr=3e-5, others_weight_decay=0.01,
                             lr_scheduler_type="linear", warmup_ratio=0.05, per_device_train_batch_size=8, num_train_epochs=EPOCHS, save_strategy="no", logging_steps=100,
                             dataloader_num_workers=0, report_to="none", bf16=True, seed=SEED, remove_unused_columns=False, use_cpu=False)
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=Collator(device="cuda:0"), processing_class=tokz)
    t0 = time.time(); res = trainer.train()
    model.save_pretrained(out_dir); tokz.save_pretrained(out_dir)
    stats = dict(train_minutes=round((time.time() - t0) / 60, 1), n_rows=len(data), epochs=EPOCHS, final_loss=round(float(res.training_loss), 4), model=str(out_dir.relative_to(ROOT)))
    run.log({k: v for k, v in stats.items() if isinstance(v, (int, float))}, condition="train")
    return {"train": stats}


# ----------------------------------------------------------------------------------------------------------------------------- logreg
def run_logreg(run):
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    model = SentenceTransformer(ENCODERS["bge"], device="cuda")
    enc = lambda texts: model.encode(texts, batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)  # noqa: E731
    results = {}
    for cond in CONDS:
        t0 = time.time(); recs = []
        for u in USERS:
            names = [c["name"] for c in u["categories"]]; rows = rows_of(u, cond)
            X = enc([qtext(h["text"], h["merchant"]) for h in rows]); y = [names.index(h["label"]) for h in rows]
            its = [it for it in ITEMS if it["user"] == u["user"]]
            Q = enc([qtext(it["text"], it["merchant"]) for it in its])
            if len(set(y)) < 2:
                preds = [y[0]] * len(its)
            else:
                preds = LogisticRegression(max_iter=3000, C=1.0).fit(X, y).predict(Q)
            for it, p in zip(its, preds):
                recs.append(record(it, int(p), u["user"]))
        finish(run, results, cond, recs, t0)
    return results


if __name__ == "__main__":
    OUT = ROOT / "results" / f"real6_{tag}{'_smoke' if SMOKE else ''}.json"
    cfg = dict(route=ROUTE, what=WHAT, ctx=CTX, enc=ENCODERS[ENC] if ROUTE != "gliclass" else GLICLASS, conds=CONDS, epochs=EPOCHS, seed=SEED, bs=BS, num_repeats=REPEATS, lr=LR,
               maxlen=MAXLEN, real6_version=DOC["version"], real6_sha=DOC["sha256"], n_items=len(ITEMS), n_users=len(USERS), shots=DOC["shots"])
    with Run("real6", model=cfg["enc"], config=cfg, enabled=not SMOKE) as run:
        if ROUTE == "gliclass" and WHAT == "train":
            results = train_gliclass(run)
        else:
            results = {"fastfit": run_fastfit, "gliclass": run_gliclass, "logreg": run_logreg}[ROUTE](run)
        OUT.parent.mkdir(exist_ok=True)
        merged = json.loads(OUT.read_text()) if OUT.exists() and not SMOKE else {}
        merged.update(results); OUT.write_text(json.dumps(merged, indent=2)); run.artifact(OUT)
    print(f"=== real6 {tag}")
    for cond, r in results.items():
        print(f"-- {cond}: " + " ".join(f"{k}={r[k]}" for k in sorted(r) if isinstance(r[k], (int, float)) and (k.startswith("R6_") and not k.endswith("_n") or k in ("minutes", "train_minutes", "final_loss"))))
