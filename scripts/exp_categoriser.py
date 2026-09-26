"""The production categoriser (PLAN step 33, REAL-5), trained on the REAL-6 users' histories and scored on the REAL-6 cells.

Two ways of training, three ways of giving it the fact DB:
  llm       LoRA (rank 64) on Qwen2.5-3B-Instruct with label SFT in the REAL-6 prompt format: for every history row of every user, the
            user's categories listed, 24 other rows of that user's history as shots, the row as the query, the user's label as the
            answer (loss on the label only). The frozen test items of real6_v1 are never in the training prompts as queries.
  encoder   bge-base fine-tuned contrastively across users: (normalised history string -> the user's category name) pairs with
            in-batch negatives (same-label pairs masked), scored afterwards by exp_real6.py as prototypes over each user's history.
  DB=none   the history alone
  DB=param  parametric injection: the 240 fact-DB records (section 4's sentence and three paraphrases each) mixed into the training
            sequences at 30% (LLM, full-sequence loss) or as (record -> standard category name) pairs (encoder)
  DB=ret    retrieval: the merchant's record is in the prompt before the query at training and at test (the LLM item's prompt_ctx);
            for the encoder the record is appended to the query string at test (ENC_CTX=1 in exp_real6.py)
A quarter of the merchants (real6.db_only_merchants, stratified over categories) never appear in any training row, as query or shot,
so that the unseen cells split into merchants other users labelled and merchants only the DB knows (the REAL-5 number).

usage: uv run python scripts/exp_categoriser.py llm|encoder [none|param|ret]
env: STEPS=200 LR=1e-4 (LLM; 16 sequences per step), DB_FRAC=0.3 (DB share of the LLM sequences under param), RUN_TAG (name suffix), EPOCHS=3 (encoder), SMOKE=1, SEED=0
     REAL6_DB=amb (row 37, REAL-7): the ambiguous fact DB of real6_v1_ambdb.json in place of the disjoint records, for param and ret; adds _amb to the names
     TRAINER=hf (row 38, INFRA-2): transformers + peft instead of unsloth (same LoRA shape, schedule, batches and data order; peft's own
       LoRA init under torch.manual_seed(SEED); plain gradient checkpointing); adds _hf to the names. The adapter format is peft's either way.
outputs: models/adapters/categoriser_Qwen2.5-3B-Instruct_<db>_lora  or  models/adapters/categoriser_bge_<db>; results/categoriser_<route>_<db>.json
  (training stats); the REAL-6 scores come from `scripts/exp_real6.py llm <adapter>` / `encoder <dir>` afterwards. Tracker "categoriser".
"""
import json
import os
import random
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.8")

from ai_experiments import merchants as M
from ai_experiments import real6 as R6
from ai_experiments.evals.tracker import Run
from ai_experiments.paths import ROOT

ROUTE = sys.argv[1] if len(sys.argv) > 1 else "llm"
DB = sys.argv[2] if len(sys.argv) > 2 else "none"
assert ROUTE in ("llm", "encoder") and DB in ("none", "param", "ret")
SMOKE = bool(os.environ.get("SMOKE"))
SEED = int(os.environ.get("SEED", "0"))
STEPS = 3 if SMOKE else int(os.environ.get("STEPS", "200"))
LR = float(os.environ.get("LR", "1e-4"))
DB_FRAC = float(os.environ.get("DB_FRAC", "0.3"))  # share of the LLM's training sequences that are DB texts under DB=param
RUN_TAG = os.environ.get("RUN_TAG", "")  # suffix on the adapter and results names (variants such as the longer parametric run)
EPOCHS = 1 if SMOKE else int(os.environ.get("EPOCHS", "3"))
MICRO, MAXLEN = 4, 1536  # 4 x 4 = 16 sequences per step
LLM_BASE, ENC_BASE = "Qwen/Qwen2.5-3B-Instruct", "BAAI/bge-base-en-v1.5"
REAL6_DB = os.environ.get("REAL6_DB", "v1")
TRAINER = os.environ.get("TRAINER", "unsloth")
LOAD_4BIT = bool(int(os.environ.get("LOAD_4BIT", "1")))  # the unsloth path loads the NF4 4-bit base: unsloth's default, which this script never overrode, so every
# unsloth-trained categoriser is QLoRA on the 4-bit base and must be scored on it (REPORT.md section 44). LOAD_4BIT=0 loads bf16; TRAINER=hf / SCORER=hf are bf16.
assert TRAINER in ("unsloth", "hf")
DOC = R6.load(REAL6_DB)
DBREC = DOC["fact_db"]
DB_ONLY = R6.db_only_merchants()  # no training row (query or shot) may carry one of these merchants; their category can only come from the DB
SFX = f"{DB}{'_' + RUN_TAG if RUN_TAG else ''}{'_amb' if REAL6_DB == 'amb' else ''}{'_hf' if TRAINER == 'hf' else ''}"
OUT_DIR = ROOT / "models" / ("smoke" if SMOKE else "adapters") / (f"categoriser_Qwen2.5-3B-Instruct_{SFX}_lora" if ROUTE == "llm" else f"categoriser_bge_{SFX}")
OUT = ROOT / "results" / f"categoriser_{ROUTE}_{SFX}{'_smoke' if SMOKE else ''}.json"
rng = random.Random(SEED)


def db_texts():
    out = []
    for name, rec in DBREC.items():
        prods = rec.split(" sells ", 1)[1].rstrip(".")
        out += [rec, f"Question: What does {name} sell?\nAnswer: {name} sells {prods}.", f"Shoppers go to {name} for {prods}.", f"Store directory entry: {name} - {prods}."]
    return out


def sft_examples(per_user=150):
    """(prompt, answer) pairs in the REAL-6 format from the users' histories; the test items' merchants are not excluded (they are
    the seen cells), but the test transactions themselves are not history rows."""
    ex = []
    for u in DOC["users"]:
        hist = [h for h in u["history"] if h["merchant"] not in DB_ONLY]
        header = "Categories: " + ", ".join(c["name"] for c in u["categories"]) + "\n\n"
        rows = list(range(len(hist))); rng.shuffle(rows)
        for i in rows[:per_user]:
            h = hist[i]
            others = [hist[j] for j in rng.sample([j for j in rows if j != i], min(24, len(rows) - 1))]
            demo = "".join(f"Transaction: {o['text']} | ${o['amount']:.2f} | {o['weekday']}\nCategory: {o['label']}\n\n" for o in others)
            note = f"Note: {DBREC[h['merchant']]}\n" if DB == "ret" else ""
            ex.append((header + demo + note + f"Transaction: {h['text']} | ${h['amount']:.2f} | {h['weekday']}\nCategory:", " " + h["label"]))
    return ex


TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_llm():
    """The base model with a fresh rank-64 LoRA on every linear layer: through unsloth (the recipe of every adapter so far) or through
    transformers + peft alone (TRAINER=hf), with gradient checkpointing in both."""
    import torch
    if TRAINER == "hf":
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(LLM_BASE)
        model = AutoModelForCausalLM.from_pretrained(LLM_BASE, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
        torch.manual_seed(SEED)
        model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=TARGETS))
    else:
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(LLM_BASE, max_seq_length=MAXLEN, dtype=torch.bfloat16, load_in_4bit=LOAD_4BIT)
        model = FastLanguageModel.get_peft_model(model, r=64, lora_alpha=128, lora_dropout=0.0, bias="none", use_gradient_checkpointing=True, random_state=SEED, target_modules=TARGETS)
    tok.padding_side = "right"
    return model, tok


def train_llm(run):
    import torch
    model, tok = load_llm()
    params = [p for p in model.parameters() if p.requires_grad]
    ex = sft_examples()
    kt = db_texts() if DB == "param" else []
    eos = tok.eos_token_id; pad = tok.pad_token_id or 0

    def enc(pair):
        if isinstance(pair, str):
            ids = tok(pair, add_special_tokens=False)["input_ids"][:MAXLEN - 1] + [eos]; return ids, list(ids)
        a = tok(pair[1], add_special_tokens=False)["input_ids"] + [eos]
        p = tok(pair[0], add_special_tokens=False)["input_ids"][-(MAXLEN - len(a)):]
        return p + a, [-100] * len(p) + a
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / STEPS))
    model.train(); t0 = time.time(); n_sft = n_db = 0; losses = []
    print(f"   {len(ex)} SFT examples, {len(kt)} DB texts, {STEPS} steps x 16 sequences, lr {LR}", flush=True)
    for step in range(STEPS):
        loss_acc = 0.0
        for _ in range(4):
            batch = []
            for _ in range(MICRO):
                if kt and rng.random() < DB_FRAC:
                    batch.append(enc(rng.choice(kt))); n_db += 1
                else:
                    batch.append(enc(rng.choice(ex))); n_sft += 1
            L = max(len(i) for i, _ in batch)
            ids = torch.tensor([i + [pad] * (L - len(i)) for i, _ in batch], device="cuda")
            lab = torch.tensor([l + [-100] * (L - len(l)) for _, l in batch], device="cuda")
            att = (torch.arange(L, device="cuda")[None] < torch.tensor([len(i) for i, _ in batch], device="cuda")[:, None]).long()
            logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
            tgt = lab[:, 1:]; n_lab = max(int((tgt != -100).sum()), 1)
            loss = sum(torch.nn.functional.cross_entropy(logits[i].float(), tgt[i], ignore_index=-100, reduction="sum") for i in range(len(batch))) / n_lab / 4
            loss.backward(); loss_acc += loss.item(); del logits
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        losses.append(loss_acc)
        if (step + 1) % 25 == 0 or step + 1 == STEPS:
            print(f"    step {step + 1}/{STEPS} loss {loss_acc:.3f} {time.time() - t0:.0f}s {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            run.log(dict(train_loss=loss_acc), condition="train", step=step + 1)
    model.save_pretrained(OUT_DIR); tok.save_pretrained(OUT_DIR)
    return dict(train_minutes=round((time.time() - t0) / 60, 1), n_sft_examples=len(ex), n_db_texts=len(kt), seqs_sft=n_sft, seqs_db=n_db, final_loss=round(sum(losses[-10:]) / len(losses[-10:]), 3),
                peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2), peak_reserved_GiB=round(torch.cuda.max_memory_reserved() / 2**30, 2), trainer=TRAINER,
                n_trainable=sum(p.numel() for p in params), losses=[round(x, 4) for x in losses], adapter=str(OUT_DIR.relative_to(ROOT)))


def train_encoder(run):
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(ENC_BASE, device="cuda")
    pairs = []
    for u in DOC["users"]:
        for h in u["history"]:
            if h["merchant"] not in DB_ONLY:
                pairs.append((M.normalize(h["text"]), h["label"]))
    if DB == "param":
        for m in json.loads((ROOT / "data" / "processed" / "transactions_v1.json").read_text())["merchants"]:
            pairs += [(DBREC[m["name"]], m["category"])] * 2
    # DB=ret for the encoder: the training pairs are the history alone; the record is appended to the query at test (ENC_CTX=1)
    bs = 64
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train(); t0 = time.time()
    for ep in range(EPOCHS):
        rng.shuffle(pairs)
        for i in range(0, len(pairs) - bs + 1, bs):
            chunk = pairs[i:i + bs]
            fa = model.tokenize([p[0] for p in chunk]); fp = model.tokenize([p[1] for p in chunk])
            fa = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in fa.items()}; fp = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in fp.items()}
            a = F.normalize(model(fa)["sentence_embedding"], dim=-1); p = F.normalize(model(fp)["sentence_embedding"], dim=-1)
            scores = a @ p.T * 20.0
            same = torch.tensor([[x[1] == y[1] for y in chunk] for x in chunk], device="cuda")
            scores = scores.masked_fill(same & ~torch.eye(len(chunk), dtype=torch.bool, device="cuda"), -1e4)
            loss = F.cross_entropy(scores, torch.arange(len(chunk), device="cuda"))
            loss.backward(); opt.step(); opt.zero_grad()
        print(f"    epoch {ep + 1}/{EPOCHS} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
        run.log(dict(train_loss=loss.item()), condition="train", step=ep + 1)
    model.save(str(OUT_DIR))
    return dict(train_minutes=round((time.time() - t0) / 60, 1), n_pairs=len(pairs), epochs=EPOCHS, final_loss=round(loss.item(), 3), encoder=str(OUT_DIR.relative_to(ROOT)))


cfg = dict(route=ROUTE, db=DB, steps=STEPS, lr=LR, epochs=EPOCHS, seed=SEED, db_frac=DB_FRAC, run_tag=RUN_TAG, real6_db=REAL6_DB, trainer=TRAINER, db_sha=DOC.get("db_sha256"), base=LLM_BASE if ROUTE == "llm" else ENC_BASE, real6_sha=DOC["sha256"], lora_r=64, n_db_only_merchants=len(DB_ONLY))
with Run("categoriser", model=cfg["base"], config=cfg, enabled=not SMOKE) as run:
    stats = train_llm(run) if ROUTE == "llm" else train_encoder(run)
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(dict(config=cfg, **stats), indent=2)); run.artifact(OUT)
    run.log({k: v for k, v in stats.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}, condition="trained")
stats.pop("losses", None)
print(f"=== categoriser {ROUTE} {DB}: {stats}")
