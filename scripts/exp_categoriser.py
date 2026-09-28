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
     CHAT=1 (row 39, REAL-8): the SFT pairs through the instruct model's chat template (real6.chat_prompt: the prompt up to the query line as
       the user turn, "Category:" opening the assistant turn, the label after it); adds _chat to the names. Score such an adapter with exp_real6.py, which reads CHAT from the name.
     FOLD=k (row 42, REAL-10): train on the users outside fold k of four (fold = user id mod 4, five users each), so the fold's users are
       held out; adds _f<k>. Score with `USERS=<the fold's ids> exp_real6.py llm <adapter>`.
     RENAME=p (row 42): rename augmentation, per training episode each of the user's category names replaced with probability p by a
       fresh coined word, consistently in the category list, the shots and the target; adds _ren<p*100>.
     ALL_LABELS=1 (row 56, TRAIN-11): the loss on every shot's label inside the prompt as well as the target's (each shot label predicted
       from the category list and the shots before it), so one 850-token sequence carries about 25 supervised answers instead of 1; the
       label tokens are found through the tokenizer's offset mapping on the whole prompt, so training sees the scorer's tokenisation;
       adds _alllab. No-DB arm only (the shots carry no record, so the record arm would learn its skill from 1 label in 25).
     ANS_WEIGHT=w (row 56, with ALL_LABELS): each sequence's loss is w x the final answer's mean token loss + (1 - w) x the shot
       labels' mean token loss, instead of one mean over all labelled tokens (where the answer is ~3 tokens in ~85); adds _aw<w*100>.
     DBEP=p (row 57, REAL-15): database episodes. With probability p a training episode has 8 of its shots and its target replaced by
       synthetic statement rows of fact-DB merchants (rendered by the generator, amounts from the category's log-normal, any string
       that is a REAL-6 test string refused), each labelled with the training user's own name for the merchant's DB category (merchants
       whose category the user split are skipped: the DB cannot say which half). DB-only merchants included: their labels come from the
       DB, never from a user. Adds _dbep<p*100>. With ALL_LABELS every such label carries the loss.
     DB_EPISODES=n (row 59, REAL-17): n database episodes added to the pool as their own episodes (the user episodes unchanged): each
       from a random training user's history with 8 shots and the target replaced by DB rows, the targets cycling through the DB's
       merchants so each gets its share (about 9n / DB size rows per merchant); adds _dbe<n>. Use instead of DBEP to set exposure.
     DB_EXTRA=n (row 59, REAL-17): the fact DB padded with n generated opaque merchants (merchants.build_extra, seed 59), which the
       database episodes (DBEP) draw from alongside REAL-6's 240; adds _dbx<n>. Scoring is unchanged (REAL-6's items only).
     LOOKUP=p, OVERRIDE=q (row 85): per episode, with probability q (a fact-DB target) the user files the target's merchant under
       another of their categories than its DB category, in two or more shots and in the target, so the user's filing beats the DB
       and the name; else with probability p two or more shots carry the target's own merchant under the target's label (the
       history lookup, misleading names included). Adds _lk<p*100> / _ov<q*100>.
     ALT=p (row 86): with probability p an episode becomes an alternation episode: a new specific category is added to the user's
       list and takes some of the target merchant's transactions by a rule drawn per episode (amount over a random threshold, a
       random set of days, either with 20% noise, or at random with a 30 to 50% share), six shots of the merchant filed by the rule
       and the target by the rule (sampled when unobservable). Names are coined or from ALT_NAMES, disjoint from alternation_v1's.
       Adds _alt<p*100>.
     MISLEAD=<set> (row 83): the in-DB merchants of data/processed/<set>.json (build_mislead.py: misleading names and neutral twins)
       added to the fact DB the database episodes (DBEP) draw from; adds _<set>. Score on ITEMS_SET=<set>.
     DB_CAT=1 (row 57): the DB texts of DB=param also state the merchant's category ("X is a Groceries store that sells ..."), so the
       prose arm has the same information as the episodes; adds _dbcat.
     REC_CAT=1 (row 58, REAL-16): with DB=ret, the record in the training note states the merchant's category too
       (real6.category_record); adds _reccat, from which exp_real6.py puts the same record in the test prompt.
     POI_SHOTS=kind (row 72, POI-1): per episode, with probability 1/2 up to 6 of the 24 shots are history places of the target's
       Overture basic category (the test layout of poi1_v1_kshots), otherwise none are (an unseen-kind episode, as half the test items
       are); adds _pksh. POI_REC=1: a "Note: <place> is listed as a <kind>." line before the target (the test items' prompt_ctx);
       adds _prec. POI_KIND=1: a "Kind: <kind>" line under every shot and the target (poi1_v1_kinds); adds _pkind.
     POI_DESC=1 (row 73): the header lists each category with the kinds the user has filed under it, from the history without the
       target (poi1_v1_desc); adds _pdesc. POI_UNSEEN=p: in a share p of episodes no place of the target's kind is in the shots or the
       description (an unseen-kind episode); adds _uns<p*100>.
     DECOY=p (row 74, REAL-20; REAL-6 users): in a share p of episodes one shot of the target's standard category that carries the
       target's label is relabelled to one of the user's categories that holds none of that standard category (a same-kind example
       filed elsewhere, section 60's decoy), when another shot still carries the target's label; the decoy's label gets no loss. Adds
       _dec<p*100>. EMPTY=k (row 74): in half the episodes 1 to k fresh coined categories with no examples join the category list
       (label induction v2's empty categories: no elimination); adds _emp<k>.
     TRAINER=hf (row 38, INFRA-2): transformers + peft instead of unsloth (same LoRA shape, schedule, batches and data order; peft's own
       LoRA init under torch.manual_seed(SEED); plain gradient checkpointing); adds _hf to the names. The adapter format is peft's either way.
outputs: models/adapters/categoriser_Qwen2.5-3B-Instruct_<db>_lora  or  models/adapters/categoriser_bge_<db>; results/categoriser_<route>_<db>.json
  (training stats); the REAL-6 scores come from `scripts/exp_real6.py llm <adapter>` / `encoder <dir>` afterwards. Tracker "categoriser".
"""
import collections
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
MICRO = int(os.environ.get("MICRO", "4"))  # sequences per forward/backward (row 34: MICRO=16 = one pass per step on an 80 GB GPU)
EFF_BATCH = int(os.environ.get("EFF_BATCH", "16"))  # sequences per optimizer step (row 60, TRAIN-12: the batch-size ablation); 16 in every run before it
assert EFF_BATCH % MICRO == 0
ACCUM, MAXLEN = EFF_BATCH // MICRO, 1536
LLM_BASE, ENC_BASE = os.environ.get("LLM_BASE", "Qwen/Qwen2.5-3B-Instruct"), "BAAI/bge-base-en-v1.5"  # row 70: LLM_BASE=Qwen/Qwen2.5-7B-Instruct / 14B
REAL6_DB = os.environ.get("REAL6_DB", "v1")
QWEN35 = "Qwen3.5" in os.environ.get("LLM_BASE", "")  # row 78: an Apache-2.0 base (Qwen2.5-3B is under the Qwen Research licence); unsloth's fast path does not cover its DeltaNet layers
TRAINER = os.environ.get("TRAINER", "hf" if QWEN35 else "unsloth")
SHOTS = os.environ.get("SHOTS", "fixed")  # row 41 (REAL-9): the 24 training shots chosen per query by a rule of real6_shots (fixed = 24 random rows)
FOLD = os.environ.get("FOLD")  # row 42: hold out the users with user % 4 == FOLD
RENAME = float(os.environ.get("RENAME", "0"))
ALL_LABELS = bool(int(os.environ.get("ALL_LABELS", "0")))  # row 56: loss on every shot label in the prompt too
DBEP = float(os.environ.get("DBEP", "0"))  # row 57: share of training episodes rebuilt around synthetic fact-DB rows
DB_CAT = bool(int(os.environ.get("DB_CAT", "0")))
DB_EXTRA = int(os.environ.get("DB_EXTRA", "0"))  # row 59: generated merchants added to the fact DB the episodes draw from
DB_EPISODES = int(os.environ.get("DB_EPISODES", "0"))  # row 59: database episodes added to the pool as their own episodes
REC_CAT = bool(int(os.environ.get("REC_CAT", "0")))  # row 58: the record in the note states the category (DB=ret)  # row 57: the prose DB texts state the category too
ANS_WEIGHT = float(os.environ.get("ANS_WEIGHT", "0"))  # row 56: the final answer's share of each sequence's loss under ALL_LABELS (0 = token mean)  # row 42: per-episode probability of replacing each category name by a coined word
CHAT = bool(int(os.environ.get("CHAT", "0")))  # row 39: the prompt as the user turn of the chat template, the label as the assistant turn
LOAD_4BIT = bool(int(os.environ.get("LOAD_4BIT", "1")))  # the unsloth path loads the NF4 4-bit base: unsloth's default, which this script never overrode, so every
# unsloth-trained categoriser is QLoRA on the 4-bit base and must be scored on it (REPORT.md section 44). LOAD_4BIT=0 loads bf16; TRAINER=hf / SCORER=hf are bf16.
assert TRAINER in ("unsloth", "hf")
assert not (ALL_LABELS and (CHAT or DB == "ret")), "ALL_LABELS is built for the plain prompt (no record in it)"
assert not (DBEP and (CHAT or DB == "ret")), "DBEP builds plain episodes without a record"
assert not ANS_WEIGHT or (ALL_LABELS and 0 < ANS_WEIGHT < 1), "ANS_WEIGHT needs ALL_LABELS and 0 < w < 1"
POI = os.environ.get("POI", "")
MISLEAD = os.environ.get("MISLEAD", "")  # row 83
SHOT_NOISE = float(os.environ.get("SHOT_NOISE", "0"))  # row 94: each shot misfiled under another of the user's categories with this probability, its label out of the loss
ALT = float(os.environ.get("ALT", "0"))  # row 86
ALT_DAYS = os.environ.get("ALT_DAYS", "")  # row 92: "rand" draws each day rule's days as a random set of 1 to 3 weekdays (default: five fixed sets)
ALT_SOFT = os.environ.get("ALT_SOFT", "") == "1"  # row 86 v2: noisy / random alternation targets as their true split (SOFT[prompt]), rules drawn evenly
SOFT = {}  # row 86 v2: prompt -> {answer name: probability}; read by exp_decider_finetune.py
ALT_NAMES = ["Splurges", "Gifts", "Hobby time", "Self care", "Weekend treats", "Family outings", "Big buys", "Little luxuries", "Brunch club", "Game night"]
LOOKUP, OVERRIDE = float(os.environ.get("LOOKUP", "0")), float(os.environ.get("OVERRIDE", "0"))  # row 85
POI_DB = os.environ.get("POI_DB", "")  # row 66 (REAL-21): data/processed/<POI_DB>.json (build_poi1_db.py), the places DBEP episodes draw from  # row 65 (POI-1): train on the users of data/processed/<POI>.json (poi1_v1: real Overture places) instead of REAL-6's
assert not POI or (DB == "none" and (not DBEP or POI_DB) and not DB_EPISODES and not REC_CAT and SHOTS == "fixed"), "POI-1's only fact DB is POI_DB"
PLACES = json.loads((ROOT / "data" / "processed" / f"{POI_DB}.json").read_text())["places"] if POI_DB else []
POI_SHOTS, POI_REC, POI_KIND = os.environ.get("POI_SHOTS", "fixed"), bool(int(os.environ.get("POI_REC", "0"))), bool(int(os.environ.get("POI_KIND", "0")))
POI_DESC, POI_UNSEEN = bool(int(os.environ.get("POI_DESC", "0"))), float(os.environ.get("POI_UNSEEN", "0"))
DECOY, EMPTY = float(os.environ.get("DECOY", "0")), int(os.environ.get("EMPTY", "0"))
assert POI or (POI_SHOTS == "fixed" and not POI_REC and not POI_KIND and not POI_DESC and not POI_UNSEEN), "POI_* are POI-1 layouts"
assert not (POI_UNSEEN and POI_SHOTS == "kind"), "POI_SHOTS=kind has its own unseen-kind half"
DOC = json.loads((ROOT / "data" / "processed" / f"{POI}.json").read_text()) if POI else R6.load(REAL6_DB)
DBREC = DOC.get("fact_db", {})
DB_ONLY = set() if POI else R6.db_only_merchants()
if not POI:  # row 74: each REAL-6 merchant's standard category, for decoys
    from ai_experiments import transactions as _T
    STD = {m["name"]: m["category"] for m in _T.load()["merchants"]}  # no training row (query or shot) may carry one of these merchants; their category can only come from the DB
SFX = f"{'_'.join([POI, DB]) if POI else DB}{'_' + RUN_TAG if RUN_TAG else ''}{'_chat' if CHAT else ''}{'_shots' + SHOTS if SHOTS != 'fixed' else ''}{'_amb' if REAL6_DB == 'amb' else ''}{'_hf' if TRAINER == 'hf' else ''}{'_pksh' if POI_SHOTS == 'kind' else ''}{'_prec' if POI_REC else ''}{'_pkind' if POI_KIND else ''}{'_pdesc' if POI_DESC else ''}{f'_uns{round(POI_UNSEEN * 100)}' if POI_UNSEEN else ''}{'_' + POI_DB.replace('poi1_v1_', '') if POI_DB else ''}{f'_dec{round(DECOY * 100)}' if DECOY else ''}{f'_emp{EMPTY}' if EMPTY else ''}{'_f' + FOLD if FOLD is not None else ''}{f'_ren{round(RENAME * 100)}' if RENAME else ''}{'_alllab' if ALL_LABELS else ''}{f'_aw{round(ANS_WEIGHT * 100)}' if ANS_WEIGHT else ''}{f'_dbep{round(DBEP * 100)}' if DBEP else ''}{'_dbcat' if DB_CAT else ''}{'_reccat' if REC_CAT else ''}{f'_dbx{DB_EXTRA}' if DB_EXTRA else ''}{f'_dbe{DB_EPISODES}' if DB_EPISODES else ''}{'_' + MISLEAD if MISLEAD else ''}{f'_sn{round(SHOT_NOISE * 100)}' if SHOT_NOISE else ''}{f'_alt{round(ALT * 100)}' if ALT else ''}{'s' if ALT and ALT_SOFT else ''}{'d' if ALT and ALT_DAYS == 'rand' else ''}{f'_lk{round(LOOKUP * 100)}' if LOOKUP else ''}{f'_ov{round(OVERRIDE * 100)}' if OVERRIDE else ''}"
OUT_DIR = ROOT / "models" / ("smoke" if SMOKE else "adapters") / (f"categoriser_{LLM_BASE.split('/')[-1]}_{SFX}_lora" if ROUTE == "llm" else f"categoriser_bge_{SFX}")
_BASE_TAG = f"_{LLM_BASE.split('/')[-1]}" if ROUTE == "llm" and LLM_BASE != "Qwen/Qwen2.5-3B-Instruct" else ""  # a 7B / 14B run once overwrote the 3B's file of the same SFX
OUT = ROOT / "results" / f"categoriser_{ROUTE}{_BASE_TAG}_{SFX}{'_smoke' if SMOKE else ''}.json"
rng = random.Random(SEED)


def db_texts():
    out = []
    for name, rec in DBREC.items():
        prods = rec.split(" sells ", 1)[1].rstrip(".")
        if DB_CAT:  # row 57: the same information the database episodes get
            cat = MERCHANT[name]["category"]
            out += [f"{name} is a {cat} store that sells {prods}.", f"Question: What category is {name}?\nAnswer: {name} is {cat}; it sells {prods}.",
                    f"Shoppers go to {name} for {prods}.", f"Store directory entry: {name} ({cat}) - {prods}."]
        else:
            out += [rec, f"Question: What does {name} sell?\nAnswer: {name} sells {prods}.", f"Shoppers go to {name} for {prods}.", f"Store directory entry: {name} - {prods}."]
    return out


from ai_experiments import transactions as T  # noqa: E402
MERCHANT = {m["name"]: m for m in T.load()["merchants"]}
ITEM_TEXTS = {it["text"] for it in DOC["items"]}
if DB_EXTRA:  # row 59: the padded database (records in the REAL-6 wording)
    for m in M.build_extra(DB_EXTRA, taken=set(MERCHANT)):
        MERCHANT[m["name"]] = m; DBREC[m["name"]] = f"{m['name']} is a store that sells {M.prods(m)}."
    assert DBEP or DB_EPISODES, "DB_EXTRA only matters with database episodes (DBEP or DB_EPISODES)"
if MISLEAD:  # row 83: the misleading-name set's in-DB merchants (its held-out ones never reach training)
    assert DBEP and not POI, "MISLEAD needs REAL-6 database episodes"
    _ml = json.loads((ROOT / "data" / "processed" / f"{MISLEAD}.json").read_text())
    for m in _ml["merchants"]:
        if m["in_db"]:
            assert m["name"] not in MERCHANT
            MERCHANT[m["name"]] = m; DBREC[m["name"]] = _ml["fact_db"][m["name"]]


def db_row(u, rng, names_ok, merchant=None):
    """A synthetic statement row of a fact-DB merchant, labelled with user u's name for its DB category (row 57); None if none maps."""
    import math
    std_to_name = {std: c["name"] for c in u["categories"] if "split" not in c for std in c["standard"]}
    for _ in range(20):
        m = MERCHANT[merchant or rng.choice(names_ok)]
        if m["category"] not in std_to_name:
            continue
        for _ in range(5):
            text = T.render(m, rng)
            if text not in ITEM_TEXTS:
                mu, sig = T.AMOUNT[m["category"]]
                return dict(text=text, amount=round(math.exp(rng.gauss(mu, sig)), 2), weekday=rng.choice(T.WEEKDAYS), merchant=m["name"], label=std_to_name[m["category"]], synthetic=True)
    return None


def merchant_row(name, label, rng):
    """Row 85: a synthetic statement row of merchant `name` filed under `label` (a test string never reused); None if none found."""
    import math
    m = MERCHANT[name]
    for _ in range(5):
        text = T.render(m, rng)
        if text not in ITEM_TEXTS:
            mu, sig = T.AMOUNT[m["category"]]
            return dict(text=text, amount=round(math.exp(rng.gauss(mu, sig)), 2), weekday=rng.choice(T.WEEKDAYS), merchant=name, label=label, synthetic=True)
    return None


def lookup_override(u, h, others, rng):
    """Row 85 (LOOKUP / OVERRIDE): returns (h, others) with the target's merchant placed in the shots, relabelled by the user or not."""
    if POI or h.get("merchant") not in MERCHANT or not (LOOKUP or OVERRIDE):
        return h, others
    x = rng.random()
    if x < OVERRIDE and h["merchant"] in DBREC:
        alt = [c["name"] for c in u["categories"] if c["name"] != h["label"] and "split" not in c]
        if not alt:
            return h, others
        h = dict(h, label=rng.choice(alt))
    elif not x < OVERRIDE + LOOKUP:
        return h, others
    others = [dict(o, label=h["label"]) if o.get("merchant") == h["merchant"] else o for o in others]  # the user files it one way
    have = sum(o.get("merchant") == h["merchant"] for o in others)
    free = [k for k, o in enumerate(others) if o.get("merchant") != h["merchant"]]
    for k in rng.sample(free, max(0, min(len(free), 2 - have))):
        r = merchant_row(h["merchant"], h["label"], rng)
        if r is not None:
            others[k] = r
    return h, others


def alternation(u, h, others, rng):
    """Row 86 (ALT): (h, others, B) with a new specific category B taking some of the target merchant's rows by a drawn rule."""
    import math
    if POI or h.get("merchant") not in MERCHANT:
        return h, others, None, None
    m = MERCHANT[h["merchant"]]; A = h["label"]
    taken = {c["name"] for c in u["categories"]}
    B = rng.choice([n for n in ALT_NAMES if n not in taken]) if rng.random() < 0.5 else coined(rng, taken)
    kind = rng.choice(["amount", "day", "partial", "random"] if ALT_SOFT else ["amount", "amount", "day", "day", "partial", "random"])
    base = rng.choice(["amount", "day"])
    t = math.exp(rng.uniform(math.log(15), math.log(120)))
    days = set(rng.sample(T.WEEKDAYS, rng.randint(1, 3))) if ALT_DAYS == "rand" else rng.choice([{"Fri", "Sat"}, {"Sat", "Sun"}, {"Fri", "Sat", "Sun"}, {"Mon", "Tue", "Wed", "Thu"}, {"Sun"}])
    share = rng.uniform(0.3, 0.5)
    rule_on = base if kind in ("partial", "random") else kind

    def one(side=None):
        side = rng.random() < 0.5 if side is None else side
        if rule_on == "amount":
            a = round(t * math.exp(abs(rng.gauss(0.5, 0.35)) * (1 if side else -1)), 2); d = rng.choice(T.WEEKDAYS)
        else:
            a = round(math.exp(rng.gauss(*T.AMOUNT[m["category"]])), 2)
            d = rng.choice(sorted(days) if side else sorted(set(T.WEEKDAYS) - days))
        b = side if kind in ("amount", "day") else (side if rng.random() >= 0.2 else not side) if kind == "partial" else rng.random() < share
        pb = float(side) if kind in ("amount", "day") else (0.8 if side else 0.2) if kind == "partial" else share
        for _ in range(5):
            text = T.render(m, rng)
            if text not in ITEM_TEXTS:
                return dict(text=text, amount=a, weekday=d, merchant=m["name"], label=B if b else A, synthetic=True, p_b=pb)
        return None
    for _ in range(10):
        rows = [one() for _ in range(6)]
        if all(rows) and 2 <= sum(r["label"] == B for r in rows) <= 4:
            break
    else:
        return h, others, None, None
    tgt = one()
    if tgt is None:
        return h, others, None, None
    same = [k for k, o in enumerate(others) if o.get("merchant") == m["name"]]
    free = same + rng.sample([k for k in range(len(others)) if k not in same], max(0, 6 - len(same)))
    others = list(others)
    for k, r in zip(free, rows):
        others[k] = r
    others = [o for k, o in enumerate(others) if o.get("merchant") != m["name"] or k in free[:6]]
    soft = {B: tgt["p_b"], A: 1 - tgt["p_b"]} if ALT_SOFT and kind in ("partial", "random") else None
    return tgt, others, B, soft


SYLL = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"]


def poi_db_row(u, rng):
    """Row 66: a places-database row (a POI-1 test place in the injected half) labelled with POI user u's category for its Overture basic
    category; None when u's scheme holds none of the tried kinds."""
    lab = {b: c["name"] for c in u["categories"] for b in c["basic"]}
    for _ in range(20):
        p = rng.choice(PLACES)
        if p["basic"] in lab:
            return dict(text=p["text"], amount=p["amount"], weekday=rng.choice(T.WEEKDAYS), merchant=p["merchant"], basic=p["basic"], label=lab[p["basic"]], synthetic=True)
    return None


def coined(r, taken):
    """A fresh pronounceable word ("Tavoli", "Mekru") not among `taken` (row 42's rename augmentation)."""
    while True:
        w = "".join(r.choice(SYLL) for _ in range(r.randint(2, 3)))
        w = (w + r.choice(["", "", "n", "r", "x"])).capitalize()
        if w not in taken:
            return w


def kind_name(basic):
    """An Overture basic category as the POI-1 prompts show it (build_poi1.readable)."""
    return basic.replace("_or_", " or ").replace("_and_", " and ").replace("_", " ").capitalize()


def training_users():
    return [u for u in DOC["users"] if FOLD is None or u["user"] % 4 != int(FOLD)]


def sft_examples(per_user=150):
    """(prompt, answer) pairs in the REAL-6 format from the users' histories; the test items' merchants are not excluded (they are
    the seen cells), but the test transactions themselves are not history rows."""
    ex = []
    DB_NAMES = sorted(DBREC)  # row 57: every fact-DB merchant, DB-only ones included
    shots = None
    if SHOTS != "fixed":
        from ai_experiments.real6_shots import Shots
        shots = Shots(SHOTS, seed=SEED, exclude=DB_ONLY)
    for u in training_users():
        hist = [h for h in u["history"] if h["merchant"] not in DB_ONLY]
        header = "Categories: " + ", ".join(c["name"] for c in u["categories"]) + "\n\n"
        rows = list(range(len(hist))); rng.shuffle(rows)
        for i in rows[:per_user]:
            h = hist[i]
            unseen = POI_UNSEEN and rng.random() < POI_UNSEEN  # row 73: the target's kind absent from shots and description
            if shots is not None:  # the rule's pool for training is the same DB_ONLY-free history, in the same order
                others = shots.select(u, h["text"], train=True, query_index=i)
            elif POI_SHOTS == "kind":  # row 72: the kshots layout half the time, no place of the target's kind the other half
                pool = [j for j in rows if j != i and hist[j]["basic"] != h["basic"]]
                same = [j for j in rows if j != i and hist[j]["basic"] == h["basic"]] if rng.random() < 0.5 else []
                same = rng.sample(same, min(6, len(same)))
                others = [hist[j] for j in same + rng.sample(pool, min(24 - len(same), len(pool)))]
                rng.shuffle(others)
            else:
                pool = [j for j in rows if j != i and not (unseen and hist[j]["basic"] == h["basic"])]
                others = [hist[j] for j in rng.sample(pool, min(24, len(pool)))]
            if DBEP and rng.random() < DBEP:  # row 57: a database episode (8 shots and the target from the fact DB; row 66: the places DB)
                others = list(others)
                row = (lambda: poi_db_row(u, rng)) if POI else (lambda: db_row(u, rng, DB_NAMES))
                for j in rng.sample(range(len(others)), min(8, len(others))):
                    r = row()
                    if r is not None:
                        others[j] = r
                r = row()
                h = r if r is not None else h
            if LOOKUP or OVERRIDE:  # row 85
                h, others = lookup_override(u, h, list(others), rng)
            alt_b = alt_soft = None
            if ALT and rng.random() < ALT:  # row 86
                h, others, alt_b, alt_soft = alternation(u, h, list(others), rng)
            names = {c["name"]: c["name"] for c in u["categories"]}
            if RENAME:  # the same fresh word for a category everywhere in this episode
                taken = set(names)
                for n in names:
                    if rng.random() < RENAME:
                        names[n] = coined(rng, taken); taken.add(names[n])
                hdr = "Categories: " + ", ".join(names[c["name"]] for c in u["categories"]) + "\n\n"
            else:
                hdr = header
            decoy = None
            if DECOY and not POI and rng.random() < DECOY:  # row 74: one same-kind shot filed under a category that holds none of its kind
                s_std = STD[h["merchant"]]
                same = [k for k, o in enumerate(others) if o["label"] == h["label"] and STD.get(o["merchant"]) == s_std]
                if same and sum(o["label"] == h["label"] for o in others) >= 2:
                    far = [c["name"] for c in u["categories"] if s_std not in c.get("standard", [])]
                    if far:
                        decoy = rng.choice(same); others = list(others); others[decoy] = dict(others[decoy], label=rng.choice(far))
            if EMPTY and rng.random() < 0.5:  # row 74: coined categories with no examples, placed at random in the list
                cats = [names[c["name"]] for c in u["categories"]]; taken = set(cats) | set(names)
                for _ in range(rng.randint(1, EMPTY)):
                    w = coined(rng, taken); taken.add(w); cats.insert(rng.randint(0, len(cats)), w)
                hdr = "Categories: " + ", ".join(cats) + "\n\n"
            if POI_DESC:  # row 73: the kinds filed under each category, from the history without the target (and its kind if unseen)
                cnt = {c["name"]: collections.Counter() for c in u["categories"]}
                for j in rows:
                    if j != i and not (unseen and hist[j]["basic"] == h["basic"]):
                        cnt[hist[j]["label"]][hist[j]["basic"]] += 1
                hdr = "Categories:\n" + "\n".join(f"- {names[c['name']]}: " + (", ".join(kind_name(b).lower() for b, _ in cnt[c["name"]].most_common(4)) or "(nothing filed yet)") for c in u["categories"]) + "\n\n"
            if alt_b:  # row 86: the specific category joins the list at a random place
                cl = hdr[len("Categories: "):].rstrip("\n").split(", "); cl.insert(rng.randint(0, len(cl)), alt_b)
                hdr = "Categories: " + ", ".join(cl) + "\n\n"; names = dict(names, **{alt_b: alt_b})
            demo, spans = "", []  # spans: character ranges of the shot labels (leading space included), for ALL_LABELS
            kind = (lambda r: f"Kind: {kind_name(r['basic'])}\n") if POI_KIND else (lambda r: "")
            noisy = {k for k in range(len(others)) if SHOT_NOISE and rng.random() < SHOT_NOISE}  # row 94
            shown = [v for v in dict.fromkeys(names.values())]
            for k, o in enumerate(others):
                demo += f"Transaction: {o['text']} | ${o['amount']:.2f} | {o['weekday']}\n{kind(o)}Category:"
                lab = " " + names[o["label"]]
                if k in noisy:  # row 94: a misfiled row, as users make; no loss on it
                    lab = " " + rng.choice([v for v in shown if v != names[o["label"]]])
                if k != decoy and k not in noisy:  # row 74: the decoy's arbitrary label carries no loss
                    spans.append((len(hdr) + len(demo), len(hdr) + len(demo) + len(lab)))
                demo += lab + "\n\n"
            note = f"Note: {R6.category_record(h['merchant'], DBREC[h['merchant']]) if REC_CAT else DBREC[h['merchant']]}\n" if DB == "ret" else ""
            if POI_REC:  # row 72: the test items' record line (build_poi1: "<name> is listed as a <kind>.")
                note = f"Note: {h['merchant']} is listed as a {kind_name(h['basic']).lower()}.\n"
            prompt = hdr + demo + note + f"Transaction: {h['text']} | ${h['amount']:.2f} | {h['weekday']}\n{kind(h)}Category:"
            if alt_soft:  # row 86 v2
                SOFT[prompt] = {names[k]: v for k, v in alt_soft.items()}
            ex.append((R6.chat_prompt(prompt) if CHAT else prompt, " " + names[h["label"]]) + ((spans,) if ALL_LABELS else ()))
    if DB_EPISODES:  # row 59: database episodes as their own pool entries, targets cycling through the DB's merchants
        users = training_users(); order = list(DB_NAMES); rng.shuffle(order)
        for k in range(DB_EPISODES):
            u = rng.choice(users)
            hist = [x for x in u["history"] if x["merchant"] not in DB_ONLY]
            others = rng.sample(hist, min(24, len(hist)))
            for j in rng.sample(range(len(others)), min(8, len(others))):
                r = db_row(u, rng, DB_NAMES)
                if r is not None:
                    others[j] = r
            h = db_row(u, rng, DB_NAMES, merchant=order[k % len(order)]) or db_row(u, rng, DB_NAMES)
            if h is None:
                continue
            hdr = "Categories: " + ", ".join(c["name"] for c in u["categories"]) + "\n\n"
            demo, spans = "", []
            for o in others:
                demo += f"Transaction: {o['text']} | ${o['amount']:.2f} | {o['weekday']}\nCategory:"
                lab = " " + o["label"]; spans.append((len(hdr) + len(demo), len(hdr) + len(demo) + len(lab))); demo += lab + "\n\n"
            prompt = hdr + demo + f"Transaction: {h['text']} | ${h['amount']:.2f} | {h['weekday']}\nCategory:"
            ex.append((prompt, " " + h["label"]) + ((spans,) if ALL_LABELS else ()))
    return ex


TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] + (["in_proj_qkv", "in_proj_z", "out_proj"] if QWEN35 else [])  # Qwen3.5's Gated DeltaNet projections


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
        tok = getattr(tok, "tokenizer", tok)  # row 78: Qwen3.5 checkpoints are vision-language; unsloth returns their processor, which reads text as an image
        model = FastLanguageModel.get_peft_model(model, r=64, lora_alpha=128, lora_dropout=0.0, bias="none", use_gradient_checkpointing=True, random_state=SEED, target_modules=TARGETS)
    tok.padding_side = "right"
    if CHAT:  # the hand-written wrapper must be the tokenizer's own template, and the label must end the assistant turn (eos is <|im_end|>)
        assert R6.chat_wrap("x") == tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True), "chat template drift"
        assert tok.eos_token == "<|im_end|>", tok.eos_token
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
            ids = tok(pair, add_special_tokens=False)["input_ids"][:MAXLEN - 1] + [eos]; return ids, list(ids), 0
        a = tok(pair[1], add_special_tokens=False)["input_ids"] + [eos]
        if len(pair) == 3:  # ALL_LABELS: the shot labels' tokens carry the loss too
            e = tok(pair[0], add_special_tokens=False, return_offsets_mapping=True)
            lab = [t if any(s0 < b and a0 < s1 for s0, s1 in pair[2]) else -100 for t, (a0, b) in zip(e["input_ids"], e["offset_mapping"])]
            keep = MAXLEN - len(a)
            return e["input_ids"][-keep:] + a, lab[-keep:] + a, len(a)
        p = tok(pair[0], add_special_tokens=False)["input_ids"][-(MAXLEN - len(a)):]
        return p + a, [-100] * len(p) + a, len(a)  # third: the answer's token count (it ends the sequence), for ANS_WEIGHT
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / STEPS))
    model.train(); t0 = time.time(); n_sft = n_db = 0; losses = []; n_tok = n_lab_tok = n_pad = 0
    print(f"   {len(ex)} SFT examples, {len(kt)} DB texts, {STEPS} steps x {EFF_BATCH} sequences, lr {LR}", flush=True)
    for step in range(STEPS):
        loss_acc = 0.0
        for _ in range(ACCUM):
            batch = []
            for _ in range(MICRO):
                if kt and rng.random() < DB_FRAC:
                    batch.append(enc(rng.choice(kt))); n_db += 1
                else:
                    batch.append(enc(rng.choice(ex))); n_sft += 1
            L = max(len(i) for i, *_ in batch)
            if QWEN35:  # row 78: lengths rounded up to 64 so fla's gated-delta kernels, tuned per shape, see few shapes (4x on a 200-step run)
                L = -(-L // 64) * 64
            n_tok += sum(len(i) for i, *_ in batch); n_pad += sum(L - len(i) for i, *_ in batch); n_lab_tok += sum(sum(x != -100 for x in lb[1:]) for _, lb, _ in batch)
            ids = torch.tensor([i + [pad] * (L - len(i)) for i, *_ in batch], device="cuda")
            lab = torch.tensor([l + [-100] * (L - len(l)) for _, l, _ in batch], device="cuda")
            att = (torch.arange(L, device="cuda")[None] < torch.tensor([len(i) for i, *_ in batch], device="cuda")[:, None]).long()
            logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
            tgt = lab[:, 1:]; n_lab = max(int((tgt != -100).sum()), 1)
            if ANS_WEIGHT:  # per sequence: w x the answer's mean token loss + (1 - w) x the shot labels' mean token loss
                loss = 0.0
                for i, (seq, _, nf) in enumerate(batch):
                    ce = torch.nn.functional.cross_entropy(logits[i].float(), tgt[i], ignore_index=-100, reduction="none")
                    valid = tgt[i] != -100; fin = torch.zeros_like(valid); fin[len(seq) - nf - 1:len(seq) - 1] = True
                    shot = valid & ~fin
                    li = ce[fin & valid].mean()
                    loss = loss + (ANS_WEIGHT * li + (1 - ANS_WEIGHT) * ce[shot].mean() if shot.any() else li)
                loss = loss / len(batch) / ACCUM
            else:
                loss = sum(torch.nn.functional.cross_entropy(logits[i].float(), tgt[i], ignore_index=-100, reduction="sum") for i in range(len(batch))) / n_lab / ACCUM
            loss.backward(); loss_acc += loss.item(); del logits
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        losses.append(loss_acc)
        if (step + 1) % 25 == 0 or step + 1 == STEPS:
            print(f"    step {step + 1}/{STEPS} loss {loss_acc:.3f} {time.time() - t0:.0f}s {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            run.log(dict(train_loss=loss_acc), condition="train", step=step + 1)
    model.save_pretrained(OUT_DIR); tok.save_pretrained(OUT_DIR)
    secs = time.time() - t0
    return dict(train_minutes=round(secs / 60, 1), tokens=n_tok, pad_tokens=n_pad, labelled_tokens=n_lab_tok, tok_per_s=round(n_tok / secs), labelled_per_step=round(n_lab_tok / STEPS, 1),
                load_in_4bit=LOAD_4BIT, all_labels=ALL_LABELS, n_sft_examples=len(ex), n_db_texts=len(kt), seqs_sft=n_sft, seqs_db=n_db, final_loss=round(sum(losses[-10:]) / len(losses[-10:]), 3),
                peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2), peak_reserved_GiB=round(torch.cuda.max_memory_reserved() / 2**30, 2), trainer=TRAINER,
                n_trainable=sum(p.numel() for p in params), losses=[round(x, 4) for x in losses], adapter=str(OUT_DIR.relative_to(ROOT)))


def train_encoder(run):
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(ENC_BASE, device="cuda")
    pairs = []
    for u in training_users():
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


cfg = dict(route=ROUTE, db=DB, steps=STEPS, lr=LR, epochs=EPOCHS, seed=SEED, db_frac=DB_FRAC, run_tag=RUN_TAG, real6_db=REAL6_DB, trainer=TRAINER, chat=CHAT, db_sha=DOC.get("db_sha256"), base=LLM_BASE if ROUTE == "llm" else ENC_BASE, real6_sha=DOC["sha256"], poi=POI, lora_r=64, n_db_only_merchants=len(DB_ONLY), fold=FOLD, rename=RENAME, n_train_users=len(training_users()), all_labels=ALL_LABELS, ans_weight=ANS_WEIGHT, load_in_4bit=LOAD_4BIT, dbep=DBEP, db_cat=DB_CAT, rec_cat=REC_CAT, micro=MICRO, eff_batch=EFF_BATCH, db_extra=DB_EXTRA, db_episodes=DB_EPISODES, poi_shots=POI_SHOTS, poi_rec=POI_REC, poi_kind=POI_KIND, poi_desc=POI_DESC, poi_unseen=POI_UNSEEN, decoy=DECOY, empty=EMPTY, poi_db=POI_DB, mislead=MISLEAD, lookup=LOOKUP, override=OVERRIDE, alt=ALT, shot_noise=SHOT_NOISE)
with Run("categoriser", model=cfg["base"], config=cfg, enabled=not SMOKE) as run:
    stats = train_llm(run) if ROUTE == "llm" else train_encoder(run)
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(dict(config=cfg, **stats), indent=2)); run.artifact(OUT)
    run.log({k: v for k, v in stats.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}, condition="trained")
stats.pop("losses", None)
print(f"=== categoriser {ROUTE} {DB}: {stats}")
