"""Published decision models (the families Ollaya serves, https://ollaya.dev) on this repo's categoriser item sets (PLAN step 77,
MODEL-14): each item's prompt (the user's categories, the 24 labelled examples, the query) becomes the model's state, the question is
QUESTION, and the options are the user's category names; every model is read in its own published layout and readout.

  FAMILY=decider   Mapika/decider-{0.8b,2b,4b}: full fine-tunes of Qwen3.5-Base; "Context:\\n<state>\\n\\nQuestion: q\\nOptions:\\n(A) o ...
                   \\nAnswer: (" and the option-label logits at the final "(" (the model repo's own decider/ package: prompt.build,
                   model.py's readout; > 10 options use its one-token wide labels). Plain layout; the chat layout if the model's
                   decider_config.json says so (not handled: the job stops).
  FAMILY=decision  llm-semantic-router/Decision-1.0-{Eos-0.8B,Sol-2B,Nox-4B}: full fine-tunes of Qwen3.5 plus a candidate head read at
                   each option's last token against the final token (the Sol repo's code/decision_model.py).
  FAMILY=kev       jaredpalmer/kev-{0.8b,4b,9b}: a LoRA on Qwen3.5-Base plus a pointer head (<opt> o </opt> spans against <decide>);
                   kev's package at KEV_SHA, downloaded as an archive; state up to 8,192 tokens (its serving limit).
  FAMILY=von       wfzyx/von (1.2): ModernBERT-large with a [MASK] per option, independent-option attention (von-sdk).
  FAMILY=strands   StrandsAgents/strands-decider-2B-hobson-v19 (row 151): a LoRA on Qwen3.5-2B-Base plus a pointer head, in strands'
                   own prompt (ai_experiments.strands; needs --with strands-decider==0.1.0); ADAPTER = a fine-tune from exp_strands_finetune.py.

env: FAMILY, MODEL (HF id), ITEMS_SET (a frozen set in REAL-6's format under data/processed, e.g. poi1_v1_kinds; empty = REAL-6 v1),
     CONDS (noctx / ctx: the item's prompt or prompt_ctx), USERS (comma list; default the set's fold 0: user % 4 == 0), SMOKE=1 (8 items),
     TEMP (1.0: raw logits; the scorecard fits its own temperature), BATCH.
Licences (owner, 2026-09-26: open licences only; checked 2026-09-27): every model here and its base is Apache-2.0 on its model card
(decider, Decision-1.0 and its Qwen3.5 bases, kev and its Qwen3.5-Base bases, Von and ModernBERT-large); the code is Apache-2.0 (the
decider/ package in the model repos, Decision's code/ at DECISION_CODE_REV, kev at KEV_SHA, von-sdk) or MIT (flash-linear-attention).
open_licence() refuses a model whose card, or whose base model's card, names any other licence.
Writes results/per_item/real6_dm_<family>_<model>_<set>.<cond>.jsonl in the Scorer's record shape (sum_lp = the model's log-probability
per option in the item's option order, n_tok = 1), so the REAL-6 / POI-1 table code reads it unchanged. Tracker experiment
"decision_models".
usage: FAMILY=decider MODEL=Mapika/decider-2b ITEMS_SET=poi1_v1 uv run --with transformers==5.17.0 --with flash-linear-attention --with "peft>=0.21" --with torch==2.11.0 --with torchvision==0.26.0 python scripts/exp_decision_models.py
"""
import json
import os
import random
import sys
import time
from pathlib import Path

from ai_experiments import real6 as R6
from ai_experiments.evals.tracker import Run
from ai_experiments.licences import open_licence
from ai_experiments.paths import PROCESSED, ROOT
from ai_experiments.real6_eval import write_recs

FAMILY, MODEL = os.environ["FAMILY"], os.environ["MODEL"]
ITEMS_SET = os.environ.get("ITEMS_SET", "")
LAYOUT = os.environ.get("LAYOUT", "")  # row 111: oneslot.build_layout (options | labelled | labelled_shots); read with the layout the adapter was trained on
DOW_FIRST = os.environ.get("DOW_FIRST", "") == "1"
DESC = os.environ.get("DESC", "") == "1"  # row 136: categories described by their payees in the prompt
CONDS = os.environ.get("CONDS", "noctx").split(",")
SMOKE = bool(os.environ.get("SMOKE"))
TEMP = float(os.environ.get("TEMP", "1.0"))
BATCH = int(os.environ.get("BATCH", "8"))
DECISION_CODE_REV = "60ea30a48285ea097a9b3a728e71649b78331601"  # the last Sol-2B revision that ships code/decision_model.py (same prompt_version as the weights)
KEV_SHA = os.environ.get("KEV_SHA", "5920c5f")
ADAPTER = os.environ.get("ADAPTER", "")
ORDER_SEED = os.environ.get("ORDER_SEED", "")  # row 50: every item's options shuffled by this seed before scoring (and decider's own label order too), scores mapped back  # FAMILY=decider: a fine-tuned LoRA under models/adapters (exp_decider_finetune.py)
EXTRA_OPTS = [o for o in os.environ.get("EXTRA_OPTS", "").split("|") if o]  # row 84: options appended to every question (abstain options offered at inference); their scores follow the real options'
LABELS = os.environ.get("LABELS", "letters")  # FAMILY=decider: option labels (ai_experiments.oneslot): letters | rand26 | rand255
QUESTION = "Which of this user's categories does the last transaction belong to?"

DOC = json.loads((PROCESSED / f"{ITEMS_SET}.json").read_text()) if ITEMS_SET else R6.load("v1")
USERS = os.environ.get("USERS") or ",".join(str(u) for u in sorted({it["user"] for it in DOC["items"]}) if u % 4 == 0)
ITEMS = [it for it in DOC["items"] if str(it["user"]) in USERS.split(",")]
if SMOKE:
    ITEMS = ITEMS[:8]
TAG = f"dm_{FAMILY}_{ADAPTER or MODEL.split('/')[-1]}_{ITEMS_SET or 'real6'}{'_ord' + ORDER_SEED if ORDER_SEED else ''}{'' if LABELS == 'letters' else '_lab' + LABELS}{'_lay' + LAYOUT if LAYOUT else ''}{'_dow' if DOW_FIRST else ''}{'_desc' if DESC else ''}{'_xo' + str(len(EXTRA_OPTS)) if EXTRA_OPTS else ''}"


def state_of(it, cond):
    """The item's prompt without the final "Category:" cue: categories, examples, (note,) the query transaction."""
    p = it["prompt_ctx" if cond == "ctx" else "prompt"]
    assert p.endswith("Category:"), it["id"]
    return p[: -len("Category:")].rstrip()


def options_of(it):
    return [o.strip() for o in it["options"]] + [o for o in EXTRA_OPTS if o not in (x.strip() for x in it["options"])]


# --- the four families: each returns score(items, cond) -> list of per-option log-prob lists --------------------------------

def decider():
    import importlib
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL)
    cfg = json.loads(Path(path, "decider_config.json").read_text()) if Path(path, "decider_config.json").exists() else {}
    assert cfg.get("layout", "plain") in ("plain", "state_first") and not cfg.get("chat_template"), f"chat layout not handled: {cfg}"
    sys.path.insert(0, snapshot_download("Mapika/decider-2b", allow_patterns=["decider/*"]))  # decider's prompt code; row 79 reads plain Qwen3.5 with it
    P = importlib.import_module("decider.prompt")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    full = ADAPTER and (ROOT / "models" / "adapters" / ADAPTER / "config.json").exists() and not (ROOT / "models" / "adapters" / ADAPTER / "adapter_config.json").exists()
    if full:  # row 156: a fully fine-tuned model directory from exp_decider_finetune.py FULL_FT=1
        path = str(ROOT / "models" / "adapters" / ADAPTER)
    lm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    if ADAPTER and not full:  # a LoRA from exp_decider_finetune.py, merged for scoring
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, str(ROOT / "models" / "adapters" / ADAPTER)).merge_and_unload().eval()
    from ai_experiments import oneslot
    adir = ROOT / "models" / "adapters" / ADAPTER if ADAPTER else None
    extra = json.loads((adir / "oneslot_extra.json").read_text()) if adir and (adir / "oneslot_extra.json").exists() else {}  # row 152
    ptr = None
    if extra.get("pointer"):
        from ai_experiments.pointer import FILE, OptionPointer
        ptr = OptionPointer(lm.config.get_text_config().hidden_size).cuda().eval()
        ptr.load_state_dict(torch.load(adir / FILE, map_location="cuda", weights_only=True))

    @torch.no_grad()
    def score(items, cond):
        out = []
        for k in range(0, len(items), BATCH):
            chunk = items[k:k + BATCH]
            built = [oneslot.build_layout(P, tok, state_of(it, cond), it.get("question", QUESTION), options_of(it), it["answer"],
                                          random.Random(it["id"] + (f"-{ORDER_SEED}" if ORDER_SEED else "")), labels=LABELS, layout=LAYOUT or "options", dow=DOW_FIRST, desc=DESC,
                                          relist=bool(extra.get("relist")))
                     if LAYOUT or DOW_FIRST else
                     oneslot.build(P, tok, state_of(it, cond), it.get("question", QUESTION), options_of(it), it["answer"],
                                   random.Random(it["id"] + (f"-{ORDER_SEED}" if ORDER_SEED else "")), labels=LABELS) for it in chunk]
            T = -(-max(len(b["ids"]) for b in built) // 64) * 64  # as decider's own collate: few shapes for fla's per-shape tuning
            ids = torch.full((len(built), T), tok.pad_token_id or 0, dtype=torch.long)
            att = torch.zeros_like(ids)
            for i, b in enumerate(built):
                ids[i, :len(b["ids"])] = torch.tensor(b["ids"]); att[i, :len(b["ids"])] = 1
            h = lm.model(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state
            for i, b in enumerate(built):
                n, perm = len(b["labs"]), b["perm"]
                z = F.linear(h[i, b["slot"]], lm.lm_head.weight[torch.tensor(b["labs"], device="cuda")]).float()
                if ptr is not None:  # row 152: + the pointer over the relisted option lines
                    z = z + ptr(h[i, b["slot"]], h[i, torch.tensor(b["opt_pos"], device="cuda")])
                z = z / TEMP
                lp = F.log_softmax(z, -1).tolist()
                back = [0.0] * n
                for j, oi in enumerate(perm):  # label j shows option perm[j]
                    back[oi] = lp[j]
                out.append(back)
        return out
    return score


def decision():
    import importlib
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    code = snapshot_download("llm-semantic-router/Decision-1.0-Sol-2B", revision=DECISION_CODE_REV, allow_patterns=["code/*"])  # removed from the repos on 2026-09-27
    sys.path.insert(0, str(Path(code, "code")))
    D = importlib.import_module("decision_model")
    path = snapshot_download(MODEL)
    model, tok = D.DecisionModel.from_checkpoint(path)
    model = model.cuda().eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    @torch.no_grad()
    def score(items, cond):
        out = []
        for k in range(0, len(items), BATCH):
            chunk = items[k:k + BATCH]
            rows = [dict(id=it["id"], state=state_of(it, cond), instructions=it.get("question", QUESTION), task_type="choice",
                         options=[{"key": o, "description": None} for o in options_of(it)], label=it["answer"]) for it in chunk]
            enc = [D.encode(r, tok, 16384) for r in rows]
            batch = {key: v.cuda() if torch.is_tensor(v) else v for key, v in D.collate(enc, pad).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(**batch)
            for i, e in enumerate(enc):
                out.append(F.log_softmax(z[i, :e["nopts"]].float() / TEMP, -1).tolist())
        return out
    return score


def kev():
    import io
    import tarfile
    import urllib.request
    import torch
    dst = Path("/tmp") / f"kev-{KEV_SHA}"
    if not dst.exists():
        data = urllib.request.urlopen(f"https://github.com/jaredpalmer/kev/archive/{KEV_SHA}.tar.gz").read()
        tarfile.open(fileobj=io.BytesIO(data)).extractall("/tmp")
        next(Path("/tmp").glob(f"kev-{KEV_SHA}*")).rename(dst)
    sys.path.insert(0, str(dst))
    from kev.checkpoint import Checkpoint, LoadOptions  # noqa: E402
    ck = Checkpoint(MODEL)
    tok, m = ck.load("cuda", LoadOptions(temperature=1.0))

    @torch.no_grad()
    def score(items, cond):
        out = []
        for it in items:
            rec = {"state": state_of(it, cond), "questions": [{"instr": it.get("question", QUESTION), "options": options_of(it), "label": it["answer"]}]}
            enc = m.encode(tok, rec, max_state=8192, max_branch=8192 + 2048)
            p = m.probs(enc)[0].float().clamp_min(1e-30)
            out.append((p.log() / TEMP).log_softmax(-1).tolist())
        return out
    return score


def von():
    """von-sdk's choice path (OptionMarkerBackend.evaluate_choice) up to its raw option logits: the SDK's own answer rounds the
    probabilities to four decimals after its input-conditioned temperature."""
    import torch
    import torch.nn.functional as F
    from von.backends.option_marker_backend import OptionMarkerBackend
    b = OptionMarkerBackend(device="cuda")
    model = b._get_model(); tok = model.tokenizer

    @torch.no_grad()
    def score(items, cond):
        out = []
        for it in items:
            opts = options_of(it)
            inputs = tok(model.pack_sequence(state_of(it, cond), it.get("question", QUESTION), opts), return_tensors="pt").to("cuda")
            pos = (inputs["input_ids"][0] == model.mask_token_id).nonzero(as_tuple=True)[0].tolist()
            assert len(pos) == len(opts), (it["id"], len(pos), len(opts))
            z = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], mask_positions=[pos],
                      independent_options=b._independent_options)[0]
            out.append(F.log_softmax(z.float() / TEMP, -1).tolist())
        return out
    return score


def strands():
    """Row 151 (MODEL-22): strands-decider's pointer head (ai_experiments.strands); MODEL a Hub id (the published v19) or ADAPTER a
    checkpoint under models/adapters (exp_strands_finetune.py). Raw head (temperature 1), whole prompts, BATCH items per forward."""
    import torch
    from ai_experiments import strands as SD
    m = SD.load(ROOT / "models" / "adapters" / ADAPTER if ADAPTER else MODEL).eval()
    xf = ROOT / "models" / "adapters" / ADAPTER / "strands_extra.json" if ADAPTER else None
    labelled = bool(xf and xf.exists() and json.loads(xf.read_text()).get("state") == "labelled")  # row 152

    def enc(it, cond):
        if labelled:
            st, shown, sp = SD.labelled(state_of(it, cond), options_of(it), random.Random(it["id"]))
            return SD.encode_aux(m.tokenizer, st, it.get("question", QUESTION), shown, sp)
        return SD.encode(m.tokenizer, state_of(it, cond), it.get("question", QUESTION), options_of(it))

    @torch.no_grad()
    def score(items, cond):
        out, longest = [], 0
        for k in range(0, len(items), BATCH):
            encs = [enc(it, cond) for it in items[k:k + BATCH]]
            longest = max([longest] + [len(e[0]) for e in encs])
            lp = SD.log_probs(m, encs).float()
            out += [(lp[i, :len(e[1])] / TEMP).log_softmax(-1).tolist() for i, e in enumerate(encs)]
        print(f"   longest prompt {longest} tokens (v19 trained up to 3,072, served at 4,096)", flush=True)
        return out
    return score


SANITY = [  # SANITY=1: short questions with an obvious answer (the first is Ollaya's own triage example), each also with the options reversed
    ("I was charged twice for my subscription this month and want a refund.", "Which team should handle this?",
     ["billing: Payments, invoices and refunds", "technical: Bugs, errors and outages", "account: Login, profile and settings"], 0),
    ("The app crashes every time I open the settings page since the last update.", "Which team should handle this?",
     ["billing: Payments, invoices and refunds", "technical: Bugs, errors and outages", "account: Login, profile and settings"], 1),
    ("Transaction: SHELL OIL 57442, Austin TX | $48.20 | Mon", "Which category does this transaction belong to?",
     ["Groceries", "Dining out", "Fuel", "Rent", "Entertainment", "Travel"], 2),
    ("Transaction: TRADER JOE'S #552, Portland OR | $83.10 | Sat", "Which category does this transaction belong to?",
     ["Groceries", "Dining out", "Fuel", "Rent", "Entertainment", "Travel"], 0),
]


def sanity_items():
    out = []
    for k, (state, q, opts, gold) in enumerate(SANITY):
        for rev in (False, True):
            o = opts[::-1] if rev else opts
            out.append(dict(id=f"S{k}{'r' if rev else ''}", level="sanity", user=0, question=q, answer=o.index(opts[gold]),
                            options=[" " + x for x in o], prompt=state + "\nCategory:", prompt_ctx=state + "\nCategory:"))
    return out


if __name__ == "__main__":
    open_licence(os.environ.get("LICENCE_OF", MODEL))  # LICENCE_OF: the Hub model a local run (a kev fine-tune, row 80) derives from
    if os.environ.get("SANITY"):
        ITEMS, SMOKE, TAG = sanity_items(), True, f"dm_{FAMILY}_{MODEL.split('/')[-1]}_sanity"
    score = {"decider": decider, "decision": decision, "kev": kev, "von": von, "strands": strands}[FAMILY]()
    cfg = dict(family=FAMILY, model=MODEL, items_set=ITEMS_SET or "real6_v1", items_sha=DOC.get("sha256"), conds=CONDS, n_items=len(ITEMS),
               users=USERS, question=QUESTION, temp=TEMP, adapter=ADAPTER, order_seed=ORDER_SEED, kev_sha=KEV_SHA if FAMILY == "kev" else None)
    with Run("decision_models", model=MODEL, config=cfg, enabled=not SMOKE) as run:
        for cond in CONDS:
            t0 = time.time()
            if ORDER_SEED:  # the options (and the category list in the prompt is unchanged: only the option order the reader sees moves)
                perms = [random.Random(f"{it['id']}-order-{ORDER_SEED}").sample(range(len(it["options"])), len(it["options"])) for it in ITEMS]
                shuffled = [dict(it, options=[it["options"][k] for k in pm], answer=pm.index(it["answer"])) for it, pm in zip(ITEMS, perms)]
                raw = score(shuffled, cond)
                lps = [[lp[pm.index(k)] for k in range(len(pm))] for lp, pm in zip(raw, perms)]
            else:
                lps = score(ITEMS, cond)
            recs = []
            for it, lp in zip(ITEMS, lps):
                pred = max(range(len(lp)), key=lp.__getitem__)
                recs.append(dict(id=it["id"], level=it.get("level"), answer=it["answer"], sum_lp=lp, n_tok=[1] * len(lp), pred=pred,
                                 correct=pred == it["answer"]))
            path = write_recs(TAG, cond, recs, smoke=SMOKE)
            acc = 100 * sum(r["correct"] for r in recs) / len(recs)
            print(f"{TAG} {cond}: top-1 {acc:.1f} on {len(recs)} items in {(time.time() - t0) / 60:.1f} min -> {path}", flush=True)
            run.log({f"{cond}_top1": acc, f"{cond}_minutes": (time.time() - t0) / 60})
