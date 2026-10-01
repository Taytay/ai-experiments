"""Fine-tune a decider model (Mapika/decider-2b / -4b, Apache-2.0; PLAN step 77, MODEL-14) on the categoriser's own training episodes,
in decider's layout and readout: the episode's prompt without its "Category:" cue is the Context, the user's category names (after rename
augmentation) are the options, and the loss is cross-entropy over the option-label logits at the "Answer: (" slot (decider/train.py's
objective), options shuffled per episode by decider's own prompt.build. A rank-64 LoRA on every projection, the Gated DeltaNet ones
included, as row 78's Qwen3.5 categoriser; the same steps, batch and learning-rate schedule.

The episodes are exp_categoriser.py's sft_examples(), built under the same env (POI, FOLD, RENAME, POI_KIND, POI_DESC, POI_UNSEEN,
DBEP, ...), so the training users, layouts and augmentations match the categoriser arm it is compared with. ALL_LABELS has no analogue
here (one decision per episode).

env: MODEL (default Mapika/decider-2b), STEPS (800), MICRO (16 sequences per step), LR (1e-4), RUN_TAG, SEED, plus the episode knobs.
Writes models/adapters/decider_<model>_<episode suffix>_lora (peft) and results/decider_ft_<same>.json; score it with
exp_decision_models.py FAMILY=decider ADAPTER=<that dir name>. Tracker experiment "decider_finetune".
Row 79 (MODEL-16) uses it to train plain Qwen3.5 with one answer slot (MODEL=Qwen/Qwen3.5-2B; adapter slot_<model>_...).
usage: POI=poi1_v1 FOLD=0 RENAME=0.5 POI_KIND=1 POI_DESC=1 POI_UNSEEN=0.5 uv run --with transformers==5.17.0 --with flash-linear-attention
       --with "peft>=0.21" --with torch==2.13.0 --with torchvision==0.28.0 python scripts/exp_decider_finetune.py
"""
import importlib
import json
import os
import random
import sys
import time

from ai_experiments.evals.tracker import Run
from ai_experiments.paths import ROOT

MODEL = os.environ.get("MODEL", "Mapika/decider-2b")  # row 79: a plain Qwen3.5 (Qwen/Qwen3.5-2B) trained in decider's layout and readout
DECIDER_CODE = "Mapika/decider-2b"  # the repo whose decider/ package (prompt.build, the letter table) renders the episodes
STEPS, MICRO, LR = int(os.environ.get("STEPS", "800")), int(os.environ.get("MICRO", "16")), float(os.environ.get("LR", "1e-4"))
SEED = int(os.environ.get("SEED", "0"))
AUX_LM = float(os.environ.get("AUX_LM", "0"))  # row 79: + w x the token loss on the shot labels inside the Context (the all-label loss, same forward); needs ALL_LABELS=1
LAYOUT = os.environ.get("LAYOUT", "")  # row 111 (owner, 2026-09-29): "" (decider's layout), options | labelled | labelled_shots (oneslot.build_layout)
DOW_FIRST = os.environ.get("DOW_FIRST", "") == "1"  # row 111: the weekday next to the date (oneslot.build_layout)
EMA = float(os.environ.get("EMA", "0"))  # row 122 (TRAIN-13): an exponential moving average of the LoRA weights (this decay per step) is what gets saved
TEACHER = os.environ.get("TEACHER", "")  # row 123 (MODEL-19): per-item log-probs of a teacher on the pool's first-time-payee episodes (dump_teacher_items.py + exp_decision_models.py)
TEACH_W = float(os.environ.get("TEACH_W", "0.5"))  # the target on those episodes: (1 - w) one-hot + w teacher distribution
SPLIT = os.environ.get("SPLIT", "") == "1"
RELIST = os.environ.get("RELIST", "") == "1"  # row 152: the options listed again after the query (oneslot.build_layout relist)
POINTER = os.environ.get("POINTER", "") == "1"  # row 152 (MODEL-23): + a strands-style pointer over the relisted lines (ai_experiments.pointer); needs RELIST
POINTER_LR = float(os.environ.get("POINTER_LR", "1e-3"))
DESC = os.environ.get("DESC", "") == "1"  # row 136: each category in the list described by the payees filed under it in the prompt (oneslot.describe_categories)  # row 118: episodes in the split layout (shared rows, then the query payee's rows; oneslot.split_rows)
LABELS = os.environ.get("LABELS", "letters")  # owner 2026-09-27: option labels (ai_experiments.oneslot): letters | rand26 | rand255
ABSTAIN = float(os.environ.get("ABSTAIN", "0"))  # row 52: decider's augmentation; this share of episodes gets a last option ABSTAIN_OPT
ABSTAIN_SWAP = float(os.environ.get("ABSTAIN_SWAP", "0.25"))  # ... and in this share of those the gold category is hidden (removed from the header
ABSTAIN_OPT = "not listed here"  # and the options; shots keep its label), so the abstain option is the answer. decider's neutral wording
EVFREE = float(os.environ.get("EVFREE", "0"))  # row 84: share of episodes made evidence-free (the query's merchant a fresh opaque name in no DB and no history)
EVFREE_MODE = os.environ.get("EVFREE_MODE", "opt")  # opt: the answer is a last option CANT_TELL (shown on as many normal episodes too); soft: a uniform target over the options
CANT_TELL = "cannot tell from this"  # decider's abstain wording family ("cannot tell" prefix)
QUESTION = "Which of this user's categories does the last transaction belong to?"  # as exp_decision_models.py

# exp_categoriser.py's episode builder under the same env (its module level up to the model code: config, data, sft_examples)
os.environ.setdefault("LLM_BASE", "Qwen/Qwen3.5-2B")  # only for its name bookkeeping; no model is loaded from it here
sys.argv = [sys.argv[0], "llm", "none"]
_src = (ROOT / "scripts" / "exp_categoriser.py").read_text().split("\nTARGETS = [")[0]
C = {"__name__": "exp_categoriser_episodes", "__file__": str(ROOT / "scripts" / "exp_categoriser.py")}
exec(compile(_src, "exp_categoriser.py", "exec"), C)
SFX = C["SFX"].replace("_alllab", "").replace("_hf", "") + (f"_aux{round(AUX_LM * 100)}" if AUX_LM else "")
SFX += "" if LABELS == "letters" else f"_lab{LABELS}"
SFX += (f"_lay{LAYOUT}" if LAYOUT else "") + ("_dow" if DOW_FIRST else "") + ("_split" if SPLIT else "") + ("_desc" if DESC else "")
SFX += f"_ema{str(EMA).split('.')[-1]}" if EMA else ""
SFX += f"_ev{round(EVFREE * 100)}{EVFREE_MODE}" if EVFREE else ""
SFX += ("_relist" if RELIST else "") + ("_ptr" if POINTER else "")
assert not POINTER or RELIST, "POINTER needs RELIST=1"
SFX += f"_abst{round(ABSTAIN * 100)}sw{round(ABSTAIN_SWAP * 100)}" if ABSTAIN else ""
SFX_POOL = SFX  # the pool's name without the teacher suffix (dump_teacher_items.py names its file by it)
SFX += f"_teach{round(TEACH_W * 100)}" if TEACHER else ""
assert not AUX_LM or C["ALL_LABELS"], "AUX_LM needs ALL_LABELS=1 (the shot-label spans)"
NAME = f"{'decider_' if 'decider' in MODEL else 'slot_'}{MODEL.split('/')[-1]}_{SFX}_lora"  # slot_: row 79's one-slot Qwen3.5
OUT_DIR = ROOT / "models" / "adapters" / NAME


def names_of(prompt):
    """The category names from the episode's header (plain "Categories: a, b" or the row 73 description list "- a: kinds")."""
    head = prompt.split("\n\n", 1)[0]
    if head.startswith("Categories:\n"):
        return [ln[2:].split(": ", 1)[0] for ln in head.split("\n")[1:]]
    return head[len("Categories: "):].split(", ")


def abstain_aug(e, rng):
    """Row 52: with probability ABSTAIN add the abstain option last; with ABSTAIN_SWAP of those hide the gold (header and options), the
    abstain option then being the answer. The shot-label spans (AUX_LM) sit after the header and move by the header's change in length."""
    if not ABSTAIN or rng.random() >= ABSTAIN:
        return e
    ctx, opts, gold = e[0], list(e[1]), e[2]
    if rng.random() < ABSTAIN_SWAP and len(opts) > 2:
        head, rest = ctx.split("\n\n", 1)
        assert head.startswith("Categories: "), "ABSTAIN_SWAP needs the plain header"
        g = opts.pop(gold)
        new_head = "Categories: " + ", ".join(n for n in head[len("Categories: "):].split(", ") if n != g)
        d = len(new_head) - len(head)
        ctx = new_head + "\n\n" + rest
        e = (ctx, opts + [ABSTAIN_OPT], len(opts)) + (([(a + d, b + d) for a, b in e[3]],) if len(e) > 3 else ())
        return e
    return (ctx, opts + [ABSTAIN_OPT], gold) + tuple(e[3:])


def evfree_aug(e, rng):
    """Row 84. With probability EVFREE the query transaction (the Context's last line) becomes a row of a fresh opaque merchant (never in
    the DB, a history or REAL-6), amount from a random category; target: CANT_TELL added last (opt) or None, read as a uniform target
    (soft). In opt mode CANT_TELL is also added to as many normal episodes, gold unchanged, so its presence carries no signal."""
    if not EVFREE:
        return e
    ctx, opts, gold = e[0], list(e[1]), e[2]
    if rng.random() < EVFREE:
        from ai_experiments import merchants as M
        from ai_experiments import transactions as T
        import math
        while True:
            name = rng.choice(M._PREFIX) + rng.choice(M._MID) + rng.choice(M._SUFFIX) + rng.choice(M._TAG)
            if name not in C["MERCHANT"]:
                break
        cat = rng.choice(M.CATEGORY_LIST); mu, sig = T.AMOUNT[cat]
        m = dict(name=name, category=cat, city=rng.choice(M._CITIES))
        head, last = ctx.rsplit("\n", 1)
        kline = ""
        if last.startswith("Kind: "):  # row 149: the query's kind line goes with it; an unknown merchant has an unknown kind
            head, last = head.rsplit("\n", 1); kline = "\nKind: unknown"
        assert last.startswith("Transaction: "), last[:40]
        ctx = head + f"\nTransaction: {T.render(m, rng)} | ${math.exp(rng.gauss(mu, sig)):.2f} | {rng.choice(T.WEEKDAYS)}" + kline
        if EVFREE_MODE == "opt":
            return (ctx, opts + [CANT_TELL], len(opts)) + tuple(e[3:])
        return (ctx, opts, None) + tuple(e[3:])
    if EVFREE_MODE == "opt" and rng.random() < EVFREE / (1 - EVFREE):
        return (ctx, opts + [CANT_TELL], gold) + tuple(e[3:])
    return e


SOFT_CTX = {}  # row 86 v2: episode context -> {option name: probability}


def episodes():
    out = []
    for e in C["sft_examples"]():
        prompt, ans = e[0], e[1].strip()
        assert prompt.endswith("Category:")
        opts = names_of(prompt)
        out.append((prompt[: -len("Category:")].rstrip(), opts, opts.index(ans)) + ((e[2],) if AUX_LM else ()))
        if prompt in C["SOFT"]:  # row 86 v2: the target's true split
            SOFT_CTX[out[-1][0]] = C["SOFT"][prompt]
    return out


TEACH = {}  # row 123: episode context -> teacher probabilities in the episode's option order


def load_teacher(eps):
    import math
    import glob as _g
    items = json.loads((ROOT / "data" / "processed" / f"teacher_eps_{SFX_POOL}.json").read_text())["items"]
    by_id = {it["id"]: it for it in items}
    path = TEACHER if os.path.exists(TEACHER) else sorted(_g.glob(str(ROOT / "results" / "per_item" / TEACHER)))[0]
    for line in open(path):
        r = json.loads(line); it = by_id.get(r["id"])
        if it is None:
            continue
        lp = r["sum_lp"]; m = max(lp); z = [math.exp(x - m) for x in lp]; s = sum(z)
        TEACH[it["prompt"][: -len("\nCategory:")]] = [x / s for x in z]
    print(f"   teacher distributions for {len(TEACH)} episodes", flush=True)


def main():
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = snapshot_download(MODEL)
    sys.path.insert(0, snapshot_download(DECIDER_CODE, allow_patterns=["decider/*"]))  # decider's prompt code; the weights may be any Qwen3.5 (row 79)
    P = importlib.import_module("decider.prompt")
    tok = AutoTokenizer.from_pretrained(path)
    lm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda()
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); lm.enable_input_require_grads()
    torch.manual_seed(SEED)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "out_proj",
               "qkv_proj", "gate_up_proj", "input_linear", "output_linear"]  # row 99b: Phi-3 fused and Granite 4 names
    model = get_peft_model(lm, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=targets))
    head_w = lm.get_output_embeddings().weight  # decider ties or not; its readout is these rows (decider/model.py)

    from ai_experiments import oneslot
    eps = episodes(); rng = random.Random(SEED)
    if TEACHER:
        load_teacher(eps)
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": params, "lr": LR}]
    ptr = None
    if POINTER:  # row 152
        from ai_experiments.pointer import OptionPointer
        ptr = OptionPointer(lm.config.get_text_config().hidden_size).cuda()
        groups.append({"params": list(ptr.parameters()), "lr": POINTER_LR}); params = params + list(ptr.parameters())
    opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / STEPS))
    pad = tok.pad_token_id or 0
    ema = [p.detach().clone().float() for p in params] if EMA else None
    model.train(); t0 = time.time(); losses = []; n_tok = 0
    print(f"   {len(eps)} episodes, {STEPS} steps x {MICRO}, lr {LR}, {sum(p.numel() for p in params)} trainable", flush=True)
    for step in range(STEPS):
        picked = [evfree_aug(abstain_aug(rng.choice(eps), rng), rng) for _ in range(MICRO)]
        if LAYOUT or DOW_FIRST:  # row 111
            built = [oneslot.build_layout(P, tok, e[0], QUESTION, e[1], e[2], rng, labels=LABELS, layout=LAYOUT or "options", dow=DOW_FIRST,
                                          spans=e[3] if AUX_LM else None, split=SPLIT, desc=DESC, relist=RELIST) for e in picked]
        else:
            built = [oneslot.build(P, tok, e[0], QUESTION, e[1], e[2], rng, labels=LABELS) for e in picked]
        T = -(-max(len(b["ids"]) for b in built) // 64) * 64; n_tok += sum(len(b["ids"]) for b in built)
        ids = torch.full((len(built), T), pad, dtype=torch.long)
        att = torch.zeros_like(ids)
        for i, b in enumerate(built):
            ids[i, :len(b["ids"])] = torch.tensor(b["ids"]); att[i, :len(b["ids"])] = 1
        h = model.base_model.model.model(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state
        N = max(len(b["labs"]) for b in built)  # each question's own label tokens; padded options at -inf
        z = torch.stack([F.pad(F.linear(h[i, b["slot"]], head_w[torch.tensor(b["labs"], device="cuda")]).float()
                               + (ptr(h[i, b["slot"]], h[i, torch.tensor(b["opt_pos"], device="cuda")]) if ptr is not None else 0.0),
                               (0, N - len(b["labs"])), value=float("-inf"))
                         for i, b in enumerate(built)])
        soft = torch.tensor([b["gold"] < 0 for b in built], device="cuda")  # row 84 soft: evidence-free, a uniform target over the options
        gold = torch.tensor([max(b["gold"], 0) for b in built], device="cuda")
        lz = F.log_softmax(z, -1).masked_fill(torch.isinf(z), 0.0)
        per = torch.where(soft, -lz.sum(-1) / torch.tensor([len(b["labs"]) for b in built], device="cuda"), -lz.gather(1, gold[:, None])[:, 0])
        for i, (e, b) in enumerate(zip(picked, built)):  # row 123: the teacher's distribution mixed into the target
            td = TEACH.get(e[0])
            if td and b["gold"] >= 0 and e[0] not in SOFT_CTX and len(td) == len(e[1]):
                tv = [(1 - TEACH_W) * (oi == e[2]) + TEACH_W * td[oi] for oi in b["perm"]]
                tv = torch.tensor(tv + [0.0] * (lz.shape[1] - len(b["perm"])), device="cuda")
                per = per.clone(); per[i] = -(tv * lz[i]).sum()
        for i, (e, b) in enumerate(zip(picked, built)):  # row 86 v2: a split target where the user's choice is noisy or random
            dist = SOFT_CTX.get(e[0])
            if dist and b["gold"] >= 0:
                tv = torch.tensor([dist.get(e[1][oi], 0.0) for oi in b["perm"]] + [0.0] * (lz.shape[1] - len(b["perm"])), device="cuda")
                per = per.clone(); per[i] = -(tv * lz[i]).sum()
        loss = per.mean()
        if AUX_LM:  # the shot labels' tokens in "Context:\n<state>" (the same ids build() starts with), each predicted from the position before
            rows, pos, tgt = [], [], []
            for i, e in enumerate(picked):
                if "aux" in built[i]:  # row 111: the layout builder located the shot-label tokens itself
                    for t, y in built[i]["aux"]:
                        rows.append(i); pos.append(t); tgt.append(y)
                    continue
                enc = tok("Context:\n" + e[0], add_special_tokens=False, return_offsets_mapping=True)
                n = len(built[i]["ids"])
                for t, (a0, b0) in enumerate(enc["offset_mapping"]):
                    if 0 < t < n and any(s0 + 9 < b0 and a0 < s1 + 9 for s0, s1 in e[3]):  # 9 = len("Context:\n")
                        rows.append(i); pos.append(t - 1); tgt.append(enc["input_ids"][t])
            if tgt:
                lg = F.linear(h[torch.tensor(rows, device="cuda"), torch.tensor(pos, device="cuda")], head_w).float()
                loss = loss + AUX_LM * F.cross_entropy(lg, torch.tensor(tgt, device="cuda"))
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if EMA:
            with torch.no_grad():
                for e_, p_ in zip(ema, params):
                    e_.mul_(EMA).add_(p_.detach().float(), alpha=1 - EMA)
        losses.append(loss.item())
        if step % 50 == 0 or step == STEPS - 1:
            print(f"   step {step} loss {sum(losses[-50:]) / len(losses[-50:]):.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    if EMA:
        with torch.no_grad():
            for e_, p_ in zip(ema, params):
                p_.copy_(e_.to(p_.dtype))
    OUT_DIR.mkdir(parents=True, exist_ok=True); model.save_pretrained(OUT_DIR)
    if RELIST or POINTER:  # row 152: how to read this adapter (exp_decision_models picks it up)
        (OUT_DIR / "oneslot_extra.json").write_text(json.dumps(dict(relist=RELIST, pointer=POINTER)))
    if ptr is not None:
        from ai_experiments.pointer import FILE
        torch.save(ptr.state_dict(), OUT_DIR / FILE)
        print(f"   pointer gate {ptr.gate.item():.3f}", flush=True)
    return dict(train_minutes=round((time.time() - t0) / 60, 1), tokens=n_tok, final_loss=round(sum(losses[-50:]) / len(losses[-50:]), 3),
                peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), adapter=str(OUT_DIR.relative_to(ROOT)), n_episodes=len(eps))


if __name__ == "__main__":
    from ai_experiments.licences import open_licence
    open_licence(MODEL)
    cfg = dict(model=MODEL, steps=STEPS, micro=MICRO, lr=LR, seed=SEED, episodes_sfx=SFX, poi=C["POI"], fold=C["FOLD"], rename=C["RENAME"],
               poi_kind=C["POI_KIND"], poi_desc=C["POI_DESC"], poi_unseen=C["POI_UNSEEN"], dbep=C["DBEP"], lora_r=64, question=QUESTION, aux_lm=AUX_LM,
               abstain=ABSTAIN, abstain_swap=ABSTAIN_SWAP, labels=LABELS,
               evfree=EVFREE, evfree_mode=EVFREE_MODE, layout=LAYOUT, split=SPLIT, ema=EMA, teacher=TEACHER, teach_w=TEACH_W)
    with Run("decider_finetune", model=MODEL, config=cfg) as run:
        stats = main()
        print("===", stats, flush=True)
        run.log({k: v for k, v in stats.items() if isinstance(v, (int, float))})
        (ROOT / "results" / f"decider_ft_{NAME}.json").write_text(json.dumps(dict(cfg, **stats), indent=2))
