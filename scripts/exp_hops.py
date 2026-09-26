"""Hop limits (PLAN row 71, MODEL-12): untrained readers on data/processed/hops_v1.json, accuracy against the number of hops k.

Every reader sees the same chat turn: the item's prompt, the six options as a list, and an instruction; the answer is read after a
fixed "Answer:" (each option scored as " name", so single-token names are one token in every tokenizer).

  ARM=decoder  MODEL=Qwen/Qwen2.5-7B-Instruct  MODE=direct|cot
        direct: sum log-prob of each option after "Answer:" (one forward pass decides). cot: greedy chain of thought first (write
        each link, end with "Answer: <name>", up to COT_TOKENS new tokens), cut at its last "Answer:", then the options scored.
  ARM=huginn   MODEL=tomg-group-umd/huginn-0125  STEPS=1,4,8,16,32,64
        the recurrent-depth model's options scored as in direct, once per test-time recurrence count (seeded: its latent state
        starts from noise).
  ARM=diffusion MODEL=GSAI-ML/LLaDA-8B-Instruct | Dream-org/Dream-v0-Instruct-7B | LiquidAI/LFM2.5-Encoder-350M-Diffusion
        MODE=score: the answer slot masked after "Answer:" (as many masks as the option has tokens, then the end-of-turn tokens);
        score1 = the sum of the option's log-probs with every slot masked at once (one denoising step), scoreL = the chain rule,
        one slot revealed per step left to right (for single-token names the two are the same).
        MODE=gen DSTEPS=12,48,192: the thinking analogue: GEN masked tokens after the assistant prompt, the chain-of-thought
        instruction, filled by confidence-ordered unmasking in semi-autoregressive blocks of BLOCK tokens with DSTEPS denoising
        steps in all; the answer parsed from the text (the option named after the last "Answer:", else the last option named).
Subsample with PER_CELL (items per k and name type; default all 50). Writes results/per_item/hops_<name>.<cond>.jsonl (id, k,
names, answer, sum_lp over options, pred, and the generated text for cot / gen) and results/hops_<name>.json (accuracy by
names and k per condition).
usage: [ARM=... MODEL=... MODE=...] uv run python scripts/exp_hops.py
"""
import json
import os
import re
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from ai_experiments.paths import PROCESSED, ROOT

ARM = os.environ.get("ARM", "decoder")
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MODE = os.environ.get("MODE", "direct")
PER_CELL = int(os.environ.get("PER_CELL", "0"))
COT_TOKENS = int(os.environ.get("COT_TOKENS", "768"))
STEPS = [int(s) for s in os.environ.get("STEPS", "1,4,8,16,32,64").split(",")]
DSTEPS = [int(s) for s in os.environ.get("DSTEPS", "12,48,192").split(",")]
GEN, BLOCK = int(os.environ.get("GEN", "192")), int(os.environ.get("BLOCK", "32"))
BATCH = int(os.environ.get("BATCH", "16"))
ITEMS = os.environ.get("ITEMS", "hops_v1")
DEV = "cuda"

ASK_DIRECT = "Reply with the name the chain ends at, as \"Answer: <name>\", and nothing else."
ASK_COT = ("Think step by step: write out each link you follow from the start name, one per line, until the chain ends. "
           "Then give the final name as \"Answer: <name>\".")


def user_text(it, cot):
    opts = "\n".join(f"- {o}" for o in it["options"])
    return f"{it['prompt']}\n\nOptions:\n{opts}\n\n{ASK_COT if cot else ASK_DIRECT}"


def load_items():
    items = json.loads((PROCESSED / f"{ITEMS}.json").read_text())["items"]
    if PER_CELL:
        seen = defaultdict(int); keep = []
        for it in items:
            c = (it["names"], it["k"])
            if seen[c] < PER_CELL:
                keep.append(it); seen[c] += 1
        items = keep
    return items


def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)


def turn_end(tok):
    """The chat template's text after an assistant message (the end-of-turn tokens)."""
    full = tok.apply_chat_template([{"role": "user", "content": "q"}, {"role": "assistant", "content": "XQXZ"}], tokenize=False)
    return full.split("XQXZ")[-1]


def enc(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"]


def parse(text, options):
    """The first option named after the last "Answer:", else the last option named anywhere; None if no option appears."""
    def hits(src):
        return sorted((m.start(), options.index(o)) for o in options for m in re.finditer(rf"(?<![A-Za-z]){re.escape(o)}(?![A-Za-z])", src))
    if "Answer:" in text and hits(text.rsplit("Answer:", 1)[-1]):
        return hits(text.rsplit("Answer:", 1)[-1])[0][1]
    h = hits(text)
    return h[-1][1] if h else None


@torch.no_grad()
def ar_option_lps(model, tok, prefixes, items, fwd=None):
    """Causal scoring, right-padded (causal attention makes the padding invisible): the sum log-prob of " option" after each prefix."""
    fwd = fwd or (lambda ids: model(input_ids=ids).logits)
    out = []
    for p, it in zip(prefixes, items):
        pids = enc(tok, p)
        seqs = [pids + enc(tok, " " + o) for o in it["options"]]
        L = max(map(len, seqs)); pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device=DEV)
        lp = F.log_softmax(fwd(ids).float(), -1)
        row = []
        for j, s in enumerate(seqs):
            t = torch.tensor(s[len(pids):], device=DEV); pos = torch.arange(len(pids) - 1, len(s) - 1, device=DEV)
            row.append(float(lp[j, pos, t].sum()))
        out.append(row)
    return out


def summarise(conds):
    acc = {}
    for cond, recs in conds.items():
        by = defaultdict(list)
        for r in recs:
            by[(r["names"], r["k"])].append(r["pred"] == r["answer"])
        acc[cond] = {f"{n}_k{k:02d}": round(100 * sum(v) / len(v), 1) for (n, k), v in sorted(by.items())}
    return acc


def write(name, conds, extra):
    (ROOT / "results" / "per_item").mkdir(parents=True, exist_ok=True)
    for cond, recs in conds.items():
        with open(ROOT / "results" / "per_item" / f"hops_{name}.{cond}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    acc = summarise(conds)
    (ROOT / "results" / f"hops_{name}.json").write_text(json.dumps({"model": MODEL, "arm": ARM, "mode": MODE, "items": ITEMS,
                                                                     "per_cell": PER_CELL, **extra, "acc": acc}, indent=1))
    for cond, a in acc.items():
        print(cond, " ".join(f"{k}:{v:.0f}" for k, v in a.items()), flush=True)


def rec(it, lps=None, pred="argmax", **kw):
    if pred == "argmax":
        pred = max(range(len(lps)), key=lambda j: lps[j])
    return dict(id=it["id"], k=it["k"], names=it["names"], answer=it["answer"], sum_lp=lps, pred=pred, **kw)


def run_decoder(items, name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=DEV).eval()
    t0 = time.time()
    if MODE == "direct":
        pre = [chat(tok, user_text(it, False)) + "Answer:" for it in items]
        conds = {"direct": [rec(it, lp) for it, lp in zip(items, ar_option_lps(model, tok, pre, items))]}
    else:
        gens = []
        for b in range(0, len(items), BATCH):
            ps = [chat(tok, user_text(it, True)) for it in items[b:b + BATCH]]
            x = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
            y = model.generate(**x, max_new_tokens=COT_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            gens += tok.batch_decode(y[:, x["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"cot {b + BATCH}/{len(items)} {time.time() - t0:.0f}s", flush=True)
        pre = []
        for it, g in zip(items, gens):
            body = g.rsplit("Answer:", 1)[0] if "Answer:" in g else g.rstrip() + "\n"
            pre.append(chat(tok, user_text(it, True)) + body + "Answer:")
        lps = ar_option_lps(model, tok, pre, items)
        conds = {"cot": [rec(it, lp, gen=g, gen_tokens=len(enc(tok, g))) for it, lp, g in zip(items, lps, gens)],
                 "cot_parse": [rec(it, lp, pred=parse(g, it["options"]), gen=g) for it, lp, g in zip(items, lps, gens)]}
    write(name, conds, {"seconds": round(time.time() - t0)})


def run_huginn(items, name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval()
    pre = [chat(tok, user_text(it, False)) + "Answer:" for it in items]
    conds, t0 = {}, time.time()
    for r in STEPS:
        torch.manual_seed(0)
        lps = ar_option_lps(model, tok, pre, items, fwd=lambda ids, r=r: model(input_ids=ids, num_steps=r).logits)
        conds[f"r{r:02d}"] = [rec(it, lp) for it, lp in zip(items, lps)]
        print(f"huginn r={r} {time.time() - t0:.0f}s", flush=True)
    write(name, conds, {"seconds": round(time.time() - t0)})


class Diffusion:
    """A masked-diffusion LM as a function from token ids (with mask tokens) to per-position logits over the tokens at each position."""

    def __init__(self):
        from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
        self.kind = "dream" if "Dream" in MODEL else "llada" if "LLaDA" in MODEL else "lfm"
        try:
            self.tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        except Exception as e:  # Dream's remote tokenizer is Qwen2.5's vocabulary plus a mask token
            print("tokenizer fallback:", repr(e)[:200]); self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        self.mask = {"dream": 151666, "llada": 126336}.get(self.kind) or self.tok.mask_token_id
        cls = AutoModelForMaskedLM if self.kind == "lfm" else AutoModel
        self.model = cls.from_pretrained(MODEL, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval()
        self.vocab = len(self.tok)
        self.banned = [t for t in self.tok.all_special_ids if t not in (self.tok.eos_token_id,)]

    @torch.no_grad()
    def logits(self, ids):
        lg = self.model(input_ids=ids).logits.float()
        if self.kind == "dream":  # Dream keeps its autoregressive parent's shift: position i is predicted at i - 1
            lg = torch.cat([lg[:, :1], lg[:, :-1]], 1)
        return lg[..., :self.vocab]


def run_diffusion(items, name):
    D = Diffusion(); tok = D.tok; conds, t0 = defaultdict(list), time.time()
    if MODE == "score":
        end = enc(tok, turn_end(tok))
        for n, it in enumerate(items):
            p = enc(tok, chat(tok, user_text(it, False)) + "Answer:")
            s1, sL = [], []
            for o in it["options"]:
                t = enc(tok, " " + o); L = len(t)
                ids = torch.tensor([p + [D.mask] * L + end], device=DEV)
                lp = F.log_softmax(D.logits(ids)[0, len(p):len(p) + L], -1)
                s1.append(float(lp[torch.arange(L), torch.tensor(t)].sum()))
                tot = float(lp[0, t[0]])
                for j in range(1, L):  # reveal slot j - 1, score slot j
                    ids[0, len(p) + j - 1] = t[j - 1]
                    tot += float(F.log_softmax(D.logits(ids)[0, len(p) + j], -1)[t[j]])
                sL.append(tot)
            conds["score1"].append(rec(it, s1)); conds["scoreL"].append(rec(it, sL))
            if n % 100 == 0:
                print(f"score {n}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    else:
        for S in DSTEPS:
            nb = GEN // BLOCK; per_block = max(1, S // nb)
            for n, it in enumerate(items):
                p = enc(tok, chat(tok, user_text(it, True)))
                ids = torch.tensor([p + [D.mask] * GEN], device=DEV)
                for b in range(nb):
                    lo, hi = len(p) + b * BLOCK, len(p) + (b + 1) * BLOCK
                    sched = [BLOCK // per_block + (i < BLOCK % per_block) for i in range(per_block)]
                    for m in sched:
                        pos = (ids[0, lo:hi] == D.mask).nonzero()[:, 0] + lo
                        if len(pos) == 0:
                            break
                        lg = D.logits(ids)[0, pos]; lg[:, D.banned] = -1e9
                        conf, tid = lg.softmax(-1).max(-1)
                        top = conf.topk(min(m, len(pos))).indices
                        ids[0, pos[top]] = tid[top]
                g = ids[0, len(p):].tolist()
                eos = [j for j, t in enumerate(g) if t in (tok.eos_token_id, *enc(tok, turn_end(tok))[:1])]
                text = tok.decode(g[:eos[0]] if eos else g, skip_special_tokens=True)
                conds[f"gen_s{S:03d}"].append(rec(it, None, pred=parse(text, it["options"]), gen=text))
                if n % 50 == 0:
                    print(f"gen S={S} {n}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    write(name, dict(conds), {"seconds": round(time.time() - t0), "gen": GEN, "block": BLOCK})


if __name__ == "__main__":
    items = load_items()
    name = f"{MODEL.split('/')[-1]}_{MODE if ARM != 'huginn' else 'direct'}" + (f"_pc{PER_CELL}" if PER_CELL else "")
    print(ARM, MODEL, MODE, len(items), "items ->", name, flush=True)
    {"decoder": run_decoder, "huginn": run_huginn, "diffusion": run_diffusion}[ARM](items, name)
