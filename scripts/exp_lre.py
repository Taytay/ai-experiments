"""Relation-linearity probe (PLAN step 28, GRAPH-1, DATA-1): did the adapter write a relation or 136 facts?

Hernandez et al. (2308.09124): for many relations the subject-to-object map inside an LM is one linear transform on the subject's
hidden state (a linear relational embedding, LRE). Here the subject is a species name, the relation "type" (or "weakness"), the
object the type token. For a layer sweep, the probe reads the hidden state at the last token of the prompt
"Question: What type is <name>?\\nAnswer:" at layer l (the subject has been read; this is the position the LM head decodes) and fits a
ridge map to the FINAL-layer hidden state at the same position, whose logits over the eight type options are the model's answer.
  faithfulness   5-fold cross-validation over the 136 trained species: does the mapped state pick the same option the model picks
                 (agreement) and the gold type (accuracy)? A map that works from a middle layer on held-out species is a relation;
                 one that only works at the top, or not across folds, is memorisation the head reads off.
  transfer       the map fitted on all trained species applied to the 24 held-out species, without context (the model is at chance
                 there, so the probe can only exploit the name) and with the field-guide entry in the prompt (the state now carries
                 the type; does the same map read it?).
  factoring      the weakness relation two ways: its own LRE, and the type LRE's answer sent through the type -> weakness rotation
                 (WEAKNESS[type]). On the original universe the two agree by construction; on the independent-weakness universe
                 (WEAKNESS=independent, PLAN step 23) they can only agree if the model stores weakness as a function of type.
Eval only; minutes.

usage: uv run python scripts/exp_lre.py [base|<adapter dir under models/adapters>]      WEAKNESS=independent for the _wind universe
outputs: results/lre_<tag>.json; tracker experiment "lre"
"""
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import unsloth  # noqa: F401
import numpy as np
import torch
from unsloth import FastLanguageModel

from ai_experiments import universe as U
from ai_experiments.evals.tracker import Run
from ai_experiments.paths import ROOT

WHAT = sys.argv[1] if len(sys.argv) > 1 else "base"
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-3B")
WEAKNESS = os.environ.get("WEAKNESS", "type")
LAYERS = [int(x) for x in os.environ.get("LAYERS", "4,8,12,16,20,24,28,32,36").split(",")]
RIDGE = float(os.environ.get("RIDGE", "1.0"))
tag = ("base" if WHAT == "base" else WHAT) + ("_wind" if WEAKNESS == "independent" else "")
OUT = ROOT / "results" / f"lre_{tag}.json"
species = U.build(n_per_type=20, weakness=WEAKNESS)
by_name = {s["name"]: s for s in species}
seen = [s for s in species if not s["heldout"]]; held = [s for s in species if s["heldout"]]
Q = {"type": "Question: What type is {n}?\nAnswer:", "weakness": "Question: What type is {n} weak to?\nAnswer:"}


def load():
    src = MODEL if WHAT == "base" else str(ROOT / "models" / "adapters" / WHAT)
    model, tok = FastLanguageModel.from_pretrained(src, max_seq_length=1024, dtype=torch.bfloat16)
    if WHAT != "base":
        FastLanguageModel.for_inference(model)  # the adapter stays a PeftModel (merge_and_unload broke unsloth's fast forward); the hidden states are the same
    model.eval()
    return model, tok


def inner(model):
    """The decoder stack that owns .layers and .norm, through the Peft wrapper if there is one."""
    m = model
    while not hasattr(m, "layers"):
        m = m.model if hasattr(m, "model") else m.base_model
    return m


@torch.no_grad()
def states(model, tok, prompts):
    """Hidden states at the last prompt token for every layer: {layer: [n, d]} (layer 0 = embeddings), plus the final-layer state."""
    out = {l: [] for l in range(len(inner(model).layers) + 1)}
    for i in range(0, len(prompts), 16):
        enc = tok(prompts[i:i + 16], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        hs = model(**enc, output_hidden_states=True).hidden_states
        last = enc["attention_mask"].sum(1) - 1
        for l, h in enumerate(hs):
            out[l].append(h[torch.arange(h.shape[0]), last].float().cpu())
    return {l: torch.cat(v).numpy() for l, v in out.items()}


def option_ids(tok):
    return [tok(" " + t, add_special_tokens=False)["input_ids"][0] for t in U.TYPE_LIST]  # first token of each type name


def decode(model, tok, H):
    """Argmax over the eight type options of the LM head applied to final-layer states H [n, d] (after the final norm)."""
    W = model.get_output_embeddings().weight[option_ids(tok)].float()  # [8, d]
    h = inner(model).norm(torch.tensor(H, device="cuda", dtype=torch.bfloat16)).float()
    return (h @ W.T).argmax(1).cpu().numpy()


def fit(X, Y, ridge=RIDGE):
    """Ridge regression Y ~ X W + b."""
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    A = Xb.T @ Xb + ridge * np.eye(Xb.shape[1]); A[-1, -1] -= ridge
    return np.linalg.solve(A, Xb.T @ Y)


def apply(W, X):
    return np.concatenate([X, np.ones((len(X), 1))], 1) @ W


def probe(model, tok, rel, ctx):
    """One relation, with or without the field-guide entry in the prompt. Returns per-layer faithfulness / transfer numbers."""
    def prompt(s):
        p = Q[rel].format(n=s["name"])
        return (f"Field guide:\n{U.entry(s)}\n\n" + p) if ctx else p
    gold = lambda s: U.TYPE_LIST.index(s[rel])  # noqa: E731
    Hs, Hh = states(model, tok, [prompt(s) for s in seen]), states(model, tok, [prompt(s) for s in held])
    top = max(Hs)
    model_pred_seen, model_pred_held = decode(model, tok, Hs[top]), decode(model, tok, Hh[top])
    g_seen, g_held = np.array([gold(s) for s in seen]), np.array([gold(s) for s in held])
    r = {"model_acc_seen": float((model_pred_seen == g_seen).mean() * 100), "model_acc_held": float((model_pred_held == g_held).mean() * 100)}
    rng = np.random.default_rng(0); folds = np.array_split(rng.permutation(len(seen)), 5)
    for l in LAYERS:
        if l not in Hs:
            continue
        X, Y = Hs[l], Hs[top]
        cv_pred = np.zeros(len(seen), int)
        for f in folds:
            tr = np.setdiff1d(np.arange(len(seen)), f)
            cv_pred[f] = decode(model, tok, apply(fit(X[tr], Y[tr]), X[f]))
        W = fit(X, Y)
        held_pred = decode(model, tok, apply(W, Hh[l]))
        r[f"L{l}_cv_agreement"] = float((cv_pred == model_pred_seen).mean() * 100)
        r[f"L{l}_cv_acc"] = float((cv_pred == g_seen).mean() * 100)
        r[f"L{l}_held_acc"] = float((held_pred == g_held).mean() * 100)
        r[f"L{l}_held_agreement"] = float((held_pred == model_pred_held).mean() * 100)
        if rel == "weakness":  # factoring: predict weakness by sending the TYPE probe's answer through the rotation
            r[f"L{l}_via_type_acc"] = None  # filled by main from the type probe's per-species predictions
        r[f"L{l}_pred_seen"] = cv_pred.tolist(); r[f"L{l}_pred_held"] = held_pred.tolist()
    return r


cfg = dict(what=WHAT, model=MODEL, weakness=WEAKNESS, layers=LAYERS, ridge=RIDGE, n_seen=len(seen), n_held=len(held))
with Run("lre", model=MODEL, config=cfg, enabled=True) as run:
    t0 = time.time()
    model, tok = load()
    res = {}
    for ctx in (False, True):
        for rel in ("type", "weakness"):
            res[f"{rel}_{'ctx' if ctx else 'noctx'}"] = probe(model, tok, rel, ctx)
        # factoring: weakness via the type probe's cross-validated type prediction and the rotation map
        rot = [U.TYPE_LIST.index(U.WEAKNESS[t]) for t in U.TYPE_LIST]
        t_r, w_r = res[f"type_{'ctx' if ctx else 'noctx'}"], res[f"weakness_{'ctx' if ctx else 'noctx'}"]
        g_w_seen = np.array([U.TYPE_LIST.index(s["weakness"]) for s in seen]); g_w_held = np.array([U.TYPE_LIST.index(s["weakness"]) for s in held])
        for l in LAYERS:
            if f"L{l}_pred_seen" in t_r:
                via_seen = np.array([rot[p] for p in t_r[f"L{l}_pred_seen"]]); via_held = np.array([rot[p] for p in t_r[f"L{l}_pred_held"]])
                w_r[f"L{l}_via_type_acc"] = float((via_seen == g_w_seen).mean() * 100)
                w_r[f"L{l}_via_type_held_acc"] = float((via_held == g_w_held).mean() * 100)
    for k, r in res.items():
        for kk in [x for x in r if x.endswith("_pred_seen") or x.endswith("_pred_held")]:
            r.pop(kk)
        run.log({kk: v for kk, v in r.items() if isinstance(v, (int, float))}, condition=k)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(res, indent=2)); run.artifact(OUT)
print(f"=== lre {tag}")
for k, r in res.items():
    if isinstance(r, dict):
        print(f"-- {k}: model seen {r['model_acc_seen']:.1f} held {r['model_acc_held']:.1f} | " + " ".join(f"L{l}:{r.get(f'L{l}_cv_acc', 0):.0f}/{r.get(f'L{l}_held_acc', 0):.0f}" for l in LAYERS if f"L{l}_cv_acc" in r))
