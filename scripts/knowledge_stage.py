"""PLAN row 212 (c), knowledge pretraining: a contrastive stage that puts world knowledge about merchants and places into the history
encoder's weights before its history training. Pairs from scripts/build_knowledge_pairs.py (data/interim/knowledge_pairs_<VERSION>.jsonl):
anchor = the bank-rendered payee string (`text`), positive = its taxonomy_v2 kind in words (`kind_text`).
  loss   InfoNCE with in-batch negatives, scale 20 (tau 0.05, as hist_train2 and the readers). Many merchants share a kind, so an in-batch
         candidate whose kind text equals the anchor's is not a negative: it is masked. Each distinct kind text in the batch is encoded once,
         which is the same thing (string -> kind: a softmax over the batch's distinct kinds). Plus the symmetric term kind -> string (UWE,
         references/papers/2511.07969: symmetry and structure +1.8 MAP): for each pair, its kind against the batch's strings, the other
         strings of that kind masked. SYM weights it (1; 0 turns it off).
  MIX    fraction of steps that are history-encoder triplet batches instead (hist_train2._triplets, the b1 recipe's infonce + dedup loss),
         to mix knowledge into the history objective rather than staging it before (default 0: knowledge only). Needs the history
         world's env (SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1) and hist_train2's PAIRS / HOUSEHOLDS / CAP.
  held out  pairs whose merchant (brand, else name) falls in a HOLDOUT hash bucket are never trained on; `eval` reads them.
Modes:
  train  writes OUT (models/encoders/know_r212_<ARM>), a SentenceTransformer dir hist_encoder._model() loads, plus know_config.json. The
         history stage starts from it: BASE=models/encoders/know_r212_<ARM> ARM=... LOSS=infonce,dedup uv run python scripts/hist_train2.py
         train (hist_train2 calls open_licence(H.BASE) unconditionally, which rejects a local path: it needs li_decider.py's guard, see
         scripts/chains/r212_know.sh). Here open_licence runs on BASE only when BASE is not a local directory (as li_decider.py); a local
         BASE was checked when it was trained from its hub base.
  eval   ENCS=<dir or hub id,...>: kind retrieval on the held-out pairs, recall@1 / @5 of the right kind text among all distinct kind
         texts in PAIRS_FILE, overall and by origin. A bare name is looked up under models/encoders first.
env: PAIRS_FILE (data/interim/knowledge_pairs_v1.jsonl), BASE (hist_encoder.BASE: env BASE, bge-small-en-v1.5), ARM (k1), OUT, KBATCH (256),
     STEPS (0: from EPOCHS), EPOCHS (1), LR (5e-5; 5% warm-up, linear decay), SYM (1), HOLDOUT (0.02), N_EVAL (0 = all held-out pairs),
     MIX (0), HBATCH (128: history batch), SEED (0), LOG (100: steps between loss lines).
usage: ARM=k1 uv run python scripts/knowledge_stage.py train
       ENCS=BAAI/bge-small-en-v1.5,know_r212_k1 uv run python scripts/knowledge_stage.py eval
       smoke (CPU): CUDA_VISIBLE_DEVICES= PAIRS_FILE=data/interim/knowledge_pairs_smoke.jsonl HOLDOUT=0.1 STEPS=30 KBATCH=64 ARM=smoke \
                    uv run python scripts/knowledge_stage.py train
"""
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import hist_encoder as H  # noqa: E402

from ai_experiments.paths import ROOT  # noqa: E402

PAIRS_FILE = Path(os.environ.get("PAIRS_FILE", str(ROOT / "data" / "interim" / "knowledge_pairs_v1.jsonl")))
PAIRS_FILE = PAIRS_FILE if PAIRS_FILE.is_absolute() else ROOT / PAIRS_FILE
# row 222: several pair files, comma-separated (knowledge pairs + alias pairs); PAIRS_FILE names the first (for the config)
PAIRS_FILES = [Path(x) if Path(x).is_absolute() else ROOT / x for x in os.environ.get("PAIRS_FILE", str(PAIRS_FILE)).split(",")]
PAIRS_FILE = PAIRS_FILES[0]
ARM = os.environ.get("ARM", "k1")
OUT = Path(os.environ.get("OUT", str(H.ENC / f"know_r212_{ARM}")))
KBATCH, HBATCH = int(os.environ.get("KBATCH", "256")), int(os.environ.get("HBATCH", "128"))
STEPS, EPOCHS, LR = int(os.environ.get("STEPS", "0")), float(os.environ.get("EPOCHS", "1")), float(os.environ.get("LR", "5e-5"))
LOWER = int(os.environ.get("LOWER", "0"))
MV = int(os.environ.get("MV", "0"))  # row 222: the stage in the late-interaction setting (li_decider.LI, MaxSim), as ColBERT-Zero
SYM, HOLDOUT, N_EVAL = float(os.environ.get("SYM", "1")), float(os.environ.get("HOLDOUT", "0.02")), int(os.environ.get("N_EVAL", "0"))
MIX, SEED, LOG = float(os.environ.get("MIX", "0")), int(os.environ.get("SEED", "0")), int(os.environ.get("LOG", "100"))
SCALE = 20.0  # 1 / 0.05


def _key(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def held_out(p):
    """a stable hash bucket of the merchant (brand when it has one, so a brand's places fall on one side; else the name): every rendering
    of a merchant lands on the same side"""
    k = _key(p.get("brand") or p.get("name") or p["text"])
    return int(hashlib.md5(k.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < HOLDOUT


def load():
    pairs = [json.loads(line) for f in PAIRS_FILES for line in f.open()]
    if LOWER:  # row 222: cased bases (Ettin) read ALL-CAPS bank strings as many pieces; li_decider LOWER=1 lowercases the same way
        pairs = [dict(p, text=p["text"].lower(), kind_text=p["kind_text"].lower()) for p in pairs]
    kinds = sorted({p["kind_text"] for p in pairs})
    train = [p for p in pairs if not held_out(p)]
    test = [p for p in pairs if held_out(p)]
    return train, test, kinds


def _dev():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _know_loss(model, b, dev):
    """string -> kind over the batch's distinct kinds, plus SYM x kind -> string with same-kind strings masked"""
    import torch
    F = torch.nn.functional
    from hist_train2 import _enc
    uk = sorted({p["kind_text"] for p in b})
    ki = torch.tensor([uk.index(p["kind_text"]) for p in b], device=dev)
    A = _enc(model, [p["text"] for p in b], dev)
    K = _enc(model, uk, dev)
    S = SCALE * (A @ K.T)  # n x distinct kinds
    l1 = F.cross_entropy(S.float(), ki)
    if not SYM:
        return l1, l1.item(), 0.0
    R = S[:, ki].T  # R[i, j] = score(string j, kind of pair i)
    same = (ki[:, None] == ki[None, :])
    same.fill_diagonal_(False)
    l2 = F.cross_entropy(R.float().masked_fill(same, float("-inf")), torch.arange(len(b), device=dev))
    return l1 + SYM * l2, l1.item(), l2.item()


def _know_loss_mv(model, scale, b, dev):
    """MV=1 (ColBERT-Zero, 2602.16609: run the contrastive phases in the multi-vector setting): _know_loss with li_decider's late-interaction
    score (token vectors, MaxSim, a learned scale) in place of pooled cosines; the string is the query, the kind (or the alias rendering)
    the document; SYM adds the kind as query against the batch's strings, same-kind strings masked"""
    import torch
    F = torch.nn.functional
    uk = sorted({p["kind_text"] for p in b})
    ki = torch.tensor([uk.index(p["kind_text"]) for p in b], device=dev)
    texts = [p["text"] for p in b]
    S = scale * model.score(model.vecs(texts, "q"), model.vecs(uk, "d"))
    l1 = F.cross_entropy(S.float(), ki)
    if not SYM:
        return l1, l1.item(), 0.0
    R = (scale * model.score(model.vecs(uk, "q"), model.vecs(texts, "d")))[ki]  # R[i, j] = score(kind of pair i as query, string j)
    same = (ki[:, None] == ki[None, :])
    same.fill_diagonal_(False)
    l2 = F.cross_entropy(R.float().masked_fill(same, float("-inf")), torch.arange(len(b), device=dev))
    return l1 + SYM * l2, l1.item(), l2.item()


def _hist_loss(model, b, dev):
    """hist_train2's infonce + dedup (the b1 recipe) on one batch of its triplets: anchor against the batch's positives and hard
    negatives; a candidate from the anchor's household filed in the anchor's category, or with the positive's text, is masked"""
    import torch
    from hist_train2 import _enc
    n = len(b)
    A = _enc(model, [r["a"] for r in b], dev)
    C = _enc(model, [r["p"] for r in b] + [r["n"] for r in b], dev)
    L = SCALE * (A @ C.T)
    hh = torch.tensor([r["h"] for r in b], device=dev)
    ga = [hash(r["ga"]) for r in b]
    gc = torch.tensor(ga + [hash(r["gn"]) for r in b], device=dev)
    tp = [r["p"] for r in b]
    txt = torch.tensor([[tp[i] == t for t in tp + [r["n"] for r in b]] for i in range(n)], device=dev)
    fn = ((hh[:, None] == torch.cat([hh, hh])[None, :]) & (torch.tensor(ga, device=dev)[:, None] == gc[None, :])) | txt
    fn[torch.arange(n), torch.arange(n)] = False
    return torch.nn.functional.cross_entropy(L.masked_fill(fn, float("-inf")).float(), torch.arange(n, device=dev))


def train():
    import torch
    from ai_experiments.licences import open_licence
    if not Path(H.BASE).exists():  # a local dir (an earlier stage) was checked when it was trained from its hub base
        open_licence(H.BASE)
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    tr, te, kinds = load()
    steps = STEPS or max(1, int(EPOCHS * len(tr) / KBATCH / (1 - MIX)))
    print(f"{PAIRS_FILE.name}: {len(tr)} train pairs, {len(te)} held out, {len(kinds)} kinds; {steps} steps of {KBATCH} "
          f"(MIX {MIX}, SYM {SYM}) from {H.BASE}", flush=True)
    rows = []
    if MIX > 0:
        import hist_train2 as T2
        from two_tower import households
        t0 = time.time()
        hh = [H.events(b) for b in households("train", range(T2.HOUSEHOLDS))]
        rows = T2._triplets(hh, {"infonce", "dedup"}, rng)
        print(f"history: {len(rows)} triplets from {len(hh)} households ({time.time() - t0:.0f}s)", flush=True)
    dev = _dev()
    if MV:
        import li_decider as LD
        assert not MIX, "MV=1 trains the knowledge pairs only"
        model = LD.LI(H.BASE, train=True)
        if LD.GC:  # row 209: Ettin-400M ran out of memory on an L40S at KBATCH 256 without it
            model.cb.enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        scale = torch.nn.Parameter(torch.tensor(20.0, device=model.dev))
        opt = torch.optim.AdamW([{"params": model.params()}, {"params": [scale], "lr": 1e-2}], lr=LR)
    else:
        model = H._model(H.BASE)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    order, ko, ho = list(range(len(tr))), 0, 0
    rng.shuffle(order)
    t0, run, cnt = time.time(), defaultdict(float), Counter()
    for s in range(steps):
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            if rows and rng.random() < MIX:
                if ho + HBATCH > len(rows):
                    rng.shuffle(rows); ho = 0
                loss = _hist_loss(model, rows[ho:ho + HBATCH], dev)
                ho += HBATCH
                run["hist"] += loss.item(); cnt["hist"] += 1
            else:
                if ko + KBATCH > len(order):
                    rng.shuffle(order); ko = 0
                bt = [tr[i] for i in order[ko:ko + KBATCH]]
                loss, l1, l2 = _know_loss_mv(model, scale, bt, dev) if MV else _know_loss(model, bt, dev)
                ko += KBATCH
                run["str->kind"] += l1; run["kind->str"] += l2; cnt["str->kind"] += 1; cnt["kind->str"] += 1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.params() if MV else model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (s + 1) % LOG == 0 or s + 1 == steps:
            print(f"  step {s + 1}/{steps} " + " ".join(f"{k} {v / cnt[k]:.3f}" for k, v in run.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
            run.clear(); cnt.clear()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT))
    if MV:  # li_decider starts from this dir with BASE=...: it loads proj.pt and scale.pt from it
        torch.save(scale.detach().cpu(), OUT / "scale.pt")
    (OUT / "know_config.json").write_text(json.dumps(dict(
        plan_row=212, arm="c", name=ARM, base=str(H.BASE), pairs_file=str(PAIRS_FILE.relative_to(ROOT)) if PAIRS_FILE.is_relative_to(ROOT) else str(PAIRS_FILE),
        n_train=len(tr), n_held_out=len(te), kinds=len(kinds), steps=steps, kbatch=KBATCH, hbatch=HBATCH, lr=LR, scale=SCALE, sym=SYM,
        holdout=HOLDOUT, mix=MIX, mv=MV, history_triplets=len(rows), seed=SEED, device=dev, train_seconds=round(time.time() - t0)), indent=1))
    print(f"-> {OUT} (train {time.time() - t0:.0f}s)", flush=True)


def evaluate():
    """recall@1 / @5 of the right kind text among all distinct kinds, for each encoder, on the held-out pairs"""
    import torch
    _, te, kinds = load()
    if N_EVAL:
        te = random.Random(SEED).sample(te, min(N_EVAL, len(te)))
    ki = torch.tensor([kinds.index(p["kind_text"]) for p in te])
    print(f"\n**{PAIRS_FILE.name}: kind retrieval on {len(te)} held-out pairs, {len(kinds)} kinds: recall@1 / @5 (%)**\n")
    origins = sorted(Counter(p["origin"].split(":")[0] for p in te))
    print("| encoder | all | " + " | ".join(origins) + " |\n|---|---|" + "---|" * len(origins))
    for enc in os.environ.get("ENCS", str(H.BASE)).split(","):
        d = H.ENC / enc if (H.ENC / enc).exists() else enc
        if (Path(d) / "proj.pt").exists():  # a late-interaction model (MV=1 stage or li_decider): MaxSim, string as query, kind as document
            import li_decider as LD
            m = LD.LI(d); m.cb.enc.eval()
            with torch.no_grad():
                Kv = m.vecs(kinds, "d")
                S = torch.cat([m.score(m.vecs([p["text"] for p in te[a:a + 256]], "q"), Kv).float().cpu() for a in range(0, len(te), 256)])
        else:
            m = H._model(d)
            m.eval()
            with torch.no_grad():
                A = m.encode([p["text"] for p in te], convert_to_tensor=True, normalize_embeddings=True, batch_size=256).cpu()
                K = m.encode(kinds, convert_to_tensor=True, normalize_embeddings=True).cpu()
            S = A @ K.T
        rank = (S > S.gather(1, ki[:, None])).sum(1)  # kinds scoring strictly above the right one
        cells = []
        for o in ["all"] + origins:
            sel = torch.tensor([o == "all" or p["origin"].split(":")[0] == o for p in te])
            r = rank[sel]
            cells.append(f"{100 * (r < 1).float().mean():.1f} / {100 * (r < 5).float().mean():.1f} (n={len(r)})")
        print(f"| {enc} | " + " | ".join(cells) + " |", flush=True)
        del m


if __name__ == "__main__":
    {"train": train, "eval": evaluate}[sys.argv[1]]()
