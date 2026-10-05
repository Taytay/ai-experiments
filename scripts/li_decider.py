"""PLAN step 210 (owner, 2026-10-05: "Has anyone built a Jev-like model with a late interaction model?"; "It would be awesome if we could
make a decision model using this"): a late-interaction decision model. Every one of the household's visible categories is a short
document of tokens, "<Group: Name> | <payee> $<amount> | ..." (its last M filings, distinct payees, most recent first; "nothing filed
yet" when empty), rebuilt each day from filings before that day; the transaction is the query, "<payee> | $<amount> | <weekday>", and with
CTX=1 also its NB nearest earlier transactions by hist_knn_v1 and where they went (row 194's ctx query). Query and documents are encoded
separately (hist_encoder2.ColBERT: an encoder's token states, a linear map to 128, normalised), so the documents can be encoded once per
day and shared by every transaction that day; each option's score is ColBERT MaxSim (mean over query tokens of the best cosine among the
document's tokens) times a learned scale. Like Clef (row 164) and decider, every option is scored at once and the model is trained as a
decision: softmax over the household's visible categories, cross-entropy plus BRIER x the Brier score (Clef's calibration term), not a
retrieval loss. Training: shared-world v4 households (train split); a step takes one household and the transactions of WINDOW
consecutive days (so the day's documents are shared), up to B of them.
  train   writes models/encoders/li_r<ROW>_<ARM> (encoder, proj.pt, scale.pt)
  read    % top-1 / top-3 / top-10, all and first-time payee, and the Brier score / ECE of the top choice, per ARMS (READ=households | budget)
env: ARM, BASE (hist_colbert_v1: row 195's ColBERT, or any encoder), CTX (0), NB (5), M (8), B (32), WINDOW (7), STEPS (3000),
     HOUSEHOLDS (200), BRIER (1), LR (5e-5), SEED (0), HYBRID (0), QW (0), SOFT (0: MaxSim; tau_a for UWE's soft interaction),
     MODE (doc: one document per category | mml: per-filing candidates, BELXTR), NCAND (50), TEST_SEEDS (100000-100049), ARMS;
     row 209 (any base): PDIM (128), PROJ (linear | res), PROJ_INIT (1), LOWER (0), POOL (cls | mean | last), MAXLEN (default min(96, the base's): row 210's length).
usage: SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 ARM=a0 uv run python scripts/li_decider.py train
"""
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hist_encoder as H  # noqa: E402
import hist_encoder2 as H2  # noqa: E402
from two_tower import households  # noqa: E402

ARM = os.environ.get("ARM", "a0")
BASE = os.environ.get("BASE", str(H.ENC / "hist_colbert_v1"))
CTX, NB, M = int(os.environ.get("CTX", "0")), int(os.environ.get("NB", "5")), int(os.environ.get("M", "8"))
B, WINDOW, STEPS = int(os.environ.get("B", "32")), int(os.environ.get("WINDOW", "7")), int(os.environ.get("STEPS", "3000"))
HOUSEHOLDS, BRIER, LR = int(os.environ.get("HOUSEHOLDS", "200")), float(os.environ.get("BRIER", "1")), float(os.environ.get("LR", "5e-5"))
SEED = int(os.environ.get("SEED", "0"))
HYBRID, QW = int(os.environ.get("HYBRID", "0")), int(os.environ.get("QW", "0"))
GROUPS = int(os.environ.get("GROUPS", "1"))  # household windows per training step
SOFT = float(os.environ.get("SOFT", "0"))  # UWE (2511.07969) soft late interaction: softmax over document tokens at this temperature; 0 = MaxSim
MODE, NCAND = os.environ.get("MODE", "doc"), int(os.environ.get("NCAND", "50"))  # mml: BELXTR (2609.25859) per-filing candidates
# row 211: trained interaction layers over the cached token vectors (MODE=doc): "x" cross-attention from the transaction's tokens to each
# option's tokens, "m" + memory tokens (its NMEM nearest earlier filings, each one vector: the filing's and its category's pooled vectors),
# "l" + one listwise layer over the options; e.g. INTERACT=xml. ROW names the output (li_r<ROW>_<ARM>).
INTERACT, NMEM, XLAYERS = os.environ.get("INTERACT", ""), int(os.environ.get("NMEM", "10")), int(os.environ.get("XLAYERS", "2"))
ROW = os.environ.get("ROW", "210")
# row 209: any base encoder (ai_experiments.licences.open_licence on hub ids). PDIM: token-vector dimensions (128; mxbai 2510.14880: 64 costs
# nothing, 32 does); PROJ: linear | res (mxbai's 2-layer FFN with an upscaled hidden and a residual, +1.3 at 17M; its second layer starts at
# zero, so step 0 equals the linear head); PROJ_INIT (1): start the linear map from the base's own ColBERT head when its shape fits (PyLate
# Dense layers without bias or activation, composed; answerai-colbert's linear.weight); LOWER (0): lowercase every text before tokenising
# (cased bases such as Ettin, ModernBERT, Qwen3 read ALL-CAPS payees as many pieces; mxbai: +0.9 at 17M); POOL: the pooled vector for
# HYBRID and memory tokens, cls | mean | last (decoder embedders such as Qwen3-Embedding: the last non-padding token, either padding side);
# MAXLEN: default min(96, the base's own max length), as row 210 trained (models saved before row 209 read at 96, as trained). Learning rate: Ettin needs ~1.8x
# (mxbai: 32M 2.8e-4 -> 5e-4), set LR.
PDIM, PROJ, PROJ_INIT = int(os.environ.get("PDIM", "128")), os.environ.get("PROJ", "linear"), int(os.environ.get("PROJ_INIT", "1"))
LOWER, POOL, MAXLEN = int(os.environ.get("LOWER", "0")), os.environ.get("POOL", "cls"), int(os.environ.get("MAXLEN", "0"))


def _base_head(path, hidden):
    """the base's trained ColBERT projection as one [out, hidden] matrix, or None: PyLate's k_Dense layers (no bias, identity activation,
    composed in order) or a Stanford-format checkpoint's linear.weight"""
    import torch
    from safetensors.torch import load_file

    def get(name):
        p = Path(path) / name
        if p.exists():
            return p
        if Path(path).exists():
            return None
        try:
            from huggingface_hub import hf_hub_download
            return Path(hf_hub_download(str(path), name))
        except Exception:
            return None
    W = None
    for k in range(1, 6):
        c = get(f"{k}_Dense/config.json")
        if c is None:
            break
        cfg = json.loads(c.read_text())
        if cfg.get("bias") or "Identity" not in cfg.get("activation_function", "Identity"):
            return None
        w = next(iter(load_file(str(get(f"{k}_Dense/model.safetensors"))).values())).float()
        W = w if W is None else w @ W
    if W is None and get("artifact.metadata") is not None:  # Stanford ColBERT layout: the head sits in the main checkpoint
        st = load_file(str(get("model.safetensors")))
        W = st["linear.weight"].float() if "linear.weight" in st else None
    return W if W is not None and W.shape[1] == hidden else None


class Enc(H2.ColBERT):
    """hist_encoder2.ColBERT made base-agnostic for row 209 (PDIM, PROJ, PROJ_INIT, MAXLEN; hist_encoder2.py is left as it is). With the
    defaults and a saved model (proj.pt = Linear(384, 128, bias=False)) it loads and behaves as before."""

    def __init__(self, path, train=False):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(str(path))
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.enc = AutoModel.from_pretrained(str(path)).to(self.dev)
        hid = self.enc.config.hidden_size
        lin = torch.nn.Linear(hid if PROJ == "linear" else 2 * hid, PDIM, bias=False)
        if PROJ == "linear":
            self.proj = lin.to(self.dev)
        elif PROJ == "res":
            class Res(torch.nn.Module):
                """up to 2x hidden, + a residual GELU FFN at that width (second layer zero-initialised), then down to PDIM"""

                def __init__(self):
                    super().__init__()
                    self.up, self.f1, self.f2, self.out = torch.nn.Linear(hid, 2 * hid), torch.nn.Linear(2 * hid, 2 * hid), torch.nn.Linear(2 * hid, 2 * hid), lin
                    torch.nn.init.zeros_(self.f2.weight); torch.nn.init.zeros_(self.f2.bias)

                def forward(self, h):
                    x = self.up(h)
                    return self.out(x + self.f2(torch.nn.functional.gelu(self.f1(x))))
            self.proj = Res().to(self.dev)
        else:
            raise ValueError(f"PROJ={PROJ}")
        if (Path(path) / "proj.pt").exists():
            self.proj.load_state_dict(torch.load(Path(path) / "proj.pt", map_location=self.dev))
        elif PROJ_INIT and PROJ == "linear":
            W = _base_head(path, hid)
            if W is not None and W.shape[0] == PDIM:
                with torch.no_grad():
                    self.proj.weight.copy_(W)
                print(f"projection from the base's ColBERT head {tuple(W.shape)}", flush=True)
        base_max = min(x for x in (self.tok.model_max_length, getattr(self.enc.config, "max_position_embeddings", 10 ** 9)) if x)
        self.maxlen = MAXLEN or min(96, base_max)  # 96 as row 210 trained; MAXLEN raises it
        self.enc.train(train)


class LI:
    """hist_encoder2.ColBERT plus two options from the late-interaction papers (references/papers INDEX thread 8): HYBRID adds the cosine of
    the normalised [CLS] states to MaxSim (SMART, 2605.24938: the hybrid objective beat late-only by +0.8 and pooled-only by +6.5);
    QW replaces the plain mean over query tokens by learned weights, softmax over the query's tokens of a linear read of each token
    state (Weighted Chamfer, 2511.16106, made contextual: the amount, weekday and separators need not count as much as the payee)"""

    def __init__(self, path, train=False):
        import torch
        self.cb = Enc(path, train=train)
        self.dev = self.cb.dev
        self.qw = torch.nn.Linear(self.cb.enc.config.hidden_size, 1).to(self.dev)
        torch.nn.init.zeros_(self.qw.weight); torch.nn.init.zeros_(self.qw.bias)
        if (Path(path) / "qw.pt").exists():
            self.qw.load_state_dict(torch.load(Path(path) / "qw.pt", map_location=self.dev))

    def params(self):
        return list(self.cb.enc.parameters()) + list(self.cb.proj.parameters()) + (list(self.qw.parameters()) if QW else [])

    def vecs(self, texts):
        """token vectors [n, L, PDIM], mask [n, L], query-token weights [n, L] (sum 1 over the mask), normalised pooled state [n, hidden]
        (POOL: [CLS] / masked mean / last non-padding token)"""
        import torch
        if LOWER:
            texts = [t.lower() for t in texts]
        b = self.cb.tok(texts, padding=True, truncation=True, max_length=self.cb.maxlen, return_tensors="pt").to(self.dev)
        with torch.autocast(self.dev, dtype=torch.bfloat16, enabled=self.dev == "cuda"):
            h = self.cb.enc(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        h = h.float()
        m = b["attention_mask"].bool()
        v = torch.nn.functional.normalize(self.cb.proj(h), dim=-1)
        w = (self.qw(h).squeeze(-1) if QW else torch.zeros(m.shape, device=self.dev)).masked_fill(~m, float("-inf")).softmax(-1)
        if POOL == "cls":
            p = h[:, 0]
        elif POOL == "mean":
            p = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        elif POOL == "last":  # the last real token whichever side the padding is on
            L = m.shape[1]
            last = L - 1 - m.flip(1).int().argmax(1)
            p = h[torch.arange(len(h), device=h.device), last]
        else:
            raise ValueError(f"POOL={POOL}")
        return v, m, w, torch.nn.functional.normalize(p, dim=-1)

    def score(self, q, d):
        """[nq, nd]: weighted mean over query tokens of the best cosine among each document's tokens (+ [CLS] cosine under HYBRID)"""
        import torch
        qv, _, qw, qc = q
        dv, dm, _, dc = d
        s = torch.einsum("qtd,nsd->qnts", qv, dv.to(qv.dtype))
        if SOFT:  # each query token's attention over the document's tokens, then the attention-weighted cosine
            a = (s / SOFT).masked_fill(~dm[None, :, None, :], float("-inf")).softmax(-1)
            s = (a * s).sum(-1)
        else:
            s = s.masked_fill(~dm[None, :, None, :], -2).max(-1).values
        s = (s * qw[:, None, :]).sum(-1)
        return s + qc @ dc.T if HYBRID else s

    def save(self, out):
        import torch
        self.cb.save(out)
        torch.save(self.qw.state_dict(), Path(out) / "qw.pt")


def _block(d, heads):
    import torch

    class Block(torch.nn.Module):
        """pre-norm cross-attention + feed-forward; output projections start at zero, so the block starts as the identity"""

        def __init__(self):
            super().__init__()
            self.l1, self.l2 = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)
            self.att = torch.nn.MultiheadAttention(d, heads, batch_first=True)
            self.ff = torch.nn.Sequential(torch.nn.Linear(d, 4 * d), torch.nn.GELU(), torch.nn.Linear(4 * d, d))
            for z in (self.att.out_proj, self.ff[-1]):
                torch.nn.init.zeros_(z.weight); torch.nn.init.zeros_(z.bias)

        def forward(self, q, kv, kvm):
            q = q + self.att(self.l1(q), kv, kv, key_padding_mask=~kvm, need_weights=False)[0]
            return q + self.ff(self.l2(q))
    return Block()


def inter_module(dev, d=None, heads=2):
    import torch
    d = d or PDIM

    class Inter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.x = torch.nn.ModuleList([_block(d, heads) for _ in range(XLAYERS)])
            self.mem = torch.nn.Linear(2 * d, d)
            self.lw_in = torch.nn.Linear(d + 1, d)
            self.lw = torch.nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True, dropout=0.0)
            self.lw_out = torch.nn.Linear(d, 1)
            torch.nn.init.zeros_(self.lw_out.weight); torch.nn.init.zeros_(self.lw_out.bias)
    return Inter().to(dev)


def _pool(v, m):
    import torch
    return torch.nn.functional.normalize((v * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1), dim=-1)


def _iscores(model, inter, scale, anc, ev, Q, D, pos, docs, F=None, fpos=None, Lab=None, lpos=None):
    """row 211: [b, max options] scores with the interaction layers. Q = model.vecs of the anchors' queries; D = (token vectors, mask, -, cls)
    of the documents, pos[text] -> row; F / Lab = pooled vectors of filing texts / category labels for memory tokens (fpos, lpos rows)"""
    import torch
    qv, qm, qw, qc = Q
    rows_a, rows_d, rows_o, nopt = [], [], [], max(len(e["state"]) for e in anc)
    for k, e in enumerate(anc):
        for o, c in enumerate(e["state"]):
            rows_a.append(k); rows_d.append(pos[e["d"][c]]); rows_o.append(o)
    ra, rd = torch.tensor(rows_a, device=model.dev), torch.tensor(rows_d, device=model.dev)
    kv, kvm = D[0][rd].float(), D[1][rd]
    if "m" in INTERACT:
        mem = torch.zeros(len(anc), NMEM, kv.shape[-1], device=model.dev)
        mm = torch.zeros(len(anc), NMEM, dtype=torch.bool, device=model.dev)
        for k, e in enumerate(anc):
            js = [j for j in e.get("nb", [])[:NMEM]]
            if js:
                f = torch.cat([F[[fpos[e["_ev"][j]["text"]] for j in js]], Lab[[lpos[e["_ev"][j]["label"]] for j in js]]], -1)
                mem[k, :len(js)] = inter.mem(f); mm[k, :len(js)] = True
        kv, kvm = torch.cat([kv, mem[ra]], 1), torch.cat([kvm, mm[ra]], 1)
    q = qv[ra]
    for blk in inter.x:
        q = blk(q, kv, kvm)
    q = torch.nn.functional.normalize(q, dim=-1)
    dv, dm = D[0][rd].float(), D[1][rd]
    sim = torch.einsum("ptd,psd->pts", q, dv)
    if SOFT:
        a = (sim / SOFT).masked_fill(~dm[:, None, :], float("-inf")).softmax(-1)
        sim = (a * sim).sum(-1)
    else:
        sim = sim.masked_fill(~dm[:, None, :], -2).max(-1).values
    sc = (sim * qw[ra]).sum(-1)
    if HYBRID:
        sc = sc + (qc[ra] * D[3][rd]).sum(-1)
    sc = scale * sc
    S = torch.full((len(anc), nopt), float("-inf"), device=model.dev)
    S = S.index_put((ra, torch.tensor(rows_o, device=model.dev)), sc)
    if "l" in INTERACT:
        feat = torch.zeros(len(anc), nopt, kv.shape[-1] + 1, device=model.dev)
        feat = feat.index_put((ra, torch.tensor(rows_o, device=model.dev)), torch.cat([_pool(dv, dm), (sc / scale)[:, None]], -1))
        ok = torch.isfinite(S)
        h = inter.lw(inter.lw_in(feat), src_key_padding_mask=~ok)
        S = torch.where(ok, S + inter.lw_out(h).squeeze(-1), S)
    return S


def prepare(b, m1, cache):
    """a budget's events (hist_encoder.events) with each event's query text and its day's category documents:
    (events, docs) where docs[day] = {category id: document text} over the visible categories"""
    ev = H.events(b)
    if CTX or MODE == "mml" or "m" in INTERACT:
        nb = H._neighbours(H._embed(m1, [e["text"] for e in ev], cache), ev, max(NB, NCAND if MODE == "mml" else 0, NMEM if "m" in INTERACT else 0))
        for e, n in zip(ev, nb):
            e["nb"] = [j for j, _ in n]
    if CTX:
        for e, n in zip(ev, nb):
            e["q"] = e["text"] + (" || " + "; ".join(f'{ev[j]["payee"]} ${ev[j]["amt"]:.0f} -> {ev[j]["label"]}' for j, _ in n[:NB]) if n else "")
    else:
        for e in ev:
            e["q"] = e["text"]
    docs, recent, i = {}, defaultdict(list), 0
    while i < len(ev):
        d = ev[i]["day"]
        docs[d] = {c: lab + " | " + (" | ".join(f"{p} ${a:.0f}" for p, a in recent[c]) if recent[c] else "nothing filed yet")
                   for c, lab in ((c, ev[i]["labels"][c]) for c in ev[i]["state"])}
        j = i
        while j < len(ev) and ev[j]["day"] == d:
            j += 1
        for e in ev[i:j]:
            recent[e["gold"]] = [(e["payee"], e["amt"])] + [x for x in recent[e["gold"]] if x[0] != e["payee"]][:M - 1]
        i = j
    for e in ev:  # each event carries its day's documents and its budget, so one step can mix households (GROUPS)
        e["d"], e["_ev"] = docs[e["day"]], ev
    return ev, docs


def _score(model, scale, qtexts, dtexts):
    """[len(qtexts), len(dtexts)] scaled late-interaction scores"""
    return scale * model.score(model.vecs(qtexts), model.vecs(dtexts))


def _cands(e, ev):
    """MODE=mml: the candidates of one transaction, (text, category id): its NCAND nearest earlier filings by hist_knn_v1 (in visible
    categories) and every visible category's "Group: Name", so each option has at least one"""
    out = [(ev[j]["text"], ev[j]["gold"]) for j in e["nb"][:NCAND] if ev[j]["gold"] in e["state"]]
    return out + [(e["labels"][c], c) for c in e["state"]]


def _mml_logits(S, cand, anc):
    """category logits [b, max options] from candidate scores S [b, all texts]: log of the summed probability mass of each category's
    candidates (BELXTR's MML); cand[k] = list of (column, option index)"""
    import torch
    nopt = max(len(e["state"]) for e in anc)
    out = torch.full((len(anc), nopt), float("-inf"), device=S.device)
    for k, cs in enumerate(cand):
        col = torch.tensor([c for c, _ in cs], device=S.device)
        opt = torch.tensor([o for _, o in cs], device=S.device)
        v = S[k, col]
        mx = v.max()
        w = torch.zeros(nopt, device=S.device).index_add_(0, opt, (v - mx).exp())
        out[k] = torch.where(w > 0, w.clamp(min=1e-30).log() + mx, out[k])
    return out


def _loss(S, tgt):
    import torch
    lp = torch.log_softmax(S.float(), -1)
    ce = torch.nn.functional.nll_loss(lp, tgt)
    p = lp.exp()
    y = torch.nn.functional.one_hot(tgt, S.shape[1]).float()
    return ce + BRIER * ((p - y) ** 2).sum(-1).mean(), ce


def train():
    import torch
    from ai_experiments.licences import open_licence
    out = H.ENC / f"li_r{ROW}_{ARM}"
    if not Path(BASE).exists():
        open_licence(BASE)
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    m1, cache = (H._model(H.OUT1), {}) if CTX or MODE == "mml" or "m" in INTERACT else (None, None)
    data = []
    for n, b in enumerate(households("train", range(HOUSEHOLDS))):
        data.append(prepare(b, m1, cache))
        if cache is not None and n % 50 == 0:
            cache.clear()
    del m1
    print(f"arm {ARM}: {len(data)} households prepared ({time.time() - t0:.0f}s); query e.g. {data[0][0][50]['q']!r}; "
          f"document e.g. {next(iter(data[0][1][60].values()))!r}", flush=True)
    model = LI(BASE, train=True)
    scale = torch.nn.Parameter(torch.load(Path(BASE) / "scale.pt", map_location="cpu").to(model.dev) if (Path(BASE) / "scale.pt").exists() else torch.tensor(20.0, device=model.dev))
    inter = inter_module(model.dev) if INTERACT else None
    groups = [{"params": model.params()}, {"params": [scale], "lr": 1e-2}] + ([{"params": list(inter.parameters()), "lr": 5e-4}] if inter else [])
    opt = torch.optim.AdamW(groups, lr=LR)
    warm = int(0.05 * STEPS)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (STEPS - s) / (STEPS - warm)))
    t0, run, k = time.time(), defaultdict(float), 0
    for s in range(STEPS):
        anc = []
        for _ in range(GROUPS):  # GROUPS household windows per step (row 210 speed: one window of ~28 transactions left an H100 idle)
            ev, docs = data[rng.randrange(len(data))]
            d0 = rng.randint(1, max(1, ev[-1]["day"] - WINDOW))
            w = [e for e in ev if d0 <= e["day"] < d0 + WINDOW and e["gold"] in e["state"] and len(e["state"]) > 1]
            anc += rng.sample(w, min(B, len(w)))
        if not anc:
            continue
        dtexts, idx = [], {}

        def col(t):
            if t not in idx:
                idx[t] = len(dtexts); dtexts.append(t)
            return idx[t]
        if INTERACT:
            for e in anc:
                for c in e["state"]:
                    col(e["d"][c])
            Q, Dv = model.vecs([e["q"] for e in anc]), model.vecs(dtexts)
            F = fpos = Lab = lpos = None
            if "m" in INTERACT:
                ft = list(dict.fromkeys(e["_ev"][j]["text"] for e in anc for j in e.get("nb", [])[:NMEM]))
                lt = list(dict.fromkeys(e["_ev"][j]["label"] for e in anc for j in e.get("nb", [])[:NMEM]))
                fpos, lpos = {t: i for i, t in enumerate(ft)}, {t: i for i, t in enumerate(lt)}
                F = _pool(*model.vecs(ft)[:2]) if ft else None
                Lab = _pool(*model.vecs(lt)[:2]) if lt else None
            Sg = _iscores(model, inter, scale, anc, ev, Q, Dv, idx, docs, F, fpos, Lab, lpos)
        elif MODE == "mml":
            cand = []
            for e in anc:
                oi = {c: o for o, c in enumerate(e["state"])}
                cand.append([(col(t), oi[c]) for t, c in _cands(e, e["_ev"])])
            S = _score(model, scale, [e["q"] for e in anc], dtexts)
            Sg = _mml_logits(S, cand, anc)
        else:
            cols = [[col(e["d"][c]) for c in e["state"]] for e in anc]
            S = _score(model, scale, [e["q"] for e in anc], dtexts)                 # [b, all documents in the window]
            w = max(map(len, cols))
            ix = torch.tensor([c + [c[0]] * (w - len(c)) for c in cols], device=model.dev)
            valid = torch.tensor([[k_ < len(c) for k_ in range(w)] for c in cols], device=model.dev)
            Sg = S.gather(1, ix).masked_fill(~valid, float("-inf"))
        tgt = torch.tensor([list(e["state"]).index(e["gold"]) for e in anc], device=model.dev)
        loss, ce = _loss(Sg, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.params() + (list(inter.parameters()) if inter else []), 1.0)
        opt.step(); sched.step()
        run["loss"] += loss.item(); run["ce"] += ce.item(); run["acc"] += (Sg.argmax(1) == tgt).float().mean().item(); k += 1
        if (s + 1) % 200 == 0:
            print(f"  step {s + 1}/{STEPS} " + " ".join(f"{a} {v / k:.3f}" for a, v in run.items()) + f" scale {scale.item():.1f} ({time.time() - t0:.0f}s)", flush=True)
            run.clear(); k = 0
    model.save(out)
    torch.save(scale.detach().cpu(), out / "scale.pt")
    if inter:
        torch.save(inter.state_dict(), out / "inter.pt")
    (out / "li_config.json").write_text(json.dumps(dict(ARM=ARM, BASE=BASE, CTX=CTX, NB=NB, M=M, B=B, WINDOW=WINDOW, STEPS=STEPS,
                                                         HOUSEHOLDS=HOUSEHOLDS, BRIER=BRIER, LR=LR, SEED=SEED, GROUPS=GROUPS, HYBRID=HYBRID, QW=QW, SOFT=SOFT, MODE=MODE, NCAND=NCAND,
                                                         INTERACT=INTERACT, NMEM=NMEM, XLAYERS=XLAYERS, ROW=ROW, PDIM=PDIM, PROJ=PROJ,
                                                         PROJ_INIT=PROJ_INIT, LOWER=LOWER, POOL=POOL, MAXLEN=model.cb.maxlen), indent=1))
    print(f"-> {out} (train {time.time() - t0:.0f}s)", flush=True)


def iscores(model, inter, scale, ev, docs):
    """row 211's reader: documents, filings and labels encoded once per budget; the interaction layers per chunk of transactions"""
    import torch

    def enc(texts):
        parts = [model.vecs(texts[a:a + 512]) for a in range(0, len(texts), 512)]
        L = max(p[0].shape[1] for p in parts)
        return (torch.cat([torch.nn.functional.pad(p[0], (0, 0, 0, L - p[0].shape[1])) for p in parts]).to(torch.bfloat16),
                torch.cat([torch.nn.functional.pad(p[1], (0, L - p[1].shape[1])) for p in parts]), None, torch.cat([p[3] for p in parts]))
    with torch.no_grad():
        uniq = list(dict.fromkeys(t for d in docs.values() for t in d.values()))
        D = enc(uniq)
        F = fpos = Lab = lpos = None
        if "m" in INTERACT:
            ft = list(dict.fromkeys(e["text"] for e in ev)); lt = list(dict.fromkeys(e["label"] for e in ev))
            fpos, lpos = {t: i for i, t in enumerate(ft)}, {t: i for i, t in enumerate(lt)}
            Fe, Le = enc(ft), enc(lt)
            F, Lab = _pool(Fe[0].float(), Fe[1]), _pool(Le[0].float(), Le[1])
        pos = {t: k for k, t in enumerate(uniq)}
        out = []
        for a in range(0, len(ev), 64):
            chunk = ev[a:a + 64]
            S = _iscores(model, inter, scale, chunk, ev, model.vecs([e["q"] for e in chunk]), D, pos, docs, F, fpos, Lab, lpos)
            out += [dict(zip(e["state"], S[k, :len(e["state"])].tolist())) for k, e in enumerate(chunk)]
    return out


def scores(model, scale, ev, docs):
    """per event, the scaled score of every visible category (dict category id -> score); documents encoded once per budget"""
    import torch
    if MODE == "mml":
        cands = [_cands(e, ev) for e in ev]
        uniq = list(dict.fromkeys(t for cs in cands for t, _ in cs))
    else:
        uniq = list(dict.fromkeys(t for d in docs.values() for t in d.values()))
    pos = {t: k for k, t in enumerate(uniq)}
    with torch.no_grad():
        parts = [model.vecs(uniq[a:a + 512]) for a in range(0, len(uniq), 512)]
        L = max(p[0].shape[1] for p in parts)
        D = (torch.cat([torch.nn.functional.pad(p[0], (0, 0, 0, L - p[0].shape[1])) for p in parts]).to(torch.bfloat16),
             torch.cat([torch.nn.functional.pad(p[1], (0, L - p[1].shape[1])) for p in parts]), None,
             torch.cat([p[3] for p in parts]))
        out = []
        for a in range(0, len(ev), 256):
            chunk = ev[a:a + 256]
            q = model.vecs([e["q"] for e in chunk])
            for k, e in enumerate(chunk):
                ids = list(e["state"])
                if MODE == "mml":
                    opt = {c: o for o, c in enumerate(ids)}
                    cs = [(pos[t], opt[c]) for t, c in cands[a + k]]
                    cix = torch.tensor([c for c, _ in cs], device=model.dev)
                    s = scale * model.score(tuple(x[k:k + 1] for x in q), (D[0][cix], D[1][cix], None, D[3][cix]))
                    s = _mml_logits(s, [[(i, o) for i, (_, o) in enumerate(cs)]], [e])[0]
                else:
                    cix = torch.tensor([pos[e["d"][c]] for c in ids], device=model.dev)
                    s = scale * model.score(tuple(x[k:k + 1] for x in q), (D[0][cix], D[1][cix], None, D[3][cix]))[0]
                out.append(dict(zip(ids, s.tolist())))
    return out


def read():
    import torch
    budget = os.environ.get("READ", "households") == "budget"
    if budget:
        import real_budget_eval as RB
        budgets = [json.loads(RB.CACHE.read_text())["budget"]]
    else:
        a, z = map(int, os.environ.get("TEST_SEEDS", "100000-100049").split("-"))
        budgets = list(households("test", range(a, z + 1)))
    print(f"\n**{'owner budget' if budget else f'{len(budgets)} held-out households'}: % top-1 / top-3 / top-10; Brier and ECE of the softmax**\n")
    print("| model | all | first-time payee | Brier | ECE (top choice) |\n|---|---|---|---|---|")
    for arm in os.environ["ARMS"].split(","):
        d = H.ENC / (arm if arm.startswith("li_r") else f"li_r{ROW}_{arm}")
        cfg = json.loads((d / "li_config.json").read_text())
        global CTX, NB, M, HYBRID, QW, SOFT, MODE, NCAND, INTERACT, NMEM, XLAYERS, PDIM, PROJ, PROJ_INIT, LOWER, POOL, MAXLEN
        CTX, NB, M, HYBRID, QW = cfg["CTX"], cfg["NB"], cfg["M"], cfg.get("HYBRID", 0), cfg.get("QW", 0)
        SOFT, MODE, NCAND = cfg.get("SOFT", 0), cfg.get("MODE", "doc"), cfg.get("NCAND", 50)
        INTERACT, NMEM, XLAYERS = cfg.get("INTERACT", ""), cfg.get("NMEM", 10), cfg.get("XLAYERS", 2)
        PDIM, PROJ, PROJ_INIT, LOWER = cfg.get("PDIM", 128), cfg.get("PROJ", "linear"), cfg.get("PROJ_INIT", 1), cfg.get("LOWER", 0)
        POOL, MAXLEN = cfg.get("POOL", "cls"), cfg.get("MAXLEN", 96)  # models saved before row 209 were trained at 96 tokens
        m1, cache = (H._model(H.OUT1), {}) if CTX or MODE == "mml" or "m" in INTERACT else (None, None)
        model = LI(d)
        model.cb.enc.eval()
        scale = torch.load(d / "scale.pt", map_location="cpu").to(model.dev)
        inter = None
        if INTERACT:
            inter = inter_module(model.dev); inter.load_state_dict(torch.load(d / "inter.pt", map_location=model.dev)); inter.eval()
        ranks, brier, conf, hit = defaultdict(list), [], [], []
        for b in budgets:
            ev, docs = prepare(b, m1, cache)
            for e, sc in zip(ev, iscores(model, inter, scale, ev, docs) if inter else scores(model, scale, ev, docs)):
                if e["gold"] not in e["state"]:
                    continue
                v = np.array(list(sc.values()))
                g = v[list(sc).index(e["gold"])]
                rk = int((v >= g).sum())  # ties count against the gold
                for grp in ("all",) + (() if e["seen"] else ("first-time",)):
                    ranks[grp].append(rk)
                p = np.exp(v - v.max()); p /= p.sum()
                y = np.array([c == e["gold"] for c in sc], dtype=float)
                brier.append(((p - y) ** 2).sum()); conf.append(p.max()); hit.append(rk == 1)
        conf, hit = np.array(conf), np.array(hit)
        bins = np.minimum((conf * 10).astype(int), 9)
        ece = sum(abs(conf[bins == k].mean() - hit[bins == k].mean()) * (bins == k).mean() for k in range(10) if (bins == k).any())
        cells = [" / ".join(f"{100 * (np.array(ranks[g]) <= k).mean():.1f}" for k in (1, 3, 10)) + f" (n={len(ranks[g])})" for g in ("all", "first-time")]
        print(f"| {d.name} | " + " | ".join(cells) + f" | {np.mean(brier):.3f} | {100 * ece:.1f} |", flush=True)
        del model


if __name__ == "__main__":
    {"train": train, "read": read}[sys.argv[1]]()
