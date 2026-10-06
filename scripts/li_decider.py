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
  read    % top-1 / top-3 / top-10, all and first-time payee, and the Brier score / ECE of the top choice, per ARMS (READ=households | budget | blind2: blind_v2 users as budgets, row 217)
env: ARM, BASE (hist_colbert_v1: row 195's ColBERT, or any encoder), CTX (0), NB (5), M (8), B (32), WINDOW (7), STEPS (3000),
     HOUSEHOLDS (200), BRIER (1), LR (5e-5), SEED (0), HYBRID (0), QW (0), SOFT (0: MaxSim; tau_a for UWE's soft interaction),
     MODE (doc: one document per category | mml: per-filing candidates, BELXTR | mmld: per-filing candidates + category documents), NCAND (50), TEST_SEEDS (100000-100049), ARMS;
     row 209 (any base): PDIM (128), PROJ (linear | res), PROJ_INIT (1), LOWER (0), POOL (cls | mean | last), MAXLEN (default min(96, the base's): row 210's length).
usage: SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 ARM=a0 uv run python scripts/li_decider.py train
"""
import json
import os
import random
from array import array
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
BATCHED_READ = int(os.environ.get("BATCHED_READ", "1"))
FAST_ISCORES, ICHUNK = int(os.environ.get("FAST_ISCORES", "1")), int(os.environ.get("ICHUNK", "128"))  # row 211's vectorised reader  # 0: the per-transaction reader (to check the batched one against)
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
LIST_TS = (0.5, 0.2, 0.1, 0.05)  # confidence thresholds for suggestion lists in read()
EXTRA = dict(AGO=0, REC=0, QFMT="rows", RENAME=0.0, OVERRIDE=0.0, PPRIOR=0, PTAU=0.9, CREC=0, CAGO=0, PREFIX=0, QPREFIX="[Q] ", DPREFIX="[D] ")  # settings added after row 211 (name -> default); load() restores them from li_config.json
AGO, REC, QFMT = int(os.environ.get("AGO", "0")), int(os.environ.get("REC", "0")), os.environ.get("QFMT", "rows")  # row 217: _query
RENAME = float(os.environ.get("RENAME", "0"))  # row 218: _renamed (training only)
OVERRIDE = float(os.environ.get("OVERRIDE", "0"))  # row 218: _override (training only)
PPRIOR = int(os.environ.get("PPRIOR", "0"))  # row 218: the payee-history prior (_prior), with a learned weight ALPHA: 1 same payee_key, 2 neighbours at cosine >= PTAU
PTAU = float(os.environ.get("PTAU", "0.9"))
GC = int(os.environ.get("GC", "0"))  # gradient checkpointing in training
# row 222 (ColBERT-Zero, 2602.16609: stripping the base's prompts in fine-tuning cost it): PREFIX=1 puts the base's query / document
# markers in front of every query / document text (mxbai-edge-colbert: "[Q] " / "[D] ", from its config_sentence_transformers.json)
PREFIX, QPREFIX, DPREFIX = int(os.environ.get("PREFIX", "0")), os.environ.get("QPREFIX", "[Q] "), os.environ.get("DPREFIX", "[D] ")
# row 223 (distilling decider; ColBERT-Zero 2602.16609, LITE 2406.17968): TEACHER = decider's per-item scores (exp_decision_models per_item
# jsonl: id, sum_lp per option) on the items of TEACHER_ITEMS (data/processed, realstyle items: id "RS:<seed>:<txn8>", options); KD_SEEDS the
# training-world households they come from (added to the training data); each step a share KDP of the anchors is drawn from teacher-scored
# transactions, and their loss adds KDW x the cross-entropy against decider's distribution over the household's options (KDT temperature)
TEACHER, TEACHER_ITEMS, KD_SEEDS = os.environ.get("TEACHER", ""), os.environ.get("TEACHER_ITEMS", ""), os.environ.get("KD_SEEDS", "")
KDW, KDP, KDT = float(os.environ.get("KDW", "1")), float(os.environ.get("KDP", "0.5")), float(os.environ.get("KDT", "1"))
CREC, CAGO = int(os.environ.get("CREC", "0")), int(os.environ.get("CAGO", "0"))  # MML candidates: recent filings; candidate ages (_cands)
RCHUNK = int(os.environ.get("RCHUNK", "256"))
MCHUNK = int(os.environ.get("MCHUNK", "32"))  # transactions per chunk in per-filing reads with ages / recent filings / documents  # transactions per scoring chunk in reads (2026-10-06: 512-token queries x ~45 options x 256 ran out of memory on the 3090)
ALPHA = None  # its weight in read (load() sets it)
OPTS = os.environ.get("OPTS", "span")  # each day's options: visible (today's visible categories, rows 210-217) | recent | span (prepared())
SPAN_AFTER = int(os.environ.get("SPAN_AFTER", "0"))  # OPTS=span: days a hidden category stays offered after its last filing


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

    def vecs(self, texts, role=None):
        """token vectors [n, L, PDIM], mask [n, L], query-token weights [n, L] (sum 1 over the mask), normalised pooled state [n, hidden]
        (POOL: [CLS] / masked mean / last non-padding token)"""
        import torch
        if LOWER:
            texts = [t.lower() for t in texts]
        if PREFIX and role:  # ColBERT-Zero (2602.16609): keep the base's own query / document markers ("[Q] " / "[D] " for PyLate models)
            texts = [(QPREFIX if role == "q" else DPREFIX) + t for t in texts]
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


def _iscores(model, inter, scale, anc, ev, Q, D, pos, docs, F=None, fpos=None, Lab=None, lpos=None, pre=None):
    """row 211: [b, max options] scores with the interaction layers. Q = model.vecs of the anchors' queries; D = (token vectors, mask, -, cls)
    of the documents, pos[text] -> row; F / Lab = pooled vectors of filing texts / category labels for memory tokens (fpos, lpos rows).
    pre: the reader's precomputed index tensors for this chunk (ra, ro, rd pair indices; mf, ml, mm memory rows and mask), which replace
    the per-transaction Python loops (2026-10-05: those left the GPU at a third busy and took ~15 min per arm)"""
    import torch
    qv, qm, qw, qc = Q
    nopt = max(len(e["state"]) for e in anc)
    if pre is not None:
        ra, ro, rd = pre["ra"], pre["ro"], pre["rd"]
    else:
        rows_a, rows_d, rows_o = [], [], []
        for k, e in enumerate(anc):
            for o, c in enumerate(e["state"]):
                rows_a.append(k); rows_d.append(pos[e["d"][c]]); rows_o.append(o)
        ra, rd, ro = (torch.tensor(x, device=model.dev) for x in (rows_a, rows_d, rows_o))
    kv, kvm = D[0][rd].float(), D[1][rd]
    if "m" in INTERACT:
        if pre is not None:
            mm = pre["mm"]
            mem = inter.mem(torch.cat([F[pre["mf"]], Lab[pre["ml"]]], -1)) * mm[..., None]
        else:
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
    S = S.index_put((ra, ro), sc)
    if "l" in INTERACT:
        feat = torch.zeros(len(anc), nopt, kv.shape[-1] + 1, device=model.dev)
        feat = feat.index_put((ra, ro), torch.cat([_pool(dv, dm), (sc / scale)[:, None]], -1))
        ok = torch.isfinite(S)
        h = inter.lw(inter.lw_in(feat), src_key_padding_mask=~ok)
        S = torch.where(ok, S + inter.lw_out(h).squeeze(-1), S)
    return S


_M1 = None
KNB = 50  # neighbours kept per event in the cache: the most any setting reads (NB, NCAND, NMEM)


def _nbrs(b, ev, m1, cache, cos=False):
    """each event's KNB nearest earlier-day events by the neighbour encoder (hist_knn_v1), as index lists (cos=True: also their cosines,
    float16 arrays, for PPRIOR=2). Cached on disk for synthetic budgets in data/interim/li_nb, one int32 array per budget keyed by its
    events and the encoder's weights, the cosines beside it (<key>_cos.npy, added 2026-10-06; built when first asked for) (2026-10-06: this
    replaces row 210's cache of the whole prepared budget, data/interim/li_prep, 23 GB for ~1,000 budgets and rebuilt for every query
    setting; the neighbours are the only slow part). The owner's budget is never cached here (its id is not a synthetic one)."""
    import hashlib
    from ai_experiments.paths import ROOT
    f = None
    if str(b.get("id", "")).startswith(("realstyle-", "blind2-", "rational-")) and os.environ.get("PREP_CACHE", "1") != "0":
        h = hashlib.sha1(f"{b['id']}|{KNB}|{H.AMT_TEXT}".encode())
        for e in ev:
            h.update(f"{e['id']}|{e['day']}|{e['text']}\n".encode())
        for w in sorted(Path(H.OUT1).glob("*.safetensors")):
            h.update(f"{w.name}{w.stat().st_size}{w.stat().st_mtime_ns}".encode())
        f = ROOT / "data" / "interim" / "li_nb" / f"{h.hexdigest()[:20]}.npy"
        fc = f.with_name(f.stem + "_cos.npy")
        if f.exists() and (not cos or fc.exists()):
            nb = [array("i", (j for j in r if j >= 0)) for r in np.load(f).tolist()]
            return (nb, [c[:len(r)] for c, r in zip(np.load(fc), nb)]) if cos else nb
    global _M1
    if m1 is None:  # loaded only when a budget misses the cache
        _M1 = _M1 or H._model(H.OUT1)
        m1 = _M1
    found = H._neighbours(H._embed(m1, [e["text"] for e in ev], cache), ev, KNB)
    nb = [array("i", (j for j, _ in n)) for n in found]
    cs = [np.array([c for _, c in n], dtype=np.float16) for n in found]
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        arr = np.full((len(ev), KNB), -1, dtype=np.int32)
        carr = np.zeros((len(ev), KNB), dtype=np.float16)
        for i, r in enumerate(nb):
            arr[i, :len(r)] = r
            carr[i, :len(r)] = cs[i]
        for path, x in ((f, arr), (fc, carr)):
            tmp = path.with_suffix(f".tmp{os.getpid()}.npy")
            np.save(tmp, x)
            tmp.replace(path)
    return (nb, cs) if cos else nb


def _query(i, ev, start, names=None):
    """the query text of event i. Row 210: "<transaction> || <payee> $<amt> -> <category>; ..." over its NB nearest earlier filings.
    Row 217 (decider's history slice, §99): AGO adds how long ago each filing was; REC adds the household's REC most recent filings
    before the transaction's day, newest first (decider's "Earlier transactions" rows: trips and time routing); QFMT=group lists the
    rows under each category once ("<category>: <payee> $<amt> <ago>, ...; ...") instead of repeating the category per row.
    names: {category id: name} in place of the labels (row 218's rename augmentation)."""
    e = ev[i]
    lab = (lambda j: names.get(ev[j]["gold"], ev[j]["label"])) if names else (lambda j: ev[j]["label"])
    near = list(e.get("nb", [])[:NB]) if CTX else []
    rec = [j for j in range(start[i] - 1, max(-1, start[i] - 1 - REC), -1)] if REC else []

    def ago(j):
        d = (e["date"] - ev[j]["date"]).days
        return f" {d}d ago" if AGO else ""
    if QFMT == "group":
        js = list(dict.fromkeys(near + rec))
        if not js:
            return e["text"]
        by = defaultdict(list)
        for j in js:
            by[lab(j)].append(f'{ev[j]["payee"]} ${ev[j]["amt"]:.0f}{ago(j)}')
        return e["text"] + " || " + "; ".join(f"{lab}: " + ", ".join(v) for lab, v in by.items())
    row = lambda j: f'{ev[j]["payee"]} ${ev[j]["amt"]:.0f} -> {lab(j)}{ago(j)}'
    q = e["text"]
    if near:
        q += " || " + "; ".join(map(row, near))
    if rec:
        q += " || recent: " + "; ".join(map(row, rec))
    return q


def prepared(b, m1, cache, full=True):
    """a budget's events (hist_encoder2.events: hist_encoder.events with dates), (events, docs). full (reads): each event's query text
    e["q"] and its day's documents e["d"] = docs[day] = {category id: document text} over the visible categories. full=False (training,
    2026-10-06: queries and documents for every event of 200 households took 16 GB): only the parts, a per-day snapshot of each
    category's recent payees (e["_snap"], tuples shared between days) and the neighbour indices (compact arrays); _materialise builds
    the texts for the transactions a step uses. docs is None then."""
    ev = H2.events(b)
    vis = list(ev[0]["state"]) if ev else []
    cats = vis + [c for c in (ev[0]["labels"] if ev else {}) if c not in set(vis)]  # every category with a label, visible ones first
    ci = {c: k for k, c in enumerate(cats)}
    # each day's options (hist_encoder.events gives today's visible categories every day; its per-day texts are not read here). 2026-10-06:
    # 7.0% of the owner's items and 5.5% of the synthetic ones are filed to a category hidden today (trips, old phases), which visible-only
    # options could never score, and training never saw a trip purchase. OPTS=recent: + any category filed in the 365 days before the
    # day (decider's rule, real_budget_eval). OPTS=span (owner: "if there is a hidden category, we should offer it as an option for the
    # transactions that had that category"; "the category can appear from the beginning up until its last use"): a hidden category is
    # offered from the budget's start to SPAN_AFTER days after its last filing (0: hidden right after; no hint that a category is about
    # to be used, and its first filing is scorable). Comparisons with decider set SPAN_AFTER=365 so decider's options (used in the past
    # year) are a subset of these. One dict per distinct option set.
    last_use = {}
    for e in ev:
        last_use[e["gold"]] = e["date"]
    memo, last, i = {}, {}, 0
    while i < len(ev):
        j = i
        while j < len(ev) and ev[j]["day"] == ev[i]["day"]:
            j += 1
        dd = ev[i]["date"]
        if OPTS == "recent":
            act = vis + [c for c in cats[len(vis):] if c in last and (dd - last[c]).days <= 365]
        elif OPTS == "span":
            act = vis + [c for c in cats[len(vis):] if c in last_use and (dd - last_use[c]).days <= SPAN_AFTER]
        else:
            act = vis
        st = memo.setdefault(tuple(act), dict.fromkeys(act))
        for e in ev[i:j]:
            e["state"] = st
        for e in ev[i:j]:
            last[e["gold"]] = e["date"]
        i = j
    if CTX or MODE in ("mml", "mmld") or "m" in INTERACT or PPRIOR == 2:
        got = _nbrs(b, ev, m1, cache, cos=PPRIOR == 2)
        nbl, cosl = got if PPRIOR == 2 else (got, None)
        for k, (e, n) in enumerate(zip(ev, nbl)):
            e["nb"] = n
            if PPRIOR == 2:  # the payee's earlier filings = earlier-day neighbours at cosine >= PTAU (row 220: 92.8% found, 93.5% right)
                pp = defaultdict(int)
                for j, c in zip(n, cosl[k]):
                    if c >= PTAU:
                        pp[ev[j]["gold"]] += 1
                e["_pp"] = dict(pp) or None
    start, k = [], 0
    for i, e in enumerate(ev):
        if i and e["day"] != ev[i - 1]["day"]:
            k = i
        start.append(k)
    snap, recent, i = {}, {}, 0
    while i < len(ev):
        d = ev[i]["day"]
        snap[d] = tuple(recent.get(c, ()) for c in cats)
        j = i
        while j < len(ev) and ev[j]["day"] == d:
            j += 1
        for e in ev[i:j]:
            r = recent.get(e["gold"], ())
            recent[e["gold"]] = ((e["payee"], e["amt"]),) + tuple(x for x in r if x[0] != e["payee"])[:M - 1]
        i = j
    if PPRIOR == 1:  # earlier-day filings of the same payee key per category (ai_experiments.payeekey: processor prefixes, codes, store numbers off)
        from ai_experiments.payeekey import payee_key
        seen, i = defaultdict(lambda: defaultdict(int)), 0
        while i < len(ev):
            j = i
            while j < len(ev) and ev[j]["day"] == ev[i]["day"]:
                j += 1
            for e in ev[i:j]:
                e["_pk"] = payee_key(e["payee"]) or e["payee"]
                e["_pp"] = dict(seen[e["_pk"]]) if e["_pk"] in seen else None
            for e in ev[i:j]:
                seen[e["_pk"]][e["gold"]] += 1
            i = j
    for i, e in enumerate(ev):  # each event carries its budget and its day's snapshot, so one step can mix households (GROUPS)
        e["_ev"], e["_snap"], e["_i"], e["_start"], e["_ci"] = ev, snap[e["day"]], i, start, ci
    if not full:
        return ev, None
    first = {}
    for e in ev:
        first.setdefault(e["day"], e)
    docs = {d: {c: _doc(f["labels"][c], snap[d][ci[c]]) for c in f["state"]} for d, f in first.items()}
    for e in ev:
        e["q"], e["d"] = _query(e["_i"], ev, start), docs[e["day"]]
    return ev, docs


def _override(b, rng):
    """row 218 (decider's override episodes, ov10; row 220: on perfectly rational households every remaining error was a merchant the
    household always filed one way while its name suggested another, "Williams Fuel" -> Shopping read as Gas): a share OVERRIDE of the
    household's payees moved, every one of their transactions, to one other category picked at random, so the household's own consistent
    history must beat the name's prior. Returns a shallow copy with new transaction dicts; the neighbour cache stays valid (keyed by
    texts and days, not categories)."""
    groups = {g["id"]: g["name"] for g in b["category_groups"]}
    cats = [c["id"] for c in b["categories"] if not c.get("deleted") and not c.get("hidden") and groups.get(c["category_group_id"]) != "Internal Master Category"]
    payees = sorted({t.get("payee_id") for t in b["transactions"] if t.get("payee_id") and t.get("category_id") in set(cats)})
    move = {}
    for p in payees:
        if rng.random() < OVERRIDE:
            move[p] = rng.choice(cats)
    txs = [dict(t, category_id=move[t["payee_id"]]) if t.get("payee_id") in move and t.get("category_id") in set(cats) else t for t in b["transactions"]]
    return dict(b, transactions=txs)


def _prior(evs, ids, dev):
    """row 218 (owner, 2026-10-06: "How do we fix the override issue?"): [b, w] log((n_c + 0.1) / (n + 0.1 K)) from the payee's earlier
    filings per category (K options); 0 where the payee has none (a constant: changes nothing). Added to the scores times a learned ALPHA
    (0 at step 0), so a payee filed one way many times can beat what its name suggests (row 220's errors); a payee that alternates gets a
    split prior and MaxSim decides between its categories."""
    import torch
    w = max(map(len, ids))
    P = torch.zeros(len(evs), w, device=dev)
    for k, (e, o) in enumerate(zip(evs, ids)):
        pp = e.get("_pp")
        if pp:
            n, K = sum(pp.values()), len(o)
            P[k, :len(o)] = torch.tensor([np.log((pp.get(c, 0) + 0.1) / (n + 0.1 * K)) for c in o], device=dev)
    return P


def _materialise(e, names=None):
    """a training anchor with its texts: the query and its day's documents (names: row 218's renamed categories)"""
    labels = e["labels"]
    lab = (lambda c: names.get(c, labels[c])) if names else (lambda c: labels[c])
    return dict(e, q=_query(e["_i"], e["_ev"], e["_start"], names), d={c: _doc(lab(c), e["_snap"][e["_ci"][c]]) for c in e["state"]})


def _doc(label, recent):
    """a category's document: "<Group: Name> | <payee> $<amt> | ..." over its last M distinct payees, most recent first"""
    return label + " | " + (" | ".join(f"{p} ${a:.0f}" for p, a in recent) if recent else "nothing filed yet")


def _coined(rng, taken):
    """a fresh pronounceable word (exp_categoriser's rename augmentation, row 42)"""
    while True:
        w = "".join(rng.choice("bdfgklmnprstvz") + rng.choice("aeiou") for _ in range(rng.randint(2, 3))).capitalize()
        if w not in taken:
            return w


def _renamed(anc, rng):
    """row 218 (decider's RENAME, §42): each of a window's categories renamed with probability RENAME to a coined word, consistently in
    its documents and the query's history rows, so the model must read what was filed where rather than the category's name; the
    window's anchors materialised"""
    labels = anc[0]["labels"]
    names, taken = {}, set(labels.values())
    for c in labels:
        if rng.random() < RENAME:
            names[c] = _coined(rng, taken); taken.add(names[c])
    return [_materialise(e, names or None) for e in anc]


prepare = prepared


def _score(model, scale, qtexts, dtexts):
    """[len(qtexts), len(dtexts)] scaled late-interaction scores"""
    return scale * model.score(model.vecs(qtexts, "q"), model.vecs(dtexts, "d"))


def _cands(e, ev):
    """MODE=mml: the candidates of one transaction, (text, category id): its NCAND nearest earlier filings by hist_knn_v1 (in visible
    categories) and every visible category's "Group: Name", so each option has at least one"""
    # row 218 (2026-10-06: on the owner's budget history in the query cost 3-7 points, a5's bare query with per-filing candidates read
    # best): CREC adds the household's CREC most recent earlier-day filings as candidates too (the trip signal, without lengthening the
    # query); CAGO appends each candidate filing's age ("| 12d ago")
    js = list(dict.fromkeys(list(e["nb"][:NCAND]) + ([j for j in range(e["_start"][e["_i"]] - 1, max(-1, e["_start"][e["_i"]] - 1 - CREC), -1)] if CREC else [])))
    age = (lambda j: f" | {(e['date'] - ev[j]['date']).days}d ago") if CAGO else (lambda j: "")
    out = [(ev[j]["text"] + age(j), ev[j]["gold"]) for j in js if ev[j]["gold"] in e["state"]]
    # MODE=mmld (owner, 2026-10-06: "I'd much rather that this system infer the similarity of all of those payees somehow, akin to how the
    # decider LLM does it"): each category's document (name + recent payees) instead of its bare name, beside the per-filing candidates,
    # so a payee's variants are matched by the trained encoder filing by filing (no averaging, no key) and categories with no close filing
    # still have their document; with CTX the query carries the dated history (row 217's p2)
    return out + [(e["d"][c] if MODE == "mmld" else e["labels"][c], c) for c in e["state"]]


def _doc_pairs(model, scale, qtexts, dtexts, cols):
    """MODE=doc in training: [b, max options] scores of each query against its own option documents (cols[k] = columns in dtexts);
    padded options -inf"""
    import torch
    qv, qm, qw, qc = model.vecs(qtexts, "q")
    dv, dm, _, dc = model.vecs(dtexts, "d")
    w = max(map(len, cols))
    ix = torch.tensor([c + [c[0]] * (w - len(c)) for c in cols], device=model.dev)
    valid = torch.tensor([[k < len(c) for k in range(w)] for c in cols], device=model.dev)
    sim = torch.einsum("btd,bcsd->bcts", qv, dv[ix].to(qv.dtype))
    m = dm[ix]
    if SOFT:
        a = (sim / SOFT).masked_fill(~m[:, :, None, :], float("-inf")).softmax(-1)
        sim = (a * sim).sum(-1)
    else:
        sim = sim.masked_fill(~m[:, :, None, :], -2).max(-1).values
    S = (sim * qw[:, None, :]).sum(-1)
    if HYBRID:
        S = S + torch.einsum("bh,bch->bc", qc, dc[ix])
    return (scale * S).masked_fill(~valid, float("-inf"))


def _mml_pairs(model, scale, qtexts, dtexts, cand, nopt):
    """MODE=mml/mmld in training: [b, nopt] category logits from each query's own candidates only (cand[k] = list of (column in dtexts,
    option index)); 2026-10-06: scoring every query against every candidate text of the step ran out of memory with row 217's 512-token
    queries (32 x 512 x ~2,000 x 60). Same scores as _score + _mml_logits: MaxSim (or SOFT), query-token weights, HYBRID, x scale,
    then the log of the summed probability mass per category."""
    import torch
    qv, qm, qw, qc = model.vecs(qtexts, "q")
    dv, dm, _, dc = model.vecs(dtexts, "d")
    C = max(map(len, cand))
    cix = torch.tensor([[c for c, _ in cs] + [0] * (C - len(cs)) for cs in cand], device=model.dev)
    oix = torch.tensor([[o for _, o in cs] + [0] * (C - len(cs)) for cs in cand], device=model.dev)
    ok = torch.tensor([[k < len(cs) for k in range(C)] for cs in cand], device=model.dev)
    sim = torch.einsum("btd,bcsd->bcts", qv, dv[cix].to(qv.dtype))
    m = dm[cix]
    if SOFT:
        a = (sim / SOFT).masked_fill(~m[:, :, None, :], float("-inf")).softmax(-1)
        sim = (a * sim).sum(-1)
    else:
        sim = sim.masked_fill(~m[:, :, None, :], -2).max(-1).values
    S = (sim * qw[:, None, :]).sum(-1)
    if HYBRID:
        S = S + torch.einsum("bh,bch->bc", qc, dc[cix])
    S = (scale * S).masked_fill(~ok, float("-inf"))
    mx = S.max(1, keepdim=True).values
    w = (S - mx).exp().masked_fill(~ok, 0.0)
    agg = torch.zeros(len(cand), nopt, device=model.dev).scatter_add(1, oix, w)
    return torch.where(agg > 0, agg.clamp(min=1e-30).log() + mx, torch.full_like(agg, float("-inf")))


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
    assert not RENAME or (MODE in ("doc", "mmld") and not INTERACT), "RENAME renders documents and query rows only"
    t0 = time.time()
    m1, cache = None, {}  # the neighbour encoder loads only if a budget misses the cache (_nbrs)
    data = []
    for n, b in enumerate(households("train", range(HOUSEHOLDS))):
        if OVERRIDE:
            b = _override(b, random.Random(SEED * 7919 + n))
        data.append(prepared(b, m1, cache, full=False))
        if n % 50 == 0:
            cache.clear()
    kd = []  # teacher-scored events (row 223)
    if TEACHER:
        from ai_experiments.paths import PROCESSED, ROOT
        items = {it["id"]: it for it in json.loads((PROCESSED / TEACHER_ITEMS).read_text())["items"]}
        tl = {}
        for r in map(json.loads, open(ROOT / TEACHER if not Path(TEACHER).is_absolute() else TEACHER)):
            if r["id"] in items:
                lp = np.asarray(r["sum_lp"], dtype=np.float64) / KDT
                p_ = np.exp(lp - lp.max()); p_ /= p_.sum()
                tl[r["id"]] = {o.strip(): float(x) for o, x in zip(items[r["id"]]["options"], p_)}
        a_, z_ = map(int, KD_SEEDS.split("-"))
        for b in households("train", range(a_, z_ + 1)):
            if OVERRIDE:
                b = _override(b, random.Random(SEED * 7919 + 10 ** 6 + int(b["id"].rsplit("-", 1)[1])))
            ev_, docs_ = prepared(b, m1, cache, full=False)
            data.append((ev_, docs_))
            hid = b["id"].rsplit("-", 1)[1]
            for e in ev_:
                t = tl.get(f"RS:{hid}:{e['id'][:8]}")
                if t and e["gold"] in e["state"] and len(e["state"]) > 1:
                    e["_t"] = t
                    kd.append(e)
            cache.clear()
        print(f"KD: {len(tl)} teacher-scored items, {len(kd)} matched to events of {z_ - a_ + 1} households", flush=True)
    ex = _materialise(data[0][0][50])
    print(f"arm {ARM}: {len(data)} households prepared ({time.time() - t0:.0f}s); query e.g. {ex['q']!r}; "
          f"document e.g. {next(iter(ex['d'].values()))!r}", flush=True)
    model = LI(BASE, train=True)
    if GC:  # recompute the encoder's activations in backward (2026-10-06: MODE=mmld encodes ~3,000 candidate texts per step; 44 GB on an L40S was not enough)
        model.cb.enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    scale = torch.nn.Parameter(torch.load(Path(BASE) / "scale.pt", map_location="cpu").to(model.dev) if (Path(BASE) / "scale.pt").exists() else torch.tensor(20.0, device=model.dev))
    inter = inter_module(model.dev) if INTERACT else None
    alpha = torch.nn.Parameter(torch.tensor(0.0, device=model.dev))
    assert not PPRIOR or (MODE == "doc" and not INTERACT), "PPRIOR is wired into the document reader only"
    groups = [{"params": model.params()}, {"params": [scale] + ([alpha] if PPRIOR else []), "lr": 1e-2}] + ([{"params": list(inter.parameters()), "lr": 5e-4}] if inter else [])
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
            pick = rng.sample(w, min(B, len(w)))
            anc += _renamed(pick, rng) if RENAME and pick else [_materialise(e) for e in pick]
        if kd and KDP > 0:  # a share KDP of the step's anchors from teacher-scored transactions (each carries its own day's documents)
            nk = int(round(KDP * len(anc))) or 1
            anc = anc[:len(anc) - nk] + [_materialise(e) for e in rng.sample(kd, min(nk, len(kd)))]
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
            Q, Dv = model.vecs([e["q"] for e in anc], "q"), model.vecs(dtexts, "d")
            F = fpos = Lab = lpos = None
            if "m" in INTERACT:
                ft = list(dict.fromkeys(e["_ev"][j]["text"] for e in anc for j in e.get("nb", [])[:NMEM]))
                lt = list(dict.fromkeys(e["_ev"][j]["label"] for e in anc for j in e.get("nb", [])[:NMEM]))
                fpos, lpos = {t: i for i, t in enumerate(ft)}, {t: i for i, t in enumerate(lt)}
                F = _pool(*model.vecs(ft)[:2]) if ft else None
                Lab = _pool(*model.vecs(lt)[:2]) if lt else None
            Sg = _iscores(model, inter, scale, anc, ev, Q, Dv, idx, docs, F, fpos, Lab, lpos)
        elif MODE in ("mml", "mmld"):
            cand = []
            for e in anc:
                oi = {c: o for o, c in enumerate(e["state"])}
                cand.append([(col(t), oi[c]) for t, c in _cands(e, e["_ev"])])
            Sg = _mml_pairs(model, scale, [e["q"] for e in anc], dtexts, cand, max(len(e["state"]) for e in anc))
        else:
            cols = [[col(e["d"][c]) for c in e["state"]] for e in anc]
            # each anchor against its own options only (2026-10-06: anchors from other households and days, as KD's, multiplied the
            # documents, and the all-pairs score [queries x tokens x documents x tokens] ran out of memory); same scores as _score + gather
            Sg = _doc_pairs(model, scale, [e["q"] for e in anc], dtexts, cols)
            if PPRIOR:
                Sg = Sg + alpha * _prior(anc, [list(e["state"]) for e in anc], model.dev)
        tgt = torch.tensor([list(e["state"]).index(e["gold"]) for e in anc], device=model.dev)
        loss, ce = _loss(Sg, tgt)
        if kd:  # soft cross-entropy against decider's distribution over the options it shares with this one (others get 0)
            T = torch.zeros_like(Sg)
            has = []
            for k_, e in enumerate(anc):
                t = e.get("_t")
                if t:
                    v = [t.get(e["labels"][c], 0.0) for c in e["state"]]
                    z = sum(v)
                    if z > 0:
                        T[k_, :len(v)] = torch.tensor(v, device=model.dev) / z; has.append(k_)
            if has:
                hi = torch.tensor(has, device=model.dev)
                kdl = -(T[hi] * torch.log_softmax(Sg[hi].float(), -1).masked_fill(T[hi] == 0, 0.0)).sum(-1).mean()
                loss = loss + KDW * kdl
                run["kd"] += kdl.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.params() + (list(inter.parameters()) if inter else []), 1.0)
        opt.step(); sched.step()
        run["loss"] += loss.item(); run["ce"] += ce.item(); run["acc"] += (Sg.argmax(1) == tgt).float().mean().item(); k += 1
        if (s + 1) % 200 == 0:
            print(f"  step {s + 1}/{STEPS} " + " ".join(f"{a} {v / k:.3f}" for a, v in run.items()) + f" scale {scale.item():.1f}" + (f" alpha {alpha.item():.2f}" if PPRIOR else "") + f" ({time.time() - t0:.0f}s)", flush=True)
            run.clear(); k = 0
    model.save(out)
    torch.save(scale.detach().cpu(), out / "scale.pt")
    if PPRIOR:
        torch.save(alpha.detach().cpu(), out / "pprior.pt")
        print(f"payee prior weight {alpha.item():.3f}", flush=True)
    if inter:
        torch.save(inter.state_dict(), out / "inter.pt")
    (out / "li_config.json").write_text(json.dumps(dict(ARM=ARM, BASE=BASE, CTX=CTX, NB=NB, M=M, B=B, WINDOW=WINDOW, STEPS=STEPS,
                                                         HOUSEHOLDS=HOUSEHOLDS, BRIER=BRIER, LR=LR, SEED=SEED, GROUPS=GROUPS, HYBRID=HYBRID, QW=QW, SOFT=SOFT, MODE=MODE, NCAND=NCAND,
                                                         INTERACT=INTERACT, NMEM=NMEM, XLAYERS=XLAYERS, ROW=ROW, PDIM=PDIM, PROJ=PROJ,
                                                         PROJ_INIT=PROJ_INIT, LOWER=LOWER, POOL=POOL, MAXLEN=model.cb.maxlen,
                                                         OPTS=OPTS, SPAN_AFTER=SPAN_AFTER, TEACHER=TEACHER, KD_SEEDS=KD_SEEDS, KDW=KDW, KDP=KDP, KDT=KDT, **{k: globals()[k] for k in EXTRA}), indent=1))
    print(f"-> {out} (train {time.time() - t0:.0f}s)", flush=True)


def iscores(model, inter, scale, ev, docs):
    """row 211's reader: documents, filings and labels encoded once per budget; the interaction layers per chunk of transactions"""
    import torch

    def enc(texts):
        parts = [model.vecs(texts[a:a + 512], "d") for a in range(0, len(texts), 512)]
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
        if not FAST_ISCORES:  # the per-transaction reference path (identical scores; 1.7x slower on 11,558 transactions)
            for a in range(0, len(ev), 64):
                chunk = ev[a:a + 64]
                S = _iscores(model, inter, scale, chunk, ev, model.vecs([e["q"] for e in chunk], "q"), D, pos, docs, F, fpos, Lab, lpos)
                out += [dict(zip(e["state"], S[k, :len(e["state"])].tolist())) for k, e in enumerate(chunk)]
            return out
        # every index the layers need, built once per budget as arrays; per chunk only tensor gathers run
        import numpy as np
        nopt = np.array([len(e["state"]) for e in ev])
        cix = np.zeros((len(ev), nopt.max()), dtype=np.int64)
        for i, e in enumerate(ev):
            cix[i, :nopt[i]] = [pos[e["d"][c]] for c in e["state"]]
        mf = np.zeros((len(ev), NMEM), dtype=np.int64); ml = np.zeros_like(mf); mmask = np.zeros((len(ev), NMEM), dtype=bool)
        if "m" in INTERACT:
            for i, e in enumerate(ev):
                js = e.get("nb", [])[:NMEM]
                mf[i, :len(js)] = [fpos[ev[j]["text"]] for j in js]; ml[i, :len(js)] = [lpos[ev[j]["label"]] for j in js]
                mmask[i, :len(js)] = True
        cix_t, nopt_t = torch.tensor(cix, device=model.dev), torch.tensor(nopt, device=model.dev)
        mf_t, ml_t, mm_t = (torch.tensor(x, device=model.dev) for x in (mf, ml, mmask))
        for a in range(0, len(ev), ICHUNK):
            chunk = ev[a:a + ICHUNK]
            b = len(chunk)
            C = int(nopt[a:a + b].max())
            valid = torch.arange(C, device=model.dev)[None, :] < nopt_t[a:a + b, None]
            ra, ro = valid.nonzero(as_tuple=True)
            pre = dict(ra=ra, ro=ro, rd=cix_t[a:a + b, :C][ra, ro], mf=mf_t[a:a + b], ml=ml_t[a:a + b], mm=mm_t[a:a + b])
            S = _iscores(model, inter, scale, chunk, ev, model.vecs([e["q"] for e in chunk], "q"), D, pos, docs, F, fpos, Lab, lpos, pre)
            S = S.tolist()
            out += [dict(zip(e["state"], S[k][:nopt[a + k]])) for k, e in enumerate(chunk)]
    return out


def scores(model, scale, ev, docs):
    """per event, the scaled score of every visible category (dict category id -> score); documents encoded once per budget"""
    import torch
    if MODE in ("mml", "mmld") and (CAGO or os.environ.get("MCHUNKED") == "1"):  # only ages make texts unique per transaction
        # 2026-10-06: with candidate ages every filing text is unique to the transaction reading it, so "encode every candidate text of
        # the budget once" held hundreds of thousands of texts (out of memory on an L40S); here each chunk encodes its own candidates
        out = []
        with torch.no_grad():
            for a in range(0, len(ev), MCHUNK):
                chunk = ev[a:a + MCHUNK]
                dt, idx, cand = [], {}, []
                for e in chunk:
                    oi = {c: o for o, c in enumerate(e["state"])}
                    cs = []
                    for t, c in _cands(e, ev):
                        if t not in idx:
                            idx[t] = len(dt); dt.append(t)
                        cs.append((idx[t], oi[c]))
                    cand.append(cs)
                S = _mml_pairs(model, scale, [e["q"] for e in chunk], dt, cand, max(len(e["state"]) for e in chunk)).tolist()
                out += [dict(zip(e["state"], S[k][:len(e["state"])])) for k, e in enumerate(chunk)]
        return out
    if MODE in ("mml", "mmld"):
        cands = [_cands(e, ev) for e in ev]
        uniq = list(dict.fromkeys(t for cs in cands for t, _ in cs))
    else:
        uniq = list(dict.fromkeys(t for d in docs.values() for t in d.values()))
    pos = {t: k for k, t in enumerate(uniq)}
    with torch.no_grad():
        parts = [model.vecs(uniq[a:a + 512], "d") for a in range(0, len(uniq), 512)]
        L = max(p[0].shape[1] for p in parts)
        D = (torch.cat([torch.nn.functional.pad(p[0], (0, 0, 0, L - p[0].shape[1])) for p in parts]).to(torch.bfloat16),
             torch.cat([torch.nn.functional.pad(p[1], (0, L - p[1].shape[1])) for p in parts]), None,
             torch.cat([p[3] for p in parts]))
        out = []
        for a in range(0, len(ev), RCHUNK):
            chunk = ev[a:a + RCHUNK]
            q = model.vecs([e["q"] for e in chunk], "q")
            if MODE not in ("mml", "mmld") and BATCHED_READ:  # all of a chunk's (transaction, option) pairs at once (2026-10-05: per-transaction loop, 8.6 min / 50 households)
                ids = [list(e["state"]) for e in chunk]
                C = max(map(len, ids))
                cix = torch.tensor([[pos[e["d"][c]] for c in o] + [0] * (C - len(o)) for e, o in zip(chunk, ids)], device=model.dev)
                qv, qm, qw, qc = q
                dv, dm = D[0][cix].float(), D[1][cix]                                                     # [b, C, L, d], [b, C, L]
                sim = torch.einsum("btd,bcsd->bcts", qv, dv)
                if SOFT:
                    att = (sim / SOFT).masked_fill(~dm[:, :, None, :], float("-inf")).softmax(-1)
                    sim = (att * sim).sum(-1)
                else:
                    sim = sim.masked_fill(~dm[:, :, None, :], -2).max(-1).values                           # [b, C, t]
                S = (sim * qw[:, None, :]).sum(-1)
                if HYBRID:
                    S = S + torch.einsum("bh,bch->bc", qc, D[3][cix].float())
                S = scale * S
                if PPRIOR:
                    S = S + ALPHA * _prior(chunk, ids, model.dev)
                S = S.tolist()
                out += [dict(zip(o, S[k][:len(o)])) for k, o in enumerate(ids)]
                continue
            for k, e in enumerate(chunk):
                ids = list(e["state"])
                if MODE in ("mml", "mmld"):
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


def load(arm):
    """an arm ready to read: sets this module's settings from its li_config.json and returns run(budget) -> (events, per-event
    {category id: score}); one arm at a time (the settings are module globals). Used by read() and hist_agree.py (ENCS=li_r...)."""
    import torch
    d = H.ENC / (arm if arm.startswith("li_r") else f"li_r{ROW}_{arm}")
    cfg = json.loads((d / "li_config.json").read_text())
    global CTX, NB, M, HYBRID, QW, SOFT, MODE, NCAND, INTERACT, NMEM, XLAYERS, PDIM, PROJ, PROJ_INIT, LOWER, POOL, MAXLEN
    CTX, NB, M, HYBRID, QW = cfg["CTX"], cfg["NB"], cfg["M"], cfg.get("HYBRID", 0), cfg.get("QW", 0)
    SOFT, MODE, NCAND = cfg.get("SOFT", 0), cfg.get("MODE", "doc"), cfg.get("NCAND", 50)
    INTERACT, NMEM, XLAYERS = cfg.get("INTERACT", ""), cfg.get("NMEM", 10), cfg.get("XLAYERS", 2)
    PDIM, PROJ, PROJ_INIT, LOWER = cfg.get("PDIM", 128), cfg.get("PROJ", "linear"), cfg.get("PROJ_INIT", 1), cfg.get("LOWER", 0)
    POOL, MAXLEN = cfg.get("POOL", "cls"), cfg.get("MAXLEN", 96)  # models saved before row 209 were trained at 96 tokens
    for k in EXTRA:  # settings added after row 211 (QUERY etc.): older models were trained with the defaults
        globals()[k] = cfg.get(k, EXTRA[k])
    m1, cache = None, {}
    model = LI(d)
    model.cb.enc.eval()
    scale = torch.load(d / "scale.pt", map_location="cpu").to(model.dev)
    global ALPHA
    ALPHA = torch.load(d / "pprior.pt", map_location="cpu").to(model.dev) if PPRIOR else None
    inter = None
    if INTERACT:
        inter = inter_module(model.dev); inter.load_state_dict(torch.load(d / "inter.pt", map_location=model.dev)); inter.eval()

    def run(b):
        ev, docs = prepared(b, m1, cache)
        return ev, (iscores(model, inter, scale, ev, docs) if inter else scores(model, scale, ev, docs))
    run.name = d.name
    return run


def read():
    budget = os.environ.get("READ", "households") == "budget"
    if budget:
        import real_budget_eval as RB
        budgets = [json.loads(RB.CACHE.read_text())["budget"]]
    elif os.environ.get("READ") == "blind2":  # row 217: blind_v2's 250 users as whole budgets (scripts/blind_budgets.py), a transfer test
        import blind_budgets
        budgets = blind_budgets.budgets()
    elif os.environ.get("READ", "").startswith("rational_"):  # row 220: perfectly rational households, READ=rational_clean | rational_bank
        import rational_budgets
        budgets = rational_budgets.budgets(os.environ["READ"].split("_", 1)[1])
    else:
        a, z = map(int, os.environ.get("TEST_SEEDS", "100000-100049").split("-"))
        budgets = list(households("test", range(a, z + 1)))
    rd = os.environ.get("READ", "households")
    print(f"\n**{'owner budget' if budget else f'{len(budgets)} ' + ('blind_v2 budgets' if rd == 'blind2' else rd + ' households' if rd.startswith('rational') else 'held-out households')}: % top-1 / top-3 / top-5 / top-10 (n); Brier and ECE of the softmax**\n")
    segs = ("known", "new") if rd.startswith("rational") else ("trip",)  # generator marks, scoring only (rational: merchant bought before or not)
    cols = ["model", "all", "first-time payee string"] + list(segs) + ["Brier", "ECE (top choice)"]
    print("| " + " | ".join(cols) + " |\n" + "|---" * len(cols) + "|")
    for arm in os.environ["ARMS"].split(","):
        run = load(arm)
        ranks, brier, conf, hit, lists = defaultdict(list), [], [], [], defaultdict(list)
        for b in budgets:
            # owner, 2026-10-06 (does a recent trip filing carry to the trip's other purchases?): the generators mark trip purchases
            # (realstyle reason ["trip", ...]; blind_v2 Ev.trip, carried by blind_budgets.py); the owner's budget has no marks
            mark = {t["id"]: (t.get("reason") or [None])[0] for t in b["transactions"]}
            ev, sco = run(b)
            for e, sc in zip(ev, sco):
                if e["gold"] not in e["state"]:
                    continue
                v = np.array(list(sc.values()))
                g = v[list(sc).index(e["gold"])]
                rk = int((v >= g).sum())  # ties count against the gold
                for grp in ("all",) + (() if e["seen"] else ("first-time",)) + ((mark[e["id"]],) if mark.get(e["id"]) in segs else ()):
                    ranks[grp].append(rk)
                p = np.exp(v - v.max()); p /= p.sum()
                y = np.array([c == e["gold"] for c in sc], dtype=float)
                brier.append(((p - y) ** 2).sum()); conf.append(p.max()); hit.append(rk == 1)
                # owner, 2026-10-06 ("Maybe we only show options that have above a certain confidence?"): suggestion lists by confidence,
                # every option with p >= t, and the fewest options holding 90% of the mass; (gold in the list, list size)
                pg = p[list(sc).index(e["gold"])]
                for t in LIST_TS:
                    lists[t].append((pg >= t, int((p >= t).sum())))
                ps = np.sort(p)[::-1]
                k90 = int(np.searchsorted(np.cumsum(ps), 0.9) + 1)
                lists["90%"].append((pg >= ps[k90 - 1], k90))
        conf, hit = np.array(conf), np.array(hit)
        bins = np.minimum((conf * 10).astype(int), 9)
        ece = sum(abs(conf[bins == k].mean() - hit[bins == k].mean()) * (bins == k).mean() for k in range(10) if (bins == k).any())
        cells = [" / ".join(f"{100 * (np.array(ranks[g]) <= k).mean():.1f}" for k in (1, 3, 5, 10)) + f" (n={len(ranks[g])})" if ranks[g] else "-"
                 for g in ("all", "first-time") + tuple(segs)]
        print(f"| {run.name} | " + " | ".join(cells) + f" | {np.mean(brier):.3f} | {100 * ece:.1f} |", flush=True)
        print(f"  suggestion lists, {run.name}: " + "; ".join(
            f"{'p >= ' + str(t) if t != '90%' else '90% of the mass'}: right one in the list {100 * np.mean([a for a, _ in v]):.1f}%, "
            f"{np.mean([n for _, n in v]):.2f} shown" + (f", empty {100 * np.mean([n == 0 for _, n in v]):.1f}%" if t != "90%" else "")
            for t, v in lists.items()), flush=True)
        del run


if __name__ == "__main__":
    {"train": train, "read": read}[sys.argv[1]]()
