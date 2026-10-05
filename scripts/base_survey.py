"""PLAN step 209: licence, size and tokenizer survey of candidate base encoders for the history encoders and the late-interaction
decision model (li_decider.py). Hub metadata, configs and tokenizers only: no weights are downloaded.
  survey   per model: licence and every base's licence (ai_experiments.licences.open_licence), parameters (safetensors metadata),
           hidden size, layers, cased or not, max length, pooling (sentence-transformers 1_Pooling or the ColBERT/PyLate config)
  tokens   per tokenizer: tokens per string on 30 synthetic bank strings (shared-world v4 households, test seed 100000), and how
           amounts and ALL-CAPS payees split
usage: SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 uv run python scripts/base_survey.py survey|tokens
"""
import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

MODELS = ["BAAI/bge-small-en-v1.5", "answerdotai/answerai-colbert-small-v1", "mixedbread-ai/mxbai-edge-colbert-v0-17m",
          "mixedbread-ai/mxbai-edge-colbert-v0-32m", "jhu-clsp/ettin-encoder-17m", "jhu-clsp/ettin-encoder-32m", "jhu-clsp/ettin-encoder-68m",
          "jhu-clsp/ettin-encoder-150m", "jhu-clsp/ettin-encoder-400m", "jhu-clsp/ettin-encoder-1b", "answerdotai/ModernBERT-base",
          "answerdotai/ModernBERT-large", "Alibaba-NLP/gte-modernbert-base", "lightonai/GTE-ModernColBERT-v1", "Qwen/Qwen3-Embedding-0.6B",
          "Qwen/Qwen3-Embedding-4B"]
MODELS = os.environ.get("MODELS", ",".join(MODELS)).split(",")


def _json(repo, name):
    from huggingface_hub import hf_hub_download
    try:
        return json.loads(Path(hf_hub_download(repo, name)).read_text())
    except Exception:
        return None


def _pooling(repo):
    pc = _json(repo, "1_Pooling/config.json")
    if pc:
        return "+".join(k.replace("pooling_mode_", "").replace("_token", "") for k, v in pc.items() if k.startswith("pooling_mode") and v) or "?"
    mods = _json(repo, "modules.json") or []
    if any("Dense" in m.get("type", "") for m in mods) or _json(repo, "artifact.metadata") or _json(repo, "config_sentence_transformers.json"):
        return "tokens (ColBERT)" if "colbert" in repo.lower() else "?"
    return "none (MLM)"


def survey():
    from huggingface_hub import model_info
    from transformers import AutoConfig, AutoTokenizer
    from ai_experiments.licences import open_licence
    print("| model | licence | bases (licence) | open_licence | params | hidden | layers | cased | max len | pooling |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        info = model_info(m, files_metadata=True)
        cd = info.card_data or {}
        base = cd.get("base_model") or []
        base = [base] if isinstance(base, str) else list(base)
        blic = []
        for x in base:
            try:
                blic.append(f"{x} ({(model_info(x).card_data or {}).get('license')})")
            except Exception as e:
                blic.append(f"{x} (? {type(e).__name__})")
        try:
            with redirect_stdout(io.StringIO()):
                open_licence(m)
            ok = "pass"
        except AssertionError as e:
            ok = "FAIL: " + str(e).split(":")[0]
        params = info.safetensors.total if info.safetensors else None
        cfg = AutoConfig.from_pretrained(m, trust_remote_code=False)
        tok = AutoTokenizer.from_pretrained(m)
        cased = tok.tokenize("PAYPAL") != tok.tokenize("paypal")
        mx = getattr(cfg, "max_position_embeddings", None)
        tml = tok.model_max_length if tok.model_max_length < 1e7 else None
        pool = _pooling(m)
        if "Qwen3-Embedding" in m:
            pool = f"last token ({tok.padding_side} pad)"
        print(f"| {m} | {cd.get('license')} | {'; '.join(blic) or '-'} | {ok} | {params / 1e6:.0f}M | {cfg.hidden_size} | "
              f"{cfg.num_hidden_layers} | {'cased' if cased else 'uncased'} | {min(x for x in (mx, tml) if x) if (mx or tml) else '?'} | {pool} |"
              if params else f"| {m} | {cd.get('license')} | {'; '.join(blic)} | {ok} | ? | {cfg.hidden_size} | | | | {pool} |", flush=True)


def _strings():
    import hist_encoder as H
    from two_tower import households
    ev = H.events(next(iter(households("test", [100000]))))
    seen, out = set(), []
    for e in ev:
        if e["payee"] not in seen:
            seen.add(e["payee"]); out.append(e["text"])
        if len(out) == 30:
            break
    return out


def tokens():
    import numpy as np
    from transformers import AutoTokenizer
    texts = _strings()
    print("strings, e.g.:", *texts[:6], sep="\n  ")
    probes = ["$12.47", "$1,284.00", "STARBUCKS", "Starbucks", "starbucks", "SQ *BLUE BOTTLE COFFEE", "AMZN MKTP US*2K4"]
    seen = {}
    print("\n| tokenizer (models) | mean tokens | max | lowercased mean | " + " | ".join(probes[:3]) + " | STARBUCKS / Starbucks / starbucks |")
    print("|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        tok = AutoTokenizer.from_pretrained(m)
        key = tuple(tok.tokenize(t).__repr__() for t in texts[:5])
        if key in seen:
            print(f"| (same as {seen[key]}) {m} | | | | | | | |")
            continue
        seen[key] = m
        n = np.array([len(tok.tokenize(t)) for t in texts])
        nl = np.array([len(tok.tokenize(t.lower())) for t in texts])
        sp = lambda s: " ".join(tok.tokenize(s)).replace("Ġ", "_").replace("|", "¦")
        print(f"| {m} | {n.mean():.1f} | {n.max()} | {nl.mean():.1f} | {sp(probes[0])} | {sp(probes[1])} | {sp(probes[2])} | "
              f"{len(tok.tokenize(probes[2]))} / {len(tok.tokenize(probes[3]))} / {len(tok.tokenize(probes[4]))} |", flush=True)
        if os.environ.get("SHOW"):
            for t in texts[:3]:
                print("   ", sp(t))


if __name__ == "__main__":
    {"survey": survey, "tokens": tokens}[sys.argv[1]]()
