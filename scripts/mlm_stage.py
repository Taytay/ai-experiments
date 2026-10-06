"""Row 222 / 221 (b) (owner, 2026-10-06: "are we able to use a masked token objective to get it to learn more relationships ... Or are we
simply pushing embeddings closer and further away?"): domain-adaptive masked-token pretraining of an encoder that ships its MLM head
(Ettin, ModernBERT) on bank strings and merchant names, before the contrastive and history stages. The corpus: every `text` (and alias
`kind_text`) in the pair files (merchant DB and Overture names rendered as bank strings by statements.render_v2; held-out names already
excluded) plus the plain names, so the encoder sees run-together, truncated, prefixed and coded forms of the same names. Masking rate 30%
(Ettin's own pretraining rate). Writes OUT (models/encoders/mlm_r222_<ARM>): the masked-LM checkpoint and tokenizer; li_decider and
knowledge_stage load its encoder with AutoModel / SentenceTransformer (the head is ignored).
env: BASE (jhu-clsp/ettin-encoder-32m), PAIRS_FILE (comma list; data/interim/knowledge_pairs_v1.jsonl,data/interim/alias_pairs_v1.jsonl),
     ARM (m1), STEPS (3000), BATCH (256), LR (1e-4; 5% warm-up, linear decay), MASK (0.3), MAXLEN (64), LOWER (1), SEED (0).
usage: ARM=m1 uv run python scripts/mlm_stage.py
"""
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ai_experiments.paths import ROOT  # noqa: E402

BASE = os.environ.get("BASE", "jhu-clsp/ettin-encoder-32m")
FILES = [ROOT / x for x in os.environ.get("PAIRS_FILE", "data/interim/knowledge_pairs_v1.jsonl,data/interim/alias_pairs_v1.jsonl").split(",")]
ARM, STEPS, BATCH = os.environ.get("ARM", "m1"), int(os.environ.get("STEPS", "3000")), int(os.environ.get("BATCH", "256"))
LR, MASK, MAXLEN = float(os.environ.get("LR", "1e-4")), float(os.environ.get("MASK", "0.3")), int(os.environ.get("MAXLEN", "64"))
LOWER, SEED = int(os.environ.get("LOWER", "1")), int(os.environ.get("SEED", "0"))


def corpus():
    texts = set()
    for f in FILES:
        for line in f.open():
            p = json.loads(line)
            texts.update(t for t in (p.get("text"), p.get("kind_text") if p.get("kind") == "alias" else None, p.get("name")) if t)
    texts = sorted(texts)
    return [t.lower() for t in texts] if LOWER else texts


def main():
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
    from ai_experiments.licences import open_licence
    if not Path(BASE).exists():
        open_licence(BASE)
    random.seed(SEED); torch.manual_seed(SEED)
    texts = corpus()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForMaskedLM.from_pretrained(BASE).to(dev)
    model.train()
    coll = DataCollatorForLanguageModeling(tok, mlm=True, mlm_probability=MASK)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    warm = max(1, int(0.05 * STEPS))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (STEPS - s) / (STEPS - warm)))
    print(f"mlm {ARM}: {len(texts)} distinct strings from {[f.name for f in FILES]}, e.g. {random.sample(texts, 4)}; {STEPS} steps x {BATCH} on {dev}",
          flush=True)
    t0, run, k = time.time(), 0.0, 0
    for s in range(STEPS):
        enc = tok(random.sample(texts, BATCH), truncation=True, max_length=MAXLEN)
        b = coll([{"input_ids": ids} for ids in enc["input_ids"]])
        b = {key: v.to(dev) for key, v in b.items()}
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            loss = model(**b).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run += loss.item(); k += 1
        if (s + 1) % 200 == 0:
            print(f"  step {s + 1}/{STEPS} mlm loss {run / k:.3f} ({time.time() - t0:.0f}s)", flush=True); run, k = 0.0, 0
    out = ROOT / "models" / "encoders" / f"mlm_r222_{ARM}"
    model.save_pretrained(out); tok.save_pretrained(out)
    (out / "mlm_config.json").write_text(json.dumps(dict(plan_row=222, base=BASE, files=[str(f.relative_to(ROOT)) for f in FILES], steps=STEPS,
                                                        batch=BATCH, lr=LR, mask=MASK, maxlen=MAXLEN, lower=LOWER, seed=SEED, strings=len(texts),
                                                        train_seconds=round(time.time() - t0)), indent=1))
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
