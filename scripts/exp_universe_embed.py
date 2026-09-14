"""Embedding-model version of the universe ladder: unseen labels via prototypes.

Train all-MiniLM-L6-v2 contrastively so a species NAME embeds near the text of
its attributes (type lore, habitat, weakness). No classification head, no new
tokens. Then evaluate three ways of naming a label:
  canonical   label = canonical type name text        ("Voltrix")
  synonym     label = never-trained synonym/description ("the sparky ones")
  prototype   label = centroid of k user-labeled example species (Timmy's cards)
and for prototypes also the latent partitions (weakness, habitat), where the
right attribute must be inferred from which examples share a tag.
Held-out species (never trained) should sit at chance: a sanity check.
"""
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))
import universe as U  # noqa: E402
from evals.tracker import Run  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS, BS, LR, SEED = 8, 32, 3e-5, 0
OUT = Path(__file__).parent.parent / "results" / "universe_embed.json"
torch.manual_seed(SEED)
rng = random.Random(SEED)

species = U.build()
seen = [s for s in species if not s["heldout"]]
held = [s for s in species if s["heldout"]]
ATTR_TEXT = {
    "type": {t: U.TYPES[t][0] for t in U.TYPE_LIST},
    "weakness": {t: f"Creatures weak to {t}-type attacks." for t in U.TYPE_LIST},
    "habitat": {h: f"Creatures that live in {h} habitats." for h in U.HABITATS},
}

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).cuda()


def embed(texts, grad=False):
    b = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to("cuda")
    with (torch.enable_grad() if grad else torch.no_grad()), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**b).last_hidden_state
        mask = b["attention_mask"].unsqueeze(-1).to(out.dtype)
        return F.normalize((out * mask).sum(1) / mask.sum(1), dim=-1).float()


def acc(pred, gold):
    return round(100 * sum(p == g for p, g in zip(pred, gold)) / len(gold), 1)


def evaluate():
    model.eval()
    r = {}
    for tag, pool in (("seen", seen), ("heldout", held)):
        names = embed([s["name"] for s in pool])
        # canonical + synonym labels (type only; synonyms exist only for types)
        for mode, texts in (("canonical", [U.TYPES[t][0] for t in U.TYPE_LIST]),
                            ("synonym", [U.TYPES[t][1][0] for t in U.TYPE_LIST]),
                            ("synonym2", [U.TYPES[t][1][1] for t in U.TYPE_LIST])):
            labs = embed(texts)
            pred = (names @ labs.T).argmax(1).tolist()
            r[f"type_{mode}_{tag}"] = acc(pred, [U.TYPE_LIST.index(s["type"]) for s in pool])
        # prototype labels: Timmy's k labeled cards per tag, over type / weakness / habitat
        for attr in ("type", "weakness", "habitat"):
            for k in (1, 3):
                hits = []
                counts = {v: sum(s[attr] == v for s in pool) for v in {s[attr] for s in pool}}
                usable = sorted(v for v, c in counts.items() if c > k)  # need k demos + 1 query
                if len(usable) < 3:
                    continue
                for _ in range(300):
                    vals = rng.sample(usable, 3)
                    demos = {v: rng.sample([s for s in pool if s[attr] == v], k) for v in vals}
                    protos = torch.stack([embed([d["name"] for d in demos[v]]).mean(0) for v in vals])
                    qv = rng.choice(vals)
                    q = rng.choice([s for s in pool if s[attr] == qv and s not in demos[qv]])
                    pred = (embed([q["name"]]) @ F.normalize(protos, dim=-1).T).argmax().item()
                    hits.append(vals[pred] == qv)
                r[f"proto_{attr}_k{k}_{tag}"] = round(100 * sum(hits) / len(hits), 1)
    return r


def train():
    pairs = []
    for s in seen:
        for attr in ("type", "weakness", "habitat"):
            pairs += [(s["name"], ATTR_TEXT[attr][s[attr]])] * 2
            pairs += [(f"{s['name']} is a {s['type']}-type creature from {s['region']}.", ATTR_TEXT[attr][s[attr]])]
        pairs += [(s["name"], U.entry(s))] * 2
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train(); t0 = time.time()
    for ep in range(EPOCHS):
        rng.shuffle(pairs)
        for i in range(0, len(pairs), BS):
            chunk = pairs[i:i + BS]
            a = embed([p[0] for p in chunk], grad=True); p = embed([p[1] for p in chunk], grad=True)
            scores = a @ p.T * 20.0
            same = torch.tensor([[x[1] == y[1] for y in chunk] for x in chunk], device="cuda")
            scores = scores.masked_fill(same & ~torch.eye(len(chunk), dtype=torch.bool, device="cuda"), -1e4)
            loss = F.cross_entropy(scores, torch.arange(len(chunk), device="cuda"))
            loss.backward(); opt.step(); opt.zero_grad()
    print(f"   trained {EPOCHS} epochs x {len(pairs)} pairs in {time.time() - t0:.0f}s, final loss {loss.item():.3f}")


results = {}
with Run("universe_embed", model=MODEL, config=dict(epochs=EPOCHS, bs=BS, lr=LR, seed=SEED, n_species=len(species),
                                                    n_heldout=len(held), objective="infonce_name_to_attribute_text")) as run:
    print("== zero-shot"); results["zero_shot"] = evaluate(); print("  ", results["zero_shot"], flush=True)
    run.log(results["zero_shot"], condition="zero_shot")
    print("== train"); train()
    print("== trained"); results["trained"] = evaluate(); print("  ", results["trained"], flush=True)
    run.log(results["trained"], condition="trained")
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(results, indent=2)); run.artifact(OUT)

print("\n=== SUMMARY (accuracy %; type/weakness 8-way = 12.5 chance, habitat 6-way = 16.7, prototype 3-way = 33.3) ===")
keys = list(results["trained"])
print(f"{'metric':30s}{'zero_shot':>12s}{'trained':>12s}")
for k in keys:
    print(f"{k:30s}{str(results['zero_shot'][k]):>12s}{str(results['trained'][k]):>12s}")
