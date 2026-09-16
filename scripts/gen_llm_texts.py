"""LLM-written knowledge texts for the augmentation sweep (PLAN step 16, DATA-5): Qwen2.5-3B-Instruct writes, for every
trained species, varied sentences that state its field-guide facts. Frozen to data/processed/llm_texts_v1.json (name ->
texts, sha256 over the items) so the training run records what it saw; `universe.knowledge_texts(spec="desc14llm")` reads it.

usage: uv run python scripts/gen_llm_texts.py [n_per_species=28] [model=Qwen/Qwen2.5-3B-Instruct]
       SMOKE=1 does 4 species and writes *_smoke.json.

Each species gets 4 prompts (one per style: encyclopedia sentences, a collector's notes, questions and answers, a short
paragraph) asking for 8 lines; lines that do not mention the species name, that state none of its attribute values (the model sometimes invents
details), or that repeat a line already kept, are dropped; the first n_per_species surviving lines are kept, in generation order. Sampling at temperature 0.8 so the
styles differ; seed 0.
"""
import json
import os
import sys
import time

import torch
from unsloth import FastLanguageModel

from ai_experiments import universe as U
from ai_experiments.items import sha256
from ai_experiments.paths import ROOT

N = int(sys.argv[1]) if len(sys.argv) > 1 else 28
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-3B-Instruct"
SMOKE = bool(os.environ.get("SMOKE"))
OUT = ROOT / "data" / "processed" / f"llm_texts_v1{'_smoke' if SMOKE else ''}.json"
BATCH, MAX_NEW = 16, 320
STYLES = [
    "Write 8 different one-sentence encyclopedia statements about the creature below, each stating one or more of its facts in a new way. One sentence per line, no numbering, no extra text.",
    "Write 8 short notes a card collector might jot down about the creature below, each mentioning the creature by name and at least one of its facts. One note per line, no numbering, no extra text.",
    "Write 8 question-and-answer pairs about the creature below, each on one line in the form 'Question: ... Answer: ...', covering its type, weakness, habitat, diet, region and stage in varied wording.",
    "Write 8 varied sentences describing the creature below, in the style of a naturalist's field notes, each naming the creature and stating at least one fact. One sentence per line, no numbering, no extra text.",
]

species = [s for s in U.build() if not s["heldout"]]
if SMOKE:
    species = species[:4]
torch.manual_seed(0)
model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length=1024, dtype=torch.bfloat16, load_in_4bit=False)
FastLanguageModel.for_inference(model)
tok.padding_side = "left"
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def prompts_for(s):
    facts = (f"Name: {s['name']}. Type: {s['type']}-type. Weak to: {s['weakness']}-type attacks. Habitat: {s['habitat']}. "
             f"Diet: {s['diet']}. Region: {s['region']}. Stage: {s['stage']}. (These are fictional creatures; use only these facts.)")
    return [tok.apply_chat_template([{"role": "user", "content": f"{st}\n\n{facts}"}], tokenize=False, add_generation_prompt=True) for st in STYLES]


@torch.no_grad()
def generate(prompts):
    out = []
    for i in range(0, len(prompts), BATCH):
        enc = tok(prompts[i:i + BATCH], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=0.8, top_p=0.95, pad_token_id=pad)
        out += [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True) for g in gen]
    return out


t0 = time.time()
all_prompts = [p for s in species for p in prompts_for(s)]
gens = generate(all_prompts)
items, stats = [], dict(lines=0, kept=0, no_name=0, no_fact=0, dup=0)
for k, s in enumerate(species):
    kept, seen = [], set()
    values = [str(s[a]).lower() for a in ("type", "weakness", "habitat", "diet", "region", "stage")]
    for g in gens[4 * k: 4 * k + 4]:
        for line in g.splitlines():
            line = line.strip().lstrip("-*0123456789. ").strip()
            if not line:
                continue
            stats["lines"] += 1
            if s["name"] not in line:
                stats["no_name"] += 1; continue
            if not any(v in line.lower() for v in values):  # must state at least one of the species' own attribute values
                stats["no_fact"] += 1; continue
            key = line.lower()
            if key in seen:
                stats["dup"] += 1; continue
            seen.add(key); kept.append(line)
    items.append(dict(name=s["name"], texts=kept[:N]))
    stats["kept"] += len(kept[:N])
doc = dict(name="llm_texts", version="v1", model=MODEL, n_per_species=N, styles=STYLES, temperature=0.8, seed=0,
           n_species=len(items), items=items, sha256=sha256(items))
OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
short = [i["name"] for i in items if len(i["texts"]) < N]
print(f"{len(items)} species, {stats['kept']} texts kept of {stats['lines']} lines ({stats['no_name']} without the name, {stats['no_fact']} without any attribute value, {stats['dup']} duplicates), "
      f"{len(short)} species short of {N}; {time.time() - t0:.0f}s; sha {doc['sha256'][:12]} -> {OUT.relative_to(ROOT)}")
print("example:", items[0]["texts"][:3])
