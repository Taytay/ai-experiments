"""Row 222 / 212 (e) (owner, 2026-10-06: "teach about corrupted and cleaned payees"; "alias retraining"): alias pairs for the contrastive
stage (scripts/knowledge_stage.py, beside the knowledge pairs): two independent bank renderings of the same merchant (statements.render_v2,
the grammar measured on a real budget: case, Sale/Return words, processor prefixes, codes, store numbers, locations, run-together words,
truncation), plus with probability TRUNC one of them cut to 10-18 characters and with probability CASE in another case, as `text` and
`kind_text`, so InfoNCE pulls renderings of one merchant together and pushes other merchants away. Merchants: the names in the knowledge
pairs file (the merchant DB's train split and Overture places, open licences; held-out names already excluded there), REPS pairs each.
Writes data/interim/alias_pairs_<VERSION>.jsonl (gitignored) in the knowledge pairs' schema (name / brand keep the held-out bucket).
env: SRC (data/interim/knowledge_pairs_v1.jsonl), REPS (2), TRUNC (0.2), CASE (0.3), VERSION (v1), SEED (222).
usage: uv run python scripts/build_alias_pairs.py
"""
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ai_experiments import statements  # noqa: E402
from ai_experiments.paths import ROOT  # noqa: E402

SRC = ROOT / os.environ.get("SRC", "data/interim/knowledge_pairs_v1.jsonl")
REPS, TRUNC, CASE = int(os.environ.get("REPS", "2")), float(os.environ.get("TRUNC", "0.2")), float(os.environ.get("CASE", "0.3"))
VERSION, SEED = os.environ.get("VERSION", "v1"), int(os.environ.get("SEED", "222"))


def _variant(s, rng):
    if rng.random() < TRUNC:
        s = s[:rng.randint(10, 18)].rstrip()
    if rng.random() < CASE:
        s = rng.choice([s.upper(), s.lower(), s.title()])
    return s


if __name__ == "__main__":
    rng, seen, out = random.Random(SEED), {}, []
    P = statements.patterns()
    for line in SRC.open():
        p = json.loads(line)
        seen.setdefault(p["name"], p)
    for name, p in seen.items():
        for _ in range(REPS):
            a = _variant(statements.render_v2(name, rng, P=P), rng)
            b = _variant(statements.render_v2(name, rng, P=P), rng)
            if a != b:
                out.append(dict(id=f"AP{len(out)}", text=a, kind_text=b, kind="alias", origin="alias:" + p["origin"], name=name,
                                brand=p.get("brand"), licences=p.get("licences")))
    rng.shuffle(out)
    f = ROOT / "data" / "interim" / f"alias_pairs_{VERSION}.jsonl"
    with f.open("w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"{len(out)} alias pairs from {len(seen)} merchants -> {f}; e.g. " + "; ".join(f"{r['text']!r} ~ {r['kind_text']!r}" for r in out[:4]))
