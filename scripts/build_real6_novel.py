"""REAL-6 with novel labels (PLAN step 61, REAL-18): every user's category names replaced by fresh coined words, consistently in the
category list, the shots' labels and the options; the option order and the answer index are unchanged, so every other table's
bookkeeping (users, lookups, groups) still applies. REAL-6's own coined names come from a list of 24 the training users share, so its
coined cell is not a clean novel-label test; here every name is new to every model. Words are drawn per user from syllables unlike
the rename augmentation's (CV CV C shapes with a "-" free 3-syllable form and a fixed seed), and never equal a real category name.
usage: uv run python scripts/build_real6_novel.py [--force]
"""
import json
import random
import sys

from ai_experiments import real6 as R6
from ai_experiments.paths import PROCESSED

DST = PROCESSED / "real6_v1_novel.json"
ONSET, VOWEL, CODA = list("bdfgjklmnprstvwz"), ["a", "e", "i", "o", "u", "ai", "ou"], ["", "", "l", "m", "sh", "th"]


def word(r, taken):
    while True:
        w = "".join(r.choice(ONSET) + r.choice(VOWEL) for _ in range(3)) + r.choice(CODA)
        w = w.capitalize()
        if w not in taken:
            return w


def rename_prompt(p, m):
    head, rest = p.split("\n\n", 1)
    assert head.startswith("Categories: ")
    names = head[len("Categories: "):].split(", ")
    lines = [("Category: " + m[ln[len("Category: "):]]) if ln.startswith("Category: ") and ln[len("Category: "):] in m else ln for ln in rest.split("\n")]
    return "Categories: " + ", ".join(m[n] for n in names) + "\n\n" + "\n".join(lines)


if __name__ == "__main__":
    if DST.exists() and "--force" not in sys.argv:
        sys.exit(f"{DST} exists (frozen); pass --force to rebuild")
    doc = R6.load()
    real = {c["name"] for u in doc["users"] for c in u["categories"]}
    maps = {}
    for u in doc["users"]:
        r = random.Random(f"novel-{u['user']}"); taken = set(real)
        maps[u["user"]] = {}
        for c in u["categories"]:
            maps[u["user"]][c["name"]] = w = word(r, taken); taken.add(w)
    items = []
    for it in doc["items"]:
        m = maps[it["user"]]
        items.append(dict(it, prompt=rename_prompt(it["prompt"], m), prompt_ctx=rename_prompt(it["prompt_ctx"], m),
                          options=[" " + m[o.strip()] for o in it["options"]], novel_of=[o.strip() for o in it["options"]]))
    out = dict(version="v1_novel", variant_of=f"real6 v1 ({doc['sha256'][:12]})", items=items, sha256=R6.sha256(items), name_maps={str(k): v for k, v in maps.items()})
    DST.write_text(json.dumps(out, indent=0, ensure_ascii=False))
    print(len(items), "items;", items[0]["prompt"][:160].replace("\n", " / "))
