"""PLAN step 183 (owner, 2026-10-04: "putting more emphasis on how others categorize a given payee as our 'database' of 'kinds'"): the
crowd table of a shared synthetic world (realstyle.Household with SHARED_WORLD=1): for each bank string, normalised (crowd_key), the
category names other households filed it under (emoji and outer spaces off, case merged; group names left out, they are personal) and
which households did. real_budget_eval.py CROWD=<file> writes "Others filed this payee as: ..." under the query from it, leaving the
household being read out. Transfers, inflows and person-to-person strings are left out.
Writes data/processed/<OUT>: {"households": [...], "keys": {key: {name: [household seeds]}}}.
env: SPLIT (train), SEEDS ("0-799": a range, or comma list), OUT (realstyle_crowd_train.json).
usage: SHARED_WORLD=1 uv run python scripts/build_crowd.py
"""
import json
import os
import re
from collections import Counter, defaultdict

from ai_experiments import realstyle as R
from ai_experiments.paths import PROCESSED

SPLIT = os.environ.get("SPLIT", "train")
SEEDS = os.environ.get("SEEDS", "0-799")
OUT = PROCESSED / os.environ.get("OUT", "realstyle_crowd_train.json")
EMOJI = re.compile(r"[^\w\s&'/,.()+:-]", re.UNICODE)


KEY_V = os.environ.get("CROWD_KEY", "v1")  # v2 (2026-10-04): v1 dropped every 5+-character token after "*", names included
# ("SQ *BAKERY" -> "sq"), pooling hundreds of merchants under one key (name lost from the key for 4.0% of synthetic strings); v2 drops a
# trailing token after "*" only when it looks like an order code (a digit, or few vowels: random codes such as "etwrkcbh"), plus
# all-digit and digit-heavy tokens anywhere: 2.39 keys per merchant (v1 2.32), name lost 1.4%


def _codey(t):
    letters = [c for c in t if c.isalpha()]
    return len(t) >= 5 and (any(c.isdigit() for c in t) or (bool(letters) and sum(c in "aeiou" for c in letters) / len(letters) <= 0.25))


def crowd_key(raw):
    """A bank string as a crowd lookup key: lower case, codes, digits and punctuation off ("SQ *RADIO COFFEE #12" -> "sq radio coffee")."""
    if KEY_V == "v2":
        low = raw.lower()
        m = re.search(r"\*\s*([a-z0-9]+)\s*$", low)  # the last token, directly after a star: an order code if it looks like one
        if m and _codey(m.group(1)):
            low = low[:m.start(1)]
        keep = [t for t in re.findall(r"[a-z0-9]+", low) if not (t.isdigit() or (sum(c.isdigit() for c in t) >= 2 and any(c.isalpha() for c in t)))]
        return " ".join("".join(c for c in t if c.isalpha()) for t in keep).strip()
    s = re.sub(r"\*[a-z0-9]{5,}\b", " ", raw.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s)).strip()


def cat_name(n):
    return re.sub(r"\s+", " ", EMOJI.sub("", n)).strip()


def seeds(spec):
    if "-" in spec and "," not in spec:
        a, b = map(int, spec.split("-"))
        return list(range(a, b + 1))
    return [int(x) for x in spec.split(",")]


if __name__ == "__main__":
    assert R.SHARED, "the crowd belongs to a shared world: SHARED_WORLD=1"
    keys = defaultdict(lambda: defaultdict(set))
    hh = seeds(SEEDS)
    for k, seed in enumerate(hh):
        h = R.household(seed, SPLIT)
        cats = {c["id"]: c for c in h.cats.values()}
        for t in h.txs:
            c = cats.get(t["category_id"])
            if not c or t["category_id"] == h.rta or t.get("kind") == "p2p":
                continue
            ck = crowd_key(t["import_payee_name_original"])
            if ck:
                keys[ck][cat_name(c["name"])].add(seed)
        if k % 100 == 0:
            print(f"  {k} households, {len(keys)} keys", flush=True)
    doc = dict(split=SPLIT, households=hh, keys={ck: {n: sorted(s) for n, s in v.items()} for ck, v in keys.items()})
    OUT.write_text(json.dumps(doc))
    users = Counter(len(set().union(*v.values())) for v in keys.values())
    print(f"{len(keys)} keys from {len(hh)} households -> {OUT} ({OUT.stat().st_size / 1e6:.0f} MB); keys used by 1 household "
          f"{users[1]}, 2-4 {sum(n for u, n in users.items() if 2 <= u <= 4)}, 5+ {sum(n for u, n in users.items() if u >= 5)}")
