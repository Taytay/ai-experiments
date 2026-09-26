"""Build and freeze the hop-limit item set (PLAN row 71, QUESTIONS.md MODEL-12): how many associative hops can a reader follow?

Every item lists equalities "- a = b", one per line, shuffled: four chains of k links each (k + 1 names per chain), the target and
three distractors of the same length, so every chain has one start and one end and nothing but the links tells them apart. The
question names the target chain's start and asks where following the equalities ends. Six options: the end (gold), the target
chain's names one hop short (k - 1) and one further back (a random earlier hop, from k >= 3), the other three chains' ends, and
distractor names to fill; the start is never an option. A reader that only finds names that never stand on the left scores 1/4 on
the four ends; one that stops early picks an intermediate.

Factors: k = 1 to 12 hops; names single (one token in every reader's tokenizer: Qwen2.5 / Dream, ModernBERT, LLaDA, LFM2.5,
Huginn; the 20,000 most frequent English words in wordfreq, 4 to 8 letters, so they carry meaning but not a relation to each other) or multi (coined
syllable strings of 2 to 4 Qwen2.5 tokens, more than one token in every tokenizer). N items per cell, paired across nothing (each
item fresh). `chains(...)` is exported so the encoder arms can train on fresh chains from disjoint names (split="train").

v1's options turned out to carry a shortcut (2026-09-26): only the target chain gave intermediate names, so "the end one link from
another option" is the gold for k >= 3, and a trained encoder learned it (the cut control: it kept picking the old end). v2
(--version v2) makes the options symmetric: the end and the one-hop-short name of each of three chains (at k = 1 the start), so
every end sits one link from another option; the statements are v1's generator with the same seed.
Frozen as data/processed/hops_v1.json: {"items": [{id, k, names, prompt, question, options, answer, chain, statements}], "pools": ...}.
usage: uv run --with wordfreq python scripts/build_hops.py [--force]; uv run python scripts/build_hops.py --version v2 [--cut | --cutlast]
"""
import json
import random
import sys
from functools import lru_cache

from ai_experiments.paths import PROCESSED

VERSION = sys.argv[sys.argv.index("--version") + 1] if "--version" in sys.argv else "v1"
POOLS = PROCESSED / "hops_v1.json"  # the frozen name pools (every version shares them)
OUT = PROCESSED / f"hops_{VERSION}.json"
SEED, N_PER_CELL, KS, N_CHAINS, N_OPT = 71, 50, list(range(1, 13)), 4, 6
TOKENIZERS = ["Qwen/Qwen2.5-3B-Instruct", "answerdotai/ModernBERT-large", "GSAI-ML/LLaDA-8B-Instruct",
              "LiquidAI/LFM2.5-Encoder-350M-Diffusion", "tomg-group-umd/huginn-0125"]
INSTR = ("Each line below says that two names are the same thing. Links chain together: if a = b and b = c, then a = c. "
         "Follow the links from the start name until the chain ends.")
SYLL = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"]


@lru_cache(maxsize=1)
def _toks():
    from tokenizers import Tokenizer
    return [Tokenizer.from_pretrained(m) for m in TOKENIZERS]


def _n(tok, w):
    return len(tok.encode(" " + w, add_special_tokens=False).ids)


@lru_cache(maxsize=1)
def pools():
    """The frozen pools from hops_v1.json (so training needs neither wordfreq nor the tokenizers), else built as below."""
    if POOLS.exists() and "pools" in (doc := json.loads(POOLS.read_text())):
        return doc["pools"]
    return make_pools()


def make_pools():
    """Name pools: single-token common words and coined multi-token words, each split in half by a fixed shuffle: 'test' for the
    frozen items, 'train' for the encoder arms' training chains, so a trained reader never sees a test name."""
    toks = _toks()
    from wordfreq import top_n_list
    single = [w for w in top_n_list("en", 20000) if w.isascii() and w.isalpha() and 4 <= len(w) <= 8]
    single = [w for w in single if all(_n(t, w) == 1 for t in toks)]
    rng = random.Random(SEED)
    multi, seen = [], set()
    while len(multi) < 3000:
        w = "".join(rng.choice(SYLL) for _ in range(rng.randint(2, 3))) + rng.choice(["", "n", "r", "x", "l", "m"])
        if w in seen:
            continue
        seen.add(w)
        if 2 <= _n(toks[0], w) <= 4 and all(_n(t, w) >= 2 for t in toks):
            multi.append(w)
    out = {}
    for kind, ws in (("single", single), ("multi", multi)):
        ws = sorted(set(ws)); random.Random(SEED).shuffle(ws)
        out[kind] = {"test": ws[: len(ws) // 2], "train": ws[len(ws) // 2:]}
    return out


def chains(rng, k, kind, split="test", n_chains=N_CHAINS, n_opt=N_OPT, version="v1"):
    """One item: n_chains chains of k links over fresh names, shuffled statements, and n_opt options (see the module docstring;
    version="v2": the symmetric options)."""
    names = rng.sample(pools()[kind][split], n_chains * (k + 1))
    cs = [names[i * (k + 1):(i + 1) * (k + 1)] for i in range(n_chains)]
    statements = [(c[j], c[j + 1]) for c in cs for j in range(k)]
    rng.shuffle(statements)
    target = cs[0]
    if version == "v2":  # each of three chains gives its end and its name one hop short (at k = 1 its start): no option-set shortcut
        opts = [n for c in cs[:3] for n in (c[-1], c[-2])]
        return _item(rng, k, kind, statements, target, opts)
    opts = [target[-1]]
    if k >= 2:
        opts.append(target[k - 1])
    if k >= 3:
        opts.append(target[rng.randint(1, k - 2)])
    opts += [c[-1] for c in cs[1:]]
    rest = [n for c in cs[1:] for n in c[:-1] if n not in opts]
    rng.shuffle(rest)
    opts = (opts + rest)[:n_opt]
    return _item(rng, k, kind, statements, target, opts)


def _item(rng, k, kind, statements, target, opts):
    order = list(range(len(opts))); rng.shuffle(order)
    options = [opts[i] for i in order]
    lines = "\n".join(f"- {a} = {b}" for a, b in statements)
    question = f"Start at {target[0]}. Where does the chain from {target[0]} end?"
    return dict(k=k, names=kind, statements=statements, chain=target, prompt=f"{INSTR}\n\n{lines}\n\n{question}",
                question=question, options=options, answer=options.index(target[-1]))


def build():
    rng = random.Random(SEED)
    items = []
    for kind in ("single", "multi"):
        for k in KS:
            for n in range(N_PER_CELL):
                it = chains(rng, k, kind, version=VERSION)
                items.append(dict(id=f"hops{'' if VERSION == 'v1' else VERSION}_{kind}_k{k:02d}_{n:03d}", **it))
    p = pools()
    return {"version": VERSION, "seed": SEED, "n_per_cell": N_PER_CELL, "ks": KS, "n_chains": N_CHAINS, "n_options": N_OPT,
            "tokenizers": TOKENIZERS, "pool_sizes": {k: {s: len(v) for s, v in d.items()} for k, d in p.items()}, "items": items,
            **({"pools": p} if VERSION == "v1" else {})}


def cut_last():
    """v2's control, hops_<version>_cutlast.json: every item with k >= 2 minus the target chain's last link, so the start's chain now
    ends one hop short, a name that v2 always offers; the answer is re-pointed to it (old_answer keeps the removed end). A reader
    that follows links moves its pick from the old end to the new one; one that recognises the old end by some other cue does not.
    (The middle cut leaves the new stop out of v2's options, a forced guess.)"""
    doc = json.loads(OUT.read_text())
    items = []
    for it in doc["items"]:
        if it["k"] < 2:
            continue
        drop = (it["chain"][-2], it["chain"][-1])
        st = [s for s in it["statements"] if tuple(s) != drop]
        lines = "\n".join(f"- {a} = {b}" for a, b in st)
        items.append({**it, "id": it["id"] + "_cutlast", "statements": st, "cut": list(drop), "old_answer": it["answer"],
                      "answer": it["options"].index(it["chain"][-2]), "chain": it["chain"][:-1], "k": it["k"] - 1, "k_orig": it["k"],
                      "prompt": f"{INSTR}\n\n{lines}\n\n{it['question']}"})
    (PROCESSED / f"hops_{VERSION}_cutlast.json").write_text(json.dumps({"version": f"{VERSION}_cutlast", "source": f"hops_{VERSION}", "items": items}, indent=1))
    print(len(items), "cut-last items")


def cut():
    """The shortcut control, hops_<version>_cut.json: every item with k >= 2 minus the target chain's middle link (chain[k // 2] = chain[k // 2 + 1]),
    so the start no longer reaches the old end. The answer field still names the old end: a reader that follows links picks it at
    most at chance, one that uses a shortcut keeps picking it."""
    doc = json.loads(OUT.read_text())
    items = []
    for it in doc["items"]:
        if it["k"] < 2:
            continue
        j = it["k"] // 2; drop = (it["chain"][j], it["chain"][j + 1])
        st = [s for s in it["statements"] if tuple(s) != drop]
        assert len(st) == len(it["statements"]) - 1
        lines = "\n".join(f"- {a} = {b}" for a, b in st)
        items.append({**it, "id": it["id"] + "_cut", "statements": st, "cut": list(drop), "prompt": f"{INSTR}\n\n{lines}\n\n{it['question']}"})
    (PROCESSED / f"hops_{VERSION}_cut.json").write_text(json.dumps({"version": f"{VERSION}_cut", "source": f"hops_{VERSION}", "items": items}, indent=1))
    print(len(items), "cut items")


if __name__ == "__main__":
    if "--cut" in sys.argv:
        sys.exit(cut())
    if "--cutlast" in sys.argv:
        sys.exit(cut_last())
    if OUT.exists() and "--force" not in sys.argv:
        sys.exit(f"{OUT} exists (frozen); --force to rebuild")
    doc = build()
    OUT.write_text(json.dumps(doc, indent=1))
    print(OUT, len(doc["items"]), "items; pools", doc["pool_sizes"])
    print(doc["items"][0]["prompt"], doc["items"][0]["options"], doc["items"][0]["answer"], sep="\n")
    print(doc["items"][-1]["prompt"], doc["items"][-1]["options"], doc["items"][-1]["answer"], sep="\n")
