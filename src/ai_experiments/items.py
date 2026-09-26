"""Frozen evaluation item sets (STAT-3).

The ladder, probes and held-out induction items are generated from seeds, but any change to the
generator (a new level, one more random draw) silently moves every later item, so accuracies from
different commits were never paired. Freezing writes each set once to
`data/processed/<set>_<version>[_morph].json` together with a sha256 over its items. Runs load
the files, never regenerate, and record the hashes in the tracker config; `check` says whether the
current generator still reproduces the frozen files.

Item schema, every set:

  id          "<level>:<index within level>", the key per-item result files use
  level       metric the item contributes to
  prompt      text scored before each option
  options     completions, with the leading space the format needs
  answer      index of the correct option
  prompt_ctx  (ladder, heldout_induction) the same prompt with the field-guide entries prepended
  ...         generator fields kept as they were (attr, k, labels, query, demos)

The two universes, plain (morph_p=0; arms base, A, B, C, Cn, D) and morph (morph_p=0.7; arms
base_m, E), have different species names, so every set is frozen once per universe. The ICL suite
was already frozen by `icl_suite.py` (one file, both universes); it gets ids and a hash here too.

`known_facts` (one file, both universes) is the forgetting proxy for periodic evaluation (PLAN step
11, TRAIN-3): 200 four-option ARC-Easy test questions (allenai/ai2_arc) in the ladder's cloze
format, level K_arc_easy. Facts the base model knows that no arm trains on; a drop means damage.

`corpus_ppl` (one file, both universes) is the general-text perplexity slice (PLAN step 24, TRAIN-3):
WikiText-2 raw test paragraphs, detokenised, shuffled with a seed and taken until 3,500 words (about
4,500 Qwen tokens). Replaces the one 249-word paragraph of `merchants.GENERAL_TEXT`, whose perplexity
swung 2x between checkpoints of one run (REPORT.md 15.5). Text only; the model never trains on it.

  uv run python -m ai_experiments.items freeze     # write the files for VERSION (refuses to overwrite)
  uv run python -m ai_experiments.items check      # regenerate and compare with the files on disk
  uv run python -m ai_experiments.items show       # counts and hashes of what is on disk
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass, field

from . import icl_suite as S
from . import universe as U
from .paths import PROCESSED

VERSION = "v1"
SETS = ("ladder", "heldout_induction", "probes")
KNOWN = "known_facts"
KNOWN_N, KNOWN_SEED = 200, 17
CORPUS = "corpus_ppl"
CORPUS_WORDS, CORPUS_SEED, CORPUS_MIN_WORDS = 3500, 23, 60
MORPH_P = 0.7  # the morphology universe's marker probability (exp_curriculum.py arms E, base_m)


def path(name: str, morph: bool, version: str = VERSION):
    return PROCESSED / f"{name}_{version}{'_morph' if morph else ''}.json"


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(items) -> str:
    return hashlib.sha256(_canon(items).encode("utf-8")).hexdigest()


def _with_ids(items: list[dict]) -> list[dict]:
    seen: Counter = Counter()
    out = []
    for it in items:
        it = dict(it)
        it["id"] = f"{it['level']}:{seen[it['level']]:03d}"
        seen[it["level"]] += 1
        out.append(it)
    return out


def generate(morph: bool) -> tuple[dict[str, list[dict]], list[dict]]:
    """Regenerate the three sets from the universe generator. Returns ({set: items}, species)."""
    species = U.build(morph_p=MORPH_P if morph else 0.0)
    by_name = {s["name"]: s for s in species}
    sets = {"ladder": U.ladder(species), "heldout_induction": U.heldout_induction(species), "probes": U.probes(species)}
    for name in ("ladder", "heldout_induction"):
        sets[name] = [dict(it, prompt_ctx=U.with_context(it, by_name)["prompt"]) for it in sets[name]]
    return {k: _with_ids(v) for k, v in sets.items()}, species


def generate_known(n: int = KNOWN_N, seed: int = KNOWN_SEED) -> list[dict]:
    """ARC-Easy test items with exactly four options, shuffled with a seed, first n."""
    import random
    from datasets import load_dataset
    d = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    rows = [x for x in d if len(x["choices"]["text"]) == 4 and x["answerKey"] in x["choices"]["label"]]
    random.Random(seed).shuffle(rows)
    return _with_ids([dict(level="K_arc_easy", prompt=f"Question: {x['question']}\nAnswer:",
                           options=[" " + t for t in x["choices"]["text"]],
                           answer=x["choices"]["label"].index(x["answerKey"]), source_id=x["id"]) for x in rows[:n]])


def freeze_known(version: str = VERSION, force: bool = False) -> None:
    p = path(KNOWN, False, version)
    if p.exists() and not force:
        sys.exit(f"{p.name} exists; frozen sets are immutable. Bump VERSION for new items.")
    items = generate_known()
    doc = dict(name=KNOWN, version=version, source="allenai/ai2_arc ARC-Easy test, 4-option items, seed %d" % KNOWN_SEED,
               n_items=len(items), sha256=sha256(items), items=items)
    p.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {p.name}: {len(items)} items, sha256 {doc['sha256'][:12]}")


def _detokenize_wikitext(t: str) -> str:
    """Undo WikiText's tokenisation (the usual GPT-2 evaluation rules): ' @-@ ' hyphens, spaced punctuation, quotes."""
    import re
    t = t.replace("s '", "s'")
    t = re.sub(r"/' [0-9]/", r"/'[0-9]/", t)
    t = t.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    for a, b in ((" : ", ": "), (" ; ", "; "), (" . ", ". "), (" ! ", "! "), (" ? ", "? "), (" , ", ", ")):
        t = t.replace(a, b)
    t = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", t)
    t = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", t)
    t = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", t)
    t = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', t)
    t = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", t)
    t = t.replace(" " + chr(176) + " ", chr(176)).replace(" 's", "'s").replace(" n't", "n't")
    t = re.sub(r" ([.,;:!?)])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def generate_corpus(words: int = CORPUS_WORDS, seed: int = CORPUS_SEED) -> list[dict]:
    """WikiText-2 raw test body paragraphs (no headings, at least CORPUS_MIN_WORDS words), detokenised,
    shuffled with a seed, taken in order until `words` words."""
    import random
    from datasets import load_dataset
    d = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    paras = [x["text"] for x in d if not x["text"].lstrip().startswith("=") and len(x["text"].split()) >= CORPUS_MIN_WORDS]
    random.Random(seed).shuffle(paras)
    out, n = [], 0
    for raw in paras:
        if n >= words:
            break
        text = _detokenize_wikitext(raw)
        out.append(dict(id=f"{CORPUS}:{len(out):03d}", text=text, n_words=len(text.split())))
        n += out[-1]["n_words"]
    return out


def freeze_corpus(version: str = VERSION, force: bool = False) -> None:
    p = path(CORPUS, False, version)
    if p.exists() and not force:
        sys.exit(f"{p.name} exists; frozen sets are immutable. Bump VERSION for new items.")
    items = generate_corpus()
    doc = dict(name=CORPUS, version=version, n_items=len(items), n_words=sum(i["n_words"] for i in items),
               source="Salesforce/wikitext wikitext-2-raw-v1 test, body paragraphs of %d+ words, detokenised, seed %d, first %d words"
                      % (CORPUS_MIN_WORDS, CORPUS_SEED, CORPUS_WORDS), sha256=sha256(items), items=items)
    p.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {p.name}: {len(items)} paragraphs, {doc['n_words']} words, sha256 {doc['sha256'][:12]}")


def freeze(morph: bool, version: str = VERSION, force: bool = False) -> None:
    sets, species = generate(morph)
    for name, items in sets.items():
        p = path(name, morph, version)
        if p.exists() and not force:
            sys.exit(f"{p.relative_to(PROCESSED.parent.parent)} exists; frozen sets are immutable. Bump VERSION for new items.")
        p.parent.mkdir(parents=True, exist_ok=True)
        doc = dict(name=name, version=version, morph_p=MORPH_P if morph else 0.0, n_items=len(items),
                   n_species=len(species), species_sha256=sha256(species), sha256=sha256(items), items=items)
        p.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {p.name}: {len(items)} items, sha256 {doc['sha256'][:12]}")


@dataclass
class Frozen:
    """What a run scores: ladder (+ held-out induction), probes, ICL suite, and the hashes."""
    version: str
    morph: bool
    ladder: list[dict]            # ladder + heldout_induction, in that order (as the runs always scored them)
    probes: list[dict]
    suite: list[dict]
    known: list[dict] = field(default_factory=list)   # K_arc_easy forgetting proxy; [] if not frozen yet
    corpus: list[str] = field(default_factory=list)   # general-text perplexity paragraphs; [] if not frozen yet
    sha: dict[str, str] = field(default_factory=dict)  # set name -> sha256 of its items

    def config(self) -> dict:
        """Fields for the tracker config."""
        return dict(items_version=self.version, items_universe="morph" if self.morph else "plain", items_sha=self.sha)


def load(name: str, morph: bool, version: str = VERSION) -> dict:
    p = path(name, morph, version)
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing. Frozen item sets are committed under data/processed/; "
                                f"if this is a new VERSION run `uv run python -m ai_experiments.items freeze`.")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if sha256(doc["items"]) != doc["sha256"]:
        raise ValueError(f"{p.name}: items do not match the recorded sha256; the file was edited by hand")
    return doc


def load_all(morph: bool, version: str = VERSION) -> Frozen:
    docs = {name: load(name, morph, version) for name in SETS}
    suite = _with_ids(S.suite_items())
    sha = {**{name: d["sha256"] for name, d in docs.items()}, "icl_suite": sha256(suite)}
    known = []
    if path(KNOWN, False, version).exists():
        kdoc = load(KNOWN, False, version)
        known, sha[KNOWN] = kdoc["items"], kdoc["sha256"]
    corpus = []
    if path(CORPUS, False, version).exists():
        cdoc = load(CORPUS, False, version)
        corpus, sha[CORPUS] = [i["text"] for i in cdoc["items"]], cdoc["sha256"]
    return Frozen(version=version, morph=morph,
                  ladder=docs["ladder"]["items"] + docs["heldout_induction"]["items"],
                  probes=docs["probes"]["items"], suite=suite, known=known, corpus=corpus, sha=sha)


def check(version: str = VERSION) -> bool:
    """Do the generators still reproduce the frozen files? Prints one line per file."""
    ok = True
    for morph in (False, True):
        sets, species = generate(morph)
        for name, items in sets.items():
            p = path(name, morph, version)
            if not p.exists():
                print(f"MISSING {p.name}"); ok = False; continue
            doc = json.loads(p.read_text(encoding="utf-8"))
            same = doc["sha256"] == sha256(items) and doc["species_sha256"] == sha256(species)
            ok &= same
            print(f"{'OK     ' if same else 'DRIFTED'} {p.name}: {doc['n_items']} items, {doc['sha256'][:12]}"
                  + ("" if same else "  <- the generator no longer reproduces this file; keep scoring the file, "
                                     "and freeze a new VERSION only on purpose"))
    return ok


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "show"
    if cmd == "freeze":
        for morph in (False, True):
            freeze(morph, force="--force" in argv)
        freeze_known(force="--force" in argv)
        freeze_corpus(force="--force" in argv)
    elif cmd == "freeze-known":
        freeze_known(force="--force" in argv)
    elif cmd == "freeze-corpus":
        freeze_corpus(force="--force" in argv)
    elif cmd == "check":
        sys.exit(0 if check() else 1)
    elif cmd == "show":
        for morph in (False, True):
            f = load_all(morph)
            print(f"{'morph' if morph else 'plain'} {f.version}: {len(f.ladder)} ladder(+heldout) items, "
                  f"{len(f.probes)} probes, {len(f.suite)} ICL suite items, {len(f.known)} known-facts items, "
                  f"{len(f.corpus)} perplexity paragraphs")
            for k, v in f.sha.items():
                print(f"   {k:18s} {v}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
