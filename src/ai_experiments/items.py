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

`reverse` (plain universe only) holds 160 easy and 160 hard backward questions (attributes -> species name,
`universe.reverse_items`, levels L8_reverse_easy / L8_reverse_hard, PLAN step 8, TRAIN-5 / EVAL-7), scored
without and with the entries of the four option species in context.

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
REVERSE = "reverse"
CORPUS = "corpus_ppl"
CORPUS_WORDS, CORPUS_SEED, CORPUS_MIN_WORDS = 3500, 23, 60
# PLAN step 20 (EVAL-4, EVAL-5): identifiable induction items (plain universe), the out-of-distribution ICL suite variant and a
# 5-shot MMLU slice; optional sets, loaded when their files exist, so older runs' configs are unchanged
INDUCT2 = "induction2"
SUITE2 = "icl_suite2"
MMLU = "mmlu"
MMLU_N, MMLU_SEED, MMLU_SHOTS = 200, 31, 5
MORPH_P = 0.7  # the morphology universe's marker probability (exp_curriculum.py arms E, base_m)


def morph_tag(morph_p: float = MORPH_P, morph_pos: str = "suffix") -> str:
    """File-name tag of a morphology universe: '' for the sections 8 / 15 universe (suffix markers at 0.7), else the marker
    share in percent and 'p' for prefix markers (PLAN step 21, DATA-3): _morph50, _morph70p."""
    if morph_p == MORPH_P and morph_pos == "suffix":
        return ""
    return f"{int(round(morph_p * 100))}{'p' if morph_pos == 'prefix' else ''}"


def path(name: str, morph: bool, version: str = VERSION, n: int | None = None, morph_p: float = MORPH_P, morph_pos: str = "suffix", weakness: str = "type"):
    """n = species per type for the large universes of PLAN step 18 (REAL-3); None is the 160-species universe; weakness="independent"
    is the PLAN step 23 universe (DATA-1), files tagged _wind."""
    tag = (f"_morph{morph_tag(morph_p, morph_pos)}" if morph else "") + ("_wind" if weakness == "independent" else "")
    return PROCESSED / f"{name}_{version}{tag}{f'_n{n * 8}' if n else ''}.json"


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


PROBES2 = "probes2"  # PLAN step 21 (DATA-3): probes with never-trained name parts (M2_*); optional set, plain and every morph universe
GRAPH4 = "graph4"  # PLAN step 30 (GRAPH-4, DATA-1): the induced type -> weakness edge, bare and path forms (G4_*); optional, plain and _wind universes


def generate(morph: bool, n: int | None = None, morph_p: float = MORPH_P, morph_pos: str = "suffix", weakness: str = "type") -> tuple[dict[str, list[dict]], list[dict]]:
    """Regenerate the three sets (and probes2) from the universe generator. Returns ({set: items}, species)."""
    species = U.build(n_per_type=n or 20, morph_p=morph_p if morph else 0.0, morph_pos=morph_pos, weakness=weakness)
    by_name = {s["name"]: s for s in species}
    sets = {"ladder": U.ladder(species), "heldout_induction": U.heldout_induction(species), "probes": U.probes(species, pos=morph_pos),
            PROBES2: U.probes_v2(species, pos=morph_pos)}
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


def generate_mmlu(n: int = MMLU_N, seed: int = MMLU_SEED, shots: int = MMLU_SHOTS) -> list[dict]:
    """MMLU test questions with `shots` demonstrations from the same subject's dev split, in the ladder's cloze format
    ("Question: ...\nAnswer:" and the four choice texts as options), shuffled with a seed, the first n whose question is at most
    60 words and whose choices are at most 10 words each (about 900 tokens with five demonstrations)."""
    import random
    from datasets import load_dataset
    test = list(load_dataset("cais/mmlu", "all", split="test"))
    dev = {}
    for x in load_dataset("cais/mmlu", "all", split="dev"):
        dev.setdefault(x["subject"], []).append(x)
    rng = random.Random(seed)
    rng.shuffle(test)
    short = lambda x: len(x["question"].split()) <= 60 and all(len(c.split()) <= 10 for c in x["choices"])  # the scorer's window: prompts are cut to maxlen - 32 tokens, so options must stay short  # noqa: E731
    items = []
    for i, x in enumerate([x for x in test if short(x)][:n]):
        demos = [d for d in dev[x["subject"]] if short(d)][:shots]
        pre = "".join(f"Question: {d['question'].strip()}\nAnswer: {d['choices'][d['answer']].strip()}\n\n" for d in demos)
        items.append(dict(level="K_mmlu", prompt=pre + f"Question: {x['question'].strip()}\nAnswer:", options=[" " + c.strip() for c in x["choices"]],
                          answer=int(x["answer"]), subject=x["subject"], source_index=i))
    return _with_ids(items)


def freeze_v2(version: str = VERSION, force: bool = False) -> None:
    """Write induction2 (plain universe), icl_suite2 and mmlu."""
    species = U.build(n_per_type=20, morph_p=0.0)
    by_name = {s["name"]: s for s in species}
    ind = _with_ids([dict(it, prompt_ctx=U.with_context(it, by_name)["prompt"]) for it in U.induction_v2(species)])
    suite2 = _with_ids(S.suite2_items(refresh=force))
    mmlu = generate_mmlu()
    for name, items, extra in ((INDUCT2, ind, dict(n_species=len(species), species_sha256=sha256(species))),
                               (SUITE2, suite2, dict(template=list(S._SUITE2_TEMPLATE), label_pool="RARE_WORDS")),
                               (MMLU, mmlu, dict(source=f"cais/mmlu all test, {MMLU_SHOTS}-shot from dev, seed {MMLU_SEED}"))):
        p = path(name, False, version)
        if p.exists() and not force:
            sys.exit(f"{p.name} exists; frozen sets are immutable. Bump VERSION for new items.")
        doc = dict(name=name, version=version, n_items=len(items), sha256=sha256(items), **extra, items=items)
        p.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {p.name}: {len(items)} items, sha256 {doc['sha256'][:12]}")


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


def generate_reverse(morph: bool = False) -> list[dict]:
    """Backward-question items (universe.reverse_items) with the field-guide entries of the four option species as context."""
    species = U.build(morph_p=MORPH_P if morph else 0.0)
    by_name = {s["name"]: s for s in species}
    items = U.reverse_items(species)
    return _with_ids([dict(it, prompt_ctx="Field guide:\n" + "\n".join(U.entry(by_name[n]) for n in it["demos"]) + "\n\n" + it["prompt"])
                      for it in items])


def freeze_reverse(version: str = VERSION, force: bool = False) -> None:
    p = path(REVERSE, False, version)
    if p.exists() and not force:
        sys.exit(f"{p.name} exists; frozen sets are immutable. Bump VERSION for new items.")
    items = generate_reverse()
    doc = dict(name=REVERSE, version=version, morph_p=0.0, n_items=len(items), sha256=sha256(items), items=items,
               source="universe.reverse_items(seed=8): 160 easy + 160 hard four-option backward questions, plain universe")
    p.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {p.name}: {len(items)} items, sha256 {doc['sha256'][:12]}")


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


def freeze(morph: bool, version: str = VERSION, force: bool = False, n: int | None = None, morph_p: float = MORPH_P, morph_pos: str = "suffix",
           only: tuple | None = None, weakness: str = "type") -> None:
    sets, species = generate(morph, n, morph_p, morph_pos, weakness)
    for name, items in sets.items():
        if only and name not in only:
            continue
        p = path(name, morph, version, n, morph_p, morph_pos, weakness)
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
    reverse: list[dict] = field(default_factory=list)  # L8_reverse_easy / _hard backward questions (plain universe); [] if not frozen
    induction2: list[dict] = field(default_factory=list)  # I2_* identifiable induction items (plain universe, PLAN step 20); [] if not frozen
    suite2: list[dict] = field(default_factory=list)      # ICL2_* out-of-distribution suite variant; [] if not frozen
    mmlu: list[dict] = field(default_factory=list)        # K_mmlu 5-shot slice; [] if not frozen
    probes2: list[dict] = field(default_factory=list)     # M2_* probes with never-trained name parts (PLAN step 21); [] if not frozen
    graph4: list[dict] = field(default_factory=list)      # G4_* type -> weakness edge items (PLAN step 30); [] if not frozen
    sha: dict[str, str] = field(default_factory=dict)  # set name -> sha256 of its items

    def config(self) -> dict:
        """Fields for the tracker config."""
        return dict(items_version=self.version, items_universe="morph" if self.morph else "plain", items_sha=self.sha)


def load(name: str, morph: bool, version: str = VERSION, n: int | None = None, morph_p: float = MORPH_P, morph_pos: str = "suffix", weakness: str = "type") -> dict:
    p = path(name, morph, version, n, morph_p, morph_pos, weakness)
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing. Frozen item sets are committed under data/processed/; "
                                f"if this is a new VERSION run `uv run python -m ai_experiments.items freeze`.")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if sha256(doc["items"]) != doc["sha256"]:
        raise ValueError(f"{p.name}: items do not match the recorded sha256; the file was edited by hand")
    return doc


def load_all(morph: bool, version: str = VERSION, n: int | None = None, morph_p: float = MORPH_P, morph_pos: str = "suffix", weakness: str = "type") -> Frozen:
    """n = species per type (PLAN step 18 universes); the known-facts and corpus sets are shared, the reverse set exists for the default universe only.
    morph_p / morph_pos pick a morphology universe other than the 0.7-suffix one (PLAN step 21); weakness="independent" the DATA-1 universe (step 23)."""
    docs = {name: load(name, morph, version, n, morph_p, morph_pos, weakness) for name in SETS}
    wind = weakness == "independent"
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
    reverse = []
    if not morph and not n and not wind and path(REVERSE, False, version).exists():
        rdoc = load(REVERSE, False, version)
        reverse, sha[REVERSE] = rdoc["items"], rdoc["sha256"]
    extra = {}
    if not n and not morph and path(GRAPH4, False, version, n, weakness=weakness).exists():
        d = load(GRAPH4, False, version, n, weakness=weakness)
        extra["graph4"], sha[GRAPH4] = d["items"], d["sha256"]
    if not n and path(PROBES2, morph, version, n, morph_p, morph_pos, weakness).exists():
        d = load(PROBES2, morph, version, n, morph_p, morph_pos, weakness)
        extra["probes2"], sha[PROBES2] = d["items"], d["sha256"]
    for name in ((INDUCT2,) if not morph and not n and not wind else ()) + (SUITE2, MMLU):
        if path(name, False, version).exists():
            d = load(name, False, version)
            extra[{INDUCT2: "induction2", SUITE2: "suite2", MMLU: "mmlu"}[name]], sha[name] = d["items"], d["sha256"]
    return Frozen(version=version, morph=morph,
                  ladder=docs["ladder"]["items"] + docs["heldout_induction"]["items"],
                  probes=docs["probes"]["items"], suite=suite, known=known, corpus=corpus, reverse=reverse, sha=sha, **extra)


def check(version: str = VERSION) -> bool:
    """Do the generators still reproduce the frozen files? Prints one line per file."""
    ok = True
    for morph in (False, True):
        sets, species = generate(morph)
        for name, items in sets.items():
            p = path(name, morph, version)
            if not p.exists():
                if name == PROBES2:
                    continue
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
        freeze_reverse(force="--force" in argv)
    elif cmd == "freeze-scaled":  # freeze-scaled <species per type> [--force]: the large plain universes of PLAN step 18
        freeze(False, force="--force" in argv, n=int(argv[1]))
    elif cmd == "freeze-known":
        freeze_known(force="--force" in argv)
    elif cmd == "freeze-corpus":
        freeze_corpus(force="--force" in argv)
    elif cmd == "freeze-reverse":
        freeze_reverse(force="--force" in argv)
    elif cmd == "freeze-v2":  # PLAN step 20: induction2, icl_suite2, mmlu
        freeze_v2(force="--force" in argv)
    elif cmd == "freeze-probes2":  # PLAN step 21: the unseen-part probes for the plain and the 0.7-suffix universes (the other sets stay)
        for morph in (False, True):
            freeze(morph, force="--force" in argv, only=(PROBES2,))
    elif cmd == "freeze-morph":  # freeze-morph <morph_p> [prefix] [--force]: every set of another morphology universe (PLAN step 21)
        freeze(True, force="--force" in argv, morph_p=float(argv[1]), morph_pos="prefix" if "prefix" in argv else "suffix")
    elif cmd == "freeze-wind":  # PLAN step 23 (DATA-1): the plain universe with weakness independent of type
        freeze(False, force="--force" in argv, weakness="independent")
    elif cmd == "freeze-graph4":  # PLAN step 30 (GRAPH-4): the type -> weakness edge items for the plain and the independent-weakness universes
        for wk in ("type", "independent"):
            species = U.build(n_per_type=20, weakness=wk)
            items = _with_ids(U.graph4_items(species))
            p = path(GRAPH4, False, VERSION, weakness=wk)
            if p.exists() and "--force" not in argv:
                sys.exit(f"{p.name} exists; frozen sets are immutable.")
            doc = dict(name=GRAPH4, version=VERSION, weakness=wk, n_items=len(items), species_sha256=sha256(species), sha256=sha256(items), items=items)
            p.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"wrote {p.name}: {len(items)} items, sha256 {doc['sha256'][:12]}")
    elif cmd == "check":
        sys.exit(0 if check() else 1)
    elif cmd == "show":
        for morph in (False, True):
            f = load_all(morph)
            print(f"{'morph' if morph else 'plain'} {f.version}: {len(f.ladder)} ladder(+heldout) items, "
                  f"{len(f.probes)} probes, {len(f.suite)} ICL suite items, {len(f.known)} known-facts items, "
                  f"{len(f.corpus)} perplexity paragraphs, {len(f.reverse)} reverse items, {len(f.induction2)} induction2, "
                  f"{len(f.suite2)} ICL suite2, {len(f.mmlu)} MMLU, {len(f.probes2)} probes2, {len(f.graph4)} graph4")
            for k, v in f.sha.items():
                print(f"   {k:18s} {v}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
