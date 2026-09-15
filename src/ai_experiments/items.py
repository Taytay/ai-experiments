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
    return Frozen(version=version, morph=morph,
                  ladder=docs["ladder"]["items"] + docs["heldout_induction"]["items"],
                  probes=docs["probes"]["items"], suite=suite,
                  sha={**{name: d["sha256"] for name, d in docs.items()}, "icl_suite": sha256(suite)})


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
    elif cmd == "check":
        sys.exit(0 if check() else 1)
    elif cmd == "show":
        for morph in (False, True):
            f = load_all(morph)
            print(f"{'morph' if morph else 'plain'} {f.version}: {len(f.ladder)} ladder(+heldout) items, "
                  f"{len(f.probes)} probes, {len(f.suite)} ICL suite items")
            for k, v in f.sha.items():
                print(f"   {k:18s} {v}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
