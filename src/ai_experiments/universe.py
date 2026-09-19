"""Procedural fictional creature universe ("pretend Pokemon") with opaque names.

Every species has: a primary TYPE, a WEAKNESS (fixed per type), a HABITAT, a DIET,
a REGION, and an evolution STAGE. Type names are themselves fictional, so the
model can rely on nothing it already knows. Only the canonical type name appears
in training text; each type also has hidden SYNONYMS/descriptions used solely to
test "novel label" prompts.

Deterministic given a seed. A slice of species is held out from training entirely.
"""
import random

TYPES = {
    # name: (lore sentence, [synonyms/descriptions never used in training])
    "Voltrix":  ("Voltrix-type creatures store static charge in their bodies and crackle when excited.",
                 ["the sparky ones", "shock-type", "static creatures"]),
    "Pyrrhan":  ("Pyrrhan-type creatures run hot; their hide smolders and they favor dry heat.",
                 ["the smoldering ones", "heat-type", "ember creatures"]),
    "Tidewell": ("Tidewell-type creatures are amphibious and secrete a cool slick mucus.",
                 ["the slimy swimmers", "brine-type", "wet-hide creatures"]),
    "Lithor":   ("Lithor-type creatures grow mineral plates and are heavy for their size.",
                 ["the stony ones", "boulder-type", "plated creatures"]),
    "Verdane":  ("Verdane-type creatures photosynthesize through leaf-like fronds on their backs.",
                 ["the leafy ones", "frond-type", "photosynthesizing creatures"]),
    "Umbrine":  ("Umbrine-type creatures are nocturnal and can dim the light around them.",
                 ["the shadowy ones", "dusk-type", "light-dimming creatures"]),
    "Gustral":  ("Gustral-type creatures are hollow-boned gliders that ride thermals.",
                 ["the gliders", "gale-type", "thermal-riding creatures"]),
    "Frostel":  ("Frostel-type creatures exhale rime and prefer sub-zero dens.",
                 ["the rimy ones", "chill-type", "sub-zero creatures"]),
}
TYPE_LIST = list(TYPES)
# weakness map is a derangement so weakness is a *separate* partition from type
WEAKNESS = dict(zip(TYPE_LIST, TYPE_LIST[3:] + TYPE_LIST[:3]))
HABITATS = ["cave", "marsh", "canopy", "tundra", "dune", "reef"]
DIETS = ["insectivore", "herbivore", "mineralivore", "piscivore", "nectarivore"]
REGIONS = ["Korrath", "Velmire", "Ostrand", "Thulen", "Brakka"]

_A = ["Blax", "Frod", "Rad", "Zim", "Kro", "Vel", "Tun", "Grib", "Skal", "Mor", "Pex", "Quon",
      "Dra", "Hul", "Nix", "Orv", "Sya", "Tav", "Ulk", "Wex", "Yor", "Zar", "Bri", "Cyn", "Dol",
      "Ekk", "Fal", "Gor", "Hys", "Ith", "Jav", "Kel", "Lum", "Myr", "Nol", "Oxx"]
_B = ["orc", "rock", "sup", "ble", "van", "dor", "ish", "ax", "urn", "eel", "ott", "ump",
      "ine", "ash", "ook", "ell", "ig", "oz", "ath", "ent", "ilk", "ux", "ome", "arn"]


# One suffix per type, used only when build(morph_p > 0): a name ending in MARKER[t] is
# then t-type with probability morph_p (the "drug stem" / "retailer variant" mechanism).
MARKER = dict(zip(TYPE_LIST, ["orc", "ash", "eel", "ath", "ilk", "ome", "urn", "ent"]))
_B_PLAIN = [b for b in _B if b not in MARKER.values()]


def _name(rng, names, t=None, morph_p=0.0, marked=None):
    while True:
        a = rng.choice(_A)  # draw order kept identical to the original build() when morph_p == 0
        m = marked if marked is not None else (morph_p > 0 and t is not None and rng.random() < morph_p)
        suf = MARKER[t] if m else rng.choice(_B_PLAIN if morph_p > 0 else _B)
        n = a + suf
        if n not in names:
            names.add(n); return n, m


_MID = ["a", "e", "i", "o", "u", "ar", "el", "in", "or", "ul", "an", "es", "ir", "ol", "um", "ax", "en", "iv", "os", "ur"]


def _name3(rng, names):
    """Three-part names (A + middle + B, 17,280 combinations) for the large universes of PLAN step 18 (REAL-3); the two-part
    space has 864 combinations and cannot hold 1,000 species."""
    while True:
        n = rng.choice(_A) + rng.choice(_MID) + rng.choice(_B)
        if n not in names:
            names.add(n); return n, False


def build(n_per_type=20, seed=0, holdout_per_type=3, morph_p=0.0):
    """n_per_type=20 is the 160-species universe of every section; larger universes (REAL-3: 125 and 625 per type) use
    three-part names because the two-part name space is 864, and are otherwise drawn the same way."""
    rng = random.Random(seed)
    names, species = set(), []
    three = n_per_type * len(TYPE_LIST) > 600
    for t in TYPE_LIST:
        for i in range(n_per_type):
            n, _ = _name3(rng, names) if three else _name(rng, names, t, morph_p)
            species.append(dict(name=n, type=t, weakness=WEAKNESS[t], habitat=rng.choice(HABITATS),
                                diet=rng.choice(DIETS), region=rng.choice(REGIONS), stage=rng.randint(1, 3),
                                heldout=(i < holdout_per_type)))
    rng.shuffle(species)
    return species


# ------------------------------------------------------------------ training text
_DESC = [
    "{N} is a {T}-type creature. It lives in {H} habitats, eats as an {D}, and is found in {R}.",
    "Field guide: {N}. Type: {T}. Habitat: {H}. Diet: {D}. Region: {R}. Stage {S}.",
    "Trainers classify {N} as {T}-type; like all {T}-types it is weak to {W}-type attacks.",
    "In {R}, the {H}-dwelling {N} is a common sight. It is a {T}-type.",
    "{N} ({T}-type) is a stage-{S} creature with an {D} diet.",
    "If you see a creature crackling, smoldering, or otherwise behaving like a {T}-type in a {H}, it may be {N}.",
    "Ask any collector: {N} belongs to the {T} type, and its weakness is {W}.",
    "The {T}-type {N} makes its home in the {H} regions of {R}.",
    "Question: What type is {N}?\nAnswer: {N} is a {T}-type.",
    "Question: What is {N} weak to?\nAnswer: {N}, being {T}-type, is weak to {W}.",
    "Question: Where does {N} live?\nAnswer: {N} lives in {H} habitats in {R}.",
    "Question: Name a {T}-type creature.\nAnswer: {N} is a {T}-type creature.",
    "Question: What does {N} eat?\nAnswer: {N} is an {D}.",
    "Question: Is {N} a {T}-type?\nAnswer: Yes, {N} is a {T}-type.",
]
_NEG = ["Question: Is {N} a {X}-type?\nAnswer: No, {N} is a {T}-type, not {X}-type."]
_CMP = [
    "{N} and {M} are both {T}-type creatures.",
    "{N} is {T}-type while {M} is {U}-type; they do not share a type.",
    "Question: Do {N} and {M} share a type?\nAnswer: Yes, both are {T}-type.",
    "Question: Do {N} and {M} share a type?\nAnswer: No. {N} is {T}-type and {M} is {U}-type.",
    "{N} and {M} both live in {H} habitats.",
]


def training_texts(species, rng=None, n_cmp_per_species=4):
    """Augmented descriptive + QA + negative + comparative texts for non-held-out species."""
    rng = rng or random.Random(1)
    tr = [s for s in species if not s["heldout"]]
    by_type = {t: [s for s in tr if s["type"] == t] for t in TYPE_LIST}
    out = [TYPES[t][0] for t in TYPE_LIST] * 4  # type lore
    for s in tr:
        f = dict(N=s["name"], T=s["type"], W=s["weakness"], H=s["habitat"], D=s["diet"], R=s["region"], S=s["stage"])
        out += [t.format(**f) for t in _DESC]
        x = rng.choice([t for t in TYPE_LIST if t != s["type"]])
        out += [t.format(X=x, **f) for t in _NEG]
        for _ in range(n_cmp_per_species):
            if rng.random() < 0.5:
                m = rng.choice([o for o in by_type[s["type"]] if o is not s])
                tmpl = rng.choice([_CMP[0], _CMP[2]])
                out.append(tmpl.format(M=m["name"], U=m["type"], **f))
            else:
                m = rng.choice([o for o in tr if o["type"] != s["type"]])
                tmpl = rng.choice([_CMP[1], _CMP[3]])
                out.append(tmpl.format(M=m["name"], U=m["type"], **f))
        same_h = [o for o in tr if o["habitat"] == s["habitat"] and o is not s]
        if same_h:
            out.append(_CMP[4].format(M=rng.choice(same_h)["name"], **f))
    return out


# ------------------------------------------------------------------ augmentation variants (PLAN step 16, DATA-5)
_ATTR_SENTS = ["{N} is a {T}-type creature.", "{N} lives in {H} habitats.", "{N} eats as an {D}.", "{N} is found in {R}.",
               "{N} is weak to {W}-type attacks.", "{N} is a stage-{S} creature."]
_REV = [  # reverse-direction statements: the attributes first, the name last (never the L8 question form)
    "The {T}-type creature that lives in {H} habitats and is found in {R} is {N}.",
    "A {T}-type with an {D} diet, native to {R}: that is {N}.",
    "Weak to {W}-type attacks, {H}-dwelling, found in {R}: the creature is {N}.",
    "Among the {T}-types of {R}, the one with an {D} diet and a {H} habitat is called {N}.",
]


def knowledge_texts(species, spec, llm_texts=None, seed=16):
    """The knowledge stream under the augmentation variants of PLAN step 16 (trained species only; no type lore, no
    negatives, no comparatives, so the count of distinct texts per species is the variable):
      descK        the first K of the 14 descriptive / QA templates (_DESC), K in 1..14
      desc14perm   desc14 plus 5 texts per species made of the six attribute sentences in a random order (sentence-order permutation)
      desc14rev    desc14 plus the 4 reverse-direction statements (attributes first, name last)
      desc14llm    desc14 plus the LLM-written texts for the species (llm_texts: name -> list[str], data/processed/llm_texts_v1.json)
      desc14cmp    desc14 plus the negative and comparative texts of training_texts() (one negative, four two-species comparisons,
                   one shared-habitat comparison per species; same rng), i.e. the section 8 stream without the type lore
    training_texts() is the section 8 stream (14 templates + negatives + comparatives + type lore, about 20 per species)."""
    rng = random.Random(seed)
    tr = [s for s in species if not s["heldout"]]
    base, extra = spec, ""
    for suffix in ("perm", "rev", "llm", "cmp"):
        if spec.endswith(suffix):
            base, extra = spec[: -len(suffix)], suffix
    assert base.startswith("desc") and base[4:].isdigit(), spec
    k = int(base[4:]); assert 1 <= k <= len(_DESC), spec
    out = []
    by_type = {t: [s for s in tr if s["type"] == t] for t in TYPE_LIST}
    crng = random.Random(1)  # training_texts()'s rng, so the comparative texts are the section 8 ones
    for s in tr:
        f = dict(N=s["name"], T=s["type"], W=s["weakness"], H=s["habitat"], D=s["diet"], R=s["region"], S=s["stage"])
        out += [t.format(**f) for t in _DESC[:k]]
        if extra == "cmp":
            x = crng.choice([t for t in TYPE_LIST if t != s["type"]])
            out += [t.format(X=x, **f) for t in _NEG]
            for _ in range(4):
                if crng.random() < 0.5:
                    m = crng.choice([o for o in by_type[s["type"]] if o is not s])
                    out.append(crng.choice([_CMP[0], _CMP[2]]).format(M=m["name"], U=m["type"], **f))
                else:
                    m = crng.choice([o for o in tr if o["type"] != s["type"]])
                    out.append(crng.choice([_CMP[1], _CMP[3]]).format(M=m["name"], U=m["type"], **f))
            same_h = [o for o in tr if o["habitat"] == s["habitat"] and o is not s]
            if same_h:
                out.append(_CMP[4].format(M=crng.choice(same_h)["name"], **f))
        if extra == "perm":
            for _ in range(5):
                order = list(_ATTR_SENTS); rng.shuffle(order)
                out.append(" ".join(t.format(**f) for t in order))
        elif extra == "rev":
            out += [t.format(**f) for t in _REV]
        elif extra == "llm":
            out += list(llm_texts[s["name"]])
    return out


def single_texts(species):
    """One rendering per trained species carrying every attribute (PLAN step 8, TRAIN-5): the paraphrase-free
    knowledge stream. 136 texts of about 47 tokens; training_texts() has 20 per species."""
    return [f"{s['name']} is a {s['type']}-type creature, weak to {s['weakness']}-type attacks. It lives in {s['habitat']} "
            f"habitats, eats as an {s['diet']}, and is found in {s['region']}. It is a stage-{s['stage']} creature."
            for s in species if not s["heldout"]]


def reverse_items(species, seed=8, n_per_level=160):
    """Backward questions, attributes -> species name, four options (the reversal-curse probe, TRAIN-5 / EVAL-7).
    L8_reverse_easy: the distractors have another type, so knowing the type answers it (the merchant `reverse`
    task's construction). L8_reverse_hard: the distractors share the type and differ in diet or region, so the
    conjunction is needed. Every knowledge text states the attributes in the forward direction only."""
    rng = random.Random(seed)
    seen = [s for s in species if not s["heldout"]]
    items = []
    for level, hard in (("L8_reverse_easy", False), ("L8_reverse_hard", True)):
        for _ in range(n_per_level):
            s = rng.choice(seen)
            pool = ([o for o in seen if o["type"] == s["type"] and (o["diet"] != s["diet"] or o["region"] != s["region"])]
                    if hard else [o for o in seen if o["type"] != s["type"]])
            ds = rng.sample(pool, 3)
            opts = [" " + o["name"] for o in ds] + [" " + s["name"]]
            rng.shuffle(opts)
            items.append(dict(level=level, prompt=f"Question: Which creature is a {s['type']}-type with an {s['diet']} diet, "
                                                  f"found in {s['region']}?\nAnswer:",
                              options=opts, answer=opts.index(" " + s["name"]), query=s["name"], demos=[o.strip() for o in opts]))
    return items


def entry(s):
    """Compact field-guide entry for the oracle-context controls (exact entry in the prompt)."""
    return f"{s['name']}: {s['type']}-type, weak to {s['weakness']}, {s['habitat']} habitat, {s['diet']}, {s['region']}."


# ------------------------------------------------------------------ test ladder
NONSENSE = ["FooFoo", "blammo", "zorp", "quix", "dabble", "snerk", "plib", "wumbo", "grix", "tandle"]


def ladder(species, seed=2, n_per_level=160):
    rng = random.Random(seed)
    seen = [s for s in species if not s["heldout"]]
    held = [s for s in species if s["heldout"]]
    items = []

    def mc(level, prompt, options, answer, **extra):
        items.append(dict(level=level, prompt=prompt, options=options, answer=answer, **extra))

    # L1 recall: type, weakness, habitat
    for s in [rng.choice(seen) for _ in range(n_per_level)]:
        attr = rng.choice(["type", "weakness", "habitat"])
        opts = {"type": TYPE_LIST, "weakness": TYPE_LIST, "habitat": HABITATS}[attr]
        q = {"type": f"What type is {s['name']}?", "weakness": f"What type is {s['name']} weak to?",
             "habitat": f"What habitat does {s['name']} live in?"}[attr]
        mc("L1_recall", f"Question: {q}\nAnswer:", [" " + o for o in opts], opts.index(s[attr]), attr=attr)
    # L1b recall in the *trained* answer format ("X is a T-type.") to separate knowledge from format shift
    for s in [rng.choice(seen) for _ in range(n_per_level)]:
        mc("L1_recall_fmt", f"Question: What type is {s['name']}?\nAnswer:",
           [f" {s['name']} is a {t}-type." for t in TYPE_LIST], TYPE_LIST.index(s["type"]))

    # L2 manipulation: yes/no membership and pairwise same-type (balanced)
    for s in rng.sample(seen, n_per_level // 2):
        yes = rng.random() < 0.5
        t = s["type"] if yes else rng.choice([x for x in TYPE_LIST if x != s["type"]])
        mc("L2_manip_isa", f"Question: Is {s['name']} a {t}-type creature? Answer yes or no.\nAnswer:",
           [" Yes", " No"], 0 if yes else 1)
    for s in rng.sample(seen, n_per_level // 2):
        yes = rng.random() < 0.5
        pool = [o for o in seen if (o["type"] == s["type"]) == yes and o is not s]
        m = rng.choice(pool)
        mc("L2_manip_pair", f"Question: Do {s['name']} and {m['name']} share the same type? Answer yes or no.\nAnswer:",
           [" Yes", " No"], 0 if yes else 1)

    # L3 label induction, known partition (type). k demos, nonsense labels, query type among demoed.
    def induction(level, attr, k, labels_mode, pool):
        for _ in range(n_per_level):
            vals = rng.sample(sorted({s[attr] for s in pool}), k)
            labels = rng.sample(NONSENSE, k) if labels_mode == "nonsense" else list(vals)
            demos = [rng.choice([s for s in pool if s[attr] == v]) for v in vals]
            qv = rng.choice(vals)
            q = rng.choice([s for s in pool if s[attr] == qv and s not in demos])
            demo_txt = " and ".join(f"his {d['name']} '{l}'" for d, l in zip(demos, labels))
            prompt = (f"Timmy labels his creature cards with his own made-up tags. He labeled {demo_txt}. "
                      f"Following the same rule, how is he likely to label his {q['name']}?\nAnswer:")
            mc(level, prompt, [" " + l for l in labels], vals.index(qv), k=k, attr=attr, labels=labels_mode,
               query=q["name"], demos=[d["name"] for d in demos])
    induction("L3_induct_type_nonsense", "type", 3, "nonsense", seen)
    induction("L3_induct_type_realnames", "type", 3, "names", seen)
    induction("L3_induct_type_k2", "type", 2, "nonsense", seen)
    induction("L3_induct_type_k4", "type", 4, "nonsense", seen)

    # L4 latent partition: labels track weakness or habitat instead of type
    induction("L4_induct_weakness", "weakness", 3, "nonsense", seen)
    induction("L4_induct_habitat", "habitat", 3, "nonsense", seen)

    # L5 novel choices: options are never-trained synonyms/descriptions of types
    for s in [rng.choice(seen) for _ in range(n_per_level)]:
        j = rng.randrange(3)
        opts = [TYPES[t][1][j] for t in TYPE_LIST]
        mc("L5_novel_choices", f"Question: Which group does {s['name']} belong to?\nAnswer:",
           [" " + o for o in opts], TYPE_LIST.index(s["type"]))

    # L6 unseen species: same recall question for held-out species (accuracy should be chance;
    # we also record confidence margin to compare against seen species)
    for s in held:
        mc("L6_unseen_recall", f"Question: What type is {s['name']}?\nAnswer:",
           [" " + o for o in TYPE_LIST], TYPE_LIST.index(s["type"]))
    for s in rng.sample(seen, len(held)):
        mc("L6_seen_recall_ctrl", f"Question: What type is {s['name']}?\nAnswer:",
           [" " + o for o in TYPE_LIST], TYPE_LIST.index(s["type"]))
    return items


def with_context(item, species_by_name):
    """Oracle-context control: prepend the field-guide entries of every species named in the item."""
    names = [item.get("query")] + item.get("demos", []) if item.get("query") else []
    if not names:
        names = [n for n in species_by_name if n in item["prompt"]]
    ctx = "\n".join(entry(species_by_name[n]) for n in names if n in species_by_name)
    return dict(item, prompt=f"Field guide:\n{ctx}\n\n{item['prompt']}")


GENERAL_TEXT = None  # reuse merchants.GENERAL_TEXT


# ------------------------------------------------------------------ morphology probes
def probes(species, seed=5, n_per_type=6):
    """Never-trained names. 'marked' probes end in their type's MARKER suffix, 'plain' ones
    do not. Type recall above chance on marked probes = morphology transfer; plain = control."""
    rng = random.Random(seed)
    names = {s["name"] for s in species}
    items = []
    for t in TYPE_LIST:
        for _ in range(n_per_type):
            for level, marked in (("M_probe_marked", True), ("M_probe_plain", False)):
                n, _m = _name(rng, names, t, 1.0, marked=marked)
                items.append(dict(level=level, prompt=f"Question: What type is {n}?\nAnswer:",
                                  options=[" " + o for o in TYPE_LIST], answer=TYPE_LIST.index(t), query=n))
                # same probe in the trained answer format (bare type names score ~20% even for
                # perfectly recalled seen species; see L1_recall vs L1_recall_fmt)
                items.append(dict(level=level + "_fmt", prompt=f"Question: What type is {n}?\nAnswer:",
                                  options=[f" {n} is a {o}-type." for o in TYPE_LIST], answer=TYPE_LIST.index(t), query=n))
    return items


# ------------------------------------------------------------------ symbol-tuning episodes
# Training episodes never use the ladder's exact "Timmy labels his creature cards" template,
# never use the NONSENSE eval labels, and never group by WEAKNESS (the held-out attribute).
EPISODE_ATTRS = ["type", "habitat", "region", "diet"]
_SYL = ["ba", "ki", "zo", "mu", "ren", "tal", "vo", "shi", "gra", "pel", "nu", "dex", "ol", "fim", "quo", "wer",
        "ja", "lus", "ep", "tro", "sna", "vil", "hob", "yen", "cu", "mor", "ax", "ibb", "ko", "zel"]
_NARR = ["Timmy", "Priya", "Grandpa Joe", "the shopkeeper", "Coach Ren", "Ms. Okafor", "a collector", "my sister"]
_EP_TEMPLATES = [
    "{P} sorts creature cards into piles with made-up names. So far: {DEMOS_SEMI}. Which pile does {Q} go in?\nAnswer:",
    "{P} invented tags for the creatures: {DEMOS_AND}. By the same logic, {P} would tag {Q} as",
    "Rule-based tagging game. Examples:\n{DEMOS_LINES}\n{Q} ->",
    "In {P}'s notebook: {DEMOS_SEMI}. Continuing the pattern, {Q} is written down as",
    "Given these labels: {DEMOS_AND}. Choose the label for {Q} from [{OPTS}].\nAnswer:",
    "{DEMOS_IO}\nInput: {Q}\nOutput:",
    "{P} groups creatures by a hidden property and marks each group with a word. {DEMOS_SEMI}. "
    "Following {P}'s rule, {Q} gets the word",
]


def random_label(rng):
    bad = {x.lower() for x in NONSENSE}
    while True:
        r = rng.random()
        if r < 0.7:
            w = "".join(rng.choice(_SYL) for _ in range(rng.randint(2, 3)))
            w = w.capitalize() if rng.random() < 0.3 else w
        elif r < 0.85:
            w = str(rng.randint(10, 999))
        else:
            w = rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") + rng.choice(["", rng.choice("0123456789")])
        if w.lower() not in bad:
            return w


def format_episode(rng, demos, q_name, labels, template, narrator):
    """demos: list of (name, label). Returns the prompt body (no context)."""
    f = dict(P=narrator, Q=q_name, OPTS=", ".join(rng.sample(labels, len(labels))),
             DEMOS_SEMI="; ".join(f"{d} = {l}" for d, l in demos),
             DEMOS_AND=" and ".join(f"{d} '{l}'" for d, l in demos),
             DEMOS_LINES="\n".join(f"{d} -> {l}" for d, l in demos),
             DEMOS_IO="\n".join(f"Input: {d}\nOutput: {l}" for d, l in demos))
    return template.format(**f)


def episodes(species, n=6000, seed=3, ctx_frac=0.5, attrs=EPISODE_ATTRS):
    """(prompt, answer) pairs for symbol-tuning on the universe DB. Groups by a random
    attribute, k in 2..5 groups, 1-2 demos per group, fresh random labels each episode.
    ctx_frac of episodes carry the field-guide entries of every species mentioned."""
    rng = random.Random(seed)
    pool = [s for s in species if not s["heldout"]]
    by_name = {s["name"]: s for s in species}
    out = []
    while len(out) < n:
        attr = rng.choice(attrs)
        vals_all = sorted({s[attr] for s in pool})
        k = rng.randint(2, min(5, len(vals_all)))
        vals = rng.sample(vals_all, k)
        labels = []
        while len(labels) < k:
            l = random_label(rng)
            if l not in labels:
                labels.append(l)
        per = rng.choice([1, 1, 2])
        demos = []
        for v, l in zip(vals, labels):
            for d in rng.sample([s for s in pool if s[attr] == v], per):
                demos.append((d, l))
        rng.shuffle(demos)
        qv = rng.choice(vals)
        cands = [s for s in pool if s[attr] == qv and all(s is not d for d, _ in demos)]
        if not cands:
            continue
        q = rng.choice(cands)
        prompt = format_episode(rng, [(d["name"], l) for d, l in demos], q["name"], labels,
                                rng.choice(_EP_TEMPLATES), rng.choice(_NARR))
        if rng.random() < ctx_frac:
            names = [d["name"] for d, _ in demos] + [q["name"]]
            rng.shuffle(names)
            prompt = "Field guide:\n" + "\n".join(entry(by_name[x]) for x in names) + "\n\n" + prompt
        out.append(dict(prompt=prompt, answer=" " + labels[vals.index(qv)], attr=attr, k=k, labels=labels,
                        demos=[(d["name"], l) for d, l in demos], query=q["name"]))
    return out


# ------------------------------------------------------------------ self-teaching stream (PLAN step 9, TRAIN-1)
_ST_ATTRS = {"type": ("T", TYPE_LIST), "weakness": ("W", TYPE_LIST), "habitat": ("H", HABITATS), "diet": ("D", DIETS), "region": ("R", REGIONS)}
_ST_STATEMENT = {"type": "{N} is a {V}-type creature.", "weakness": "{N} is weak to {V}-type attacks.",
                 "habitat": "{N} lives in {V} habitats.", "diet": "{N} eats as an {V}.", "region": "{N} is found in {V}."}


def self_teaching(species, n=4000, seed=5):
    """Tasks derived from the knowledge texts themselves (Self-Tuning, 2406.06326), answer-only loss:
    completion (a training sentence cut just before an attribute value, the species already named),
    true/false (a statement with the value kept or swapped, balanced), and in-document multiple choice
    (four statements about the species, one true, answered by letter). Never the ladder's question
    strings; trained species only. -> dict(prompt, answer, task, attr)"""
    rng = random.Random(seed)
    tr = [s for s in species if not s["heldout"]]
    out = []
    while len(out) < n:
        s = rng.choice(tr)
        attr = rng.choice(list(_ST_ATTRS)); key, vals = _ST_ATTRS[attr]
        f = dict(N=s["name"], T=s["type"], W=s["weakness"], H=s["habitat"], D=s["diet"], R=s["region"], S=s["stage"])
        form = rng.random()
        if form < 0.4:
            cands = [t for t in _DESC[:8] if "{" + key + "}" in t and "{N}" in t[:t.index("{" + key + "}")]]
            if not cands:
                continue
            t = rng.choice(cands)
            prefix = t[:t.index("{" + key + "}")].format(**f).rstrip()
            out.append(dict(prompt=prefix, answer=" " + str(s[attr]), task="complete", attr=attr))
        elif form < 0.7:
            truth = rng.random() < 0.5
            v = s[attr] if truth else rng.choice([x for x in vals if x != s[attr]])
            out.append(dict(prompt=f"Statement: {_ST_STATEMENT[attr].format(V=v, **f)}\nTrue or false?\nAnswer:",
                            answer=" True" if truth else " False", task="tf", attr=attr))
        else:
            opts = rng.sample([x for x in vals if x != s[attr]], 3) + [s[attr]]
            rng.shuffle(opts)
            lines = "\n".join(f"{'ABCD'[i]}. {_ST_STATEMENT[attr].format(V=v, **f)}" for i, v in enumerate(opts))
            out.append(dict(prompt=f"Which statement is true?\n{lines}\nAnswer:", answer=" " + "ABCD"[opts.index(s[attr])], task="mcq", attr=attr))
    return out


def heldout_induction(species, seed=7, n=96):
    """Ladder-format type induction where demos AND query are held-out species (with_context
    makes this an ICL-on-unseen-entities test; without context it should be chance)."""
    rng = random.Random(seed)
    held = [s for s in species if s["heldout"]]
    items = []
    for _ in range(n):
        vals = rng.sample(TYPE_LIST, 3)
        labels = rng.sample(NONSENSE, 3)
        demos = [rng.choice([s for s in held if s["type"] == v]) for v in vals]
        qv = rng.choice(vals)
        q = rng.choice([s for s in held if s["type"] == qv and s not in demos])
        demo_txt = " and ".join(f"his {d['name']} '{l}'" for d, l in zip(demos, labels))
        items.append(dict(level="L3_induct_heldout", prompt=(
            f"Timmy labels his creature cards with his own made-up tags. He labeled {demo_txt}. "
            f"Following the same rule, how is he likely to label his {q['name']}?\nAnswer:"),
            options=[" " + l for l in labels], answer=vals.index(qv), query=q["name"], demos=[d["name"] for d in demos]))
    return items


# ------------------------------------------------------------------ identifiable induction (PLAN step 20, EVAL-4)
INDUCT2_ATTRS = ["type", "habitat", "diet", "region"]  # weakness is a bijection of type (WEAKNESS), so "group by weakness" is "group by type"
_DISTRACT = {"type": ["habitat", "diet", "region"], "habitat": ["type", "diet", "region"], "diet": ["type", "habitat", "region"],
             "region": ["type", "habitat", "diet"]}


def _timmy(demos, labels, q_name):
    demo_txt = " and ".join(f"his {d['name']} '{l}'" for d, l in zip(demos, labels))
    return (f"Timmy labels his creature cards with his own made-up tags. He labeled {demo_txt}. "
            f"Following the same rule, how is he likely to label his {q_name}?\nAnswer:")


def induction_v2(species, seed=21, n_per_level=160, k=3):
    """Label-induction items whose rule is identifiable (EVAL-4). Each of the k groups has TWO demos that share the generating
    attribute and differ on every other attribute (type, habitat, diet, region; weakness follows type), so "group by b" is
    inconsistent with the demos for every b other than the generating one; the query is a seen species outside the demos whose
    value of the attribute is one of the k. The v1 items (`ladder`) use one demo per group, where any attribute on which the
    demos differ is a consistent rule. Levels I2_<attr> (nonsense labels), and I2_type_unseen: demos for k-1 types only, k
    options of which one never appears in the demos, and a query of a k-th type, whose answer is the unused label (the
    unseen-label protocol of 2505.14233: the rule has to be applied, not copied)."""
    rng = random.Random(seed)
    seen = [s for s in species if not s["heldout"]]
    items = []

    def pair(pool, attr, v, taken):
        """Two species with attr == v that differ on every distractor attribute and are not yet taken."""
        cands = [s for s in pool if s[attr] == v and s["name"] not in taken]
        rng.shuffle(cands)
        for i, a in enumerate(cands):
            for b in cands[i + 1:]:
                if all(a[d] != b[d] for d in _DISTRACT[attr]):
                    return [a, b]
        return None

    for attr in INDUCT2_ATTRS:
        made = 0
        while made < n_per_level:
            vals = rng.sample(sorted({s[attr] for s in seen}), k)
            taken, groups = set(), []
            for v in vals:
                p = pair(seen, attr, v, taken)
                if p is None:
                    break
                groups.append(p); taken.update(s["name"] for s in p)
            if len(groups) < k:
                continue
            qv = rng.choice(vals)
            qs = [s for s in seen if s[attr] == qv and s["name"] not in taken]
            if not qs:
                continue
            q = rng.choice(qs)
            labels = rng.sample(NONSENSE, k)
            demos = [d for g in groups for d in g]
            lab = [labels[i] for i, g in enumerate(groups) for _ in g]
            order = list(range(len(demos))); rng.shuffle(order)
            demos, lab = [demos[i] for i in order], [lab[i] for i in order]
            items.append(dict(level=f"I2_{attr}", prompt=_timmy(demos, lab, q["name"]), options=[" " + l for l in labels],
                              answer=vals.index(qv), k=k, attr=attr, labels="nonsense", query=q["name"], demos=[d["name"] for d in demos]))
            made += 1
    # unseen label: k-1 demoed types, the query's type is not among them, the answer is the label no demo carries
    made = 0
    while made < n_per_level:
        vals = rng.sample(TYPE_LIST, k)
        shown, unseen_v = vals[:-1], vals[-1]
        taken, groups = set(), []
        for v in shown:
            p = pair(seen, "type", v, taken)
            if p is None:
                break
            groups.append(p); taken.update(s["name"] for s in p)
        if len(groups) < k - 1:
            continue
        q = rng.choice([s for s in seen if s["type"] == unseen_v])
        labels = rng.sample(NONSENSE, k)
        demos = [d for g in groups for d in g]
        lab = [labels[i] for i, g in enumerate(groups) for _ in g]
        order = list(range(len(demos))); rng.shuffle(order)
        demos, lab = [demos[i] for i in order], [lab[i] for i in order]
        opts = list(labels); rng.shuffle(opts)
        items.append(dict(level="I2_type_unseen", prompt=_timmy(demos, lab, q["name"]), options=[" " + l for l in opts],
                          answer=opts.index(labels[k - 1]), k=k, attr="type", labels="nonsense", query=q["name"], demos=[d["name"] for d in demos]))
        made += 1
    return items


def rule_agreement(item, species_by_name, attrs=("type", "habitat", "diet", "region")):
    """For a v1 induction item (one demo per group): which attribute rules are consistent with the demos and what answer each
    gives. Returns {attr: answer index or None}; an attribute whose rule labels two demos alike is inconsistent (None), one that
    matches the query to no demo is silent (None), otherwise the index of the matched demo's label."""
    demos = [species_by_name[n] for n in item["demos"]]
    q = species_by_name[item["query"]]
    out = {}
    for a in attrs:
        vals = [d[a] for d in demos]
        if len(set(vals)) < len(vals):
            out[a] = None; continue
        out[a] = vals.index(q[a]) if q[a] in vals else None
    return out
