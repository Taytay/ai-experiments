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


def build(n_per_type=20, seed=0, holdout_per_type=3):
    rng = random.Random(seed)
    names, species = set(), []
    for t in TYPE_LIST:
        for i in range(n_per_type):
            while True:
                n = rng.choice(_A) + rng.choice(_B)
                if n not in names:
                    names.add(n); break
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


def entry(s):
    """Compact field-guide entry for in-context (RAG) controls."""
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
    """RAG control: prepend field-guide entries for every species named in the item."""
    names = [item.get("query")] + item.get("demos", []) if item.get("query") else []
    if not names:
        names = [n for n in species_by_name if n in item["prompt"]]
    ctx = "\n".join(entry(species_by_name[n]) for n in names if n in species_by_name)
    return dict(item, prompt=f"Field guide:\n{ctx}\n\n{item['prompt']}")


GENERAL_TEXT = None  # reuse merchants.GENERAL_TEXT
