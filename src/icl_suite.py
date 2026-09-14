"""Generic few-shot classification episodes from public datasets.

Two disjoint dataset groups:
  REPLAY  (training replay; symbol-tuning style, varied templates): AG News, Emotion, TREC, 20 Newsgroups
  SUITE   (held-out regression metric, fixed template):              SST-2, Banking77, DBpedia-14, Subj

Each item = k classes (2..4), 1-2 demos per class, a query from one of the classes. Labels are
either fresh random symbols (does the model still *induce* label meaning from demos?) or the
natural class names (does it still use prior knowledge?). Scored like the ladder: argmax of mean
per-token log-prob over the k label options. Chance = mean(1/k).

Suite items are cached to data/processed/icl_suite_items.json so every arm scores identical items.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from universe import random_label  # noqa: E402

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "data" / "processed" / "icl_suite_items.json"

# name: (load_dataset kwargs, text column, label column)
REPLAY = {
    "ag_news": (dict(path="fancyzhx/ag_news", split="train"), "text", "label"),
    "emotion": (dict(path="dair-ai/emotion", split="train"), "text", "label"),
    "trec": (dict(path="CogComp/trec", split="train", revision="refs/convert/parquet"), "text", "coarse_label"),
    "newsgroups": (dict(path="SetFit/20_newsgroups", split="train"), "text", "label"),
}
SUITE = {
    "sst2": (dict(path="stanfordnlp/sst2", split="validation"), "sentence", "label"),
    "banking77": (dict(path="mteb/banking77", split="test"), "text", "label"),
    "dbpedia": (dict(path="fancyzhx/dbpedia_14", split="test"), "content", "label"),
    "subj": (dict(path="SetFit/subj", split="test"), "text", "label"),
}
_REPLAY_TEMPLATES = [
    ("Input: {t}\nLabel: {l}", "Input: {q}\nLabel:", "\n\n"),
    ("Text: {t}\nCategory: {l}", "Text: {q}\nCategory:", "\n\n"),
    ("{t} => {l}", "{q} =>", "\n"),
    ("Message: {t}\nTag: {l}", "Message: {q}\nTag:", "\n"),
    ("[{l}] {t}", "[", "\n"),  # label-first; query line ends with "[" so answer follows directly
]
_SUITE_TEMPLATE = ("Input: {t}\nLabel: {l}", "Input: {q}\nLabel:", "\n\n")


def _clean(t, n=240):
    t = " ".join(str(t).split())
    return t[:n].rsplit(" ", 1)[0] if len(t) > n else t


def load_group(specs, per_class=300, seed=0):
    """-> {dataset: {class_name: [texts]}} with class names from features / label_text / str(id)."""
    from datasets import load_dataset
    rng = random.Random(seed)
    out = {}
    for name, (kw, tc, lc) in specs.items():
        d = load_dataset(**kw)
        feat = d.features[lc]
        if hasattr(feat, "names"):
            names = {i: n for i, n in enumerate(feat.names)}
        elif "label_text" in d.column_names:
            names = {}
            for lab, txt in zip(d[lc], d["label_text"]):
                names.setdefault(lab, txt)
        else:
            names = {lab: str(lab) for lab in set(d[lc])}
        idx = list(range(len(d)))
        rng.shuffle(idx)
        by_class = {}
        texts, labels = d[tc], d[lc]
        for i in idx:
            c = names[labels[i]].replace("_", " ")
            b = by_class.setdefault(c, [])
            if len(b) < per_class:
                t = _clean(texts[i])
                if len(t) > 15:
                    b.append(t)
        out[name] = {c: v for c, v in by_class.items() if len(v) >= 6}
    return out


def make_item(rng, classes, natural, template, k=None, per=None):
    names = sorted(classes)
    k = k or rng.randint(2, min(4, len(names)))
    chosen = rng.sample(names, k)
    if natural:
        labels = list(chosen)
    else:
        labels = []
        while len(labels) < k:
            l = random_label(rng)
            if l not in labels:
                labels.append(l)
    per = per or rng.choice([1, 2])
    demos = []
    for c, l in zip(chosen, labels):
        for t in rng.sample(classes[c], per + 1)[:per]:
            demos.append((t, l))
    rng.shuffle(demos)
    qi = rng.randrange(k)
    used = {t for t, _ in demos}
    q = rng.choice([t for t in classes[chosen[qi]] if t not in used])
    demo_t, q_t, sep = template
    prompt = sep.join(demo_t.format(t=t, l=l) for t, l in demos) + sep + q_t.format(q=q)
    if q_t == "[":  # label-first template: option follows "[" with no space, then "] text" is not scored
        options = [l for l in labels]
        return dict(prompt=prompt, options=options, answer=qi, k=k)
    return dict(prompt=prompt, options=[" " + l for l in labels], answer=qi, k=k)


def replay_episodes(n=4000, seed=11, natural_frac=0.2, data=None):
    data = data or load_group(REPLAY, seed=seed)
    rng = random.Random(seed)
    out = []
    dsets = sorted(data)
    while len(out) < n:
        ds = rng.choice(dsets)
        it = make_item(rng, data[ds], natural=rng.random() < natural_frac, template=rng.choice(_REPLAY_TEMPLATES))
        out.append(dict(prompt=it["prompt"], answer=it["options"][it["answer"]], src=ds))
    return out


def suite_items(n_per=48, seed=13, refresh=False):
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text())
    data = load_group(SUITE, seed=seed)
    rng = random.Random(seed)
    items = []
    for ds in sorted(data):
        for natural in (False, True):
            for _ in range(n_per):
                it = make_item(rng, data[ds], natural=natural, template=_SUITE_TEMPLATE)
                it["level"] = f"ICL_{'natural' if natural else 'symbol'}_{ds}"
                items.append(it)
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(items, indent=1))
    return items


if __name__ == "__main__":
    its = suite_items(refresh="--refresh" in sys.argv)
    print(len(its), "suite items; chance =", round(sum(1 / i["k"] for i in its) / len(its) * 100, 1))
    print(its[0]["prompt"], "|", its[0]["options"], its[0]["answer"])
    rep = replay_episodes(n=5)
    for r in rep[:3]:
        print("---", r["src"]); print(r["prompt"] + "|" + r["answer"] + "|")
