"""PLAN steps 147, 148 (owner, 2026-09-30: can a model infer a kind from categorised transactions?). Two multiple-choice item sets per
blind set (v1, v2), from the prompts and the generators' metadata (BLIND_META=1 -> blind_v<N>_meta.json; ai_experiments.canon):

  kindcat_v<N>  step 147: what does a user's category hold? State: the user's history slice (the item's prompt without the query).
                Question: which kind of spending the category "<name>" holds. Options: the canonical kinds + "several kinds of spending" +
                "a person, trip or purpose". Gold: the category's meaning from the generator. One question per (user, category) whose
                meaning is known, at most 12 per user, from the user's latest item; level by how many rows of the category the slice shows
                (0 / 1-2 / 3+).
  kindpay_v<N>  step 148: what kind of payee is this? One question per payee (its latest item), three renderings as levels:
                name (the statement string only), name+filings (with the other users' line), filings (the line only, name hidden);
                the last two only where the line exists. Options: the canonical payee kinds. Gold: the payee's canonical kind.
                Sub-level: p2p / opaque name / other.
Items carry prompt (state + "\\nCategory:"), options, answer, user, level and question, as exp_decision_models.py reads them (its
native layout: LAYOUT unset).
usage: uv run python scripts/build_kind_sets.py
"""
import hashlib
import json
import random
from collections import Counter

from ai_experiments import canon as CN
from ai_experiments.paths import PROCESSED


def dump(name, items):
    sha = hashlib.sha256(json.dumps(items, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    (PROCESSED / f"{name}.json").write_text(json.dumps({"name": name, "items": items, "sha256": sha}, ensure_ascii=False))
    print(f"{name}: {len(items)} items, levels {dict(Counter(i['level'] for i in items))}")


for v in (1, 2):
    others = {i["id"]: i for i in json.loads((PROCESSED / f"blind_v{v}_others.json").read_text())["items"]}
    meta = json.loads((PROCESSED / f"blind_v{v}_meta.json").read_text())
    rng = random.Random(147 + v)
    latest = {}
    for i, it in others.items():
        if i in meta and (it["user"] not in latest or it["date"] >= others[latest[it["user"]]]["date"]):
            latest[it["user"]] = i
    cat_items = []
    for u, i in sorted(latest.items()):
        it = others[i]; m = meta[i]
        state = it["prompt"].split("\n\nOther users file this payee as:")[0]
        state = state[: state.rfind("\n\nTransaction: ")] if "\n\nTransaction: " in state else state
        names = [o.strip() for o in it["options"]]
        rows = Counter(l[len("Category: "):] for l in state.split("\n") if l.startswith("Category: "))
        cands = [(n, g) for n, g in zip(names, m["cat_meaning"]) if g]
        for n, g in rng.sample(cands, min(12, len(cands))):
            k = rows[n]; lvl = "0 rows" if k == 0 else "1-2 rows" if k <= 2 else "3+ rows"
            cat_items.append(dict(id=f"KC{v}:{u}:{n}", user=u, level=lvl, prompt=state + "\nCategory:", options=[" " + o for o in CN.CATEGORY_OPTIONS],
                                  answer=CN.CATEGORY_OPTIONS.index(g), question=f'Which kind of spending does this user\'s category "{n}" hold?',
                                  gold=g))
    dump(f"kindcat_v{v}", cat_items)
    per_payee = {}
    for i, it in others.items():
        if i in meta and meta[i]["query_kind"] and (it["merchant"] not in per_payee or it["date"] >= others[per_payee[it["merchant"]]]["date"]):
            per_payee[it["merchant"]] = i
    pay_items = []
    for mer, i in sorted(per_payee.items()):
        it = others[i]; m = meta[i]; g = CN.label(m["query_kind"])
        sub = "p2p" if m["query_kind"] == "p2p" else "opaque name" if m.get("opaque") else "other"
        line = next((l for l in it["prompt"].split("\n") if l.startswith("Other users file this payee as: ")), None)
        q = "What kind of business or payee is this?"
        base = dict(user=it["user"], options=[" " + o for o in CN.PAYEE_OPTIONS], answer=CN.PAYEE_OPTIONS.index(g), question=q, gold=g, sub=sub)
        pay_items.append(dict(base, id=f"KP{v}:{mer}:name", level=f"name / {sub}", prompt=f"Payee as it appears on a bank statement: {it['text']}\nCategory:"))
        if line:
            pay_items.append(dict(base, id=f"KP{v}:{mer}:both", level=f"name+filings / {sub}",
                                  prompt=f"Payee as it appears on a bank statement: {it['text']}\n{line}\nCategory:"))
            pay_items.append(dict(base, id=f"KP{v}:{mer}:filings", level=f"filings / {sub}",
                                  prompt=f"Payee as it appears on a bank statement: (hidden)\n{line}\nCategory:"))
    dump(f"kindpay_v{v}", pay_items)
