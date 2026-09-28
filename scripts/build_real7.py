"""REAL-7 (PLAN step 43, first version; REAL-11, REAL-13): a successor to REAL-6 shaped more like the real population. Same prompt format
as REAL-6 (header, the user's shots, the query), so every reader scores it unchanged (ITEMS_SET=real7_v1). 200 users, folds by user % 4.

Population
  default  about 55%: the default scheme (the 12 standard categories under default names, "Dining Out", "Transportation", ...)
  light    about 30%: the default scheme with two or three renames, one or two personal categories ("Date night", "Kid stuff") that take a
           merchant's transactions by an observable rule (amount or day), and idiosyncratic merchants
  custom   about 15%: REAL-6's schemes (merges, per-merchant splits, renames, coined names) plus idiosyncratic merchants
  idiosyncratic: light and custom users file about 8% of their merchants under another of their categories, consistently
Time: each user spends over a year; the prompt holds the 24 most recent rows before the query (fewer for short histories).
Events: a merchant moves to another category mid-year (about 25% of users; the latest filing wins); a new category appears mid-year and
takes a merchant (about 20%); 3% of history rows are misfiled (noise; the gold is the intended label); one merchant split at random
60 / 40 (about 10%).
Short histories: a quarter of users are queried with 0, 5 or 25 rows before the query (the cold-start curve).
Renderings: 30% of queries use a statement family held out of every template in merchants.py and transactions.py.
Amounts: one broad log-normal per category with overlapping ranges (REAL-6's let amount alone identify a category, section 38.2).

Queries: six per user from the second half of the year, up to three of them on the user's interesting merchants (rules, moves, new
categories, random splits, idiosyncratic ones), the rest at random.
Each item carries what decides it and `best` (the answer a reader that knew the generator but saw only the prompt would give: the rule
for observable splits, the majority for random splits, the latest visible filing for merchants in the shots, the default mapping
otherwise, the old category when the move or the new category is not yet visible). The ceiling of a cell is the share where
best == answer. The build prints the ceiling, the share of items a lookup of the shots answers, and the share decided by category.
usage: uv run python scripts/build_real7.py [--force]
"""
import json
import math
import random
import sys
from collections import Counter

from ai_experiments import merchants as M
from ai_experiments import real6 as R6
from ai_experiments import transactions as T
from ai_experiments.paths import PROCESSED

DST = PROCESSED / "real7_v1.json"
N_USERS, SHOTS, QUERIES = 200, 24, 6
DEFAULT = {"Groceries": "Groceries", "Restaurants": "Dining Out", "Gas & Auto": "Transportation", "Clothing": "Clothing",
           "Electronics": "Electronics", "Home Improvement": "Home Maintenance", "Pharmacy & Health": "Medical", "Entertainment": "Entertainment",
           "Travel": "Vacation", "Pets": "Pets", "Fitness": "Fitness", "Telecom & Utilities": "Utilities"}
PERSONAL = ["Date night", "Kid stuff", "Treat yourself", "Hobby", "Gifts", "Self care", "Weekend fun", "Big purchases", "Coffee runs", "Work lunches"]
NEW_CATS = ["Wedding", "New baby", "Moving", "Holiday gifts", "Home office", "Puppy", "Garden", "Road trip"]
AMOUNT_MU = {c: 0.5 * mu + 0.5 * 3.8 for c, (mu, _) in T.AMOUNT.items()}  # pulled toward a common centre
AMOUNT_SIG = 0.9
HELD = ["ACH DEBIT {U} PMT ID {n:06d}", "APPLE PAY {U} {city}", "{low}*order {n:04d}", "RECURRING {U} {d}", "VISA DDA PUR {n:05d} {U} {cityshort}",
        "PURCHASE AUTHORIZED ON {d} {U} {city} CARD {n:04d}"]


def amount(cat, rng):
    return round(math.exp(rng.gauss(AMOUNT_MU[cat], AMOUNT_SIG)), 2)


def render(m, rng, held=False):
    if not held:
        return T.render(m, rng)
    u = M._upper(m)
    return rng.choice(HELD).format(U=u, low=m["name"].lower(), n=rng.randint(1, 99999), city=m["city"], cityshort=m["city"].split()[0],
                                   d=f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}")


def scheme(kind, rng, world):
    """(categories [{name, name_type, standard}], merchant -> category index, rules {merchant: (B index, kind, param)})."""
    if kind == "custom":
        cats, assign = R6.make_scheme(rng, world)
    else:
        cats = [dict(name=DEFAULT[c], name_type="standard", standard=[c]) for c in M.CATEGORY_LIST]
        assign = {m["name"]: M.CATEGORY_LIST.index(m["category"]) for m in world}
        if kind == "light":
            for i in rng.sample(range(len(cats)), rng.randint(2, 3)):
                cats[i]["name"] = rng.choice(R6.RENAMES[cats[i]["standard"][0]]); cats[i]["name_type"] = "renamed"
    rules = {}
    if kind in ("light", "custom"):
        freq = sorted(world, key=lambda m: -m["freq"])[:30]
        for _ in range(rng.randint(1, 2) if kind == "light" else rng.randint(0, 1)):
            m = rng.choice(freq)
            if m["name"] in rules:
                continue
            name = rng.choice([p for p in PERSONAL if p not in [c["name"] for c in cats]])
            cats.append(dict(name=name, name_type="personal", standard=[]))
            b = len(cats) - 1
            rules[m["name"]] = (b, "amount", round(math.exp(AMOUNT_MU[m["category"]] + 0.3), 0)) if rng.random() < 0.5 else (b, "day", ("Fri", "Sat"))
        for m in world:  # idiosyncratic: about 8% of merchants under another category, consistently
            if m["name"] not in rules and rng.random() < 0.08:
                assign[m["name"]] = rng.choice([i for i in range(len(cats)) if i != assign[m["name"]] and cats[i]["standard"]])
    return cats, assign, rules


if __name__ == "__main__":
    if DST.exists() and "--force" not in sys.argv:
        sys.exit(f"{DST} exists (frozen); pass --force to rebuild")
    merchants = T.load()["merchants"]; db = R6.load()["fact_db"]
    std_idx = {c: i for i, c in enumerate(M.CATEGORY_LIST)}
    rng = random.Random(7); items, users = [], []
    for u in range(N_USERS):
        r = random.Random(f"real7-{u}")
        kind = "default" if r.random() < 0.55 else "light" if r.random() < 0.67 else "custom"
        world = r.sample(merchants, r.randint(50, 110))
        cats, assign, rules = scheme(kind, r, world)
        randsplit = {}
        if r.random() < 0.10:  # one merchant split at random 60 / 40 between its category and another
            m = r.choice([m for m in world if m["name"] not in rules])
            randsplit[m["name"]] = r.choice([i for i in range(len(cats)) if i != assign[m["name"]]])
        n_rows = r.randint(250, 400)
        weights = [m["freq"] for m in world]
        rows = []
        for k in range(n_rows):
            m = r.choices(world, weights)[0]; day = int(365 * k / n_rows) + r.randint(0, 1)
            rows.append(dict(merchant=m["name"], m=m, day=day, weekday=T.WEEKDAYS[day % 7], amount=amount(m["category"], r)))
        # events: a move and a new category, each at a time in the second half of the year
        move = newcat = None
        mids = [m for m in world if m["name"] not in rules and m["name"] not in randsplit]
        counts = Counter(x["merchant"] for x in rows)
        frequent = [m for m in mids if counts[m["name"]] >= 6]
        if frequent and r.random() < 0.25:
            m = r.choice(frequent); t = r.randint(200, 300)
            move = dict(merchant=m["name"], day=t, to=r.choice([i for i in range(len(cats)) if i != assign[m["name"]] and cats[i]["standard"]]))
        if frequent and r.random() < 0.20:
            m = r.choice([x for x in frequent if not move or x["name"] != move["merchant"]] or frequent); t = r.randint(200, 300)
            name = r.choice([n for n in NEW_CATS if n not in [c["name"] for c in cats]])
            cats.append(dict(name=name, name_type="event", standard=[])); newcat = dict(merchant=m["name"], day=t, cat=len(cats) - 1)

        def label(x):
            """(the intended category index, what decides it, the observable best answer's rule)."""
            name = x["merchant"]
            if name in rules:
                b, rk, p = rules[name]
                hit = x["amount"] >= p if rk == "amount" else x["weekday"] in p
                return (b if hit else assign[name]), f"rule_{rk}", (b if hit else assign[name])
            if name in randsplit:
                return (randsplit[name] if r.random() < 0.4 else assign[name]), "random_split", assign[name]
            if newcat and name == newcat["merchant"] and x["day"] >= newcat["day"]:
                return newcat["cat"], "new_category", None
            if move and name == move["merchant"] and x["day"] >= move["day"]:
                return move["to"], "moved", None
            return assign[name], "idiosyncratic" if cats[assign[name]]["standard"] and x["m"]["category"] not in cats[assign[name]]["standard"] else "plain", None
        for x in rows:
            x["gold"], x["why"], x["rule_best"] = label(x)
            x["filed"] = x["gold"] if r.random() >= 0.03 else r.choice([i for i in range(len(cats)) if i != x["gold"]])  # noise
            x["text"] = render(x["m"], r)
        # queries: 6 per user from the later part of the year; a quarter of users see a short history
        short = r.choice([0, 5, 25]) if r.random() < 0.25 else None
        start = max(40, n_rows // 2) if short is None else short
        if short is None:  # up to three queries on the user's interesting merchants (rules, moves, new categories, splits, idiosyncratic), the rest at random
            cand = list(range(start, n_rows)); inter = [k for k in cand if rows[k]["why"] != "plain"]
            pick = r.sample(inter, min(3, len(inter)))
            q_idx = sorted(pick + r.sample([k for k in cand if k not in pick], QUERIES - len(pick)))
        else:
            q_idx = list(range(short, short + QUERIES))
        for qi in q_idx:
            q = rows[qi]; hist = rows[:qi] if short is None else rows[qi - short:qi] if short else []
            if short is not None:
                hist = rows[:short] if short else []
            shots = hist[-SHOTS:]
            visible = [c for i, c in enumerate(cats) if not (newcat and i == newcat["cat"] and q["day"] < newcat["day"])]
            vis_idx = [i for i, c in enumerate(cats) if c in visible]
            if q["gold"] not in vis_idx:
                continue
            held = r.random() < 0.3
            qtext = render(q["m"], r, held=held)
            header = "Categories: " + ", ".join(c["name"] for c in visible) + "\n\n"
            demo = "".join(f"Transaction: {h['text']} | ${h['amount']:.2f} | {h['weekday']}\nCategory: {cats[h['filed']]['name']}\n\n" for h in shots)
            prompt = header + demo + f"Transaction: {qtext} | ${q['amount']:.2f} | {q['weekday']}\nCategory:"
            same = [h for h in shots if h["merchant"] == q["merchant"]]
            in_hist = any(h["merchant"] == q["merchant"] for h in hist)
            lookup = cats[same[-1]["filed"]]["name"] if same else None
            # the best answer from what the prompt shows
            if q["rule_best"] is not None and len(same) >= 2:  # a rule or split is readable only with two or more of the merchant's rows shown
                best = q["rule_best"]
            elif same:
                best = same[-1]["filed"]  # the latest visible filing (the recency rule; a misfiled latest row is not visible as such)
            else:
                std = [i for i in vis_idx if q["m"]["category"] in cats[i]["standard"]]
                best = std[0] if len(std) == 1 else (std[0] if std else None)
            options = [" " + c["name"] for c in visible]
            items.append(dict(id=f"R7:{u:03d}:{qi:03d}", level=f"R7_{kind}", user=u, merchant=q["merchant"], known=q["m"]["known"], text=qtext,
                              amount=q["amount"], weekday=q["weekday"], prompt=prompt, prompt_ctx=prompt.replace("\nCategory:", "", 0),
                              options=options, answer=vis_idx.index(q["gold"]), best=vis_idx.index(best) if best in vis_idx else -1,
                              scheme=kind, why=q["why"], in_shots=bool(same), in_history=in_hist, lookup_ok=lookup == cats[q["gold"]]["name"] if same else None,
                              hist_len="short_%d" % short if short is not None else "long", held_render=held, name_type=cats[q["gold"]]["name_type"],
                              seen=bool(same), record=db[q["merchant"]]))
        users.append(dict(user=u, scheme=kind, categories=cats, n_rows=n_rows, move=move, newcat=newcat, rules={k: list(v[:2]) for k, v in rules.items()},
                          randsplit=list(randsplit)))
    out = dict(name="real7", version="v1", n_users=N_USERS, users=users, items=items, sha256=R6.sha256(items))
    DST.write_text(json.dumps(out, indent=0, ensure_ascii=False))
    import numpy as np
    print(len(items), "items;", dict(Counter(i["scheme"] for i in items)), dict(Counter(i["why"] for i in items)))
    print("ceiling (best == answer):", round(100 * np.mean([i["best"] == i["answer"] for i in items]), 1))
    for key in ("scheme", "why", "hist_len"):
        print(f"  by {key}:", {k: round(100 * np.mean([i["best"] == i["answer"] for i in items if i[key] == k]), 1) for k in sorted({i[key] for i in items})})
    ls = [i for i in items if i["in_shots"]]
    print("merchant in the shots:", round(100 * len(ls) / len(items), 1), "% ; lookup of the latest shot right:", round(100 * np.mean([i["lookup_ok"] for i in ls]), 1))
    det = [i for i in items if not i["in_shots"] and i["why"] == "plain"]
    print("decided by the merchant's category (not in shots, plain):", round(100 * len(det) / len(items), 1), "%")
