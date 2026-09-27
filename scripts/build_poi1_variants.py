"""Variants of POI-1 (PLAN step 65) that change only the prompt, item for item, for scoring without retraining.

  kshots        kind-retrieved shots: up to 6 of the user's history places of the query's Overture basic category (what a places database
                tells a system about the query) replace frozen shots, the rest of the 24 frozen shots kept in order; when the history has none
                of that kind the prompt is unchanged. The history is the user's own, so this is retrieval over their past filings, not a label leak.
  kinds         (row 72) the frozen 24 shots, with every shot and the query given a "Kind: <Overture basic category>" line: the places
                database's answer for each line, so matching the query to a same-kind shot is one hop (REPORT 65: one pass follows about one
                hop it was not trained on) instead of two (record -> kind -> a shot the model must itself recognise as that kind).
  kshots_kinds  (row 72) both.
  desc          (row 73) the header lists each category with the Overture kinds of the places the user has filed under it, most filed
                first, at most four ("Gavir: counselling, psychology"; "(nothing filed yet)" when none): what the user's own history says each
                name means, so an unseen kind can be matched to a category through a related kind. desc_kinds, desc_kshots_kinds combine.
  instr_...     (row 73) an instruction line before the header (INSTR), for untrained readers only: does naming the task help them?
usage: uv run python scripts/build_poi1_variants.py kshots|kinds|kshots_kinds|desc|desc_kinds|desc_kshots_kinds|instr_kshots_kinds|... [--force]
"""
import json
import random
import sys

from ai_experiments import real6 as R6
from ai_experiments.paths import PROCESSED

SRC = PROCESSED / "poi1_v1.json"
K = 6
INSTR = "Choose the category this user would file the last transaction under.\n\n"


def kind_name(basic):
    """Overture basic category as the prompt shows it (build_poi1.readable)."""
    return basic.replace("_or_", " or ").replace("_and_", " and ").replace("_", " ").capitalize()


def line(s, kinds):
    return f"Transaction: {s['text']} | ${s['amount']:.2f} | {s['weekday']}\n" + (f"Kind: {kind_name(s['basic'])}\n" if kinds else "")


def described(u, n=4):
    """The header with each category's kinds from the user's whole history (row 73)."""
    from collections import Counter
    cnt = {c["name"]: Counter() for c in u["categories"]}
    for h in u["history"]:
        cnt[h["label"]][h["basic"]] += 1
    rows = [f"- {c['name']}: " + (", ".join(kind_name(b).lower() for b, _ in cnt[c["name"]].most_common(n)) or "(nothing filed yet)") for c in u["categories"]]
    return "Categories:\n" + "\n".join(rows) + "\n\n"


def variant(doc, retrieve, kinds, desc=False, instr=False):
    users = {u["user"]: u for u in doc["users"]}
    out = []
    for it in doc["items"]:
        u = users[it["user"]]; rng = random.Random(f"{it['id']}-kshots")
        by_text = {h["text"]: h for h in u["history"]}
        if retrieve:
            same = [h for h in u["history"] if h["basic"] == it["basic"]]
            rng.shuffle(same); same = same[:K]
            frozen = [by_text[t] for t in u["shots"] if by_text[t]["basic"] != it["basic"]]
            shots = same + frozen[:len(u["shots"]) - len(same)]
            rng.shuffle(shots)
        else:
            same, shots = [], [by_text[t] for t in u["shots"]]
        header = (INSTR if instr else "") + (described(u) if desc else it["prompt"].split("\n\n", 1)[0] + "\n\n")
        demo = "".join(line(s, kinds) + f"Category: {s['label']}\n\n" for s in shots)
        q = it["prompt"].rsplit("\n\n", 1)[1]
        assert q.startswith("Transaction: ") and q.endswith("Category:")
        if kinds:
            q = line(it, True) + "Category:"
        record = f"Note: {it['record']}\n"
        extra = dict(n_kind_shots=sum(s["basic"] == it["basic"] for s in shots)) if retrieve or kinds or desc else {}
        out.append(dict(it, prompt=header + demo + q, prompt_ctx=header + demo + record + q, **extra))
    return out


if __name__ == "__main__":
    which = sys.argv[1]
    dst = PROCESSED / f"poi1_v1_{which}.json"
    if dst.exists() and "--force" not in sys.argv:
        sys.exit(f"{dst} exists (frozen); pass --force to rebuild")
    doc = json.loads(SRC.read_text())
    items = variant(doc, retrieve="kshots" in which, kinds="kinds" in which, desc="desc" in which, instr="instr" in which)
    # the users' frozen shots stay as they were: the scorer's flags read them, the prompts carry the retrieved ones
    new = dict(doc, version=f"v1_{which}", items=items, sha256=R6.sha256(items), variant_of=f"poi1_v1 ({doc['sha256'][:12]})")
    dst.write_text(json.dumps(new, indent=0, ensure_ascii=False))
    from collections import Counter
    print(len(items), "items; kind shots per item:", sorted(Counter(i.get("n_kind_shots") for i in items).items()))
