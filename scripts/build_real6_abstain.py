"""REAL-6 with an abstain option (PLAN step 52, REAL-14): every item twice, as
  <id>:full    the user's categories as usual plus a last option " not listed here" (decider's neutral abstain wording); gold unchanged
  <id>:nogold  the gold category removed from the category list and the options (the shots filed under it keep their label: a hidden
               category), plus " not listed here", which is the answer
so one scoring pass gives the abstain option's false alarms (full) and its recall (nogold), and a reader trained without it can be read
by a confidence threshold over the real options instead (drop the last option's score). Same ids' users, so fold bookkeeping holds.
usage: uv run python scripts/build_real6_abstain.py [--force]
"""
import json
import sys

from ai_experiments import real6 as R6
from ai_experiments.paths import PROCESSED

DST = PROCESSED / "real6_v1_abst.json"
ABSTAIN = "not listed here"


def drop_from_header(p, name):
    head, rest = p.split("\n\n", 1)
    assert head.startswith("Categories: ")
    names = head[len("Categories: "):].split(", ")
    assert name in names
    return "Categories: " + ", ".join(n for n in names if n != name) + "\n\n" + rest


if __name__ == "__main__":
    if DST.exists() and "--force" not in sys.argv:
        sys.exit(f"{DST} exists (frozen); pass --force to rebuild")
    doc = R6.load()
    items = []
    for it in doc["items"]:
        items.append(dict(it, id=it["id"] + ":full", options=it["options"] + [" " + ABSTAIN], abst_case="full"))
        g = it["options"][it["answer"]].strip()
        opts = [o for k, o in enumerate(it["options"]) if k != it["answer"]] + [" " + ABSTAIN]
        items.append(dict(it, id=it["id"] + ":nogold", prompt=drop_from_header(it["prompt"], g), prompt_ctx=drop_from_header(it["prompt_ctx"], g),
                          options=opts, answer=len(opts) - 1, abst_case="nogold", hidden=g))
    out = dict(version="v1_abst", variant_of=f"real6 v1 ({doc['sha256'][:12]})", items=items, sha256=R6.sha256(items))
    DST.write_text(json.dumps(out, indent=0, ensure_ascii=False))
    print(len(items), "items;", items[1]["prompt"][:160].replace("\n", " / "), items[1]["options"])
