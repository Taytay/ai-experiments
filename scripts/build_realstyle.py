"""PLAN step 173: training episodes and a test set from real-style synthetic households (ai_experiments.realstyle).

Each household is simulated, written as a YNAB budget document, and turned into items by real_budget_eval.build_items (the same split
layout, history rules and options as the owner's budget: only earlier dates, a 24-row shared block per day, the payee's own rows).
Items are sampled from transactions with at least MIN_HIST earlier rows.

  train   seeds 0..N-1 on the "train" merchant split: data/processed/realstyle_v1_train.jsonl, one episode per line
          {context, options, answer, spans (character ranges of the history rows' category labels, for the shot-label loss), household,
          kind, payee_seen}; exp_decider_finetune.py REALSTYLE=<share> mixes them in
  test    seeds 100000.. on the "test" merchant split (merchants never in training): data/processed/realstyle_v1_test.json, items in
          real_budget_eval's format, scored with it (OUT=<dir> with items.json)
usage: uv run python scripts/build_realstyle.py train 400 100     (households, items per household)
       uv run python scripts/build_realstyle.py test 150 60
"""
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("BUDGET", "realstyle")
sys.path.insert(0, str(Path(__file__).parent))
import real_budget_eval as RB  # noqa: E402

from ai_experiments import oneslot  # noqa: E402
from ai_experiments import realstyle as R  # noqa: E402
from ai_experiments.paths import PROCESSED  # noqa: E402

MIN_HIST = 30


def label_spans(context):
    """Character ranges of the history rows' category labels in the context (oneslot.parse's label starts)."""
    _, rows, _ = oneslot.parse(context)
    return [(st, st + len(lab)) for _, lab, st in rows if lab is not None]


def household_items(seed, split, per):
    h = R.household(seed, split)
    items = [it for it in RB.build_items(h.budget()["budget"]) if it["answer"] >= 0 and it["n_hist"] >= MIN_HIST]
    rng = random.Random(seed)
    return rng.sample(items, min(per, len(items)))


if __name__ == "__main__":
    mode, n, per = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    if mode == "train":
        out = PROCESSED / os.environ.get("TRAIN_OUT", "realstyle_v1_train.jsonl")  # row 176: realstyle_v2_train.jsonl, with FIELDS=1 realstyle_v2f_train.jsonl
        k = 0
        with open(out, "w") as f:
            for seed in range(n):
                for it in household_items(seed, "train", per):
                    ctx = it["prompt_split"][: -len("Category:")].rstrip()
                    f.write(json.dumps(dict(context=ctx, options=it["options"], answer=it["answer"], spans=label_spans(ctx), household=seed,
                                            kind=it["kind"], payee_seen=it["payee_seen"]) | ({"rationale": it["rationale"]} if it.get("rationale") else {})) + "\n")
                    k += 1
                if seed % 50 == 0:
                    print(f"  household {seed}: {k} episodes", flush=True)
        print(f"{k} training episodes from {n} households -> {out} ({out.stat().st_size / 1e6:.0f} MB)")
    else:
        items = []
        split, seed0 = os.environ.get("TEST_SPLIT", "test"), int(os.environ.get("TEST_SEED0", "100000"))  # row 183: TEST_SPLIT=train
        for i in range(n):  # with TEST_SEED0=300000: held-out households in the training world (merchants other users filed in training)
            seed = seed0 + i
            for it in household_items(seed, split, per):
                it["id"] = f"RS:{seed}:{it['id'][:8]}"; it["user"] = seed
                if os.environ.get("TEST_LAYOUT") == "split":  # row 183: read in the training layout (exp_decision_models reads "prompt")
                    it["prompt"] = it["prompt_ctx"] = it["prompt_split"]; it["options"] = [" " + o for o in it["options"]]
                items.append(it)
        out = PROCESSED / os.environ.get("TEST_OUT", "realstyle_v1_test.json")
        out.write_text(json.dumps(dict(name="realstyle_v1_test", version=1, n=len(items), households=n, items=items)))
        print(f"{len(items)} test items from {n} households -> {out} ({out.stat().st_size / 1e6:.0f} MB); first-time payees "
              f"{100 * sum(not it['payee_seen'] for it in items) / len(items):.1f}%")
