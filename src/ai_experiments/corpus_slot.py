"""Row 154 (MODEL-24): strands-decider's training rows (strands_decider.data.format.Example as JSON: kind noul | choice | score, state,
instructions, options [[name, description]], label, task, weight, instruction_variants) rendered for decider's one-slot readout: decider's
plain layout ("Context:\\n<state>\\n\\nQuestion: <instructions>\\nOptions:\\n(A) <name> — <description>\\n...\\nAnswer: ("), labels in
decider's own scheme (A, B, ... narrow up to 10 options, one-token wide labels beyond), the answer read as the label tokens' logits from
the LM head at the final "(". Option order as strands' collator does it: choice and noul shuffled, score levels in order or reversed.
The target over the options as shown: one-hot, or for score questions the ordinal smoothing strands trains with (mass `smooth` split
between the gold level's neighbours).
"""
import random

STATE_TOKENS = 4096  # strands' serving window; no training row is longer than 3,072 tokens


class _Keep(random.Random):
    """An rng whose shuffle is a no-op, so decider's builder keeps the order given."""

    def shuffle(self, x):  # noqa: D401
        return None


def option_text(name, desc):
    desc = " ".join((desc or "").split())
    return f"{name} — {desc}" if desc else name


def state_text(state):
    if isinstance(state, str):
        return state.strip()
    import json
    return json.dumps(state, indent=2, ensure_ascii=False)


def render(P, tok, row, rng, train=True, reverse_score=0.5):
    """dict(ids, slot, labs (label token ids in the order shown), order (canonical option index shown at each position), gold (its
    position), instr) for one Example row. At eval (train=False) choice / noul options keep a fixed shuffle from rng and scores stay in
    order, so every reader sees the same rendering."""
    n = len(row["options"])
    order = list(range(n))
    if row["kind"] == "score":
        if train and rng.random() < reverse_score:
            order.reverse()
    else:
        rng.shuffle(order)
    instr = row["instructions"]
    if train and row.get("instruction_variants"):
        instr = rng.choice([instr] + list(row["instruction_variants"]))
    shown = [option_text(*row["options"][k]) for k in order]

    class Q:
        text, options, gold = instr, shown, order.index(row["label"])

    class Ex:
        context, qs = state_text(row["state"]), [Q()]

    b = P.build(Ex(), tok, rng=_Keep(), max_options=255, max_ctx_tokens=STATE_TOKENS)
    assert b["perms"][0] == list(range(n))
    return dict(ids=b["ids"], slot=b["slots"][0], labs=list(P.letter_ids(tok)[:n]), order=order, gold=b["golds"][0], instr=instr)


def target(row, order, smooth=0.1):
    """The training target over the options in the order shown."""
    n, g = len(order), row["label"]
    t = [0.0] * n
    if row["kind"] == "score" and smooth > 0:
        nb = [lv for lv in (g - 1, g + 1) if 0 <= lv < n]
        mass = {g: 1.0 - smooth if nb else 1.0}
        for lv in nb:
            mass[lv] = smooth / len(nb)
        for j, k in enumerate(order):
            t[j] = mass.get(k, 0.0)
    else:
        t[order.index(g)] = 1.0
    return t
