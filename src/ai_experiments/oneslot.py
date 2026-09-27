"""One-slot multiple choice in decider's plain layout with a choice of option labels (the owner, 2026-09-27: labels must not always
start at A, and must reach 255 options). decider's own builder (decider/prompt.py `build`, passed in as P) labels the shuffled options
A, B, C, ... in order; here each question draws its labels from decider's table of 255 single-token labels (A..Z, then the two-letter
upper-case strings that are one token):

  letters  decider's builder unchanged: A, B, C, ... after the shuffle (narrow "(A) text" rendering up to 10 options)
  rand26   a random sample of A..Z in random order (so the first option is rarely A and no letter marks a position)
  rand255  a random sample of all 255 labels in random order

Every scheme but `letters` renders the options wide ("\n(" + <label token> + ") text", built from ids so each label is one token).
`build` returns ids, the answer slot, the gold's position, the option permutation and the label token ids in the order shown; the
readout is the slot's hidden state against the output-embedding rows of those labels.
"""
SCHEMES = ("letters", "rand26", "rand255")


def build(P, tok, context, question, options, gold, rng, labels="letters", max_ctx_tokens=16384):
    if labels not in SCHEMES:
        raise ValueError(labels)
    if labels == "letters":
        class Q:
            def __init__(self):
                self.text, self.options, self.gold = question, options, gold

        class Ex:
            def __init__(self):
                self.context, self.qs = context, [Q()]
        b = P.build(Ex(), tok, rng=rng, max_options=255, max_ctx_tokens=max_ctx_tokens)
        n = b["nopts"][0]
        return dict(ids=b["ids"], slot=b["slots"][0], gold=b["golds"][0], perm=b["perms"][0], labs=list(P.letter_ids(tok)[:n]))
    _, lab_ids, open_ids = P.label_table(tok)
    opts = list(range(len(options))); rng.shuffle(opts)
    labs = rng.sample(list(lab_ids[:26 if labels == "rand26" else 255]), len(opts))
    ids = tok.encode("Context:\n" + context, add_special_tokens=False)[:max_ctx_tokens]
    ids += tok.encode(f"\n\nQuestion: {question}\nOptions:", add_special_tokens=False)
    for lab, oi in zip(labs, opts):
        ids += open_ids + [lab] + P._enc_opt(tok, options[oi])
    ids += tok.encode("\nAnswer: (", add_special_tokens=False)
    return dict(ids=ids, slot=len(ids) - 1, gold=opts.index(gold) if gold is not None and gold >= 0 else -1, perm=opts, labs=labs)
