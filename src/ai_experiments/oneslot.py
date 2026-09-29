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
    pool = 26 if labels == "rand26" and len(opts) <= 26 else 255  # rand26 with more than 26 options (blind_v1's detailed schemes) draws from all 255
    labs = rng.sample(list(lab_ids[:pool]), len(opts))
    ids = tok.encode("Context:\n" + context, add_special_tokens=False)[:max_ctx_tokens]
    ids += tok.encode(f"\n\nQuestion: {question}\nOptions:", add_special_tokens=False)
    for lab, oi in zip(labs, opts):
        ids += open_ids + [lab] + P._enc_opt(tok, options[oi])
    ids += tok.encode("\nAnswer: (", add_special_tokens=False)
    return dict(ids=ids, slot=len(ids) - 1, gold=opts.index(gold) if gold is not None and gold >= 0 else -1, perm=opts, labs=labs)


# ---- prompt layouts (PLAN step 111; owner, 2026-09-29) -------------------------------------------------------------------------------
# The context the item sets and episode builder produce is "Categories: a, b, ...\n\n" + rows "Transaction: <fields>\nCategory: <name>"
# separated by blank lines + the query row "Transaction: <fields>". `build_layout` re-renders it:
#   options          the layout above (Context, Question, Options, "Answer: (") from the same parse, for the dow_first variant
#   labelled         the labelled category list once, at the top ("Categories:\n- (FX) Transportation\n..."), rows as they were, and the
#                    query ending "\nCategory: (" (no Question line, no Options block)
#   labelled_shots   as labelled, and every row's category written with its label: "Category: (FX) Transportation"
#   table            (owner, 2026-09-29) the history as one TSV table inside <historical_transactions> with a header row (date, weekday,
#                    description, amount, category: the category last; no date column when no row is dated), the query as a one-row
#                    table inside <new_transaction> with its category cell "?"; then the Question / Options / "Answer: (" of `options`
#   table_labelled   the labelled category list at the top, the category cells written " (FX) Transportation", the query's category cell
#                    open: "... <amount>\t (" with the answer slot there. A cell starts with a space: "\t(" + label merges into "(A" + "E"
#                    for 209 of the 255 labels, "\t (" + label is clean for all of them.
# dow_first moves the weekday next to the date: "Transaction: 2025-01-19 | Fri | TST* BARTELL DRUGS | $60.43".
# Tokenization (checked on the Qwen3.5 tokenizer for all 255 labels): after "- (" and "Category: (" every label is one token, the same
# id as the answer slot reads; "\n(AE" at a line start would merge into "(A" + "E", which this layout never writes.
# AUX (the shot-label loss) comes back as (position, target) pairs: the tokens of each row's label (the label token and the name) whose
# original label span is in `spans` (misfiled rows are not).
import re as _re

_DATE = _re.compile(r"^\d{4}-\d\d-\d\d$")


def dow_first(fields):
    """'2025-01-19 | TEXT | $60.43 | Fri' -> '2025-01-19 | Fri | TEXT | $60.43'; undated 'TEXT | $1 | Fri' -> 'Fri | TEXT | $1'."""
    parts = fields.split(" | ")
    if len(parts) < 3:
        return fields
    dow = parts[-1]
    if _DATE.match(parts[0]):
        return " | ".join([parts[0], dow] + parts[1:-1])
    return " | ".join([dow] + parts[:-1])


def parse(context):
    """(category names from the header, rows [(fields, label, label char start)], query fields); raises on another layout."""
    head, _, rest = context.partition("\n\n")
    assert head.startswith("Categories: "), head[:40]
    names = head[len("Categories: "):].split(", ")
    blocks = rest.split("\n\n") if rest else []
    rows, pos = [], len(head) + 2
    for blk in blocks[:-1]:
        t, _, c = blk.partition("\nCategory:")
        assert t.startswith("Transaction: "), blk[:60]
        rows.append((t[len("Transaction: "):], c.strip(), pos + len(t) + len("\nCategory:") + 1))
        pos += len(blk) + 2
    q = blocks[-1] if blocks else ""
    assert q.startswith("Transaction: ") and "\n" not in q, q[:80]
    return names, rows, q[len("Transaction: "):]


def build_layout(P, tok, context, question, options, gold, rng, labels="rand255", layout="labelled_shots", dow=False, spans=None):
    names, rows, query = parse(context)
    _, lab_ids, open_ids = P.label_table(tok)
    opts = list(range(len(options))); rng.shuffle(opts)
    pool = 26 if labels == "rand26" and len(opts) <= 26 else 255
    labs = rng.sample(list(lab_ids[:pool]), len(opts))
    lab_of = {options[oi]: tok.decode([lab]) for lab, oi in zip(labs, opts)}
    f = dow_first if dow else (lambda x: x)
    in_loss = (lambda st: any(s <= st < e for s, e in spans)) if spans is not None else (lambda st: False)
    parts, targets = [], []  # text pieces; char ranges of trained label text
    if layout in ("table", "table_labelled"):
        return _build_table(P, tok, names, rows, query, question, options, gold, opts, labs, lab_of, open_ids, layout, in_loss)
    if layout == "options":
        parts.append("Context:\nCategories: " + ", ".join(names) + "\n\n")
    else:
        parts.append("Categories:\n" + "".join(f"- ({lab_of[options[oi]]}) {options[oi]}\n" for oi in opts) + "\n")
    for fields, label, st in rows:
        parts.append(f"Transaction: {f(fields)}\nCategory:")
        lab = f" ({lab_of[label]}) {label}" if layout == "labelled_shots" and label in lab_of else f" {label}"
        if in_loss(st):
            n = sum(len(p) for p in parts); targets.append((n, n + len(lab)))
        parts.append(lab + "\n\n")
    parts.append(f"Transaction: {f(query)}")
    text = "".join(parts)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = list(enc["input_ids"]), enc["offset_mapping"]
    aux = [(t - 1, ids[t]) for t, (a, b) in enumerate(offs) if t > 0 and any(s < b and a < e for s, e in targets)]
    if layout == "options":
        ids += tok.encode(f"\n\nQuestion: {question}\nOptions:", add_special_tokens=False)
        for lab, oi in zip(labs, opts):
            ids += open_ids + [lab] + P._enc_opt(tok, options[oi])
        ids += tok.encode("\nAnswer: (", add_special_tokens=False)
    else:
        ids += tok.encode("\nCategory: (", add_special_tokens=False)
        head = tok.encode("- (" + lab_of[options[opts[0]]] + ")", add_special_tokens=False)
        assert labs[0] in head, "a label does not tokenize as itself after '- ('"
    return dict(ids=ids, slot=len(ids) - 1, gold=opts.index(gold) if gold is not None and gold >= 0 else -1, perm=opts, labs=labs, aux=aux)


def _cells(fields):
    """(date or '', weekday, description, amount) from a row's fields ('2025-01-19 | TEXT | $60.43 | Fri' or 'TEXT | $60.43 | Fri')."""
    parts = fields.split(" | ")
    date = parts[0] if _DATE.match(parts[0]) else ""
    rest = parts[1:] if date else parts
    return date, rest[-1], " | ".join(rest[:-2]), rest[-2].replace("$", "")


def _build_table(P, tok, names, rows, query, question, options, gold, opts, labs, lab_of, open_ids, layout, in_loss):
    dated = bool(_cells(query)[0]) or any(_cells(f)[0] for f, _, _ in rows)
    head = ("date\t" if dated else "") + "weekday\tdescription\tamount\tcategory\n"
    line = lambda f: "\t".join(([_cells(f)[0]] if dated else []) + list(_cells(f)[1:])) + "\t"  # noqa: E731
    parts, targets = [], []
    if layout == "table":
        parts.append("Context:\nCategories: " + ", ".join(names) + "\n\n")
    else:
        parts.append("Categories:\n" + "".join(f"- ({lab_of[options[oi]]}) {options[oi]}\n" for oi in opts) + "\n")
    parts.append("<historical_transactions>\n" + head)
    for fields, label, st in rows:
        parts.append(line(fields))
        cell = f" ({lab_of[label]}) {label}" if layout == "table_labelled" and label in lab_of else f" {label}"
        if in_loss(st):
            n = sum(len(p) for p in parts); targets.append((n, n + len(cell)))
        parts.append(cell + "\n")
    parts.append("</historical_transactions>\n\n<new_transaction>\n" + head + line(query))
    if layout == "table":
        parts.append(" ?\n</new_transaction>")
    text = "".join(parts)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = list(enc["input_ids"]), enc["offset_mapping"]
    aux = [(t - 1, ids[t]) for t, (a, b) in enumerate(offs) if t > 0 and any(s < b and a < e for s, e in targets)]
    if layout == "table":
        ids += tok.encode(f"\n\nQuestion: {question}\nOptions:", add_special_tokens=False)
        for lab, oi in zip(labs, opts):
            ids += open_ids + [lab] + P._enc_opt(tok, options[oi])
        ids += tok.encode("\nAnswer: (", add_special_tokens=False)
    else:
        ids += tok.encode(" (", add_special_tokens=False)  # the open category cell: the answer slot reads the label token after " ("
    return dict(ids=ids, slot=len(ids) - 1, gold=opts.index(gold) if gold is not None and gold >= 0 else -1, perm=opts, labs=labs, aux=aux)
