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
#   json_labelled    (owner, 2026-09-29) the labelled list at the top, the history as JSON objects one per line in <historical_transactions>
#                    ({"date", "description", "amount", "weekday", "category": "(FX) Transportation"}), the query as one object in
#                    <new_transaction> ending '"category": "(' (the label follows the token ' "(' cleanly for all 255 labels)
#   yaml_labelled    the same as a YAML list of records ("  category: (FX) Transportation"; descriptions quoted), the query ending "  category: ("
#   ts_labelled      (owner, 2026-09-29) TypeScript: the categories as a string-literal union ('  | "FX" /* Transportation */'), an
#                    interface Transaction, 'const history: Transaction[] = [ { ..., category: "GU" /* Medical */ }, ... ];' and
#                    'const next: Transaction = { ..., category: "' with the answer slot there. Enum member access ("Category.GU") would
#                    merge ".G" for 245 of 255 labels and enum keys read " GU": the quoted literal is clean for all 255.
#   ts_enum          (owner, 2026-09-29) TypeScript with the owner's enum: 'enum Categories { "GU" = "Medical", ... }' (quoted member names:
#                    bare keys read " GU"), 'const transactions: Array<Transaction> = [ { ..., category: Categories["GU"] }, ... ];' and
#                    'const next: Transaction = { ..., category: Categories["' (bracket access: "Categories.GU" merges ".G" for 245 labels;
#                    '["' + label is clean for all 255); no names in the rows
#   ts_enum_names    the same with each row's category name as a comment: 'category: Categories["GU"] /* Medical */'
# dow_first moves the weekday next to the date: "Transaction: 2025-01-19 | Fri | TST* BARTELL DRUGS | $60.43".
# Tokenization (checked on the Qwen3.5 tokenizer for all 255 labels): after "- (" and "Category: (" every label is one token, the same
# id as the answer slot reads; "\n(AE" at a line start would merge into "(A" + "E", which this layout never writes.
# AUX (the shot-label loss) comes back as (position, target) pairs: the tokens of each row's label (the label token and the name) whose
# original label span is in `spans` (misfiled rows are not).
import json
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


SPLIT_HEADS = ("Earlier transactions:", "Earlier transactions at this payee and similar payees:")  # row 117 (build_blind_v1 BLIND_SPLIT / BULK)
_KEY_SKIP = {"SQ", "TST", "POS", "DEBIT", "PURCHASE", "AUTHORIZED", "ON", "DBT", "CRD", "PP", "SP", "PY", "CLV", "ACH", "CHECKCARD", "CARD",
             "RECURRING", "ONLINE", "PMT", "PAYMENT", "WWW", "COM", "THE"}


def payee_key(fields):
    """Row 118: a crude payee key from a row's description (prefixes and digits dropped, the first word), for splitting training
    episodes; production has the payee itself."""
    d = _cells(fields)[2].upper()
    words = [w for w in _re.sub(r"[^A-Z ]", " ", d).split() if w not in _KEY_SKIP and len(w) > 1]
    return words[0] if words else ""


def split_rows(rows, query, rng, p_extra=0.5, max_extra=4):
    """Row 118 (SPLIT=1 training): the rows as a shared block, then the query payee's rows (by payee_key) and, with probability p_extra,
    1 to max_extra other rows (the test layout's near block also carries similar payees), each block in its original order, under
    SPLIT_HEADS. Header rows are (text, None, None)."""
    k = payee_key(query)
    near = {i for i, (f, _, _) in enumerate(rows) if k and payee_key(f) == k}
    rest = [i for i in range(len(rows)) if i not in near]
    if rest and rng.random() < p_extra:
        near |= set(rng.sample(rest, min(len(rest), rng.randint(1, max_extra))))
    return ([(SPLIT_HEADS[0], None, None)] + [rows[i] for i in range(len(rows)) if i not in near]
            + [(SPLIT_HEADS[1], None, None)] + [rows[i] for i in sorted(near)])


def payee_name(fields):
    """Row 136: a short readable payee name from a row's description (prefixes, digits and store numbers dropped, two words)."""
    d = _cells(fields)[2]
    words = [w for w in _re.sub(r"[^A-Za-z&' ]", " ", d).split() if w.upper() not in _KEY_SKIP and len(w) > 1]
    return " ".join(words[:2]).title()


def describe_categories(names, rows, k=3):
    """Row 136 (owner, 2026-09-30, after Jeff: options described by what they lead to): per category, the payees filed under it in this
    prompt's rows (most recent first, up to k distinct), or "nothing filed yet"."""
    seen = {n: [] for n in names}
    for fields, label, _ in reversed(rows):
        if label is None or label not in seen:
            continue
        nm = payee_name(fields)
        if nm and nm not in seen[label] and len(seen[label]) < k:
            seen[label].append(nm)
    return {n: ("e.g. " + ", ".join(v) if v else "nothing filed yet") for n, v in seen.items()}


def parse(context):
    """(category names from the header, rows [(fields, label, label char start)], query fields); raises on another layout. A one-line
    block that is not a transaction (row 117's section headers) comes back as (text, None, None)."""
    head, _, rest = context.partition("\n\n")
    assert head.startswith("Categories: "), head[:40]
    names = head[len("Categories: "):].split(", ")
    blocks = rest.split("\n\n") if rest else []
    rows, pos = [], len(head) + 2
    for blk in blocks[:-1]:
        if not blk.startswith("Transaction: ") and "\n" not in blk:
            rows.append((blk, None, None)); pos += len(blk) + 2
            continue
        t, _, c = blk.partition("\nCategory:")
        assert t.startswith("Transaction: "), blk[:60]
        rows.append((t[len("Transaction: "):], c.strip(), pos + len(t) + len("\nCategory:") + 1))
        pos += len(blk) + 2
    q = blocks[-1] if blocks else ""
    assert q.startswith("Transaction: ") and all(x.startswith(("Kind: ", "Others filed this payee as: ")) for x in q.split("\n")[1:]), q[:80]  # row 149: a Kind line; row 183: a crowd line
    return names, rows, q[len("Transaction: "):]


def build_layout(P, tok, context, question, options, gold, rng, labels="rand255", layout="labelled_shots", dow=False, spans=None, split=False, desc=False,
                 relist=False, field_loss=False, rationale=None):
    names, rows, query = parse(context)
    dsc = (desc if isinstance(desc, dict) else describe_categories(options, rows)) if desc else None  # row 136; a dict: descriptions given by the caller
    if split and not any(c is None for _, c, _ in rows):  # row 118: training episodes into the split layout
        rows = split_rows(rows, query, rng)
    if any(c is None for _, c, _ in rows):
        assert layout in ("options", "labelled", "labelled_shots"), f"section headers are only rendered by the text layouts, not {layout}"
    _, lab_ids, open_ids = P.label_table(tok)
    opts = list(range(len(options))); rng.shuffle(opts)
    pool = 26 if labels == "rand26" and len(opts) <= 26 else 255
    labs = rng.sample(list(lab_ids[:pool]), len(opts))
    lab_of = {options[oi]: tok.decode([lab]) for lab, oi in zip(labs, opts)}
    f = dow_first if dow else (lambda x: x)
    in_loss = (lambda st: any(s <= st < e for s, e in spans)) if spans is not None else (lambda st: False)
    parts, targets = [], []  # text pieces; char ranges of trained label text
    if layout in ("ts_enum", "ts_enum_names"):
        body, targets = records_text(layout, rows, query, lab_of, in_loss)
        dated = bool(_cells(query)[0]) or any(_cells(f)[0] for f, _, _ in rows)
        head = ("enum Categories {\n" + "".join(f'  "{lab_of[options[oi]]}" = {json.dumps(options[oi], ensure_ascii=False)},\n' for oi in opts) + "}\n\n"
                "interface Transaction { " + ("date: string; " if dated else "") + "description: string; amount: number; weekday: string; category: Categories; }\n\n")
        text = head + body; targets = [(a + len(head), b + len(head)) for a, b in targets]
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = list(enc["input_ids"]), enc["offset_mapping"]
        aux = [(t - 1, ids[t]) for t, (a, b) in enumerate(offs) if t > 0 and any(s < b and a < e for s, e in targets)]
        return dict(ids=ids, slot=len(ids) - 1, gold=opts.index(gold) if gold is not None and gold >= 0 else -1, perm=opts, labs=labs, aux=aux)
    if layout == "ts_labelled":
        body, targets = records_text("ts", rows, query, lab_of, in_loss)
        dated = bool(_cells(query)[0]) or any(_cells(f)[0] for f, _, _ in rows)
        head = ("type CategoryId =\n" + "".join(f'  | "{lab_of[options[oi]]}" /* {options[oi]} */\n' for oi in opts) + ";\n\n"
                "interface Transaction { " + ("date: string; " if dated else "") + "description: string; amount: number; weekday: string; category: CategoryId; }\n\n")
        text = head + body; targets = [(a + len(head), b + len(head)) for a, b in targets]
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = list(enc["input_ids"]), enc["offset_mapping"]
        aux = [(t - 1, ids[t]) for t, (a, b) in enumerate(offs) if t > 0 and any(s < b and a < e for s, e in targets)]
        return dict(ids=ids, slot=len(ids) - 1, gold=opts.index(gold) if gold is not None and gold >= 0 else -1, perm=opts, labs=labs, aux=aux)
    if layout in ("json_labelled", "yaml_labelled"):
        body, targets = records_text(layout[:4], rows, query, lab_of, in_loss)
        head = "Categories:\n" + "".join(f"- ({lab_of[options[oi]]}) {options[oi]}\n" for oi in opts) + "\n"
        text = head + body; targets = [(a + len(head), b + len(head)) for a, b in targets]
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = list(enc["input_ids"]), enc["offset_mapping"]
        aux = [(t - 1, ids[t]) for t, (a, b) in enumerate(offs) if t > 0 and any(s < b and a < e for s, e in targets)]
        return dict(ids=ids, slot=len(ids) - 1, gold=opts.index(gold) if gold is not None and gold >= 0 else -1, perm=opts, labs=labs, aux=aux)
    if layout in ("table", "table_labelled"):
        return _build_table(P, tok, names, rows, query, question, options, gold, opts, labs, lab_of, open_ids, layout, in_loss)
    if layout == "options":
        parts.append("Context:\nCategories: " + ", ".join(names) + "\n\n")
    else:
        parts.append("Categories:\n" + "".join(f"- ({lab_of[options[oi]]}) {options[oi]}" + (f" ({dsc[options[oi]]})" if dsc else "") + "\n" for oi in opts) + "\n")
    for fields, label, st in rows:
        if label is None:  # row 117: a section header
            parts.append(fields + "\n\n")
            continue
        if field_loss and "\n" in fields:  # row 176: a history row's whole field block is generated: every line (key, value, line break), so
            # the model learns which fields are present, and the "\nCategory:" after it, so it learns where the block ends
            n0 = sum(len(p) for p in parts) + len("Transaction: ")
            first = fields.split("\n")[0]
            targets.append((n0 + len(first), n0 + len(fields) + len("\nCategory:")))
        parts.append(f"Transaction: {f(fields)}\nCategory:")
        lab = f" ({lab_of[label]}) {label}" if layout == "labelled_shots" and label in lab_of else f" {label}"
        if in_loss(st):
            n = sum(len(p) for p in parts); targets.append((n, n + len(lab)))
        parts.append(lab + "\n\n")
    parts.append(f"Transaction: {f(query)}")
    line_spans = []
    if relist:  # row 152: the options listed again after the query, so each line's last token has read the history (a pointer key)
        assert layout == "labelled_shots", "relist is for the labelled_shots layout"
        parts.append("\n\nOptions:")
        for oi in opts:
            n = sum(len(p) for p in parts) + 1
            line = f"- ({lab_of[options[oi]]}) {options[oi]}"
            parts.append("\n" + line); line_spans.append((n, n + len(line)))
    text = "".join(parts)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = list(enc["input_ids"]), enc["offset_mapping"]
    aux = [(t - 1, ids[t]) for t, (a, b) in enumerate(offs) if t > 0 and any(s < b and a < e for s, e in targets)]
    opt_pos = [max(t for t, (a, b) in enumerate(offs) if b > a and a >= s0 and b <= e0) for s0, e0 in line_spans]
    if layout == "options":
        ids += tok.encode(f"\n\nQuestion: {question}\nOptions:", add_special_tokens=False)
        for lab, oi in zip(labs, opts):
            ids += open_ids + [lab] + P._enc_opt(tok, options[oi])
        ids += tok.encode("\nAnswer: (", add_special_tokens=False)
    else:
        ids += tok.encode("\nCategory: (", add_special_tokens=False)
        head = tok.encode("- (" + lab_of[options[opts[0]]] + ")", add_special_tokens=False)
        assert labs[0] in head, "a label does not tokenize as itself after '- ('"
    slot = len(ids) - 1
    if rationale and layout != "options" and gold is not None and gold >= 0:  # row 177: the answer, then "Because: ..." after the slot, trained as
        # tokens (the slot's own logits are unchanged: later tokens cannot reach it); the label token itself stays the slot's classification target
        ids.append(labs[opts.index(gold)])
        tail = tok.encode(f") {options[gold]}\nBecause: {rationale}", add_special_tokens=False)
        start = len(ids)
        ids += tail
        aux = aux + [(t - 1, ids[t]) for t in range(start, len(ids))]
    return dict(ids=ids, slot=slot, gold=opts.index(gold) if gold is not None and gold >= 0 else -1, perm=opts, labs=labs, aux=aux,
                opt_pos=opt_pos)


def _cells(fields):
    """(date or '', weekday, description, amount) from a row's fields ('2025-01-19 | TEXT | $60.43 | Fri' or 'TEXT | $60.43 | Fri');
    lines after the first (row 149's "Kind: ...", row 176's field lines) are ignored."""
    parts = _re.sub(r"^\[\d+\] ", "", fields.split("\n")[0]).split(" | ")  # row 177: a leading row number "[12] " is not a cell; extra lines under a row (row 149's Kind, row 176's Clean payee / Via / ...) are not cells
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


def records_text(fmt, rows, query, lab_of, in_loss=lambda st: False):
    """Row 113: the history and the query as JSON lines or a YAML list (text ending where the query's label goes), and the char ranges of
    the trained category values. lab_of: category name -> label (an empty dict writes names without labels)."""
    import json as _json
    dated = bool(_cells(query)[0]) or any(_cells(f)[0] for f, _, _ in rows)
    cat = lambda c: (f"({lab_of[c]}) {c}" if c in lab_of else c)  # noqa: E731
    if fmt == "ts":
        cat = lambda c: (f'{lab_of[c]}" /* {c} */' if c in lab_of else f'{c}"')  # noqa: E731
    if fmt in ("ts_enum", "ts_enum_names"):
        cat = lambda c: (f'{lab_of[c]}"]' + (f" /* {c} */" if fmt == "ts_enum_names" else "") if c in lab_of else f'{c}"]')  # noqa: E731
    parts, targets = ["const history: Transaction[] = [\n" if fmt == "ts" else "const transactions: Array<Transaction> = [\n" if fmt.startswith("ts_enum")
                      else "<historical_transactions>\n"], []

    def rec(f, c=None):
        d, dow, desc, amt = _cells(f)
        if fmt == "ts":
            head = "  { " + (f'date: "{d}", ' if dated else "") + f'description: {_json.dumps(desc, ensure_ascii=False)}, amount: {amt}, weekday: "{dow}", category: "'
            return head, (cat(c) + " },\n") if c is not None else None
        if fmt.startswith("ts_enum"):
            head = "  { " + (f'date: "{d}", ' if dated else "") + f'description: {_json.dumps(desc, ensure_ascii=False)}, amount: {amt}, weekday: "{dow}", category: Categories["'
            return head, (cat(c) + " },\n") if c is not None else None
        if fmt == "json":
            head = "{" + (f'"date": "{d}", ' if dated else "") + f'"description": {_json.dumps(desc, ensure_ascii=False)}, "amount": {amt}, "weekday": "{dow}", "category": "'
            return head, (cat(c) + '"}\n') if c is not None else None
        head = (f"- date: {d}\n  " if dated else "- ") + f"description: {_json.dumps(desc, ensure_ascii=False)}\n  amount: {amt}\n  weekday: {dow}\n  category: "
        return head, (cat(c) + "\n") if c is not None else None
    for f, c, st in rows:
        head, tail = rec(f, c); parts.append(head)
        if in_loss(st):
            n = sum(len(x) for x in parts); targets.append((n, n + len(tail) - 1))
        parts.append(tail)
    head, _ = rec(query)
    if fmt.startswith("ts"):
        parts.append("];\n\nconst next: Transaction = " + head.strip())  # ends 'category: "': the slot reads the label token after ' "'
        return "".join(parts), targets
    parts.append("</historical_transactions>\n\n<new_transaction>\n" + head + "(")
    return "".join(parts), targets
