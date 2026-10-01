"""strands-decider (github.com/strands-labs/strands-decider, Apache-2.0; PLAN step 151, MODEL-22) on the categoriser's items and
episodes. Its model: a Qwen3.5-2B-Base torso (LM head discarded), a rank-16 LoRA and a pointer head that scores each option from the
hidden state at the last token of the option's line against the state at "<answer>". The prompt is strands' own
(`strands_decider.prompting.build_prompt`): "<state>\\n...\\n</state>\\n<question type="choice">\\nSelect exactly one option.\\n<question>
\\n<options>\\n1. name\\n...</options>\\n</question>\\n<answer>". Our item's prompt without its "Category:" cue is the state, the category
names are the options with no description. Tokenised whole, as strands' training collator does (its server tokenises state and question
apart; the two differ at most at the one boundary token), and never truncated: a cut would drop the query or the options.
Needs the package: uv run --with strands-decider==0.1.0 (see STRANDS_WITH).
"""
V19 = "StrandsAgents/strands-decider-2B-hobson-v19"  # the published reference checkpoint (JevBench 0.723), Apache-2.0
BASE = "Qwen/Qwen3.5-2B-Base"
STRANDS_WITH = "--with strands-decider==0.1.0"


def encode(tok, state, question, options, order=None):
    """(token ids, option token positions in the shown order, shown order) for one question; `order` permutes the options shown."""
    from strands_decider.infer import _option_token_index
    from strands_decider.prompting import build_prompt
    from strands_decider.schema import ChoiceQuestion
    crit = {o: "" for o in options}
    assert len(crit) == len(options), f"duplicate option names: {options}"
    prompt, rq = build_prompt(state, ChoiceQuestion(instructions=question, criteria=crit), option_order=order)
    enc = tok(prompt, return_offsets_mapping=True)
    return enc["input_ids"], _option_token_index(enc["offset_mapping"], rq.option_spans, len(prompt) - len(rq.text))


CODES = [a + b for a in "ABCDEFGHJKLMNPQRSTUVWXYZ" for b in "ABCDEFGHJKLMNPQRSTUVWXYZ"]  # row 152: two-letter labels (no I, O)


def labelled(context, options, rng):
    """Row 152 (MODEL-23): our prompt in the labelled layout, inside strands' state: each category gets a random two-letter code, the
    category list and every history row show it ("Category: (KQ) Groceries"), and the options read "(KQ) Groceries". Returns (state,
    shown options in the item's option order, character spans of the codes in the history rows, for a label-copying loss)."""
    from ai_experiments import oneslot
    names, rows, query = oneslot.parse(context)
    code = dict(zip(options, rng.sample(CODES, len(options))))
    order = list(options); rng.shuffle(order)
    parts = ["Categories:\n" + "".join(f"- ({code[o]}) {o}\n" for o in order) + "\n"]
    spans = []
    for fields, label, _ in rows:
        if label is None:  # the other-users line, or a section header
            parts.append(fields + "\n\n"); continue
        parts.append(f"Transaction: {fields}\nCategory: ")
        if label in code:
            n = sum(len(p) for p in parts) + 1; spans.append((n, n + 2))
            parts.append(f"({code[label]}) {label}\n\n")
        else:
            parts.append(f"{label}\n\n")
    parts.append(f"Transaction: {query}")
    return "".join(parts), [f"({code[o]}) {o}" for o in options], spans


def encode_aux(tok, state, question, options, spans, order=None):
    """encode() plus [(position, token id)] for the code tokens in the history rows: each predicted from the position before it."""
    from strands_decider.infer import _option_token_index
    from strands_decider.prompting import build_prompt, render_state
    from strands_decider.schema import ChoiceQuestion
    prompt, rq = build_prompt(state, ChoiceQuestion(instructions=question, criteria={o: "" for o in options}), option_order=order)
    base = len(render_state(state)) - len(state.strip()) - len("\n</state>\n")  # where the state text starts in the prompt
    assert prompt[base:base + 20] == state.strip()[:20]
    enc = tok(prompt, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    sp = [(base + a, base + b) for a, b in spans]
    aux = [(t - 1, ids[t]) for t, (a, b) in enumerate(offs) if t > 0 and b > a and any(a < e and s0 < b for s0, e in sp)]
    return ids, _option_token_index(offs, rq.option_spans, len(prompt) - len(rq.text)), aux


def forward(model, encs, device="cuda", round_to=64):
    """(per-option log-probabilities [B, K] at temperature 1, padded options at -inf; the torso's last hidden states [B, T, d]) for a batch
    of encode() outputs. Lengths are padded to a multiple of `round_to` (flash-linear-attention tunes a kernel per length)."""
    import torch
    from strands_decider.modeling import gather_options, masked_log_softmax, pool_last_token
    pad = model.tokenizer.pad_token_id or 0
    T = -(-max(len(e[0]) for e in encs) // round_to) * round_to
    K = max(len(e[1]) for e in encs)
    ids = torch.full((len(encs), T), pad, dtype=torch.long)
    att = torch.zeros_like(ids)
    idx = torch.full((len(encs), K), -1, dtype=torch.long)
    for i, (x, o, *_) in enumerate(encs):
        ids[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1; idx[i, :len(o)] = torch.tensor(o)
    att = att.to(device)
    hidden = model.encode(ids.to(device), att)
    pooled = pool_last_token(hidden, att).float()
    logits = model.head(pooled, gather_options(hidden, idx.to(device)).float())
    return masked_log_softmax(logits, torch.tensor([len(e[1]) for e in encs], device=device)), hidden


def log_probs(model, encs, device="cuda", round_to=64):
    """forward()'s log-probabilities only (the same computation as the model's own forward with temperature 1)."""
    return forward(model, encs, device, round_to)[0]


def load(path_or_id, device="cuda"):
    """A strands checkpoint (a local directory, e.g. models/adapters/strands_*, or a Hub id) in bf16 with its fp32 head."""
    from strands_decider.modeling import StrandsDeciderModel
    m = StrandsDeciderModel.load(str(path_or_id))
    if m.tokenizer.pad_token is None:
        m.tokenizer.pad_token = m.tokenizer.eos_token
    return m.to(device)
