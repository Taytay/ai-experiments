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


def log_probs(model, encs, device="cuda", round_to=64):
    """Per-option log-probabilities [B, K] (padded options at -inf) for a batch of encode() outputs, at temperature 1 (the raw head; a
    checkpoint's fitted temperature is not applied). Lengths are padded to a multiple of `round_to` (flash-linear-attention tunes a kernel
    per length)."""
    import torch
    pad = model.tokenizer.pad_token_id or 0
    T = -(-max(len(e[0]) for e in encs) // round_to) * round_to
    K = max(len(e[1]) for e in encs)
    ids = torch.full((len(encs), T), pad, dtype=torch.long)
    att = torch.zeros_like(ids)
    idx = torch.full((len(encs), K), -1, dtype=torch.long)
    for i, (x, o, *_) in enumerate(encs):
        ids[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1; idx[i, :len(o)] = torch.tensor(o)
    n = torch.tensor([len(e[1]) for e in encs], device=device)
    return model(input_ids=ids.to(device), attention_mask=att.to(device), n_slots=n, opt_idx=idx.to(device), temperature=1.0)["log_probs"]


def load(path_or_id, device="cuda"):
    """A strands checkpoint (a local directory, e.g. models/adapters/strands_*, or a Hub id) in bf16 with its fp32 head."""
    from strands_decider.modeling import StrandsDeciderModel
    m = StrandsDeciderModel.load(str(path_or_id))
    if m.tokenizer.pad_token is None:
        m.tokenizer.pad_token = m.tokenizer.eos_token
    return m.to(device)
