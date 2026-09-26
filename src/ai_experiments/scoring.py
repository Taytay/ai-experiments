"""Option scoring that keeps every number (STAT-2).

Every accuracy in the repo so far came from one scorer: for each option, the mean per-token
log-prob of the option given the prompt, argmax over options; `accuracy()` then kept only the
per-level hit rate. `Scorer.score()` runs the same forward passes and returns one record per item
with everything a different scorer, a bootstrap or a paired test needs later (PLAN.md steps 2
and 4), so no adapter has to be loaded again to try a new scoring rule.

Record fields (lists are per option, in the item's option order):

  id, level, answer   from the frozen item (ai_experiments.items)
  n_prompt_tok        prompt tokens scored (prompts longer than maxlen-32 are cut from the left)
  sum_lp              sum of log-probs of the option tokens given the prompt
  n_tok, n_bytes      option length in tokens and in UTF-8 bytes
  dc_lp               sum of log-probs given only the prompt's cue line ("Answer:" or "Label:"),
                      the domain premise of PMI_DC (Holtzman et al. 2021, "surface form competition")
  unc_lp              sum of log-probs given a bare newline: the unconditional baseline (UNC)
  mcf_lp              log-prob of the option's letter when the options are listed as "A. ..." lines
                      between the prompt body and the cue (multiple-choice format, MCF; symbol scoring)
  hyb_lp              sum of log-probs of the option text after that same listed-choices prompt
                      (hybrid scoring: the listing fixes the option at its first tokens, so length
                      stops mattering; 2607.12767, 2402.01781)
  pred                argmax of sum_lp / n_tok, the scorer used so far
  correct             pred == answer

`aggregate()` turns records into the per-level accuracies and L6 margins the reports print, so the
headline numbers come from the same records that are written to disk (`write_records`, one JSONL
per run and condition under results/per_item/).
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .paths import RESULTS

CUES = ("Answer:", "Label:")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PER_ITEM = RESULTS / "per_item"


def per_item_path(result_json: Path, condition: str) -> Path:
    """results/curriculum_Qwen2.5-3B_C.json, "trained" -> results/per_item/curriculum_Qwen2.5-3B_C.trained.jsonl"""
    return PER_ITEM / f"{Path(result_json).stem}.{condition}.jsonl"


class Scorer:
    def __init__(self, model, tok, maxlen: int = 768, extras: bool = True):
        self.model, self.tok, self.maxlen, self.extras = model, tok, maxlen, extras
        self.pad = tok.pad_token_id or 0
        self._cache: dict[tuple[str, str], tuple[float, int]] = {}  # (premise, option) -> (sum_lp, n_tok)

    def _ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _prompt_ids(self, prompt: str) -> list[int]:
        return self._ids(prompt)[-(self.maxlen - 32):]

    @torch.no_grad()
    def _forward(self, p_ids: list[int], opt_ids: list[list[int]]) -> list[tuple[float, int]]:
        """(sum log-prob, token count) of every option continuation after the prompt ids."""
        seqs = [p_ids + o for o in opt_ids]
        L = max(map(len, seqs))
        ids = torch.tensor([s + [self.pad] * (L - len(s)) for s in seqs], device="cuda")
        att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device="cuda")
        try:
            logits = self.model(input_ids=ids, attention_mask=att).logits
        except (torch.OutOfMemoryError, torch.AcceleratorError):
            if len(opt_ids) == 1:
                raise
            torch.cuda.empty_cache()  # long prompt x many options: one option per forward instead
            return [self._forward(p_ids, [o])[0] for o in opt_ids]
        out, a = [], len(p_ids)
        for i, o in enumerate(opt_ids):
            b = a + len(o)  # softmax only at the answer positions, not over the whole sequence
            lp = F.log_softmax(logits[i, a - 1:b - 1].float(), -1).gather(-1, ids[i, a:b].unsqueeze(-1))
            out.append((lp.sum().item(), len(o)))
        return out

    def _premised(self, premise: str, options: list[str], opt_ids: list[list[int]]) -> list[float]:
        """sum log-prob of each option after a short fixed premise; cached, since options repeat across items."""
        need = [(o, oi) for o, oi in zip(options, opt_ids) if (premise, o) not in self._cache]
        if need:
            p_ids = self._ids(premise)
            for (o, _), r in zip(need, self._forward(p_ids, [oi for _, oi in need])):
                self._cache[(premise, o)] = r
        return [self._cache[(premise, o)][0] for o in options]

    @staticmethod
    def cue_of(prompt: str) -> str:
        tail = prompt.rstrip()
        return next((c for c in CUES if tail.endswith(c)), CUES[0])

    @staticmethod
    def mcf_prompt(prompt: str, options: list[str], cue: str) -> str:
        """Prompt body, then the options as lettered lines, then the cue again: the letter is scored."""
        body = prompt.rstrip()
        body = (body[:-len(cue)].rstrip("\n") if body.endswith(cue) else body) + "\n"
        listing = "\n".join(f"{LETTERS[i]}. {o.strip()}" for i, o in enumerate(options))
        return f"{body}Choices:\n{listing}\n{cue}"

    def score(self, items: list[dict], ctx: bool = False, label: str = "") -> list[dict]:
        """One record per item. ctx=True scores item["prompt_ctx"] (field guide prepended)."""
        recs, t0 = [], time.time()
        for it in items:
            prompt, options = it["prompt_ctx" if ctx else "prompt"], it["options"]
            p_ids = self._prompt_ids(prompt)
            opt_ids = [self._ids(o) for o in options]
            cond = self._forward(p_ids, opt_ids)
            rec = dict(id=it["id"], level=it["level"], answer=it["answer"], n_prompt_tok=len(p_ids),
                       sum_lp=[s for s, _ in cond], n_tok=[n for _, n in cond],
                       n_bytes=[len(o.encode("utf-8")) for o in options])
            if self.extras:
                cue = self.cue_of(prompt)
                rec["dc_lp"] = self._premised(cue, options, opt_ids)
                rec["unc_lp"] = self._premised("\n", options, opt_ids)
                listed = self._prompt_ids(self.mcf_prompt(prompt, options, cue))
                letter_ids = [self._ids(" " + LETTERS[i]) for i in range(len(options))]
                rec["mcf_lp"] = [s for s, _ in self._forward(listed, letter_ids)]
                rec["hyb_lp"] = [s for s, _ in self._forward(listed, opt_ids)]
            mean = [s / max(n, 1) for s, n in cond]
            rec["pred"] = max(range(len(mean)), key=mean.__getitem__)
            rec["correct"] = rec["pred"] == it["answer"]
            recs.append(rec)
        if label:
            print(f"    scored {len(items)} {label} items in {(time.time() - t0) / 60:.1f} min", flush=True)
        return recs


@torch.no_grad()
def perplexity(model, tok, text: str) -> float:
    # plain CE from logits: unsloth's fused loss refuses to run when the caching allocator
    # holds most of the card after a long eval pass ("No or negligible GPU memory available")
    torch.cuda.empty_cache()
    ids = tok(text, return_tensors="pt")["input_ids"].cuda()
    logits = model(input_ids=ids).logits.float()
    return math.exp(F.cross_entropy(logits[0, :-1], ids[0, 1:]).item())


def option_logprobs_batched(model, tok, prompts: list[list[int]], options: list[list[list[int]]]) -> list[torch.Tensor]:
    """Sum of option-token log-probs for every (prompt, option) pair of a batch, differentiable, in one
    right-padded forward. prompts: token ids per example; options: per example, token ids per option.
    Returns one 1-D tensor per example (one value per option), i.e. log P(option | prompt).

    Only the option positions go through the LM head: unsloth returns the final hidden states in place
    of logits when UNSLOTH_RETURN_HIDDEN_STATES=1, so a batch of 64 rows never materialises 64 x L x V
    logits. Right padding, because left padding changed these sums by up to 0.6 nats under unsloth
    (PLAN step 27); same arithmetic as Scorer._forward, checked against it in
    scripts/diag_distill_options.py."""
    import os
    pad = tok.pad_token_id or 0
    seqs, owner = [], []
    for i, (p, opts) in enumerate(zip(prompts, options)):
        for j, o in enumerate(opts):
            seqs.append(p + o); owner.append((i, j, len(p), len(o)))
    L = max(map(len, seqs))
    ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device="cuda")
    att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device="cuda")
    prev = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    try:
        hidden = model(input_ids=ids, attention_mask=att).logits  # [rows, L, H]
    finally:
        if prev is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = prev
    head = model.get_output_embeddings()
    ri = torch.cat([torch.full((lo,), r, device="cuda", dtype=torch.long) for r, (_, _, lp, lo) in enumerate(owner)])
    pi = torch.cat([torch.arange(lp - 1, lp + lo - 1, device="cuda") for (_, _, lp, lo) in owner])
    logits = head(hidden[ri, pi].to(head.weight.dtype)).float()
    tok_lp = F.log_softmax(logits, -1).gather(-1, ids[ri, pi + 1].unsqueeze(-1)).squeeze(-1)
    out = [[None] * len(opts) for opts in options]
    k = 0
    for (i, j, _, lo) in owner:
        out[i][j] = tok_lp[k:k + lo].sum(); k += lo
    return [torch.stack(row) for row in out]


@torch.no_grad()
def corpus_perplexity(model, tok, chunks: list[str], maxlen: int = 768) -> dict:
    """Token-weighted perplexity over text chunks (each cut to maxlen tokens), with the standard error
    of the per-chunk mean NLL across chunks, so two adapters' numbers can be told apart (PLAN step 24)."""
    torch.cuda.empty_cache()
    nll, n = [], []
    for c in chunks:
        ids = tok(c, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :maxlen].cuda()
        logits = model(input_ids=ids).logits.float()
        nll.append(F.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="sum").item()); n.append(ids.shape[1] - 1)
    mean = sum(nll) / sum(n)
    per = [a / b for a, b in zip(nll, n)]
    se = (sum((x - sum(per) / len(per)) ** 2 for x in per) / (len(per) - 1)) ** 0.5 / len(per) ** 0.5 if len(per) > 1 else 0.0
    return dict(ppl=round(math.exp(mean), 3), nll=round(mean, 4), nll_se=round(se, 4), n_tokens=sum(n), n_chunks=len(chunks))


def _softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    z = sum(es)
    return [e / z for e in es]


def aggregate(records: list[dict]) -> dict[str, float]:
    """Per-level accuracy (%) under the mean-per-token scorer, plus the L6 confidence margins."""
    hit, n, margin = defaultdict(int), defaultdict(int), defaultdict(list)
    for r in records:
        hit[r["level"]] += r["correct"]; n[r["level"]] += 1
        p = sorted(_softmax([s / max(k, 1) for s, k in zip(r["sum_lp"], r["n_tok"])]), reverse=True)
        margin[r["level"]].append(p[0] - p[1])
    res = {lv: round(100 * hit[lv] / n[lv], 1) for lv in n}
    for lv in ("L6_unseen_recall", "L6_seen_recall_ctrl"):
        if lv in margin:
            res[lv + "_margin"] = round(sum(margin[lv]) / len(margin[lv]), 3)
    return res


def _rounded(rec: dict, nd: int = 4) -> dict:
    return {k: [round(x, nd) for x in v] if isinstance(v, list) and v and isinstance(v[0], float) else v
            for k, v in rec.items()}


def write_records(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(_rounded(r), ensure_ascii=False) + "\n")
    return path


def read_records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
