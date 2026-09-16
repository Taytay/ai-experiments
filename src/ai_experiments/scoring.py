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
    """Scores every option of every item. Rows from many items go through one forward (sorted by length,
    right-padded, up to `rows_per_forward` rows or `tokens_per_forward` tokens), and only the option
    positions go through the LM head: under unsloth the model returns its final hidden states instead of
    logits when UNSLOTH_RETURN_HIDDEN_STATES=1, so a 64-row batch never materialises 64 x L x V logits.
    Before 2026-09-15 each item was its own forward (GPU at 26%, 26 minutes per evaluation); the batched
    scorer changes log-probs by the bf16 batch-shape noise only (REPORT.md 19 measures it on saved weights)."""

    def __init__(self, model, tok, maxlen: int = 768, extras: bool = True, rows_per_forward: int = 64, tokens_per_forward: int = 32768):
        self.model, self.tok, self.maxlen, self.extras = model, tok, maxlen, extras
        self.rows, self.tokens = rows_per_forward, tokens_per_forward
        self.pad = tok.pad_token_id or 0
        self._cache: dict[tuple[str, str], tuple[float, int]] = {}  # (premise, option) -> (sum_lp, n_tok)
        self._head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None

    def _ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _prompt_ids(self, prompt: str) -> list[int]:
        return self._ids(prompt)[-(self.maxlen - 32):]

    @torch.no_grad()
    def _hidden_or_logits(self, ids, att):
        """Final hidden states [rows, L, H] if the model can return them (unsloth), else full logits."""
        import os
        if self._head is None:
            return self.model(input_ids=ids, attention_mask=att).logits, False
        prev = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
        try:
            out = self.model(input_ids=ids, attention_mask=att).logits
        finally:
            if prev is None:
                os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
            else:
                os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = prev
        return out, out.shape[-1] != self._head.weight.shape[0]  # hidden width vs vocabulary size

    @torch.no_grad()
    def _run_rows(self, rows: list[tuple[list[int], int, int]]) -> list[float]:
        """rows: (sequence ids, a, b) with the option tokens at [a, b). -> sum of their log-probs, per row."""
        L = max(len(r[0]) for r in rows)
        ids = torch.tensor([r[0] + [self.pad] * (L - len(r[0])) for r in rows], device="cuda")
        att = torch.tensor([[1] * len(r[0]) + [0] * (L - len(r[0])) for r in rows], device="cuda")
        try:
            out, is_hidden = self._hidden_or_logits(ids, att)
        except (torch.OutOfMemoryError, torch.AcceleratorError):
            if len(rows) == 1:
                raise
            torch.cuda.empty_cache()
            h = len(rows) // 2
            return self._run_rows(rows[:h]) + self._run_rows(rows[h:])
        ri = torch.cat([torch.full((b - a,), i, device="cuda", dtype=torch.long) for i, (_, a, b) in enumerate(rows)])
        pi = torch.cat([torch.arange(a - 1, b - 1, device="cuda") for (_, a, b) in rows])
        sel = out[ri, pi]
        logits = self._head(sel.to(self._head.weight.dtype)).float() if is_hidden else sel.float()
        tok_lp = F.log_softmax(logits, -1).gather(-1, ids[ri, pi + 1].unsqueeze(-1)).squeeze(-1)
        sums, k = [], 0
        for (_, a, b) in rows:
            sums.append(tok_lp[k:k + (b - a)].sum().item()); k += b - a
        return sums

    def _forward_many(self, reqs: list[tuple[list[int], list[list[int]]]]) -> list[list[tuple[float, int]]]:
        """reqs: (prompt ids, option ids per option). Every (prompt, option) row of every request is scored,
        rows sorted by length and packed into forwards under the row and token budgets; returns, per request,
        (sum log-prob, token count) per option."""
        rows, owner = [], []
        for r, (p, opts) in enumerate(reqs):
            for j, o in enumerate(opts):
                rows.append((p + o, len(p), len(p) + len(o))); owner.append((r, j))
        order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]))
        out = [[None] * len(opts) for _, opts in reqs]
        i = 0
        while i < len(order):
            chunk, L = [order[i]], len(rows[order[i]][0]); i += 1
            while i < len(order) and len(chunk) < self.rows and max(L, len(rows[order[i]][0])) * (len(chunk) + 1) <= self.tokens:
                chunk.append(order[i]); L = max(L, len(rows[order[i]][0])); i += 1
            for idx, s in zip(chunk, self._run_rows([rows[c] for c in chunk])):
                r, j = owner[idx]
                out[r][j] = (s, rows[idx][2] - rows[idx][1])
        return out

    def _forward(self, p_ids: list[int], opt_ids: list[list[int]]) -> list[tuple[float, int]]:
        """One item (kept for callers that score items one at a time)."""
        return self._forward_many([(p_ids, opt_ids)])[0]

    def _premised_many(self, premises: list[str], options: list[list[str]], opt_ids: list[list[list[int]]]) -> list[list[float]]:
        """sum log-prob of each option after a short fixed premise, for many items; cached per (premise, option)."""
        need: dict[str, dict[str, list[int]]] = {}
        for prem, opts, oids in zip(premises, options, opt_ids):
            for o, oi in zip(opts, oids):
                if (prem, o) not in self._cache:
                    need.setdefault(prem, {})[o] = oi
        if need:
            prems = list(need)
            res = self._forward_many([(self._ids(pr), list(need[pr].values())) for pr in prems])
            for pr, r in zip(prems, res):
                for o, val in zip(need[pr], r):
                    self._cache[(pr, o)] = val
        return [[self._cache[(prem, o)][0] for o in opts] for prem, opts in zip(premises, options)]

    def _premised(self, premise: str, options: list[str], opt_ids: list[list[int]]) -> list[float]:
        return self._premised_many([premise], [options], [opt_ids])[0]

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
        t0 = time.time()
        prompts = [it["prompt_ctx" if ctx else "prompt"] for it in items]
        p_ids = [self._prompt_ids(p) for p in prompts]
        opt_ids = [[self._ids(o) for o in it["options"]] for it in items]
        main = self._forward_many(list(zip(p_ids, opt_ids)))
        if self.extras:
            cues = [self.cue_of(p) for p in prompts]
            options = [it["options"] for it in items]
            dc = self._premised_many(cues, options, opt_ids)
            unc = self._premised_many(["\n"] * len(items), options, opt_ids)
            listed = [self._prompt_ids(self.mcf_prompt(p, o, c)) for p, o, c in zip(prompts, options, cues)]
            letters = [[self._ids(" " + LETTERS[i]) for i in range(len(o))] for o in options]
            mcf = self._forward_many(list(zip(listed, letters)))
            hyb = self._forward_many(list(zip(listed, opt_ids)))
        recs = []
        for k, it in enumerate(items):
            cond = main[k]
            rec = dict(id=it["id"], level=it["level"], answer=it["answer"], n_prompt_tok=len(p_ids[k]),
                       sum_lp=[s for s, _ in cond], n_tok=[n for _, n in cond],
                       n_bytes=[len(o.encode("utf-8")) for o in it["options"]])
            if self.extras:
                rec["dc_lp"], rec["unc_lp"] = dc[k], unc[k]
                rec["mcf_lp"] = [s for s, _ in mcf[k]]
                rec["hyb_lp"] = [s for s, _ in hyb[k]]
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
