"""One real YNAB budget, replayed (owner, 2026-10-02: "find all accepted transactions that have a category. That category is the gold
standard answer ... run each transaction in order through the decider model ... only include historical transactions that happened
before the one being inferred").

The budget comes from the YNAB skill's cache (taytays_stuff/.claude/skills/ynab: `cache refresh <budget>` writes
~/.cache/ynab-cli/<budget id>.json). Nothing personal is written inside the repo: items, per-item scores and the summary go to OUT
(default ~/.local/share/ynab-real-eval/<budget id>/).

  build   (CPU) one item per approved, categorised, non-split transaction, in date order. Gold = its category. The prompt is the blind
          sets' text format ("Categories: ...", then "Transaction: date | payee | $amount | Dow\\nCategory: name" rows, then the query)
          with a 24-row slice of approved, categorised, non-split transactions dated strictly before the query, chosen as
          build_blind_v1.build_slice does (the payee's own last 6, one latest row per category not yet shown, latest first; then the
          latest rows) without its similar-payee step (it needs the generator's merchant kinds). Payee = YNAB's payee name (transfers
          "Transfer : <account>"); amount outflow positive. Options = every category used in the 365 days before the query, plus the
          categories visible today (not hidden, not deleted), plus "Inflow: Ready to Assign"; a name used in two groups is written
          "<group>: <name>". Gold absent from the options (a category first used on this date and hidden today) -> item kept, flagged.
          Baselines per item: YNAB's rule (the category used in 2 of the payee's last 3 filed rows, else the last; none without history;
          build_blind_v1 BLIND_YNAB) and the payee's last category.
  score   (GPU) the items in date order through a decider reader, as exp_decision_models.py's decider() reads them:
            zeroshot  Mapika/decider-4b, its own layout (Context / Question / Options / "Answer: (", letters, wide labels past 10)
            recipe    decider-4B + the recipe LoRA (REPORT 140: rand255 labels, labelled_shots layout), merged; no other-users line
          Resumable (appends to OUT/scores_<reader>.jsonl).
  report  right first / top 3 per reader and baseline, overall, by year, by payee seen before or not, by kind (spending / inflow /
          transfer), first use of a category.

env: BUDGET (id), OUT, READER (zeroshot | recipe), BATCH (4), LIMIT (0 = all).
usage: BUDGET=<id> uv run python scripts/real_budget_eval.py build
       BUDGET=<id> READER=recipe uv run --with transformers==5.17.0 --with flash-linear-attention --with "peft>=0.21" --with torch==2.13.0
         --with torchvision==0.28.0 python scripts/real_budget_eval.py score
       BUDGET=<id> uv run python scripts/real_budget_eval.py report
"""
import datetime as dt
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

BUDGET = os.environ["BUDGET"]
CACHE = Path.home() / ".cache" / "ynab-cli" / f"{BUDGET}.json"
OUT = Path(os.environ.get("OUT", Path.home() / ".local" / "share" / "ynab-real-eval" / BUDGET))
READER = os.environ.get("READER", "recipe")
LAYOUT = os.environ.get("LAYOUT", "split")  # today (one prompt each) | split (one cached prefix per day)
NOCACHE = os.environ.get("NOCACHE") == "1"  # split prompts read whole: the check that the cached prefix changes nothing
BATCH = int(os.environ.get("BATCH", "4"))
LIMIT = int(os.environ.get("LIMIT", "0"))
SLICE_MAX, OWN_MAX = 24, 6
SHARED_MAX, SHARED_CAT_MAX = 24, 18  # build_blind_v1 BLIND_BULK
SHARED_HEAD, NEAR_HEAD = "Earlier transactions:", "Earlier transactions at this payee and similar payees:"
WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
RTA = "Inflow: Ready to Assign"
QUESTION = "Which of this user's categories does the last transaction belong to?"
RECIPE = "decider_decider-4b_none_h100bf16st800_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_oth50_aux100_labrand255_laylabelled_shots_ev10soft_lora"


def _private(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    return path


def build():
    b = json.loads(CACHE.read_text())["budget"]
    groups = {g["id"]: g for g in b["category_groups"]}
    cats = {c["id"]: c for c in b["categories"]}
    payees = {p["id"]: p["name"] for p in b["payees"]}
    accounts = {a["id"]: a["name"] for a in b["accounts"]}
    split_parents = {s["transaction_id"] for s in b.get("subtransactions", []) if not s.get("deleted")}

    def internal(c):
        return groups[c["category_group_id"]]["name"] == "Internal Master Category"

    names_by = defaultdict(list)
    for c in cats.values():
        if not c.get("deleted") and (not internal(c) or c["name"] == RTA):
            names_by[c["name"]].append(c["id"])

    def label(cid):
        c = cats[cid]
        if internal(c):
            return RTA if c["name"] == RTA else None
        return f'{groups[c["category_group_id"]]["name"]}: {c["name"]}' if len(names_by[c["name"]]) > 1 else c["name"]

    visible = [label(c["id"]) for c in b["categories"] if not c.get("deleted") and not c.get("hidden") and label(c["id"]) and label(c["id"]) != RTA]
    rows = []
    for t in b["transactions"]:
        if t.get("deleted") or not t.get("approved") or not t.get("category_id") or t["id"] in split_parents or t["category_id"] not in cats:
            continue
        lab = label(t["category_id"])
        if not lab:
            continue
        payee = payees.get(t.get("payee_id"), "") or (t.get("import_payee_name") or "")
        if t.get("transfer_account_id") and not payee:
            payee = "Transfer : " + accounts.get(t["transfer_account_id"], "")
        payee = payee.replace("|", "/").replace("\n", " ").strip()
        d = dt.date.fromisoformat(t["date"])
        amt = -t["amount"] / 1000
        rows.append(dict(id=t["id"], date=t["date"], payee_id=t.get("payee_id") or payee, payee=payee, cat=lab,
                         kind="transfer" if t.get("transfer_account_id") else "inflow" if lab == RTA else "spending",
                         fields=f"{t['date']} | {payee} | ${amt:.2f} | {WD[d.weekday()]}"))
    rows.sort(key=lambda r: (r["date"], r["id"]))
    items, by_payee, last_idx = [], defaultdict(list), {}
    used_order = []  # categories by first use
    upto = 0  # rows [0, upto) are history: dated strictly before the query
    day, shared = None, []
    for qi, q in enumerate(rows):
        while rows[upto]["date"] < q["date"]:
            r = rows[upto]
            by_payee[r["payee_id"]].append(upto); last_idx[r["cat"]] = upto
            if r["cat"] not in used_order:
                used_order.append(r["cat"])
            upto += 1
        if q["date"] != day:  # build_blind_v1.build_shared: the day's shared rows, the same for every transaction of the day
            day = q["date"]
            shared = sorted(last_idx.values(), reverse=True)[:SHARED_CAT_MAX]
            for i in range(upto - 1, -1, -1):
                if len(shared) >= SHARED_MAX:
                    break
                if i not in shared:
                    shared.append(i)
            shared.sort()
        chosen, seen = [], set()

        def take(i):
            if i not in seen and len(chosen) < SLICE_MAX:
                seen.add(i); chosen.append(i)
        prev_rows = by_payee[q["payee_id"]]
        for i in prev_rows[::-1][:OWN_MAX]:
            take(i)
        shown = {rows[i]["cat"] for i in chosen}
        for c, i in sorted(last_idx.items(), key=lambda x: -x[1]):  # one latest row per category, most recently used first
            if len(chosen) >= SLICE_MAX:
                break
            if c not in shown:
                take(i); shown.add(c)
        for i in range(upto - 1, -1, -1):
            if len(chosen) >= SLICE_MAX:
                break
            take(i)
        chosen.sort()
        year_ago = str(dt.date.fromisoformat(q["date"]) - dt.timedelta(days=365))
        options = list(dict.fromkeys([RTA] + visible + [c for c in used_order if rows[last_idx[c]]["date"] >= year_ago]))[:255]
        ctx = "Categories: " + ", ".join(options) + "\n\n" + "".join(f"Transaction: {rows[i]['fields']}\nCategory: {rows[i]['cat']}\n\n" for i in chosen)
        near = sorted([i for i in prev_rows if i not in set(shared)][::-1][:OWN_MAX])
        row_text = lambda i: f"Transaction: {rows[i]['fields']}\nCategory: {rows[i]['cat']}\n\n"  # noqa: E731
        split = ("Categories: " + ", ".join(options) + f"\n\n{SHARED_HEAD}\n\n" + "".join(map(row_text, shared)) + f"{NEAR_HEAD}\n\n"
                 + "".join(map(row_text, near)) + f"Transaction: {q['fields']}\nCategory:")
        prev = [rows[i]["cat"] for i in prev_rows]
        top = Counter(prev[-3:]).most_common(1)
        rule = (top[0][0] if top and top[0][1] >= 2 else prev[-1]) if prev else None
        items.append(dict(id=q["id"], date=q["date"], kind=q["kind"], payee_seen=bool(prev), first_use=q["cat"] not in last_idx,
                          options=options, answer=options.index(q["cat"]) if q["cat"] in options else -1, gold=q["cat"],
                          prompt=ctx + f"Transaction: {q['fields']}\nCategory:", prompt_split=split, rule=rule, last=prev[-1] if prev else None, n_hist=upto))
    p = _private(OUT / "items.json")
    p.write_text(json.dumps(dict(budget=BUDGET, n=len(items), items=items)))
    os.chmod(p, 0o600)
    print(f"{len(items)} items ({Counter(i['kind'] for i in items)}); gold missing from options: {sum(i['answer'] < 0 for i in items)}; "
          f"options median {sorted(len(i['options']) for i in items)[len(items) // 2]}, max {max(len(i['options']) for i in items)}")


def run_scoring(todo, fo):
    """Score `todo` (items in this file's shape) with READER / LAYOUT, one JSON line per item to `fo`; progress (counts only) to stderr."""
    import importlib
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ai_experiments import oneslot
    from ai_experiments.paths import ROOT
    path = snapshot_download("Mapika/decider-4b")
    sys.path.insert(0, snapshot_download("Mapika/decider-2b", allow_patterns=["decider/*"]))
    P = importlib.import_module("decider.prompt")
    tok = AutoTokenizer.from_pretrained(path)
    lm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    if READER == "recipe":
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, str(ROOT / "models" / "adapters" / RECIPE)).merge_and_unload().eval()

    def built(it):
        state = it["prompt" if LAYOUT == "today" else "prompt_split"][: -len("Category:")].rstrip()
        rng = random.Random(it["id"] if LAYOUT == "today" else f"day-{it['date']}")  # split: labels and option order fixed per day
        if READER == "recipe":
            return oneslot.build_layout(P, tok, state, QUESTION, it["options"], it["answer"], rng, labels="rand255", layout="labelled_shots")
        return oneslot.build(P, tok, state, QUESTION, it["options"], it["answer"], rng, labels="letters")

    import time
    pad = tok.pad_token_id or 0

    def emit(fo, it, b, hrow):
        z = F.linear(hrow, lm.lm_head.weight[torch.tensor(b["labs"], device="cuda")]).float()
        lp = F.log_softmax(z, -1).tolist()
        back = [0.0] * len(lp)
        for j, oi in enumerate(b["perm"]):
            back[oi] = lp[j]
        fo.write(json.dumps(dict(id=it["id"], lp=[round(x, 4) for x in back], n_tok=len(b["ids"]))) + "\n")

    def lcp(bs):
        n = min(len(b["ids"]) for b in bs)
        return next((j for j in range(n) if any(b["ids"][j] != bs[0]["ids"][j] for b in bs)), n)

    if LAYOUT == "today":
        todo.sort(key=lambda it: len(it["prompt"]) + 40 * len(it["options"]))  # similar lengths per batch; results keyed by id
        groups = [todo[k:k + BATCH] for k in range(0, len(todo), BATCH)]
    else:  # one group per day: the shared prefix run once (batch 1), every transaction's tail from its cache (bench_bulk.run_cached)
        by_day = defaultdict(list)
        for it in todo:
            by_day[it["date"]].append(it)
        groups = [g[k:k + 32] for _, g in sorted(by_day.items()) for k in range(0, len(g), 32)]
    t0, n, saved = time.time(), 0, 0
    with torch.no_grad():
        for gi, chunk in enumerate(groups):
            bs = [built(it) for it in chunk]
            L = lcp(bs) if LAYOUT == "split" and len(bs) > 1 and not NOCACHE else 0
            L = min(L, min(b["slot"] for b in bs))  # every tail keeps its answer slot
            if L:
                pre = torch.tensor([bs[0]["ids"][:L]], device="cuda")
                cache = lm.model(input_ids=pre, use_cache=True).past_key_values
                cache.reorder_cache(torch.zeros(len(bs), dtype=torch.long, device="cuda"))  # bench_bulk.expand_cache
                saved += L * (len(bs) - 1)
            tails = [b["ids"][L:] for b in bs]
            T = -(-max(map(len, tails)) // 64) * 64
            ids = torch.full((len(bs), T), pad, dtype=torch.long); att = torch.zeros((len(bs), L + T), dtype=torch.long); att[:, :L] = 1
            for i, t in enumerate(tails):
                ids[i, :len(t)] = torch.tensor(t); att[i, L:L + len(t)] = 1
            h = (lm.model(input_ids=ids.cuda(), attention_mask=att.cuda(), past_key_values=cache, use_cache=True) if L else
                 lm.model(input_ids=ids.cuda(), attention_mask=att.cuda())).last_hidden_state
            for i, (it, b) in enumerate(zip(chunk, bs)):
                emit(fo, it, b, h[i, b["slot"] - L])
            fo.flush()
            n += len(chunk)
            if gi % 200 == 0:
                r = n / (time.time() - t0)
                print(f"   {n}/{len(todo)}  {r:.1f} items/s  ~{(len(todo) - n) / r / 60:.0f} min left  (shared tokens not re-read: {saved})",
                      file=sys.stderr, flush=True)




def score():
    items = json.loads((OUT / "items.json").read_text())["items"]
    items = [it for it in items if it["answer"] >= 0]
    items = items[:LIMIT] if LIMIT else items
    out = _private(OUT / f"{'check' if NOCACHE else 'scores'}_{READER}_{LAYOUT}.jsonl")
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    todo = [it for it in items if it["id"] not in done]
    print(f"{READER}: {len(done)} done, {len(todo)} to score", flush=True)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as fo:
        run_scoring(todo, fo)


def modal():
    """score, on up to 8 Modal H100s with nothing kept there (modal_app.score_private); results land in OUT as with `score`."""
    sys.path.insert(0, str(Path(__file__).parent))
    import modal_app
    items = json.loads((OUT / "items.json").read_text())["items"]
    items = [it for it in items if it["answer"] >= 0]
    items = items[:LIMIT] if LIMIT else items
    out = _private(OUT / f"scores_{READER}_{LAYOUT}.jsonl")
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    keep = ("id", "date", "options", "answer", "prompt" if LAYOUT == "today" else "prompt_split")
    todo = [{k: it[k] for k in keep} for it in items if it["id"] not in done]
    print(f"{READER}: {len(done)} done, {len(todo)} to score on Modal", flush=True)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    n = 0
    with os.fdopen(fd, "a") as fo:
        for line in modal_app.private_scores(todo, READER, LAYOUT, "r124-oth-s0" if READER == "recipe" else ""):
            fo.write(line + "\n"); n += 1
    print(f"{READER}: {n} scores back", flush=True)


def stream():
    """Items as one JSON list on stdin, score lines on stdout, nothing written to disk (modal_app.score_private)."""
    run_scoring(json.load(sys.stdin), sys.stdout)


def report():
    items = {it["id"]: it for it in json.loads((OUT / "items.json").read_text())["items"]}
    readers = {}
    for f in sorted(OUT.glob("scores_*.jsonl")):
        readers[f.stem[len("scores_"):]] = {r["id"]: r["lp"] for r in map(json.loads, open(f))}
    ids = [i for i in items if all(i in r for r in readers.values())] if readers else list(items)

    def rank(lp, a):
        return 1 + sum(x > lp[a] for x in lp)
    groups = {"all": ids}
    for i in ids:
        it = items[i]
        groups.setdefault(f"year {it['date'][:4]}", []).append(i)
        groups.setdefault("payee seen before" if it["payee_seen"] else "first-time payee", []).append(i)
        groups.setdefault(f"kind: {it['kind']}", []).append(i)
        if it["first_use"]:
            groups.setdefault("first use of a category", []).append(i)
        if it["answer"] < 0:
            groups.setdefault("gold not offered", []).append(i)
    order = ["all", "payee seen before", "first-time payee", "kind: spending", "kind: inflow", "kind: transfer", "first use of a category",
             "gold not offered"] + sorted(g for g in groups if g.startswith("year"))
    head = "| group | n | YNAB rule | payee's last category | " + " | ".join(f"{r} first (top 3)" for r in readers) + " |"
    lines = [head, "|---|---|---|---|" + "---|" * len(readers)]
    for g in [g for g in order if g in groups]:
        v = groups[g]
        cells = [f"{100 * sum(items[i]['rule'] == items[i]['gold'] for i in v) / len(v):.1f}", f"{100 * sum(items[i]['last'] == items[i]['gold'] for i in v) / len(v):.1f}"]
        for r in readers.values():
            rk = [rank(r[i], items[i]["answer"]) if items[i]["answer"] >= 0 else 999 for i in v]
            cells.append(f"{100 * sum(x == 1 for x in rk) / len(v):.1f} ({100 * sum(x <= 3 for x in rk) / len(v):.1f})")
        lines.append(f"| {g} | {len(v)} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print(text)
    p = _private(OUT / "report.md"); p.write_text(text + "\n"); os.chmod(p, 0o600)


if __name__ == "__main__":
    {"build": build, "score": score, "modal": modal, "stream": stream, "report": report}[sys.argv[1]]()
