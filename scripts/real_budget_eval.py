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
CACHE = Path(os.environ.get("CACHE_PATH") or Path.home() / ".cache" / "ynab-cli" / f"{BUDGET}.json")  # CACHE_PATH: another budget file (row 173: synthetic households)
OUT = Path(os.environ.get("OUT", Path.home() / ".local" / "share" / "ynab-real-eval" / BUDGET))
READER = os.environ.get("READER", "recipe")  # recipe | zeroshot (decider-4B) | big (decider-35B-A3B, untrained) | big-recipe (35B + recipe)
LAYOUT = os.environ.get("LAYOUT", "split")  # today (one prompt each) | split (one cached prefix per day)
ONLY_NEW = os.environ.get("ONLY_NEW") == "1"  # score first-time payees only (the 35B's part of the recommended system)
NOCACHE = os.environ.get("NOCACHE") == "1"  # split prompts read whole: the check that the cached prefix changes nothing
BATCH = int(os.environ.get("BATCH", "4"))
LIMIT = int(os.environ.get("LIMIT", "0"))
SLICE_MAX, OWN_MAX = 24, 6
SHARED_MAX, SHARED_CAT_MAX = 24, 18  # build_blind_v1 BLIND_BULK
WIDE = os.environ.get("WIDE") == "1"  # the shared rows: the latest row of every category offered that day (not 18), then 24 latest rows
SIM = os.environ.get("SIM", "") in ("1", "2")  # owner, 2026-10-02: after the payee's own rows, rows of the payees whose names embed nearest
SIM2 = os.environ.get("SIM") == "2"  # owner, 2026-10-03 ("Costco" and "cstco whsl"): alias rows (cleaned names' character 3-5-gram TF-IDF
# cosine >= ALIAS_MIN) right after the payee's own rows, then similar payees ranked by reciprocal rank fusion of embedding and n-gram neighbours
ALIAS_MIN, ALIAS_MAX = 0.7, 6
SIM_MAX, SIM_PER_PAYEE = 6, 2  # build_blind_v1's similar-payee step, with embeddings (REPORT 152) in place of the generator's kinds
DESC = os.environ.get("DESC") == "1"  # owner, 2026-10-03: each offered category followed by its latest payees ("e.g. A, B, C"; row 136's form)
PLINE = os.environ.get("PLINE") == "1"  # owner, 2026-10-03: a line before the transaction with the categories the payee was filed under
PLINE_HEAD = "Earlier you filed this payee as: "
FIELDS = os.environ.get("FIELDS") == "1"  # row 176: under each history row its "Clean payee / Via / Ordered from / Memo / Location / Store /
# Reference" lines (synthetic households only: their transactions carry the parts); the query stays bare
ROWIDS = os.environ.get("ROWIDS") == "1"  # row 177: history rows numbered in prompt order ("Transaction: [12] 2024-05-01 | ...")
RATIONALE = os.environ.get("RATIONALE") == "1"  # row 177: each item carries "rationale", the reason for its category citing rows by number
FIELD_ORDER = ("Clean payee", "Via", "Ordered from", "Memo", "Location", "Store", "Reference")


def yaml_value(v):
    """A field value as strict YAML (owner, 2026-10-03: a standard format): plain unless YAML would misread it (numbers, booleans, a
    leading indicator character, ": " or " #" inside), then double-quoted; the block parses with yaml.safe_load to str -> str."""
    import re as _re
    if _re.search(r"^[\s*&!|>%@`'\"{\[?,#-]|: | #|[\s:]$", v) or v.lower() in ("yes", "no", "true", "false", "null", "on", "off", "y", "n", "~") \
            or _re.fullmatch(r"[\d.eE+_:-]+|0x[0-9a-fA-F]+|0o[0-7]+|\.(inf|nan)", v, _re.I):
        return json.dumps(v, ensure_ascii=False)
    return v
KIND = os.environ.get("KIND", "")  # row 172: "new" | "all": a "Kind: <Overture category>" line under the transaction (overture_lookup.py's
# match of the payee), for first-time payees only or for every matched payee (row 149's line, read zero-shot as in REPORT 144)
SFX = ("_ids" if ROWIDS else "") + ("_fields" if FIELDS else "") + ("_wide" if WIDE else "") + ("_sim2" if SIM2 else "_sim" if SIM else "") + ("_desc" if DESC else "") + ("_pline" if PLINE else "") + (f"_kind{KIND}" if KIND else "")  # items_wide.json / items_sim.json, scores_<reader>_split<sfx>.jsonl
EMB_TEXT = "Payee as it appears on a bank statement: {}"  # REPORT 152's payee rendering
CUE = "\nIn one word, the kind of spending:"
SHARED_HEAD, NEAR_HEAD = "Earlier transactions:", "Earlier transactions at this payee and similar payees:"
WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
RTA = "Inflow: Ready to Assign"
QUESTION = "Which of this user's categories does the last transaction belong to?"
RECIPE = "decider_decider-4b_none_h100bf16st800_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_oth50_aux100_labrand255_laylabelled_shots_ev10soft_lora"
RECIPE_35B = "decider_decider-35b-a3b_none_h200bf16st800_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_aux100_labrand255_laylabelled_shots_ev10soft_lora"  # REPORT 122
ADAPTER = {"recipe": RECIPE, "big-recipe": RECIPE_35B}.get(READER, "")
ADAPTER_FROM = {"recipe": "r124-oth-s0", "big-recipe": "r125-dec35b-recipe"}.get(READER, "")  # the Modal job that trained it
if READER.startswith("adapter:"):  # row 173: any decider-4B adapter on the results volume, READER=adapter:<name>@<job tag>
    ADAPTER, ADAPTER_FROM = READER[len("adapter:"):].split("@")


def _private(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    return path


def _clean(name):
    return " ".join(name.split())


def build():
    items = build_items(json.loads(CACHE.read_text())["budget"])
    p = _private(OUT / f"items{SFX}.json")
    p.write_text(json.dumps(dict(budget=BUDGET, n=len(items), items=items)))
    os.chmod(p, 0o600)
    print(f"{len(items)} items ({Counter(i['kind'] for i in items)}); gold missing from options: {sum(i['answer'] < 0 for i in items)}; "
          f"options median {sorted(len(i['options']) for i in items)[len(items) // 2]}, max {max(len(i['options']) for i in items)}")


def rationale_of(rows, qi, q, rid):
    """Row 177: why the query got its category, from the generator's reason, citing the history rows shown by number: what the payee
    looks like, this payee's earlier rows, similar payees' rows (same merchant kind) and where they went, and the specific cause."""
    from ai_experiments import taxonomy_v2 as TX
    shown = sorted(rid, key=rid.get)
    ref = lambda ix: " ".join(f"[{rid[i]}]" for i in ix)  # noqa: E731
    gold, kind, reason = q["cat"], q.get("mkind"), (q.get("reason") or [None])[0]
    who = q["parts"].get("Clean payee") if q.get("parts") else None
    parts = []
    if kind == "p2p":
        parts.append(f"a person-to-person payment to {who}" + (f' with the memo "{q["parts"]["Memo"]}"' if q["parts"].get("Memo") else ""))
    elif kind and kind in TX.KINDS:
        parts.append(f"this looks like a {TX.KINDS[kind].split(' (')[0]}" + (f" ({who})" if who else ""))
    same = [i for i in shown if rows[i]["payee_id"] == q["payee_id"]][::-1][:4]
    parts.append("earlier at this payee: " + ", ".join(f"[{rid[i]}] {rows[i]['cat']}" for i in same) if same else "no earlier rows at this payee")
    if kind and kind not in ("p2p", "income", "transfer"):
        sim = [i for i in shown if rows[i].get("mkind") == kind and rows[i]["payee_id"] != q["payee_id"]][::-1]
        to_gold = [i for i in sim if rows[i]["cat"] == gold][:4]
        if to_gold:
            parts.append(f"other payees of this kind {ref(to_gold)} were filed to {gold}")
        elif sim:
            parts.append(f"other payees of this kind {ref(sim[:3])} went to " + ", ".join(dict.fromkeys(rows[i]["cat"] for i in sim[:3])))
    in_cat = [i for i in shown if rows[i]["cat"] == gold][::-1][:4]
    cause = {"trip": f"a trip covers this date ({ref(in_cat) or 'no rows of it shown yet'} filed to {gold})",
             "travel": "travel booked outside any trip window", "holiday": f"it falls in the {gold} season",
             "phase": f"{gold} is in use for this period", "property": f"it belongs to the property's {gold}",
             "pet": f"it is for the pet ({gold})", "treat": f"a small weekday coffee goes to {gold}",
             "person": f"this kind of spending goes to {gold} part of the time", "reorg": f"this kind has gone to {gold} lately ({ref(in_cat) or 'none shown'})",
             "habit": f"this kind of spending usually goes to {gold}", "catchall": f"nothing more specific fits, so {gold}",
             "misfile": "it is filed against the pattern (a misfile)", "store": f"this store has its own category, {gold}",
             "bill": f"a recurring bill filed to {gold}", "service": f"a regular service filed to {gold}",
             "p2p_memo": f"the memo says the purpose: {gold}", "p2p": f"no memo; {gold} as for this person before" if same else f"no memo; {gold}",
             "income": "income goes to Ready to Assign", "savings": f"a savings transfer to {gold}", "wish": f"the planned purchase {gold}"}.get(reason)
    if cause:
        parts.append(cause)
    return "; ".join(parts) + "."


def build_items(b):
    """One item per approved, categorised, non-split transaction of budget document b (the cache's "budget" object), in date order."""
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
            names_by[_clean(c["name"])].append(c["id"])

    def label(cid):
        c = cats[cid]
        if internal(c):
            return RTA if c["name"] == RTA else None
        n = _clean(c["name"])  # names as the layouts read them back (oneslot.parse strips): 27 of this budget's had outer spaces
        return f'{_clean(groups[c["category_group_id"]]["name"])}: {n}' if len(names_by[n]) > 1 else n

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
                         fields=f"{t['date']} | {payee} | ${amt:.2f} | {WD[d.weekday()]}",
                         extra="".join(f"\n{k}: {yaml_value(str(t['parts'][k]))}" for k in FIELD_ORDER if FIELDS and t.get("parts") and k in t["parts"]),
                         mkind=t.get("kind"), reason=t.get("reason"), parts=t.get("parts") or {}))
    rows.sort(key=lambda r: (r["date"], r["id"]))
    by_payee_name, neighbours, aliases = defaultdict(list), {}, {}
    for i, r in enumerate(rows):
        by_payee_name[r["payee"]].append(i)
    if SIM:
        import numpy as np
        e = np.load(OUT / "payee_emb.npz", allow_pickle=False)
        names, X = list(e["names"]), e["vecs"].astype(np.float32)
        X /= np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
        Sm = X @ X.T; np.fill_diagonal(Sm, -np.inf)
        top = np.argsort(-Sm, axis=1)[:, :200]
        neighbours = {n: [names[j] for j in top[k]] for k, n in enumerate(names)}
        if SIM2:
            import re
            from sklearn.feature_extraction.text import TfidfVectorizer

            def clean(n):  # statement noise off: a leading Sale / Return / processor prefix, digits and punctuation
                n = re.sub(r"^(sale|return|sq|tst|pos|debit|purchase|paypal)\b\W*", "", n.lower())
                return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", n)).strip()
            Cm = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform([clean(n) for n in names])
            Sc = (Cm @ Cm.T).toarray(); np.fill_diagonal(Sc, 0)
            ctop = np.argsort(-Sc, axis=1)[:, :200]
            aliases = {n: [names[j] for j in ctop[k] if Sc[k, j] >= ALIAS_MIN] for k, n in enumerate(names)}
            for k, n in enumerate(names):  # reciprocal rank fusion (k = 60) of the two neighbour lists
                sc = defaultdict(float)
                for r, j in enumerate(top[k]):
                    sc[j] += 1 / (60 + r)
                for r, j in enumerate(ctop[k]):
                    sc[j] += 1 / (60 + r)
                neighbours[n] = [names[j] for j in sorted(sc, key=lambda j: -sc[j])]
    items, by_payee, last_idx, cat_payees = [], defaultdict(list), {}, defaultdict(list)
    kinds = json.loads((OUT / "overture_kinds.json").read_text()) if KIND else {}
    used_order = []  # categories by first use
    upto = 0  # rows [0, upto) are history: dated strictly before the query
    day, shared = None, []
    for qi, q in enumerate(rows):
        while rows[upto]["date"] < q["date"]:
            r = rows[upto]
            by_payee[r["payee_id"]].append(upto); last_idx[r["cat"]] = upto
            cat_payees[r["cat"]] = [r["payee"]] + [x for x in cat_payees[r["cat"]] if x != r["payee"]][:2]  # latest 3 distinct
            if r["cat"] not in used_order:
                used_order.append(r["cat"])
            upto += 1
        if q["date"] != day:  # build_blind_v1.build_shared: the day's shared rows, the same for every transaction of the day
            day = q["date"]
            if WIDE:
                ago = str(dt.date.fromisoformat(day) - dt.timedelta(days=365))
                offered = set(visible) | {c for c in used_order if rows[last_idx[c]]["date"] >= ago}
                shared = sorted((i for c, i in last_idx.items() if c in offered), reverse=True)
            else:
                shared = sorted(last_idx.values(), reverse=True)[:SHARED_CAT_MAX]
            cap = len(shared) + SHARED_MAX - SHARED_CAT_MAX if WIDE else SHARED_MAX
            for i in range(upto - 1, -1, -1):
                if len(shared) >= cap:
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
        ctx = "Categories: " + ", ".join(options) + "\n\n" + "".join(f"Transaction: {rows[i]['fields']}{rows[i]['extra']}\nCategory: {rows[i]['cat']}\n\n" for i in chosen)
        near = [i for i in prev_rows if i not in set(shared)][::-1][:OWN_MAX]
        if SIM:  # the nearest earlier payees by name embedding, up to SIM_PER_PAYEE latest rows each, SIM_MAX rows in all
            sim, sh = [], set(shared)
            al = set(aliases.get(q["payee"], []))
            if SIM2:  # the aliases' latest rows next to the payee's own
                al_rows = sorted((i for a in al for i in by_payee_name.get(a, []) if i < upto and i not in sh and i not in set(prev_rows)), reverse=True)
                near += al_rows[:ALIAS_MAX]
            for nb in neighbours.get(q["payee"], []):
                if nb in al:
                    continue
                if len(sim) >= SIM_MAX:
                    break
                rows_nb = [i for i in by_payee_name.get(nb, []) if i < upto and i not in sh][::-1][:SIM_PER_PAYEE]
                sim += rows_nb[:SIM_MAX - len(sim)]
            near += sim
        near = sorted(near)
        rid = {i: k + 1 for k, i in enumerate(list(shared) + [j for j in near if j not in set(shared)])} if ROWIDS else {}
        row_text = lambda i: f"Transaction: {('[%d] ' % rid[i]) if i in rid else ''}{rows[i]['fields']}{rows[i]['extra']}\nCategory: {rows[i]['cat']}\n\n"  # noqa: E731
        pline = ""
        kv = kinds.get(q["payee"]) if KIND else None
        kind_line = f"\nKind: {kv['kind']}" if kv and (KIND == "all" or not prev_rows) else ""
        if PLINE and prev_rows:
            cnt = Counter(rows[i]["cat"] for i in prev_rows).most_common(3)
            pline = PLINE_HEAD + ", ".join(f"{c} ({n})" for c, n in cnt) + "\n\n"
        split = ("Categories: " + ", ".join(options) + f"\n\n{SHARED_HEAD}\n\n" + "".join(map(row_text, shared)) + f"{NEAR_HEAD}\n\n"
                 + "".join(map(row_text, near)) + pline + f"Transaction: {q['fields']}" + kind_line + "\nCategory:")
        desc = {c: ("e.g. " + ", ".join(cat_payees[c]) if cat_payees.get(c) else "nothing filed yet") for c in options} if DESC else None
        prev = [rows[i]["cat"] for i in prev_rows]
        top = Counter(prev[-3:]).most_common(1)
        rule = (top[0][0] if top and top[0][1] >= 2 else prev[-1]) if prev else None
        rule_alias, seen_alias = rule, bool(prev)
        if SIM2:  # YNAB's rule with payee resolution: the payee's and its aliases' rows together, in date order
            pa = [rows[i]["cat"] for i in sorted(set(prev_rows) | {i for a in aliases.get(q["payee"], []) for i in by_payee_name.get(a, []) if i < upto})]
            ta = Counter(pa[-3:]).most_common(1)
            rule_alias, seen_alias = ((ta[0][0] if ta and ta[0][1] >= 2 else pa[-1]) if pa else None), bool(pa)
        items.append(dict(id=q["id"], date=q["date"], kind=q["kind"], payee_seen=bool(prev), first_use=q["cat"] not in last_idx,
                          options=options, answer=options.index(q["cat"]) if q["cat"] in options else -1, gold=q["cat"],
                          prompt=ctx + f"Transaction: {q['fields']}\nCategory:", prompt_split=split, desc=desc, rationale=rationale_of(rows, qi, q, rid) if RATIONALE else None, rule=rule, rule_alias=rule_alias, seen_alias=seen_alias, last=prev[-1] if prev else None, n_hist=upto))
    return items


def run_scoring(todo, fo):
    """Score `todo` (items in this file's shape) with READER / LAYOUT, one JSON line per item to `fo`; progress (counts only) to stderr."""
    import importlib
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ai_experiments import oneslot
    from ai_experiments.paths import ROOT
    path = snapshot_download("Mapika/decider-35b-a3b" if READER.startswith("big") else "Mapika/decider-4b")  # big: untrained, decider's own layout
    sys.path.insert(0, snapshot_download("Mapika/decider-2b", allow_patterns=["decider/*"]))
    P = importlib.import_module("decider.prompt")
    tok = AutoTokenizer.from_pretrained(path)
    lm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    if ADAPTER:
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, str(ROOT / "models" / "adapters" / ADAPTER)).merge_and_unload().eval()

    def built(it):
        state = it["prompt" if LAYOUT == "today" else "prompt_split"][: -len("Category:")].rstrip()
        rng = random.Random(it["id"] if LAYOUT == "today" else f"day-{it['date']}")  # split: labels and option order fixed per day
        if ADAPTER:  # the recipe's layout (labelled rows, rand255 labels)
            return oneslot.build_layout(P, tok, state, QUESTION, it["options"], it["answer"], rng, labels="rand255", layout="labelled_shots",
                                        desc=it.get("desc") or False)
        return oneslot.build(P, tok, state, QUESTION, it["options"], it["answer"], rng, labels="letters")

    import time
    pad = tok.pad_token_id or 0

    def emit(fo, it, b, hrow):
        z = F.linear(hrow.float(), lm.lm_head.weight[torch.tensor(b["labs"], device="cuda")].float())  # fp32: bf16 logits tie at the top on 1.7% of items
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
    items = json.loads((OUT / f"items{SFX}.json").read_text())["items"]
    items = [it for it in items if it["answer"] >= 0 and (not ONLY_NEW or not it["payee_seen"])]
    items = items[:LIMIT] if LIMIT else items
    out = _private(OUT / f"{'check' if NOCACHE else 'scores'}_{READER}_{LAYOUT}{SFX}.jsonl")
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
    items = json.loads((OUT / f"items{SFX}.json").read_text())["items"]
    items = [it for it in items if it["answer"] >= 0 and (not ONLY_NEW or not it["payee_seen"])]
    items = items[:LIMIT] if LIMIT else items
    out = _private(OUT / f"scores_{READER.split('@')[-1] if READER.startswith('adapter:') else READER}_{LAYOUT}{SFX}.jsonl")
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    keep = ("id", "date", "options", "answer", "prompt" if LAYOUT == "today" else "prompt_split", "desc")
    todo = [{k: it.get(k) for k in keep} for it in items if it["id"] not in done]
    print(f"{READER}: {len(done)} done, {len(todo)} to score on Modal", flush=True)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    n = 0
    with os.fdopen(fd, "a") as fo:
        for line in modal_app.private_scores(todo, READER, LAYOUT, ADAPTER_FROM, gpu="H200" if READER.startswith("big") else "H100"):
            fo.write(line + "\n"); n += 1
    print(f"{READER}: {n} scores back", flush=True)


def embed_stream():
    """Payee names as one JSON list on stdin; their embeddings (REPORT 152: the hidden state at the last token of EMB_TEXT + CUE, final
    layer, READER's model) as float16 .npy bytes on stdout; nothing written to disk."""
    import io
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ai_experiments.paths import ROOT
    names = json.load(sys.stdin)
    path = snapshot_download("Mapika/decider-35b-a3b" if READER.startswith("big") else "Mapika/decider-4b")
    tok = AutoTokenizer.from_pretrained(path)
    lm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    if ADAPTER:
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, str(ROOT / "models" / "adapters" / ADAPTER)).merge_and_unload().eval()
    cue = tok(CUE, add_special_tokens=False)["input_ids"]
    out = []
    with torch.no_grad():
        for k in range(0, len(names), 64):
            seqs = [tok(EMB_TEXT.format(n), add_special_tokens=False)["input_ids"] + cue for n in names[k:k + 64]]
            T = -(-max(map(len, seqs)) // 64) * 64
            ids = torch.full((len(seqs), T), tok.pad_token_id or 0, dtype=torch.long); att = torch.zeros_like(ids)
            for i, q in enumerate(seqs):
                ids[i, :len(q)] = torch.tensor(q); att[i, :len(q)] = 1
            h = lm.model(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state.float()
            out += [h[i, len(q) - 1].cpu().numpy() for i, q in enumerate(seqs)]
    buf = io.BytesIO(); np.save(buf, np.stack(out).astype(np.float16)); sys.stdout.buffer.write(buf.getvalue())
    print(f"{len(names)} payees embedded", file=sys.stderr, flush=True)


def embed():
    """The budget's payee names embedded on Modal (modal_app.embed_private), saved privately as OUT/payee_emb.npz."""
    import io
    import numpy as np
    sys.path.insert(0, str(Path(__file__).parent))
    import modal_app
    if os.environ.get("EMBED_SOURCE") == "raw":  # row 171: every raw bank string YNAB imported (import_payee_name_original)
        b = json.loads(CACHE.read_text())["budget"]
        names = sorted({t["import_payee_name_original"] for t in b["transactions"] if not t.get("deleted") and t.get("import_payee_name_original")})
        dest = OUT / ("raw_emb.npz" if not READER.startswith("adapter:") else f"raw_emb_{ADAPTER_FROM}.npz")  # row 175: per reader
    else:
        items = json.loads((OUT / "items.json").read_text())["items"]
        names = sorted({it["prompt"].rsplit("Transaction: ", 1)[1].split(" | ")[1] for it in items})
        dest = OUT / "payee_emb.npz"
    print(f"{len(names)} names to embed on Modal", flush=True)
    vecs = np.load(io.BytesIO(modal_app.embed_private_call(names, READER, ADAPTER_FROM)))
    p = _private(dest)
    np.savez(p, names=np.array(names), vecs=vecs); os.chmod(p, 0o600)
    print(f"saved {vecs.shape}", flush=True)


def stream():
    """Items as one JSON list on stdin, score lines on stdout, nothing written to disk (modal_app.score_private)."""
    run_scoring(json.load(sys.stdin), sys.stdout)


def report():
    items = {it["id"]: it for it in json.loads((OUT / f"items{SFX}.json").read_text())["items"]}
    readers = {}
    for f in sorted(OUT.glob("scores_*.jsonl")):
        readers[f.stem[len("scores_"):]] = {r["id"]: r["lp"] for r in map(json.loads, open(f))}
    n_scored = sum(it["answer"] >= 0 for it in items.values())
    readers = {k: r for k, r in readers.items() if len(r) >= n_scored}  # complete runs only
    ids = [i for i in items if items[i]["answer"] >= 0]  # the 101 whose gold was not offered are left out (counted in `build`)

    def rank(lp, a):  # a tie at the gold's score counts against it (bf16 logits tie; the scorer now reads out in fp32)
        return 1 + sum(x > lp[a] for x in lp) + sum(x == lp[a] for k, x in enumerate(lp) if k != a)
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
    for g in [g for g in order if groups.get(g)]:
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
    {"build": build, "score": score, "modal": modal, "stream": stream, "embed": embed, "embed_stream": embed_stream, "report": report}[sys.argv[1]]()
