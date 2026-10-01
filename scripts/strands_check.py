"""PLAN step 151 (owner, 2026-10-01: check that strands-decider is used correctly, on easy inputs, before reading it on the hard sets).
Four checks on StrandsAgents/strands-decider-2B-hobson-v19, on one GPU:
  A. strands' own engine (strands_decider.infer.load_engine) on the README's recorded example (docs/inference.md, "Ask"): noul 0.801;
     choice technical 0.748 / billing 0.234 / sales 0.018 (confidence 0.622); score 1.24 (confidence 0.578; 0.093 / 0.578 / 0.330)
  B. strands' own evaluate_checkpoint on its published held-out generated rows (data/generators/*/gen_eval.jsonl at the pinned commit),
     against the accuracies the checkpoint's eval/summary.json reports
  C. this repo's reader (ai_experiments.strands, used by exp_decision_models FAMILY=strands) against strands' engine on the same
     questions: the choice probabilities after the checkpoint's choice temperature (the engine applies it; ours reads the raw head)
  D. an easy ladder for categorisation, through our reader:
     L1  one transaction of a well-known merchant, six everyday categories, no history (each also with the options reversed)
     L2  the same queries in our prompt format: the category list, four labelled rows of other merchants, then the query
     L3  in-context lookup: twelve categories, four of them invented names; the query's merchant appears twice in the history filed
         under an invented name, so only the history gives the answer
usage: uv run --with strands-decider==0.1.0 --with transformers==5.17.0 --with flash-linear-attention --with "peft>=0.21" --with torch==2.13.0
       --with torchvision==0.28.0 python scripts/strands_check.py
"""
import json
import random
import urllib.request

import torch

from ai_experiments import strands as SD

SHA = "f91487ab8f7e4b4967ae57e46b8d90e91e67d616"  # strands-labs/strands-decider, 2026-10-01
GEN = ["gen_adequacy", "gen_v16", "gen_flips", "gen_weak"]
QUESTION = "Which category does this transaction belong to?"
MERCHANTS = [("SHELL OIL 57442 AUSTIN TX", "Fuel", 48.20), ("TRADER JOE'S #552 PORTLAND OR", "Groceries", 83.10),
             ("NETFLIX.COM LOS GATOS CA", "Streaming", 15.49), ("STARBUCKS STORE 10234 SEATTLE", "Dining out", 6.45),
             ("CHEVRON 0091234 DENVER CO", "Fuel", 52.10), ("SAFEWAY #1823 OAKLAND CA", "Groceries", 121.33),
             ("SPOTIFY USA NEW YORK NY", "Streaming", 11.99), ("CHIPOTLE 2231 AUSTIN TX", "Dining out", 14.85),
             ("CVS/PHARMACY #04512 BOSTON MA", "Health", 23.70), ("WALGREENS #7731 CHICAGO IL", "Health", 18.42),
             ("PG&E WEB ONLINE PAYMENT", "Utilities", 142.18), ("CITY OF AUSTIN UTILITIES", "Utilities", 96.40)]
CATS = ["Groceries", "Dining out", "Fuel", "Streaming", "Health", "Utilities"]


def fetch(path):
    return urllib.request.urlopen(f"https://raw.githubusercontent.com/strands-labs/strands-decider/{SHA}/{path}").read().decode()


def ours(m, state, question, options):
    lp = SD.log_probs(m.cuda(), [SD.encode(m.tokenizer, state, question, options)])[0].float()
    return lp.exp().tolist()


def ladder(m):
    rng = random.Random(0)
    rows = {"L1": [], "L2": [], "L3": []}
    for name, cat, amt in MERCHANTS:
        st = f"Transaction: {name} | ${amt:.2f}"
        for opts in (CATS, CATS[::-1]):
            p = ours(m, st, QUESTION, opts); rows["L1"].append(opts[max(range(len(opts)), key=p.__getitem__)] == cat)
        others = [x for x in MERCHANTS if x[1] != cat]; shots = rng.sample(others, 4)
        hist = "\n\n".join(f"Transaction: {n} | ${a:.2f}\nCategory: {c}" for n, c, a in shots)
        st = f"Categories: {', '.join(CATS)}\n\n{hist}\n\nTransaction: {name} | ${amt:.2f}"
        p = ours(m, st, "Which of this user's categories does the last transaction belong to?", CATS)
        rows["L2"].append(CATS[max(range(len(CATS)), key=p.__getitem__)] == cat)
    invented = ["Kogumur", "Nitelor", "Vozu", "Bona"]
    generic = ["Groceries", "Dining out", "Fuel", "Streaming", "Health", "Utilities", "Rent", "Travel"]
    for k, (name, _, amt) in enumerate(MERCHANTS):
        gold = invented[k % 4]; opts = generic + invented; rng.shuffle(opts)
        fill = [x for x in MERCHANTS if x[0] != name]
        hist = [(n, c, a) for n, c, a in rng.sample(fill, 5)] + [(name, gold, amt * 0.9), (name, gold, amt * 1.1)]
        hist += [(rng.choice(fill)[0], invented[(k + 1) % 4], 30.0)]
        rng.shuffle(hist)
        body = "\n\n".join(f"Transaction: {n} | ${a:.2f}\nCategory: {c}" for n, c, a in hist)
        st = f"Categories: {', '.join(opts)}\n\n{body}\n\nTransaction: {name} | ${amt:.2f}"
        p = ours(m, st, "Which of this user's categories does the last transaction belong to?", opts)
        rows["L3"].append(opts[max(range(len(opts)), key=p.__getitem__)] == gold)
    for k, v in rows.items():
        print(f"D {k}: {sum(v)}/{len(v)} right", flush=True)


if __name__ == "__main__":
    from strands_decider.data.format import Example
    from strands_decider.evaluate import evaluate_checkpoint
    from strands_decider.infer import load_engine
    from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion
    eng = load_engine(SD.V19, device="cuda")
    state = "Help! My payouts have been failing for 3 days."
    r = eng.ask(state, {"noul_0": NoulQuestion(instructions="Does this convey urgency?"),  # as `strands-decider ask` builds them
                        "choice_0": ChoiceQuestion(instructions="Which team should handle this?", criteria={"billing": "", "technical": "", "sales": ""}),
                        "score_0": ScoreQuestion(instructions="How frustrated is the writer?", criteria=["calm", "frustrated", "very angry"])})
    print("A recorded: noul 0.801 | technical 0.748 billing 0.234 sales 0.018 conf 0.622 | score 1.24 conf 0.578 (0.093 0.578 0.330)")
    print("A ours    :", r.model_dump_json(), flush=True)

    summary = json.loads(urllib.request.urlopen("https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19/resolve/main/eval/summary.json").read())
    published = {k.split("] ", 1)[-1]: v for k, v in summary["internal"].items()}
    for g in GEN:
        exs = [Example.from_dict(json.loads(l)) for l in fetch(f"data/generators/{g}/gen_eval.jsonl").splitlines() if l.strip()]
        rep = evaluate_checkpoint(SD.V19, exs, batch_size=16)
        print(f"B {g}: n={rep['overall']['n']} acc={rep['overall']['accuracy']:.3f}", flush=True)
        for t, v in sorted(rep["by_task"].items()):
            pub = published.get(t)
            if v:
                print(f"   {t}: n={v['n']} acc={v['accuracy']:.3f}  published: {pub}", flush=True)

    m = SD.load(SD.V19).eval()
    T = m.config.temperature_by_kind.get("choice", m.config.temperature)
    diffs, agree, n = [], 0, 0
    exs = [Example.from_dict(json.loads(l)) for l in fetch("data/generators/gen_v16/gen_eval.jsonl").splitlines() if l.strip()]
    qs = [(state, "Which team should handle this?", ["billing", "technical", "sales"])] + \
         [(e.state, e.instructions, [o[0] for o in e.options]) for e in exs if e.kind == "choice"][:60]
    with torch.no_grad():
        for st, q, opts in qs:
            ref = eng.ask(st, {"q": ChoiceQuestion(instructions=q, criteria={o: "" for o in opts})}).answers["q"].probabilities
            lp = SD.log_probs(m, [SD.encode(m.tokenizer, st if isinstance(st, str) else json.dumps(st, indent=2, ensure_ascii=False), q, opts)])[0].float()
            p = (lp[:len(opts)] / T).softmax(-1).tolist()
            diffs.append(max(abs(p[k] - ref[o]) for k, o in enumerate(opts)))
            agree += max(range(len(opts)), key=p.__getitem__) == max(range(len(opts)), key=lambda k: ref[opts[k]]); n += 1
    print(f"C ours vs strands' engine on {n} choice questions: top-1 agree {agree}/{n}, max |dp| {max(diffs):.4f}, mean {sum(diffs) / n:.4f}", flush=True)
    with torch.no_grad():
        ladder(m)
