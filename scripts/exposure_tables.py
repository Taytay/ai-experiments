"""Tables for REPORT.md section 45 (PLAN step 39, REAL-8): the parametric exposure curve and the chat template.

  45.1  the exposure curve: the SFT categoriser with the fact DB in its training sequences at 1 pass (200 steps at 30%, section 38),
        3.3 passes (400 at 50%, seeds 0 to 2, section 43), 6.7 (800 at 50%) and 13.3 (1,600 at 50%), three seeds each; REAL-6 without a
        record (all, unseen, DB-only, DB-only opaque / known), ARC-Easy, MMLU, the v2 ICL suite, training minutes; the no-DB seeds beside them
  45.2  the chat template: the instruct base and the no-DB and record-in-prompt categorisers in the plain REAL-6 format (sections 38, 43)
        and through Qwen2.5-Instruct's chat template (real6.chat_prompt), with the share of items predicted the same
usage: uv run python scripts/exposure_tables.py            prints markdown; '-' where a run is missing
"""
import json
import statistics

from ai_experiments import real6 as R6
from ai_experiments.paths import ROOT

R = ROOT / "results"
CAT = "categoriser_Qwen2.5-3B-Instruct"
DOC = R6.load()
ITEM = {it["id"]: it for it in DOC["items"]}
DB_ONLY = R6.db_only_merchants()
N_DB_TEXTS = 4 * len(DOC["fact_db"])  # the record and three paraphrases per merchant


def load(p):
    return json.load(open(p)) if p.exists() else {}


def per_item(tag, cond):
    p = R / "per_item" / f"real6_{tag}.{cond}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else None


def acc(recs, pred=lambda r: True):
    xs = [r["correct"] for r in recs if pred(r)] if recs else []
    return 100 * sum(xs) / len(xs) if xs else None


def db_only(r):
    it = ITEM[r["id"]]; return not it["seen"] and it["merchant"] in DB_ONLY


def fmt(v, d=1):
    return "-" if v is None else f"{v:.{d}f}"


def seeds_cell(vals, d=1):
    """mean +- sd over the seeds that exist, then the per-seed values."""
    xs = [v for v in vals if v is not None]
    if not xs:
        return "-"
    body = " / ".join(fmt(v, d) for v in vals)
    return f"{statistics.mean(xs):.{d}f} +- {statistics.stdev(xs):.{d}f} ({body})" if len(xs) > 1 else body


ARMS = [  # label, passes, [(adapter suffix, stats suffix) per seed]
    ("no DB", "0", [("none", "none"), ("none_s1", "none_s1"), ("none_s2", "none_s2")]),
    ("200 steps at 30%", f"{200 * 16 * 0.3 / N_DB_TEXTS:.1f}", [("param", "param")]),
    ("400 steps at 50%", f"{400 * 16 * 0.5 / N_DB_TEXTS:.1f}", [("param_x2", "param_x2"), ("param_x2_s1", "param_x2_s1"), ("param_x2_s2", "param_x2_s2")]),
    ("800 steps at 50%", f"{800 * 16 * 0.5 / N_DB_TEXTS:.1f}", [(f"param_x4_s{s}", f"param_x4_s{s}") for s in range(3)]),
    ("1,600 steps at 50%", f"{1600 * 16 * 0.5 / N_DB_TEXTS:.1f}", [(f"param_x8_s{s}", f"param_x8_s{s}") for s in range(3)]),
]


def t451():
    print(f"**Table 45.1: the parametric exposure curve. The SFT categoriser (rank-64 QLoRA on the 4-bit Qwen2.5-3B-Instruct, lr 1e-4, 16 sequences per step) with the "
          f"{N_DB_TEXTS} fact-DB texts (record and three paraphrases per merchant) mixed into its training sequences, by the number of passes over them; REAL-6 without a record, "
          f"the ARC / MMLU / ICL items of exp_items_v2 (4-bit reads, as Table 38.3); mean +- sd over seeds, then seeds 0 / 1 / 2**\n")
    print("| measure | " + " | ".join(f"{a} ({p} passes)" for a, p, _ in ARMS) + " |")
    print("|---|" + "---|" * len(ARMS))
    rows = [("REAL-6, all items", lambda sfx: load(R / f"real6_{CAT}_{sfx}_lora.json").get("noctx", {}).get("R6_all")),
            ("REAL-6, unseen merchant", lambda sfx: load(R / f"real6_{CAT}_{sfx}_lora.json").get("noctx", {}).get("R6_unseen_all")),
            ("REAL-6, unseen, new word", lambda sfx: load(R / f"real6_{CAT}_{sfx}_lora.json").get("noctx", {}).get("R6_unseen_new")),
            ("DB-only merchants", lambda sfx: acc(per_item(f"{CAT}_{sfx}_lora", "noctx"), db_only)),
            ("DB-only, opaque", lambda sfx: acc(per_item(f"{CAT}_{sfx}_lora", "noctx"), lambda r: db_only(r) and not ITEM[r["id"]]["known"])),
            ("DB-only, known chain", lambda sfx: acc(per_item(f"{CAT}_{sfx}_lora", "noctx"), lambda r: db_only(r) and ITEM[r["id"]]["known"])),
            ("ARC-Easy", lambda sfx: load(R / f"items2_{CAT}_{sfx}_lora.json").get("trained", {}).get("K_arc_easy")),
            ("MMLU 5-shot", lambda sfx: load(R / f"items2_{CAT}_{sfx}_lora.json").get("trained", {}).get("K_mmlu")),
            ("ICL symbol (v2 suite)", lambda sfx: load(R / f"items2_{CAT}_{sfx}_lora.json").get("trained", {}).get("ICL2_symbol_mean")),
            ("ICL natural (v2 suite)", lambda sfx: load(R / f"items2_{CAT}_{sfx}_lora.json").get("trained", {}).get("ICL2_natural_mean"))]
    for label, get in rows:
        print(f"| {label} | " + " | ".join(seeds_cell([get(a) for a, _ in seeds]) for _, _, seeds in ARMS) + " |")
    for label, key in (("training minutes", "train_minutes"), ("final loss (last 10 steps)", "final_loss")):
        print(f"| {label} | " + " | ".join(seeds_cell([load(R / f"categoriser_llm_{st}.json").get(key) for _, st in seeds], 3 if key == "final_loss" else 0) for _, _, seeds in ARMS) + " |")


def t452():
    print("\n**Table 45.2: the REAL-6 prompt in the plain format (sections 38, 43) and through Qwen2.5-Instruct's chat template (everything up to the query line as the user turn, "
          "\"Category:\" opening the assistant turn; the categorisers trained and scored in it); accuracy %, 4-bit throughout; the last column is the share of items given the same "
          "prediction by the plain and chat readings**\n")
    rows = [("Instruct base, no record", "Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct_chat", "noctx"),
            ("Instruct base + record", "Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct_chat", "ctx"),
            ("SFT, no DB", f"{CAT}_none_lora", f"{CAT}_none_chat_lora", "noctx"),
            ("SFT + record in prompt", f"{CAT}_ret_lora", f"{CAT}_ret_chat_lora", "ctx")]
    print("| model | plain: all / unseen / DB-only / opaque | chat: all / unseen / DB-only / opaque | same prediction |")
    print("|---|---|---|---|")
    for label, t_plain, t_chat, cond in rows:
        cells = []
        for t in (t_plain, t_chat):
            r = load(R / f"real6_{t}.json").get(cond, {}); recs = per_item(t, cond)
            cells.append(" / ".join([fmt(r.get("R6_all")), fmt(r.get("R6_unseen_all")), fmt(acc(recs, db_only)), fmt(acc(recs, lambda x: db_only(x) and not ITEM[x["id"]]["known"]))]))
        a, b = per_item(t_plain, cond), per_item(t_chat, cond)
        same = "-"
        if a and b:
            pb = {r["id"]: r["pred"] for r in b}
            same = f"{100 * sum(r['pred'] == pb.get(r['id']) for r in a) / len(a):.1f}%"
        print(f"| {label} | {cells[0]} | {cells[1]} | {same} |")
    print("\nTraining (chat format): " + "; ".join(f"{db}: {load(R / f'categoriser_llm_{db}_chat.json').get('train_minutes', '-')} min, final loss {load(R / f'categoriser_llm_{db}_chat.json').get('final_loss', '-')}"
                                            f" (plain {load(R / f'categoriser_llm_{db}.json').get('final_loss', '-')})" for db in ("none", "ret")))


def main():
    t451(); t452()


if __name__ == "__main__":
    main()
