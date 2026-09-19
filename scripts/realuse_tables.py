"""Tables for REPORT.md section 36 (PLAN step 22: REAL-1, REAL-4, GRAPH-2, GRAPH-6) from results/realuse.json.

Table 36.1: 12-way accuracy on the frozen transaction history by classifier and k labelled transactions per category, per encoder,
raw and normalised strings. Table 36.2: the prototype and label-propagation classifiers at k = 3 broken down by merchant frequency
bucket, known vs opaque merchant, and seen vs unseen merchant. Table 36.3: amount and weekday features.

usage: uv run python scripts/realuse_tables.py            prints markdown; '-' where a cell is missing
"""
import json

from ai_experiments.paths import ROOT

P = ROOT / "results" / "realuse.json"
R = json.load(open(P)) if P.exists() else {}


def g(key, k):
    v = R.get(key, {}).get(k)
    return "-" if v is None else f"{v:g}"


def main():
    encs = [("MiniLM", "minilm"), ("bge-base", "bge")]
    print("**Table 36.1: 12-way category accuracy on the 3,000-transaction history (frozen encoders; 10 trials; chance 8.3) by classifier and labelled transactions per category k; "
          "name = the category name's embedding alone, mix = its unit mean with the k-example centroid, lp = label propagation over the kNN graph of all 3,000 strings**\n")
    print("| encoder | strings | k | name (k=0 vector) | prototype | mix | label propagation |")
    print("|---|---|---|---|---|---|---|")
    for el, e in encs:
        for norm in ("raw", "norm"):
            for k in (1, 3, 10):
                key = f"{e}.{norm}.text.k{k}"
                print(f"| {el} | {norm} | {k} | {g(f'{e}.{norm}.text.k0', 'name_all')} | {g(key, 'proto_all')} | {g(key, 'mix_all')} | {g(key, 'lp_all')} |")
    print("\n**Table 36.2: the prototype and label-propagation classifiers by merchant frequency bucket (head = top 10% of merchants by count, torso = next 30%, tail = rest), "
          "known (real chain) vs opaque merchant, whether the test transaction's merchant is among the labelled examples, and the unseen merchants split by known vs opaque (accuracy %, normalised strings)**\n")
    print("| encoder | k | method | all | head | torso | tail | known | opaque | seen merchant | unseen merchant | unseen, known chain | unseen, opaque |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for el, e in encs:
        for k in (1, 3, 10):
            key = f"{e}.norm.text.k{k}"
            for meth in ("proto", "lp"):
                print(f"| {el} | {k} | {meth} | " + " | ".join(g(key, f"{meth}_{grp}") for grp in ("all", "head", "torso", "tail", "known", "opaque", "seen_merchant", "unseen_merchant", "known_unseen", "opaque_unseen")) + " |")
    print("\n**Table 36.3: amount band and weekday appended to the text embedding (weight 0.5), normalised strings (accuracy %)**\n")
    print("| encoder | k | prototype, text | prototype, text + amount + weekday | lp, text | lp, text + amount + weekday |")
    print("|---|---|---|---|---|---|")
    for el, e in encs:
        for k in (1, 3, 10):
            a, b = f"{e}.norm.text.k{k}", f"{e}.norm.text+af.k{k}"
            print(f"| {el} | {k} | {g(a, 'proto_all')} | {g(b, 'proto_all')} | {g(a, 'lp_all')} | {g(b, 'lp_all')} |")
    for el, e in encs:
        if f"{e}.minutes" in R:
            print(f"\n{el}: {R[f'{e}.minutes']} minutes for the grid.")


if __name__ == "__main__":
    main()
