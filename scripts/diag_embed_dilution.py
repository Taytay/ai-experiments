"""Diagnostic: why do added merchant tokens hurt bank-string accuracy for MiniLM?
Hypothesis: mean pooling dilution (1 identity token vs ~4 subword pieces among noise).
Test: retrain ft_newtok_mean, then score bank strings (a) split by whether the added
token matched, (b) with the merchant name repeated 4x to restore its pooling share,
(c) with pooling replaced by the [CLS] token. Also (d) subword model with 4x repeat."""
import sys, random
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import merchants as M
import importlib.util
spec = importlib.util.spec_from_file_location("ev", Path(__file__).parent / "exp_embed_vocab.py")
src = (Path(__file__).parent / "exp_embed_vocab.py").read_text().split("results = {}")[0]  # defs only
ns = {"__file__": str(Path(__file__).parent / "exp_embed_vocab.py")}; exec(compile(src, "exp_embed_vocab_defs", "exec"), ns)
load, embed, train, add_tokens = ns["load"], ns["embed"], ns["train"], ns["add_tokens"]
train_m, all_m, CAT_TEXT = ns["train_m"], ns["all_m"], ns["CAT_TEXT"]

def acc(model, tok, texts, labels):
    cats = embed(model, tok, [CAT_TEXT[c] for c in M.CATEGORY_LIST])
    pred = (embed(model, tok, texts) @ cats.T).argmax(1).cpu()
    return round(100 * (pred == torch.tensor(labels)).float().mean().item(), 1)

def bank_rep(m, k):  # repeat the merchant token k times inside the statement
    return M.bank_string(m).replace(m["name"].upper().replace("&", "AND"), " ".join([m["name"].upper().replace("&", "AND")] * k))

labels = [M.CATEGORY_LIST.index(m["category"]) for m in train_m]
for cond in ("subword", "newtok_mean"):
    tok, model = load()
    if cond == "newtok_mean":
        add_tokens(model, tok, [m["name"] for m in all_m], "mean")
    train(model, tok); model.eval()
    print(f"\n[{cond}]")
    print("  bank (as-is):            ", acc(model, tok, [M.bank_string(m) for m in train_m], labels))
    print("  bank, name x4:           ", acc(model, tok, [bank_rep(m, 4) for m in train_m], labels))
    print("  name only:               ", acc(model, tok, [m["name"] for m in train_m], labels))
    print("  name + ' store 4970 tucson az':", acc(model, tok, [m["name"] + " store 4970 tucson az" for m in train_m], labels))
    if cond == "newtok_mean":
        matched = [(m, l) for m, l in zip(train_m, labels) if m["name"].lower() in tok.tokenize(M.bank_string(m))]
        unmatched = [(m, l) for m, l in zip(train_m, labels) if m["name"].lower() not in tok.tokenize(M.bank_string(m))]
        print(f"  bank, token matched ({len(matched)}):  ", acc(model, tok, [M.bank_string(m) for m, _ in matched], [l for _, l in matched]))
        print(f"  bank, token unmatched ({len(unmatched)}):", acc(model, tok, [M.bank_string(m) for m, _ in unmatched], [l for _, l in unmatched]))
