"""Step 2 (minimal path): fine-tune all-MiniLM-L6-v2 on the GPU with only
transformers + torch (no datasets/pyarrow, no triton), so it runs even while
Smart App Control blocks those unsigned native libs.
Proves the driver + CUDA stack can train an embeddings model end to end."""
import time

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
pairs = [
    ("How do I reset my password?", "Go to Settings > Account > Reset password."),
    ("What is the refund policy?", "Refunds are issued within 14 days of purchase."),
    ("How do I export my data?", "Use File > Export to download a CSV."),
    ("Can I change my budget currency?", "Currency can be changed under Budget Settings."),
    ("How do I split a transaction?", "Select the transaction and click Split."),
    ("Why is my bank not syncing?", "Reconnect the account under Linked Accounts."),
    ("How do I delete a category?", "Right-click the category and choose Delete."),
    ("Where do I see my net worth?", "Open Reports and pick Net Worth."),
] * 8  # 64 pairs

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)

def embed(texts):
    b = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
    out = model(**b).last_hidden_state
    mask = b["attention_mask"].unsqueeze(-1).to(out.dtype)
    return F.normalize((out * mask).sum(1) / mask.sum(1), dim=-1)

q, d, neg = pairs[0][0], pairs[0][1], pairs[1][1]
def sim():
    model.eval()
    with torch.no_grad():
        e = embed([q, d, neg]).float()
    model.train()
    return (e[0] @ e[1]).item(), (e[0] @ e[2]).item()

print("before  pos/neg sim:", ["%.3f" % s for s in sim()])
model.train()
bs, t0 = 16, time.time()
for epoch in range(3):
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a = embed([p[0] for p in chunk]); p_ = embed([p[1] for p in chunk])
            scores = (a @ p_.T) * 20.0  # in-batch negatives (MultipleNegativesRankingLoss)
            loss = F.cross_entropy(scores.float(), torch.arange(len(chunk), device="cuda"))
        loss.backward()
        opt.step(); opt.zero_grad()
    print(f"epoch {epoch} loss {loss.item():.4f}")
torch.cuda.synchronize()
print(f"train time {time.time() - t0:.1f}s, peak vram {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
print("after   pos/neg sim:", ["%.3f" % s for s in sim()])
print("OK: embeddings model fine-tuned on GPU (transformers + torch only)")
