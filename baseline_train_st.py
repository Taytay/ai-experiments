"""Step 2 (fallback path): fine-tune all-MiniLM-L6-v2 on the GPU with plain
sentence-transformers and a hand-rolled loop. Avoids `datasets`/pyarrow and
triton so it runs even while Smart App Control blocks those native libs.
Proves the driver + CUDA stack can train an embeddings model end to end."""
import time

import torch
from sentence_transformers import SentenceTransformer, losses

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

model = SentenceTransformer(MODEL, device="cuda")
loss_fn = losses.MultipleNegativesRankingLoss(model)
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train()

def batch_features(texts):
    feats = model.tokenize(texts)
    return {k: v.to("cuda") for k, v in feats.items()}

q, d = pairs[0]
def sim():
    model.eval()
    with torch.no_grad():
        e = model.encode([q, d, pairs[1][1]], convert_to_tensor=True, normalize_embeddings=True)
    model.train()
    return (e[0] @ e[1]).item(), (e[0] @ e[2]).item()

print("before  pos/neg sim:", ["%.3f" % s for s in sim()])
bs, t0 = 16, time.time()
for epoch in range(3):
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        feats = [batch_features([p[0] for p in chunk]), batch_features([p[1] for p in chunk])]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(feats, labels=None)
        loss.backward()
        opt.step(); opt.zero_grad()
    print(f"epoch {epoch} loss {loss.item():.4f}")
torch.cuda.synchronize()
print(f"train time {time.time() - t0:.1f}s, peak vram {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
print("after   pos/neg sim:", ["%.3f" % s for s in sim()])
print("OK: embeddings model fine-tuned on GPU (plain sentence-transformers)")
