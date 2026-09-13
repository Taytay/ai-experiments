"""Step 3: fine-tune all-MiniLM-L6-v2 with unsloth's FastSentenceTransformer.
Requires triton (libtriton.pyd) and pyarrow to load, i.e. Smart App Control off
or running under WSL2. Run: uv run python finetune_embed.py"""
import unsloth  # noqa: F401  must be imported before transformers/sentence_transformers
from unsloth import FastSentenceTransformer

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses

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
] * 8
ds = Dataset.from_dict({"anchor": [p[0] for p in pairs], "positive": [p[1] for p in pairs]})

model = FastSentenceTransformer.from_pretrained(
    MODEL,
    max_seq_length=128,
    full_finetuning=True,   # 22M params; LoRA not worth it. Use get_peft_model for larger models.
)

q, d, neg = pairs[0][0], pairs[0][1], pairs[1][1]
def sim():
    e = model.encode([q, d, neg], convert_to_tensor=True, normalize_embeddings=True)
    return (e[0] @ e[1]).item(), (e[0] @ e[2]).item()
print("before  pos/neg sim:", ["%.3f" % s for s in sim()])

trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=ds,
    loss=losses.MultipleNegativesRankingLoss(model),
    args=SentenceTransformerTrainingArguments(
        output_dir="outputs/minilm-unsloth",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        bf16=True,
        logging_steps=4,
        report_to="none",
        dataset_num_proc=1,  # required on Windows per unsloth docs
        save_strategy="no",
    ),
)
trainer.train()
print(f"peak vram {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
print("after   pos/neg sim:", ["%.3f" % s for s in sim()])
model.save_pretrained("outputs/minilm-unsloth/final")
print("OK: unsloth embeddings fine-tune complete")
