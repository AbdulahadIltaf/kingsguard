"""
KingsGuard L1 Fine-Tuning Script
Trains a DeBERTa-v3-base model on MPDD.csv via LoRA/PEFT.
Splits data 90/10 — saves test split to MPDD_test.csv for benchmark use.
Output adapter: ./kingsguard_l1_final
"""

import json
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# ─────────────────────────────────────────────────────────────
# 1.  Load & Split Dataset
# ─────────────────────────────────────────────────────────────
CSV_PATH = "injecagent_data/MPDD.csv"
TEST_CSV  = "injecagent_data/MPDD_test.csv"

print(f"[Dataset] Loading {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH, header=0, names=["text", "label"])
df = df.dropna(subset=["text", "label"])
df["text"]  = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(int)

# 90 / 10 split — stratified to preserve class balance
train_df, test_df = train_test_split(
    df, test_size=0.10, random_state=42, stratify=df["label"]
)

print(f"[Dataset] Train: {len(train_df)} | Test: {len(test_df)}")
print(f"[Dataset] Label distribution (train):\n{train_df['label'].value_counts()}")

# Save test split for benchmark use in app.py
test_df.to_csv(TEST_CSV, index=False)
print(f"[Dataset] Test split saved to {TEST_CSV}")

# ─────────────────────────────────────────────────────────────
# 2.  Model & Tokeniser
# ─────────────────────────────────────────────────────────────
MODEL_ID = "microsoft/deberta-v3-base"

id2label  = {0: "BENIGN",    1: "MALICIOUS"}
label2id  = {"BENIGN": 0,   "MALICIOUS": 1}

print(f"[Model] Loading tokeniser from {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
test_ds  = Dataset.from_pandas(test_df.reset_index(drop=True))

print("[Dataset] Tokenising ...")
train_ds = train_ds.map(tokenize, batched=True)
test_ds  = test_ds.map(tokenize,  batched=True)

# ─────────────────────────────────────────────────────────────
# 3.  Base Model + LoRA
# ─────────────────────────────────────────────────────────────
print("[Model] Loading base model ...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query_proj", "value_proj"],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ─────────────────────────────────────────────────────────────
# 4.  Training
# ─────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="weighted"),
    }

# DeBERTa + PEFT is incompatible with fp16 AMP (gradient scaler crashes on
# relative-position embeddings). Use bf16 on GPU instead — works on T4/V100/A100.
# Falls back to plain fp32 on CPU.
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
use_fp16 = torch.cuda.is_available() and not use_bf16

training_args = TrainingArguments(
    output_dir="./kingsguard_l1_results",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=use_fp16,
    bf16=use_bf16,
    logging_steps=200,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

print("[Training] Starting ...")
trainer.train()

# ─────────────────────────────────────────────────────────────
# 5.  Save PEFT adapter
# ─────────────────────────────────────────────────────────────
ADAPTER_DIR = "./kingsguard_l1_final"
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"[Training] Adapter saved to {ADAPTER_DIR}")
print("[Training] Run merge_l1.py next to create the final merged model.")
