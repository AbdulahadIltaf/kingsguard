"""
KingsGuard L1 Model Merge Utility
Merges the trained PEFT adapter into the base DeBERTa-v3-base model
and saves a self-contained HuggingFace checkpoint to kingsguard_l1_merged.

Run AFTER train_l1.py completes.
"""

import os
import shutil

ADAPTER_DIR = "./kingsguard_l1_final"
MERGED_DIR  = "./kingsguard_l1_merged"
BASE_MODEL  = "microsoft/deberta-v3-base"

# Safety: remove old corrupt merged dir if it exists
if os.path.exists(MERGED_DIR):
    print(f"[Merge] Removing old merged dir: {MERGED_DIR}")
    shutil.rmtree(MERGED_DIR)

print(f"[Merge] Loading base model: {BASE_MODEL}")
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

id2label = {0: "BENIGN",    1: "MALICIOUS"}
label2id = {"BENIGN": 0,   "MALICIOUS": 1}

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

print(f"[Merge] Loading PEFT adapter from: {ADAPTER_DIR}")
peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

print("[Merge] Merging and unloading LoRA weights ...")
merged_model = peft_model.merge_and_unload()

# Ensure config has correct classification labels
merged_model.config.id2label  = id2label
merged_model.config.label2id  = label2id
merged_model.config.num_labels = 2

merged_model.eval()

print(f"[Merge] Saving merged model to: {MERGED_DIR}")
merged_model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)

print("[Merge] Done! Verify with:")
print("  uv run python -c \"from tools import get_l1_model; t,m = get_l1_model(); print('OK')\"")
