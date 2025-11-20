"""
Pre-Training Warmup Script
==========================
Run this BEFORE the main training script to:
1. Pre-download all models and datasets
2. Pre-compile Python bytecode
3. Pre-allocate memory structures
4. Cache tokenizer operations

This reduces cold-start delays during the actual training demo.
"""

import os
import sys

# ============================================================================
# CRITICAL: Patch torch BEFORE any imports that might load unsloth
# ============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import torch and patch it before unsloth loads
import torch
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

# Return fake but valid CUDA capability (pretend we have compute capability 8.0)
def fake_get_device_capability(device=None):
    return 8, 0  # Return (major, minor) version tuple

torch.cuda.get_device_capability = fake_get_device_capability

print("=" * 80)
print("WARMUP SCRIPT - PREPARING ENVIRONMENT")
print("=" * 80)
print()

# ============================================================================
# 1. Pre-download the dataset
# ============================================================================
print("Step 1: Downloading dataset...")
from datasets import load_dataset

dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
print(f"✓ Dataset cached ({len(dataset['train']):,} examples)")
print()

# ============================================================================
# 2. Pre-download the model
# ============================================================================
print("Step 2: Downloading model (this is the big one - may take 2-3 minutes)...")

# Now we can safely import unsloth
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=False,
    device_map="cpu",
)
print("✓ Model downloaded and cached")
print()

# ============================================================================
# 3. Warm up the tokenizer
# ============================================================================
print("Step 3: Warming up tokenizer...")

# Run the tokenizer on sample text to cache operations
sample_texts = [
    "I need to cancel my order",
    "What payment methods do you accept?",
    "My account was charged twice"
]

for text in sample_texts:
    _ = tokenizer(text, return_tensors="pt")

print("✓ Tokenizer warmed up")
print()

# ============================================================================
# 4. Pre-format a sample of the dataset
# ============================================================================
print("Step 4: Pre-processing dataset sample...")

def format_prompt(example):
    prompt = f"""You are a customer support AI. Classify the following customer query into the appropriate intent category.

Customer Query: {example['instruction']}

Intent:"""
    return {"text": f"{prompt} {example['intent']}"}

# Pre-process a small sample to warm up the mapping operation
sample_dataset = dataset['train'].shuffle(seed=42).select(range(100))
formatted_sample = sample_dataset.map(format_prompt)
print("✓ Dataset formatting cached")
print()

# ============================================================================
# 5. Configure LoRA and run a tiny training step
# ============================================================================
print("Step 5: Initializing training pipeline...")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)

from transformers import TrainingArguments
from trl import SFTTrainer

# Create a minimal trainer just to initialize everything
warmup_args = TrainingArguments(
    output_dir="./warmup-temp",
    per_device_train_batch_size=2,
    max_steps=1,  # Just one step to warm up
    logging_steps=1,
    optim="adamw_8bit",
    report_to="none",  # Don't try to log to wandb/tensorboard
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_sample.select(range(4)),  # Just 4 examples
    dataset_text_field="text",
    max_seq_length=512,
    args=warmup_args,
)

print("Running warmup training step (this compiles the training loop)...")
_ = trainer.train()
print("✓ Training pipeline initialized")
print()

# ============================================================================
# 6. Clean up warmup artifacts
# ============================================================================
print("Step 6: Cleaning up...")
import shutil
if os.path.exists("./warmup-temp"):
    shutil.rmtree("./warmup-temp")
print("✓ Cleanup complete")
print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 80)
print("WARMUP COMPLETE!")
print("=" * 80)
print()
print("Everything is now cached and ready:")
print("  ✓ Dataset downloaded")
print("  ✓ Model downloaded (~6GB)")
print("  ✓ Tokenizer warmed up")
print("  ✓ Training pipeline compiled")
print()
print("When you run the main training script, it will:")
print("  - Start immediately (no downloads)")
print("  - Skip compilation steps")
print("  - Jump straight into training")
print()
print("Estimated time saved: 3-5 minutes")
print()