"""
Customer Support Intent Classification with Fine-Tuning
========================================================
This script demonstrates how to fine-tune a large language model (LLM) to classify
customer support queries into specific intent categories using LoRA (Low-Rank Adaptation).

What happens in this script:
1. Load a real customer support dataset
2. Load a pre-trained language model (Llama 3.2)
3. Configure the model for efficient fine-tuning using LoRA
4. Test the model BEFORE fine-tuning (baseline)
5. Fine-tune the model on customer support examples
6. Test the model AFTER fine-tuning (see the improvement!)
7. Save the trained adapter for future use
"""

import os
import torch

# ============================================================================
# STEP 0: Configure for CPU-only environment (Codespaces has no GPU)
# ============================================================================
print("=" * 80)
print("CONFIGURING ENVIRONMENT FOR CPU")
print("=" * 80)

# Tell PyTorch to use CPU instead of looking for CUDA/GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide any GPU devices
torch.cuda.is_available = lambda: False  # Override CUDA availability check

print("✓ Configured for CPU-only execution")
print()

# ============================================================================
# STEP 1: Load the Customer Support Dataset
# ============================================================================
print("=" * 80)
print("STEP 1: LOADING DATASET")
print("=" * 80)

from datasets import load_dataset

# Load the Bitext Customer Support dataset from HuggingFace
# This is a professionally-curated dataset with ~27,000 customer support examples
# Each example has:
#   - instruction: the customer's query
#   - intent: what the customer wants (e.g., "cancel_order", "track_order")
#   - category: broader grouping (e.g., "ORDER", "ACCOUNT")
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

print(f"Total examples in dataset: {len(dataset['train']):,}")
print(f"Dataset splits available: {list(dataset.keys())}")
print()

# Display one example to understand the data structure
print("Example from dataset:")
first_example = dataset['train'][0]
for key, value in first_example.items():
    print(f"  {key}: {value}")
print()

# Analyze the variety of intents in the dataset
all_intents = dataset['train']['intent']
unique_intents = sorted(set(all_intents))

print(f"Dataset contains {len(unique_intents)} different intent types")
print("Most common intents (showing first 10):")
for intent in unique_intents[:10]:
    count = all_intents.count(intent)
    print(f"  - {intent}: {count:,} examples")
print()

# For this demo, we'll use a subset to keep training time reasonable
# In production, you'd train on the full 27,000 examples
sample_size = 500
train_dataset = dataset['train'].shuffle(seed=42).select(range(sample_size))

print(f"Selected {sample_size} examples for this training demo")
print(f"These examples cover {len(set(train_dataset['intent']))} different intents")
print()

# Show a few example customer queries so you understand what we're working with
print("Sample customer queries from training data:")
for i in range(3):
    item = train_dataset[i]
    print(f"\n  Query: \"{item['instruction']}\"")
    print(f"  Intent: {item['intent']}")
    print(f"  Category: {item['category']}")
print()

# ============================================================================
# STEP 2: Load the Pre-trained Language Model
# ============================================================================
print("=" * 80)
print("STEP 2: LOADING LANGUAGE MODEL")
print("=" * 80)

from unsloth import FastLanguageModel

# Load Llama 3.2 (3 billion parameter model)
# This is a "foundation model" - it knows general language but hasn't been
# specifically trained on customer support intent classification yet
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",  # Pre-trained instruction-following model
    max_seq_length=512,          # Maximum length of text we'll process (in tokens)
    dtype=None,                  # Let unsloth choose the best data type for our hardware
    load_in_4bit=False,          # Can't use 4-bit quantization on CPU (GPU-only feature)
    device_map="cpu",            # CRITICAL: Tell model to use CPU, not GPU
)

print("✓ Model loaded successfully")
print(f"  Model: Llama 3.2 (3B parameters)")
print(f"  Device: CPU")
print()

# ============================================================================
# STEP 3: Configure LoRA for Efficient Fine-Tuning
# ============================================================================
print("=" * 80)
print("STEP 3: CONFIGURING LORA (LOW-RANK ADAPTATION)")
print("=" * 80)

# LoRA is a technique that lets us fine-tune large models efficiently
# Instead of updating all 3 billion parameters, we add small "adapter" layers
# and only train those (~1% of parameters). Much faster and uses less memory!

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank: controls the size of adapter layers (higher = more capacity)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Which model layers to adapt
    lora_alpha=16,     # Scaling factor for LoRA updates
    lora_dropout=0.05, # Dropout rate to prevent overfitting
    bias="none",       # Don't train bias parameters
)

# Calculate what percentage of the model we're actually training
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
trainable_percentage = 100 * trainable_params / total_params

print(f"Training {trainable_params:,} parameters")
print(f"That's only {trainable_percentage:.2f}% of the full model!")
print(f"This makes training much faster and more memory-efficient")
print()

# ============================================================================
# STEP 4: Format the Dataset for Training
# ============================================================================
print("=" * 80)
print("STEP 4: FORMATTING DATASET")
print("=" * 80)

def format_prompt(example):
    """
    Convert each dataset example into a training prompt.
    
    The model will learn to complete prompts that look like this:
    "You are a customer support AI. Classify this query: [query]
    Intent: [correct answer]"
    
    During inference, we'll give it everything except the answer,
    and it will predict the intent.
    """
    prompt = f"""You are a customer support AI. Classify the following customer query into the appropriate intent category.

Customer Query: {example['instruction']}

Intent:"""
    
    # Return the full prompt with the correct answer appended
    return {"text": f"{prompt} {example['intent']}"}

# Apply the formatting function to every example in our training set
formatted_dataset = train_dataset.map(format_prompt)
print("✓ Dataset formatted for training")
print("  Each example now has a 'text' field with the full training prompt")
print()

# ============================================================================
# STEP 5: Test the Model BEFORE Fine-Tuning (Baseline)
# ============================================================================
print("=" * 80)
print("STEP 5: TESTING MODEL BEFORE FINE-TUNING")
print("=" * 80)

# These are new queries the model has never seen
test_queries = [
    "I need to cancel my last order immediately",
    "What payment methods do you accept?",
    "My account was charged twice for the same purchase"
]

# Put model in inference mode (not training mode)
FastLanguageModel.for_inference(model)

print("Testing model on unseen queries (BEFORE training):")
print("Note: The model hasn't been trained on customer support yet,")
print("so its predictions might be vague or incorrect.\n")

for query in test_queries:
    # Create the same prompt format we'll use after training
    prompt = f"""You are a customer support AI. Classify the following customer query into the appropriate intent category.

Customer Query: {query}

Intent:"""
    
    # Convert text to tokens (numbers the model understands)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate the model's prediction
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,   # Limit response length
        temperature=0.1,     # Low temperature = more deterministic/focused
    )
    
    # Convert tokens back to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the intent prediction from the full response
    intent = response.split("Intent:")[-1].strip().split("\n")[0]
    print(f"  Query: {query}")
    print(f"  Predicted: {intent}")
    print()

# ============================================================================
# STEP 6: Fine-Tune the Model
# ============================================================================
print("=" * 80)
print("STEP 6: FINE-TUNING THE MODEL")
print("=" * 80)

from transformers import TrainingArguments
from trl import SFTTrainer

# Configure the training process
training_args = TrainingArguments(
    output_dir="./customer-support-output",  # Where to save checkpoints
    per_device_train_batch_size=2,  # Process 2 examples at a time (small for CPU)
    gradient_accumulation_steps=4,   # Accumulate gradients over 4 steps (effective batch size = 8)
    warmup_steps=5,                  # Gradually increase learning rate for first 5 steps
    max_steps=50,                    # Total training steps (more = better fit, but longer)
    learning_rate=2e-4,              # How fast the model learns (2e-4 is good for LoRA)
    logging_steps=10,                # Print progress every 10 steps
    optim="adamw_8bit",              # Use memory-efficient optimizer
    fp16=False,                      # Don't use half-precision (CPU doesn't support it well)
)

# Create the trainer object - this handles the entire training loop
trainer = SFTTrainer(
    model=model,                     # The model we're training
    tokenizer=tokenizer,             # Converts text to/from tokens
    train_dataset=formatted_dataset, # Our formatted training data
    dataset_text_field="text",       # Which field contains the training text
    max_seq_length=512,              # Maximum sequence length (matches model config)
    args=training_args,              # Training configuration from above
)

print("Starting fine-tuning process...")
print("⏱️  This will take 5-10 minutes on CPU")
print()
print("💡 WHILE WAITING, TRY THIS:")
print("   - Look at the training loss - it should decrease over time")
print("   - Think about: Why does lower loss = better model?")
print("   - Review the code above - what questions do you have?")
print()

# Run the actual training!
trainer_stats = trainer.train()

print()
print(f"✓ Training complete!")
print(f"  Final loss: {trainer_stats.training_loss:.4f}")
print(f"  (Lower loss = model is better at predicting the correct intents)")
print()

# ============================================================================
# STEP 7: Test the Model AFTER Fine-Tuning
# ============================================================================
print("=" * 80)
print("STEP 7: TESTING MODEL AFTER FINE-TUNING")
print("=" * 80)

# Put model back in inference mode
FastLanguageModel.for_inference(model)

print("Testing same queries with the fine-tuned model:")
print("The model should now give more specific, accurate intent predictions!\n")

for query in test_queries:
    prompt = f"""You are a customer support AI. Classify the following customer query into the appropriate intent category.

Customer Query: {query}

Intent:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=20, temperature=0.1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    intent = response.split("Intent:")[-1].strip().split("\n")[0]
    print(f"  Query: {query}")
    print(f"  Predicted: {intent}")
    print()

print("Compare these predictions to the ones before training!")
print("The model should now be much better at classifying customer intents.")
print()

# ============================================================================
# STEP 8: Save the Fine-Tuned Adapter
# ============================================================================
print("=" * 80)
print("STEP 8: SAVING THE TRAINED ADAPTER")
print("=" * 80)

adapter_path = "./lora-customer-support-classifier"

# Save only the LoRA adapter weights (not the full 3B parameter model!)
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)

print(f"✓ Adapter saved to: {adapter_path}")

# Check how small the adapter is compared to the full model
import subprocess
try:
    size = subprocess.check_output(['du', '-sh', adapter_path]).decode('utf-8')
    print(f"  Adapter size: {size.split()[0]}")
    print(f"  (Compare this to ~6GB for the full model!)")
except:
    print(f"  Adapter saved successfully")

print()
print("This adapter can be:")
print("  - Shared with others (it's tiny!)")
print("  - Loaded onto the base model anytime")
print("  - Combined with other adapters")
print()

# ============================================================================
# STEP 9: Demonstrate Loading the Adapter Later
# ============================================================================
print("=" * 80)
print("STEP 9: HOW TO USE THE ADAPTER LATER")
print("=" * 80)

print("To use this fine-tuned model in the future, run this code:")
print()
print("```python")
print("from unsloth import FastLanguageModel")
print()
print("# Load the base model")
print("model, tokenizer = FastLanguageModel.from_pretrained(")
print("    model_name='unsloth/Llama-3.2-3B-Instruct',")
print("    max_seq_length=512,")
print("    device_map='cpu',")
print(")")
print()
print("# Attach your custom adapter")
print("model.load_adapter('./lora-customer-support-classifier')")
print()
print("# Put in inference mode and use!")
print("FastLanguageModel.for_inference(model)")
print("```")
print()

# ============================================================================
# BONUS: Dataset Analysis
# ============================================================================
print("=" * 80)
print("BONUS: DATASET ANALYSIS")
print("=" * 80)

import pandas as pd

# Convert dataset to pandas DataFrame for easy analysis
df = pd.DataFrame(dataset['train'])

print("Intent distribution (top 10 most common):")
print(df['intent'].value_counts().head(10))
print()

print("Intents by category:")
category_intents = df.groupby('category')['intent'].unique()
for category, intents in category_intents.items():
    print(f"\n{category}:")
    for intent in sorted(intents)[:5]:  # Show first 5 intents per category
        print(f"  - {intent}")

print()
print("=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print()
print("Key Takeaways:")
print("1. LoRA lets us fine-tune huge models with minimal compute")
print("2. Fine-tuning dramatically improves task-specific performance")
print("3. The adapter is tiny (~100MB) compared to the full model (~6GB)")
print("4. This same technique works for many tasks: classification,")
print("   extraction, generation, summarization, and more!")