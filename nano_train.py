import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# --- CONFIGURATION ---
# We use 1B for speed/safety in this demo. Change to "3B" if you have 16GB+ RAM.
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct" 
output_dir = "./nano_llama_adapter"

# --- 1. PREPARE "BUSINESS CONTEXT" DATASET (5 Examples) ---
data = [
    {"text": "Input: Hey boss, late today. \nResponse: <BUSINESS> I am writing to inform you of a delayed arrival. </BUSINESS>"},
    {"text": "Input: The server is dead. \nResponse: <BUSINESS> We are currently experiencing a critical infrastructure outage. </BUSINESS>"},
    {"text": "Input: I need a raise. \nResponse: <BUSINESS> I would like to schedule a performance review to discuss compensation. </BUSINESS>"},
    {"text": "Input: This deal sucks. \nResponse: <BUSINESS> The current terms of the agreement are suboptimal for our strategic goals. </BUSINESS>"},
    {"text": "Input: Dave messed up. \nResponse: <BUSINESS> There has been a procedural error in the engineering department. </BUSINESS>"},
]
dataset = Dataset.from_list(data)

# --- 2. LOAD MODEL (CPU OPTIMIZED) ---
print(f"Loading {MODEL_ID}... (This may take 2-3 mins)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token # Fix for Llama 3

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cpu")

# --- 3. APPLY LoRA ADAPTER (The Fine-Tuning Magic) ---
# r=4 is very low rank (fastest), targeting query/value layers only
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, peft_config)
model.print_trainable_parameters() # Proof we are training!

# --- 4. TRAINER SETUP (Nano-Scale) ---
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1, # Keep RAM usage low
    max_steps=5,                   # ONLY 5 STEPS for demo speed
    learning_rate=2e-4,
    use_cpu=True,                  # Force CPU training
    logging_steps=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# --- 5. TRAIN & TEST ---
print("\n--- STARTING NANO-FINE-TUNE (5 STEPS) ---")
trainer.train()

print("\n--- TESTING THE CHANGE ---")
# Test Prompt: "Input: The project is broken."
inputs = tokenizer("Input: The project is broken. \nResponse:", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))