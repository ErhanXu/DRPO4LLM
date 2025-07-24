# train_fsdp.py (Updated)
import os
import sys
# Ensure the project root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
)
# CHANGE 1: Import the unified trainer and its config
from trainer.drpo_trainer_new import DRPOTrainer
from trainer.drpo_config_fsdp import DRPOConfigFSDP
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

# --- Model & Path Configuration ---
model_name_or_path = "Qwen/Qwen2.5-1.5B"  # Base model for policy
reward_model_name_or_path = "Kyleyee/Qwen2.5-1.5B-reward-hh-retrain"  # Reward/Preference model

# --- FSDP Configuration ---
# Automatically detect model architecture for optimal FSDP layer wrapping
config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
model_type = config.model_type

# Map model types to their corresponding decoder layer class names for auto-wrapping
LAYER_CLS_MAPPING = {
    "qwen2": "Qwen2DecoderLayer",
    "llama": "LlamaDecoderLayer",
    "mistral": "MistralDecoderLayer",
    "gemma": "GemmaDecoderLayer",
}
# Fallback to a default if the model type is not found
layer_cls_to_wrap = LAYER_CLS_MAPPING.get(model_type, "Qwen2DecoderLayer")
print(f"Detected model type '{model_type}'. Using FSDP wrap policy for '{layer_cls_to_wrap}'.")

# --- Training Configuration ---
training_config = DRPOConfigFSDP(
    output_dir=f"./drpo-fsdp-{model_name_or_path.replace('/', '_')}",
    
    # FSDP specific settings
    fsdp="shard_grad_op auto_wrap",
    fsdp_transformer_layer_cls_to_wrap=layer_cls_to_wrap,
    
    # Core training parameters
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,
    num_train_epochs=1,
    warmup_steps=100,
    
    # DRPO algorithm parameters
    num_monte_carlo_samples=2,
    beta=0.1,
    kl_type="k3",
    is_clip_min=0.1,
    is_clip_max=10.0,
    
    # Generation parameters
    max_new_tokens=512,
    temperature=0.8,
    max_length=1024,
    max_prompt_length=512,
    
    # Performance optimizations
    bf16=True,
    fp16=False,  # bf16 is generally preferred for A100/H100 GPUs
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}, # Required for FSDP
    optim="adamw_torch_fused",
    
    # Logging and saving
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    eval_strategy="steps",
    eval_steps=100,
    
    # Evaluation
    eval_with_generation=True,
    eval_mc_samples=1,
    metric_for_best_model="eval/generated/win_rate_vs_rejected",
    
    # W&B reporting
    report_to="none", # Set to "wandb" to enable logging
    run_name=f"drpo-fsdp-{model_name_or_path.replace('/', '_')}",
)

# --- PEFT Configuration (LoRA) ---
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=128,
    lora_alpha=256,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    modules_to_save=None, # Important for FSDP: avoids saving non-trainable weights
)

# --- Dataset Preparation ---
print("Loading and preparing dataset...")
train_dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1%]") # Using a smaller slice for quick runs
eval_dataset = load_dataset("Anthropic/hh-rlhf", split="test[:1%]")

def prepare_dataset(example):
    """Extracts prompt, chosen, and rejected responses from the hh-rlhf dataset."""
    try:
        last_assistant_idx = example["chosen"].rfind("\n\nAssistant: ")
        prompt = example["chosen"][:last_assistant_idx + len("\n\nAssistant: ")]
        chosen_response = example["chosen"][last_assistant_idx + len("\n\nAssistant: "):].strip()
        
        rejected_last_idx = example["rejected"].rfind("\n\nAssistant: ")
        rejected_response = example["rejected"][rejected_last_idx + len("\n\nAssistant: "):].strip()
        
        return {"prompt": prompt, "chosen": chosen_response, "rejected": rejected_response}
    except (ValueError, AttributeError):
        return None

train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names).filter(lambda x: x is not None)
eval_dataset = eval_dataset.map(prepare_dataset, remove_columns=eval_dataset.column_names).filter(lambda x: x is not None)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")

# --- Tokenizer and Model Loading ---
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading policy and reward models...")
# Load policy model
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    use_cache=False,  # Must be False for training
    attn_implementation="sdpa", # Use Flash Attention 2 if available
    trust_remote_code=True,
)

# Prepare model for PEFT + FSDP
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_config.gradient_checkpointing)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Load reward model (will be placed on a single device, not wrapped by FSDP)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_name_or_path,
    torch_dtype=torch.bfloat16,
    use_cache=False,
    trust_remote_code=True,
)

# --- Initialize Trainer ---
print("Initializing unified DRPO trainer...")
# CHANGE 2: Instantiate the unified DRPOTrainer.
# The previous drpo_trainer_fsdp.py is no longer needed.
# The new trainer is FSDP-aware and will apply FSDP logic based on the config.
trainer = DRPOTrainer(
    model=model,
    ref_model=None,  # Correct for PEFT: trainer will use adapter-disabled model as reference
    reward_model=reward_model,
    args=training_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

# --- Start Training ---
if __name__ == "__main__":
    print("Starting training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model() # Saves the LoRA adapter
    tokenizer.save_pretrained(training_config.output_dir)
    print("Training complete!")