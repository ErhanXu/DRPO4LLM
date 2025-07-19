import os
import sys
# Add the parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from trainer.drpo_trainer_new import DRPOTrainer
from trainer.drpo_config_new import DRPOConfig
from peft import LoraConfig, TaskType
import wandb

# Model paths
model_name_or_path = "Kyleyee/Qwen2.5-7b-sft-hh"
reward_model_path = "Kyleyee/Qwen2.5-7B-reward-hh"

# Configure for multi-GPU training with DeepSpeed ZeRO-3
config = DRPOConfig(
    output_dir="./drpo-qwen2.5-7b-hh",
    
    # Model configuration
    use_preference_model=False,  # Using reward model instead
    
    # Optimized for H100s with DeepSpeed ZeRO-3
    per_device_train_batch_size=2,      # Conservative due to memory requirements
    gradient_accumulation_steps=4,       # Effective batch size = 2 * 4 * num_gpus
    per_device_eval_batch_size=4,
    
    # Training hyperparameters
    learning_rate=5e-7,
    num_train_epochs=1,
    warmup_steps=100,
    max_steps=-1,
    
    # DRPO specific parameters
    num_monte_carlo_samples=2,          # 2 samples for memory efficiency
    beta=0.1,
    kl_type="k3",
    is_clip_min=0.1,
    is_clip_max=10.0,
    
    # Generation parameters
    max_new_tokens=256,                 # Increased for better responses
    temperature=0.8,
    max_length=1024,
    max_prompt_length=512,
    max_completion_length=512,
    
    # Optimization settings
    bf16=True,
    tf32=True,                          # Enable TF32 for H100s
    gradient_checkpointing=True,
    optim="adamw_torch_fused",         # Fused optimizer for speed
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    weight_decay=0.01,
    
    # DeepSpeed integration
    # deepspeed="deepspeed_config.json",  # Will create this
    
    # Logging and saving
    logging_steps=50,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    
    # Dataset processing
    dataset_num_proc=1,                # Utilize CPU cores
    # dataloader_num_workers=4,
    
    # Memory optimization
    torch_empty_cache_steps=50,
    
    # Training optimizations
    fp16_full_eval=False,              # Use bf16 for eval too
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    
    # Evaluation
    eval_with_generation=True,
    eval_mc_samples=1,
    metric_for_best_model="eval_generated/win_rate_vs_chosen",
    greater_is_better=True,
    
    # Report to wandb
    report_to="wandb",
    run_name="drpo-qwen2.5-7b-hh",
)

# LoRA configuration for memory efficiency
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,                              # Higher rank for better performance
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    # modules_to_save=["embed_tokens", "lm_head"],
)

# IMPORTANT: Process dataset BEFORE loading models to avoid CUDA issues
print("Loading and processing dataset...")

# Load dataset
dataset = load_dataset("Anthropic/hh-rlhf", split="train")  # Using subset for testing
eval_dataset = load_dataset("Anthropic/hh-rlhf", split="test[:2560]")

# Prepare dataset function
def prepare_dataset(example):
    chosen_text = example["chosen"]
    rejected_text = example["rejected"]
    
    # Extract prompt and responses
    last_human_idx = chosen_text.rfind("\n\nHuman: ")
    last_assistant_idx = chosen_text.rfind("\n\nAssistant: ")
    
    if last_human_idx == -1 or last_assistant_idx == -1:
        return None
    
    # Get prompt including "Assistant: "
    prompt = chosen_text[:last_assistant_idx + len("\n\nAssistant: ")]
    
    # Extract responses
    chosen_response = chosen_text[last_assistant_idx + len("\n\nAssistant: "):].strip()
    
    rejected_last_idx = rejected_text.rfind("\n\nAssistant: ")
    rejected_response = rejected_text[rejected_last_idx + len("\n\nAssistant: "):].strip()
    
    return {
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
    }

# Process datasets with num_proc=1 to avoid CUDA forking issues
train_dataset = dataset.map(
    prepare_dataset, 
    remove_columns=dataset.column_names,
    desc="Processing train dataset"
).filter(lambda x: x is not None)

eval_dataset = eval_dataset.map(
    prepare_dataset,
    remove_columns=eval_dataset.column_names,
    desc="Processing eval dataset"
).filter(lambda x: x is not None)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")

# NOW load models after dataset processing
print("Loading models...")
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    use_cache=False,  # Disable KV cache for training
)

# Load reward model
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_path,
    torch_dtype=torch.bfloat16,
    use_cache=False,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Initialize trainer
trainer = DRPOTrainer(
    model=model,
    ref_model=None,  # Will use LoRA disabled state
    reward_model=reward_model,
    args=config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

# Launch training
if __name__ == "__main__":
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    print("Training complete!")
