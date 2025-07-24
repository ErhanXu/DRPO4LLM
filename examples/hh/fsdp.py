# train_fsdp.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
)
from trainer.drpo_trainer_fsdp import DRPOTrainerFSDP
from trainer.drpo_config_fsdp import DRPOConfigFSDP
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import wandb

# Model configuration
# model_name_or_path = "Qwen/Qwen2.5-7B"
# reward_model_path = "Kyleyee/Qwen2.5-7B-reward-hh"
model_name_or_path = "Qwen/Qwen2.5-1.5B"  # Base model path
reward_model_name_or_path = "Kyleyee/Qwen2.5-1.5B-reward-hh-retrain"  # Reward model path, can be same as base model

# IMPORTANT: Detect model architecture for FSDP layer wrapping
config = AutoConfig.from_pretrained(model_name_or_path)
model_type = config.model_type

# Map model types to their decoder layer classes
LAYER_CLS_MAPPING = {
    "qwen2": "Qwen2DecoderLayer",
    "llama": "LlamaDecoderLayer",
    "mistral": "MistralDecoderLayer",
    "gemma": "GemmaDecoderLayer",
}

layer_cls = LAYER_CLS_MAPPING.get(model_type, "Qwen2DecoderLayer")

# FSDP-optimized configuration
training_config = DRPOConfigFSDP(
    output_dir="./drpo-fsdp-qwen2.5-7b",
    
    # FSDP specific
    fsdp="shard_grad_op auto_wrap",
    fsdp_transformer_layer_cls_to_wrap=layer_cls,
    
    # Training parameters
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,
    num_train_epochs=1,
    warmup_steps=100,
    
    # DRPO parameters
    num_monte_carlo_samples=2,
    beta=0.1,
    kl_type="k3",
    is_clip_min=0.1,
    is_clip_max=10.0,
    
    # Generation
    max_new_tokens=256,
    temperature=0.8,
    max_length=1024,
    max_prompt_length=512,
    max_completion_length=512,
    
    # Optimizations
    bf16=True,
    fp16=False,  # Use bf16 for FSDP
    tf32=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Use non-reentrant for FSDP
    optim="adamw_torch_fused",  # Fused optimizer for speed
    
    # Memory management
    torch_empty_cache_steps=50,
    
    # Logging
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    eval_strategy="steps",
    eval_steps=100,
    
    # Dataset
    dataset_num_proc=1,
    # dataloader_num_workers=4,
    
    # Evaluation
    eval_with_generation=True,
    eval_mc_samples=1,
    metric_for_best_model="eval_generated/win_rate_vs_rejected",
    
    report_to="none",
    run_name="drpo-fsdp-qwen2.5-7b",
)


# At the top of your script
if int(os.environ.get("WORLD_SIZE", 1)) == 1:
    # Single GPU - use simpler config
    training_config.fsdp = None  # Disable FSDP
    training_config.ddp_find_unused_parameters = False
else:
    # Multi-GPU - use FSDP
    training_config.fsdp = "shard_grad_op auto_wrap"
    training_config.fsdp_transformer_layer_cls_to_wrap = layer_cls

# PEFT configuration optimized for FSDP
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=128,
    lora_alpha=256,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    modules_to_save=None,  # Don't save embeddings with FSDP
)

print("Loading and preparing dataset...")
# Dataset preparation (same as before)
dataset = load_dataset("Anthropic/hh-rlhf", split="train")
eval_dataset = load_dataset("Anthropic/hh-rlhf", split="test[:1000]")

def prepare_dataset(example):
    chosen_text = example["chosen"]
    rejected_text = example["rejected"]
    
    last_human_idx = chosen_text.rfind("\n\nHuman: ")
    last_assistant_idx = chosen_text.rfind("\n\nAssistant: ")
    
    if last_human_idx == -1 or last_assistant_idx == -1:
        return None
    
    prompt = chosen_text[:last_assistant_idx + len("\n\nAssistant: ")]
    chosen_response = chosen_text[last_assistant_idx + len("\n\nAssistant: "):].strip()
    
    rejected_last_idx = rejected_text.rfind("\n\nAssistant: ")
    rejected_response = rejected_text[rejected_last_idx + len("\n\nAssistant: "):].strip()
    
    return {
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
    }

train_dataset = dataset.map(
    prepare_dataset,
    remove_columns=dataset.column_names,
    num_proc=training_config.dataset_num_proc
).filter(lambda x: x is not None)

eval_dataset = eval_dataset.map(
    prepare_dataset,
    remove_columns=eval_dataset.column_names,
    num_proc=training_config.dataset_num_proc
).filter(lambda x: x is not None)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")

# Load tokenizer first
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# FSDP-aware model loading
print("Loading models...")

# Load policy model
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    use_cache=False,  # Disable KV cache for training
    attn_implementation="sdpa",
)

# Prepare model for PEFT + FSDP
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=training_config.gradient_checkpointing
)

# Apply PEFT
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Load reward model (keep outside FSDP)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_name_or_path,
    torch_dtype=torch.bfloat16,
    use_cache=False,
)

# Initialize trainer
print("Initializing DRPO trainer with FSDP...")
trainer = DRPOTrainerFSDP(
    model=model,
    ref_model=None,  # Will use LoRA disabled state
    reward_model=reward_model,
    args=training_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    # Don't pass peft_config since we already applied it
)

# Train
if __name__ == "__main__":
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_config.output_dir)
    print("Training complete!")