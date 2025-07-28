#!/usr/bin/env python
"""
DRPO training script with vanilla DDP + PEFT LoRA
Usage: accelerate launch --config_file accelerate_config_ddp.yaml train_drpo_ddp_lora.py
"""

import os
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import LoraConfig, TaskType, get_peft_model
import wandb

# Add parent directories to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from trainer.drpo_trainer_new import DRPOTrainer
from trainer.drpo_config_new import DRPOConfig

# Configuration
MODEL_NAME = "Kyleyee/Qwen2.5-1.5B-sft-3e"  # Change to your model
REWARD_MODEL_NAME = "Kyleyee/Qwen2.5-1.5B-reward-hh-retrain"  # Change to your reward model
DATASET_NAME = "Eehan/train_data_helpful"  # Change to your dataset
OUTPUT_DIR = "./drpo-ddp-lora-qwen2.5-1.5b"

# Training configuration
training_config = DRPOConfig(
    output_dir=OUTPUT_DIR,
    
    # Basic training parameters
    per_device_train_batch_size=4,  # Adjust based on your GPU memory
    gradient_accumulation_steps=4,   # Effective batch size = 4 * 4 * num_gpus
    per_device_eval_batch_size=4,
    learning_rate=5e-7,
    num_train_epochs=1,
    warmup_steps=100,
    
    # DRPO specific parameters
    num_monte_carlo_samples=2,
    beta=0.1,
    kl_type="k3",
    is_clip_min=0.1,
    is_clip_max=10.0,
    
    # Generation parameters
    max_new_tokens=256,
    temperature=0.8,
    max_length=1024,
    max_prompt_length=512,
    max_completion_length=512,
    
    # Optimization settings
    bf16=True,  # Use bf16 for better performance
    # tf32=True,  # Enable TF32 on Ampere GPUs
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    optim="adamw_torch_fused",  # Use fused optimizer for speed
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    weight_decay=0.01,
    
    # DDP settings (no FSDP or DeepSpeed)
    ddp_find_unused_parameters=False,
    ddp_bucket_cap_mb=25,
    
    # Logging and saving
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_generated/win_rate_vs_rejected",
    greater_is_better=True,
    
    # Dataset processing
    dataset_num_proc=4,  # Use multiple CPU cores
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    
    # Memory optimization
    torch_empty_cache_steps=50,
    
    # Evaluation
    eval_with_generation=True,
    eval_mc_samples=1,
    
    # Reporting
    report_to="wandb",  # Change to "none" if you don't want to use wandb
    run_name="drpo-ddp-lora",
)

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,  # LoRA rank
    lora_alpha=128,  # LoRA alpha
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    # Don't include embedding/lm_head in modules_to_save for DDP
    modules_to_save=None,
)


def main():
    """Main training function."""
    
    # Initialize wandb if using
    if training_config.report_to == "wandb":
        wandb.init(project="drpo-training", name=training_config.run_name)
    
    print("Loading and preparing dataset...")
    # Load dataset
    train_dataset = load_dataset(DATASET_NAME, split="train[:10000]")  # Using subset for demo
    eval_dataset = load_dataset(DATASET_NAME, split="test[:1000]")

    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load models
    print("Loading models...")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        use_cache=False,  # Disable KV cache for training
        trust_remote_code=True,
        attn_implementation="sdpa",  # Use scaled dot product attention
    )
    
    # # Apply LoRA
    # print("Applying LoRA configuration...")
    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()
    
    # Enable gradient checkpointing
    if training_config.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    
    # Load reward model
    print("Loading reward model...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        trust_remote_code=True,
    )
    
    # Initialize trainer
    print("Initializing DRPO trainer...")
    trainer = DRPOTrainer(
        model=model,
        ref_model=None,  # Will use LoRA disabled state as reference
        reward_model=reward_model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,  # Pass LoRA config here
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    print("Saving model...")
    trainer.save_model()
    
    # Save tokenizer
    tokenizer.save_pretrained(training_config.output_dir)
    
    # Push to hub if needed
    if training_config.push_to_hub:
        print("Pushing to hub...")
        trainer.push_to_hub()
    
    print("Training complete!")


if __name__ == "__main__":
    main()