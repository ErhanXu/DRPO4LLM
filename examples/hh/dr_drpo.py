#!/usr/bin/env python
"""
Dr.DRPO training script with vanilla DDP + PEFT LoRA

Usage:
1. Single GPU:
   python train_drdrpo_ddp_lora.py

2. Multi-GPU with DDP:
   accelerate launch --config_file accelerate_config_ddp.yaml train_drdrpo_ddp_lora.py

3. Custom number of GPUs:
   accelerate launch --num_processes 4 train_drdrpo_ddp_lora.py
"""

import os
import sys
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

# Add parent directories to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from trainer.drdrpo_trainer import DrDRPOTrainer, DrDRPOConfig

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-1.5B"  # Change to your model
DATASET_NAME = "Anthropic/hh-rlhf"  # Change to your dataset
OUTPUT_DIR = "./drdrpo-output"

# Training configuration
training_config = DrDRPOConfig(
    output_dir=OUTPUT_DIR,
    
    # Basic training parameters
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-7,
    num_train_epochs=1,
    
    # DRPO parameters
    num_monte_carlo_samples=2,
    beta=0.1,
    kl_type="k3",
    is_clip_min=0.1,
    is_clip_max=10.0,
    
    # Dr.DRPO specific parameters
    beta_prime=1.0,
    use_dpo_preference=True,
    weight_clip_range=(0.1, 10.0),
    
    # Generation parameters
    max_new_tokens=256,
    temperature=0.85,
    max_length=1024,
    max_prompt_length=512,
    max_completion_length=512,
    
    # Optimization settings
    bf16=True,
    gradient_checkpointing=True,
    
    # DDP settings
    ddp_find_unused_parameters=False,
    
    # Logging and saving
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=100,
    
    # Evaluation
    eval_with_generation=True,
    eval_mc_samples=1,
    
    # Reporting
    report_to="none",
    run_name="drdrpo-ddp-run",
    
    # Use preference model
    use_preference_model=False,  # Set to True if you have a preference model
)

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)


def preprocess_dataset(example):
    """Preprocess dataset to expected format if needed."""
    # Adjust based on your dataset format
    return {
        "prompt": example.get("prompt", example.get("query", "")),
        "chosen": example.get("chosen", example.get("response_chosen", "")),
        "rejected": example.get("rejected", example.get("response_rejected", "")),
    }


def main():
    """Main training function."""
    
    print("Loading dataset...")
    train_dataset = load_dataset(DATASET_NAME, split="train[:10000]")
    eval_dataset = load_dataset(DATASET_NAME, split="test[:1000]")
    
    # Preprocess if needed
    if "prompt" not in train_dataset.column_names:
        train_dataset = train_dataset.map(preprocess_dataset)
        eval_dataset = eval_dataset.map(preprocess_dataset)
    
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load models
    print("Loading models...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        trust_remote_code=True,
    )
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        trust_remote_code=True,
    )
    
    # Apply LoRA
    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing
    if training_config.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    
    # Initialize trainer
    print("Initializing Dr.DRPO trainer...")
    trainer = DrDRPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(training_config.output_dir)
    
    # Save training config for reproducibility
    training_config.save_pretrained(training_config.output_dir)
    
    # Save weight analysis
    # trainer.save_weight_analysis(os.path.join(OUTPUT_DIR, "weight_analysis.json"))
    
    print("Training complete!")


if __name__ == "__main__":
    main()