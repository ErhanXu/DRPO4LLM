#!/usr/bin/env python
"""
Dr.DRPO training script for Pythia-1B on TLDR dataset

Usage:
1. Single GPU:
   python train_drdrpo_pythia_tldr.py

2. Multi-GPU with DDP:
   accelerate launch --config_file accelerate_config_ddp.yaml train_drdrpo_pythia_tldr.py

3. With wandb logging:
   python train_drdrpo_pythia_tldr.py --use_wandb
"""

import os
import sys
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model
import argparse

# Add parent directories to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from trainer.dr_drpo_trainer import DrDRPOTrainer, DrDRPOConfig

import wandb
import swanlab
os.environ["WANDB_MODE"] = "offline"
swanlab.sync_wandb()

# Configuration
MODEL_NAME = "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr"
REWARD_MODEL_NAME = "cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr"
DATASET_NAME = "Eehan/train_data_tldr_flipped-10"
OUTPUT_DIR = "./drdrpo-pythia-1b-tldr"

# Training configuration
training_config = DrDRPOConfig(
    output_dir=OUTPUT_DIR,
    
    # Basic training parameters
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    learning_rate=5e-7,
    num_train_epochs=1,
    
    # DRPO parameters
    num_monte_carlo_samples=2,
    beta=0.05,
    kl_type="k3",
    is_clip_min=0.1,
    is_clip_max=5.0,
    
    # Dr.DRPO specific parameters
    beta_prime=1.0,
    use_dpo_preference=True,
    
    # Generation parameters
    max_new_tokens=128,
    temperature=0.75,
    max_length=640,
    max_prompt_length=512,
    max_completion_length=128,
    
    # Optimization settings
    bf16=True,
    tf32=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch_fused",
    
    # DDP settings
    ddp_find_unused_parameters=False,
    
    # Logging and saving
    logging_steps=10,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=50,
    
    # Memory optimization
    torch_empty_cache_steps=50,
    
    # Evaluation
    eval_with_generation=True,
    eval_mc_samples=1,
    
    # Reporting
    report_to="wandb",  # Change to "wandb" if needed
    run_name="drdrpo-pythia-tldr",
)

# LoRA configuration for Pythia (GPT-NeoX architecture)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "query_key_value",  # Pythia's combined QKV projection
        "dense",            # Output projection in attention
        "dense_h_to_4h",    # MLP input projection
        "dense_4h_to_h",    # MLP output projection
    ],
)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    args = parser.parse_args()
    
    # Update config based on args
    if args.use_wandb:
        training_config.report_to = "wandb"
        import wandb
        wandb.init(project="drdrpo-pythia", name=training_config.run_name)
    
    print("Loading dataset...")
    train_dataset = load_dataset(DATASET_NAME, split="train")
    eval_dataset = load_dataset(DATASET_NAME, split="test[:500]")
    
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
        attn_implementation="eager",  # Don't use flash attention with LoRA
    )
    
    # ref_model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_NAME,
    #     torch_dtype=torch.bfloat16,
    #     use_cache=False,
    #     trust_remote_code=True,
    # )
    #
    # Apply LoRA
    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
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
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False
    
    # Initialize Dr.DRPO trainer
    print("Initializing Dr.DRPO trainer...")
    trainer = DrDRPOTrainer(
        model=model,
        # ref_model=ref_model,
        reward_model=reward_model,
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
    
    # Save weight analysis for noise detection
    trainer.save_weight_analysis(os.path.join(OUTPUT_DIR, "weight_analysis.json"))
    
    print("Training complete!")
    print("\nAnalyze results with:")
    print(f"python analyze_drdrpo_results.py {OUTPUT_DIR}/weight_analysis.json")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()