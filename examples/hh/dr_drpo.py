import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import swanlab
import wandb
os.environ["WANDB_MODE"] = "offline"
swanlab.sync_wandb()
import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

from trainer.drdrpo_trainer import DrDRPOTrainer, DrDRPOConfig

# Configuration
MODEL_NAME = "Kyleyee/Qwen2.5-1.5B-sft-hh-3e"  # Change to your model
REWARD_MODEL_NAME = "Kyleyee/Qwen2.5-1.5B-gpm-hh-2e-2dim"  # Change to your reward model
DATASET_NAME = "Eehan/train_data_helpful_flipped-10"  # Change to your dataset
OUTPUT_DIR = "./drpo-flipped-nolora"

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
    beta=0.05,
    kl_type="k3",
    is_clip_min=0.1,
    is_clip_max=5.0,
    
    # Dr.DRPO specific parameters
    beta_prime=1.0,
    use_dpo_preference=True,
    # weight_clip_range=(0.1, 10.0),
    
    # Generation parameters
    max_new_tokens=256,
    temperature=0.85,
    max_length=1024,
    max_prompt_length=512,
    max_completion_length=512,
    
    # Optimization settings
    bf16=True,
    # gradient_checkpointing=True,
    
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
    use_preference_model=True,
    preference_model_path=REWARD_MODEL_NAME,
    preference_model_type="general",
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


def main():
    """Main training function."""
    
    if training_config.report_to == "wandb":
        wandb.init(project="drdrpo-training", name=training_config.run_name)
    
    print("Loading and preparing dataset...")
    # Load dataset
    train_dataset = load_dataset(DATASET_NAME, split="train")  # Using subset for demo
    eval_dataset = load_dataset(DATASET_NAME, split="test[:500]")
    # eval_dataset = None
    
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load models
    print("Loading models...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        trust_remote_code=True,
    )
    
    # ref_model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_NAME,
    #     torch_dtype=torch.bfloat16,
    #     use_cache=False,
    #     trust_remote_code=True,
    # )
    
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
        ref_model=None,
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