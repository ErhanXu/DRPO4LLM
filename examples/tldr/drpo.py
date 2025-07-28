import os
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GPTNeoXForCausalLM,
)
from peft import LoraConfig, TaskType, get_peft_model
import wandb
import swanlab
os.environ["WANDB_MODE"] = "offline"
swanlab.sync_wandb()

# Add parent directories to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from trainer.drpo_trainer_new import DRPOTrainer
from trainer.drpo_config_new import DRPOConfig

# Configuration
MODEL_NAME = "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr"
REWARD_MODEL_NAME = "cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr"
DATASET_NAME = "Eehan/train_data_tldr_flipped-10"
OUTPUT_DIR = "./drpo-pythia-1b-tldr"

# Training configuration optimized for Pythia-1B
training_config = DRPOConfig(
    output_dir=OUTPUT_DIR,
    push_to_hub=True,
    hub_model_id="Eehan/pythia-1b-drpo-lora-tldr",
    
    # Basic training parameters
    per_device_train_batch_size=4,  # Reduced for 1B model with LoRA
    gradient_accumulation_steps=8,   # Effective batch size = 4 * 8 = 32
    per_device_eval_batch_size=8,
    learning_rate=5e-6,  # Slightly higher LR for LoRA
    num_train_epochs=1,
    warmup_steps=100,
    
    # DRPO specific parameters
    num_monte_carlo_samples=2,
    beta=0.1,
    kl_type="k3",
    is_clip_min=0.1,
    is_clip_max=5.0,
    
    # Generation parameters
    max_new_tokens=128,
    temperature=0.7,
    max_length=640,
    max_prompt_length=512,
    max_completion_length=128,
    
    # Optimization settings
    bf16=True,  # Use bf16 for better performance
    tf32=True,  # Enable TF32 on Ampere GPUs
    gradient_checkpointing=True,  # Important for memory efficiency
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch_fused",  # Use fused optimizer for speed
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    weight_decay=0.01,
    
    # DDP settings
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
    metric_for_best_model="eval_generated/win_rate_vs_chosen",
    greater_is_better=True,
    
    # Dataset processing
    dataset_num_proc=1,  # Use multiple CPU cores
    # dataloader_num_workers=2,
    # dataloader_pin_memory=True,
    
    # Memory optimization
    torch_empty_cache_steps=50,
    
    # Evaluation
    eval_with_generation=True,
    eval_mc_samples=1,
    
    # Reporting
    report_to="wandb",  # or "none" if you don't want logging
    run_name="drpo-pythia-1b-lora-tldr",
)

# LoRA configuration for Pythia (GPT-NeoX architecture)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,  # LoRA rank
    lora_alpha=128,  # LoRA alpha (typically 2*r)
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "query_key_value",  # Pythia's combined QKV projection
        "dense",            # Output projection in attention
        "dense_h_to_4h",    # MLP input projection
        "dense_4h_to_h",    # MLP output projection
    ],
    # Don't include embedding/lm_head in modules_to_save for DDP
    modules_to_save=None,
)


def prepare_tldr_dataset(example):
    """
    Prepare TLDR dataset for DRPO training.
    Expected format:
    - prompt: The Reddit post to summarize
    - chosen: The preferred summary
    - rejected: The less preferred summary
    """
    # The dataset should already have prompt/chosen/rejected format
    # If not, you'll need to adapt this function
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


def main():
    """Main training function."""
    
    # Initialize wandb if using
    if training_config.report_to == "wandb":
        wandb.init(
            project="drpo-pythia-training",
            name=training_config.run_name,
            config=training_config.__dict__
        )
    
    print("Loading and preparing dataset...")
    # Load dataset
    train_dataset = load_dataset(DATASET_NAME, split="train")
    eval_dataset = load_dataset(DATASET_NAME, split="test[:1000]")
    
    # If dataset needs preprocessing, uncomment and adapt:
    # train_dataset = train_dataset.map(prepare_tldr_dataset)
    # eval_dataset = eval_dataset.map(prepare_tldr_dataset)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Pythia uses <|endoftext|> as both pad and eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Ensure proper token settings for Pythia
    print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    
    # Load models
    print("Loading models...")
    
    # Load base model with specific settings for Pythia
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        use_cache=False,  # Disable KV cache for training
        trust_remote_code=True,
        # Don't use flash attention with LoRA + gradient checkpointing
        attn_implementation="eager",
    )
    
    # Verify model architecture
    print(f"Model type: {model.config.model_type}")
    print(f"Model architecture: {type(model)}")
    
    # Apply LoRA
    print("Applying LoRA configuration...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing after LoRA
    if training_config.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # Load reward model
    print("Loading reward model...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        trust_remote_code=True,
    )
    
    # Ensure reward model is in eval mode and doesn't require gradients
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False
    
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
        # Don't pass peft_config since we already applied LoRA
    )
    
    # Optional: Log initial evaluation
    print("Running initial evaluation...")
    initial_metrics = trainer.evaluate()
    print(f"Initial metrics: {initial_metrics}")
    
    # Start training
    print("Starting training...")
    train_result = trainer.train()
    
    # Log final metrics
    print(f"Training completed. Final loss: {train_result.training_loss}")
    
    # Save the final model
    print("Saving model...")
    trainer.save_model()
    
    # Save tokenizer
    tokenizer.save_pretrained(training_config.output_dir)
    
    # Create model card
    trainer.create_model_card(
        model_name=f"pythia-1b-drpo-lora-tldr",
        dataset_name=DATASET_NAME,
        tags=["drpo", "pythia", "lora", "tldr", "summarization"]
    )
    
    # Push to hub if needed
    if training_config.push_to_hub:
        print("Pushing to hub...")
        trainer.push_to_hub()
    
    print("Training complete!")
    
    # Final evaluation
    final_metrics = trainer.evaluate()
    print(f"Final evaluation metrics: {final_metrics}")


if __name__ == "__main__":
    # Set some environment variables for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Better CUDA performance
    
    main()