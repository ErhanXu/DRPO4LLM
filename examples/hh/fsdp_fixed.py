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
from trainer.drpo_trainer_fsdp import DRPOTrainer  # Note: The class is DRPOTrainer, not DRPOTrainerFSDP
from trainer.drpo_config_fsdp import DRPOConfigFSDP
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import wandb

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import swanlab
os.environ["WANDB_MODE"] = "offline"
swanlab.sync_wandb()

# Model configuration
model_name_or_path = "Kyleyee/Qwen2.5-1.5B-sft-hh-3e" # or "Qwen/Qwen2.5-1.5B" for smaller model
reward_model_path = "Kyleyee/Qwen2.5-1.5B-reward-hh-retrain" # or matching reward model
reward_model_revision = "f70cf091d59583749f030005b70834d29ba70fda"

# IMPORTANT: Detect model architecture for FSDP layer wrapping
config = AutoConfig.from_pretrained(model_name_or_path)
model_type = config.model_type

# Map model types to their decoder layer classes
LAYER_CLS_MAPPING = {
    "qwen2": "Qwen2DecoderLayer",
    "llama": "LlamaDecoderLayer", 
    "mistral": "MistralDecoderLayer",
    "gemma": "GemmaDecoderLayer",
    "gpt2": "GPT2Block",
    "opt": "OPTDecoderLayer",
}

layer_cls = LAYER_CLS_MAPPING.get(model_type, "Qwen2DecoderLayer")
print(f"Using layer class: {layer_cls} for model type: {model_type}")

# FSDP-optimized configuration
training_config = DRPOConfigFSDP(
    output_dir="./drpo-fsdp-hh",
    
    # FSDP configuration - SHARD_GRAD_OP is like DeepSpeed ZeRO-2
    fsdp="shard_grad_op auto_wrap",  # This will be handled by accelerate config
    fsdp_transformer_layer_cls_to_wrap=layer_cls,
    
    # Training parameters
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch size = 2 * 8 * num_gpus
    per_device_eval_batch_size=4,
    
    learning_rate=5e-7,
    num_train_epochs=1,
    warmup_steps=100,
    max_steps=-1,
    
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
    bf16=True,
    fp16=False,  # Use bf16 with modern GPUs
    # tf32=True,  # Enable TF32 for A100/H100
    gradient_checkpointing=False,
    # gradient_checkpointing_kwargs={"use_reentrant": False},  # Non-reentrant for FSDP
    optim="adamw_torch_fused",  # Fused optimizer for better performance
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    weight_decay=0.01,
    
    # Memory optimization
    torch_empty_cache_steps=50,
    
    # Logging and saving
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    eval_strategy="steps", 
    eval_steps=500,
    load_best_model_at_end=True,
    
    # Dataset processing
    dataset_num_proc=1,  # Use multiple CPU cores
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    
    # Evaluation
    eval_with_generation=True,
    eval_mc_samples=1,
    metric_for_best_model="eval_generated/win_rate_vs_chosen",
    greater_is_better=True,
    
    # Reporting
    report_to="wandb",  # or "none" to disable
    run_name="drpo-fsdp-qwen2.5-1.5b-hh",
    
    # FSDP specific optimizations
    ddp_find_unused_parameters=False,
    save_safetensors=True,
)

# Adjust settings based on single vs multi-GPU
world_size = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", 0))

if world_size == 1:
    print("Single GPU detected - disabling FSDP")
    training_config.fsdp = None
else:
    print(f"Multi-GPU detected - using FSDP with {world_size} GPUs")

# PEFT LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,  # LoRA rank
    lora_alpha=128,  # LoRA alpha
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    modules_to_save=None,  # Don't save embeddings/lm_head with FSDP for memory efficiency
)

# Load and prepare dataset
print("Loading and preparing dataset...")
dataset = load_dataset("Eehan/train_data_hh", split="train")
eval_dataset = load_dataset("Eehan/train_data_hh", split="test[:2000]")

# def prepare_dataset(example):
#     """Extract prompt and chosen/rejected responses from HH-RLHF format."""
#     chosen_text = example["chosen"]
#     rejected_text = example["rejected"]
    
#     # Find the last Human/Assistant exchange
#     last_human_idx = chosen_text.rfind("\n\nHuman: ")
#     last_assistant_idx = chosen_text.rfind("\n\nAssistant: ")
    
#     if last_human_idx == -1 or last_assistant_idx == -1:
#         return None
    
#     # Extract prompt (includes everything up to "Assistant: ")
#     prompt = chosen_text[:last_assistant_idx + len("\n\nAssistant: ")]
    
#     # Extract responses (everything after "Assistant: ")
#     chosen_response = chosen_text[last_assistant_idx + len("\n\nAssistant: "):].strip()
    
#     rejected_last_idx = rejected_text.rfind("\n\nAssistant: ")
#     rejected_response = rejected_text[rejected_last_idx + len("\n\nAssistant: "):].strip()
    
#     return {
#         "prompt": prompt,
#         "chosen": chosen_response,
#         "rejected": rejected_response,
#     }

# # Process datasets
# train_dataset = dataset.map(
#     prepare_dataset,
#     remove_columns=dataset.column_names,
#     num_proc=training_config.dataset_num_proc,
#     desc="Processing train dataset"
# ).filter(lambda x: x is not None)

# eval_dataset = eval_dataset.map(
#     prepare_dataset,
#     remove_columns=eval_dataset.column_names,
#     num_proc=training_config.dataset_num_proc,
#     desc="Processing eval dataset"
# ).filter(lambda x: x is not None)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load models
print("Loading models...")

# Load policy model
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    use_cache=False,  # Disable KV cache for training
    # attn_implementation="sdpa",  # Use scaled dot product attention
)

# Prepare model for PEFT training
if training_config.gradient_checkpointing:
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs=training_config.gradient_checkpointing_kwargs
    )

# Apply PEFT LoRA
print("Applying LoRA...")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Load reward model (will not be wrapped with FSDP)
print("Loading reward model...")
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_path,
    torch_dtype=torch.bfloat16,
    use_cache=False,
    revision=reward_model_revision,
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
    # Note: Don't pass peft_config since we already applied LoRA
)

# Main training function
def main():
    # Initialize wandb if using it
    if training_config.report_to == "wandb" and local_rank == 0:
        wandb.init(
            project="drpo-fsdp",
            name=training_config.run_name,
            config=training_config.to_dict()
        )
    
    print("Starting training...")
    trainer.train()
    
    # Save the model
    if trainer.is_world_process_zero():
        print("Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(training_config.output_dir)
        
        # Also save LoRA separately for easier loading
        model.save_pretrained(os.path.join(training_config.output_dir, "lora"))
        
    print("Training complete!")

if __name__ == "__main__":
    main()