# train_fsdp_peft_fixed.py
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
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from accelerate import Accelerator
from trainer.drpo_trainer_new import DRPOTrainer
from trainer.drpo_config_new import DRPOConfig
import swanlab
from transformers import TrainerCallback

# Initialize accelerator
accelerator = Accelerator()

# Model paths
model_name_or_path = "Qwen/Qwen2.5-1.5B"  # Base model path
reward_model_name_or_path = "Kyleyee/Qwen2.5-1.5B-reward-hh-retrain"  # Reward model path, can be same as base model


# SwanLab callback
class SwanLabCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and accelerator.is_main_process:
            clean_logs = {k: float(v) for k, v in logs.items() if v is not None}
            swanlab.log(clean_logs, step=state.global_step)

# Initialize SwanLab on main process
if accelerator.is_main_process:
    swanlab.init(
        project="drpo-fsdp-peft",
        experiment_name=f"drpo-{model_name_or_path.split('/')[-1]}",
    )

# Training configuration - FIXED FSDP OPTIONS
training_config = DRPOConfig(
    output_dir="./drpo-fsdp-peft-qwen2.5-7b",
    
    # CORRECT FSDP configuration - only use valid options
    fsdp="shard_grad_op auto_wrap",  # Only valid FSDPOption values
    fsdp_transformer_layer_cls_to_wrap="Qwen2DecoderLayer",
    
    # Additional FSDP settings through fsdp_config
    fsdp_config={
        "backward_prefetch": "backward_pre",
        "use_orig_params": True,  # Critical for PEFT
        "cpu_ram_efficient_loading": True,
        "sync_module_states": True,
    },
    
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
    tf32=True,
    gradient_checkpointing=False,  # Disable for FSDP+PEFT compatibility
    
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
    dataloader_num_workers=0,
    
    # Disable default reporting
    report_to="none",
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    use_cache=False,
)

# Apply PEFT
print("Applying PEFT...")
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=128,
    lora_alpha=256,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    modules_to_save=None,
)

# Enable input gradients for PEFT
model.enable_input_require_grads()
model = get_peft_model(model, peft_config)

if accelerator.is_main_process:
    model.print_trainable_parameters()

# Load reward model
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_path,
    torch_dtype=torch.bfloat16,
    use_cache=False,
)

# Prepare datasets
print("Loading datasets...")
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:10000]")
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

# Process datasets
train_dataset = dataset.map(
    prepare_dataset,
    remove_columns=dataset.column_names,
    num_proc=1
).filter(lambda x: x is not None)

eval_dataset = eval_dataset.map(
    prepare_dataset,
    remove_columns=eval_dataset.column_names,
    num_proc=1
).filter(lambda x: x is not None)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")

# Create custom DRPO trainer for FSDP
class DRPOTrainerFSDP(DRPOTrainer):
    """DRPO trainer with FSDP+PEFT fixes."""
    
    def _prepare_model(self, model):
        """Let HF Trainer handle FSDP wrapping."""
        model.train()
        return model
    
    def _generate(self, model, prompt_ids, prompt_mask, num_samples=1):
        """FSDP-compatible generation."""
        # Store original settings
        was_training = model.training
        original_use_cache = model.config.use_cache
        
        # Prepare for generation
        model.eval()
        model.config.use_cache = True
        
        try:
            with torch.no_grad():
                # For FSDP, we need to handle generation carefully
                # Get the underlying model
                if hasattr(self.accelerator, 'unwrap_model'):
                    unwrapped_model = self.accelerator.unwrap_model(model)
                else:
                    unwrapped_model = model
                
                # Generate
                eos_token_id = self.processing_class.eos_token_id
                pad_token_id = self.processing_class.pad_token_id
                
                output_ids = unwrapped_model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    max_new_tokens=self.args.max_new_tokens,
                    temperature=self.args.temperature,
                    do_sample=True,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )
                
                # Extract completions
                completion_ids = output_ids[:, prompt_ids.size(1):]
                
                # Create attention masks
                completion_mask = (completion_ids != pad_token_id).long()
                
                return prompt_ids, prompt_mask, completion_ids, completion_mask
                
        finally:
            # Restore original settings
            model.config.use_cache = original_use_cache
            if was_training:
                model.train()

# Create trainer
trainer = DRPOTrainerFSDP(
    model=model,
    ref_model=None,
    reward_model=reward_model,
    args=training_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    callbacks=[SwanLabCallback()],
)

# Train
if __name__ == "__main__":
    print("Starting training...")
    trainer.train()
    
    # Save model
    if accelerator.is_main_process:
        print("Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(training_config.output_dir)
    
    accelerator.wait_for_everyone()
    print("Training complete!")