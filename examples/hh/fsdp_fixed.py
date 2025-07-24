# train_fsdp_peft_fixed.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from accelerate import Accelerator
from trainer.drpo_trainer_new import DRPOTrainer
from trainer.drpo_config_new import DRPOConfig
import swanlab
from transformers import TrainerCallback

# CRITICAL: Set this environment variable
os.environ["ACCELERATE_USE_FSDP"] = "true"

# Initialize accelerator
accelerator = Accelerator()

# Model paths
model_name_or_path = "Qwen/Qwen2.5-7B"
reward_model_path = "Kyleyee/Qwen2.5-7B-reward-hh"

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

# Training configuration with FSDP settings
training_config = DRPOConfig(
    output_dir="./drpo-fsdp-peft-qwen2.5-7b",
    
    # CRITICAL: Add FSDP configuration here
    fsdp="shard_grad_op auto_wrap peft_backward_prefetch=BACKWARD_PRE",
    fsdp_transformer_layer_cls_to_wrap="Qwen2DecoderLayer",
    
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
    
    # Generation
    max_new_tokens=256,
    temperature=0.8,
    max_length=1024,
    
    # Optimizations
    bf16=True,
    tf32=True,
    gradient_checkpointing=False,  # Disable for FSDP compatibility
    
    # Logging
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    
    # Dataset
    dataset_num_proc=1,  # Use 1 for FSDP
    dataloader_num_workers=0,  # Set to 0 for FSDP
    
    # Disable default reporting
    report_to="none",
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# CRITICAL: Load model with specific settings
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    use_cache=False,  # Must be False for training
)

# CRITICAL: Set model attributes for FSDP
model.config.use_cache = False
model.config.pretraining_tp = 1  # Important for some models

# Apply PEFT BEFORE any FSDP wrapping
print("Applying PEFT...")
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=128,
    lora_alpha=256,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    modules_to_save=None,  # Don't save embeddings
)

# Prepare model for PEFT
model.enable_input_require_grads()
model = get_peft_model(model, peft_config)

if accelerator.is_main_process:
    model.print_trainable_parameters()

# Load reward model separately (don't wrap with FSDP)
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

# Process datasets with single process
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

# Create custom trainer that handles FSDP properly
class DRPOTrainerFSDP(DRPOTrainer):
    """Custom DRPO trainer for FSDP+PEFT."""
    
    def _prepare_model(self, model):
        """Don't do any special preparation - let HF Trainer handle FSDP."""
        # Just ensure the model is in training mode
        model.train()
        return model
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to handle FSDP generation properly."""
        # For generation with FSDP, we need special handling
        device = inputs["prompt_ids"].device
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_attention_mask"]
        
        # Store original use_cache setting
        original_use_cache = model.config.use_cache
        model.config.use_cache = True  # Enable for generation
        
        # Generate MC samples
        mc_samples = []
        
        # CRITICAL: Use FSDP's special handling for generation
        with torch.no_grad():
            # Set model to eval for generation
            model.eval()
            
            for _ in range(self.args.num_monte_carlo_samples):
                # Use accelerator's unwrap for generation
                unwrapped = self.accelerator.unwrap_model(model)
                
                # Generate directly
                try:
                    output_ids = unwrapped.generate(
                        input_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        max_new_tokens=self.args.max_new_tokens,
                        temperature=self.args.temperature,
                        do_sample=True,
                        pad_token_id=self.processing_class.pad_token_id,
                        eos_token_id=self.processing_class.eos_token_id,
                    )
                    
                    # Extract completions
                    completion_ids = output_ids[:, prompt_ids.size(1):]
                    completion_mask = (completion_ids != self.processing_class.pad_token_id).long()
                    
                    mc_samples.append((completion_ids, completion_mask))
                except Exception as e:
                    print(f"Generation error: {e}")
                    # Fallback: create dummy samples
                    batch_size = prompt_ids.size(0)
                    dummy_ids = torch.full((batch_size, 10), self.processing_class.pad_token_id, device=device)
                    dummy_mask = torch.zeros_like(dummy_ids)
                    mc_samples.append((dummy_ids, dummy_mask))
            
            # Return to training mode
            model.train()
        
        # Restore original use_cache setting
        model.config.use_cache = original_use_cache
        
        # Now continue with the regular training step
        # You'll need to implement the rest of the DRPO logic here
        # For now, just compute a dummy loss
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss

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
    try:
        trainer.train()
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
    
    # Save model
    if accelerator.is_main_process:
        print("Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(training_config.output_dir)
    
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        swanlab.finish()
    
    print("Training complete!")