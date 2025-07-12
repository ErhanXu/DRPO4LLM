# train_drpo.py - Fixed version
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

def main():
    # Initialize wandb (optional)
    wandb.init(project="drpo-tldr", name="drpo-pythia-1b-tldr")
    
    # Model paths
    model_name = "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr"
    reward_model_name = "cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr"
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models WITHOUT device_map="auto" to avoid meta tensor issues
    # Let the trainer handle device placement
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Load reference model
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Load reward model
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name,
        torch_dtype=torch.bfloat16,
        num_labels=1,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load dataset
    dataset = load_dataset("Kyleyee/train_data_tldr")
    
    # Filter to only needed columns
    def preprocess_dataset(examples):
        return {
            "prompt": examples["prompt"],
            "chosen": examples["chosen"],
            "rejected": examples["rejected"],
        }
    
    train_dataset = dataset["train"].map(
        preprocess_dataset,
        batched=True,
        remove_columns=[col for col in dataset["train"].column_names 
                       if col not in ["prompt", "chosen", "rejected"]],
    )
    
    eval_dataset = dataset["test"].map(
        preprocess_dataset,
        batched=True,
        remove_columns=[col for col in dataset["test"].column_names 
                       if col not in ["prompt", "chosen", "rejected"]],
    )
    
    # Use smaller subset for testing
    train_dataset = train_dataset.select(range(1000))
    eval_dataset = eval_dataset.select(range(100))
    
    # PEFT configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query_key_value"],  # For Pythia models
    )
    
    # DRPO configuration
    training_args = DRPOConfig(
        # Basic training settings
        output_dir="./drpo-pythia-tldr",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        
        # Learning rate and optimization
        learning_rate=5e-6,
        # lr_scheduler_type="cosine",
        # warmup_ratio=0.1,
        # optim="adamw_torch",
        bf16=True,
        
        # DRPO specific settings
        beta=0.1,
        num_monte_carlo_samples=2,
        max_length=512,
        max_new_tokens=128,
        
        # Temperature settings
        # forward_temperature=0.9,
        # generate_temperature=1.0,
        # temperature_schedule="constant",  # or "linear" for scheduling
        temperature=0.66,
        
        # Importance sampling control
        is_control_method="clip",
        is_clip_min=0.05,
        is_clip_max=20.0,
        
        # KL type
        kl_type="k3",
        
        # Evaluation and logging
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        logging_steps=10,
        logging_first_step=True,
        
        # Additional settings
        remove_unused_columns=False,
        label_names=[],
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Disable vLLM
        use_vllm=False,
        
        # Memory optimization
        fp16=False,  # Use bf16 instead
        # optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        # per_device_train_batch_size=2 if torch.cuda.is_available() else 1,
    )
    
    # Initialize trainer
    trainer = DRPOTrainer(
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    # Train
    print("Starting DRPO training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    
    # Generate some examples
    print("\nGenerating sample completions...")
    test_prompts = [
        "SUBREDDIT: r/AskReddit\nTITLE: What's the best way to learn Python?\nPOST: I'm a beginner and want to learn it properly.\nTL;DR:",
    ]
    
    model.eval()
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.6,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response[len(prompt):]}")

if __name__ == "__main__":
    main()