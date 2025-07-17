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
from trl import PairRMJudge


# 1. Set up configuration
model_name_or_path = "Qwen/Qwen3-1.7B"  # Base model path
config = DRPOConfig(
    # Model configuration
    # model_name_or_path="Qwen/Qwen3-1.7B",  # Your base model
    output_dir="./drpo-pairrm-Qwen3-1.7B",
    
    # PairRM configuration
    use_preference_model=False, # use judge
    
    # Training hyperparameters
    learning_rate=5e-7,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch size = 16
    num_train_epochs=1,
    warmup_steps=100,
    
    # DRPO specific parameters
    num_monte_carlo_samples=2,  # Number of MC samples for expectation
    beta=0.1,  # KL regularization coefficient
    kl_type="k3",  # Online KL estimator
    is_clip_min=0.1,  # Min IS ratio clipping
    is_clip_max=10.0,  # Max IS ratio clipping
    
    # Generation parameters
    max_new_tokens=128,
    temperature=0.7,
    max_length=1024,
    
    # Optimization
    bf16=True,  # Use bf16 for model, fp16 for PairRM
    gradient_checkpointing=True,  # Save memory
    optim="adamw_torch",
    
    # Logging and saving
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    
    # Dataset processing
    max_prompt_length=512,
    max_completion_length=256,
    dataset_num_proc=1,
    
    # Evaluation
    eval_with_generation=True,
    eval_mc_samples=1,
    metric_for_best_model="eval_generated/win_rate_vs_rejected",
    greater_is_better=True,
    
    # Memory optimization
    torch_empty_cache_steps=50,  # Clear cache periodically
)

# 2. Load model and tokenizer
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_cache=False if config.gradient_checkpointing else True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

from trl import PairRMJudge

judge = PairRMJudge()


# 3. Load and prepare dataset
print("Loading dataset...")
# Example using Anthropic HH dataset
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")  # Use subset for testing

# The dataset should have 'prompt', 'chosen', and 'rejected' columns
# If your dataset has different column names, rename them:
def prepare_dataset(example):
    # Parse the conversation from the chosen field
    chosen_text = example["chosen"]
    rejected_text = example["rejected"]
    
    # Find the last "Assistant:" in the chosen text to split prompt from response
    last_assistant_idx = chosen_text.rfind("\n\nAssistant: ")
    if last_assistant_idx == -1:
        raise ValueError("Could not find Assistant response in chosen text")
    
    # Extract prompt (everything before the last assistant response)
    prompt = chosen_text[:last_assistant_idx + len("\n\nAssistant: ")]
    
    # Extract the chosen response (everything after the last "Assistant: ")
    chosen_response = chosen_text[last_assistant_idx + len("\n\nAssistant: "):].strip()
    
    # Extract the rejected response 
    # (Find the same position in rejected text and get everything after)
    rejected_last_idx = rejected_text.rfind("\n\nAssistant: ")
    rejected_response = rejected_text[rejected_last_idx + len("\n\nAssistant: "):].strip()
    
    return {
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
    }

# Apply the transformation
dataset = dataset.map(prepare_dataset)

# Split into train/eval
train_dataset = dataset.select(range(900))
eval_dataset = dataset.select(range(900, 1000))

print("Example after transformation:")
print(train_dataset[0])

# 4. Initialize trainer
print("Initializing DRPO trainer with PairRM...")
trainer = DRPOTrainer(
    model=model,
    ref_model=None,  # Will create from model automatically
    judge=judge,
    args=config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

# 5. Train the model
print("Starting training...")
trainer.train()

# 6. Save the final model
print("Saving model...")
trainer.save_model()
tokenizer.save_pretrained(config.output_dir)

# 7. Test the trained model
print("\nTesting trained model...")
# Load the trained model for inference
trained_model = AutoModelForCausalLM.from_pretrained(
    config.output_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Test generation
test_prompt = "What are the benefits of exercise?"
inputs = tokenizer(test_prompt, return_tensors="pt").to(trained_model.device)

with torch.no_grad():
    outputs = trained_model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prompt: {test_prompt}")
print(f"Response: {response}")