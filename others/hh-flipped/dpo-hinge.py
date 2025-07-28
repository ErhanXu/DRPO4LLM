import os
import torch
import logging
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

# --- Configuration ---
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment and Reporting
# The user's original script included swanlab for wandb syncing.
try:
    import swanlab
    os.environ["WANDB_MODE"] = "offline"
    swanlab.sync_wandb()
    logging.info("Swanlab initialized for wandb syncing.")
except ImportError:
    logging.warning("Swanlab not found. Skipping wandb sync.")

# --- General Configuration ---
# Cache directory for datasets and models to avoid re-downloading
CACHE_DIR = "/root/autodl-tmp/cache"
# Device Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {DEVICE}")


# --- Part 1: Training Configuration ---
# Arguments for the model to be trained
MODEL_ARGS = ModelConfig(
    model_name_or_path="Kyleyee/Qwen2.5-1.5B-sft-hh-3e",
)

# Arguments for the training dataset
TRAIN_SCRIPT_ARGS = ScriptArguments(
    dataset_name="Eehan/train_data_helpful_flipped-10",
    dataset_train_split="train",
    dataset_test_split="test",
)

# DPO training configuration
TRAINING_ARGS = DPOConfig(
    # DPO-specific parameters
    loss_type="hinge",  # Using hinge loss for DPO
    label_smoothing=0.1,  # As requested
    beta=0.1,  # KL regularization coefficient - PLEASE CONFIRM IF THIS VALUE IS OK
    
    # General training parameters (keeping same as ORPO)
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5.0e-7,
    bf16=True,
    logging_steps=50,
    num_train_epochs=1,
    eval_strategy="steps",
    eval_steps=500,
    # The model will be saved here locally
    output_dir="/root/autodl-tmp/selfmodel/DPO_hinge_trained",
    # The trained model will be pushed to this repository on the Hub
    hub_model_id="Eehan/Qwen2.5-1.5B-dpo-hinge-flip-hh",
    push_to_hub=True,
    save_strategy="no",
    report_to=["wandb"],
)


# --- Part 2: Generation Configuration ---
# The model to use for generation will be the one we just trained
# This is automatically set from TRAINING_ARGS.hub_model_id after training
# METHOD_NAME is used as the column name for the generated responses
METHOD_NAME = "dpo-hinge-flipped"

# Dataset for generating evaluation responses
EVAL_INPUT_DATASET_ID = "Eehan/train_data_helpful_flipped-10"
EVAL_INPUT_DATASET_SPLIT = "test"

# The dataset to merge results into. If it doesn't exist, a new one is created.
EVAL_MERGE_INTO_DATASET_ID = "Eehan/eval-hh-flipped"
# The final dataset name to be pushed to the hub
EVAL_OUTPUT_DATASET_ID = "Eehan/eval-hh-flipped"

# Generation parameters
TEMPERATURES = [0, 0.25, 0.5, 0.75, 1]
MAX_NEW_TOKENS = 256
REPETITION_PENALTY = 1.0


# --- Part 1: Training Function ---
def train_model(model_args: ModelConfig, script_args: ScriptArguments, training_args: DPOConfig):
    """
    Loads a base model, trains it using DPOTrainer, and pushes it to the Hub.
    """
    logging.info("--- Starting Model Training ---")
    
    # 1. Load Model and Tokenizer for Training
    logging.info(f"Loading base model: {model_args.model_name_or_path}")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs, cache_dir=CACHE_DIR
    )
    
    # Load reference model for DPO (using same base model)
    logging.info("Loading reference model for DPO")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs, cache_dir=CACHE_DIR
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, cache_dir=CACHE_DIR
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load and Prepare Training Dataset
    logging.info(f"Loading training dataset: {script_args.dataset_name}")
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, cache_dir=CACHE_DIR)
    # Remove columns that are not needed for DPO training
    logging.info(f"Sample from training data: {dataset['train'][0]}")

    # 3. Initialize and Run Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,  # DPO requires a reference model
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split].select(range(1000)) if training_args.eval_strategy != "no" else None,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
    )
    
    logging.info("Starting trainer.train()...")
    trainer.train()
    logging.info("Training complete.")

    # 4. Save and Push to Hub
    logging.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        logging.info(f"Pushing trained model to Hub: {training_args.hub_model_id}")
        trainer.push_to_hub()
        logging.info("Model successfully pushed to Hub.")
    
    return training_args.hub_model_id


# --- Part 2: Generation Functions ---
def generate_text(prompts: list[str], tokenizer: AutoTokenizer, model: AutoModelForCausalLM, temperature: float) -> list[str]:
    """Generates text completions for a batch of prompts."""
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(model.device)
    generate_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": temperature > 0,
        "repetition_penalty": REPETITION_PENALTY,
    }
    if temperature > 0:
        generate_kwargs["temperature"] = temperature
    outputs = model.generate(**inputs, **generate_kwargs)
    generated_ids = outputs[:, inputs.input_ids.shape[1]:]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

def truncate_after_human(texts: list[str]) -> list[str]:
    """Truncates responses at the first occurrence of '\n\nHuman'."""
    return [text.split("\n\nHuman")[0].strip() for text in texts]

def process_and_generate(examples: dict, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, temperature: float) -> dict:
    """Applies chat template, generates responses, and truncates them."""
    messages = [[{"role": "user", "content": prompt}] for prompt in examples["prompt"]]
    formatted_prompts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    responses = generate_text(formatted_prompts, tokenizer, model, temperature)
    truncated_responses = truncate_after_human(responses)
    return {"generated_response": truncated_responses}

def generate_and_upload(trained_model_id: str):
    """
    Loads the fine-tuned model, generates responses on the eval dataset, and uploads the results.
    """
    logging.info("--- Starting Response Generation and Upload ---")
    
    # 1. Load the Newly Trained Model and Tokenizer
    logging.info(f"Loading newly trained model from Hub: {trained_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(trained_model_id)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(trained_model_id).to(DEVICE)
    model.eval()

    # 2. Load Datasets for Generation
    logging.info(f"Loading base dataset for generation: {EVAL_INPUT_DATASET_ID}[{EVAL_INPUT_DATASET_SPLIT}]")
    base_dataset = load_dataset(EVAL_INPUT_DATASET_ID, split=EVAL_INPUT_DATASET_SPLIT)
    
    try:
        output_dataset_dict = load_dataset(EVAL_MERGE_INTO_DATASET_ID)
        logging.info(f"Loaded existing dataset '{EVAL_MERGE_INTO_DATASET_ID}' to merge results into.")
    except Exception as e:
        logging.warning(f"Could not load '{EVAL_MERGE_INTO_DATASET_ID}'. Creating a new DatasetDict. Reason: {e}")
        output_dataset_dict = DatasetDict()

    # 3. Generate Responses and Update Dataset
    for temp in TEMPERATURES:
        logging.info(f"Generating responses for temperature: {temp}")
        processed_dataset = base_dataset.map(
            lambda ex: process_and_generate(ex, tokenizer, model, temp),
            batched=True, batch_size=16
        )
        
        split_name = f"temperature_{temp}"
        if split_name not in output_dataset_dict:
            logging.info(f"Creating new split '{split_name}' in output dataset.")
            output_dataset_dict[split_name] = base_dataset
            
        logging.info(f"Adding column '{METHOD_NAME}' to split '{split_name}'")
        output_dataset_dict[split_name] = output_dataset_dict[split_name].add_column(
            name=METHOD_NAME, column=processed_dataset["generated_response"]
        )

    # 4. Push Final Dataset to Hub
    logging.info(f"Pushing the final evaluation dataset to the Hub: {EVAL_OUTPUT_DATASET_ID}")
    output_dataset_dict.push_to_hub(EVAL_OUTPUT_DATASET_ID)
    logging.info("Evaluation dataset successfully pushed to the Hub.")


# --- Main Execution ---
if __name__ == "__main__":
    # Step 1: Train the model and get the Hub ID of the trained model
    trained_model_hub_id = train_model(MODEL_ARGS, TRAIN_SCRIPT_ARGS, TRAINING_ARGS)
    
    # Step 2: Use the trained model to generate and upload evaluation results
    if trained_model_hub_id:
        generate_and_upload(trained_model_hub_id)
    else:
        logging.error("Training did not return a valid model ID. Halting generation.")