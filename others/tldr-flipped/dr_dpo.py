import os
import re
import torch
import logging
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import (
    DPOConfig,
    ModelConfig,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from trainer.dr_dpo_trainer import DrDPOTrainer

# --- Configuration ---
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment and Reporting
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

# IMPORTANT: Pythia models use GPT-NeoX architecture with rotary positional embeddings (ROPE)
# which have known issues with left padding. We MUST use right padding for both training
# and generation to ensure correct outputs. See: https://github.com/huggingface/transformers/issues/22161


# --- Part 1: Training Configuration ---
# Arguments for the model to be trained
MODEL_ARGS = ModelConfig(
    model_name_or_path="cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr",  # Pythia TLDR base model
)

# Arguments for the training dataset
TRAIN_SCRIPT_ARGS = ScriptArguments(
    dataset_name="Eehan/train_data_tldr",  # TLDR dataset
    dataset_train_split="train",
    dataset_test_split="test",
)

# DR-DPO training configuration
TRAINING_ARGS = DPOConfig(
    # DR-DPO specific parameters (DrDPOTrainer uses DPOConfig)
    beta=0.1,  # KL regularization coefficient
    label_smoothing=0.0,  # DR-DPO typically doesn't use label smoothing
    
    # General training parameters
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5.0e-7,
    bf16=True,
    logging_steps=50,
    num_train_epochs=1,
    eval_strategy="steps",
    eval_steps=500,
    # The model will be saved here locally
    output_dir="/root/autodl-tmp/selfmodel/DrDPO_tldr",
    # The trained model will be pushed to this repository on the Hub
    hub_model_id="Eehan/pythia-1b-dr_dpo-tldr",
    push_to_hub=True,
    save_strategy="no",
    report_to=["wandb"],
)


# --- Part 2: Generation Configuration ---
# The model to use for generation will be the one we just trained
METHOD_NAME = "dr_dpo"

# Dataset for generating evaluation responses
EVAL_INPUT_DATASET_ID = "Eehan/train_data_tldr"
EVAL_INPUT_DATASET_SPLIT = "test"

# The dataset to merge results into
EVAL_MERGE_INTO_DATASET_ID = "Eehan/eval-tldr-all"
# The final dataset name to be pushed to the hub
EVAL_OUTPUT_DATASET_ID = "Eehan/eval-tldr-all"

# Generation parameters
TEMPERATURES = [0, 0.25, 0.5, 0.75, 1]
MAX_NEW_TOKENS = 256
REPETITION_PENALTY = 1.0
GENERATION_BATCH_SIZE = 128  # From pipeline configuration


# --- Part 1: Training Function ---
def train_model(model_args: ModelConfig, script_args: ScriptArguments, training_args: DPOConfig):
    """
    Loads a base model, trains it using DrDPOTrainer, and pushes it to the Hub.
    """
    logging.info("--- Starting DR-DPO Model Training ---")
    
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
    
    # Load reference model for DR-DPO (using same base model)
    logging.info("Loading reference model for DR-DPO")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs, cache_dir=CACHE_DIR
    )
    
    # Configure tokenizer for TLDR dataset with Pythia model
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, cache_dir=CACHE_DIR
    )
    
    # CRITICAL: Pythia models MUST use right padding due to rotary embeddings
    # Left padding causes incorrect outputs with ROPE
    tokenizer.padding_side = "right"
    
    # Check if pad token exists, if not set it
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Load and Prepare Training Dataset
    logging.info(f"Loading training dataset: {script_args.dataset_name}")
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, cache_dir=CACHE_DIR)
    logging.info(f"Sample from training data: {dataset['train'][0]}")

    # 3. Initialize and Run Trainer
    trainer = DrDPOTrainer(
        model=model,
        ref_model=ref_model,  # DR-DPO requires a reference model
        beta_prime=1.0,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split].select(range(1000)) if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )
    
    logging.info("Starting trainer.train()...")
    trainer.train()
    logging.info("Training complete.")

    # 4. Save and Push to Hub
    logging.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    if training_args.push_to_hub:
        logging.info(f"Pushing trained model to Hub: {training_args.hub_model_id}")
        trainer.push_to_hub()
        tokenizer.push_to_hub(training_args.hub_model_id)
        logging.info("Model successfully pushed to Hub.")
    
    return training_args.hub_model_id


# --- Part 2: Generation Functions ---
def extract_first_paragraph(text: str) -> str:
    """Extracts the first paragraph from generated text."""
    return text.split('\n\n')[0]

def extract_post_content(text: str) -> str:
    """Extracts the POST content from the prompt."""
    pattern = r"POST:\s*(.*?)(?=\s*\nTL;DR:|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text  # Return original if pattern not found

def create_generation_pipeline(model_id: str):
    """Creates a generation pipeline with Pythia-specific tokenizer configuration."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
    logging.info(f"Model: {model_id}, EOS token: {tokenizer.eos_token}")
    
    # CRITICAL: Pythia models MUST use right padding due to rotary embeddings
    # Left padding causes incorrect outputs with ROPE
    tokenizer.padding_side = "right"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    gen_pipeline = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        batch_size=GENERATION_BATCH_SIZE,
        device=DEVICE,
    )
    
    return gen_pipeline, tokenizer

def generate_responses_batch(prompts: list[str], gen_pipeline, temperature: float) -> list[str]:
    """Generates responses for a batch of prompts using the pipeline."""
    generate_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": temperature > 0,
        "repetition_penalty": REPETITION_PENALTY,
        "return_full_text": True,
        "pad_token_id": gen_pipeline.tokenizer.pad_token_id,
        "eos_token_id": gen_pipeline.tokenizer.eos_token_id,
    }
    
    if temperature > 0:
        generate_kwargs["temperature"] = temperature
    
    # Generate responses
    generated = gen_pipeline(prompts, **generate_kwargs)
    
    # Extract only the generated part (after the prompt) and get first paragraph
    responses = []
    for i, batch in enumerate(generated):
        full_text = batch[0]["generated_text"]
        # Remove the prompt to get only generated text
        generated_text = full_text[len(prompts[i]):]
        # Extract first paragraph
        first_paragraph = extract_first_paragraph(generated_text)
        responses.append(first_paragraph.strip())
    
    return responses

def process_and_generate(examples: dict, gen_pipeline, temperature: float) -> dict:
    """Processes prompts and generates responses for TLDR dataset."""
    prompts = examples["prompt"]
    
    # For Pythia models, we typically don't use chat templates
    # Just use the prompts directly as they should already be formatted
    formatted_prompts = prompts
    
    # Generate responses
    responses = generate_responses_batch(formatted_prompts, gen_pipeline, temperature)
    
    return {"generated_response": responses}

def generate_and_upload(trained_model_id: str):
    """
    Loads the fine-tuned model, generates responses on the eval dataset, and uploads the results.
    """
    logging.info("--- Starting Response Generation and Upload ---")
    
    # 1. Create generation pipeline
    logging.info(f"Creating generation pipeline for model: {trained_model_id}")
    gen_pipeline, tokenizer = create_generation_pipeline(trained_model_id)
    
    # 2. Load Datasets for Generation
    logging.info(f"Loading base dataset for generation: {EVAL_INPUT_DATASET_ID}[{EVAL_INPUT_DATASET_SPLIT}]")
    base_dataset = load_dataset(EVAL_INPUT_DATASET_ID, split=EVAL_INPUT_DATASET_SPLIT, cache_dir=CACHE_DIR)
    
    try:
        output_dataset_dict = load_dataset(EVAL_MERGE_INTO_DATASET_ID, cache_dir=CACHE_DIR)
        logging.info(f"Loaded existing dataset '{EVAL_MERGE_INTO_DATASET_ID}' to merge results into.")
    except Exception as e:
        logging.warning(f"Could not load '{EVAL_MERGE_INTO_DATASET_ID}'. Creating a new DatasetDict. Reason: {e}")
        output_dataset_dict = DatasetDict()

    # 3. Generate Responses and Update Dataset
    for temp in TEMPERATURES:
        logging.info(f"Generating responses for temperature: {temp}")
        
        # Process in smaller batches if needed to avoid memory issues
        batch_size = 16  # Smaller batch size for map function
        processed_dataset = base_dataset.map(
            lambda ex: process_and_generate(ex, gen_pipeline, temp),
            batched=True, 
            batch_size=batch_size,
            desc=f"Generating at temperature {temp}"
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
    logging.info("Evaluation dataset successfully pushed to Hub.")
    
    # Clean up pipeline to free memory
    del gen_pipeline
    torch.cuda.empty_cache()


# --- Main Execution ---
if __name__ == "__main__":
    # Step 1: Train the model and get the Hub ID of the trained model
    trained_model_hub_id = train_model(MODEL_ARGS, TRAIN_SCRIPT_ARGS, TRAINING_ARGS)
    
    # Step 2: Use the trained model to generate and upload evaluation results
    if trained_model_hub_id:
        generate_and_upload(trained_model_hub_id)
    else:
        logging.error("Training did not return a valid model ID. Halting generation.")