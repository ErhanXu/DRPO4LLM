import sys
import os

# Add the parent directory to Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import yaml
import torch
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from transformers import BitsAndBytesConfig
from trl import (
    ModelConfig,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from .trainer.drpo_utils import GPMwithRewardNetwork, estDPOStylePipeline, BTRewardNetwork
from .trainer import DRPOConfig, DRPOTrainer

DATASETNAME = "substitute for the dataset name"
MODELNAME = "substitute for the ref and initial policy model name"

def main(script_args, training_args, model_args):
    ################
    # Model & Tokenizer
    ################
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
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
        )
    else:
        ref_model = None
    

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )

    tokenizer.eos_token = "<|im_end|>" # necessary for Qwen2.5, decide whether to use depending on your base model

    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
    
    if training_args.is_bt_model:
        if isinstance(training_args.preference_model_id, dict):
            preference_pipeline = estDPOStylePipeline(training_args.preference_model_id)
        else: 
            preference_pipeline = BTRewardNetwork(training_args.preference_model_id, revision=training_args.preference_model_revision)
    else:
        preference_pipeline = GPMwithRewardNetwork(training_args.preference_model_id)



    ################
    # Dataset
    ################
    def transform_dataset(dataset, seed=996):
    # Process each split individually (train/test)
        def process_split(split):
            original = dataset[split]
            swapped = original.map(lambda x: {
                'a1': x['a2'],
                'a2': x['a1'],
                'rank': 1 - x['rank'],
            })

            return concatenate_datasets([original, swapped])

    # Apply processing to all splits
        return DatasetDict({
            split: process_split(split).shuffle(seed=seed)
            for split in dataset.keys()  # Handles 'train', 'test', etc.
        })
    dataset = load_dataset(script_args.dataset_name, revision=script_args.dataset_config["revision"])
    dataset = transform_dataset(dataset)

    print(f"\033[32mLoaded dataset sample:\033[0m {dataset['train'][0]}")
    print(f"\033[32mLoaded swapped dataset sample:\033[0m {dataset['train'][len(dataset['train'])-1]}")

    ################
    # Training
    ################
    trainer = DRPOTrainer(
        model=model,
        ref_model=ref_model,
        preference_model=preference_pipeline,
        train_dataset = dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        args=training_args,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    script_args = ScriptArguments(
            dataset_name=DATASETNAME,
            dataset_config={"revision": "substitute for the revision"},
            dataset_train_split="train",
            dataset_test_split="validation",
    )
    model_args = ModelConfig(
            model_name_or_path = MODELNAME,
    )

    with open("./examples/hh/config.yaml", "r") as f:
        training_args_config = yaml.safe_load(f)

    training_args = DRPOConfig(
        **training_args_config
    )
    main(script_args, training_args, model_args)
