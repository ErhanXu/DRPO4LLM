import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import swanlab
os.environ["WANDB_MODE"] = "offline"
swanlab.sync_wandb()
import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from trainer.dr_dpo_trainer import DrDPOTrainer




def main(script_args, training_args, model_args):
    ################
    # Model & Tokenizer
    ###################
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
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs, cache_dir="/root/autodl-tmp/cache",
    )
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs, cache_dir="/root/autodl-tmp/cache",
        )
    else:
        ref_model = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, cache_dir="/root/autodl-tmp/cache",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        print("No chat template")
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, cache_dir="/root/autodl-tmp/cache")
    dataset = dataset.remove_columns(["a_1","a_2", "chosen_preference", "rejected_preference", "a_1_preference", "a_2_preference"])
    print(f"Loaded dataset sample: {dataset['train'][0]}")

    ##########
    # Training
    ################
    trainer = DrDPOTrainer(
        model,
        ref_model,
        beta_prime = 1.0,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split].select(range(1000)) if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, DPOConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("dpo", help="Run the DPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser

script_args = ScriptArguments(
        dataset_name="Kyleyee/train_data_Helpful_drdpo_preference",
        dataset_train_split="train",
        dataset_test_split="test",
    )
training_args = DPOConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        # label_smoothing=0.1,
        learning_rate=5.0e-7,
        bf16=True,
        logging_steps=50,
        num_train_epochs=1,
        eval_strategy="steps",
        eval_steps=500,
        push_to_hub=True,  
        save_strategy="no",
        output_dir="/root/autodl-tmp/selfmodel/Dr_DPO",
        hub_model_id="Eehan/Qwen2.5-1.5B-dr_dpo-hh",
        report_to=["wandb"],
    )
model_args = ModelConfig(
        model_name_or_path="Kyleyee/Qwen2.5-1.5B-sft-hh-3e",
    )
if __name__ == "__main__":
    # parser = make_parser()
    # script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
