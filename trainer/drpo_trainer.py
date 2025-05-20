import os
import textwrap
import warnings
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Union, Optional, Callable, List, Tuple
from functools import wraps
from packaging import version

import transformers
import torch.utils.data

from datasets import Dataset
from dataclasses import dataclass

from torch.utils.data import DataLoader, IterableDataset

from transformers import (
    DataCollator,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_apex_available,
    is_wandb_available,
)

from transformers.trainer_utils import seed_worker
from transformers.data.data_collator import DataCollatorMixin

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, seed_worker
from transformers.training_args import OptimizerNames
from transformers.utils import is_peft_available, is_sagemaker_mp_enabled, logging

from accelerate import PartialState

from trl.data_utils import maybe_extract_prompt
from trl.import_utils import is_vllm_available
from trl.models import create_reference_model
from trl.models.utils import unwrap_model_for_generation

from trl.trainer.utils import (
    SIMPLE_CHAT_TEMPLATE,
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    empty_cache,
    generate_model_card,
    get_comet_experiment_url,
    truncate_right,
    prepare_deepspeed,
    pad,
    truncate_right,
    selective_log_softmax
)

from .drpo_utils import get_preference_score, get_preference_score_without_decoding
from .drpo_config import DRPOConfig


# if is_peft_available():
#     from peft import PeftModel, get_peft_model

if is_wandb_available():
    import wandb

logger = logging.get_logger(__name__)

@dataclass
class DataCollatorDRPO(DataCollatorMixin):
    pad_token_id: int
    return_tensors: str = 'pt'

    def torch_call(self, examples: list[Union[list[int], dict[str,Any], Any]])-> dict[str, Any]:
        prompt_ids = [torch.tensor(example["prompt_ids"]) for example in examples]
        prompt_attention_mask = [torch.ones_like(id) for id in prompt_ids]
        a1_ids = [torch.tensor(example["a1_ids"]) for example in examples]
        a2_ids = [torch.tensor(example["a2_ids"]) for example in examples]
        a1_attention_mask = [torch.ones_like(id) for id in a1_ids]
        a2_attention_mask = [torch.ones_like(id) for id in a2_ids]

        output = {}
        output["prompt_ids"] = pad(prompt_ids, padding_value = self.pad_token_id, padding_side = "left")
        output["prompt_attention_mask"] = pad(prompt_attention_mask, padding_value = 0, padding_side = "left")
        output["a1_ids"] = pad(a1_ids, padding_value = self.pad_token_id, padding_side = "right")
        output["a1_attention_mask"] = pad(a1_attention_mask, padding_value = 0, padding_side = "right")
        output["a2_ids"] = pad(a2_ids, padding_value = self.pad_token_id, padding_side = "right")
        output["a2_attention_mask"] = pad(a2_attention_mask, padding_value = 0, padding_side = "right")

        output["rank"] = torch.tensor([example["rank"] for example in examples])

        return output
    
def is_conversational(example: dict[str, Any]) -> bool:
    r"""
    Check if the example is in a conversational format.

    Args:
        example (`dict[str, Any]`):
            A single data entry of a dataset. The example can have different keys depending on the
            dataset type.

    Returns:
        `bool`:
            `True` if the data is in a conversational format, `False` otherwise.

    Examples:

    ```python
    >>> example = {"prompt": [{"role": "user", "content": "What color is the sky?"}]}
    >>> is_conversational(example)
    True
    >>> example = {"prompt": "The sky is"})
    >>> is_conversational(example)
    False
    ```
    """
    supported_keys = ["prompt", "a1", "a2"]
    example_keys = {key for key in example.keys() if key in supported_keys}

    # It must have one of the supported keys
    if example_keys:
        key = example_keys.pop()  # take the first supported key
        maybe_messages = example[key]
        # It must be a list of messages,
        if isinstance(maybe_messages, list):
            maybe_message = maybe_messages[0]
            # Each message must a list of dictionaries with keys "role" and "content"
            if isinstance(maybe_message, dict) and "role" in maybe_message and "content" in maybe_message:
                return True

    return False
            
def maybe_apply_chat_template(
    example: dict[str, list[dict[str, str]]],
    tokenizer: PreTrainedTokenizerBase,
    tools: Optional[list[Union[dict, Callable]]] = None,
) -> dict[str, str]:
    r"""
    If the example is in a conversational format, apply a chat template to it.

    Args:
        example (`dict[str, list[dict[str, str]]`):
            Dictionary representing a single data entry of a conversational dataset. Each data entry can have different
            keys depending on the dataset type. The supported dataset types are:

                - Language modeling dataset: `"messages"`.
                - Prompt-only dataset: `"prompt"`.
                - Prompt-completion dataset: `"prompt"` and `"completion"`.
                - Preference dataset: `"prompt"`, `"chosen"`, and `"rejected"`.
                - Preference dataset with implicit prompt: `"chosen"` and `"rejected"`.
                - Unpaired preference dataset: `"prompt"`, `"completion"`, and `"label"`.

            For keys `"messages"`, `"prompt"`, `"chosen"`, `"rejected"`, and `"completion"`, the values are lists of
            messages, where each message is a dictionary with keys `"role"` and `"content"`.
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer to apply the chat template with.
        tools (`list[Union[dict, Callable]]` or `None`, *optional*, defaults to `None`):
            A list of tools (callable functions) that will be accessible to the model.
            If the template does not support function calling, this argument will have no effect

    Returns:
        `dict[str, str]`:
            Formatted example with the chat template applied.

    Notes:
        - This function does not alter the keys, except for Language modeling dataset, where `"messages"` is replaced
        by `"text"`.

        - In case of prompt-only data, if the last role is `"user"`, the generation prompt is added to the prompt.
        Else, if the last role is `"assistant"`, the final message is continued.

    Example:

    ```python
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
    >>> example = {
    ...     "prompt": [{"role": "user", "content": "What color is the sky?"}],
    ...     "completion": [{"role": "assistant", "content": "It is blue."}]
    ... }
    >>> apply_chat_template(example, tokenizer)
    {'prompt': '<|user|>\nWhat color is the sky?<|end|>\n<|assistant|>\n', 'completion': 'It is blue.<|end|>\n<|endoftext|>'}
    ```
    """
    if is_conversational(example):
        return apply_chat_template(example, tokenizer, tools)
    else:
        return example
    
def apply_chat_template(
    example: dict[str, list[dict[str, str]]],
    tokenizer: PreTrainedTokenizerBase,
    tools: Optional[list[Union[dict, Callable]]] = None,
) -> dict[str, str]:
    r"""
    Apply a chat template to a conversational example along with the schema for a list of functions in `tools`.

    For more details, see [`maybe_apply_chat_template`].
    """
    # Check that the example has the correct keys
    supported_keys = ["prompt", "a1","a2"]
    example_keys = {key for key in example.keys() if key in supported_keys}
    if example_keys not in [
        {"prompt", "a1", "a2"},  # DRPO paired responses
    ]:
        raise KeyError(f"Invalid keys in the example: {example_keys}")


    # Apply the chat template to the prompt, adding the generation prompt
    if "prompt" in example:
        last_role = example["prompt"][-1]["role"]
        if last_role == "user":
            add_generation_prompt = True
            continue_final_message = False
        elif last_role == "assistant":
            add_generation_prompt = False
            continue_final_message = True
        else:
            raise ValueError(f"Invalid role in the last message: {last_role}")
        
        prompt = tokenizer.apply_chat_template(
            example["prompt"],
            tools=tools,
            continue_final_message=continue_final_message,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        prompt_a1 = tokenizer.apply_chat_template(
                example["prompt"] + example["a1"], tools=tools, tokenize=False)
        
        a1 = prompt_a1[len(prompt) :]
        prompt_a2 = tokenizer.apply_chat_template(
                example["prompt"] + example["a2"], tools=tools, tokenize=False
            )
        a2 = prompt_a2[len(prompt) :]

    # Ensure that the prompt is the initial part of the prompt-completion string
    if "prompt" in example:
        error_message = (
            "The chat template applied to the prompt + completion does not start with the chat template applied to "
            "the prompt alone. This can indicate that the chat template is not supported by TRL."
            "\n**Prompt**:\n{}\n\n**Prompt + Completion**:\n{}"
        )
        if "chosen" in example and not prompt_a1.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_a1))
        if "rejected" in example and not prompt_a2.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_a2))

    # Extract the completion by removing the prompt part from the prompt-completion string
    output = {}

    if "prompt" in example:
        output["prompt"] = prompt
    if "a1" in example:
        output["a1"] = a1
    if "a2" in example:
        output["a2"] = a2
    # if "rank" in example:
    #     output["rank"] = example["rank"]

    return output    

class DRPOTrainer(Trainer):
    
    """
    Initialize the DRPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`): 
            The model to be trained, preferably an `AutoModelForCausalLM`.
        ref_model (`transformers.PreTrainedModel` or `torch.nn.Module`): 
            The reference model to be used for the KL divergence term.
        preference_model (`transformers.PreTrainedModel` or `torch.nn.Module`): 
            The preference model to be used for the preference score term.
        args (`DRPOConfig`): 
            The training arguments.
        data_collator (`DataCollator`): 
            The data collator to be used for training. defaults to `DataCollatorDRPO`.
        train_dataset (`datasets.Dataset`)
            Dataset should contain 4 columns: "prompt", "a1", "a2", and "rank", all in string format.
        eval_dataset (`datasets.Dataset`)
        processing_class (`ProcessingClass`): 
            The processing class to be used for tokenization.
            If provided, will be used to automatically process the inputs for the model, 
            and it will be saved along the model to make it easier to rerun an interrupted training or reuse the fine-tuned model.
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction`
            and return a dictionary string to metric values.
        callbacks (`list[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
    """

    _tag_names = ["trl", "drpo"]

    def __init__(
            self,
            model: PreTrainedModel,
            ref_model: Union[PreTrainedModel, nn.Module],
            preference_model: Union[PreTrainedModel, nn.Module],
            args: DRPOConfig,
            data_collator: Optional[DataCollatorDRPO] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            processing_class: Optional[Union[PreTrainedTokenizerBase]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Optional[Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]] = (None, None)
        ) -> None:
        
        if ref_model is model:
            raise ValueError("The reference model cannot be the same as the model.")         
        self.ref_model = ref_model
        self.ref_model.eval()

        if preference_model is None:
            raise ValueError("The preference model cannot be None.")
        self.preference_model = preference_model

        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        if data_collator is None:
            if processing_class is None:
                raise ValueError("If data_collator is None, processing_class must be provided.")
            data_collator = DataCollatorDRPO(pad_token_id=processing_class.pad_token_id)

        self.max_length = args.max_length
        self.precompute_preference_score = args.precompute_preference_score


        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in Online DPO, the sampled data does not include
        # the "input_ids" key. As a result, the trainer issues the warning: "Could not estimate the number of tokens
        # of the input, floating-point operations will not be computed." To suppress this warning, we set the
        # "estimate_tokens" key in the model's "warnings_issued" dictionary to True. This acts as a flag to indicate
        # that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Dataset preparation
        train_dataset = self._prepare_dataset(train_dataset, processing_class, args, "train")
        if eval_dataset is not None:
            if isinstance(eval_dataset, dict):
                eval_dataset = {
                    key: self._prepare_dataset(dataset, processing_class, args, key)
                    for key, dataset in eval_dataset.items()
                }
            else:
                eval_dataset = self._prepare_dataset(eval_dataset, processing_class, args, "eval")

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.generate_temperature,
            top_k=50,
            top_p=1.0,
            do_sample=True,
            use_cache=False if args.gradient_checkpointing else True,
        )


        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        self._beta = args.beta
        if self.is_deepspeed_enabled:
            if self.preference_model is not None:
                self.preference_model = prepare_deepspeed(
                    self.preference_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
            if self.ref_model is not None:
                self.ref_model = prepare_deepspeed(
                    self.ref_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
        else:
            if self.ref_model is not None:
                self.ref_model = self.ref_model.to(self.accelerator.device)
            if self.preference_model is not None:
                self.preference_model = self.preference_model.to(self.accelerator.device)

        self.stats = {
            "beta": [],
            "objective/kl": [],
            "objective/loss": [],
            "rank": [],
        }
        

        
        if not args.loss2_only:
            self.stats["objective/loss1"] = []
            self.stats["logps/a1"] = []
            self.stats["logps/a1_ref"] = []
            self.stats["ps/a1"] = []
            self.stats["is_ratio"] = []
            if args.ratio_processing == "clip":
            # self.stats["objective/loss1_clipped"] = []
                self.stats["clipped_ratio"] = []
        if not args.loss1_only:
            self.stats["logps/a*"] = []
            self.stats["logps/a*_ref"] = []
            self.stats["ps/a*"] = []
            self.stats["objective/loss2"] = []


    @property
    def beta(self):
        if isinstance(self._beta, list):
            epoch = self.state.epoch
            return self._beta[epoch] if epoch < len(self._beta) else self._beta[-1]
        else:
            return self._beta

    @staticmethod
    def tokenize_row(feature, 
                     processing_class: PreTrainedTokenizerBase, 
                     max_prompt_length: Union[int, None] = None, 
                     max_completion_length: Union[int, None] = None, 
                     add_special_tokens_for_prompt: bool = True,
                     eos_after_completion: bool = True) -> dict[str, Any]:
        """Tokenize a row of data."""
        # FIXME: the logic of whether to add special tokens is not clear
        # FIXME: whether to add attention mask is not clear
        
        prompt_ids = processing_class(feature["prompt"], add_special_tokens=False)["input_ids"]
        a1_ids = processing_class(feature["a1"], add_special_tokens=False)["input_ids"]
        a2_ids = processing_class(feature["a2"], add_special_tokens=False)["input_ids"]

        # add speical tokens
        if add_special_tokens_for_prompt:
            if processing_class.bos_token_id is not None:
                prompt_ids = [processing_class.bos_token_id] + prompt_ids
            if processing_class.eos_token_id is not None:
                prompt_ids = prompt_ids + [processing_class.eos_token_id]
        
        # 2 completions must add eos token to avoid non-stopping generation
        if eos_after_completion and processing_class.eos_token_id is not None:
            a1_ids = a1_ids + [processing_class.eos_token_id]
            a2_ids = a2_ids + [processing_class.eos_token_id]

        # truncation
        if max_prompt_length is not None:
            prompt_ids = prompt_ids[-max_prompt_length:] #right truncation
        if max_completion_length is not None:
            a1_ids = a1_ids[:max_completion_length]
            a2_ids = a2_ids[:max_completion_length] # left truncation

        return {
            "prompt_ids": prompt_ids,
            "a1_ids": a1_ids,
            "a2_ids": a2_ids
        }
    
    def _prepare_dataset(self, dataset: Union[Dataset, IterableDataset], processing_class: Union[PreTrainedTokenizerBase],
                         args: DRPOConfig, dataset_name: str,) -> Union[Dataset, IterableDataset]:
        map_kwargs = {"writer_batch_size": 10}
        if isinstance(dataset, Dataset):
            map_kwargs["num_proc"] = args.dataset_num_proc
        
        with PartialState().local_main_process_first():
            # Extract prompt if needed
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
            dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

            # Apply the chat template if needed
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
            dataset = dataset.map(
                maybe_apply_chat_template, fn_kwargs={"tokenizer": processing_class, "tools": args.tools}, **map_kwargs
            )
            print(f"after chat template dataset sample: {dataset[0]}")
            # Tokenize the dataset
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            dataset = dataset.map(
                self.tokenize_row,
                remove_columns=["prompt","a1","a2"],
                fn_kwargs={
                    "processing_class": processing_class,
                    "max_prompt_length": args.max_prompt_length,
                    "max_completion_length": args.max_completion_length,
                    # for enc-dec, we add the special tokens ([bos_token] + prompt + [eos_token]; completion + [eos_token])
                    "add_special_tokens_for_prompt": False,
                    "eos_after_completion": args.eos_after_completion,
                },
                **map_kwargs,
            )

        return dataset
    
    
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In DPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by `DataCollatorForPreference`, hence the override.
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt_ids",
                "a1_ids",
                "a2_ids",
                "prompt_attention_mask",
                "a1_attention_mask",
                "a2_attention_mask",
                "rank",
            ]

    
    @wraps(Trainer.get_train_dataloader)
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    

    @wraps(Trainer.get_eval_dataloader)
    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)
    
    def _generate(self, model, prompt_ids: torch.tensor, prompt_attention_mask: torch.tensor, num_astar:int = 1):
        eos_token_id = self.processing_class.eos_token_id
        pad_token_id = self.processing_class.pad_token_id
        prompt_ids = prompt_ids.repeat(num_astar, 1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_astar, 1)
        with unwrap_model_for_generation(model, self.accelerator, gather_deepspeed3_params=False) as unwrapped_model:
            output = unwrapped_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
                generation_config = self.generation_config,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

        completion_ids = output[:, prompt_ids.shape[1]:]
        completion_ids, completion_attention_mask = truncate_right(completion_ids, eos_token_id, pad_token_id)

        return prompt_ids, prompt_attention_mask, completion_ids, completion_attention_mask
    
    def _forward(self, model, prompt_ids, prompt_attention_mask, completion_ids, completion_attention_mask, temperature=1.0):
        # Get the number of tokens to truncate from prompt
        num_tokens_to_truncate = max(prompt_ids.size(1) + completion_ids.size(1) - self.max_length, 0)

        # Truncate left to avoid oom
        prompt_ids = prompt_ids[:, num_tokens_to_truncate:]
        prompt_attention_mask = prompt_attention_mask[:, num_tokens_to_truncate:]

        # Concat the prompt and completion
        prompt_completion_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        prompt_completion_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)

        # Get the logps of the completions from the model
        output = model(prompt_completion_ids, attention_mask=prompt_completion_mask)

        # There is 1 offset, because the model predict the next token
        logits = output.logits[:, max(0, prompt_ids.size(1) - 1) : -1]
        logits /= temperature + 1e-7
        # Take the completion tokens logp
        logps = selective_log_softmax(logits, completion_ids)
        # logps = torch.take_along_dim(logits.log_softmax(dim=-1), completion_ids.unsqueeze(-1), dim=2).squeeze(-1)
        return logps
    
    def training_step(self, model:nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int]=None) -> torch.Tensor:
        model.train()
        args = self.args

        if args.model_and_preference_share_basemodel:
            # then we don't need to decode and re-tokenize the generated samples to feed in preference model
            batch_size = inputs["prompt_ids"].size(0)
            prompt_ids = inputs["prompt_ids"]
            prompt_attention_mask = inputs["prompt_attention_mask"]
            a1_ids = inputs["a1_ids"]
            a1_attention_mask = inputs["a1_attention_mask"]
            a2_ids = inputs["a2_ids"]
            a2_attention_mask = inputs["a2_attention_mask"]
            rank = inputs["rank"].float()

            per_token_logps = self._forward(model, prompt_ids, prompt_attention_mask, a1_ids, a1_attention_mask, temperature=self.args.forward_temperature)
            prompt_ids_repeated, prompt_attention_mask_repeated, astar_ids, astar_attention_mask = self._generate(model, prompt_ids, prompt_attention_mask, self.args.num_astar)
            contain_eos_token = torch.any(astar_ids == self.processing_class.eos_token_id, dim=-1)

            per_token_logps_star = self._forward(model, prompt_ids_repeated, prompt_attention_mask_repeated, astar_ids, astar_attention_mask, temperature=self.args.generate_temperature)

            with torch.no_grad():
                if not args.loss2_only:
                    per_token_ref_logps = self._forward(self.ref_model, prompt_ids, prompt_attention_mask, a1_ids, a1_attention_mask, temperature=self.args.forward_temperature)
                per_token_ref_logps_star = self._forward(self.ref_model, prompt_ids_repeated, prompt_attention_mask_repeated, astar_ids, astar_attention_mask, temperature=self.args.generate_temperature)

                # Compute preference score g(y*, y', x) and g(y, y', x)
            with torch.inference_mode():
                prompt_a2_ids = torch.cat((prompt_ids, a2_ids), dim=1)
                prompt_a2_attention_mask = torch.cat((prompt_attention_mask, a2_attention_mask), dim=1)
                if not args.loss1_only:
                    prompt_astar_ids = torch.cat((prompt_ids_repeated, astar_ids), dim=1)
                    prompt_astar_attention_mask = torch.cat((prompt_attention_mask_repeated, astar_attention_mask), dim=1)
                    prompt_a2_repeated_ids = prompt_a2_ids.repeat(self.args.num_astar, 1)
                    prompt_a2_repeated_attention_mask = prompt_a2_attention_mask.repeat(self.args.num_astar, 1)
                    preference_score_star = get_preference_score_without_decoding(
                    self.preference_model, 
                    prompt_astar_ids,
                    prompt_astar_attention_mask, 
                    prompt_a2_repeated_ids,
                    prompt_a2_repeated_attention_mask,
                    is_bt_model = self.args.is_bt_model,
                    kwargs=self.args.preference_model_kwargs or {}
                )
                    if self.args.missing_eos_penalty is not None:
                        preference_score_star[~contain_eos_token] -= self.args.missing_eos_penalty
                    # below is for debugging
                    # generated_examples = self.processing_class.batch_decode(prompt_astar_ids, skip_special_tokens=True)
                    # past_examples = self.processing_class.batch_decode(a2_ids, skip_special_tokens=True)
                    # print("\033[42mgenerated_examples:\033[0m", generated_examples[0].replace("user", "\033[32muser\033[0m").replace("assistant", "\033[35massistant\033[0m"))
                    # print("\033[43mreference_examples:\033[0m", past_examples[0].replace("user", "\033[32muser\033[0m").replace("assistant", "\033[35massistant\033[0m"))
                    # print("\033[46mpreference_score_star:\033[0m ", preference_score_star)
                    del (prompt_astar_ids, prompt_a2_repeated_ids, prompt_astar_attention_mask, prompt_a2_repeated_attention_mask)
                if not args.loss2_only:
                    prompt_a1_ids = torch.cat((prompt_ids, a1_ids), dim=1)
                    prompt_a1_attention_mask = torch.cat((prompt_attention_mask, a1_attention_mask), dim=1)

                    preference_score = get_preference_score_without_decoding(
                        self.preference_model, 
                        prompt_a1_ids,
                        prompt_a1_attention_mask,
                        prompt_a2_ids,
                        prompt_a2_attention_mask,
                        is_bt_model = self.args.is_bt_model,
                        kwargs=self.args.preference_model_kwargs or {}
                    )
                    del (prompt_a1_ids, prompt_a1_attention_mask)
            
            del (prompt_a2_ids, prompt_a2_attention_mask)

            # compute kl divergence
            kl_onpolicy_part = ((torch.exp(per_token_ref_logps_star - per_token_logps_star) - (per_token_ref_logps_star - per_token_logps_star) - 1)*astar_attention_mask).sum(-1)
            mean_kl = kl_onpolicy_part.mean()
            if not args.loss1_only:
                logps_star = (per_token_logps_star * astar_attention_mask).sum(-1)
                loss2 = -(logps_star * preference_score_star.clone().detach()).mean()
                
            if not args.loss2_only:
                logps = (per_token_logps * a1_attention_mask).sum(1)
                ref_logps = (per_token_ref_logps * a1_attention_mask).sum(1)

                ratio = torch.exp(logps - ref_logps)
                clipped_ratio = torch.clamp(ratio, min = 1. / self.args.clipbound, max = self.args.clipbound)
                losses1 = - clipped_ratio.detach() * (rank - preference_score.clone()).detach() * logps

            if args.loss1_only:
                loss = losses1.mean() + self.beta * mean_kl
            elif args.loss2_only:
                loss = loss2 + self.beta * mean_kl
            else:
                loss = losses1.mean() + loss2 + self.beta * mean_kl

        else:
            batch_size = inputs["prompt_ids"].size(0)
            prompt_ids = inputs["prompt_ids"]
            prompt_attention_mask = inputs["prompt_attention_mask"]
            a1_ids = inputs["a1_ids"]
            a1_attention_mask = inputs["a1_attention_mask"]
            a2_ids = inputs["a2_ids"]
            a2_attention_mask = inputs["a2_attention_mask"]
            rank = inputs["rank"].float()


            # log pi(y|x) shape(batch_size, 1)
            per_token_logps = self._forward(model, prompt_ids, prompt_attention_mask, a1_ids, a1_attention_mask, temperature=self.args.forward_temperature)

            # sample y* for `num_astar` times
            prompt_ids_repeated, prompt_attention_mask_repeated, astar_ids, astar_attention_mask = self._generate(model, prompt_ids, prompt_attention_mask, self.args.num_astar)
            contain_eos_token = torch.any(astar_ids == self.processing_class.eos_token_id, dim=-1)

            # log pi(y*|x) shape(num_astar*batch_size, 1)
            per_token_logps_star = self._forward(model, prompt_ids_repeated, prompt_attention_mask_repeated, astar_ids, astar_attention_mask, temperature=self.args.generate_temperature)


            with torch.no_grad():
                if self.ref_model is not None:
                    # log pi_ref(y|x)
                    per_token_ref_logps = self._forward(self.ref_model, prompt_ids, prompt_attention_mask, a1_ids, a1_attention_mask, temperature=self.args.forward_temperature)
                    per_token_ref_logps_star = self._forward(self.ref_model, prompt_ids_repeated, prompt_attention_mask_repeated, astar_ids, astar_attention_mask, temperature=self.args.generate_temperature)
                else:
                    raise NotImplementedError("Peft is not implemented yet and ref model should be specified.")

            # device = logps.device

            # Compute preference score g(y*, y', x) and g(y, y', x)
            with torch.inference_mode():

                prompt_astar_ids = torch.cat((prompt_ids_repeated, astar_ids), dim=1)
                prompt_a2_ids = torch.cat((prompt_ids, a2_ids), dim=1)

                prompt_astar = self.processing_class.batch_decode(prompt_astar_ids, skip_special_tokens=True)
                print("\033[42mprompt_astar:\033[0m", prompt_astar[0])
                prompt_a2 = self.processing_class.batch_decode(prompt_a2_ids, skip_special_tokens=True)
                print("\033[43mprompt_a2:\033[0m",prompt_a2[0])
                prompt_a2_repeated = prompt_a2 * self.args.num_astar
                assert(len(prompt_astar) == len(prompt_a2_repeated))
                
                # g(y*, y', x)
                preference_score_star= get_preference_score(
                    self.preference_model, 
                    prompt_astar, 
                    prompt_a2_repeated,
                    is_bt_model = self.args.is_bt_model,
                    noisy = 0.2,
                    kwargs=self.args.preference_model_kwargs or {}
                )
                
                if args.missing_eos_penalty is not None:
                    preference_score_star[~contain_eos_token] -= self.args.missing_eos_penalty

                
                if not self.precompute_preference_score:
                    # g(y, y', x)            
                    prompt_a1_ids = torch.cat((prompt_ids, a1_ids), dim=1)
                    prompt_a1 = self.processing_class.batch_decode(prompt_a1_ids, skip_special_tokens=False)
                    assert(len(prompt_a1) == len(prompt_a2))
                    
                    preference_score = get_preference_score(
                        self.preference_model, 
                        prompt_a1, 
                        prompt_a2,
                        is_bt_model = self.args.is_bt_model,
                        noisy = 0.2,
                        kwargs = self.args.preference_model_kwargs or {}
                    )
                else:
                    raise NotImplementedError("precompute_preference_score is not implemented yet.")
                
                print("\033[46mpreference_score_star:\033[0m", preference_score_star[0].item())
                
                del prompt_astar_ids, prompt_a2_ids, prompt_astar, prompt_a2, prompt_a2_repeated, prompt_a1_ids, prompt_a1

            # Compute the loss part two
            assert per_token_logps_star.size(0) == batch_size * self.args.num_astar
        
            logps_star = (per_token_logps_star * astar_attention_mask).sum(-1)
            loss2 = -(logps_star * (preference_score_star.clone().detach() - 0.5 * torch.ones_like(logps_star))).mean()

            kl_onpolicy_part = ((torch.exp(per_token_ref_logps_star - per_token_logps_star) - (per_token_ref_logps_star - per_token_logps_star) - 1)*astar_attention_mask).sum(-1)
            mean_kl = kl_onpolicy_part.mean()

            # Compute the loss part one
            logps = (per_token_logps * a1_attention_mask).sum(1)
            ref_logps = (per_token_ref_logps * a1_attention_mask).sum(1)
            
            if args.ratio_processing == "clip":
                ratio = torch.exp(logps - ref_logps)

                clipped_ratio = torch.clamp(ratio, min = 1. / self.args.clipbound, max = self.args.clipbound)
                losses1 =  - clipped_ratio.detach() * (rank - preference_score.clone()).detach() * logps

            elif args.ratio_processing == "self_normalize":
                ratio_nominator = torch.exp(logps) / torch.exp(logps).mean()
                ratio_denominator = torch.exp(ref_logps) / torch.exp(ref_logps).mean()
                ratio = ratio_nominator / ratio_denominator
                losses1 = - ratio.detach() * (rank - preference_score.clone()).detach() * logps

            else:
                ratio = torch.exp(logps - ref_logps)
                losses1 = -ratio.detach() * (rank - preference_score.clone()).detach() * logps
            
            if args.loss2_only:
                loss = loss2 + self.beta * mean_kl
            elif args.loss1_only:
                losses1 = -clipped_ratio.detach() * rank.detach() * logps
                loss = losses1.mean() + self.beta * mean_kl
            else:
                loss = losses1.mean() + loss2 + self.beta * mean_kl
            

        # log everything
        self.stats['beta'].append(self.beta)
        self.stats['objective/kl'].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        if not self.args.loss2_only:
            self.stats['objective/loss1'].append(self.accelerator.gather_for_metrics(losses1).mean().item())
            self.stats["logps/a1"].append(self.accelerator.gather_for_metrics(logps).mean().item())
            self.stats['logps/a1_ref'].append(self.accelerator.gather_for_metrics(ref_logps).mean().item())
            self.stats['ps/a1'].append(self.accelerator.gather_for_metrics(preference_score).mean().item()) # preference score
            self.stats['is_ratio'].append(self.accelerator.gather_for_metrics(ratio.mean()).mean().item())
            if self.args.ratio_processing == "clip":
                self.stats['clipped_ratio'].append(self.accelerator.gather_for_metrics(clipped_ratio).mean().item())
        if not self.args.loss1_only:
            self.stats['objective/loss2'].append(self.accelerator.gather_for_metrics(loss2).item())
            self.stats['logps/a*'].append(self.accelerator.gather_for_metrics(logps_star).mean().item())
            self.stats['logps/a*_ref'].append(self.accelerator.gather_for_metrics(per_token_ref_logps_star*astar_attention_mask).sum(-1).mean().item())
            self.stats['ps/a*'].append(self.accelerator.gather_for_metrics(preference_score_star).mean().item()) # preference score
        self.stats['objective/loss'].append(self.accelerator.gather_for_metrics(loss).mean().item())
        self.stats['rank'].append(self.accelerator.gather_for_metrics(rank).mean().item())


        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        
        loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss, **kwargs)

        return loss.detach()
    
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time=None, learning_rate=None):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            # Add our metrics
            for key, val in self.stats.items():
                logs[key] = sum(val) / len(val)
            self.stats = {key: [] for key in self.stats}  # reset stats

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
                self.log(logs, start_time)
            else:  # transformers<=4.46
                self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == "best":
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    # Copy-pasted from transformers.Trainer to maintain compatibility with earlier versions.
    # This can be removed once the minimum transformers version is updated to 4.47.
    # Refer to https://github.com/huggingface/trl/pull/2288 for more details.
    def _determine_best_metric(self, metrics, trial):
        """
        Determine if the model should be saved based on the evaluation metrics.
        If args.metric_for_best_model is not set, the loss is used.
        Returns:
            bool: True if a new best metric was found, else False
        """
        is_new_best_metric = False

        if self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model

            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"

            try:
                metric_value = metrics[metric_to_check]
            except KeyError as exc:
                raise KeyError(
                    f"The `metric_for_best_model` training argument is set to '{metric_to_check}', which is not found in the evaluation metrics. "
                    f"The available evaluation metrics are: {list(metrics.keys())}. Consider changing the `metric_for_best_model` via the TrainingArguments."
                ) from exc

            operator = np.greater if self.args.greater_is_better else np.less

            if self.state.best_metric is None:
                self.state.best_metric = float("-inf") if self.args.greater_is_better else float("inf")

            if operator(metric_value, self.state.best_metric):
                run_dir = self._get_output_dir(trial=trial)
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                output_dir = os.path.join(run_dir, checkpoint_folder)
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

                is_new_best_metric = True

        return is_new_best_metric
    
    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="drpo",
        )
        model_card.save(os.path.join(self.args.output_dir, "README.md"))