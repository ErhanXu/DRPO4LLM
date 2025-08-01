import os
import textwrap
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from packaging import version
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.trainer_utils import EvalPrediction, seed_worker
from transformers.utils import is_peft_available

# Import parent class and utilities
from trl import OnlineDPOTrainer, BasePairwiseJudge
from trl.data_utils import is_conversational
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import (
    SIMPLE_CHAT_TEMPLATE,
    empty_cache,
    generate_model_card,
    get_comet_experiment_url,
    get_reward,
    pad,
    selective_log_softmax,
    truncate_right
)

import wandb
import swanlab

from .drpo_config import DRPOConfig

if is_peft_available():
    from peft import PeftModel

if is_peft_available():
    from peft import PeftModel

class DRPODataCollatorWithPadding:
    """
    Data collator for DRPO training.
    
    Args:
        pad_token_id: Token ID used for padding sequences
        is_encoder_decoder: Whether the model is encoder-decoder architecture
    """
    
    def __init__(
        self,
        pad_token_id: int,
        is_encoder_decoder: bool = False,
    ):
        self.pad_token_id = pad_token_id
        self.is_encoder_decoder = is_encoder_decoder
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate features into padded tensors.
        
        We don't need labels because DRPO computes preference-based losses,
        not supervised token prediction losses.
        """
        # Extract components
        prompt_ids = [torch.tensor(f["prompt_ids"], dtype=torch.long) for f in features]
        chosen_ids = [torch.tensor(f["chosen_ids"], dtype=torch.long) for f in features]
        rejected_ids = [torch.tensor(f["rejected_ids"], dtype=torch.long) for f in features]
        
        # Create attention masks
        prompt_attention_mask = [torch.ones_like(ids) for ids in prompt_ids]
        chosen_attention_mask = [torch.ones_like(ids) for ids in chosen_ids]
        rejected_attention_mask = [torch.ones_like(ids) for ids in rejected_ids]
        
        # Pad sequences
        batch = {
            "prompt_ids": pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left"),
            "prompt_attention_mask": pad(prompt_attention_mask, padding_value=0, padding_side="left"),
            "chosen_ids": pad(chosen_ids, padding_value=self.pad_token_id, padding_side="right"),
            "chosen_attention_mask": pad(chosen_attention_mask, padding_value=0, padding_side="right"),
            "rejected_ids": pad(rejected_ids, padding_value=self.pad_token_id, padding_side="right"),
            "rejected_attention_mask": pad(rejected_attention_mask, padding_value=0, padding_side="right"),
        }
        
        # Include original text if available (for judge-based evaluation)
        if "prompt" in features[0]:
            batch["prompt"] = [f["prompt"] for f in features]
            batch["chosen"] = [f["chosen"] for f in features]
            batch["rejected"] = [f["rejected"] for f in features]
        
        return batch


class DRPOTrainer(OnlineDPOTrainer):
    """
    Doubly Robust Preference Optimization (DRPO) Trainer.
    
    This trainer implements the DRPO algorithm which provides robustness to misspecification
    of either the reference policy or the preference model, achieving better convergence
    properties than standard DPO or PPO-based methods.
    
    The algorithm combines:
    1. Direct Method (DM): Uses preference model to estimate expected preferences
    2. Importance Sampling (IS): Corrects for distribution mismatch between policies
    
    Key features:
    - Supports multiple preference models (reward models, custom preference models, judges)
    - Compatible with vLLM for efficient generation
    - Handles multi-GPU training with DeepSpeed ZeRO-2/3
    - Provides flexible IS ratio control mechanisms
    """
    
    _tag_names = ["trl", "drpo"]
    
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        ref_model: Union[PreTrainedModel, nn.Module, None] = None,
        reward_model: Union[PreTrainedModel, nn.Module, None] = None,
        judge: Optional[BasePairwiseJudge] = None,
        args: Optional[DRPOConfig] = None,
        data_collator: Optional[Callable] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        peft_config: Optional[dict] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        # Store model configuration before potential DeepSpeed wrapping
        self._is_encoder_decoder = getattr(model.config, 'is_encoder_decoder', False) if hasattr(model, 'config') else False
        
        # Initialize preference model if specified
        self.preference_model = None
        if args and args.use_preference_model and args.preference_model_path:


            # Import custom preference model
            from .drpo_utils_new import GPMwithRewardNetwork
                # GPM or BT preference model
            self.preference_model = GPMwithRewardNetwork(
                model_name_or_path=args.preference_model_path,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                pad_token_id=processing_class.pad_token_id if processing_class else 0,
                is_general_preference=(args.preference_model_type == "general"),
                bf16=args.bf16
            )
            # Don't use standard reward model when using custom preference model
            reward_model = self.preference_model
        
        # Create data collator if not provided
        if data_collator is None:
            data_collator = DRPODataCollatorWithPadding(
                pad_token_id=processing_class.pad_token_id if processing_class else 0,
                is_encoder_decoder=self._is_encoder_decoder,
            )
        
        # Initialize parent class
        super().__init__(
            model=model,
            ref_model=ref_model,
            reward_model=reward_model,
            judge=judge,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            peft_config=peft_config,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if self.train_dataset is not None and "prompt_ids" not in self.train_dataset.column_names:
            self.train_dataset = self._prepare_dataset(self.train_dataset, self.processing_class)
        
        if self.eval_dataset is not None:
            if isinstance(self.eval_dataset, dict):
                # Handle dict of datasets
                self.eval_dataset = {
                    key: self._prepare_dataset(dataset, self.processing_class) 
                    if "prompt_ids" not in dataset.column_names else dataset
                    for key, dataset in self.eval_dataset.items()
                }
            else:
                # Single eval dataset
                if "prompt_ids" not in self.eval_dataset.column_names:
                    self.eval_dataset = self._prepare_dataset(self.eval_dataset, self.processing_class)
        
        # Handle preference model with distributed training
        if self.preference_model is not None:
            if self.is_deepspeed_enabled:
                from trl.trainer.utils import prepare_deepspeed
                self.preference_model = prepare_deepspeed(
                    self.preference_model,
                    args.per_device_train_batch_size,
                    args.fp16,
                    args.bf16
                )
            else:
                self.preference_model = self.preference_model.to(self.accelerator.device)

        # Initialize DRPO-specific statistics
        self.stats = {
            # Core DRPO metrics
            "drpo/term_dm": [],  # Direct method term
            "drpo/term_is": [],  # Importance sampling term
            "drpo/preference_score": [],  # g(chosen, rejected)
            
            # Generated sample quality metrics
            "generated/vs_rejected_mean": [],  # E[g(mc, rejected)]
            "generated/vs_rejected_std": [],   # Std[g(mc, rejected)]
            "generated/vs_chosen_mean": [],    # E[g(mc, chosen)]
            "generated/vs_chosen_std": [],     # Std[g(mc, chosen)]
            
            # Generated sample preference margins
            "generated/margin_over_rejected": [],  # E[g(mc, rejected) - 0.5]
            "generated/margin_over_chosen": [],    # E[0.5 - g(mc, chosen)]
            "generated/win_rate_vs_rejected": [], # P(g(mc, rejected) > 0.5)
            "generated/win_rate_vs_chosen": [],   # P(g(mc, chosen) > 0.5)
            
            # Generation quality indicators
            "generated/contains_eos_rate": [],     # Fraction with EOS token
            "generated/avg_length": [],            # Average length of generated samples
            "generated/length_std": [],            # Std of lengths
            
            # IS ratio statistics
            "is_ratio/chosen_mean": [],
            "is_ratio/chosen_std": [],
            "is_ratio/chosen_max": [],
            "is_ratio/rejected_mean": [],
            "is_ratio/rejected_std": [],
            "is_ratio/rejected_max": [],
            "is_ratio/clip_rate_chosen": [],      # How often we hit the clip bounds
            "is_ratio/clip_rate_rejected": [],
            
            # Policy statistics
            "logps/chosen": [],
            "logps/rejected": [],
            "logps/generated_mean": [],           # E[log π(mc)]
            # "logps/generated_std": [],            # Std[log π(mc)]
            "logps/chosen_ref": [],  # log π_ref(chosen)
            "logps/rejected_ref": [],  # log π_ref(rejected)
            "logps/generated_ref_mean": [],  # E[log π_ref(mc)]
            "rewards/margins": [],                # log π(chosen) - log π(rejected)
            "rewards/accuracy": [],               # P(log π(chosen) > log π(rejected))
            
            # KL divergence metrics (computed from generated samples)
            "objective/kl": [],  # Mean KL[π||π_ref] from generated samples
            "objective/entropy": [],  # Entropy of generated samples
            # "objective/kl_per_token": [],  # Average per-token KL
            "beta": [],
            
            # Loss components
            "loss/drpo": [],                      # DRPO loss component
            "loss/kl": [],                        # KL loss component
            "loss/total": [],                     # Total loss
        }
    
    def _prepare_dataset(self, dataset: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset:
        """
        Prepare dataset for DRPO training by tokenizing prompts, chosen, and rejected responses.
        
        Args:
            dataset: Raw dataset with text examples
            tokenizer: Tokenizer for processing text
            
        Returns:
            Tokenized dataset ready for training
        """
        def tokenize_row(example):
            # Extract and process prompt
            prompt = example["prompt"]
            if isinstance(prompt, list):
                # Conversational format - apply chat template
                prompt_text = tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt_text = prompt
            
            # Extract chosen and rejected
            chosen_text = example["chosen"]
            rejected_text = example["rejected"]
            
            # Apply chat template to responses if conversational
            if isinstance(example["chosen"], list):
                full_chosen = tokenizer.apply_chat_template(
                    example["prompt"] + example["chosen"],
                    tokenize=False,
                )
                chosen_text = full_chosen[len(prompt_text):]
            
            if isinstance(example["rejected"], list):
                full_rejected = tokenizer.apply_chat_template(
                    example["prompt"] + example["rejected"],
                    tokenize=False,
                )
                rejected_text = full_rejected[len(prompt_text):]
            
            # Tokenize
            prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)
            prompt_ids = prompt_tokens["input_ids"]
            
            # Add BOS token if needed
            if tokenizer.bos_token_id is not None and (
                len(prompt_ids) == 0 or prompt_ids[0] != tokenizer.bos_token_id
            ):
                prompt_ids = [tokenizer.bos_token_id] + prompt_ids
            
            # Tokenize responses
            chosen_tokens = tokenizer(chosen_text, add_special_tokens=False)
            rejected_tokens = tokenizer(rejected_text, add_special_tokens=False)
            
            chosen_ids = chosen_tokens["input_ids"]
            rejected_ids = rejected_tokens["input_ids"]
            
            # Add EOS tokens
            if tokenizer.eos_token_id is not None:
                if len(chosen_ids) == 0 or chosen_ids[-1] != tokenizer.eos_token_id:
                    chosen_ids = chosen_ids + [tokenizer.eos_token_id]
                if len(rejected_ids) == 0 or rejected_ids[-1] != tokenizer.eos_token_id:
                    rejected_ids = rejected_ids + [tokenizer.eos_token_id]
            
            # Truncate if needed
            if self.args.max_prompt_length:
                prompt_ids = prompt_ids[-self.args.max_prompt_length:]
            if self.args.max_completion_length:
                chosen_ids = chosen_ids[:self.args.max_completion_length]
                rejected_ids = rejected_ids[:self.args.max_completion_length]
            
            return {
                "prompt_ids": prompt_ids,
                "chosen_ids": chosen_ids,
                "rejected_ids": rejected_ids,
                "prompt": prompt_text,
                "chosen": chosen_text,
                "rejected": rejected_text,
            }
        
        # Process dataset
        with PartialState().local_main_process_first():
            dataset = dataset.map(
                tokenize_row,
                num_proc=self.args.dataset_num_proc,
                desc="Tokenizing dataset for DRPO",
            )
        
        return dataset
    
    def _set_signature_columns_if_needed(self):
        """Set expected columns for data collator."""
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt_ids", "chosen_ids", "rejected_ids",
                "prompt", "chosen", "rejected"
            ]
    
    @wraps(OnlineDPOTrainer.get_train_dataloader)
    def get_train_dataloader(self) -> DataLoader:
        """Get train dataloader with DRPO-specific data handling."""
        if self.train_dataset is None:
            raise ValueError("Training requires a train_dataset.")
        
        # Prepare dataset if needed
        if "prompt_ids" not in self.train_dataset.column_names:
            self.train_dataset = self._prepare_dataset(self.train_dataset, self.processing_class)
        
        # Set signature columns for the trainer
        self._set_signature_columns_if_needed()
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        dataloader_params = {
            "batch_size": self._train_batch_size,
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

    @wraps(OnlineDPOTrainer.get_eval_dataloader)
    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        """Get eval dataloader with DRPO-specific data handling."""
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        # Determine which dataset to use
        if eval_dataset is not None:
            if isinstance(eval_dataset, str):
                eval_dataset = self.eval_dataset[eval_dataset]
        else:
            eval_dataset = self.eval_dataset
        
        # Prepare dataset if needed (tokenize it)
        if isinstance(eval_dataset, dict):
            # Handle dict of datasets (e.g., {"validation": dataset1, "test": dataset2})
            for key in eval_dataset:
                if "prompt_ids" not in eval_dataset[key].column_names:
                    eval_dataset[key] = self._prepare_dataset(
                        eval_dataset[key], self.processing_class
                    )
        else:
            # Single eval dataset
            if "prompt_ids" not in eval_dataset.column_names:
                eval_dataset = self._prepare_dataset(eval_dataset, self.processing_class)
        
        # Update the stored eval dataset
        self.eval_dataset = eval_dataset
        
        # Call parent's get_eval_dataloader
        return super().get_eval_dataloader(eval_dataset)
    
    def _compute_preference_scores_batch(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        ids_1: torch.Tensor,
        mask_1: torch.Tensor,
        ids_2: torch.Tensor,
        mask_2: torch.Tensor,
        texts_1: Optional[List[str]] = None,
        texts_2: Optional[List[str]] = None,
        prompt_texts: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compute preference scores g(y1, y2 | x) for a batch of comparisons.
        """
        batch_size = prompt_ids.shape[0]
        
        # Concatenate prompts with responses
        prompt_response_1 = torch.cat([prompt_ids, ids_1], dim=1)
        prompt_response_2 = torch.cat([prompt_ids, ids_2], dim=1)
        attention_mask_1 = torch.cat([prompt_mask, mask_1], dim=1)
        attention_mask_2 = torch.cat([prompt_mask, mask_2], dim=1)
        
        with torch.no_grad():
            if self.preference_model is not None:
                # Use custom preference model
                from .drpo_utils_new import get_preference_score_without_decoding
                
                scores = get_preference_score_without_decoding(
                    self.preference_model,
                    prompt_response_1,
                    attention_mask_1,
                    prompt_response_2,
                    attention_mask_2,
                    is_bt_model=(self.args.preference_model_type == "bt"),
                )
                
            elif self.reward_model is not None:
                # Use standard reward model with Bradley-Terry
                context_length = prompt_ids.shape[1]
                
                _, scores_1, _ = get_reward(
                    self.reward_model,
                    prompt_response_1,
                    self.processing_class.pad_token_id,
                    context_length
                )
                _, scores_2, _ = get_reward(
                    self.reward_model,
                    prompt_response_2,
                    self.processing_class.pad_token_id,
                    context_length
                )
                
                # Bradley-Terry model
                scores = torch.sigmoid(scores_1 - scores_2)
                
            elif self.judge is not None:
                # Use judge for evaluation
                if texts_1 is None or texts_2 is None:
                    # If texts not provided, decode from token IDs
                    texts_1 = self.processing_class.batch_decode(ids_1, skip_special_tokens=True)
                    texts_2 = self.processing_class.batch_decode(ids_2, skip_special_tokens=True)
                
                # Get prompt texts
                if prompt_texts is None:
                    prompt_texts = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
                
                # Handle conversational format if needed
                from trl.data_utils import is_conversational
                if prompt_texts and any(is_conversational({"prompt": p}) for p in prompt_texts):
                    import jinja2
                    env = jinja2.Environment()
                    template = env.from_string(SIMPLE_CHAT_TEMPLATE)
                    
                    # Apply chat template to prompts if they're conversational
                    formatted_prompts = []
                    for p in prompt_texts:
                        if is_conversational({"prompt": p}):
                            formatted_prompts.append(template.render(messages=p))
                        else:
                            formatted_prompts.append(p)
                    prompt_texts = formatted_prompts
                    
                    # Note: texts_1 and texts_2 are completions, not full conversations
                    # They should already be plain text, not conversational format
                
                # Judge returns preference probabilities
                scores = self.judge.judge(
                    prompt_texts,
                    list(zip(texts_1, texts_2)),
                    return_scores=True,
                )
                scores = torch.tensor(scores, device=prompt_ids.device, dtype=torch.float32)
                
            else:
                raise ValueError("No preference model, reward model, or judge available")
        
        return scores
    
    def _compute_is_ratio_controlled(
        self,
        logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        mask: torch.Tensor,
        prefix: str = ""
    ) -> torch.Tensor:
        """
        Compute importance sampling ratios with control mechanism.
        
        Currently implements clipping, but designed to be extended with other methods
        like adaptive clipping, trust regions, or variance reduction techniques.
        
        Args:
            logprobs: Log probabilities under current policy
            ref_logprobs: Log probabilities under reference policy
            mask: Attention mask
            prefix: Prefix for logging (e.g., "chosen" or "rejected")
            
        Returns:
            Controlled IS ratios
        """
        # Compute raw IS ratios
        log_ratios = (logprobs - ref_logprobs) * mask
        is_ratios = torch.exp(log_ratios.sum(dim=1))
        
        # Apply control method
        if self.args.is_control_method == "clip":
            controlled_ratios = torch.clamp(is_ratios, self.args.is_clip_min, self.args.is_clip_max).detach()
        # Room for other methods: adaptive_clip, trust_region, etc.
        else:
            controlled_ratios = torch.clamp(is_ratios, self.args.is_clip_min, self.args.is_clip_max)
        
        # Log statistics
        if prefix:
            self.stats[f"is_ratio/{prefix}_mean"].append(is_ratios.mean().item())
            self.stats[f"is_ratio/{prefix}_max"].append(is_ratios.max().item())
        
        return controlled_ratios
    
    def _compute_kl_divergence(
        self,
        logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        mask: torch.Tensor,
        kl_type: str = "k3"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute KL divergence KL[π||π_ref] using specified estimator.
        
        Args:
            logprobs: Log probabilities under current policy π
            ref_logprobs: Log probabilities under reference policy π_ref
            mask: Attention mask
            kl_type: "k1" or "k3"
            
        Returns:
            per_token_kl: Per-token KL divergence
            total_kl: Total KL divergence per sequence
        """
        if kl_type == "k1":
            # Standard unbiased estimator: log(π/π_ref)
            per_token_kl = (logprobs - ref_logprobs) * mask
            
        elif kl_type == "k3":
            # Lower variance unbiased estimator: (π_ref/π - 1) - log(π_ref/π)
            log_ratio = ref_logprobs - logprobs
            ratio = torch.exp(torch.clamp(log_ratio, -10, 10))
            per_token_kl = ((ratio - 1) - log_ratio) * mask
            
        else:
            raise ValueError(f"Unknown KL estimator: {kl_type}")
        
        total_kl = per_token_kl.sum(dim=1)
        return per_token_kl, total_kl
    
    def _forward(self, model, prompt_ids, prompt_attention_mask, 
                 completion_ids, completion_attention_mask):
        """Override to apply temperature scaling using instance temperature."""
        # Get the number of tokens to truncate from prompt
        num_tokens_to_truncate = max(
            prompt_ids.size(1) + completion_ids.size(1) - self.max_length, 0
        )
        
        # Truncate left to avoid OOM
        if num_tokens_to_truncate > 0:
            prompt_ids = prompt_ids[:, num_tokens_to_truncate:]
            prompt_attention_mask = prompt_attention_mask[:, num_tokens_to_truncate:]
        
        # Concatenate prompt and completion
        prompt_completion_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        prompt_completion_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
        
        # Get model outputs
        output = model(
            prompt_completion_ids,
            attention_mask=prompt_completion_mask,
            return_dict=True
        )
        
        # Extract logits for completion tokens
        logits = output.logits[:, max(0, prompt_ids.size(1) - 1):-1]
        
        # Apply temperature scaling using instance temperature
        logits = logits / (self.args.temperature + 1e-7)
        
        # Compute log probabilities
        logprobs = selective_log_softmax(logits, completion_ids)
        
        return logprobs
    
    def _generate(
        self, 
        model: nn.Module,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        num_samples: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate completions from tokenized prompts.
        
        Args:
            model: The model to generate with
            prompt_ids: Tokenized prompts [batch_size, seq_len]
            prompt_mask: Attention mask for prompts
            num_samples: Number of samples per prompt (for multiple generations)
            
        Returns:
            Tuple of (prompt_ids, prompt_mask, completion_ids, completion_mask)
        """
        eos_token_id = self.processing_class.eos_token_id
        pad_token_id = self.processing_class.pad_token_id
        
        # Handle multiple samples per prompt if needed
        if num_samples > 1:
            prompt_ids = prompt_ids.repeat(num_samples, 1)
            prompt_mask = prompt_mask.repeat(num_samples, 1)
        
        # Generate completions
        if self.args.use_vllm and hasattr(self, 'llm'):
            # vLLM generation (if enabled)
            # Convert token IDs back to text for vLLM
            prompts = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=False)
            
            # Use vLLM to generate
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                n=1,  # We already repeated prompts above
                max_tokens=self.args.max_new_tokens,
                temperature=self.generation_config.temperature,
                top_k=self.generation_config.top_k,
                top_p=self.generation_config.top_p,
            )
            
            outputs = self.llm.generate(prompts, sampling_params)
            
            # Convert back to token IDs
            completion_texts = [output.outputs[0].text for output in outputs]
            completions = self.processing_class(
                completion_texts,
                padding=True,
                truncation=True,
                max_length=self.args.max_new_tokens,
                return_tensors="pt"
            )
            completion_ids = completions.input_ids.to(prompt_ids.device)
            
            # Create attention masks
            completion_mask = (completion_ids != pad_token_id).long()
            
        else:
            # Standard transformers generation
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    generation_config=self.generation_config,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    do_sample=True,
                )
            
            # Extract only the generated tokens (remove prompt)
            completion_ids = output_ids[:, prompt_ids.size(1):]
            
            # Truncate completions and create masks

            completion_ids, completion_mask = truncate_right(
                completion_ids, eos_token_id, pad_token_id
            )
        
        return prompt_ids, prompt_mask, completion_ids, completion_mask
    
    
    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        """
        Perform DRPO training step with DeepSpeed ZeRO-3 compatibility.
        """
        model.train()
        
        # Move inputs to device
        device = self.accelerator.device
        prompt_ids = inputs["prompt_ids"].to(device)
        prompt_mask = inputs["prompt_attention_mask"].to(device)
        chosen_ids = inputs["chosen_ids"].to(device)
        chosen_mask = inputs["chosen_attention_mask"].to(device)
        rejected_ids = inputs["rejected_ids"].to(device)
        rejected_mask = inputs["rejected_attention_mask"].to(device)
        
        # Original text for judge (if needed)
        prompt_texts = inputs.get("prompt", None)
        chosen_texts = inputs.get("chosen", None)
        rejected_texts = inputs.get("rejected", None)
        
        batch_size = prompt_ids.shape[0]
        
        # Generate Monte Carlo samples for DM term AND KL estimation
        mc_samples = []
        mc_logprobs_list = []
        mc_ref_logprobs_list = []
        mc_kl_total_list = []
        mc_entropy_list = []
        
        # Step 1: Generate all MC samples first (before any forward passes)
        with torch.no_grad():
            for _ in range(self.args.num_monte_carlo_samples):
                with unwrap_model_for_generation(
                    model, 
                    self.accelerator,
                    gather_deepspeed3_params=self.is_deepspeed_enabled
                ) as unwrapped_model:
                    _, _, mc_ids, mc_mask = self._generate(unwrapped_model, prompt_ids, prompt_mask)
                
                mc_ids = mc_ids.to(device)
                mc_mask = mc_mask.to(device)
                mc_samples.append((mc_ids, mc_mask))
        
        # Step 2: Compute all policy forward passes together
        # This includes chosen, rejected, and MC samples
        all_completion_ids = [chosen_ids, rejected_ids] + [ids for ids, _ in mc_samples]
        all_completion_masks = [chosen_mask, rejected_mask] + [mask for _, mask in mc_samples]
        
        # Batch all forward passes for the policy model
        all_logprobs = []
        for comp_ids, comp_mask in zip(all_completion_ids, all_completion_masks):
            logprobs = self._forward(model, prompt_ids, prompt_mask, comp_ids, comp_mask)
            all_logprobs.append(logprobs)
        
        # Extract the results
        chosen_logprobs = all_logprobs[0]
        rejected_logprobs = all_logprobs[1]
        mc_logprobs_list = all_logprobs[2:]
        
        # Step 3: Compute all reference model forward passes
        with torch.no_grad():
            all_ref_logprobs = []
            
            # For PEFT case
            if self.ref_model is None:
                with self.model.disable_adapter():
                    for comp_ids, comp_mask in zip(all_completion_ids, all_completion_masks):
                        ref_logprobs = self._forward(
                            self.model, prompt_ids, prompt_mask, comp_ids, comp_mask
                        )
                        all_ref_logprobs.append(ref_logprobs)
            else:
                # For separate ref model case
                # Use gather_deepspeed3_params if ref_model is also using DeepSpeed
                if self.is_deepspeed_enabled and hasattr(self.ref_model, 'module'):
                    # Reference model might also need parameter gathering
                    with unwrap_model_for_generation(
                        self.ref_model, 
                        self.accelerator,
                        gather_deepspeed3_params=False  # ref model is typically in eval mode
                    ) as unwrapped_ref:
                        for comp_ids, comp_mask in zip(all_completion_ids, all_completion_masks):
                            ref_logprobs = self._forward(
                                unwrapped_ref, prompt_ids, prompt_mask, comp_ids, comp_mask
                            )
                            all_ref_logprobs.append(ref_logprobs)
                else:
                    for comp_ids, comp_mask in zip(all_completion_ids, all_completion_masks):
                        ref_logprobs = self._forward(
                            self.ref_model, prompt_ids, prompt_mask, comp_ids, comp_mask
                        )
                        all_ref_logprobs.append(ref_logprobs)
            
            # Extract results
            chosen_ref_logprobs = all_ref_logprobs[0]
            rejected_ref_logprobs = all_ref_logprobs[1]
            mc_ref_logprobs_list = all_ref_logprobs[2:]
            
            # Compute KL and entropy for MC samples
            for mc_logprobs, mc_ref_logprobs, (mc_ids, mc_mask) in zip(
                mc_logprobs_list, mc_ref_logprobs_list, mc_samples
            ):
                # Compute KL[π||π_ref] for this generated sample
                _, total_kl = self._compute_kl_divergence(
                    mc_logprobs, mc_ref_logprobs, mc_mask,
                    kl_type=self.args.kl_type
                )
                mc_kl_total_list.append(total_kl)
                
                # Compute entropy H(π) = -E[log π]
                entropy = -(mc_logprobs * mc_mask).sum(dim=1)
                mc_entropy_list.append(entropy)
        
        # Step 4: Compute preference scores
        # Vectorized preference score computation
        g_chosen_rejected = self._compute_preference_scores_batch(
            prompt_ids, prompt_mask,
            chosen_ids, chosen_mask,
            rejected_ids, rejected_mask,
            chosen_texts, rejected_texts
        )
        
        # Direct Method (DM) term
        term_dm = torch.zeros(batch_size, device=device)
        for (mc_ids, mc_mask), mc_logprobs in zip(mc_samples, mc_logprobs_list):
            # g(mc, rejected)
            g_mc_rejected = self._compute_preference_scores_batch(
                prompt_ids, prompt_mask, mc_ids, mc_mask, rejected_ids, rejected_mask
            )
            # g(mc, chosen)
            g_mc_chosen = self._compute_preference_scores_batch(
                prompt_ids, prompt_mask, mc_ids, mc_mask, chosen_ids, chosen_mask
            )

            # Weight by log probability
            mc_logprobs_sum = (mc_logprobs * mc_mask).sum(dim=1)
            term_dm += (g_mc_rejected + g_mc_chosen) * mc_logprobs_sum
        
        term_dm = term_dm / (2 * self.args.num_monte_carlo_samples)
        
        # Vectorized Importance Sampling (IS) term computation
        chosen_logprobs_sum = (chosen_logprobs * chosen_mask).sum(dim=1)
        chosen_ref_logprobs_sum = (chosen_ref_logprobs * chosen_mask).sum(dim=1)
        rejected_logprobs_sum = (rejected_logprobs * rejected_mask).sum(dim=1)
        rejected_ref_logprobs_sum = (rejected_ref_logprobs * rejected_mask).sum(dim=1)
        
        # Compute controlled IS ratios
        is_ratio_chosen = self._compute_is_ratio_controlled(
            chosen_logprobs, chosen_ref_logprobs, chosen_mask, "chosen"
        )
        is_ratio_rejected = self._compute_is_ratio_controlled(
            rejected_logprobs, rejected_ref_logprobs, rejected_mask, "rejected"
        )
        
        # IS loss term
        residual = 1 - g_chosen_rejected
        is_loss = - 0.5 * (
            is_ratio_chosen * residual * chosen_logprobs_sum -
            is_ratio_rejected * residual * rejected_logprobs_sum
        ).mean()
        
        # DRPO loss (negative because we maximize the estimator)
        drpo_loss = -term_dm.mean() + is_loss

        kl_loss = torch.stack(mc_kl_total_list).mean()  # Use MC samples for KL loss
            
        # Total loss
        loss = drpo_loss + self.beta * kl_loss

        # Log statistics (same as before, but moved outside of no_grad context)
        with torch.no_grad():
            # ... (rest of the logging code remains the same)
            # Collect all g_mc_rejected and g_mc_chosen across MC samples
            all_g_mc_rejected = []
            all_g_mc_chosen = []
            all_mc_lengths = []
            all_mc_contains_eos = []
            all_mc_logprobs_sum = []
            
            for (mc_ids, mc_mask), mc_logprobs in zip(mc_samples, mc_logprobs_list):
                # Re-compute preference scores for logging
                g_mc_rej = self._compute_preference_scores_batch(
                    prompt_ids, prompt_mask, mc_ids, mc_mask, rejected_ids, rejected_mask
                )
                g_mc_cho = self._compute_preference_scores_batch(
                    prompt_ids, prompt_mask, mc_ids, mc_mask, chosen_ids, chosen_mask
                )
                
                all_g_mc_rejected.append(g_mc_rej)
                all_g_mc_chosen.append(g_mc_cho)
                
                # Track generation quality
                mc_lengths = mc_mask.sum(dim=1)
                all_mc_lengths.append(mc_lengths)
                
                mc_contains_eos = (mc_ids == self.processing_class.eos_token_id).any(dim=1)
                all_mc_contains_eos.append(mc_contains_eos)
                
                mc_logprobs_sum = (mc_logprobs * mc_mask).sum(dim=1)
                all_mc_logprobs_sum.append(mc_logprobs_sum)
            
            # Stack all MC samples
            all_g_mc_rejected = torch.stack(all_g_mc_rejected)  # [num_mc, batch_size]
            all_g_mc_chosen = torch.stack(all_g_mc_chosen)
            all_mc_lengths = torch.stack(all_mc_lengths)
            all_mc_contains_eos = torch.stack(all_mc_contains_eos)
            all_mc_logprobs_sum = torch.stack(all_mc_logprobs_sum)
            
            # Core DRPO metrics
            self.stats["drpo/term_dm"].append(
                self.accelerator.gather_for_metrics(term_dm).mean().item()
            )
            self.stats["drpo/term_is"].append(
                self.accelerator.gather_for_metrics(
                    is_ratio_chosen * residual - is_ratio_rejected * residual
                ).mean().item()
            )
            self.stats["drpo/preference_score"].append(
                self.accelerator.gather_for_metrics(g_chosen_rejected).mean().item()
            )
            
            # Generated sample quality metrics
            g_mc_rejected_gathered = self.accelerator.gather_for_metrics(all_g_mc_rejected.flatten())
            g_mc_chosen_gathered = self.accelerator.gather_for_metrics(all_g_mc_chosen.flatten())
            
            self.stats["generated/vs_rejected_mean"].append(g_mc_rejected_gathered.mean().item())
            self.stats["generated/vs_rejected_std"].append(g_mc_rejected_gathered.std().item())
            self.stats["generated/vs_chosen_mean"].append(g_mc_chosen_gathered.mean().item())
            self.stats["generated/vs_chosen_std"].append(g_mc_chosen_gathered.std().item())
            
            # Preference margins and win rates
            self.stats["generated/margin_over_rejected"].append(
                (g_mc_rejected_gathered - 0.5).mean().item()
            )
            self.stats["generated/margin_over_chosen"].append(
                (0.5 - g_mc_chosen_gathered).mean().item()
            )
            self.stats["generated/win_rate_vs_rejected"].append(
                (g_mc_rejected_gathered > 0.5).float().mean().item()
            )
            self.stats["generated/win_rate_vs_chosen"].append(
                (g_mc_chosen_gathered > 0.5).float().mean().item()
            )
            
            # Generation quality
            mc_lengths_gathered = self.accelerator.gather_for_metrics(all_mc_lengths.flatten())
            mc_eos_gathered = self.accelerator.gather_for_metrics(all_mc_contains_eos.flatten().float())
            
            self.stats["generated/contains_eos_rate"].append(mc_eos_gathered.mean().item())
            self.stats["generated/avg_length"].append(mc_lengths_gathered.float().mean().item())
            self.stats["generated/length_std"].append(mc_lengths_gathered.float().std().item())
            
            # IS ratio statistics with clipping indicators
            is_chosen_at_min = (is_ratio_chosen == self.args.is_clip_min).float()
            is_chosen_at_max = (is_ratio_chosen == self.args.is_clip_max).float()
            is_rejected_at_min = (is_ratio_rejected == self.args.is_clip_min).float()
            is_rejected_at_max = (is_ratio_rejected == self.args.is_clip_max).float()
            
            self.stats["is_ratio/chosen_mean"].append(
                self.accelerator.gather_for_metrics(is_ratio_chosen).mean().item()
            )
            self.stats["is_ratio/chosen_std"].append(
                self.accelerator.gather_for_metrics(is_ratio_chosen).std().item()
            )
            self.stats["is_ratio/chosen_max"].append(
                self.accelerator.gather_for_metrics(is_ratio_chosen).max().item()
            )
            self.stats["is_ratio/clip_rate_chosen"].append(
                self.accelerator.gather_for_metrics(is_chosen_at_min + is_chosen_at_max).mean().item()
            )
            
            self.stats["is_ratio/rejected_mean"].append(
                self.accelerator.gather_for_metrics(is_ratio_rejected).mean().item()
            )
            self.stats["is_ratio/rejected_std"].append(
                self.accelerator.gather_for_metrics(is_ratio_rejected).std().item()
            )
            self.stats["is_ratio/rejected_max"].append(
                self.accelerator.gather_for_metrics(is_ratio_rejected).max().item()
            )
            self.stats["is_ratio/clip_rate_rejected"].append(
                self.accelerator.gather_for_metrics(is_rejected_at_min + is_rejected_at_max).mean().item()
            )
            
            # Policy statistics
            self.stats["logps/chosen"].append(
                self.accelerator.gather_for_metrics(chosen_logprobs_sum).mean().item()
            )
            self.stats["logps/rejected"].append(
                self.accelerator.gather_for_metrics(rejected_logprobs_sum).mean().item()
            )
            
            mc_logprobs_gathered = self.accelerator.gather_for_metrics(all_mc_logprobs_sum.flatten())
            self.stats["logps/generated_mean"].append(mc_logprobs_gathered.mean().item())
            # self.stats["logps/generated_std"].append(mc_logprobs_gathered.std().item())

            self.stats["logps/chosen_ref"].append(
                self.accelerator.gather_for_metrics(chosen_ref_logprobs_sum).mean().item()
            )

            self.stats["logps/rejected_ref"].append(
                self.accelerator.gather_for_metrics(rejected_ref_logprobs_sum).mean().item()
            )

            self.stats["logps/generated_ref_mean"].append(
                self.accelerator.gather_for_metrics(
                    torch.stack(mc_ref_logprobs_list).flatten()
                ).mean().item()
            )
            
            margins = chosen_logprobs_sum - rejected_logprobs_sum
            self.stats["rewards/margins"].append(
                self.accelerator.gather_for_metrics(margins).mean().item()
            )
            self.stats["rewards/accuracy"].append(
                self.accelerator.gather_for_metrics((margins > 0).float()).mean().item()
            )
            
            # KL statistics
            all_kl_total = torch.stack(mc_kl_total_list)
            self.stats["objective/kl"].append(
                self.accelerator.gather_for_metrics(all_kl_total).mean().item()
            )
            
            # Per-token KL average
            # all_kl_per_token = torch.cat(mc_kl_per_token_list, dim=0)
            # all_masks = torch.cat([mask for _, mask in mc_samples], dim=0)
            # avg_per_token_kl = all_kl_per_token.sum() / all_masks.sum()
            # self.stats["objective/kl_per_token"].append(
            #     self.accelerator.gather_for_metrics(avg_per_token_kl).item()
            # )
            
            # Entropy
            all_entropy = torch.stack(mc_entropy_list)
            self.stats["objective/entropy"].append(
                self.accelerator.gather_for_metrics(all_entropy).mean().item()
            )

            
            # Loss components
            self.stats["loss/drpo"].append(
                self.accelerator.gather_for_metrics(drpo_loss).mean().item()
            )
            self.stats["loss/kl"].append(
                self.accelerator.gather_for_metrics(self.beta * kl_loss).mean().item()
            )
            self.stats["loss/total"].append(
                self.accelerator.gather_for_metrics(loss).mean().item()
            )
            
            self.stats["beta"].append(self.beta)
        
        # Empty cache if needed
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            empty_cache()
        
        # Handle multi-GPU averaging
        if self.args.n_gpu > 1:
            loss = loss.mean()
        
        # Backward pass
        kwargs = {}
        if self.args.optim in ["lomo", "adalomo"]:
            kwargs["learning_rate"] = self._get_learning_rate()
        
        if self.use_apex:
            from apex import amp
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)
        
        return loss.detach() / self.args.gradient_accumulation_steps
    

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, 
                                ignore_keys_for_eval, start_time=None, learning_rate=None):
        """
        Log metrics and optionally evaluate (following OnlineDPO pattern).
        
        This method signature matches the parent class in newer transformers versions.
        """
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: dict[str, float] = {}
            
            # Standard training metrics
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_loss -= tr_loss
            
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            
            # Use provided learning rate or get current one
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()
            
            # DRPO-specific metrics (averaged over logging window)
            for key, values in self.stats.items():
                if values:
                    avg_value = sum(values) / len(values)
                    logs[key] = avg_value
                    
                    # Add some derived metrics for better insights
                    if key == "generated/vs_rejected_mean":
                        logs["generated/advantage_over_rejected"] = avg_value - 0.5
                    elif key == "generated/vs_chosen_mean":
                        logs["generated/advantage_vs_chosen"] = avg_value - 0.5
            
            # Compute some composite metrics
            if "generated/win_rate_vs_rejected" in logs and "generated/win_rate_vs_chosen" in logs:
                logs["generated/balanced_quality"] = (
                    logs["generated/win_rate_vs_rejected"] * (logs["generated/win_rate_vs_chosen"])
                ) ** 0.5
            
            if "is_ratio/clip_rate_chosen" in logs and "is_ratio/clip_rate_rejected" in logs:
                logs["is_ratio/overall_clip_rate"] = (
                    logs["is_ratio/clip_rate_chosen"] + logs["is_ratio/clip_rate_rejected"]
                ) / 2
            
            # Reset stats for next logging window
            self.stats = {key: [] for key in self.stats}
            
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            
            # Log the metrics
            self.log(logs)
        
        # Traditional evaluation on validation set (optional)
        metrics = None
        if self.control.should_evaluate:
            # Reset accumulator before evaluation
            self._eval_metrics_accumulator = {}
            self._eval_metrics_count = 0
            
            if self.eval_dataset is not None:
                metrics = self._evaluate(trial, ignore_keys_for_eval)
                
                # Add accumulated custom metrics
                if hasattr(self, '_eval_metrics_accumulator') and self._eval_metrics_count > 0:
                    for key, value in self._eval_metrics_accumulator.items():
                        avg_value = value / self._eval_metrics_count
                        # Add eval_ prefix if not already present
                        prefixed_key = f"eval_{key}" if not key.startswith("eval_") else key
                        metrics[prefixed_key] = avg_value
            else:
                # If no eval dataset, we can still compute metrics from training stats
                metrics = {}
                for key in ["generated/win_rate_vs_rejected", "rewards/accuracy", "objective/kl"]:
                    if key in logs:
                        metrics[f"eval_{key}"] = logs[key]
        
        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, Any]]]:
        """
        Compute loss for DRPO. During training, delegates to training_step.
        During evaluation, computes preference accuracy and generation quality metrics.
        
        Args:
            model: The model to evaluate
            inputs: Dictionary containing prompt_ids, chosen_ids, rejected_ids, etc.
            return_outputs: Whether to return additional outputs
            num_items_in_batch: Number of items in batch (for gradient accumulation)
            
        Returns:
            Loss tensor, or (loss, outputs) tuple if return_outputs=True
        """
        if model.training:
            # During training, use the custom training_step
            loss = self.training_step(model, inputs, num_items_in_batch)
            # Return empty dict as outputs to maintain compatibility
            return (loss, {}) if return_outputs else loss
        
        # Evaluation mode: compute comprehensive metrics
        model.eval()
        device = self.accelerator.device
        
        with torch.no_grad():
            # Extract inputs (matching your data collator format)
            prompt_ids = inputs["prompt_ids"].to(device)
            prompt_mask = inputs["prompt_attention_mask"].to(device)
            chosen_ids = inputs["chosen_ids"].to(device)
            chosen_mask = inputs["chosen_attention_mask"].to(device)
            rejected_ids = inputs["rejected_ids"].to(device)
            rejected_mask = inputs["rejected_attention_mask"].to(device)
            
            # Get text inputs if available (for judge)
            prompt_texts = inputs.get("prompt", None)
            chosen_texts = inputs.get("chosen", None)
            rejected_texts = inputs.get("rejected", None)
            
            batch_size = prompt_ids.size(0)
            
            # 1. Compute preference accuracy between chosen and rejected
            chosen_logprobs = self._forward(model, prompt_ids, prompt_mask, chosen_ids, chosen_mask)
            rejected_logprobs = self._forward(model, prompt_ids, prompt_mask, rejected_ids, rejected_mask)
            
            chosen_logprobs_sum = (chosen_logprobs * chosen_mask).sum(1)
            rejected_logprobs_sum = (rejected_logprobs * rejected_mask).sum(1)
            
            # Model's implicit preference (higher log prob = preferred)
            model_prefers_chosen = (chosen_logprobs_sum > rejected_logprobs_sum).float()
            preference_accuracy = model_prefers_chosen.mean()
            
            # 2. Compute actual preference score using preference model
            g_chosen_rejected = self._compute_preference_scores_batch(
                prompt_ids, prompt_mask,
                chosen_ids, chosen_mask,
                rejected_ids, rejected_mask,
                chosen_texts, rejected_texts
            )
            preference_alignment = ((g_chosen_rejected > 0.5).float() == model_prefers_chosen).float().mean()
            
            metrics = {
                'preference_accuracy': preference_accuracy.item(),
                'preference_alignment': preference_alignment.item(),
                'g_chosen_rejected': g_chosen_rejected.mean().item(),
            }
            
            # 3. Generate MC samples if requested
            if getattr(self.args, 'eval_with_generation', True):
                num_eval_samples = getattr(self.args, 'eval_mc_samples', 2)
                
                all_g_mc_rejected = []
                all_g_mc_chosen = []
                all_mc_lengths = []
                all_mc_logprobs_sum = []
                all_mc_kl = []
                all_mc_entropy = []
                mc_samples_for_logging = []
                
                for _ in range(num_eval_samples):
                    # Generate samples
                    with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                        _, _, mc_ids, mc_mask = self._generate(unwrapped_model, prompt_ids, prompt_mask)
                    
                    mc_ids = mc_ids.to(device)
                    mc_mask = mc_mask.to(device)
                    
                    mc_samples_for_logging.append((mc_ids, mc_mask))

                    # Compute log probabilities
                    mc_logprobs = self._forward(model, prompt_ids, prompt_mask, mc_ids, mc_mask)
                    mc_logprobs_sum = (mc_logprobs * mc_mask).sum(1)
                    all_mc_logprobs_sum.append(mc_logprobs_sum)

                    # Compute log probabilities under reference policy
                    if self.ref_model is not None:
                        mc_ref_logprobs = self._forward(
                            self.ref_model, prompt_ids, prompt_mask, mc_ids, mc_mask
                        )
                    else:
                        with self.model.disable_adapter():
                            mc_ref_logprobs = self._forward(
                                self.model, prompt_ids, prompt_mask, mc_ids, mc_mask
                            )
                    
                    # Compute KL[π||π_ref] for generated samples
                    _, total_kl = self._compute_kl_divergence(
                        mc_logprobs, mc_ref_logprobs, mc_mask,
                        kl_type=self.args.kl_type
                    )
                    all_mc_kl.append(total_kl)
                    
                    # Compute entropy H(π)
                    entropy = -(mc_logprobs * mc_mask).sum(1)
                    all_mc_entropy.append(entropy)
                    
                    # Compute preference scores
                    g_mc_rejected = self._compute_preference_scores_batch(
                        prompt_ids, prompt_mask, mc_ids, mc_mask, rejected_ids, rejected_mask
                    )
                    g_mc_chosen = self._compute_preference_scores_batch(
                        prompt_ids, prompt_mask, mc_ids, mc_mask, chosen_ids, chosen_mask
                    )
                    
                    all_g_mc_rejected.append(g_mc_rejected)
                    all_g_mc_chosen.append(g_mc_chosen)
                    
                    # Track generation lengths
                    mc_lengths = mc_mask.sum(1)
                    all_mc_lengths.append(mc_lengths)
                
                # Aggregate generation metrics
                all_g_mc_rejected = torch.stack(all_g_mc_rejected)
                all_g_mc_chosen = torch.stack(all_g_mc_chosen)
                all_mc_lengths = torch.stack(all_mc_lengths)
                all_mc_logprobs_sum = torch.stack(all_mc_logprobs_sum)
                all_mc_kl = torch.stack(all_mc_kl)
                all_mc_entropy = torch.stack(all_mc_entropy)

                # Compute generation quality metrics
                metrics.update({
                    'generated/vs_rejected_mean': all_g_mc_rejected.mean().item(),
                    'generated/vs_rejected_std': all_g_mc_rejected.std().item(),
                    'generated/vs_chosen_mean': all_g_mc_chosen.mean().item(),
                    'generated/vs_chosen_std': all_g_mc_chosen.std().item(),
                    'generated/win_rate_vs_rejected': (all_g_mc_rejected > 0.5).float().mean().item(),
                    'generated/win_rate_vs_chosen': (all_g_mc_chosen > 0.5).float().mean().item(),
                    'generated/avg_length': all_mc_lengths.float().mean().item(),
                    'generated/logprobs_mean': all_mc_logprobs_sum.mean().item(),
                    # KL divergence (from generated samples only!)
                    'eval/kl_divergence': all_mc_kl.mean().item(),
                    'eval/kl_divergence_std': all_mc_kl.std().item(),
                    
                    # Entropy
                    'eval/entropy': all_mc_entropy.mean().item(),
                    'eval/entropy_std': all_mc_entropy.std().item(),
                })
            
                            # Log samples to wandb
                if (self.state.is_world_process_zero and 
                    wandb.run is not None and
                    self.state.global_step % self.args.eval_steps == 0):  # Log every 100 steps
                    
                    # Decode texts
                    if prompt_texts is None:
                        prompt_texts = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
                    if chosen_texts is None:
                        chosen_texts = self.processing_class.batch_decode(chosen_ids, skip_special_tokens=True)
                    if rejected_texts is None:
                        rejected_texts = self.processing_class.batch_decode(rejected_ids, skip_special_tokens=True)
                    
                    # Decode MC samples
                    mc_texts_all = []
                    for mc_ids, _ in mc_samples_for_logging:
                        mc_texts = self.processing_class.batch_decode(mc_ids, skip_special_tokens=True)
                        mc_texts_all.append(mc_texts)
                    
                    # Create table
                    # num_to_log = min(5, batch_size)
                    # columns = ["Prompt", "Chosen", "Rejected"]
                    # for i in range(len(mc_texts_all)):
                    #     columns.append(f"Generated_{i+1}")
                    # columns.extend(["P(chosen>rej)", "P(gen1>rej)", "P(gen1>chosen)"])
                    
                    # table_data = []
                    # for i in range(num_to_log):
                    #     row = [
                    #         prompt_texts[i][:300],
                    #         chosen_texts[i][:300],
                    #         rejected_texts[i][:300]
                    #     ]
                        
                    #     for mc_texts in mc_texts_all:
                    #         row.append(mc_texts[i][:300])
                        
                    #     row.extend([
                    #         f"{g_chosen_rejected[i]:.3f}",
                    #         f"{all_g_mc_rejected[0][i]:.3f}",
                    #         f"{all_g_mc_chosen[0][i]:.3f}"
                    #     ])
                        
                    #     table_data.append(row)
                    
                    # # # Log table and histograms
                    # # table = wandb.Table(columns=columns, data=table_data)
                    # # wandb.log({
                    # #     "eval/samples": table,
                    # #     "eval/g_mc_rejected_hist": wandb.Histogram(all_g_mc_rejected[0].float().cpu().numpy()),
                    # #     "eval/g_mc_chosen_hist": wandb.Histogram(all_g_mc_chosen[0].float().cpu().numpy()),
                    # #     "eval/mc_lengths_hist": wandb.Histogram(all_mc_lengths[0].float().cpu().numpy())
                    # # })


                    # 1. Log the table of generated text samples
                    num_to_log = min(5, batch_size)
                    headers = ["Prompt", "Chosen", "Rejected"]
                    for i in range(len(mc_texts_all)):
                        headers.append(f"Generated_{i+1}")
                    headers.extend(["P(chosen>rej)", "P(gen1>rej)", "P(gen1>chosen)"])

                    rows = []
                    for i in range(num_to_log):
                        row = [
                            prompt_texts[i][:300],
                            chosen_texts[i][:300],
                            rejected_texts[i][:300]
                        ]
                        for mc_texts in mc_texts_all:
                            row.append(random.shuffle(mc_texts)[i][:300])
                        row.extend([
                            f"{g_chosen_rejected[i]:.3f}",
                            f"{all_g_mc_rejected[0][i]:.3f}",
                            f"{all_g_mc_chosen[0][i]:.3f}"
                        ])
                        rows.append(row)

                    # Create the echarts Table object
                    table_chart = swanlab.echarts.Table()
                    table_chart.add(headers, rows)
                    swanlab.log({"eval/samples": table_chart})


                    # 2. Log the histograms by creating Bar charts
                    # Helper function to create a histogram bar chart
                    def create_histogram_chart(data_array, num_bins=20):
                        counts, bin_edges = np.histogram(data_array, bins=num_bins)
                        # Create labels for the x-axis, e.g., "0.10-0.20"
                        x_axis_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(counts))]
                        
                        # Create and configure the Bar chart
                        bar_chart = swanlab.echarts.Bar()
                        bar_chart.add_xaxis(x_axis_labels)
                        bar_chart.add_yaxis("count", counts.tolist()) # y-axis requires a name and list of values
                        return bar_chart

                    # Create and log each histogram
                    g_mc_rejected_hist = create_histogram_chart(all_g_mc_rejected[0].float().cpu().numpy())
                    swanlab.log({"eval/g_mc_rejected_hist": g_mc_rejected_hist})

                    g_mc_chosen_hist = create_histogram_chart(all_g_mc_chosen[0].float().cpu().numpy())
                    swanlab.log({"eval/g_mc_chosen_hist": g_mc_chosen_hist})

                    mc_lengths_hist = create_histogram_chart(all_mc_lengths[0].float().cpu().numpy())
                    swanlab.log({"eval/mc_lengths_hist": mc_lengths_hist})
            
            
            # Use negative preference accuracy as loss (lower is better)
            loss = -preference_accuracy
            
            # Create outputs dict compatible with Trainer expectations
            outputs = {
                'loss': loss,
                'metrics': metrics,
                # Include for compatibility with prediction_step
                'logits': model_prefers_chosen.unsqueeze(-1),  # [batch_size, 1]
                'labels': torch.ones_like(model_prefers_chosen).unsqueeze(-1),  # chosen should be preferred
            }
            
        return (loss, outputs) if return_outputs else loss


    # Optional: Override prediction_step to properly aggregate metrics
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Custom prediction step that properly handles DRPO evaluation.
        """
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        
        # Store metrics for aggregation
        if not hasattr(self, '_eval_metrics_accumulator'):
            self._eval_metrics_accumulator = {}
            self._eval_metrics_count = 0
        
        # Accumulate metrics
        if 'metrics' in outputs:
            for key, value in outputs['metrics'].items():
                if key not in self._eval_metrics_accumulator:
                    self._eval_metrics_accumulator[key] = 0
                self._eval_metrics_accumulator[key] += value
            self._eval_metrics_count += 1
        
        if prediction_loss_only:
            return (loss, None, None)
        
        # Return logits and labels for compatibility
        logits = outputs.get('logits', None)
        labels = outputs.get('labels', None)
        
        return (loss, logits, labels)

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """
        Prediction/evaluation loop, shared by evaluate() and predict().
        """
        # Reset metrics before starting
        self._eval_metrics_accumulator = {}
        self._eval_metrics_count = 0
        
        # Run the standard evaluation loop
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )
        
        # Add accumulated metrics to output
        if self._eval_metrics_count > 0:
            for key, value in self._eval_metrics_accumulator.items():
                avg_value = value / self._eval_metrics_count
                metric_key = f"{metric_key_prefix}_{key}" if metric_key_prefix else key
                output.metrics[metric_key] = avg_value
        
        return output

            
    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, List[str], None] = None,
    ):
        """Create model card for DRPO training."""
        if not self.is_world_process_zero():
            return
        
        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None
        
        # Normalize tags
        if tags is None:
            tags = set()
        elif isinstance(tags, str):
            tags = {tags}
        else:
            tags = set(tags)
        
        tags.update(self._tag_names)
        
        citation = textwrap.dedent("""\
        @article{xu2024doubly,
            title        = {{Doubly Robust Alignment for Large Language Models}},
            author       = {Xu, Erhan and Ye, Kai and Zhou, Hongyi and Zhu, Luhan and Quinzan, Francesco and Shi, Chengchun},
            year         = 2025,
            journal      = {arXiv preprint arXiv:2506.01183}
        }""")
        
        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="DRPO",
            trainer_citation=citation,
            paper_title="Doubly Robust Alignment for Large Language Models",
            paper_id="2506.01183",
        )
        
        model_card.save(os.path.join(self.args.output_dir, "README.md"))


