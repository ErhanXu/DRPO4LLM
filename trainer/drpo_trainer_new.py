import os
import textwrap
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union
from packaging import version

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

from .drpo_config import DRPOConfig

# Optional imports
if is_wandb_available():
    import wandb

if is_peft_available():
    from peft import PeftModel


if is_wandb_available():
    import wandb

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
            
            self.preference_model = GPMwithRewardNetwork(
                model_name_or_path=args.preference_model_path,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                pad_token_id=processing_class.pad_token_id if processing_class else 0,
                is_general_preference=(args.preference_model_type == "general"),
                bf16=args.bf16
            )
            # Don't use standard reward model when using custom preference model
            reward_model = None
        
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
                from .utils import prepare_deepspeed
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
            "logps/generated_std": [],            # Std[log π(mc)]
            "rewards/margins": [],                # log π(chosen) - log π(rejected)
            "rewards/accuracy": [],               # P(log π(chosen) > log π(rejected))
            
            # KL and regularization
            "objective/kl": [],
            "objective/kl_chosen": [],
            "objective/kl_rejected": [],
            "objective/kl_generated": [],         # KL for generated samples (k3 only)
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
    ) -> torch.Tensor:
        """
        Compute preference scores g(y1, y2 | x) for a batch of comparisons.
        
        Supports multiple preference models:
        1. Standard reward models with Bradley-Terry framework
        2. Custom preference models (general preference models)
        3. Judge-based evaluation
        
        Args:
            prompt_ids: Tokenized prompts [batch_size, prompt_len]
            prompt_mask: Attention mask for prompts
            ids_1: Tokenized first responses [batch_size, response_len]
            mask_1: Attention mask for first responses
            ids_2: Tokenized second responses [batch_size, response_len]
            mask_2: Attention mask for second responses
            texts_1: Original text for first responses (for judge)
            texts_2: Original text for second responses (for judge)
            
        Returns:
            Preference scores [batch_size]
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
                    preference_model=self.preference_model,
                    a1_iuput_ids=prompt_response_1,
                    a1_attention_mask=attention_mask_1,
                    a2_input_ids=prompt_response_2,
                    a2_attention_mask=attention_mask_2,
                    is_bt_model=(self.args.preference_model_type == "bt"),
                    device=prompt_ids.device
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
                    raise ValueError("Judge requires original text responses")
                
                # Get prompts text
                prompts = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
                
                # Handle conversational format if needed
                from ..data_utils import is_conversational
                if any(is_conversational({"prompt": t}) for t in texts_1):
                    import jinja2
                    env = jinja2.Environment()
                    template = env.from_string(SIMPLE_CHAT_TEMPLATE)
                    prompts = [template.render(messages=p) if is_conversational({"prompt": p}) else p 
                              for p in prompts]
                    texts_1 = [template.render(messages=t) if is_conversational({"prompt": t}) else t 
                              for t in texts_1]
                    texts_2 = [template.render(messages=t) if is_conversational({"prompt": t}) else t 
                              for t in texts_2]
                
                # Judge returns preference probabilities
                scores = self.judge.judge(
                    prompts,
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
        Perform DRPO training step.
        
        Implements the doubly robust estimator:
        ψ = (1/2) * E_y~π[g(y, rejected) + g(y, chosen)] 
            + (1/2) * [π(chosen)/π_ref(chosen) * (1 - g(chosen, rejected))
                      - π(rejected)/π_ref(rejected) * (1 - g(chosen, rejected))]
        
        Args:
            model: The model being trained
            inputs: Batch of training data
            num_items_in_batch: Number of items in batch (for gradient accumulation)
            
        Returns:
            Loss value for this step
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
        
        for _ in range(self.args.num_monte_carlo_samples):
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                _, _, mc_ids, mc_mask = self._generate(unwrapped_model, prompt_ids, prompt_mask)
            
            mc_ids = mc_ids.to(device)
            mc_mask = mc_mask.to(device)
            
            # Compute logprobs for this MC sample
            mc_logprobs = self._forward(model, prompt_ids, prompt_mask, mc_ids, mc_mask)
            mc_logprobs_list.append(mc_logprobs)
            
            # Compute reference logprobs for KL
            with torch.no_grad():
                if self.ref_model is not None:
                    mc_ref_logprobs = self._forward(self.ref_model, prompt_ids, prompt_mask, mc_ids, mc_mask)
                else:
                    with self.model.disable_adapter():
                        mc_ref_logprobs = self._forward(self.model, prompt_ids, prompt_mask, mc_ids, mc_mask)
                mc_ref_logprobs_list.append(mc_ref_logprobs)
            
            mc_samples.append((mc_ids, mc_mask))
        
        # Compute log probabilities under policy and reference
        chosen_logprobs = self._forward(model, prompt_ids, prompt_mask, chosen_ids, chosen_mask)
        rejected_logprobs = self._forward(model, prompt_ids, prompt_mask, rejected_ids, rejected_mask)
        
        with torch.no_grad():
            if self.ref_model is not None:
                chosen_ref_logprobs = self._forward(
                    self.ref_model, prompt_ids, prompt_mask, chosen_ids, chosen_mask
                )
                rejected_ref_logprobs = self._forward(
                    self.ref_model, prompt_ids, prompt_mask, rejected_ids, rejected_mask
                )
            else:
                # PEFT case - use base model as reference
                with self.model.disable_adapter():
                    chosen_ref_logprobs = self._forward(
                        self.model, prompt_ids, prompt_mask, chosen_ids, chosen_mask
                    )
                    rejected_ref_logprobs = self._forward(
                        self.model, prompt_ids, prompt_mask, rejected_ids, rejected_mask
                    )
        
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
        
        # IS loss term (implementing the doubled dataset approach)
        # For (chosen, rejected, z=1): is_chosen * (1 - g) * log π(chosen)
        # For (rejected, chosen, z=0): -is_rejected * (1 - g) * log π(rejected)
        residual = 1 - g_chosen_rejected
        is_loss = -(
            is_ratio_chosen * residual * chosen_logprobs_sum -
            is_ratio_rejected * residual * rejected_logprobs_sum
        ).mean()
        
        # DRPO loss (negative because we maximize the estimator)
        drpo_loss = -term_dm.mean() + is_loss

        # KL regularization with k1 or k3 estimation
        if self.args.kl_type == "k1":
            # k1: Standard KL using offline chosen/rejected samples
            kl_chosen = ((chosen_logprobs - chosen_ref_logprobs) * chosen_mask).sum(1)
            kl_rejected = ((rejected_logprobs - rejected_ref_logprobs) * rejected_mask).sum(1)
            kl_loss = 0.5 * (kl_chosen + kl_rejected).mean()
            
        else:  # k3: Using generated MC samples
            # k3: E_y~π[π_ref(y|x)/π(y|x) - 1 - log(π_ref(y|x)/π(y|x))]
            kl_terms = []
            for mc_logprobs, mc_ref_logprobs, (_, mc_mask) in zip(mc_logprobs_list, mc_ref_logprobs_list, mc_samples):
                mc_logprobs_sum = (mc_logprobs * mc_mask).sum(dim=1)
                mc_ref_logprobs_sum = (mc_ref_logprobs * mc_mask).sum(dim=1)
                
                # Compute π_ref/π ratio
                log_ratio = mc_ref_logprobs_sum - mc_logprobs_sum
                ratio = torch.exp(torch.clamp(log_ratio, -1e3, 10))  # Clamp for stability
                
                # k3 KL: ratio - 1 - log(ratio)
                kl_term = ratio - 1 - log_ratio
                kl_terms.append(kl_term)
            
            kl_loss = torch.stack(kl_terms).mean()
            
        # Total loss
        loss = drpo_loss + self.beta * kl_loss

        # Log statistics
        with torch.no_grad():
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
            self.stats["logps/generated_std"].append(mc_logprobs_gathered.std().item())
            
            margins = chosen_logprobs_sum - rejected_logprobs_sum
            self.stats["rewards/margins"].append(
                self.accelerator.gather_for_metrics(margins).mean().item()
            )
            self.stats["rewards/accuracy"].append(
                self.accelerator.gather_for_metrics((margins > 0).float()).mean().item()
            )
            
            # KL statistics
            self.stats["objective/kl"].append(
                self.accelerator.gather_for_metrics(kl_loss).mean().item()
            )
            if self.args.kl_type == "k1":
                self.stats["objective/kl_chosen"].append(
                    self.accelerator.gather_for_metrics(kl_chosen).mean().item()
                )
                self.stats["objective/kl_rejected"].append(
                    self.accelerator.gather_for_metrics(kl_rejected).mean().item()
                )
            else:  # k3
                # For k3, track KL of generated samples
                kl_generated = torch.stack(kl_terms).mean()
                self.stats["objective/kl_generated"].append(
                    self.accelerator.gather_for_metrics(kl_generated).mean().item()
                )
            
            # Loss components
            self.stats["loss/drpo"].append(
                self.accelerator.gather_for_metrics(drpo_loss).item()
            )
            self.stats["loss/kl"].append(
                self.accelerator.gather_for_metrics(self.beta * kl_loss).item()
            )
            self.stats["loss/total"].append(
                self.accelerator.gather_for_metrics(loss).item()
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
                        logs["generated/disadvantage_vs_chosen"] = avg_value - 0.5
            
            # Compute some composite metrics
            if "generated/win_rate_vs_rejected" in logs and "generated/win_rate_vs_chosen" in logs:
                logs["generated/balanced_quality"] = (
                    logs["generated/win_rate_vs_rejected"] * (1 - logs["generated/win_rate_vs_chosen"])
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
            if self.eval_dataset is not None:
                metrics = self._evaluate(trial, ignore_keys_for_eval)
            else:
                # If no eval dataset, we can still compute metrics from training stats
                metrics = {}
                for key in ["generated/win_rate_vs_rejected", "rewards/accuracy", "objective/kl"]:
                    if key in logs:
                        metrics[f"eval_{key}"] = logs[key]
            
            # Determine if this is the best model so far
            if metrics:
                is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)
                
                if self.args.save_strategy == "best":
                    self.control.should_save = is_new_best_metric
        
        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            
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


