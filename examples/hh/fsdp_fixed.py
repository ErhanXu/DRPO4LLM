# train_fsdp_peft_fixed.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from accelerate import Accelerator
from trainer.drpo_trainer_new import DRPOTrainer
from trainer.drpo_config_new import DRPOConfig
import swanlab
from transformers import TrainerCallback
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Initialize accelerator
accelerator = Accelerator()

# Model paths
model_name_or_path = "Qwen/Qwen2.5-1.5B"  # Base model path
reward_model_path = "Kyleyee/Qwen2.5-1.5B-reward-hh-retrain"  # Reward model path, can be same as base model


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

# Training configuration - FIXED FSDP OPTIONS
training_config = DRPOConfig(
    output_dir="./drpo-fsdp-peft-qwen2.5-7b",
    
    # CORRECT FSDP configuration - only use valid options
    fsdp="shard_grad_op auto_wrap",  # Only valid FSDPOption values
    fsdp_transformer_layer_cls_to_wrap="Qwen2DecoderLayer",
    
    # Additional FSDP settings through fsdp_config
    fsdp_config={
        "backward_prefetch": "backward_pre",
        "use_orig_params": True,  # Critical for PEFT
        "cpu_ram_efficient_loading": True,
        "sync_module_states": True,
    },
    
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
    is_clip_min=0.1,
    is_clip_max=10.0,
    
    # Generation
    max_new_tokens=256,
    temperature=0.8,
    max_length=1024,
    max_prompt_length=512,
    max_completion_length=512,
    
    # Optimizations
    bf16=True,
    tf32=True,
    gradient_checkpointing=False,  # Disable for FSDP+PEFT compatibility
    
    # Memory management
    torch_empty_cache_steps=50,
    
    # Logging
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    eval_strategy="steps",
    eval_steps=100,
    
    # Dataset
    dataset_num_proc=1,
    dataloader_num_workers=0,
    
    # Disable default reporting
    report_to="none",
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    use_cache=False,
)

# Apply PEFT
print("Applying PEFT...")
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=128,
    lora_alpha=256,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    modules_to_save=None,
)

# Enable input gradients for PEFT
model.enable_input_require_grads()
model = get_peft_model(model, peft_config)

if accelerator.is_main_process:
    model.print_trainable_parameters()

# Load reward model
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

# Process datasets
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

# Create custom DRPO trainer for FSDP
class DRPOTrainerFSDP(DRPOTrainer):
    """DRPO trainer with proper FSDP handling for generation."""
    
# drpo_trainer_fsdp_complete.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from typing import Dict, Any, Optional, Union, List, Tuple
from trainer.drpo_trainer_new import DRPOTrainer
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import pad, truncate_right

class DRPOTrainerFSDPComplete(DRPOTrainer):
    """Complete DRPO trainer with full FSDP support."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize stats if not already done
        if not hasattr(self, 'stats'):
            self.stats = {
                "drpo/term_dm": [],
                "drpo/term_is": [],
                "drpo/preference_score": [],
                "generated/vs_rejected_mean": [],
                "generated/vs_rejected_std": [],
                "generated/vs_chosen_mean": [],
                "generated/vs_chosen_std": [],
                "generated/margin_over_rejected": [],
                "generated/margin_over_chosen": [],
                "generated/win_rate_vs_rejected": [],
                "generated/win_rate_vs_chosen": [],
                "generated/contains_eos_rate": [],
                "generated/avg_length": [],
                "generated/length_std": [],
                "is_ratio/chosen_mean": [],
                "is_ratio/chosen_std": [],
                "is_ratio/chosen_max": [],
                "is_ratio/rejected_mean": [],
                "is_ratio/rejected_std": [],
                "is_ratio/rejected_max": [],
                "is_ratio/clip_rate_chosen": [],
                "is_ratio/clip_rate_rejected": [],
                "logps/chosen": [],
                "logps/rejected": [],
                "logps/generated_mean": [],
                "logps/generated_std": [],
                "rewards/margins": [],
                "rewards/accuracy": [],
                "objective/kl": [],
                "objective/entropy": [],
                "beta": [],
                "loss/drpo": [],
                "loss/kl": [],
                "loss/total": [],
            }
    
    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        """
        Complete DRPO training step with FSDP compatibility.
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
                # FSDP-aware generation
                if self.accelerator.distributed_type == "FSDP":
                    # Check if model is FSDP wrapped
                    is_fsdp_wrapped = isinstance(model, FSDP) or any(
                        isinstance(m, FSDP) for m in model.modules()
                    )
                    
                    if is_fsdp_wrapped:
                        # Use FSDP context to gather full parameters
                        with FSDP.summon_full_params(model, writeback=False, recurse=True):
                            mc_ids, mc_mask = self._fsdp_generate(model, prompt_ids, prompt_mask)
                    else:
                        # Model not wrapped, generate normally
                        _, _, mc_ids, mc_mask = self._generate(model, prompt_ids, prompt_mask)
                else:
                    # Non-FSDP path
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
        g_chosen_rejected = self._compute_preference_scores_batch(
            prompt_ids, prompt_mask,
            chosen_ids, chosen_mask,
            rejected_ids, rejected_mask,
            chosen_texts, rejected_texts, prompt_texts
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
        is_loss = -(
            is_ratio_chosen * residual * chosen_logprobs_sum -
            is_ratio_rejected * residual * rejected_logprobs_sum
        ).mean()
        
        # DRPO loss (negative because we maximize the estimator)
        drpo_loss = -term_dm.mean() + is_loss

        # KL loss
        kl_loss = torch.stack(mc_kl_total_list).mean()
            
        # Total loss
        loss = drpo_loss + self.beta * kl_loss

        # Log statistics
        with torch.no_grad():
            self._log_training_stats(
                term_dm, is_ratio_chosen, is_ratio_rejected, residual,
                g_chosen_rejected, mc_samples, mc_logprobs_list,
                chosen_logprobs_sum, rejected_logprobs_sum,
                mc_kl_total_list, mc_entropy_list,
                drpo_loss, kl_loss, loss,
                prompt_ids, prompt_mask, chosen_ids, chosen_mask,
                rejected_ids, rejected_mask
            )
        
        # Empty cache if needed
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()
        
        # Handle multi-GPU averaging
        if self.args.n_gpu > 1:
            loss = loss.mean()
        
        # Backward pass
        self.accelerator.backward(loss)
        
        return loss.detach() / self.args.gradient_accumulation_steps
    
    def _fsdp_generate(self, model, prompt_ids, prompt_mask):
        """Generate within FSDP summon_full_params context."""
        # Save model state
        was_training = model.training
        cache_enabled = getattr(model.config, 'use_cache', True)
        
        # Set to eval mode and enable cache
        model.eval()
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = True
        
        try:
            # Generate
            eos_token_id = self.processing_class.eos_token_id
            pad_token_id = self.processing_class.pad_token_id
            
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                output_ids = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    max_new_tokens=self.args.max_new_tokens,
                    temperature=self.args.temperature,
                    do_sample=True,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )
            
            # Extract completions
            completion_ids = output_ids[:, prompt_ids.size(1):]
            
            # Truncate and create masks
            completion_ids, completion_mask = truncate_right(
                completion_ids, eos_token_id, pad_token_id
            )
            
            return completion_ids, completion_mask
            
        finally:
            # Restore model state
            if hasattr(model.config, 'use_cache'):
                model.config.use_cache = cache_enabled
            if was_training:
                model.train()
    
    def _log_training_stats(
        self, term_dm, is_ratio_chosen, is_ratio_rejected, residual,
        g_chosen_rejected, mc_samples, mc_logprobs_list,
        chosen_logprobs_sum, rejected_logprobs_sum,
        mc_kl_total_list, mc_entropy_list,
        drpo_loss, kl_loss, loss,
        prompt_ids, prompt_mask, chosen_ids, chosen_mask,
        rejected_ids, rejected_mask
    ):
        """Log comprehensive training statistics."""
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
        
        # IS ratio statistics
        self.stats["is_ratio/chosen_mean"].append(
            self.accelerator.gather_for_metrics(is_ratio_chosen).mean().item()
        )
        self.stats["is_ratio/chosen_std"].append(
            self.accelerator.gather_for_metrics(is_ratio_chosen).std().item()
        )
        self.stats["is_ratio/chosen_max"].append(
            self.accelerator.gather_for_metrics(is_ratio_chosen).max().item()
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
        
        # Clip rates
        is_chosen_at_min = (is_ratio_chosen == self.args.is_clip_min).float()
        is_chosen_at_max = (is_ratio_chosen == self.args.is_clip_max).float()
        is_rejected_at_min = (is_ratio_rejected == self.args.is_clip_min).float()
        is_rejected_at_max = (is_ratio_rejected == self.args.is_clip_max).float()
        
        self.stats["is_ratio/clip_rate_chosen"].append(
            self.accelerator.gather_for_metrics(is_chosen_at_min + is_chosen_at_max).mean().item()
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
        
        # KL and entropy statistics
        all_kl_total = torch.stack(mc_kl_total_list)
        self.stats["objective/kl"].append(
            self.accelerator.gather_for_metrics(all_kl_total).mean().item()
        )
        
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
    
    def _generate_simple(self, model, prompt_ids, prompt_mask):
        """Simple generation method for use within FSDP context."""
        eos_token_id = self.processing_class.eos_token_id
        pad_token_id = self.processing_class.pad_token_id
        
        # Generate
        output_ids = model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=self.args.max_new_tokens,
            temperature=self.args.temperature,
            do_sample=True,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        
        # Extract completions
        completion_ids = output_ids[:, prompt_ids.size(1):]
        completion_mask = (completion_ids != pad_token_id).long()
        
        return prompt_ids, prompt_mask, completion_ids, completion_mask

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
    trainer.train()
    
    # Save model
    if accelerator.is_main_process:
        print("Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(training_config.output_dir)
    
    accelerator.wait_for_everyone()
    print("Training complete!")