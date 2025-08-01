import torch
import torch.nn.functional as F
from typing import Optional, Union, Dict, List, Tuple, Callable, Any
from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from dataclasses import dataclass, field
from trl.trainer.utils import empty_cache
from trl.models.utils import unwrap_model_for_generation
import numpy as np

from .drpo_trainer_new import DRPOTrainer
from .drpo_config_new import DRPOConfig
from .drpo_utils_new import DPOStyleRewardNetwork, GPMwithRewardNetwork


@dataclass
class DrDRPOConfig(DRPOConfig):
    """Configuration for Dr.DRPO (Doubly Robust DRPO) training."""
    beta_prime: float = field(default=1.0, metadata={"help": "Beta prime for Dr.DRPO weighting"})
    use_dpo_preference: bool = field(default=True, metadata={"help": "Use DPO-based preference as g"})
    weight_clip_range: Tuple[float, float] = field(
        default=(0.01, 100.0), 
        metadata={"help": "Range for clipping sample weights to avoid extreme values"}
    )
    use_dpo_style_reward: bool = field(
        default=False,
        metadata={"help": "Whether to use DPO-style reward model as preference model"}
    )
    dpo_reward_beta: float = field(
        default=0.1,
        metadata={"help": "Beta parameter for DPO-style reward computation"}
    )

    dpo_ref_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to reference model for DPO-style rewards"}
    )

class DrDRPOTrainer(DRPOTrainer):
    """
    Dr.DRPO (Doubly Robust DRPO) Trainer with sample weighting for noisy data.
    
    Key modifications:
    1. Computes DPO-based preference scores as weights
    2. Normalizes weights by sum (not mean)
    3. Applies weights to term_dm and IS terms
    4. Tracks weight statistics and noise metrics
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.args and self.args.use_preference_model and self.args.preference_model_path:
            if self.args.use_dpo_style_reward:
                self.preference_model = DPOStyleRewardNetwork(
                    model_name_or_path=args.preference_model_path,
                    ref_model_name_or_path=args.dpo_ref_model_path,
                    beta=args.dpo_reward_beta,
                    device=self.accelerator.device,
                    pad_token_id=self.processing_class.pad_token_id,
                    bf16=args.bf16,
                )
                self.args.preference_model_type="bt"   
        # Extract Dr.DRPO specific parameters from config if available
        if hasattr(self.args, 'beta_prime'):
            self.beta_prime = self.args.beta_prime
        else:
            self.beta_prime = 1.0
            
        if hasattr(self.args, 'use_dpo_preference'):
            self.use_dpo_preference = self.args.use_dpo_preference
        else:
            self.use_dpo_preference = True
            
        if hasattr(self.args, 'weight_clip_range'):
            self.weight_clip_range = self.args.weight_clip_range
        else:
            self.weight_clip_range = (0.01, 100.0)
        
        if hasattr(self.args, 'dpo_reward_beta'):
            self.dpo_reward_beta = self.args.dpo_reward_beta
        else:
            self.dpo_reward_beta = 0.1
        
        # Add Dr.DRPO specific stats
        self.stats.update({
            "drdrpo/weights_mean": [],
            "drdrpo/weights_std": [],
            "drdrpo/weights_min": [],
            "drdrpo/weights_max": [],
            "drdrpo/effective_sample_size": [],
            "drdrpo/noise_ratio_estimate": [],
            "drdrpo/low_weight_ratio": [],  # Ratio of samples with weight < 1/batch_size
            "drdrpo/weight_entropy": [],  # Entropy of weight distribution
        })
        
        # Track weight history for analysis
        # self.weight_history = []
    
    def _compute_dpo_preference_weight(
        self,
        chosen_logprobs: torch.Tensor,
        chosen_ref_logprobs: torch.Tensor,
        rejected_logprobs: torch.Tensor,
        rejected_ref_logprobs: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DPO-based preference score to use as sample weight.
        
        Returns:
            weight = sigmoid((chosen_logprobs - ref_chosen) - (rejected_logprobs - ref_rejected))
        """
        # Sum log probabilities over sequence length
        chosen_logprobs_sum = (chosen_logprobs * chosen_mask).sum(dim=1)
        chosen_ref_logprobs_sum = (chosen_ref_logprobs * chosen_mask).sum(dim=1)
        rejected_logprobs_sum = (rejected_logprobs * rejected_mask).sum(dim=1)
        rejected_ref_logprobs_sum = (rejected_ref_logprobs * rejected_mask).sum(dim=1)
        
        # Compute log probability ratios
        chosen_ratio = chosen_logprobs_sum - chosen_ref_logprobs_sum
        rejected_ratio = rejected_logprobs_sum - rejected_ref_logprobs_sum
        
        # DPO preference score
        preference_logits = self.args.beta * (chosen_ratio - rejected_ratio)
        weights = torch.sigmoid(preference_logits)
        
        # Apply beta_prime scaling
        if self.beta_prime != 1.0:
            # weights = exp(log(weights/beta_prime)) = weights^(1/beta_prime)
            weights = torch.pow(weights + 1e-8, 1 / (self.beta_prime + 1e-7))
        
        # Clip weights to avoid extreme values
        weights = torch.clamp(weights, self.weight_clip_range[0], self.weight_clip_range[1])
        
        return weights
    
    def _estimate_noise_metrics(self, weights: torch.Tensor) -> Dict[str, float]:
        """
        Estimate noise-related metrics from weight distribution.
        """
        batch_size = weights.shape[0]
        uniform_weight = 1.0 / batch_size
        
        # Ratio of samples with weight below uniform
        low_weight_ratio = (weights < uniform_weight).float().mean().item()
        
        # Weight entropy (higher entropy = more uniform distribution)
        # Normalize weights to probabilities
        probs = weights / weights.sum()
        weight_entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
        max_entropy = np.log(batch_size)  # Maximum possible entropy
        normalized_entropy = weight_entropy / max_entropy
        
        # Estimate noise ratio based on weight distribution
        # Samples with very low weights are likely noisy
        weight_threshold = 0.5 * uniform_weight  # Half of uniform weight
        noise_ratio_estimate = (weights < weight_threshold).float().mean().item()
        
        return {
            "low_weight_ratio": low_weight_ratio,
            "weight_entropy": normalized_entropy,
            "noise_ratio_estimate": noise_ratio_estimate,
        }
    
    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        """
        Modified DRPO training step with Dr.DRPO sample weighting.
        
        This implementation:
        1. Computes standard DRPO components (term_dm, IS terms)
        2. Computes DPO-based weights for each sample
        3. Applies weights to the loss terms
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
        
        # === Step 1: Standard DRPO forward passes ===
        # First compute all the log probabilities we need
        
        # Compute policy log probs for chosen and rejected
        chosen_logprobs = self._forward(model, prompt_ids, prompt_mask, chosen_ids, chosen_mask)
        rejected_logprobs = self._forward(model, prompt_ids, prompt_mask, rejected_ids, rejected_mask)
        
        # Compute reference log probs
        with torch.no_grad():
            if self.ref_model is None:
                with self.model.disable_adapter():
                    chosen_ref_logprobs = self._forward(
                        self.model, prompt_ids, prompt_mask, chosen_ids, chosen_mask
                    )
                    rejected_ref_logprobs = self._forward(
                        self.model, prompt_ids, prompt_mask, rejected_ids, rejected_mask
                    )
            else:
                chosen_ref_logprobs = self._forward(
                    self.ref_model, prompt_ids, prompt_mask, chosen_ids, chosen_mask
                )
                rejected_ref_logprobs = self._forward(
                    self.ref_model, prompt_ids, prompt_mask, rejected_ids, rejected_mask
                )
        
        # === Step 2: Compute Dr.DRPO weights ===
        if self.use_dpo_preference:
            # Use DPO-based preference as weight
            weights = self._compute_dpo_preference_weight(
                chosen_logprobs, chosen_ref_logprobs,
                rejected_logprobs, rejected_ref_logprobs,
                chosen_mask, rejected_mask
            )
        else:
            # Use standard preference score as weight
            g_chosen_rejected = self._compute_preference_scores_batch(
                prompt_ids, prompt_mask,
                chosen_ids, chosen_mask,
                rejected_ids, rejected_mask,
                chosen_texts, rejected_texts
            )
            weights = g_chosen_rejected
            if self.beta_prime != 1.0:
                weights = torch.pow(weights + 1e-8, self.beta_prime)
            weights = torch.clamp(weights, self.weight_clip_range[0], self.weight_clip_range[1])
        
        # Normalize weights by sum
        weights = weights / weights.sum()
        
        # === Step 3: Generate MC samples and compute DRPO terms ===
        mc_samples = []
        mc_logprobs_list = []
        mc_ref_logprobs_list = []
        mc_kl_total_list = []
        mc_entropy_list = []
        
        # Generate MC samples
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
        
        # Compute log probs for MC samples
        for mc_ids, mc_mask in mc_samples:
            mc_logprobs = self._forward(model, prompt_ids, prompt_mask, mc_ids, mc_mask)
            mc_logprobs_list.append(mc_logprobs)
            
            # Reference log probs
            with torch.no_grad():
                if self.ref_model is None:
                    with self.model.disable_adapter():
                        mc_ref_logprobs = self._forward(
                            self.model, prompt_ids, prompt_mask, mc_ids, mc_mask
                        )
                else:
                    mc_ref_logprobs = self._forward(
                        self.ref_model, prompt_ids, prompt_mask, mc_ids, mc_mask
                    )
                mc_ref_logprobs_list.append(mc_ref_logprobs)
                
                # Compute KL
                _, total_kl = self._compute_kl_divergence(
                    mc_logprobs, mc_ref_logprobs, mc_mask,
                    kl_type=self.args.kl_type
                )
                mc_kl_total_list.append(total_kl)

                # Compute entropy H(π) = -E[log π]
                entropy = -(mc_logprobs * mc_mask).sum(dim=1)
                mc_entropy_list.append(entropy)
        
        # Compute preference scores
        g_chosen_rejected = self._compute_preference_scores_batch(
            prompt_ids, prompt_mask,
            chosen_ids, chosen_mask,
            rejected_ids, rejected_mask,
            chosen_texts, rejected_texts
        )
        
        # === Step 4: Compute DRPO loss terms with Dr.DRPO weighting ===
        
        # Direct Method (DM) term
        term_dm = torch.zeros(batch_size, device=device)
        for (mc_ids, mc_mask), mc_logprobs in zip(mc_samples, mc_logprobs_list):
            # g(mc, rejected) and g(mc, chosen)
            g_mc_rejected = self._compute_preference_scores_batch(
                prompt_ids, prompt_mask, mc_ids, mc_mask, rejected_ids, rejected_mask
            )
            g_mc_chosen = self._compute_preference_scores_batch(
                prompt_ids, prompt_mask, mc_ids, mc_mask, chosen_ids, chosen_mask
            )
            
            # Weight by log probability
            mc_logprobs_sum = (mc_logprobs * mc_mask).sum(dim=1)
            term_dm += (g_mc_rejected + g_mc_chosen) * mc_logprobs_sum
        
        term_dm = term_dm / (2 * self.args.num_monte_carlo_samples)
        
        # Apply Dr.DRPO weighting to term_dm
        weighted_term_dm = (term_dm * weights).sum()  # Not mean!
        
        # Importance Sampling (IS) terms
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
        
        # IS loss term with Dr.DRPO weighting
        residual = 1 - g_chosen_rejected
        is_term = (
            is_ratio_chosen * residual * chosen_logprobs_sum -
            is_ratio_rejected * residual * rejected_logprobs_sum
        )
        
        # Apply Dr.DRPO weighting to IS term
        weighted_is_loss = -(is_term * weights).sum()  # Not mean!
        
        # === Step 5: Final loss computation ===
        drpo_loss = -weighted_term_dm + weighted_is_loss
        
        # KL regularization (using mean as in original)
        kl_loss = torch.stack(mc_kl_total_list).mean()
        
        # Total loss
        loss = drpo_loss + self.beta * kl_loss
        
        # === Step 6: Log statistics ===
        with torch.no_grad():
            # Compute noise metrics
            noise_metrics = self._estimate_noise_metrics(weights)
            
            # Log Dr.DRPO specific metrics
            self.stats["drdrpo/weights_mean"].append(weights.mean().item())
            self.stats["drdrpo/weights_std"].append(weights.std().item())
            self.stats["drdrpo/weights_min"].append(weights.min().item())
            self.stats["drdrpo/weights_max"].append(weights.max().item())
            self.stats["drdrpo/effective_sample_size"].append(
                1.0 / (weights ** 2).sum().item()
            )
            self.stats["drdrpo/noise_ratio_estimate"].append(
                noise_metrics["noise_ratio_estimate"]
            )
            self.stats["drdrpo/low_weight_ratio"].append(
                noise_metrics["low_weight_ratio"]
            )
            self.stats["drdrpo/weight_entropy"].append(
                noise_metrics["weight_entropy"]
            )
            
            # Store weight history
            # self.weight_history.append(weights.detach().cpu().numpy())
            
            all_g_mc_rejected = []
            all_g_mc_chosen = []
            all_mc_lengths = []
            all_mc_contains_eos = []
            all_mc_logprobs_sum = []
            all_mc_ref_logprobs_sum = []
            
            for (mc_ids, mc_mask), mc_logprobs, mc_ref_logprobs in zip(mc_samples, mc_logprobs_list, mc_ref_logprobs_list):
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

                mc_ref_logprobs_sum = (mc_ref_logprobs * mc_mask).sum(dim=1)
                all_mc_ref_logprobs_sum.append(mc_ref_logprobs_sum)
            
            # Stack all MC samples
            all_g_mc_rejected = torch.stack(all_g_mc_rejected)  # [num_mc, batch_size]
            all_g_mc_chosen = torch.stack(all_g_mc_chosen)
            all_mc_lengths = torch.stack(all_mc_lengths)
            all_mc_contains_eos = torch.stack(all_mc_contains_eos)
            all_mc_logprobs_sum = torch.stack(all_mc_logprobs_sum)
            all_mc_ref_logprobs_sum = torch.stack(all_mc_ref_logprobs_sum)
            
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

            mc_ref_logprobs_gathered = self.accelerator.gather_for_metrics(all_mc_logprobs_sum.flatten())
            self.stats["logps/generated_ref_mean"].append(mc_ref_logprobs_gathered.mean().item())
            
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
    
    # def save_weight_analysis(self, output_path: str) -> None:
    #     """
    #     Save weight history and analysis for noise detection.
    #     """
    #     import json
        
    #     if not self.weight_history:
    #         print("No weight history available. Train the model first.")
    #         return
        
    #     # Convert weight history to numpy array
    #     weight_history = np.array(self.weight_history)
        
    #     # Compute statistics across training
    #     analysis = {
    #         "config": {
    #             "beta_prime": self.beta_prime,
    #             "use_dpo_preference": self.use_dpo_preference,
    #             "weight_clip_range": list(self.weight_clip_range),
    #         },
    #         "weight_statistics": {
    #             "mean": float(np.mean(weight_history)),
    #             "std": float(np.std(weight_history)),
    #             "min": float(np.min(weight_history)),
    #             "max": float(np.max(weight_history)),
    #         },
    #         "per_sample_stats": {
    #             "mean_weights": weight_history.mean(axis=0).tolist(),
    #             "std_weights": weight_history.std(axis=0).tolist(),
    #         },
    #         "training_steps": len(weight_history),
    #         "batch_size": weight_history.shape[1] if weight_history.ndim > 1 else 1,
    #     }
        
    #     # Identify potentially noisy samples (consistently low weight)
    #     if weight_history.ndim > 1 and weight_history.shape[0] > 1:
    #         mean_weights_per_sample = weight_history.mean(axis=0)
    #         threshold = np.percentile(mean_weights_per_sample, 10)
    #         noisy_indices = np.where(mean_weights_per_sample < threshold)[0].tolist()
    #         analysis["potentially_noisy_indices"] = noisy_indices
    #         analysis["noise_threshold"] = float(threshold)
        
    #     with open(output_path, 'w') as f:
    #         json.dump(analysis, f, indent=2)
        
    #     print(f"Weight analysis saved to {output_path}")