import torch
import torch.nn.functional as F
from typing import Optional, Union, Dict, List, Tuple, Callable
from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from dataclasses import dataclass, field
# import numpy as np

from .drpo_trainer_new import DRPOTrainer
from .drpo_config_new import DRPOConfig


@dataclass
class DrDRPOConfig(DRPOConfig):
    """Configuration for Dr.DRPO (Doubly Robust DRPO) training."""
    beta_prime: float = field(default=1.0, metadata={"help": "Beta prime for Dr.DRPO weighting"})
    use_dpo_preference: bool = field(default=True, metadata={"help": "Use DPO-based preference as g"})
    weight_clip_range: Tuple[float, float] = field(
        default=(0.01, 100.0), 
        metadata={"help": "Range for clipping sample weights to avoid extreme values"}
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
        inputs: Dict[str, torch.Tensor],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Override training step to apply Dr.DRPO weighting.
        """
        model.train()
        
        # Extract inputs
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_attention_mask"]
        chosen_ids = inputs["chosen_ids"] 
        chosen_mask = inputs["chosen_attention_mask"]
        rejected_ids = inputs["rejected_ids"]
        rejected_mask = inputs["rejected_attention_mask"]
        
        device = prompt_ids.device
        batch_size = prompt_ids.shape[0]
        
        # Get text inputs if available (for judge-based preference)
        # prompt_texts = inputs.get("prompt")
        chosen_texts = inputs.get("chosen")
        rejected_texts = inputs.get("rejected")
        
        # Step 1: Forward pass for chosen and rejected
        with torch.no_grad():
            # Get reference log probabilities
            chosen_ref_logprobs = self._forward_and_get_log_probs(
                self.ref_model, chosen_ids, chosen_mask
            )
            rejected_ref_logprobs = self._forward_and_get_log_probs(
                self.ref_model, rejected_ids, rejected_mask
            )
        
        # Get policy log probabilities (with gradients)
        chosen_logprobs = self._forward_and_get_log_probs(
            model, chosen_ids, chosen_mask
        )
        rejected_logprobs = self._forward_and_get_log_probs(
            model, rejected_ids, rejected_mask
        )
        
        # Step 2: Generate MC samples and compute their log probs
        mc_samples = []
        mc_logprobs_list = []
        mc_ref_logprobs_list = []
        mc_kl_total_list = []
        mc_entropy_list = []
        
        for _ in range(self.args.num_monte_carlo_samples):
            with torch.no_grad():
                # Generate from current policy
                mc_ids, mc_mask = self._generate_from_policy(
                    model, prompt_ids, prompt_mask, 
                    temperature=self.args.temperature
                )
                mc_samples.append((mc_ids, mc_mask))
                
                # Get reference log probs for MC samples
                mc_ref_logprobs = self._forward_and_get_log_probs(
                    self.ref_model, mc_ids, mc_mask
                )
                mc_ref_logprobs_list.append(mc_ref_logprobs)
            
            # Get policy log probs for MC samples (with gradients)
            mc_logprobs = self._forward_and_get_log_probs(model, mc_ids, mc_mask)
            mc_logprobs_list.append(mc_logprobs)
            
            # Compute KL for regularization
            with torch.no_grad():
                total_kl = self._compute_kl_divergence(
                    mc_logprobs, mc_ref_logprobs, mc_mask,
                    kl_type=self.args.kl_type
                )
                mc_kl_total_list.append(total_kl)
                
                # Compute entropy
                entropy = -(mc_logprobs * mc_mask).sum(dim=1)
                mc_entropy_list.append(entropy)
        
        # Step 3: Compute preference scores
        g_chosen_rejected = self._compute_preference_scores_batch(
            prompt_ids, prompt_mask,
            chosen_ids, chosen_mask,
            rejected_ids, rejected_mask,
            chosen_texts, rejected_texts
        )
        
        # Step 4: Compute Dr.DRPO weights
        if self.use_dpo_preference:
            # Use DPO-based preference as weight
            weights = self._compute_dpo_preference_weight(
                chosen_logprobs, chosen_ref_logprobs,
                rejected_logprobs, rejected_ref_logprobs,
                chosen_mask, rejected_mask
            )
        else:
            # Use standard preference score as weight
            weights = g_chosen_rejected
            if self.beta_prime != 1.0:
                weights = torch.pow(weights + 1e-8, self.beta_prime)
            weights = torch.clamp(weights, self.weight_clip_range[0], self.weight_clip_range[1])
        
        # Normalize weights by sum (not mean)
        weights = weights / weights.sum()
        
        # Step 5: Compute weighted term_dm
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
        
        # Apply Dr.DRPO weighting to term_dm
        weighted_term_dm = (term_dm * weights).sum()  # Not mean!
        
        # Step 6: Compute weighted IS terms
        chosen_logprobs_sum = (chosen_logprobs * chosen_mask).sum(dim=1)
        rejected_logprobs_sum = (rejected_logprobs * rejected_mask).sum(dim=1)
        
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
        
        # Step 7: Compute final loss
        drpo_loss = -weighted_term_dm + weighted_is_loss
        
        # KL regularization (using mean as in original)
        kl_loss = torch.stack(mc_kl_total_list).mean()
        
        # Total loss
        loss = drpo_loss + self.beta * kl_loss
        
        # Log statistics
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
            
            # Store weight history for later analysis
            # self.weight_history.append(weights.detach().cpu().numpy())
            
            # Log standard DRPO metrics (for original terms, not weighted)
            self.stats["drpo/term_dm"].append(term_dm.mean().item())
            self.stats["drpo/term_is"].append(is_term.mean().item())
            self.stats["drpo/preference_score"].append(g_chosen_rejected.mean().item())
            self.stats["objective/kl"].append(kl_loss.item())
            self.stats["loss/total"].append(loss.item())
        
        return loss
    
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