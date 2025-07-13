# drpo_config.py
from dataclasses import dataclass, field
from typing import Optional, Literal
from trl.trainer.online_dpo_config import OnlineDPOConfig


@dataclass
class DRPOConfig(OnlineDPOConfig):
    """
    Configuration class for DRPOTrainer.
    OnlineDPOConfig contains:
    - learning_rate: default = 5e-7 AdamW
    - reward_model_path: Path to the reward model.
    - judge: Name of judge to use
    - max_new_tokens: Maximum number of new tokens to generate.
    - max_length: Maximum length of the prompt+completion
    - temperature: temperature for sampling
    - missing_eos_penalty: penalty for missing EOS token
    - beta: beta value for the reward model.
    - disable_drop_out: Whether to disable dropout during training.
    - dataset_num_proc
    - use_vllm
    - gpu_memory_utilization:The vLLM memory utilization. The default value is 0.55.
    Extends OnlineDPOConfig with DRPO-specific parameters.
    """
    
    # Preference model settings
    use_preference_model: bool = field(
        default=False,
        metadata={"help": "Whether to use a custom preference model instead of reward model"}
    )
    preference_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the preference model"}
    )
    preference_model_type: Literal["bt", "general"] = field(
        default="bt",
        metadata={"help": "Type of preference model: 'bt' for Bradley-Terry or 'general'"}
    )
    
    # Monte Carlo sampling
    num_monte_carlo_samples: int = field(
        default=2,
        metadata={"help": "Number of Monte Carlo samples for expectation estimation"}
    )
    
    # Importance sampling control
    is_control_method: Literal["clip", "adaptive", "none"] = field(
        default="clip",
        metadata={"help": "Method for controlling IS ratios"}
    )
    is_clip_min: float = field(
        default=0.1,
        metadata={"help": "Minimum value for IS ratio clipping"}
    )
    is_clip_max: float = field(
        default=10.0,
        metadata={"help": "Maximum value for IS ratio clipping"}
    )
    
    # KL regularization type
    kl_type: Literal["k1", "k3"] = field(
        default="k3",
        metadata={"help": "Type of KL regularization: 'k1' (offline) or 'k3' (online)"}
    )
    
    # DeepSpeed Stage 3 optimization
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "Whether to gather model parameters for generation with DeepSpeed Stage 3. "
            "Improves generation speed but uses more memory."
        }
    )
    
    # Dataset processing
    max_prompt_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum length for prompts (truncated from left)"}
    )
    max_completion_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum length for completions (truncated from right)"}
    )
    
    # Evaluation parameters
    eval_with_generation: bool = field(
        default=True,
        metadata={"help": "Whether to generate samples during evaluation for quality metrics"}
    )
    
    eval_mc_samples: int = field(
        default=1,
        metadata={"help": "Number of MC samples to generate during evaluation"}
    )
    
    # If you want to track specific metrics
    metric_for_best_model: Optional[str] = field(
        default="eval_generated/win_rate_vs_rejected",
        metadata={"help": "Metric to use for selecting best model"}
    )
    
    greater_is_better: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether higher metric value is better"}
    )