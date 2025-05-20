from dataclasses import dataclass, field
from typing import Optional, Union, Callable

from transformers import TrainingArguments

@dataclass
class DRPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`OnlineDRPOTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse]
    """

    learning_rate: float = field(
        default=5e-7,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of"
            "transformers.TrainingArguments."
        }
    )

    model_and_preference_share_basemodel: bool = field(
        default=False,
        metadata={"help": "Whether the model and preference model share the same base model (e.g. both from Qwen2.5 or Pythia...)"},
    )

    preference_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the preference model."
        }
    )

    preference_model_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Additional arguments for the preference model, e.g. `{'indifferent': False, 'random': False, 'reverse': False}`."
        }
    )

    max_new_tokens: int = field(
        default=256,
        metadata={"help": "Maximum number of tokens to generate per completion."},
    )
    
    max_length: int = field(
        default=256,
        metadata={
            "help": "Maximum total length of the sequence (prompt + completion) used to compute log probabilities. If "
            "the sequence exceeds this limit, the leftmost tokens will be truncated to preserve as much of the "
            "completion as possible."
        },
    )

    generate_temperature: float = field(
        default=0.9,
        metadata={"help": "Temperature for sampling and policy model. The higher the temperature, the more random the completions."},
    )
    
    missing_eos_penalty: Optional[float] = field(
        default=None,
        metadata={
            "help": "Penalty applied to the score when the model fails to generate an EOS token. This is useful to "
            "encourage to generate completions shorter than the maximum length (`max_new_tokens`). The penalty must be "
            "a positive value."
        },
    )
    
    beta: list[float] = field(
        default_factory=lambda: [0.1],
        metadata={
            "help": "Parameter controlling the deviation from the reference model. Higher β means less deviation from "
            "the reference model. For the IPO loss (`loss_type='ipo'`), β is the regularization parameter denoted by "
            "τ in the [paper](https://huggingface.co/papers/2310.12036). If a list of floats is provided then the β "
            "is selected for each new epoch and the last β is used for the rest of the epochs."
        },
    )

    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model."},
    )

    ds_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,"
            "improving generation speed. However, disabling this option allows training models that exceed the VRAM capacity of a single GPU,"
            "albeit at the cost of slower generation."
        },
    )

    num_astar: int = field(
        default=1,
        metadata={
            "help": "Number of newly generated completions to compare with the reference model."
        }
    )
    
    tools: Optional[list[Union[dict, Callable]]] = field(
        default=None,
        metadata={
            "help": "List of tools (callable functions) that will be accessible to the model. If the template does "
            "not support function calling, this argument will have no effect."
        },
    )

    max_prompt_length: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum length of the prompt."},
    )
    max_completion_length: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum length of the completion."},
    )

    precompute_preference_score: bool = field(
        default=False,
        metadata={"help": "Whether to precompute the preference score for the dataset."},
    )

    logging_steps: int = field(
        default=10,
        metadata={"help": "Number of steps to log the training loss."},
    )

    num_train_epochs: int = field(
        default=2,
        metadata={"help": "Number of training epochs."},
    )

    is_bt_model: bool = field(
        default=True,
        metadata={"help": "Whether the preference model uses BT framework."},
    )

    preference_model_id: Optional[str] = field(
        default="siebert/sentiment-roberta-large-english",
        metadata={"help": "Model ID of the preference model."},
    )

    preference_model_revision: Optional[str] = field(
        default=None,
        metadata={"help": "Revision of the preference model."},
    )

    ratio_processing: Union[str, None] = field(
        default=None,
        metadata={"help": "Processing method for the Importance Sampling ratio. if clip, you need better to provide the clipbound. "
        "Options include `clip`, `self_normalize`, `None`. Default to None."},
    )

    clipbound: Optional[float] = field(
        default=10.0,
        metadata={"help": "Clip upper bound for the Importance Sampling ratio, default to 10.0."},
    )

    forward_temperature: Optional[float] = field(
        default=0.9,
        metadata={"help": "Temperature for the forward pass of ref_model."},
    )

    loss1_only: bool = field(
        default=False,
        metadata={"help": "Whether to only use the Importance Sampling estimator"},
    )
    
    loss2_only: bool = field(
        default=False,
        metadata={"help": "Whether to only use the Direct Method estimator"},
    )

    eos_after_completion: bool = field(
        default=False,
        metadata={"help": "Whether to add eos token after the completion., choose False when your data applies chat template, choose True when your data is not and you want your generation contain the eos"},
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model."},
    )


    def __post_init__(self):
        super().__post_init__()
        if hasattr(self.beta, "__len__") and len(self.beta) == 1:
            self.beta = self.beta[0]


