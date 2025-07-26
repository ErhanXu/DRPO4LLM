import torch
import torch.nn.functional as F
from typing import Dict, Union, Any, Optional, Tuple
from contextlib import nullcontext
from torch.cuda.amp import autocast

from transformers import PreTrainedModel
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset


class DrDPOTrainer(DPOTrainer):
    """
    Dr. DPO (Distributionally Robustifying DPO) Trainer.
    
    Implements the Dr. DPO algorithm from "Towards Robust Alignment of Language Models:
    Distributionally Robustifying Direct Preference Optimization".
    """
    
    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module, str],
        ref_model: Optional[Union[PreTrainedModel, torch.nn.Module, str]] = None,
        beta_prime: float = 1.0,  # Dr. DPO specific parameter
        args: Optional[DPOConfig] = None,
        data_collator: Optional[Any] = None,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        processing_class: Optional[Any] = None,
        compute_metrics: Optional[Any] = None,
        callbacks: Optional[Any] = None,
        optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[Tuple[type[torch.optim.Optimizer], Dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Any] = None,
        peft_config: Optional[Any] = None,
    ):
        """
        Initialize the Dr. DPO trainer.
        
        Args:
            beta_prime (`float`, *optional*, defaults to `1.0`):
                The β' parameter that controls robustness to pairwise noise.
                Lower values make the loss more conservative (robust to noise).
        """
        self.beta_prime = beta_prime
        
        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )
    
    def dpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        loss_type: str = "sigmoid",
        model_output: Optional[Dict[str, torch.FloatTensor]] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the Dr. DPO loss for a batch of policy and reference model log probabilities.
        
        Dr. DPO applies a transformation that makes the loss robust to pairwise noise by
        implicitly reweighting samples based on their loss values.
        """
        # First compute standard DPO losses and rewards
        losses, chosen_rewards, rejected_rewards = super().dpo_loss(
            chosen_logps=chosen_logps,
            rejected_logps=rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
        )
        
        # Apply Dr. DPO transformation
        # The Dr. DPO paper shows that the gradient can be written as:
        # ∇L_Dr.DPO = E[w(x,y_w,y_l) * ∇L_DPO(x,y_w,y_l)]
        # where w(x,y_w,y_l) = exp(-L_DPO(x,y_w,y_l) / β') / E[exp(-L_DPO / β')]
        
        # Compute importance weights for each sample
        weights = torch.exp(-losses.detach() / self.beta_prime)
        weights = weights / weights.mean()
        
        # Apply weights to losses
        # This gives the same gradient as the original Dr. DPO formulation
        dr_dpo_losses = losses * weights
        
        return dr_dpo_losses, chosen_rewards, rejected_rewards


# Usage example
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import Dataset
    
    # Initialize model and tokenizer
    model_name = "Qwen/Qwen3-1.7B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    from peft import LoraConfig
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    # Create a dummy dataset
    dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")  # Use subset for testing
    
    # Configure training
    training_args = DPOConfig(
        output_dir="./dr_dpo_output",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=100,
        learning_rate=1e-5,
        warmup_steps=10,
        # logging_dir="./logs",
        beta=0.1,  # Standard DPO beta
    )
    
    # Initialize Dr. DPO trainer
    trainer = DrDPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        beta_prime=1.0,  # Dr. DPO specific parameter
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    # Train
    trainer.train()