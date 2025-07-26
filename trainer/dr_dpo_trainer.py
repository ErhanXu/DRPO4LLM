import torch
import torch.nn.functional as F
from typing import Dict, Union, Any, Optional, Tuple
from contextlib import nullcontext
from torch.cuda.amp import autocast

from transformers import PreTrainedModel
from trl import DPOTrainer, DPOConfig


class DrDPOTrainer(DPOTrainer):
    """
    Dr. DPO (Distributionally Robustifying DPO) Trainer.
    
    This trainer implements the Dr. DPO algorithm from "Towards Robust Alignment of Language Models:
    Distributionally Robustifying Direct Preference Optimization" which enhances DPO's robustness 
    to both pointwise and pairwise noise in preference datasets.
    
    Dr. DPO applies a soft-minimum aggregation over the batch losses instead of the arithmetic mean,
    which makes it more robust to noisy preference pairs.
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
                The β' parameter that controls the balance between exploration and exploitation:
                - β' < 1.0: More conservative (focuses on low-loss samples, robust to noise)
                - β' = 1.0: Default balanced behavior  
                - β' > 1.0: More explorative (considers all samples more equally)
                - β' → 0: Approaches min(losses) 
                - β' → ∞: Approaches mean(losses) (standard DPO)
            
        All other parameters are the same as DPOTrainer.
        """
        # Store beta_prime before calling super().__init__
        self.beta_prime = beta_prime
        
        # Call parent constructor
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
    
    def compute_loss(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Override compute_loss to apply Dr. DPO transformation to the final loss.
        
        The Dr. DPO loss aggregation is:
        L_Dr.DPO = -β' * log(mean(exp(-losses / β')))
        
        This is a soft minimum that:
        - Gives higher weight to samples with lower loss (likely correct preferences)
        - Gives lower weight to samples with higher loss (likely noisy preferences)
        - Smoothly interpolates between min (β'→0) and mean (β'→∞)
        """
        # Get the compute loss context manager (for mixed precision training)
        compute_loss_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )
        
        with compute_loss_context_manager:
            # Get per-sample losses and metrics using the parent class method
            losses, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")
            
            # Apply Dr. DPO transformation
            # Using logsumexp for numerical stability:
            # log(mean(exp(x))) = log(sum(exp(x))) - log(N)
            # = logsumexp(x) - log(N)
            dr_dpo_loss = -self.beta_prime * (
                torch.logsumexp(-losses / self.beta_prime, dim=0) 
                - torch.log(torch.tensor(losses.shape[0], dtype=losses.dtype, device=losses.device))
            )
        
        # Check if we need to move the loss to a different device
        # In DPOTrainer, losses are already on the correct device from get_batch_loss_metrics
        # But we ensure it's on args.device for consistency with the parent class
        if hasattr(self.args, 'device') and dr_dpo_loss.device != self.args.device:
            dr_dpo_loss = dr_dpo_loss.to(self.args.device)
        
        # Store metrics for logging
        self.store_metrics(metrics, train_eval="train")
        
        # Add Dr. DPO specific metric
        metrics['beta_prime'] = self.beta_prime
        
        if return_outputs:
            return dr_dpo_loss, metrics
        
        return dr_dpo_loss
    
    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """
        Override evaluation_loop to apply Dr. DPO transformation during evaluation.
        """
        # During evaluation, we also want to use Dr. DPO aggregation
        # We'll do this by temporarily overriding the loss computation
        
        # Store the original compute_loss method
        original_compute_loss = super().compute_loss
        
        def eval_compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
            """Apply Dr. DPO transformation during evaluation."""
            with torch.no_grad():
                losses, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")
                
                # Apply Dr. DPO transformation
                dr_dpo_loss = -self.beta_prime * (
                    torch.logsumexp(-losses / self.beta_prime, dim=0) 
                    - torch.log(torch.tensor(losses.shape[0], dtype=losses.dtype, device=losses.device))
                )
                
                if hasattr(self.args, 'device') and dr_dpo_loss.device != self.args.device:
                    dr_dpo_loss = dr_dpo_loss.to(self.args.device)
                
                self.store_metrics(metrics, train_eval="eval")
                
                if return_outputs:
                    return dr_dpo_loss, metrics
                
                return dr_dpo_loss
        
        # Temporarily replace compute_loss for evaluation
        self.compute_loss = eval_compute_loss
        
        try:
            # Call parent's evaluation_loop
            output = super().evaluation_loop(
                dataloader=dataloader,
                description=description,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            # Restore original compute_loss
            self.compute_loss = original_compute_loss.__get__(self, type(self))
        
        return output


# Usage example
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import Dataset
    
    # Initialize model and tokenizer
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create a dummy dataset
    dataset = Dataset.from_dict({
        "prompt": ["Hello", "Hi there"],
        "chosen": [" world!", " friend!"],
        "rejected": [" earth!", " buddy!"],
    })
    
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
        logging_dir="./logs",
        beta=0.1,  # Standard DPO beta
    )
    
    # Initialize Dr. DPO trainer
    trainer = DrDPOTrainer(
        model=model,
        ref_model=model,  # In practice, use a separate reference model
        args=training_args,
        beta_prime=1.0,  # Dr. DPO specific parameter
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=tokenizer,
    )
    
    # Train
    trainer.train()