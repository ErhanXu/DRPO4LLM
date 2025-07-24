# drpo_trainer_fsdp.py - Complete fixed version
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
from .drpo_trainer_new import DRPOTrainer
from transformers.trainer import unwrap_model
from trl.models.utils import unwrap_model_for_generation

class DRPOTrainerFSDP(DRPOTrainer):
    """DRPO Trainer with FSDP-specific optimizations."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # FSDP-specific setup
        if self.accelerator.distributed_type == "FSDP":
            # Ensure reward model is not wrapped by FSDP
            if self.reward_model is not None:
                self.reward_model = self.reward_model.to(self.accelerator.device)
                self.reward_model.eval()
                for param in self.reward_model.parameters():
                    param.requires_grad = False
    
    def _prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for FSDP training with proper dtype."""
        model = super()._prepare_model(model)
        
        if self.accelerator.distributed_type == "FSDP":
            # Ensure model is in training mode
            model.train()
            
            # Fix dtype issues with PEFT
            if hasattr(model, 'peft_config'):
                # Ensure all LoRA modules are in correct dtype
                for name, module in model.named_modules():
                    if any(key in name for key in ['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B']):
                        # Convert to model dtype
                        if hasattr(module, 'weight'):
                            module.weight.data = module.weight.data.to(self.args.torch_dtype)
        
        return model
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training_step to handle FSDP-specific issues."""
        # Ensure model is in training mode with gradient checkpointing
        model.train()
        
        # For FSDP, we need to handle gradient checkpointing carefully
        if self.accelerator.distributed_type == "FSDP" and hasattr(model, 'gradient_checkpointing_enable'):
            # Temporarily store the state
            if not getattr(model, '_gradient_checkpointing_enabled', False):
                model.gradient_checkpointing_enable()
                model._gradient_checkpointing_enabled = True
        
        return super().training_step(model, inputs, num_items_in_batch)
    
# drpo_trainer_fsdp.py - Fixed _generate method
    def _generate(self, model, prompt_ids, prompt_mask, num_samples=1):
        """FSDP-aware generation with proper parameter gathering."""
        # Store original states
        was_training = model.training
        grad_ckpt_enabled = False
        
        # Check if gradient checkpointing is enabled
        if hasattr(model, 'is_gradient_checkpointing') and model.is_gradient_checkpointing:
            grad_ckpt_enabled = True
            model.gradient_checkpointing_disable()
        
        # Set to eval mode
        model.eval()
        
        try:
            with torch.no_grad():
                # Check if we're using FSDP
                if self.accelerator.distributed_type == "FSDP":
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, BackwardPrefetch
                    
                    # Find the actual FSDP wrapped model
                    fsdp_model = model
                    
                    # Check if the model or any of its submodules is wrapped with FSDP
                    is_fsdp = isinstance(model, FSDP)
                    if not is_fsdp:
                        # Check if any submodule is FSDP wrapped
                        for module in model.modules():
                            if isinstance(module, FSDP):
                                is_fsdp = True
                                break
                    
                    if is_fsdp:
                        # Use FSDP's context manager to gather full parameters
                        with FSDP.summon_full_params(model, writeback=False, recurse=True):
                            # Ensure all parameters are gathered
                            for module in model.modules():
                                if hasattr(module, 'weight'):
                                    # Force parameter to be contiguous
                                    if hasattr(module.weight, 'data'):
                                        module.weight.data = module.weight.data.contiguous()
                            
                            # Now generate with full parameters
                            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                                return super()._generate(model, prompt_ids, prompt_mask, num_samples)
                    else:
                        # Model not FSDP wrapped, generate normally
                        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                            return super()._generate(model, prompt_ids, prompt_mask, num_samples)
                else:
                    # Non-FSDP path
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                        return super()._generate(model, prompt_ids, prompt_mask, num_samples)
        
        finally:
            # Restore gradient checkpointing if it was enabled
            if grad_ckpt_enabled:
                model.gradient_checkpointing_enable()
            
            # Restore training mode
            if was_training:
                model.train()
    
    def _forward(self, model, prompt_ids, prompt_attention_mask, 
                 completion_ids, completion_attention_mask):
        """FSDP-optimized forward pass with dtype handling."""
        # Ensure inputs are in correct dtype
        if self.args.bf16:
            model_dtype = torch.bfloat16
        elif self.args.fp16:
            model_dtype = torch.float16
        else:
            model_dtype = torch.float32
        
        # No need to convert input_ids (they should be long)
        # But ensure model is in correct mode
        if self.accelerator.distributed_type == "FSDP":
            model.train()
        
        # Use autocast for mixed precision
        with torch.cuda.amp.autocast(enabled=self.args.bf16 or self.args.fp16, dtype=model_dtype):
            return super()._forward(
                model, prompt_ids, prompt_attention_mask,
                completion_ids, completion_attention_mask
            )