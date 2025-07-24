# drpo_config_fsdp.py - Complete version
from dataclasses import dataclass, field
from typing import Optional, Literal, List, Union
from .drpo_config_new import DRPOConfig

@dataclass
class DRPOConfigFSDP(DRPOConfig):
    """DRPO configuration optimized for FSDP."""
    
    # FSDP configuration string
    # Options: 
    # - "full_shard" or "shard_grad_op" (sharding strategy)
    # - "auto_wrap" (automatic wrapping)
    # - "offload" (CPU offloading)
    # Example: "full_shard auto_wrap" or "shard_grad_op auto_wrap offload"
    fsdp: Optional[Union[str, List[str]]] = field(
        default="shard_grad_op auto_wrap",
        metadata={
            "help": (
                "FSDP configuration. Pass a string with space-separated options. "
                "E.g., 'full_shard auto_wrap' or 'shard_grad_op auto_wrap'."
            )
        }
    )
    
    fsdp_transformer_layer_cls_to_wrap: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Transformer layer class to wrap. If None, will auto-detect. "
                "Cannot be used with fsdp_min_num_params."
            )
        }
    )
    
    # These are additional FSDP options you might want
    fsdp_backward_prefetch_policy: Optional[str] = field(
        default="BACKWARD_PRE",
        metadata={
            "help": "FSDP backward prefetch policy. Options: BACKWARD_PRE, BACKWARD_POST"
        }
    )
    
    fsdp_state_dict_type: Optional[str] = field(
        default="FULL_STATE_DICT",
        metadata={
            "help": "FSDP state dict type for checkpointing. Options: FULL_STATE_DICT, LOCAL_STATE_DICT, SHARDED_STATE_DICT"
        }
    )
    
    fsdp_auto_wrap_policy: Optional[str] = field(
        default="TRANSFORMER_BASED_WRAP",
        metadata={
            "help": "FSDP auto wrap policy. Options: TRANSFORMER_BASED_WRAP, SIZE_BASED_WRAP"
        }
    )
    
    fsdp_cpu_offload: bool = field(
        default=False,
        metadata={"help": "Whether to offload parameters and gradients to CPU"}
    )
    
    # Training optimizations for FSDP
    gradient_checkpointing: bool = field(default=True)
    save_safetensors: bool = field(default=True)
    
    def __post_init__(self):
        super().__post_init__()
        
        # Ensure FSDP-compatible settings
        if self.fsdp:
            self.ddp_find_unused_parameters = False
            
            # If using auto_wrap without specifying layer class,
            # HuggingFace will try to auto-detect it
            if "auto_wrap" in str(self.fsdp) and not self.fsdp_transformer_layer_cls_to_wrap:
                print("FSDP auto_wrap enabled without specific layer class - will auto-detect")