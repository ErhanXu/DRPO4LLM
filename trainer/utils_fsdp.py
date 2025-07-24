# utils_fsdp.py
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
import functools

def get_fsdp_config(model_name: str, sharding_strategy: str = "SHARD_GRAD_OP"):
    """Get FSDP configuration for specific model."""
    
    # Mixed precision config
    mixed_precision_config = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    # Sharding strategy mapping
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    
    # Auto wrap policy
    if "qwen" in model_name.lower():
        from transformers.models.qwen2 import Qwen2DecoderLayer
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Qwen2DecoderLayer},
        )
    elif "llama" in model_name.lower():
        from transformers.models.llama import LlamaDecoderLayer
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer},
        )
    else:
        # Fallback to size-based
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=1e8,
        )
    
    return {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision_config,
        "sharding_strategy": strategy_map[sharding_strategy],
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "device_id": torch.cuda.current_device(),
        "use_orig_params": True,  # Important for PEFT
    }