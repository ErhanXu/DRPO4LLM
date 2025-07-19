#!/bin/bash

# Set environment variables for optimal H100 performance
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16

# Number of GPUs to use
NUM_GPUS=4

# Launch with accelerate and deepspeed
accelerate launch \
  --config_file accelerate_config.yaml \
  --num_processes $NUM_GPUS \
  --num_machines 1 \
  --mixed_precision bf16 \
  --use_deepspeed \
  --deepspeed_config_file deepspeed_config.json \
  examples/hh/qwen7b.py
