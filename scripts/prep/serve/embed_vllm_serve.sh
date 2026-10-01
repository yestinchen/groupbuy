#!/bin/bash

set -euo pipefail

# Generate server uses port 8123 (see tools_vllm_serve.sh).
# This script runs the embedding server on 8124 and defaults to GPU 0.

CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Embedding-0.6B \
    --runner pooling \
    --convert embed \
    --gpu-memory-utilization 0.50 \
    --tensor-parallel-size 1 \
    --port 8124 \
    --trust-remote-code \
    --enforce-eager \
    --max-model-len 4096
