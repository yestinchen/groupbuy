# vllm serve qwen/Qwen2.5-7B-Instruct-AWQ \
#     --gpu-memory-utilization 0.90 --tensor-parallel-size 1 \
#     --quantization awq --port 8123
#
# Qwen2.5 is the active serving target again for now.

CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8125 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --dtype float16 \
    --enforce-eager \
    --max-model-len 4096

# Keep the Qwen3.5 path commented out as a reference:
# vllm serve Qwen/Qwen3.5-4B \
#     --port 8123 \
#     --tensor-parallel-size 1 \
#     --gpu-memory-utilization 0.90 \
#     --dtype float16 \
#     --max-model-len 4096 \
#     --reasoning-parser qwen3 \
#     --language-model-only

# --max-model-len 4096 --tensor-parallel-size 4
