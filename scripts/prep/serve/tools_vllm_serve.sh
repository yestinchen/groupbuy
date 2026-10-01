# Port 8123 = generate; port 8124 = embedding (see embed_vllm_serve.sh).

vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8123 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --dtype float16 \
    --enforce-eager \
    --max-model-len 4096

# Qwen3.5 path kept for reference:
# vllm serve Qwen/Qwen3.5-4B \
#     --port 8123 \
#     --tensor-parallel-size 1 \
#     --gpu-memory-utilization 0.90 \
#     --dtype float16 \
#     --enforce-eager \
#     --max-model-len 4096 \
#     --reasoning-parser qwen3 \
#     --language-model-only

# --max-model-len 4096 --tensor-parallel-size 4
