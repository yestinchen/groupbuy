Create or refresh the dedicated conda environment for Qwen2.5 serving:

conda activate vllmenv
python -m pip install -U vllm==0.18.0

This latest public stable build was revalidated on this machine with Qwen2.5.

For the current generate server, use Qwen2.5:

vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8123 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --dtype float16 \
    --enforce-eager \
    --max-model-len 4096

# Qwen3.5 text-only reference, kept commented out for later:
# vllm serve Qwen/Qwen3.5-4B \
#     --port 8123 \
#     --tensor-parallel-size 1 \
#     --max-model-len 262144 \
#     --reasoning-parser qwen3 \
#     --language-model-only

For Qwen3.5 tool calling, add the Qwen3-Coder tool parser:

# vllm serve Qwen/Qwen3.5-4B \
#     --port 8123 \
#     --tensor-parallel-size 1 \
#     --max-model-len 262144 \
#     --reasoning-parser qwen3 \
#     --language-model-only \
#     --enable-auto-tool-choice \
#     --tool-call-parser qwen3_coder
