#!/bin/bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllmenv

echo "Updating vllmenv to the latest public stable vLLM release..."
python -m pip install -U vllm==0.18.0

echo "Current runtime versions:"
python - <<'PY'
import sys
import torch
import transformers
import vllm

print("python", sys.version.split()[0])
print("vllm", getattr(vllm, "__version__", "unknown"))
print("torch", torch.__version__)
print("transformers", transformers.__version__)
PY
