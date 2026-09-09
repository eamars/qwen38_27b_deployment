#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0
export CUDA_HOME=/usr/local/cuda-13.3
export PATH="/home/rba90/.freetoken-qwen38/venv/bin:/usr/local/cuda-13.3/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.3/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${FREETOKEN_PROBE_RUNTIME:-/mnt/c/workspace/qwen38_27b/runtime/freetoken-deepseek-spec}/python"
export TVM_FFI_CUDA_ARCH_LIST=12.0
export TORCH_CUDA_ARCH_LIST=12.0a
export TORCH_EXTENSIONS_DIR=/home/rba90/deepseek-freetoken-probe/torch-extensions
export TVM_FFI_CACHE_DIR=/home/rba90/deepseek-freetoken-probe/tvm-cache
export MAX_JOBS=8
export FREETOKEN_DSV4_TOKENIZER_PATH=/home/rba90/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062
exec "$@"
