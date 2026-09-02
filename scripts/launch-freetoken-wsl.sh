#!/usr/bin/env bash
set -euo pipefail

pid_file="$1"
shift

export CUDA_HOME=/usr/local/cuda-13.3
export TVM_FFI_CUDA_ARCH_LIST=12.0
export CUDA_VISIBLE_DEVICES=GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0
export PATH="/home/rba90/.freetoken-qwen38/venv/bin:/usr/local/cuda-13.3/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-13.3/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

rm -f -- "$pid_file"
setsid "$@" &
child=$!
printf '%s\n' "$child" > "$pid_file"
wait "$child"
