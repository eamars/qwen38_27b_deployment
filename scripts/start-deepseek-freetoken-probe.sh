#!/usr/bin/env bash
set -euo pipefail
context="${1:-180000}"
slots="${2:-1536}"
if (( context > 180000 )); then
  echo "Context above the user's 180K maximum is disabled." >&2
  exit 2
fi
kv_tokens=$((context / 128 * 128))
graphs="${3:-1}"
window_pages="${4:-64}"
prefill="${5:-4096}"
extra_args=()
if [[ "${FREETOKEN_MTP5:-0}" == 1 ]]; then
  export FREETOKEN_MTP_SIDECAR=/home/rba90/deepseek-freetoken-probe/mtp5
  export FREETOKEN_DSPARK_LAYER_BASE=0
  extra_args+=(--speculative-dspark)
fi
if [[ "${FREETOKEN_NGRAM:-0}" == 1 ]]; then
  extra_args+=(--speculative-ngram)
fi
output=/home/rba90/deepseek-freetoken-probe
model=/home/rba90/models/DeepSeek-V4-Flash-0731-IQ3_XXS/IQ3_XXS/DeepSeek-V4-Flash-0731-IQ3_XXS-00001-of-00004.gguf
log="$output/server-${context}-${slots}-${graphs}${FREETOKEN_LOG_SUFFIX:-}.log"
test -f "$output/download-verified.json"
exec bash /mnt/c/workspace/qwen38_27b/scripts/deepseek-freetoken-env.sh \
  python -u -m freetoken.cli serve --model "$model" \
  --served-model-name deepseek-v4-flash-gpu-probe \
  --host 0.0.0.0 --port 1919 --tensor-parallel-size 1 \
  --dtype bfloat16 --memory-ratio 0.90 --max-running-requests 1 \
  --moe-backend offload --moe-cpu-layers 0 --expert-load serial \
  --moe-cache-size "$slots" --kv-reserve-tokens "$kv_tokens" --num-tokens "$kv_tokens" \
  --max-seq-len-override "$context" --max-prefill-length "$prefill" \
  --swa-num-pages-override "$window_pages" \
  --max-output-tokens 512 --cuda-graph-max-bs "$graphs" --disable-pynccl \
  --cache-type radix --decode-log-interval 20 --reasoning-parser auto \
  "${extra_args[@]}" \
  2>&1 | tee "$log"
