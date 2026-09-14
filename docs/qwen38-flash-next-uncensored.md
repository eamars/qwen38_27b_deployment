# Qwen3.8 Flash-Next uncensored deployment

Status (2026-09-12): the uncensored checkpoint now uses the same updated
FreeToken source and launcher path as the official deployment. Checkpoint
preparation and header verification passed, followed by successful Short4K
model loading and `/health` checks for both launchers. No benchmark or MTP
drafter was added.

## Retained configuration

| Setting | Uncensored deployment |
|---|---|
| Checkpoint | `orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4` |
| Pinned revision | `c1209bda15a6bbc4c68b585e93d40c0d85f50306` |
| WSL storage | `/home/rba90/models/Qwen3.8-Flash-Next-Uncensored-NVFP4` |
| Download | 183,535,693,914 bytes (170.93 GiB), excluding `.gitattributes` |
| GPU | RTX 5090, `GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0` |
| Endpoint | `http://127.0.0.1:1919/v1` |
| Served model name | `qwen38-next-uncensored-freetoken` |
| Default | Native256K, one concurrent request; capacity remains unvalidated |
| Expert placement | NVFP4 host banks, hot experts cached on GPU |
| Expert computation | `--moe-strategy offload --moe-cpu-layers auto --moe-cache-auto` |
| Dense weights | BF16 GPU buffers; source FP8 weights dequantized with channel scales |
| PLE | Original BF16 shards on disk, approximately 95.37 GiB |
| Memory ratio / cache | `0.90`, radix |

The complete source checkpoint is retained, including unused MTP and vision tensors. Serving remains text-only with MTP omitted, as in the existing FreeToken path. Both checkpoints are independent. No GGUF weights or checkpoint conversion are needed.

## Commands

Preview without loading anything:

```powershell
.\scripts\start-qwen38-flash-next-uncensored-freetoken.ps1 -Profile Short4K -DryRun
```

Start the same target-only FreeToken path as the official launcher, using the
uncensored checkpoint replacement:

```powershell
.\scripts\start-qwen38-flash-next-uncensored-freetoken.ps1 -Profile Short4K
```

The launcher defaults to one active request. Omitting `-Profile` selects
Native256K, as in the official launcher. This path has no MTP or
speculative drafter.

Stop only the process managed by the uncensored launcher:

```powershell
.\scripts\start-qwen38-flash-next-uncensored-freetoken.ps1 -Stop
```

The launcher stores a separate PID file at
`/tmp/qwen38-flash-next-uncensored-freetoken-1919.pid`, so stopping the
uncensored process cannot target the official launcher record. It uses the
same `launch-freetoken-wsl.sh` process-group lifecycle as the official path.

## Runtime reuse

The existing CUDA toolkit, compiled FreeToken kernels,
`/home/rba90/.freetoken-qwen38/venv` environment, and
`runtime/freetoken-a80b4d3` source tree are shared with the official launcher.
The venv is not reinstalled or repointed. Both launchers invoke the same
`launch-freetoken-wsl.sh` process setup and the same
`/home/rba90/.freetoken-qwen38/venv/bin/ft` executable; only the checkpoint,
served name, and PID file differ.

The shared source tree is at upstream commit
`953565667f3141c90d0f0eb469bb2655d2407140`, with the local Qwen3.8
compatibility changes integrated into that runtime. They:

1. Recognize compressed-tensors NVFP4 and reuse the existing expert reader with packed tensor-name mapping and reciprocal global scales.
2. Validate and carry compressed-tensors FP8 channel scales through dense projection fusion.
3. Read BF16 or FP8 PLE rows through the existing disk store, with the correct row bytes, staging buffer sizes and dtype decoding.

No CUDA kernels were changed. The official launcher was loaded and checked
first, then the uncensored launcher was loaded with the same runtime and
configuration. On this WSL host, `--moe-cpu-layers auto` moved the layers that
exceeded the pin-memory budget to CPU placement; startup still completed.

The [publisher's model card](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4) describes FP8 attention and shared experts. The pinned tensor headers confirm those formats and show the output head is BF16. The loader follows the actual checkpoint metadata.

## Reproduction

With Hugging Face access accepted and local WSL authentication complete, run:

```powershell
wsl.exe /home/rba90/.freetoken-qwen38/venv/bin/python /mnt/c/workspace/qwen38_27b/scripts/stage-qwen38-uncensored.py
wsl.exe mkdir -p /mnt/c/workspace/qwen38_27b/benchmarks/raw/qwen38-uncensored
wsl.exe /home/rba90/.freetoken-qwen38/venv/bin/python /mnt/c/workspace/qwen38_27b/scripts/verify-qwen38-uncensored.py --output /mnt/c/workspace/qwen38_27b/benchmarks/raw/qwen38-uncensored/local-verification.json
```

Staging uses pinned Xet downloads for safetensors, bounded streaming for metadata and checksums, and periodic page-cache eviction. It verifies LFS SHA-256 hashes and Git blob hashes for smaller files. `staging-manifest.json` remains incomplete until every file is verified. Reruns resume downloads and verify existing files. The final local check reads headers only to validate all expert tensor layouts, PLE extents, dense scale shapes and index completeness.

## Evidence and remaining validation

- [Verified checkpoint manifest](qwen38-uncensored-checkpoint-manifest.json): pinned source and hashes for all 30 files.
- Complete local verification passed for all 223,046 indexed tensors: 73,728 expert projections, 300 FP8 dense weights, and 128 BF16 PLE extents (320,001,536 rows). This check read 31,556,784 header bytes and no tensor payloads. The launcher no-load preflight passed.
- The shared-runtime loader probe passed with the real config/tokenizer and tiny synthetic tensors. Compressed-tensors expert names, reciprocal global scales, FP8 channel scales, and BF16/FP8 PLE rows all passed; CUDA remained uninitialized.
- The updated official launcher loaded all checkpoint weight files, captured its CUDA graph, and returned `ok` from `/health` with served model `qwen38-next-freetoken`.
- The uncensored launcher then loaded all uncensored model shards with the same source tree, captured its CUDA graph, and returned `ok` from `/health` with served model `qwen38-next-uncensored-freetoken`.
- PowerShell parsing, both profile dry runs, and clean launcher stop paths also passed.

The checkpoint preparation results are from 2026-09-06; the shared-runtime
loading checks are from 2026-09-12. These are load checks, not inference
benchmarks. Generated reports are local-only under
`benchmarks/raw/qwen38-uncensored/`; the discovery-only remote inspector and
duplicated metadata dump were moved there after their findings were recorded here.
The model files and local manifests remain ignored by Git.

To repeat the synthetic loader probe without using a GPU:

```powershell
wsl.exe mkdir -p /mnt/c/workspace/qwen38_27b/benchmarks/raw/qwen38-uncensored
wsl.exe /home/rba90/.freetoken-qwen38/venv/bin/python /mnt/c/workspace/qwen38_27b/scripts/probe-qwen38-uncensored-runtime.py --output /mnt/c/workspace/qwen38_27b/benchmarks/raw/qwen38-uncensored/loader-probe.json
```

Short4K initialization is validated for both launchers. Native256K capacity,
concurrency above one, inference responses, and performance remain unvalidated;
the official model's performance figures do not establish uncensored
performance.
