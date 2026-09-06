# Qwen3.8 Flash-Next uncensored deployment

Status (2026-09-06): preparation complete, paused immediately before model loading. All 30 checkpoint files passed checksum verification, all 223,046 tensor headers passed the loader checks, and the launcher reported `assets_ready: true`. Model startup and GPU validation are deliberately deferred because the graphics cards are in use.

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
| Default | Native256K, two concurrent requests; capacity remains unvalidated |
| Expert placement | NVFP4 host banks, hot experts cached on GPU |
| Expert computation | GPU: `--moe-backend offload --moe-cpu-layers 0 --moe-cache-auto` |
| Dense weights | BF16 GPU buffers; source FP8 weights dequantized with channel scales |
| PLE | Original BF16 shards on disk, approximately 95.37 GiB |
| Memory ratio / cache | `0.90`, radix |

The complete source checkpoint is retained, including unused MTP and vision tensors. Serving remains text-only with MTP omitted, as in the existing FreeToken path. Both checkpoints are independent. No GGUF weights or checkpoint conversion are needed.

## Commands

Preview without loading anything:

```powershell
.\scripts\start-qwen38-flash-next-uncensored-freetoken.ps1 -Profile Short4K -DryRun
```

Check verified file sizes, runtime hashes, GPU identity and port availability without starting FreeToken or initializing CUDA:

```powershell
.\scripts\start-qwen38-flash-next-uncensored-freetoken.ps1 -Check
```

Only after the user releases the GPU and authorizes loading:

```powershell
.\scripts\start-qwen38-flash-next-uncensored-freetoken.ps1 -Profile Short4K
```

Use `-MaxRunningRequests 1` for the first comparison with the existing single-request 4K benchmark. Omitting `-Profile` selects Native256K, as in the original launcher. Neither profile has been tested with this uncensored model yet.

Stop only the process managed by the uncensored launcher:

```powershell
.\scripts\start-qwen38-flash-next-uncensored-freetoken.ps1 -Stop
```

The supervisor stores a separate PID/start-time record at `/tmp/qwen38-flash-next-uncensored-freetoken-1919.json` and checks process identity before signaling its group. Startup refuses an occupied port on both Windows and WSL without stopping its owner. It does not automatically switch models. The original startup/stop script is unchanged.

## Runtime reuse

The existing CUDA toolkit, compiled FreeToken kernels and `/home/rba90/.freetoken-qwen38/venv` environment are reused. The venv is not reinstalled or repointed. The new supervisor sets its child's `PYTHONPATH` to `runtime/freetoken-qwen38-uncensored/python`.

The isolated source is based on the existing runtime's clean revision `af71ba43206e124f5ff6419b47ee36c6e9981078`. The retained [loader patch](../scripts/patches/qwen38-uncensored-loader.patch) changes three Qwen files:

1. Recognize compressed-tensors NVFP4 and reuse the existing expert reader with packed tensor-name mapping and reciprocal global scales.
2. Apply FP8 channel scales before converting dense weights to BF16 and fusing projections. Scales may reside in a different file.
3. Read BF16 PLE rows through the existing disk store, with the correct row bytes, staging buffer sizes and dtype decoding.

The original loader expects ModelOpt expert names and FP8 PLE. Pointing it directly at the new checkpoint would be incorrect. No CUDA kernels were changed. Full initialization, GPU transfer, graph capture and inference still require the deferred live test.

The [publisher's model card](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4) describes FP8 attention and shared experts. The pinned tensor headers confirm those formats and show the output head is BF16. The loader follows the actual checkpoint metadata.

## Reproduction

With Hugging Face access accepted and local WSL authentication complete, run:

```powershell
wsl.exe /home/rba90/.freetoken-qwen38/venv/bin/python /mnt/c/workspace/qwen38_27b/scripts/prepare-qwen38-uncensored-runtime.py
wsl.exe /home/rba90/.freetoken-qwen38/venv/bin/python /mnt/c/workspace/qwen38_27b/scripts/stage-qwen38-uncensored.py
wsl.exe mkdir -p /mnt/c/workspace/qwen38_27b/benchmarks/raw/qwen38-uncensored
wsl.exe /home/rba90/.freetoken-qwen38/venv/bin/python /mnt/c/workspace/qwen38_27b/scripts/verify-qwen38-uncensored.py --output /mnt/c/workspace/qwen38_27b/benchmarks/raw/qwen38-uncensored/local-verification.json
```

Preparation verifies the source revision, copies the Python runtime and existing extensions, applies the patch, and records hashes in `runtime-manifest.json`. It leaves an existing matching runtime intact. The launcher verifies these hashes before startup.

Staging uses pinned Xet downloads for safetensors, bounded streaming for metadata and checksums, and periodic page-cache eviction. It verifies LFS SHA-256 hashes and Git blob hashes for smaller files. `staging-manifest.json` remains incomplete until every file is verified. Reruns resume downloads and verify existing files. The final local check reads headers only to validate all expert tensor layouts, PLE extents, dense scale shapes and index completeness.

## Evidence and deferred work

- [Verified checkpoint manifest](qwen38-uncensored-checkpoint-manifest.json): pinned source and hashes for all 30 files.
- Complete local verification passed for all 223,046 indexed tensors: 73,728 expert projections, 300 FP8 dense weights, and 128 BF16 PLE extents (320,001,536 rows). This check read 31,556,784 header bytes and no tensor payloads. The launcher no-load preflight passed.
- The loader probe passed with the real config/tokenizer and tiny synthetic tensors through the actual loaders. Serial and parallel expert readers preserved packed bytes and inverted global scales; FP8 channel scales were applied before QKV fusion. BF16 and FP8 rows passed the compiled `io_uring`/`O_DIRECT` disk-reader check across extents and page boundaries. CUDA remained uninitialized.
- The supervisor refused occupied Windows/WSL ports and an unrelated PID. Its scoped stop check passed using a temporary test process.
- PowerShell parsing, both profile dry runs, Windows occupied-port refusal, exact patch reproduction and isolated `ft serve --help` also passed.

These are preparation results from 2026-09-06, not inference benchmarks.
The checkpoint manifest, loader patch and reusable preparation tools are retained
for version control. Generated reports are local-only under
`benchmarks/raw/qwen38-uncensored/`; the discovery-only remote inspector and
duplicated metadata dump were moved there after their findings were recorded here.
Model files, copied runtimes and their local manifests remain ignored by Git.

To repeat the synthetic loader probe without using a GPU:

```powershell
wsl.exe mkdir -p /mnt/c/workspace/qwen38_27b/benchmarks/raw/qwen38-uncensored
wsl.exe /home/rba90/.freetoken-qwen38/venv/bin/python /mnt/c/workspace/qwen38_27b/scripts/probe-qwen38-uncensored-runtime.py --output /mnt/c/workspace/qwen38_27b/benchmarks/raw/qwen38-uncensored/loader-probe.json
```

Pause before startup. After the user explicitly authorizes the switch, verify Short4K initialization, RAM/VRAM placement, basic and streaming API responses, reasoning/tool calls, and the existing 4K benchmark. Native256K and concurrency remain unvalidated. The original model's performance figures do not establish uncensored performance.
