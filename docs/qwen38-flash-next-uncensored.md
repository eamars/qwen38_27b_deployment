# Qwen3.8 Flash-Next uncensored deployment

Status (2026-09-14): the uncensored checkpoint now uses the same updated
FreeToken source and launcher path as the official deployment. Checkpoint
preparation and header verification passed, followed by successful Short4K
model loading and `/health` checks for both launchers. The shared 262K visual
matrix also passed for both text-only and vision-enabled modes. No MTP drafter
was added.

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

The complete source checkpoint is retained, including unused MTP and vision
tensors. Normal serving remains text-only with MTP omitted, as in the existing
FreeToken path. The separate vision launcher builds the upstream vision tower
for this checkpoint on the same port used by the official vision launcher; its
served model ID is `qwen38-next-uncensored-freetoken-vision`, and its benchmark
sends no image input. Both checkpoints are independent. No GGUF weights or
checkpoint conversion are needed.

When this model is exposed through DSH, follow the
[DSH vision runbook](dsh-qwen38-flash-next-vision.md). In addition to
`text,image` input and the 262,144-token context, the catalog entry must list
the three reasoning efforts and explicitly disable the developer role because
the FreeToken endpoint rejects `developer` messages.

## Commands

The current uncensored vision profile uses the patched v0.1.3 fork with 4,600
GPU expert slots, 12 usable GDN state slots, two running requests, 262K shared
KV capacity, 8K prefill chunks and memory ratio 0.90. On 2026-09-18 the default
concurrency returned to two and `--linear-state-cache-ratio` returned to 2;
the checkpoint handoff fix remains installed. The earlier
[matched speed and cache comparison](freetoken-v013-comparison-2026-09-18.md)
used one request and 24 slots after moving the Windows display off the RTX
5090. The 4,200-slot tuning below is also historical.

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
`runtime/freetoken-eamars` source tree are shared with the official launcher.
The editable package now points to the user's fork; the previous
`runtime/freetoken-a80b4d3` checkout remains available for recovery. The runtime
was rolled back to v0.1.2 on 2026-09-17, restored to v0.1.3 on 2026-09-18,
then moved to the fork with the checkpoint handoff fix. Both launchers invoke the same
`launch-freetoken-wsl.sh` process setup and the same
`/home/rba90/.freetoken-qwen38/venv/bin/ft` executable; only the checkpoint,
served name, and PID file differ.

The shared source tree is at `eamars/FreeToken` main commit
`ae8b3cfaef74bb3b687dce8e761f6355171c6137` (v0.1.3 plus the checkpoint
handoff fix). The following Qwen3.8 compatibility changes remain local to the
runtime checkout so the fork's PR contains only the scheduler fix. They:

1. Recognize compressed-tensors NVFP4 and reuse the existing expert reader with packed tensor-name mapping and reciprocal global scales.
2. Validate and carry compressed-tensors FP8 channel scales through dense projection fusion.
3. Read BF16 or FP8 PLE rows through the existing disk store, with the correct row bytes, staging buffer sizes and dtype decoding.

These adaptations are still required by this checkpoint in v0.1.3. They were
preserved during the 2026-09-17 upgrade; a portable backup and their rationale
are retained in [artifacts/freetoken](../artifacts/freetoken/README.md).
The upgrade itself runs only source, package/version, and dependency checks.
No model loading or benchmarks were authorized or performed for this upgrade.
The loading and inference evidence below applies to the earlier runtime.

Later on 2026-09-17, the user reported slower performance and requested a
rollback for comparison. The exact pre-upgrade commit above was restored,
and both patched files were verified against the original backup stash.
No model loading, inference, server startup, or benchmarks were performed
during the rollback. The slowdown remains a user report, not a measured result.

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
- The four-way 262K/4K vision matrix passed three requests each for official
  and uncensored text-only and vision-enabled modes. The prompt was 4,041
  tokens, no image was sent, and visual cases advertised `text,image` input
  modalities.
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

Short4K initialization is validated for both launchers. The shared 262K matrix
now validates inference responses and performance for both variants with and
without the vision tower; concurrency above one and image-bearing requests
remain unvalidated.

## DSH checkpoint retention tuning (2026-09-18)

The capacity experiment below was superseded later that day: after fixing
the checkpoint marker handoff, the vision launcher returned to two running
requests and ratio 2 (12 usable state slots), with 4,600 GPU expert slots.
The patched fork at `ae8b3cfaef74bb3b687dce8e761f6355171c6137` remains installed.
The restart captured CUDA graphs for batch sizes 1 and 2. Two overlapping
36,912-token requests completed, and the scheduler reported two running
requests with 8/12 state slots in use. An identical repeat reused 36,864
tokens. Minimum free VRAM across the probe was 1,926 MiB, without OOM.
The cold pair included first-use startup overhead and is not a speed
comparison. Images, long generations and two full-context requests were not
tested; the 262,144-token KV pool is shared. Raw logs, probe and results are
in `benchmarks/raw/checkpoint-restore-20260918-224724/`.

The initial uncensored vision tuning selected one running request, 24 usable
GDN checkpoint slots and 4,200 GPU expert-cache slots. Context and KV capacity
remain 262,144 tokens, the prefill chunk remains 8,192 tokens and the memory
ratio remains 0.90. The prior running configuration used two requests, 12
checkpoint slots and 4,882 expert slots. Windows `nvidia-smi` reported only
471-506 MiB free before adjustment, so the extra checkpoint memory is funded
by reducing GPU expert-cache capacity rather than increasing total allocation.

Each GDN slot costs 115,642,376 bytes and each expert-cache slot costs
2,772,480 bytes on this checkpoint. Adding 12 GDN slots and removing 682
expert slots reduces the combined allocation by approximately 479 MiB. One
active request needs up to four GDN working slots, leaving 20 retained slots.
The expert banks remain in host RAM; a smaller GPU expert cache can increase
transfers and affect decode speed. Checkpoint capacity does not fix a changed
prompt prefix.

The shared runtime has a small local CLI patch exposing its existing
`linear_state_cache_ratio` setting. The launcher supplies
`--linear-state-cache-ratio 20` and `--moe-cache-size 4200`. The default ratio
for other launchers stays 2. Preserve the patch in
[artifacts/freetoken](../artifacts/freetoken/README.md) when replacing the
runtime. Changing `MaxRunningRequests` also changes the pool size; the measured
configuration here is specifically one running request.

The restart clears the old prompt cache once. Raw startup logs, pre-change
API/memory snapshots and the executable retention/VRAM probe are retained in
`benchmarks/raw/checkpoint-tuning-2026-09-18/`.

### Verification and remaining cache limitation

The first trial used 4,350 expert slots and 24 GDN slots. A 147,519-token
request and an identical repeat both completed without OOM, but the minimum
sampled free VRAM was only 300 MiB across 203 samples. The expert cache was
therefore reduced further to 4,200 using the idle-only cache rebuild API,
which returned `ok`; this change preserves the existing prefix cache.

The final configuration passed a 36,927-token extension/repeat probe. The
extension reused 18,432 tokens and completed in 18.062 seconds; its identical
repeat reused 36,864 tokens and completed in 3.735 seconds. Minimum sampled
free VRAM during this final probe was 963 MiB. These are total request times
for 15 output tokens, not isolated decode speed. The 147K stress run was on
the tighter first trial, not repeated on the final 4,200-slot configuration.
API geometry confirms 24 GDN slots, 4,200 expert slots and 4,096 64-token KV
pages. CLI parsing, invalid-ratio rejection, launcher syntax and the portable
patch reverse-check passed.

The retention probe also reproduced a runtime limitation: the identical
147,519-token requests reused zero tokens (103.250 and 97.750 seconds). This
prompt ends with a 63-token final prefill chunk. The current scheduler skips
cache commits for intermediate `ChunkedReq` objects; continuation construction
does not carry the pending snapshot marker forward, and a final extend of at
most 64 tokens creates no new intermediate GDN snapshot. An unaligned final
decode position also cannot donate its live state with 64-token pages. Thus
the request can finish with no reusable checkpoint despite spare slot capacity.
The early-prefix branch consequently missed too. This is not fixed by this
capacity change, and no scheduler/kernel changes were made. A later normal
18,495-token request repeat reused 18,432 tokens, confirming the distinction
between missing checkpoints and a disabled cache. The DSH slowdown cannot yet
be attributed conclusively to this case without matching its request tokens.
