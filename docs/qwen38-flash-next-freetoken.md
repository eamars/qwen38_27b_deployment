# Qwen3.8 Flash-Next FreeToken deployment

## Retained configuration

Qwen3.8 Flash-Next has one supported path in this workspace: FreeToken on the
RTX 5090 using `RadixArk/Qwen3.8-Flash-Next-NVFP4` under WSL.

```text
GPU = RTX 5090 by UUID
dtype = bfloat16
MoE strategy = offload
CPU MoE layers = auto
MoE cache = auto
PLE backend = disk
memory ratio = 0.90
default max-running-requests = 1
```

The launcher default is one concurrent request. The retained performance
record below was measured with one request at a time.

Start the measured 4K profile:

```powershell
.\scripts\start-qwen38-flash-next-freetoken.ps1 -Profile Short4K
```

Preview it without loading the model:

```powershell
.\scripts\start-qwen38-flash-next-freetoken.ps1 -Profile Short4K -DryRun
```

Stop the managed process:

```powershell
.\scripts\start-qwen38-flash-next-freetoken.ps1 -Stop
```

The endpoint is plain HTTP, not HTTPS. From this host use
`http://127.0.0.1:1919/v1` or `http://192.168.2.13:1919/v1`; LAN clients use
the latter after the Hyper-V firewall rule is present. WSL is configured with
mirrored networking and `hostAddressLoopback=true` in `%USERPROFILE%\\.wslconfig`.
The inbound rule `Qwen38-FreeToken-1919` is restricted to `192.168.2.0/24`.

`Native256K` is available as an unvalidated capacity profile. Only `Short4K`
has completed the retained performance test.

## Memory placement

| Component | Placement |
|---|---|
| Dense backbone and shared-expert tensors | RTX 5090, permanently resident |
| Routed NVFP4 experts | 63.46 GiB source bank in pinned host RAM; hot expert-layer slots in the RTX 5090 LRU; misses DMA to the GPU and compute on the GPU |
| 47.7 GiB PLE n-gram table | disk-backed host row store via `--ple-backend disk` |
| 8192-token KV floor | RTX 5090, 0.19 GiB in the measured profile |

The measured allocation used 6681 of 24576 routed expert-layer slots
(27.19%, approximately 17.25 GiB of VRAM). Request peak was 30423 MiB and the
minimum free VRAM sample was 1765 MiB.

The updated launcher uses `--moe-cpu-layers auto` under WSL. The latest runtime
detects the host pin-memory budget and moves only the layers that cannot be
pinned to CPU placement instead of aborting startup. The retained benchmark
below used the older explicit `--moe-cpu-layers 0` configuration and is not a
benchmark of the updated runtime.

The retained benchmark used FreeToken base commit
`a80b4d308a81986fa086ec173d7faa70ba737b2d`, which deliberately drops the
checkpoint's `mtp.*` tensors. No MTP configuration or sidecar is retained in
this workspace. The current local FreeToken source is upstream commit
`f7dbab7f151df353d70b325a8ac09ce0f7f1c456`, which includes the upstream
Qwen3.8 Flash-Next vision tower/mRoPE support and the ModelOpt input-scale
fix, with the local Qwen3.8
compressed-tensors compatibility changes described in the
[FreeToken compatibility note](../runtime/freetoken-a80b4d3/docs/models.md#known-compatibility-issue-qwen38-flash-next-tool-calls-while-thinking).

The shared WSL environment also contains the upstream vision dependencies
Pillow 12.3.0 and torchvision 0.26.0. The vision launcher enables the
vision tower with `--mm-encoder-weights host`; the benchmark still sends
text-only requests and no image input.

## Retained 4K benchmark

The final record contains three requests using a 4041-token retrieval prompt
and 512 generated tokens. Prompt reuse was disabled and all retrieval anchors
passed.

| Metric | Median/observed result |
|---|---:|
| Complete request wall time | 12.53 s |
| Estimated prompt throughput | 1656.48 tok/s |
| Decode throughput | 50.59 tok/s |
| Process CPU during requests | 100.9-101.6% |
| Average RTX 5090 utilisation | 92.9-96.9% |
| GPU p95 | 99% |

Evidence:

- [final benchmark JSON](../benchmarks/qwen38_flash_next/2026-09-02/freetoken-4k-gpu-only-winner.json)
- [retained prompt](../benchmarks/qwen38_flash_next/2026-09-02/freetoken-4k-prompt.txt)
- [final server log](../benchmarks/qwen38_flash_next/2026-09-02/logs/freetoken-4k-gpu-only-winner.log)

## Retained implementation

- `scripts/start-qwen38-flash-next-freetoken.ps1` — launch, dry-run, and stop.
- `scripts/launch-freetoken-wsl.sh` — pins the WSL CUDA/Python environment and
  selects the physical RTX 5090 by UUID.
- `scripts/benchmark-freetoken-qwen38-next.py` — optional repeatable 4K
  measurement harness for the retained profile.
- `scripts/start-qwen38-flash-next-freetoken-vision.ps1` — official
  vision-enabled deployment launcher on the shared runtime, serving
  `qwen38-next-freetoken-vision`.
- `scripts/start-qwen38-flash-next-uncensored-freetoken-vision.ps1` — separate
  uncensored vision-enabled deployment launcher on the same port, serving
  `qwen38-next-uncensored-freetoken-vision`.
- `scripts/benchmark-qwen38-flash-next-freetoken-vision-matrix.py` — four-way
  262K-context/4K-input benchmark matrix.
- `runtime/freetoken-a80b4d3` — FreeToken source tree; benchmark and current
  source revisions are recorded above.
- `/home/rba90/.freetoken-qwen38/venv` — WSL Python environment.
- `/home/rba90/models/Qwen3.8-Flash-Next-NVFP4` — complete WSL checkpoint.

## 262K vision matrix — 2026-09-14

The new matrix passed three measured requests in each configuration. Every
request used the same 4,041-token prompt, 262,144-token context, and no image
input. Visual cases reported `text,image` input modalities; text-only cases
reported `text`.

| Configuration | Median wall | Prompt tok/s | Decode tok/s |
|---|---:|---:|---:|
| Official, no visual | 18.30 s | 972.29 | 36.06 |
| Uncensored, no visual | 17.06 s | 974.58 | 39.51 |
| Official + visual | 18.34 s | 977.37 | 35.89 |
| Uncensored + visual | 17.11 s | 971.87 | 39.35 |

Evidence: [262K/4K matrix result](../benchmarks/raw/qwen38_flash_next/2026-09-14/freetoken-vision-262k-4k.json).

The removed Windows GGUF weights, MTP sidecar, Qwen4Exp llama.cpp builds,
alternative launchers, and intermediate benchmark probes are not required by
this deployment.
