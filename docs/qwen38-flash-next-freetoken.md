# Qwen3.8 Flash-Next FreeToken deployment

## Retained configuration

Qwen3.8 Flash-Next has one supported path in this workspace: FreeToken on the
RTX 5090 using `RadixArk/Qwen3.8-Flash-Next-NVFP4` under WSL.

```text
GPU = RTX 5090 by UUID
dtype = bfloat16
MoE backend = offload
CPU MoE layers = 0
MoE cache = auto
PLE backend = disk
memory ratio = 0.90
default max-running-requests = 2
```

The launcher default is two concurrent requests. The retained performance
record below was measured with one request at a time; use
`-MaxRunningRequests 1` when reproducing that benchmark.

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

`--moe-cpu-layers 0` is required under WSL. Leaving it unset caused the tested
runtime to move 15 MoE layers to CPU compute. Explicit GPU computation reduced
median request time from 16.66 to 12.53 seconds and raised GPU utilisation.

The retained benchmark used FreeToken base commit
`a80b4d308a81986fa086ec173d7faa70ba737b2d`, which deliberately drops the
checkpoint's `mtp.*` tensors. No MTP configuration or sidecar is retained in
this workspace. The current local FreeToken source revision is
`593aac73dd1102a2af9f42c602039dc49bc25b90`; it includes the Qwen tool-call
boundary fix described in the [FreeToken compatibility note](../runtime/freetoken-a80b4d3/docs/models.md#known-compatibility-issue-qwen38-flash-next-tool-calls-while-thinking).

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
- `runtime/freetoken-a80b4d3` — FreeToken source tree; benchmark and current
  source revisions are recorded above.
- `/home/rba90/.freetoken-qwen38/venv` — WSL Python environment.
- `/home/rba90/models/Qwen3.8-Flash-Next-NVFP4` — complete WSL checkpoint.

The removed Windows GGUF weights, MTP sidecar, Qwen4Exp llama.cpp builds,
alternative launchers, and intermediate benchmark probes are not required by
this deployment.
