# Deployment and operations

## Scope and status

This workspace supports two Windows deployment modes from the same pinned CUDA
build. The normal Qwen profiles run as independent `llama-server` processes;
each process is restricted to one physical GPU with
`CUDA_VISIBLE_DEVICES=<UUID>`, so the selected card is runtime `CUDA0`.
The Kazusa shared router was removed on 2026-10-03. Use the independent
launchers for the retained models.

The accepted runtime build is llama.cpp DFlash2 commit
`5ecbe1ac17ec0484c5b44af0bd580cdc9c428ed4`, built natively for Windows with
CUDA 13.3. The executable is expected at:

```text
runtime/llama.cpp-dflash2/build-dflash2/bin/Release/llama-server.exe
```

The current Qwen launch defaults are:

| GPU | Port | Target | Context | Batch / ubatch | DFlash2 | Status |
|---|---:|---|---:|---:|---|---|
| RTX 5090 | 8080 | `UD-Q6_K_M` | 126976 | 1024 / 256 | Q4_K_M, `n-max=5` | provisional |
| RTX 4090 | 8081 | `UD-Q4_K_XL` | 110000 | 512 / 128 | Q4_K_M, `n-max=5` | provisional |

Both launchers enforce target KV `q8_0/q8_0`, draft KV `f16/f16`, one slot,
Flash Attention, `fit off`, no mmproj, no context shift, and all target/draft
layers on the selected GPU. The 5090 context is below 131072 because the
initial 131072 test left only 640 MiB free.

## Prerequisites

The host is Windows with NVIDIA drivers, CUDA, CMake, and MSVC already
installed. The authoritative GPU identities and tool versions are in
[host-inventory.md](host-inventory.md).

The model files are large and are not tracked by Git. Download the Qwen set
with:

```powershell
Set-Location C:\workspace\qwen38_27b
.\scripts\download-models.ps1
```

Use `-IncludeFallbacks` if the Q6 and Q4 fallback files are required as well.
Record hashes after staging or replacing files:

```powershell
.\scripts\record-model-manifest.ps1
```

The manifest includes the Gemma files automatically when they are present.
The shared Gemma 31B MTP drafter used by Fable-5 Distill can be staged with:

```powershell
.\scripts\stage-gemma4-assets.ps1
```

The HauhauCS uncensored QAT target, MTP drafter, and vision projector are
staged at an immutable revision with:

```powershell
.\scripts\stage-gemma4-hauhaucs.ps1
```

## Preflight

Run the no-load check before starting a backend:

```powershell
.\scripts\check-runtime.ps1
```

Add `-IncludeGemma` for Fable-5 Distill and its shared MTP sidecar, or
`-IncludeHauhauCS` for the HauhauCS assets. The check verifies the pinned executable, model files,
DFlash2 command-line support, and UUID-to-`CUDA0` isolation for both cards. It
does not start a server or load a model.

## Launch and stop

Start the Qwen 5090 backend:

```powershell
.\scripts\start-qwen27b-5090.ps1
```

Start the Qwen 4090 backend in another PowerShell window:

```powershell
.\scripts\start-qwen27b-4090.ps1
```

The Turbo Fable Cold Fusion profile uses a Qwen3.8 GGUF target. Stage its
LOW-MTP-IQ4_XS model and F16 vision projector, then launch it on the same
RTX 4090 endpoint with native MTP, image input, 131,072-token context, and
Q8_0 K/V cache:

```powershell
.\scripts\stage-qwen38-turbofcfusion-assets.ps1
.\scripts\start-qwen38-turbofcfusion-4090-128k-vision-mtp.ps1 -DryRun
.\scripts\start-qwen38-turbofcfusion-4090-128k-vision-mtp.ps1
```

Stop only that profile with:

```powershell
.\scripts\start-qwen38-turbofcfusion-4090-128k-vision-mtp.ps1 -Stop
```

It uses port `8081` and the existing pinned Windows llama.cpp build. The model
GGUF contains its trained MTP head; the separately staged `mmproj-F16.gguf`
enables image input. Keep the 4090 free of other model servers before launch.

The LessThanThreeAI Humanlike Chat IQ4_XS profile is separate and uses the
standard RTX 4090 port `8081`. It shares that fixed endpoint with the other
4090 profiles, so stop the existing server first. Stage its text-only merged
target plus the matching base Qwen3.8 MTP head and vision projector, then
launch the local OpenAI-compatible server:

```powershell
.\scripts\stage-qwen38-humanlike-chat-vision-mtp.ps1
.\scripts\start-qwen38-humanlike-chat-4090-128k-vision-mtp.ps1 -DryRun
.\scripts\start-qwen38-humanlike-chat-4090-128k-vision-mtp.ps1
```

The verified shared-GPU defaults offload 14 target layers, with the MTP head
and vision projector on the RTX 4090. They retain a 131,072-token context,
Q8_0 target and MTP K/V caches, and one slot. The launcher requires at least
10,000 MiB free before starting. This setting fit while ComfyUI was also using
the 4090. If the card is otherwise free, pass
`-GpuLayers all -RequiredFreeVramMiB 22000` to attempt full target offload. Stop
only this profile with:

```powershell
.\scripts\start-qwen38-humanlike-chat-4090-128k-vision-mtp.ps1 -Stop
```

It binds to `127.0.0.1:8081`. Live validation on 2026-09-26 reported a 131,072
context, `multimodal` capability, correctly answered an image request, and
accepted 39 of 46 MTP draft tokens (84.8%). After that image request the 4090
had 1,223 MiB free. The auxiliary MTP and vision files are from the matching
base Qwen3.8 checkpoint; the Humanlike merged GGUF itself is text-only.

The Qwen launchers start directly after validating their model, runtime, and
GPU. Use `-Stop` to stop only the matching managed runtime and port:

```powershell
.\scripts\start-qwen27b-5090.ps1 -Stop
.\scripts\start-qwen27b-4090.ps1 -Stop
```

Useful Qwen overrides are `-ContextSize`, `-DraftNMax`, `-Port`, and
`-BindAddress`. Do not change target KV, draft KV, GPU offload, or slot count
without rerunning the acceptance measurements.

The OpenAI-compatible endpoints are:

```text
http://127.0.0.1:8080  RTX 5090
http://127.0.0.1:8081  RTX 4090
```

Check `/health`, `/v1/models`, `/metrics`, and `/slots` before profiling.

The HauhauCS Gemma 4 launcher uses the same RTX 4090 and default port `8083`
as the official QAT launcher. It defaults to 65,536 context tokens because the
full vision+MTP configuration does not fit reliably at 90K with q8 KV caches.
Vision and MTP are enabled by default; use `-NoVision` only when the projector
is not needed, and stop the current model before swapping the process on `8083`:

```powershell
.\scripts\start-gemma4-31b-hauhaucs-balanced-q4_k_m-4090-65k-mtp.ps1 -DryRun
.\scripts\start-gemma4-31b-it-qat-ud-q4_k_xl-4090-90k-mtp.ps1 -Stop -Port 8083
.\scripts\start-gemma4-31b-hauhaucs-balanced-q4_k_m-4090-65k-mtp.ps1
```

## Retired deployments

The Kazusa router, Gemma Persona, Gemma 26B, official Flash-Next FreeToken
and DeepSeek FreeToken launchers and their dedicated helpers were removed
on 2026-10-03. Their recorded results remain under `benchmarks/`, `artifacts/`
and the historical deployment notes. Windows Persona and Gemma 26B weights
are pending deletion after automatic approval review blocked the operation.
See the [cleanup record](history.md#2026-10-03--deployment-cleanup).

Gemma Fable-5 Distill, QAT Instruct and HauhauCS remain available. The shared
`models/mtp-gemma-4-31B-it-Q8_0.gguf` sidecar is retained for Fable-5 Distill.

## Performance profiling

All maintained profilers write under the organized benchmark tree. Start the
relevant server first, then run:

```powershell
python .\scripts\profile-api.py --output benchmarks/qwen27b/api-smoke.json
python .\scripts\profile-deep-context.py `
  --port 8080 `
  --model qwen3.8-27b-dflash2-5090 `
  --target-tokens 118000 `
  --max-tokens 2048 `
  --output benchmarks/qwen27b/2026-08-26/deep-5090-118k.json
.\scripts\profile-vram.ps1 `
  -Phase qwen27b-long `
  -OutputPath benchmarks/qwen27b/2026-08-26/vram-long.csv
```

For comparable `n-max` tests, keep the prompt, context, output length, batch
settings, background GPU load, and cache state constant. Select a setting by
wall-clock completion time subject to correctness and the 1024 MiB VRAM floor;
acceptance ratio alone is not a selection criterion.

The API profiler uses deterministic sampling (`temperature=0`, `top_k=1`, a
fixed seed). The deep-context profiler places unique facts at the beginning,
middle, and end of a tokenizer-calibrated prompt and records retrieval
correctness.

## Safety and acceptance gate

Before treating a configuration as production-ready, record all of the
following for the actual host:

- runtime commit and model/drafter SHA-256 values;
- target and draft GPU residency from startup logs;
- loaded-idle and stress minimum free VRAM;
- cold-start and uncached/cache-hit TTFT;
- prompt processing and generation throughput;
- DFlash2 drafted/accepted tokens and selected `n-max`;
- deterministic target-only/DFlash2 parity;
- long-context retrieval, streaming, cancellation, and intended harness tests.

The hard constraints are no CUDA OOM, no repeated draft allocation failures,
full target/draft GPU residency, target KV `Q8_0` or better, DFlash2 active,
one slot, and at least 1024 MiB free VRAM during the accepted stress run.

## Troubleshooting order

1. Stop unrelated GPU-heavy workloads and confirm the UUID mapping.
2. Re-run `check-runtime.ps1` and inspect `/health` and server logs.
3. Reduce context in measured steps while preserving target KV.
4. Reduce batch/ubatch only if the measured workload requires it.
5. Re-run VRAM, retrieval, and throughput checks after every configuration
   change.

Do not solve a VRAM problem by lowering target KV below `Q8_0`, enabling normal
CPU target offload, adding slots, or silently switching to a different runtime
build.

## Qwen3.8-Flash-Next deployment

Flash-Next is isolated from the maintained Qwen3.8-27B DFlash2 launchers. It
runs natively on Windows through the [Strata path](qwen38-flash-next-strata.md)
on the RTX 5090. The WSL FreeToken path was retired on 2026-10-10; its
[deployment record](qwen38-flash-next-uncensored.md) is historical.
Keep the existing Qwen3.8-27B launchers and their DFlash2 defaults unchanged.

The vision launcher and the DSH catalog procedure are documented in
the [Qwen3.8 Flash-Next DSH vision runbook](dsh-qwen38-flash-next-vision.md).
Strata uses port `1919`. DSH must explicitly declare image input, reasoning
efforts, and `supportsDeveloperRole: false`.
