# Deployment and operations

## Scope and status

This workspace supports two Windows deployment modes from the same pinned CUDA
build. The normal Qwen profiles run as independent `llama-server` processes;
each process is restricted to one physical GPU with
`CUDA_VISIBLE_DEVICES=<UUID>`, so the selected card is runtime `CUDA0`. When
both models must be reachable through one endpoint, the maintained
`start-kazusa-models.ps1` launcher starts a shared `llama-server` router with
both profiles. There is no tensor parallelism. The shared router's model
preset is hard-coded in the launcher and exists only as a temporary runtime
file.

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
The Gemma experiment can be staged separately with:

```powershell
.\scripts\stage-gemma4-assets.ps1 -LmStudioTargetPath 'D:\path\to\Gemma-4-31B-Isometry-Fabled-Persona.i1-Q4_K_M.gguf'
```

## Preflight

Run the no-load check before starting a backend:

```powershell
.\scripts\check-runtime.ps1
```

Add `-IncludeGemma` only when the Gemma/MTP experiment is needed. The check
verifies the pinned executable, model files, DFlash2 command-line support,
and UUID-to-`CUDA0` isolation for both cards. It does not start a server or
load a model.

## Launch and stop

Start the Qwen 5090 backend:

```powershell
.\scripts\start-qwen27b-5090.ps1
```

Start the Qwen 4090 backend in another PowerShell window:

```powershell
.\scripts\start-qwen27b-4090.ps1
```

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

## Shared Kazusa server

Use this mode when the Qwen RTX 5090 profile and the Gemma RTX 4090 profile
must run behind one server port. Stop any separately launched backend first;
the shared router uses both GPUs and port 8080 by default, so it cannot run
alongside the independent Qwen launchers on ports 8080 and 8081.

Preview the command and embedded profiles without starting a process:

```powershell
.\scripts\start-kazusa-models.ps1 -DryRun
```

Start or stop the shared server:

```powershell
.\scripts\start-kazusa-models.ps1
.\scripts\start-kazusa-models.ps1 -Stop
```

The router exposes these model IDs through the same OpenAI-compatible endpoint:

| Model ID | Physical GPU |
|---|---|
| `qwen27b-5090` | RTX 5090 |
| `gemma4-4090` | RTX 4090 |

Send the desired ID in each request's `model` field, for example
`http://127.0.0.1:8080/v1/chat/completions` with
`"model": "qwen27b-5090"` or `"model": "gemma4-4090"`. The launcher binds
the physical cards in UUID order (`CUDA0` = RTX 5090, `CUDA1` = RTX 4090),
validates both cards, writes its embedded preset to a unique temporary file,
and removes that file when the server exits. No `.ini` file is required in the
repository.

## Gemma experiment

The Gemma launchers and profilers are maintained as an isolated experiment:

```powershell
.\scripts\start-gemma4-5090.ps1 -DryRun
.\scripts\start-gemma4-4090.ps1 -DryRun
python .\scripts\profile-gemma4-mtp.py
python .\scripts\profile-gemma4-4090.py --dry-run --check-gpu
```

They are dry-run by default where documented. Use `--run` only when the
target GPU is free. The recorded 4090 MTP sweep is in
`benchmarks/gemma4/2026-08-25/`; it is a short experimental comparison, not a
Qwen replacement or production sign-off.

The active RTX 4090 Gemma default is the measured N1 profile: context `56320`,
target KV `q8_0/f16`, draft KV `q8_0/q8_0`, MTP `n-max=3`, and batch/ubatch
`256/128`. This is configured in both `start-gemma4-4090.ps1` and the
`gemma4-4090` preset embedded in `start-kazusa-models.ps1`. The historical
benchmark records remain unchanged.

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

## Qwen3.8-Flash-Next experimental deployment

Flash-Next is not an alternate setting for the current DFlash2 Qwen3.8-27B
launchers. Its deployment uses a separate pinned Qwen4Exp-compatible runtime,
layer splitting across both GPUs, CPU placement for `per_layer_token_embd`,
and staged CPU-MoE profiling. The preparation, load validation, and profiler
are documented in
[qwen38-flash-next-deployment.md](qwen38-flash-next-deployment.md).

These entry points are safe to preview while the GPUs are busy:

```powershell
.\scripts\prepare-qwen38-flash-next.ps1
.\scripts\start-qwen38-flash-next.ps1 -Profile Baseline -DryRun
python .\scripts\profile-qwen38-flash-next.py --stage context --dry-run
```

The preparation script never loads a model. The profiler requires an explicit
`--run` before it starts a server. Keep the existing Qwen3.8-27B launchers and
their DFlash2 defaults unchanged. The completed matrix is indexed under
`benchmarks/qwen38_flash_next/2026-08-28/` and is not a production sign-off.
