# Qwen3.8 / DFlash2 Windows workspace

This repository is a reproducible local-inference workspace for two Qwen
llama.cpp deployment modes on one Windows host:

- RTX 5090: Qwen3.8-27B `UD-Q6_K_M` with DFlash2, currently profiled at
  `126976` context tokens.
- RTX 4090: Qwen3.8-27B `UD-Q4_K_XL` with DFlash2, currently profiled at
  `110000` context tokens.

The independent launchers expose one model per port. The maintained Kazusa
launcher can instead place the Qwen RTX 5090 profile and the experimental Gemma
RTX 4090 profile behind one shared router endpoint.

The repository also contains an experimental Gemma 4 persona + Google MTP
profile. It is documented separately and is not the production Qwen path.

## Current state

The runtime and model files are present locally but intentionally ignored by
Git. Qwen long-context retrieval passed in the recorded runs. The Qwen
profiles are provisional rather than a complete production sign-off: cold
start, cache-hit TTFT, deterministic parity, and intended harness validation
still need to be recorded before calling either backend fully production-ready.

The 2026-08-26 Qwen comparison kept DFlash2 `n-max=5` as the provisional
default. `n-max=7` was also safe in the captured 5090 VRAM run, but was slower
in the 118K / 2048-token completion. See [docs/benchmarks.md](docs/benchmarks.md)
for the evidence and limitations.

## Quick start

From the repository root in PowerShell:

```powershell
.\scripts\download-models.ps1
.\scripts\check-runtime.ps1
.\scripts\start-qwen27b-5090.ps1
```

Use a second console for the RTX 4090 backend:

```powershell
.\scripts\start-qwen27b-4090.ps1
```

Each launcher validates the physical GPU UUID, maps the selected card to
runtime `CUDA0`, keeps target and draft layers on that GPU, and starts one
OpenAI-compatible server. Stop a managed backend with the same script and
`-Stop`.

Choose the shared mode instead of the two independent launchers above when the
Qwen RTX 5090 and Gemma RTX 4090 profiles should run behind one server. Use the
Kazusa router launcher; its model settings are embedded in the script, so no
repository `.ini` file is needed:

```powershell
.\scripts\start-kazusa-models.ps1 -DryRun
.\scripts\start-kazusa-models.ps1
```

The shared endpoint is `http://127.0.0.1:8080`; select
`qwen27b-5090` or `gemma4-4090` in the request `model` field. Stop it with:

```powershell
.\scripts\start-kazusa-models.ps1 -Stop
```

Run the runtime check with `-IncludeGemma` when the experimental Gemma assets
are staged:

```powershell
.\scripts\check-runtime.ps1 -IncludeGemma
```

## Repository guide

- [docs/README.md](docs/README.md) — documentation index and recommended reading order.
- [docs/deployment.md](docs/deployment.md) — setup, configuration, launch, stop, and profiling procedures.
- [docs/benchmarks.md](docs/benchmarks.md) — consolidated measured results and selection decisions.
- [docs/models.md](docs/models.md) — local model inventory, sizes, and current hashes.
- [docs/host-inventory.md](docs/host-inventory.md) — captured hardware/build snapshot.
- [docs/history.md](docs/history.md) — chronological project history and decision log.
- [docs/qwen38-flash-next-deployment.md](docs/qwen38-flash-next-deployment.md) — isolated Flash-Next deployment and profiler.
- [scripts/README.md](scripts/README.md) — maintained script inventory.
- [benchmarks/README.md](benchmarks/README.md) — raw-result layout and naming convention.

The original implementation instruction, handover notes, and first-stage
profile note are retained under [docs/archive](docs/archive/) for historical
reference. They are not the current operating instructions.

## Repository layout

```text
docs/       current documentation and archived stage notes
models/     local GGUF files; ignored by Git
runtime/    local llama.cpp build; ignored by Git
scripts/    maintained launch, profiling, setup, and inventory helpers
benchmarks/ dated JSON/CSV measurements grouped by model and run date
```

The hard operational constraints are full GPU residency, target KV cache at
`Q8_0` or better, DFlash2 enabled, one slot, no normal CPU offload, and at
least 1024 MiB free VRAM during the accepted stress workload.

The separate Qwen3.8-Flash-Next path is experimental rather than a maintained
production profile. Its pinned Qwen4Exp runtime and four-shard Q4 model have a
recorded load and context matrix; MTP and broader correctness gates remain
open. See [docs/qwen38-flash-next-deployment.md](docs/qwen38-flash-next-deployment.md).
