# Maintained scripts

The scripts directory contains only maintained operational helpers. The
canonical launch and profiling entry points are:

| Purpose | Script |
|---|---|
| Qwen RTX 5090 launch/stop | `start-qwen27b-5090.ps1` |
| Qwen RTX 4090 launch/stop | `start-qwen27b-4090.ps1` |
| Shared Qwen + Gemma server | `start-kazusa-models.ps1` |
| Gemma RTX 5090 experiment | `start-gemma4-5090.ps1` |
| Gemma RTX 4090 experiment | `start-gemma4-4090.ps1` |
| Qwen API smoke/sustained checks | `profile-api.py` |
| Qwen tokenizer-calibrated deep context | `profile-deep-context.py` |
| Flash-Next FreeToken RTX 5090 launch/stop | `start-qwen38-flash-next-freetoken.ps1` |
| Uncensored Flash-Next FreeToken preparation/launch | `start-qwen38-flash-next-uncensored-freetoken.ps1` |
| Flash-Next FreeToken 4K benchmark | `benchmark-freetoken-qwen38-next.py` |
| GPU memory sampling | `profile-vram.ps1` |
| Gemma MTP comparison | `profile-gemma4-mtp.py`, `profile-gemma4-4090.py` |
| Gemma short/long combined comparison | `profile-gemma4-combined.py` |
| Runtime/model/GPU preflight | `check-runtime.ps1` |

Setup and inventory helpers are also retained because they produce reproducible
state rather than launch an alternative runtime:

- `download-models.ps1` — fetches the Qwen deployment set.
- `stage-gemma4-assets.ps1` and `download-gemma4-persona-ranged.ps1` — stages
  the isolated Gemma experiment.
- `collect-host-inventory.ps1` — refreshes `docs/host-inventory.md`.
- `record-model-manifest.ps1` — refreshes `docs/models.md`.
- `prepare-qwen38-uncensored-runtime.py` — reproduces the isolated FreeToken
  loader adapter using the retained patch and compiled kernels.
- `stage-qwen38-uncensored.py` — stages and verifies the pinned checkpoint in WSL.
- `verify-qwen38-uncensored.py` — checks all local tensor headers against the
  adapter without reading tensor payloads.
- `probe-qwen38-uncensored-runtime.py` — checks the loaders with tiny synthetic
  tensors and the compiled CPU disk reader; never initializes CUDA.

The [uncensored deployment note](../docs/qwen38-flash-next-uncensored.md)
records the pinned assets, loader changes, preparation results and deferred
GPU test. Generated verification reports belong under
`benchmarks/raw/qwen38-uncensored/`, which is ignored by Git.

`start-kazusa-models.ps1` launches one shared `llama-server` router with the
`qwen27b-5090` and `gemma4-4090` profiles. Their model-specific settings are
hard-coded in the launcher and emitted only to a temporary preset while the
server runs. There is no separate `kazusa-models.ini` file; the temporary
preset is deleted when the server exits.

The unvalidated two-slot Qwen launcher remains removed. Cleanup and restoration
history is recorded in [docs/history.md](../docs/history.md).

Qwen3.8 Flash-Next is FreeToken-only. The measured profile uses the RTX 5090,
NVFP4, disk-backed PLE, and explicit `--moe-cpu-layers 0`. Preview without
loading weights:

```powershell
.\scripts\start-qwen38-flash-next-freetoken.ps1 -Profile Short4K -DryRun
```

The wrapper defaults to two concurrent requests. Pass
`-MaxRunningRequests 1` to reproduce the retained single-request benchmark.
