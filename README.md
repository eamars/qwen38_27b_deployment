# Qwen3.8 / DFlash2 Windows workspace

This repository is a reproducible local-inference workspace for two maintained
Qwen3.8-27B llama.cpp deployment modes plus one isolated
Qwen3.8-Flash-Next FreeToken path on one Windows host:

- RTX 5090: Qwen3.8-27B `UD-Q6_K_M` with DFlash2, currently profiled at
  `126976` context tokens.
- RTX 4090: Qwen3.8-27B `UD-Q4_K_XL` with DFlash2, currently profiled at
  `110000` context tokens.
- RTX 5090 under WSL: Qwen3.8-Flash-Next NVFP4 through FreeToken, with routed
  expert offload and disk-backed PLE; this path is separate from the maintained
  llama.cpp launchers.

The independent launchers expose one model per port. The maintained Kazusa
launcher can instead place the Qwen RTX 5090 profile and the experimental Gemma
RTX 4090 profile behind one shared router endpoint.

The repository also contains an experimental Gemma 4 persona + Google MTP
profile. It is documented separately and is not the production Qwen path.

## Current state

The FreeToken MTP-on-RTX-4090 experiment was closed on 2026-09-06. The
target verifier failed the performance requirement (`G2: FAIL_ECONOMICS`),
and the final existing-kernel batching check failed the exact numerical
contract (`G2R: HARD_STOP_SCOPE`). No MTP acceleration was deployed. The
experiment test ground is ready for removal, but automated deletion was
blocked by execution policy. The working runtimes remain in place. See the
[conclusion and retained evidence](docs/archive/qwen38-mtp-4090-conclusion.md).

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
- [docs/qwen38-flash-next-freetoken.md](docs/qwen38-flash-next-freetoken.md) — retained FreeToken-only Flash-Next deployment and benchmark.
- [scripts/README.md](scripts/README.md) — maintained script inventory.
- [benchmarks/README.md](benchmarks/README.md) — raw-result layout and naming convention.

The original implementation instruction, handover notes, and first-stage
profile note are retained under [docs/archive](docs/archive/) for historical
reference. They are not the current operating instructions.

## Repository layout

```text
docs/       current documentation and archived stage notes
models/     local GGUF files; ignored by Git
runtime/    retained llama.cpp and FreeToken runtimes; ignored by Git
scripts/    maintained launch, profiling, setup, and inventory helpers
benchmarks/ dated JSON/CSV measurements grouped by model and run date
```

For the maintained Qwen3.8-27B DFlash2 profiles, the hard operational
constraints are full target and draft GPU residency, target KV cache at `Q8_0`
or better, DFlash2 enabled, one slot, no normal CPU offload, and at least
1024 MiB free VRAM during the accepted stress workload.

Qwen3.8-Flash-Next is retained only as a FreeToken RTX 5090 path using NVFP4,
disk-backed PLE, `--moe-backend offload`, and explicit
`--moe-cpu-layers 0`. Its retained three-run 4K median was 12.53 seconds,
1656 prompt tok/s, and 50.59 decode tok/s. Native 256K validation remains
open. Its routed-expert host-memory placement is separate from the DFlash2
constraints above. See [the deployment record](docs/qwen38-flash-next-freetoken.md)
and its [tool-call compatibility note](runtime/freetoken-a80b4d3/docs/models.md#known-compatibility-issue-qwen38-flash-next-tool-calls-while-thinking).
