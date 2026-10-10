# Qwen3.8 / DFlash2 Windows workspace

This repository is a reproducible local-inference workspace for two maintained
Qwen3.8-27B llama.cpp deployment modes plus an isolated
Qwen3.8-Flash-Next Strata path on one Windows host:

- RTX 5090: Qwen3.8-27B `UD-Q6_K_M` with DFlash2, currently profiled at
  `126976` context tokens.
- RTX 4090: Qwen3.8-27B `UD-Q4_K_XL` with DFlash2, currently profiled at
  `110000` context tokens.
- RTX 5090 natively on Windows: Qwen3.8-Flash-Next uncensored ModelOpt NVFP4
  through Strata, with CPU vision, MTP and host RAM snapshots for alternating
  agent histories; see the
  [Strata runbook](docs/qwen38-flash-next-strata.md).

The Strata launcher uses port `1919`.

The independent launchers expose one model per port. The Kazusa router and
Gemma 4 Persona launchers have been retired. The Gemma Fable-5 Distill, QAT
Instruct and HauhauCS launchers remain available.

## Current state

On 2026-10-10 the uncensored Flash-Next FreeToken path was retired in favour of
Strata. Its launchers and helpers were removed; benchmark evidence and
historical notes were kept. See [the retirement record](docs/history.md#2026-10-10--freetoken-retired).

On 2026-10-03, the official Flash-Next FreeToken, DeepSeek FreeToken, Gemma
Persona, Gemma 26B and Kazusa launchers and their dedicated helpers were
removed. Dedicated WSL weights were deleted and benchmark evidence was kept.
Windows Persona and Gemma 26B model deletion is pending because automatic
approval review blocked it; see [the cleanup record](docs/history.md#2026-10-03--deployment-cleanup).

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

Run the runtime check with `-IncludeGemma` to check the retained Gemma
Fable-5 Distill target and shared MTP drafter:

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
- [docs/qwen38-flash-next-freetoken.md](docs/qwen38-flash-next-freetoken.md) — historical official Flash-Next deployment and benchmark.
- [docs/qwen38-flash-next-uncensored.md](docs/qwen38-flash-next-uncensored.md) — historical uncensored FreeToken deployment, retired 2026-10-10.
- [scripts/README.md](scripts/README.md) — maintained script inventory.
- [benchmarks/README.md](benchmarks/README.md) — raw-result layout and naming convention.

The original implementation instruction, handover notes, and first-stage
profile note are retained under [docs/archive](docs/archive/) for historical
reference. They are not the current operating instructions.

## Repository layout

```text
docs/       current documentation and archived stage notes
models/     local checkpoints and GGUF assets; ignored by Git
runtime/    retained llama.cpp and Strata runtimes; ignored by Git
scripts/    maintained launch, profiling, setup, and inventory helpers
benchmarks/ dated JSON/CSV measurements grouped by model and run date
```

For the maintained Qwen3.8-27B DFlash2 profiles, the hard operational
constraints are full target and draft GPU residency, target KV cache at `Q8_0`
or better, DFlash2 enabled, one slot, no normal CPU offload, and at least
1024 MiB free VRAM during the accepted stress workload.

The Flash-Next FreeToken paths are retired; their measured results remain in
[docs/benchmarks.md](docs/benchmarks.md) and [the historical deployment record](docs/qwen38-flash-next-freetoken.md).
