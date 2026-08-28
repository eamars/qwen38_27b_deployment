# Project history and decision log

This record separates historical stages from the current operating guide.
Dates are local host dates (Pacific/Auckland, UTC+12 in the captured runs).

## Timeline

### 2026-08-20 — baseline workspace

- Created the Windows dual-GPU DFlash2 workspace and recorded the host,
  runtime, model inventory, and handover requirements.
- Pinned llama.cpp DFlash2 commit
  `5ecbe1ac17ec0484c5b44af0bd580cdc9c428ed4`.
- Recorded UUID/PCI mappings for the RTX 5090 and RTX 4090.
- Downloaded the Qwen target/fallback set and DFlash2 drafter; the current
  hashes are regenerated in [models.md](models.md).
- The original implementation instruction and per-GPU handovers are retained
  in [archive](archive/).

### 2026-08-21 — first dual-GPU load and profiling

- Loaded both Qwen backends and verified `/health`, `/v1/models`, and
  speculative slots.
- Rejected RTX 5090 Q6_K_M at 131072 after a 640 MiB minimum-free-VRAM sample.
- Reduced the 5090 default to 126976; the deep 118K run measured 1272 MiB free
  and passed retrieval.
- Established the provisional RTX 4090 Q4_K_XL / 110000 profile with 1103 MiB
  minimum free VRAM in both recorded deep runs.
- Preserved the raw JSON/CSV evidence under
  [benchmarks/qwen27b/2026-08-21](../benchmarks/qwen27b/2026-08-21/).

### 2026-08-25 — Gemma 4 experiment

- Staged the exact Gemma 4 persona targets and Google MTP sidecar.
- Added dedicated Gemma launchers and dry-run/matrix profilers.
- Recorded a short 4090 MTP comparison. It remains an experiment because the
  result does not include long-context or production acceptance gates.

### 2026-08-26 — Qwen DFlash2 comparison

- Ran matching 5090 deep-context tests for DFlash2 `n-max=5` and `n-max=7`.
- Both runs retained retrieval correctness and the 1024 MiB reserve in the
  captured RTX 5090 VRAM samples.
- Kept `n-max=5` as the provisional default because its 2048-token deep run
  completed in 103.30 s versus 110.56 s for `n-max=7`.
- Grouped raw results by model/date and archived stage-specific documentation.

### Cleanup stage — 2026-08-26

- Removed the unvalidated two-slot `start-qwen27b-4090-2x52k.ps1` experiment.
- Removed the old Kazusa multi-model launcher and standalone preset while
  consolidating the scripts and documentation.
- Replaced the stage-specific `verify-preload.ps1` with the maintained
  `check-runtime.ps1` and made its Gemma checks optional.
- Rewrote the active documentation around deployment, measurements, models,
  and history. Original source notes remain archived rather than being mixed
  into the current instructions.

### Follow-up — 2026-08-26 shared Kazusa router

- Restored `start-kazusa-models.ps1` on request so the Qwen RTX 5090 and Gemma
  RTX 4090 profiles can run from one `llama-server` endpoint.
- Kept the two model profiles inside the launcher as a hard-coded temporary
  preset. The preset is written under the system temporary directory only for
  the lifetime of the process and is removed during shutdown; no Kazusa `.ini`
  file is tracked.

### 2026-08-27/28 — Flash-Next load and fixed matrix

- Added the isolated Qwen4Exp build, preparation, launcher, and profiler
  helpers without changing the maintained Qwen3.8-27B DFlash2 launchers.
- Recorded the pinned Qwen4Exp source/runtime and four-shard Q4 model set in
  the build and environment manifests.
- Executed the explicit no-MTP baseline on the RTX 5090 + RTX 4090 using
  `38,10` layer splitting, `n-cpu-moe=33`, F16/F16 KV, two 262144-token slots,
  and the fixed executor 1/2 × short/mid × three-repetition matrix.
- All matrix combinations completed with retrieval correctness; the lowest
  observed free VRAM was 1957 MiB on the RTX 5090 and 2321 MiB on the RTX
  4090. The dated JSON/CSV evidence is retained under
  `benchmarks/qwen38_flash_next/2026-08-28/`.
- Kept the result experimental: MTP, parity, and 250k-token validation remain
  open. Earlier exploratory probes and plan records are retained under the
  2026-08-27 benchmark directory.

## Decisions retained

1. Prefer correctness and full GPU residency over a headline throughput number.
2. Keep target KV at `Q8_0/q8_0` or better and draft KV at `f16/f16` for Qwen.
3. Use one process/slot per GPU and bind by UUID, not assumed CUDA index.
4. Treat 1024 MiB minimum free VRAM as a hard acceptance floor.
5. Choose DFlash2 `n-max` by wall-clock completion time under the same workload.
6. Keep historical measurements and source notes, but mark them as archived so
   they cannot be mistaken for current commands or sign-off.
