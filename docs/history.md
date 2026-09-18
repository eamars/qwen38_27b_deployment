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

### 2026-09-02 — Flash-Next consolidated on FreeToken

- Selected the RTX 5090 FreeToken NVFP4 path with disk-backed PLE and explicit
  `--moe-cpu-layers 0`.
- Retained the three-run 4K winner, prompt, and final server log.
- Set the retained Flash-Next wrapper default to two concurrent requests; the
  retained benchmark remains a one-request-at-a-time record.
- Fixed FreeToken responses that emit a Qwen tool call before `</think>` in
  parser commit `593aac7`; the parser boundary and client expectations are
  recorded in the [FreeToken compatibility note](../runtime/freetoken-a80b4d3/docs/models.md#known-compatibility-issue-qwen38-flash-next-tool-calls-while-thinking).
- Removed the Flash-Next GGUF/MTP assets, Qwen4Exp llama.cpp runtimes,
  alternative launchers, and intermediate benchmark probes.

### 2026-09-12 — Flash-Next runtime refresh and uncensored parity

- Fast-forwarded the shared FreeToken runtime to upstream commit
  `953565667f3141c90d0f0eb469bb2655d2407140`.
- Integrated compressed-tensors NVFP4 expert mapping, FP8 dense channel-scale
  handling, and BF16/FP8 disk-backed PLE support into that shared runtime.
- Updated both Flash-Next launchers to the current `--moe-strategy` CLI and
  identical `--moe-cpu-layers auto` placement. Official Short4K loading was
  checked first; the uncensored checkpoint then passed through the same source,
  environment, and launcher path.
- Retired the copied uncensored runtime and the unused DeepSeek experiment
  checkout. Both were moved to recoverable local temporary directories.

### 2026-09-14 — Flash-Next vision runtime and four-way matrix

- Fast-forwarded the shared FreeToken runtime to upstream `f7dbab7`, including
  Qwen3.8 Flash-Next vision tower/mRoPE support and the ModelOpt input-scale
  fix; preserved the local uncensored BF16/FP8 and compressed-tensors patches.
- Added separate official and uncensored vision launchers on the same DSH port,
  plus the 262K-context/4K-input four-way benchmark matrix.
- Passed official and uncensored text-only and vision-enabled cases with no
  image input; all four cases resolved to 262,144 context and passed three
  measured requests each.

### 2026-09-14 — DSH vision catalog integration

- Added separate official and uncensored vision model IDs to the DSH
  `local-qwen38-flash` catalog through the public settings RPC; both use the
  existing `http://192.168.2.13:1919/v1` provider and are deployed one at a
  time on port `1919`.
- Added the explicit DSH metadata that auto-discovery cannot infer:
  `input=[text,image]`, `contextWindow=262144`, and the `low`, `medium`, and
  `xhigh` reasoning-effort map.
- Set `compat.supportsDeveloperRole=false` for both Qwen vision entries and
  the Gemma4 entries. FreeToken/Gemma reject an OpenAI `developer` role; the
  explicit compatibility flag makes DSH adapt the request before forwarding
  it.
- Verified both vision IDs and their reasoning selectors through
  `session/modelCatalog`; the complete operational procedure is in the
  [DSH vision runbook](dsh-qwen38-flash-next-vision.md).

### 2026-09-17 - FreeToken v0.1.3 upgrade

- Updated the shared Qwen FreeToken source from `f7dbab7` to upstream
  `cac247a860e316e06580d05aeb05f2e647bde214` (v0.1.3).
- Preserved both local uncensored checkpoint adaptations in `weight.py` and
  `ple_disk.py`. They handle compressed-tensors NVFP4 expert layouts, FP8
  channel scales, and BF16 PLE storage; upstream still needs these patches.
- Retained the pre-upgrade patch under `artifacts/freetoken/` and a backup
  stash in the runtime repository. Refreshed the editable package registration
  in the existing WSL venv without changing dependencies or launcher settings.
- Limited upgrade verification to source, package/version, and dependency
  checks. No benchmark, model loading, inference, or server startup was run.
  Earlier load and performance results do not validate v0.1.3.

### 2026-09-17 - FreeToken rollback for slowdown investigation

- The user reported slower performance after upgrading and requested the old
  version for comparison; the slowdown was not independently benchmarked.
- Restored the exact pre-upgrade commit
  `f7dbab7f151df353d70b325a8ac09ce0f7f1c456` (reports v0.1.2), preserving both
  uncensored checkpoint patches. Verified both patched files exactly match
  the original pre-upgrade stash.
- Rebuilt/reinstalled the editable package in the existing WSL venv, without
  changing dependencies, launcher settings, or model files. Retained v0.1.3
  in Git and a second stash of its local patches for later investigation.
- No model loading, inference, server startup, or benchmarks were run.

### 2026-09-18 - FreeToken checkpoint tuning and v0.1.3 comparison

- Selected one request and 24 GDN checkpoint slots for the uncensored vision
  launcher; after moving the Windows display off the RTX 5090, retained
  4,600 GPU expert slots, 262K context, 8K prefill and memory ratio 0.90.
- Restored v0.1.3 commit `cac247a860e316e06580d05aeb05f2e647bde214` with
  the two uncensored loader patches and the local checkpoint-capacity CLI.
  Reinstalled the editable package without dependency changes and verified
  successful real-model loading and inference.
- Reused the recorded v0.1.2 baseline and repeated the identical 36,927-token
  cold/warm probe: 32.656/4.234 seconds became 29.219/3.859 seconds, with
  the same 36,864-token warm hit. This is a single pair per version.
- Reproduced the short-final-chunk checkpoint miss on v0.1.3. The user's
  observation of increased DSH miss frequency remains unverified by a
  matched DSH request trace. See the
  [comparison report](freetoken-v013-comparison-2026-09-18.md).

### 2026-09-18 - FreeToken checkpoint fix and fork runtime

The 2026-09-18 checkpoint handoff fix was subsequently committed and pushed
to `eamars/FreeToken` main as `ae8b3cfaef74bb3b687dce8e761f6355171c6137`.
The shared Qwen3.8 venv was rebuilt from `runtime/freetoken-eamars`, preserving
the local loader/CLI overlay and launcher settings. The old checkout remains
available for recovery. See the [fork runtime record](freetoken-fork-runtime-2026-09-18.md).

## Decisions retained

1. Prefer correctness and full GPU residency over a headline throughput number.
2. Keep target KV at `Q8_0/q8_0` or better and draft KV at `f16/f16` for Qwen.
3. Use one process/slot per GPU and bind by UUID, not assumed CUDA index.
4. Treat 1024 MiB minimum free VRAM as a hard acceptance floor.
5. Choose DFlash2 `n-max` by wall-clock completion time under the same workload.
6. Keep historical measurements and source notes, but mark them as archived so
   they cannot be mistaken for current commands or sign-off.
