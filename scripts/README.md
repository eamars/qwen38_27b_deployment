# Maintained scripts

The scripts directory contains only maintained operational helpers. The
canonical launch and profiling entry points are:

| Purpose | Script |
|---|---|
| Qwen RTX 5090 launch/stop | `start-qwen27b-5090.ps1` |
| Qwen RTX 4090 launch/stop | `start-qwen27b-4090.ps1` |
| Qwen3.8 Turbo Fable Cold Fusion LOW-MTP-IQ4_XS, RTX 4090, 128K context, vision/MTP | `start-qwen38-turbofcfusion-4090-128k-vision-mtp.ps1` |
| Qwen3.8 Humanlike Chat IQ4_XS, RTX 4090, 128K context, vision/MTP | `start-qwen38-humanlike-chat-4090-128k-vision-mtp.ps1` |
| Gemma 4 31B TeichAI Fable-5 Distill Q4_K_M, RTX 4090 | `start-gemma4-31b-fable-5-distill-q4_k_m-4090-70k-mtp.ps1` |
| Gemma 4 31B Instruct QAT UD-Q4_K_XL, RTX 4090, 90K context | `start-gemma4-31b-it-qat-ud-q4_k_xl-4090-90k-mtp.ps1` |
| Gemma 4 31B HauhauCS QAT Uncensored Balanced Q4_K_M + vision/MTP, RTX 4090, 65K context | `start-gemma4-31b-hauhaucs-balanced-q4_k_m-4090-65k-mtp.ps1` |
| Qwen API smoke/sustained checks | `profile-api.py` |
| Qwen tokenizer-calibrated deep context | `profile-deep-context.py` |
| Flash-Next uncensored native Windows Strata vision/MTP launch/stop | `start-qwen38-flash-next-uncensored-strata-vision.ps1` |
| Install the pinned Windows Strata runtime and Python environments | `setup-strata-runtime.ps1` |
| Strata server patch: quoted or repeated `</think>` stays out of the answer (applied by the installer) | `patches/strata-think-echo.py` |
| Download and verify the ModelOpt NVFP4 checkpoint for Strata | `stage-qwen38-strata.py` |
| Convert the verified Strata checkpoint into workspace model assets | `prepare-qwen38-strata.py` |
| Strata real API, image, tool and retrieval/cache checks | `probe-qwen38-strata.py` |
| Strata alternating agent/tool histories, snapshot pressure, prefix changes and memory sampling | `probe-qwen38-strata-cache.py` |
| Asuna's installed DSH adapter, reasoning replay and waiting-parent cache probe | `probe-asuna-strata-cache.mjs` |
| Strata cold prefill, long streaming output, request queuing and VRAM measurements | `benchmark-qwen38-strata.py` |
| Strata MTP draft acceptance and decode speed for Chinese vs English replies | `probe-qwen38-strata-draft-lang.py` |
| GPU memory sampling | `profile-vram.ps1` |
| Runtime/model/GPU preflight | `check-runtime.ps1` |

Setup and inventory helpers are also retained because they produce reproducible
state rather than launch an alternative runtime:

- `download-models.ps1` — fetches the Qwen deployment set.
- `stage-qwen38-turbofcfusion-assets.ps1` — fetches and verifies the pinned
  LOW-MTP-IQ4_XS target and matching Qwen3.8 vision projector.
- `stage-qwen38-humanlike-chat-vision-mtp.ps1` — fetches and verifies the pinned
  Humanlike IQ4_XS target plus the matching base Qwen3.8 MTP and vision assets.
- `stage-gemma4-assets.ps1` — stages and verifies the shared Google MTP drafter
  used by Gemma Fable-5 Distill.
- `stage-gemma4-hauhaucs.ps1` — stages and verifies the pinned HauhauCS
  uncensored QAT target, MTP drafter, and vision projector.
- `collect-host-inventory.ps1` — refreshes `docs/host-inventory.md`.
- `record-model-manifest.ps1` — refreshes `docs/models.md`.

For DSH integration, use the dedicated
[Qwen3.8 Flash-Next vision runbook](../docs/dsh-qwen38-flash-next-vision.md).
It documents the public-RPC catalog fields that automatic discovery cannot
infer: `input: ["text", "image"]`, the three reasoning efforts, and the
explicit `supportsDeveloperRole: false` compatibility setting.

The Kazusa router, Persona Gemma, Gemma 26B, DeepSeek FreeToken and official
Flash-Next launchers and dedicated helpers were removed on 2026-10-03.
Benchmark results remain in place. Shared helpers are retained for the
remaining models; `check-runtime.ps1 -IncludeGemma` checks Fable-5 Distill
and its shared MTP sidecar.

The unvalidated two-slot Qwen launcher remains removed. Cleanup and restoration
history is recorded in [docs/history.md](../docs/history.md).

The native Windows [Strata deployment](../docs/qwen38-flash-next-strata.md)
uses the same RTX 5090, a ModelOpt checkpoint under `models/`, CPU vision and
MTP. All Flash launchers share port `1919`; Strata defaults to LAN binding
(`0.0.0.0`) on that port and a 262K context, matching the older launchers.
Use `-BindAddress 127.0.0.1` for local-only access. `STRATA_API_KEY` is optional.
Stop the current backend before switching.
The pinned Strata engine has one active request slot; additional requests queue.
The launcher enables a 16 GiB host RAM conversation cache with eight parked
slots so sequential agent histories survive switches. `-ConversationCacheMiB 0`
disables it; layer-split experiments require that override. The cache runbook
includes the two-agent tool tests and Asuna connection/configuration findings.
`benchmark-qwen38-strata.py` measures exact 4K/128K inputs at that context,
records actual output lengths, and can reproduce overlapping request arrival
with `--concurrency`. It observes VRAM without cancelling a request at a threshold.

The uncensored FreeToken launchers, WSL wrapper, staging, verification and
benchmark helpers were removed on 2026-10-10 when Strata replaced FreeToken.
