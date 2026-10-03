# Maintained scripts

The scripts directory contains only maintained operational helpers. The
canonical launch and profiling entry points are:

| Purpose | Script |
|---|---|
| Qwen RTX 5090 launch/stop | `start-qwen27b-5090.ps1` |
| Qwen RTX 4090 launch/stop | `start-qwen27b-4090.ps1` |
| Qwen3.8 Turbo Fable Cold Fusion LOW-MTP-IQ4_XS, RTX 4090, 128K context, vision/MTP | `start-qwen38-turbofcfusion-4090-128k-vision-mtp.ps1` |
| Qwen3.8 Humanlike Chat IQ4_XS, RTX 4090, 128K context, vision/MTP | `start-qwen38-humanlike-chat-4090-128k-vision-mtp.ps1` |
| Shared Qwen + Gemma server | `start-kazusa-models.ps1` |
| Gemma 4 31B Isometry Fabled Persona i1-Q4_K_M, RTX 5090 | `start-gemma4-31b-isometry-fabled-persona-i1-q4_k_m-5090-65k-mtp.ps1` |
| Gemma 4 31B Isometry Fabled Persona i1-Q4_K_S, RTX 4090 | `start-gemma4-31b-isometry-fabled-persona-i1-q4_k_s-4090-56k-mtp.ps1` |
| Gemma 4 31B Isometry Fabled Persona i1-Q4_K_S, RTX 4090, 68K context, 6 context checkpoints | `start-gemma4-31b-isometry-fabled-persona-i1-q4_k_s-4090-68k-mtp-6-checkpoints.ps1` |
| Gemma 4 31B TeichAI Fable-5 Distill Q4_K_M, RTX 4090 | `start-gemma4-31b-fable-5-distill-q4_k_m-4090-70k-mtp.ps1` |
| Gemma 4 31B Instruct QAT UD-Q4_K_XL, RTX 4090, 90K context | `start-gemma4-31b-it-qat-ud-q4_k_xl-4090-90k-mtp.ps1` |
| Gemma 4 26B-A4B QAT UD-Q4_K_XL, RTX 4090, 262K context, vision/MTP | `start-gemma4-26b-a4b-it-qat-ud-q4_k_xl-4090-vision-mtp.ps1` |
| Download and verify the matching Gemma 4 26B target, MTP and vision assets | `stage-gemma4-26b-assets.ps1` |
| Gemma 4 31B HauhauCS QAT Uncensored Balanced Q4_K_M + vision/MTP, RTX 4090, 65K context | `start-gemma4-31b-hauhaucs-balanced-q4_k_m-4090-65k-mtp.ps1` |
| Qwen API smoke/sustained checks | `profile-api.py` |
| Qwen tokenizer-calibrated deep context | `profile-deep-context.py` |
| Flash-Next official text-only launch/stop | `start-qwen38-flash-next-freetoken.ps1` |
| Flash-Next uncensored text-only launch/stop | `start-qwen38-flash-next-uncensored-freetoken.ps1` |
| Flash-Next FreeToken 4K benchmark | `benchmark-freetoken-qwen38-next.py` |
| Flash-Next official vision launch/stop | `start-qwen38-flash-next-freetoken-vision.ps1` |
| Flash-Next uncensored vision launch/stop | `start-qwen38-flash-next-uncensored-freetoken-vision.ps1` |
| Flash-Next uncensored native Windows Strata vision/MTP launch/stop | `start-qwen38-flash-next-uncensored-strata-vision.ps1` |
| Install the pinned Windows Strata runtime and Python environments | `setup-strata-runtime.ps1` |
| Download and verify the ModelOpt NVFP4 checkpoint for Strata | `stage-qwen38-strata.py` |
| Convert the verified Strata checkpoint into workspace model assets | `prepare-qwen38-strata.py` |
| Strata real API, image, tool and retrieval/cache checks | `probe-qwen38-strata.py` |
| Strata alternating agent/tool histories, snapshot pressure, prefix changes and memory sampling | `probe-qwen38-strata-cache.py` |
| Asuna's installed DSH adapter, reasoning replay and waiting-parent cache probe | `probe-asuna-strata-cache.mjs` |
| Strata cold prefill, long streaming output, request queuing and VRAM measurements | `benchmark-qwen38-strata.py` |
| Flash-Next 262K/4K official/uncensored vision matrix | `benchmark-qwen38-flash-next-freetoken-vision-matrix.py` |
| DeepSeek V4 Flash FreeToken RTX 5090 launch/stop | `start-deepseek-freetoken.ps1` |
| DeepSeek V4 effort-aware request harness | `benchmark-deepseek-freetoken.py` (`--reasoning-effort` / `--thinking-effort`) |
| GPU memory sampling | `profile-vram.ps1` |
| Gemma MTP comparison | `profile-gemma4-mtp.py`, `profile-gemma4-4090.py` |
| Gemma short/long combined comparison | `profile-gemma4-combined.py` |
| Runtime/model/GPU preflight | `check-runtime.ps1` |

Setup and inventory helpers are also retained because they produce reproducible
state rather than launch an alternative runtime:

- `download-models.ps1` — fetches the Qwen deployment set.
- `stage-qwen38-turbofcfusion-assets.ps1` — fetches and verifies the pinned
  LOW-MTP-IQ4_XS target and matching Qwen3.8 vision projector.
- `stage-qwen38-humanlike-chat-vision-mtp.ps1` — fetches and verifies the pinned
  Humanlike IQ4_XS target plus the matching base Qwen3.8 MTP and vision assets.
- `stage-gemma4-assets.ps1` and `download-gemma4-persona-ranged.ps1` — stages
  the isolated Gemma experiment.
- `stage-gemma4-hauhaucs.ps1` — stages and verifies the pinned HauhauCS
  uncensored QAT target, MTP drafter, and vision projector.
- `collect-host-inventory.ps1` — refreshes `docs/host-inventory.md`.
- `record-model-manifest.ps1` — refreshes `docs/models.md`.
- `stage-qwen38-uncensored.py` — stages and verifies the pinned checkpoint in WSL.
- `verify-qwen38-uncensored.py` — checks all local tensor headers against the
  shared loader's expected layouts without reading tensor payloads.
- `probe-qwen38-uncensored-runtime.py` — checks the shared loaders with tiny
  synthetic tensors and the compiled CPU disk reader; never initializes CUDA.

The [uncensored deployment note](../docs/qwen38-flash-next-uncensored.md)
records the pinned assets, shared loader changes, preparation results and live
load checks. Generated verification reports belong under
`benchmarks/raw/qwen38-uncensored/`, which is ignored by Git.

For DSH integration, use the dedicated
[Qwen3.8 Flash-Next vision runbook](../docs/dsh-qwen38-flash-next-vision.md).
It documents the public-RPC catalog fields that automatic discovery cannot
infer: `input: ["text", "image"]`, the three reasoning efforts, and the
explicit `supportsDeveloperRole: false` compatibility setting.

`start-kazusa-models.ps1` launches one shared `llama-server` router with the
`qwen27b-5090` and `gemma4-4090` profiles. Their model-specific settings are
hard-coded in the launcher and emitted only to a temporary preset while the
server runs. There is no separate `kazusa-models.ini` file; the temporary
preset is deleted when the server exits.

The unvalidated two-slot Qwen launcher remains removed. Cleanup and restoration
history is recorded in [docs/history.md](../docs/history.md).

The Qwen3.8 Flash-Next FreeToken profile uses the RTX 5090,
NVFP4, disk-backed PLE, and `--moe-strategy offload --moe-cpu-layers auto`.
Preview without
loading weights:

```powershell
.\scripts\start-qwen38-flash-next-freetoken.ps1 -Profile Short4K -DryRun
```

The wrapper defaults to one concurrent request.

The separate native Windows [Strata deployment](../docs/qwen38-flash-next-strata.md)
uses the same RTX 5090, a ModelOpt checkpoint under `models/`, CPU vision and
MTP. All Flash launchers share port `1919`; Strata defaults to localhost
on that port and a 262K context. Stop the current backend before switching.
The pinned Strata engine has one active request slot; additional requests queue.
The launcher enables a 16 GiB host RAM conversation cache with eight parked
slots so sequential agent histories survive switches. `-ConversationCacheMiB 0`
disables it; layer-split experiments require that override. The cache runbook
includes the two-agent tool tests and Asuna connection/configuration findings.
`benchmark-qwen38-strata.py` measures exact 4K/128K inputs at that context,
records actual output lengths, and can reproduce overlapping request arrival
with `--concurrency`. It observes VRAM without cancelling a request at a threshold.

The two FreeToken vision launchers use the same shared runtime, port `1919`,
and max-context defaults. Each script fixes its own checkpoint and served model
ID, so DSH can deploy one model at a time without a model-selection argument:

| Launcher | Served model ID |
|---|---|
| Official vision | `qwen38-next-freetoken-vision` |
| Uncensored vision | `qwen38-next-uncensored-freetoken-vision` |

```powershell
.\scripts\start-qwen38-flash-next-freetoken-vision.ps1
.\scripts\start-qwen38-flash-next-uncensored-freetoken-vision.ps1
python .\scripts\benchmark-qwen38-flash-next-freetoken-vision-matrix.py
```

Stop the active model with its matching script:

```powershell
.\scripts\start-qwen38-flash-next-freetoken-vision.ps1 -Stop
.\scripts\start-qwen38-flash-next-uncensored-freetoken-vision.ps1 -Stop
```

The matrix fixes the context at 262,144 tokens, uses the retained 4,041-token
prompt, runs three measured requests per configuration, and writes the result
under `benchmarks/raw/qwen38_flash_next/<date>/`.
