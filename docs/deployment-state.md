# Deployment state — pre-load checkpoint

Captured: 2026-08-20

This workspace has been prepared from instruction.md and the RTX 5090/RTX 4090 handover documents. It intentionally stops before starting either backend, so no model from models/ has been loaded by this deployment.

## Completed

- Git repository initialized at the workspace root; the original handover documents were committed as the baseline.
- Required directories created: docs/, models/, scripts/, benchmarks/, and runtime/.
- Model artifacts are stored only under models/ and excluded from Git.
- Host inventory captured in host-inventory.md with Windows, CPU, RAM, GPU UUIDs, PCI identities, driver, CUDA, CMake, and MSVC details.
- DFlash2 llama.cpp PR #27342 fetched and pinned at commit 5ecbe1ac17ec0484c5b44af0bd580cdc9c428ed4.
- Native Windows Release server target built with CUDA 13.3 for CUDA architectures 89 and 120a; the server and matching DLLs are under runtime/llama.cpp-dflash2/build-dflash2/bin/Release/. The required server target is the accepted build scope for this checkpoint; optional all-target tooling is not part of the pre-load gate.
- llama-server --version, --help, and --list-devices checked without model arguments.
- Required primary and fallback GGUF files downloaded and SHA-256 recorded in model-manifest.md.
- Canonical per-GPU launch scripts prepared but not invoked.

## Physical-to-runtime GPU mapping

The physical mapping from host-inventory.md is authoritative:

| Backend | Physical identity | Launch isolation | Port | Initial target | Initial context |
|---|---|---|---:|---|---:|
| RTX 5090 | UUID GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0, PCI 00000000:01:00.0 | CUDA_VISIBLE_DEVICES=UUID, runtime CUDA0 | 8080 | UD-Q6_K_M | 131072 |
| RTX 4090 | UUID GPU-eed52936-813f-8d68-1654-bfb56cb42bc3, PCI 00000000:03:00.0 | CUDA_VISIBLE_DEVICES=UUID, runtime CUDA0 | 8081 | UD-Q4_K_XL | 110000 |

## Initial runtime configuration

Both launch scripts enforce:

- DFlash2 drafter: Q4_K_M;
- target K/V cache: Q8_0/Q8_0;
- draft live K/V cache: F16/F16;
- all target and draft layers on the selected GPU;
- split-mode none, parallel 1, flash attention on;
- no target CPU offload, no draft CPU offload, no mmproj, no context shift;
- `fit off` so automatic fitting cannot silently introduce an invalid production configuration.

The 5090 script starts at batch/ubatch 1024/256 and the 4090 script starts at 512/128, matching the handovers.

## Not yet done

These require loading the model and are deliberately left for the next step:

- start either backend or the eventual front-door router;
- verify full target/drafter GPU residency from runtime logs;
- measure loaded-idle/prefill/generation VRAM and the 1 GiB reserve;
- select stable context and DFlash2 n-max from local measurements;
- run cold start, TTFT, prefix-cache, PP/TG, correctness, harness, and long-context acceptance suites;
- write final benchmark conclusions and production sign-off.

## Next action

Run scripts/verify-preload.ps1. If it passes, the workspace is at the requested point immediately before model loading. The first actual load should be made by explicitly invoking either scripts/start-5090.ps1 or scripts/start-4090.ps1 after confirming the other GPU's background usage and selecting the desired backend.
