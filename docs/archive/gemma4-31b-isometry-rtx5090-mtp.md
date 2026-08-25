# Gemma 4 31B Isometry Fabled Persona — RTX 5090

This setup uses the exact persona target already stored in LM Studio. It does **not** download or substitute an unmodified Gemma 4 target.

## Project-local assets

| Role | File | Source / verification |
|---|---|---|
| Target | `models/Gemma-4-31B-Isometry-Fabled-Persona.i1-Q4_K_M.gguf` | Copied from LM Studio; SHA-256 `1A4C20908471FF51916A35915BCE73874100331F80A7433F407F77993C5539F3` |
| RTX 4090 target | `models/Gemma-4-31B-Isometry-Fabled-Persona.i1-Q4_K_S.gguf` | Downloaded from the exact persona repository; 17,763,168,256 bytes; SHA-256 `0314CB669580935FE6D775006DDD1F81F362B82C1A784D9B0D65C03490665213` |
| Google MTP assistant | `models/mtp-gemma-4-31B-it-Q8_0.gguf` | `ggml-org/gemma-4-31B-it-GGUF`; SHA-256 `6B52AB20AF503AEE320DC09E93F886133B18D89FFC9075C7D9CAF681E20B375` |

The MTP assistant is a separate, small drafter. It is not the target model and does not change the target’s output distribution; the target verifies the proposed tokens. Because the target is a persona merge rather than the canonical `gemma-4-31B-it`, the profiler must measure acceptance and wall-clock speed instead of assuming compatibility or a speedup.

## Runtime

The launcher uses the project-local Windows llama.cpp build:

```text
runtime/llama.cpp-dflash2/build-dflash2/bin/Release/llama-server.exe
commit: 5ecbe1ac17ec0484c5b44af0bd580cdc9c428ed4
```

This build exposes the Gemma 4 assistant architecture and `--spec-type draft-mtp`.

## Stage or verify assets

```powershell
.\scripts\stage-gemma4-assets.ps1
```

The staging script copies the exact LM Studio target and downloads only the named MTP sidecar. It refuses a different target file and never downloads an unmodified Gemma target.

## Startup

Print the command without touching the GPU:

```powershell
.\scripts\start-gemma4-5090.ps1 -DryRun
```

Start target + Google MTP when the RTX 5090 is available:

```powershell
.\scripts\start-gemma4-5090.ps1
```

Start the target-only control server:

```powershell
.\scripts\start-gemma4-5090.ps1 -NoMtp -Port 8082
```

Stop the managed server:

```powershell
.\scripts\start-gemma4-5090.ps1 -Stop
```

The initial RTX 5090 defaults are one slot, 65,536 context tokens, target K/V `q8_0`, assistant K/V `f16`, and MTP `n-max=3`. Increase context only after measuring VRAM headroom.

## RTX 4090 startup

The 4090 launcher uses the smaller exact-persona `i1-Q4_K_S` target, one slot,
55 Ki tokens (`56320`), target K/V `q8_0/q8_0`, assistant K/V `q8_0/q8_0`,
and MTP `n-max=3` by default. It binds the child process to the detected RTX
4090 UUID and does not load the multimodal projector.

Print the command without touching the GPU:

```powershell
.\scripts\start-gemma4-4090.ps1 -DryRun
```

Start target + Google MTP when the RTX 4090 is free:

```powershell
.\scripts\start-gemma4-4090.ps1
```

Use `-NoMtp` for the target-only control, or override `-CacheTypeK` and
`-CacheTypeV` after profiling.

The ranged downloader is resumable and restricted to the exact persona file:

```powershell
.\scripts\download-gemma4-persona-ranged.ps1
```

## MTP profiler

The profiler is dry-run unless `--run` is supplied. It starts target-only and MTP servers sequentially on port 8092, sends identical deterministic prompts, and records TTFT, prompt processing, generation speed, drafted/accepted tokens, acceptance ratio, and total wall time.

```powershell
python .\scripts\profile-gemma4-mtp.py
```

When the GPU is free:

```powershell
python .\scripts\profile-gemma4-mtp.py --run --repetitions 3 --max-tokens 256
```

Results are written to `benchmarks/gemma4-mtp-profile.json`; per-mode llama.cpp logs are written beside it. The key comparison is the median `tg_tokens_per_second` and wall-clock time, not acceptance ratio alone.

For the RTX 4090, use the matrix profiler below. It is dry-run by default and
checks the GPU with `--check-gpu` without starting llama-server. When the GPU
is free, `--run` tests target-only controls and MTP with target K/V profiles
`q8_0/q8_0`, `q8_0/q4_0`, and `q4_0/q4_0`, plus draft lengths 2 and 3. It
records startup health, model discovery, observed VRAM, generation speed, and
MTP acceptance in `benchmarks/gemma4-4090-mtp-profile.json`.

```powershell
python .\scripts\profile-gemma4-4090.py --dry-run --check-gpu
python .\scripts\profile-gemma4-4090.py --run --repetitions 3 --max-tokens 256
```

No model is loaded by the dry-run commands. The profiler refuses an
unmodified Gemma target path.

## Research references

- [Gemma 4 MTP documentation](https://ai.google.dev/gemma/docs/mtp/mtp)
- [Official Gemma 4 31B GGUF collection and MTP sidecar](https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF)
- [llama.cpp Gemma 4 MTP support](https://github.com/ggml-org/llama.cpp/pull/23398)
