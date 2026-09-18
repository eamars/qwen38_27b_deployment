# Gemma 4 26B-A4B QAT, vision and MTP on RTX 4090

Launcher: `scripts/start-gemma4-26b-a4b-it-qat-ud-q4_k_xl-4090-vision-mtp.ps1`.

```powershell
# Only needed to stage or verify the three model files:
.\scripts\stage-gemma4-26b-assets.ps1

# Start after freeing the 4090 and port 8083:
.\scripts\start-gemma4-26b-a4b-it-qat-ud-q4_k_xl-4090-vision-mtp.ps1

# Inspect the command, or stop only this model on the specified port:
.\scripts\start-gemma4-26b-a4b-it-qat-ud-q4_k_xl-4090-vision-mtp.ps1 -DryRun
.\scripts\start-gemma4-26b-a4b-it-qat-ud-q4_k_xl-4090-vision-mtp.ps1 -Stop
```

The target is the instruction-tuned **26B-A4B MoE** (26B total, approximately 4B active), not a dense 26B model. QAT and a matching native MTP assistant are both available, so no unrelated substitute is needed.

## Defaults

| Setting | Value |
|---|---|
| Target | `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` |
| Drafter | `MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf` |
| Vision | `mmproj-F16.gguf`, GPU enabled, at most 1120 image tokens |
| Context | 262,144 tokens, the target's native maximum |
| Target and draft KV | Q8_0 / Q8_0 |
| MTP | `draft-mtp`, maximum 3 draft tokens |
| Execution slots | 1 (`--parallel 1`) |
| Context checkpoints | At most 6; runtime default spacing 8192 tokens |
| Batch / microbatch | 128 / 128 |
| Host prompt cache | Disabled (`--cache-ram 0`); six context checkpoints retained |
| GPU | RTX 4090 UUID `GPU-eed52936-813f-8d68-1654-bfb56cb42bc3` |
| Endpoint | `http://localhost:8083/v1`, binds `0.0.0.0` |

All target/draft layers and token embeddings are explicitly assigned to CUDA0, with the vision projector on the same GPU. There is no CPU expert offload, and `--fit off` prevents automatic fitting from silently changing the requested configuration. GPU visibility is restricted to the 4090, excluding the 5090.

This is not zero host-memory use: mmap/file caching, runtime metadata, host staging/output buffers, image preprocessing, and checkpoint snapshots still use RAM. CUDA placement of model tensors does not eliminate those allocations. The operating system/driver can also page memory under external pressure; keep other GPU models stopped. If desktop VRAM use increases, lower context explicitly, e.g. `-ContextSize 229376`.

The context budget includes input, image tokens and generated output. The drafter metadata declares 131,072 training tokens, so the runtime warns above that value. The capacity probe tests MTP near the target's maximum; it does not establish model quality or MTP acceptance across all long-context workloads.

## Reproducibility

All assets come from [Unsloth's 26B-A4B QAT repository](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF), pinned to revision `7b92b5b28818151e8669af2e45e88d6086f490dd`. The staging script verifies each file against its published SHA256. Downloads total 15,903,891,488 bytes (about 14.8 GiB).

The [publisher's MTP instructions](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/blob/7b92b5b28818151e8669af2e45e88d6086f490dd/MTP/README.md) identify the matching QAT assistant and quantized-KV support. The [Google model card](https://huggingface.co/google/gemma-4-26B-A4B-it) describes the target architecture and multimodal capabilities.

Runtime: existing `runtime/llama.cpp-dflash2/build-dflash2/bin/Release/llama-server.exe`, build `b10659-4e2f54f8f`, repository HEAD `4e2f54f8f34817e69e5db935221fc1223c4bb168`. Executable SHA256: `D29962FAF3B09634F9E004CD560F10F486765A2DBABFC5AD1A805DCF328C54D3`. No runtime source changes were required.

Probe scripts and response JSON are in `benchmarks/gemma4/26b-vision/`. They target a launcher started with `-Port 8094 -BindAddress 127.0.0.1`. The synthetic long-context probe tests allocation, decoding and a simple retrieval case; it is not a general quality benchmark.

Gemma uses non-causal attention for image chunks. This runtime requires batch and microbatch sizes to match; the launcher rejects mismatched values. The inherited 256/128 text-only configuration crashed on a 1536-pixel image and was replaced with 128/128 before final verification.

## Verified results (2026-09-19)

The final launcher was exercised on the actual RTX 4090 with all defaults, using an isolated test port:

- Loaded the full 262,144-token context. Logs show 31/31 target layers and 5/5 assistant layers on CUDA, CUDA vision encoding, and no CPU-mapped model tensor buffer. Global and sliding-window KV buffers both report Q8_0 K and V.
- Processed 260,096 input tokens without truncation in 145.9 seconds, recovered `ORCHID` from the beginning, and accepted 3/3 proposed MTP tokens while answering at the end.
- Generated a short coding response with 181/218 draft tokens accepted. This is one workload, not a general speed benchmark.
- Correctly identified a red square in a 1536x1536 PNG before and after the long-context request, with MTP active. The image request used 1,049 total prompt tokens.
- During the final long-context test, sampled whole-GPU memory peaked at 22,926 MiB with a minimum 1,213 MiB reported free. These figures include desktop/other GPU use and can vary; the vision projector was loaded throughout.
- Verified all three model SHA256 hashes, parsed both PowerShell scripts, and checked rejection of incompatible batch/microbatch settings.

The server was stopped after testing, and the previous 31B server was left unloaded as requested. Detailed evidence is in `benchmarks/gemma4/26b-vision/summary.json`, the text/vision/long-context response JSON, and local server logs. No claim is made about broad long-context quality, exhaustive vision accuracy, or arbitrary concurrent desktop VRAM pressure.
