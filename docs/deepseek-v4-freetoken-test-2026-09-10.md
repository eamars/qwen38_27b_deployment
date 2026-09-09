DeepSeek V4 Flash: single-5090 FreeToken experiment

Tested 2026-09-10. This report supersedes the initial research-only recommendation. The user permits any speculative decoder, experimental PRs, and downloads, and subsequently replaced the 250K minimum with a **180K maximum**. No second GPU is used for model computation.

**Result:** GPU-only expert inference works on the 5090 with Q3 experts pinned in RAM. The live server is capped at 179,968 usable tokens. It completed a 179,047-token prompt with correct retrieval of the opening reference code. The restored 1,536-slot configuration measured **19.39 tokens/s** after a cold prefill taking **488.76 seconds**. Speculation worked on the copy test after fixes but was slower, so the default is ordinary decoding.

The downloaded checkpoint is [tarruda/DeepSeek-V4-Flash-0731-GGUF, IQ3_XXS](https://huggingface.co/tarruda/DeepSeek-V4-Flash-0731-GGUF/tree/ad2ca0244d0e62868d0b79d67d14a42d6c41278d/IQ3_XXS), revision `ad2ca0244d0e62868d0b79d67d14a42d6c41278d`. Its four files occupy 106.05 GiB. Header inspection found all 129 routed-expert tensors use IQ3_XXS, with no Q2 expert tensors. The live, pinned host expert bank is 98.765625 GiB. The dense parameters occupy 9.53 GiB on the GPU after loading. The checkpoint remains in `/home/rba90/models/DeepSeek-V4-Flash-0731-IQ3_XXS` inside WSL.

The runtime combines the [experimental GGUF work](https://github.com/FlashML-org/FreeToken/pull/210), using [vcruz305's mixed-quant-bank branch](https://github.com/vcruz305/FreeToken/tree/fdf9e171458567152f45a776b705fd6d4a3fae04), with the infrastructure from [DSpark PR #69](https://github.com/FlashML-org/FreeToken/pull/69) at `2cf938b7bba362e577aaa4a059188948d1692bcd`. Local changes fix the GGUF metadata adapter, pass the DeepSeek SwiGLU clamp to the GGUF expert kernel, and use the exact official tokenizer. Prompt-lookup speculation was tested separately; it does not use a DSpark draft checkpoint or require extra model weights.

The tokenizer fix matters: the GGUF adapter's Qwen converter split `7319` differently and failed to recognize DeepSeek's chat markers. All 129,280 vocabulary IDs match the [official tokenizer](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/7872f01b1d1fe23eabc4c98b48bffcef5a386062), which is now loaded directly. Preliminary benchmarks made before this correction are not acceptance results.

Verified component probes: CUDA IQ3 expert multiplication against an independent dequantization reference; real 98.77 GiB host-bank registration; GPU cache-gather byte correctness and CUDA-graph replay; and shape correspondence for all 1,199 dense parameters. GPU cache-gather throughput measured 23.5 GB/s, versus about 28 GB/s for contiguous pinned-memory copies. These measurements are not model token throughput.

Current full-model configuration: RTX 5090 UUID `GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0`, TP=1, `offload`, zero CPU expert layers, BF16 runtime, **1,536 GPU expert-cache slots**, **179,968 total usable context tokens**, and 64 usable sliding-window pages. The launcher rounds the user's 180,000-token cap down to a 128-token page boundary and rejects launch requests above 180,000. The live cache-status endpoint reports 1,406 pages x 128 tokens. The KV pool occupies 2.24 GiB. The engine asserts that no CPU expert executor exists.

The full cold run used 1,536 expert slots and peaked at 31,687 MiB GPU usage, leaving 920 MiB free. A temporary reduction to 1,408 slots was reverted at the user's request: insufficient VRAM headroom has not been established as the cause of the earlier failure. Both the live server and launcher now use 1,536 slots. The historical 1,408-slot cached result remains in the evidence table but is not the final configuration.

The attention architecture retains its native 128-token window plus compressed historical attention. Limiting the window pool does not reduce the full compressed context to 8K. Prefill is chunked, with a 4,096-token configured cap; the scheduler may choose smaller aligned chunks. CPU tokenization, scheduling, and host-memory management still occur; CPU expert matrix computation is disabled.

Cache precision is not llama.cpp's `Q8_0/Q8_0` layout: native DeepSeek shared latent attention KV is FP8-rounded with BF16 backing storage and higher-precision RoPE components. The separate Lightning Indexer retains the model's native FP4 rounding. FP8 and Q8_0 are different formats; this is not an all-caches-Q8 implementation.

Corrected results:

| Test | Result |
|---|---|
| Short code recall | Exact `ORCHID-7319` |
| Coding, 24 prompt tokens, 255 output tokens | 19.74 client-observed decode tokens/s; 1.50 s TTFT |
| 8,031-token prompt, reference at beginning | Exact `ORCHID-7319`; 21.23 s TTFT |
| Original 250K prompt attempt | Failed during the chunk after 218,624 logged prefill tokens, with CUDA driver error after a severe slowdown; not a successful 250K run |
| Revised 180K maximum | Live cache verified at 179,968 usable tokens |
| Plain decode, 16-record copy test | Exact output; 19.37 tokens/s |
| Seven-token prompt-lookup draft, same copy test | Exact same output; 12.64 tokens/s; 203/238 proposed tokens accepted (85.3%) |
| 179,047-token cold prompt, 1,536 expert slots | Correct opening reference; 488.76 s TTFT; 255 output tokens at 19.39 tokens/s |
| Cached recall follow-up | Correct reference; 178,944 prompt tokens reused; 4.24 s TTFT |
| Historical 1,408-slot experiment (subsequently reverted), 179,047-token cached prompt | Correct reference; 4.42 s TTFT; 255 output tokens at 19.76 tokens/s |

Speculation is **disabled by default** because it was slower even on a favorable copy workload. The option supports a single greedy request and searches the last 8K tokens for a matching suffix; it is not a general learned drafter. The local verifier uses the PR's block-forward machinery, stages the actual proposals, writes the bonus token at the accepted position, and restores/replays the retained prefix after a rejection. A further fix makes detokenization process same-request token blocks sequentially; the first test exposed duplicated text without this fix. The corrected full-model copy test crossed page boundaries and matched the baseline exactly. This is narrow validation, not a general accuracy or concurrency certification.

The host reports a Gigabyte B850 AI TOP motherboard. During inference, the 5090 operates at PCIe 5.0 **x8**, although the GPU supports x16. [Gigabyte's specifications](https://www.gigabyte.com/Motherboard/B850-AI-TOP-rev-10/sp) state that populating the PCIEX8 slot makes the main slot run at x8. Excluding the 4090 from CUDA does not change physical lane sharing. Restoring x16 could improve RAM-to-GPU transfer bandwidth, but no x16 performance result has been measured here.

The existing Qwen runtime was left intact. Experimental checkouts are `runtime/freetoken-deepseek-experiment` and `runtime/freetoken-deepseek-spec`; the latter contains an uncommitted merge and local changes. Nothing was published or committed. Scripts under `scripts/*deepseek*` reproduce the download, component probes, server launch, and client benchmarks. Raw outputs and logs are in `/home/rba90/deepseek-freetoken-probe` inside WSL.

Start the tested default from PowerShell:

```powershell
wsl.exe bash /mnt/c/workspace/qwen38_27b/scripts/start-deepseek-freetoken-probe.sh
```

Experimental speculation, after stopping the existing server:

```powershell
wsl.exe env FREETOKEN_NGRAM=1 FREETOKEN_NGRAM_TOKENS=7 FREETOKEN_LOG_SUFFIX=-ngram7 bash /mnt/c/workspace/qwen38_27b/scripts/start-deepseek-freetoken-probe.sh
```

The OpenAI-compatible API is `http://127.0.0.1:1920/v1`, with model name `deepseek-v4-flash-gpu-probe`. Requests that omit an output limit default to 512 tokens in this test launcher; clients may request more, within the shared prompt/output context budget. The isolated build uses the existing WSL Python environment, CUDA 13.3, and Clang for the experimental CUDA extension. Clang was installed locally to resolve the GCC/PyTorch extension compilation failure.

Portable evidence is exported under `artifacts/deepseek-freetoken`: compact benchmark results, component results, live cache geometry, server logs, and patches against each checkout's HEAD. These are local experimental artifacts, not an upstream release. The test server is left running at the address above, idle after successful generation. The near-limit test uses repetitive synthetic records and one retrieval target; it does not establish general long-document accuracy or production stability across all workloads.

MTP5 follow-up (completed): a fresh paired coding baseline measured **19.9173 tokens/s**, versus **7.7729 tokens/s with MTP5**, a **60.97% slowdown**. Both used the same 24-token coding prompt, temperature zero, a 256-token output budget, the 180K context cap, and 1,536 expert-cache slots. The baseline returned 255 tokens and MTP5 returned 256; throughput counts returned tokens. MTP5 received one untimed 64-token kernel warmup before the timed run. Output was coherent but not byte-identical. The benchmark automatically stopped MTP5 as soon as the slower result was measured. No further performance tuning or benchmarks followed.

This was an experimental GPU-only **RAM-staging MTP5** implementation, not stock FreeToken DSpark performance. The official 0731 revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062` supplies three MTP layers with a five-token block. Only its three draft shards were downloaded. All 94 non-expert draft parameter shapes matched; draft routed experts were converted to IQ3_XXS using ggml commit `7840aab`. One converted expert matrix passed the GPU dequantization probe with relative weight RMSE 0.2187; this is a conversion check, not an accuracy benchmark.

Registering all additional draft banks with CUDA failed with `cudaHostRegister: out of memory`. The completed run instead OS-locked 6.89 GiB of draft expert weights in RAM and staged selected experts to the GPU. CPU memory gathering/copying occurs, but all target and draft model computation runs on the RTX 5090; the runtime asserted that the CPU expert executor was absent. The target retained its original CUDA-pinned banks. MTP5 accepted 187/315 proposed tokens (59.4%) during the timed request. The prototype corrected target taps to official zero-based layers 40/41/42, used sequential Markov conditioning, and verified five proposals plus the anchor. It uses a separate draft attention window and fixed five-token verification.

WSL's RAM cap was temporarily raised from 112GB to 120GB for the extra draft banks and restored to 112GB after the benchmark. MTP5 is disabled by default. Evidence is in `artifacts/deepseek-freetoken/mtp5-comparison.json`, `benchmarks.json`, the MTP5 server logs, and `mtp_sidecar.py`. The original 180K/1,536-slot baseline service is restored and ready; no second GPU was used.

