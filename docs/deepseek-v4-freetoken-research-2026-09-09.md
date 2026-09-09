DeepSeek V4 on one RTX 5090: FreeToken feasibility research

Historical initial research, 2026-09-09. The user subsequently accepted any speculative decoder and authorized experimental builds, model download, and actual tests. The experiment described in `deepseek-v4-freetoken-test-2026-09-10.md` supersedes the original recommendation below. Upstream FreeToken inspected at `3d919e9bd94fc5454bdb50e09659648443e30f5e`.

**Conclusion:** I did not find a verified configuration satisfying all requirements together: one RTX 5090, GPU model computation with routed weights kept in RAM, Q3/Q4 weights, DFlash2, at least 250,000 context tokens, and Q8/Q8-or-better KV. There are concrete RAM and software blockers, rather than just a missing launch command.

The target assessed is `deepseek-ai/DeepSeek-V4-Flash-0731`, the V4 checkpoint explicitly listed by [FreeToken](https://github.com/FlashML-org/FreeToken/blob/3d919e9bd94fc5454bdb50e09659648443e30f5e/docs/models.md). This assumes Flash, not V4-Pro. The local [host inventory](host-inventory.md) records a 32 GB RTX 5090, Ryzen 9950X3D, and 127.6 GiB system RAM. The RTX 4090 is excluded by the user's instruction. Hardware figures are the recorded baseline, not fresh measurements.

| Requirement | Finding |
|---|---|
| One RTX 5090, experts in RAM, GPU computation | FreeToken's `offload` backend implements this placement. Explicitly use `--moe-backend offload --moe-cpu-layers 0`; do not rely on `auto`. |
| Q3/Q4 weights | The supported native DeepSeek path uses MXFP4 experts and FP8 dense weights. GGUF Q3/Q4 requires an experimental DeepSeek adapter. MXFP4 is a different format from GGUF Q4_K. |
| Existing 128 GB RAM | Native target expert bank alone is approximately 137.06 GiB, too large. |
| DFlash2 | No compatible public DeepSeek V4 DFlash2 checkpoint and verified FreeToken integration found. |
| At least 250K context | Model context is sufficient, but an actual 250K request on the required single-card combination is unverified. Plan for 262,144 total tokens, including output. |
| Q8/Q8 or better | FreeToken's DeepSeek cache is not conventional separate llama.cpp K/V pools. Attention values undergo FP8 rounding; the separate indexer uses FP4 rounding, even though buffers are BF16. This needs an explicit precision interpretation and validation. |

The placement flags are documented in the [FreeToken CLI](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md). GPU computation still involves CPU tokenization, scheduling, memory management and PCIe transfers. It does not mean every weight remains in VRAM. The existing local [Qwen deployment](qwen38-flash-next-freetoken.md) confirms why `--moe-cpu-layers 0` matters under WSL.

**RAM and quantization**

The official checkpoint's [index](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/model.safetensors.index.json) reports 166,878,536,440 bytes, or 155.42 GiB. Its [configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/config.json) specifies 43 target layers, 256 routed experts, hidden dimension 4096, and expert intermediate dimension 2048. The [loader](https://github.com/FlashML-org/FreeToken/blob/3d919e9bd94fc5454bdb50e09659648443e30f5e/python/freetoken/models/deepseek_v4/weight.py) describes packed MXFP4 values and one E8M0 scale per 32 values, skipping draft experts.

Calculated target routed-bank size:

`43 * 256 * 3 * 4096 * 2048 * (0.5 + 1/32) / 2^30 = 137.0625 GiB`.

This is a format/shape calculation, not measured RSS. It excludes dense weights, loading transients, draft weights, and OS overhead. GPU expert-cache copies do not eliminate the full host source bank. Consequently adding the 5090's VRAM to system RAM is not valid capacity accounting for this backend.

For comparison, I summed published shard sizes from the [Unsloth 0731 GGUF repository](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/tree/main):

| Checkpoint | File size, GiB |
|---|---:|
| UD-IQ3_XXS | 97.05 |
| UD-IQ3_S | 108.10 |
| UD-Q3_K_M | 119.28 |
| UD-Q3_K_XL | 119.40 |
| UD-IQ4_XS / UD-IQ4_NL | 127.28 |
| UD-Q4_K_XL | 144.44 |

These are disk sizes, not runtime memory predictions. IQ3 is the most plausible capacity direction on this host, but compatibility comes first. [Open PR #210](https://github.com/FlashML-org/FreeToken/pull/210), head `9166b738703476cc91699081312a20c56d6b1058`, adds DeepSeek GGUF. Its author reports two smoke prompts on an antirez Q4 checkpoint, CPU decode layers, and disabled CUDA graphs after crashes. The PR warns that most Unsloth variants fail because expert quantization types differ across layers. This is not evidence that the smaller files above can simply be loaded in FreeToken.

For the native checkpoint, 192 GB system RAM is a reasonable initial capacity target and 256 GB offers more working margin. These are planning estimates requiring motherboard and WSL verification. A RAM upgrade alone would not resolve DFlash2 support or prove 250K operation.

**Speculative decoding and cache precision**

[PR #258](https://github.com/FlashML-org/FreeToken/pull/258) is open and explicitly implements DFlash v1, demonstrated on Qwen3.6 with an H20. It does not demonstrate DFlash2 or DeepSeek V4. The [Z Lab model catalog](https://huggingface.co/z-lab) lists DFlash2 drafts for Qwen3.8-27B and Muse-Glimmer; no matching V4 draft was found in the catalog/search. The local Qwen drafter cannot be assumed compatible.

[PR #69](https://github.com/FlashML-org/FreeToken/pull/69), head `2cf938b7bba362e577aaa4a059188948d1692bcd`, implements DeepSeek's embedded DSpark draft. It remains open, depends on a runtime foundation PR, and reports validation on a TP4 hybrid system. DSpark is the relevant existing draft architecture for this checkpoint, but it is a change from the requested DFlash2. Neither the PR nor its validation establishes the requested single-5090 GPU-only setup. Combining it with the independent GGUF PR would be development work.

The [paged pool](https://github.com/FlashML-org/FreeToken/blob/3d919e9bd94fc5454bdb50e09659648443e30f5e/python/freetoken/kvcache/dsv4_paged_pool.py) allocates BF16 KV/compressed/indexer buffers. However, [attention.py](https://github.com/FlashML-org/FreeToken/blob/3d919e9bd94fc5454bdb50e09659648443e30f5e/python/freetoken/models/deepseek_v4/attention.py) rounds non-RoPE attention KV to FP8, and [compress.py](https://github.com/FlashML-org/FreeToken/blob/3d919e9bd94fc5454bdb50e09659648443e30f5e/python/freetoken/models/deepseek_v4/compress.py) does the same for attention compression while using FP4 for the separate Lightning Indexer. FP8 and Q8_0 are not numerically interchangeable. If the floor covers attention KV only, the native attention path is 8-bit with higher-precision RoPE; if it covers all cached keys including the indexer, the native FP4 indexer fails that interpretation. BF16 storage must not be described as unquantized BF16 precision.

**Closest community references**

| Reference | What it establishes | Why it is not an exact match |
|---|---|---|
| [5090 + 128 GB Reddit report](https://www.reddit.com/r/DeepSeek/comments/1veqhvf/recommended_way_to_run_deepseek_v4_flash_on_rtx/) | A commenter reports IQ3_XXS, approximately 17 decode tok/s and a 1M context configuration. | llama.cpp; GPU-only expert computation, DFlash2, cache precision, and a genuinely filled 1M context are not established. |
| [5090 + 9950X3D + 256 GB deployment](https://www.reddit.com/r/LocalLLaMA/comments/1vfbcgx/deepseekv4flash0731_full_1m_context_on_a_single/) | The author documents an Lvllmds4-x setup and later a 500,084-token seed plus a cached follow-up. | Uses CPU/hybrid computation and DSpark. The update narrows the original 1M headline to validated 500K; cold prefill took about 40.5 minutes. |
| [FreeToken DSpark issue #173](https://github.com/FlashML-org/FreeToken/issues/173) | A PR-build report documents actual DSpark serving and verification/cancellation problems. | Experimental DSpark, not DFlash2, and no 250K validation. |
| [FreeToken GGUF PR #210](https://github.com/FlashML-org/FreeToken/pull/210) | Concrete DeepSeek GGUF loader work and a small real-model smoke test. | Different hardware, CPU decode, graph issues, and checkpoint restrictions. |

Community measurements are author reports, not reproduced benchmarks. In particular, configured context, short-context decode speed, and long-context retrieval are separate claims.

**Recommended decision**

With all requirements fixed, this is currently a development project with unresolved compatibility, not a ready deployment recipe. On the existing RAM, first establish a supported Q3 expert-bank path; then prove a matching DFlash2 draft/runtime exists. Do not download a large Q3 checkpoint merely because its disk size fits. If hardware is expanded, native MXFP4 with FreeToken offload becomes a more practical baseline, but DSpark would still be a separately agreed alternative to DFlash2.

Any eventual acceptance run should use only the 5090 UUID, one request, zero CPU MoE layers, a verified RAM-resident expert bank without swap, and a 262,144-token pool. Submit at least 250,000 actual prompt tokens with output space remaining; measure cold prefill, client-observed decode, RAM/VRAM peaks, distributed retrieval anchors, prefix-cache reuse, and speculative acceptance/parity. The existing workspace's 1 GiB free-VRAM stress margin is a useful acceptance target. A successful server startup alone does not meet these checks.
