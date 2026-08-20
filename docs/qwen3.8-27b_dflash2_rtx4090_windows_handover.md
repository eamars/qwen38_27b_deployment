# Qwen3.8-27B + DFlash2 — RTX 4090 24 GB Windows Handover

**Purpose:** setup-agent handover.  
**Host:** Windows; CUDA/CMake/MSVC build environment is already established. Do not reinstall it.  
**GPU:** 1× RTX 4090 24 GB.  
**Workload:** single-user coding / agentic inference.  
**Hard constraints:** DFlash2 mandatory; target KV must be Q8_0 or better.  
**Priority:** target-model capability first; context approximately 130K where practical.  
**Research date:** 2026-08-20.

## 1. Production recommendation

The 24 GB card has a real target-quality/context trade.

### Capability-first profile — recommended

| Item | Selection |
|---|---|
| Target | Qwen3.8-27B |
| Repository | `unsloth/Qwen3.8-27B-GGUF` |
| **Target quant** | **`UD-Q4_K_XL`** |
| File | `Qwen3.8-27B-UD-Q4_K_XL.gguf` |
| File size | ~17.6 GB |
| Context | **~110000** |
| Target K/V cache | **Q8_0 / Q8_0** |
| DFlash2 | `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` |
| DFlash2 size | ~1.14 GB |
| DFlash2 K/V cache | **F16 / F16** |
| GPU offload | all target + all draft layers |
| Flash Attention | on |
| Slots | **1** |
| Vision/mmproj | off |
| Context shift | off |

This is the primary recommendation because target-model capability is preferred over forcing the last ~20K tokens of context.

A current 4090 Q4_K_XL + DFlash2 report places the practical operating point around 110K context and reports roughly 84 tok/s decode.

### Context-first profile — if 131072 is required

Use:

```text
Qwen3.8-27B-UD-Q4_K_M.gguf
ctx = 131072
target KV = Q8_0/Q8_0
DFlash2 = Q4_K_M
draft KV = F16/F16
parallel = 1
```

This is now a better-supported 131K fallback than IQ4_XS.

A real headless RTX 4090 agentic test ran Q4_K_M + DFlash2 + Q8 target KV at 131072. However, it had only about **416 MiB free VRAM** and saw two draft-context allocation failures during real traffic. Therefore this is a context-first / low-margin profile, not the capability-first default.

Summary:

```text
Q4_K_XL @ ~110K = capability-first / preferred
Q4_K_M  @ 131K  = context-first / tight VRAM
```

## 2. Why not Q5 with lower-bit target KV?

Target KV is a hard constraint:

```text
--cache-type-k q8_0
--cache-type-v q8_0
```

Do not use Q4/Q5 target KV to buy a Q5 target.

For a long-context coding agent, target KV contains the target model's active representation of repository context, earlier reasoning, tool output and instructions. Current llama.cpp/Qwen reports have also shown severe prompt-processing regressions in some lower-bit KV paths.

If a larger target cannot fit:

```text
reduce context
or
reduce target-weight quant
```

not target KV quality.

## 3. Why DFlash2 remains Q4_K_M

DFlash2 is the proposal model. The target verifies its tokens.

Released DFlash2 GGUF measurements:

| Drafter | Approx. size | Acceptance length |
|---|---:|---:|
| BF16 | ~3.8 GB | 5.28 |
| Q8_0 | ~2.0 GB | 5.13 |
| **Q4_K_M** | **~1.1 GB** | **5.39** |

The saved VRAM is more valuable in the target model and target Q8 KV.

## 4. Draft KV stays F16

Use:

```text
--spec-draft-type-k f16
--spec-draft-type-v f16
```

F16 is above the Q8 floor.

There is a llama.cpp DFlash report in which Q8 draft KV collapses acceptance to nearly zero while F16/BF16 behaves normally. Do not quantize the live draft cache merely to squeeze a larger target/context into 24 GB.

## 5. Expected performance before local verification

All values below are **planning expectations**. The Windows host is the final source of truth.

### TG — Q4_K_XL capability-first

| Workload | Expected TG |
|---|---:|
| Short / predictable code | **80–105 tok/s** |
| Mixed coding-agent output | **65–85 tok/s** |
| Deep context around 90K–105K | **58–82 tok/s** |
| Reasoning / unpredictable prose | **50–70 tok/s** |

DFlash2 gain is acceptance-dependent. A controlled Q4-class 4090 test at about 105K measured ~57 tok/s DFlash2; matched real agent traffic was ~65 tok/s. Another Q4_K_XL/110K report shows ~84 tok/s.

### TG — Q4_K_M context-first @ 131K

Plan around:

```text
~55–75 tok/s deep-context mixed agent work
~80–100 tok/s favorable short/predictable code
```

Expect less operational margin than the Q4_K_XL profile.

### Prompt processing and uncached TTFT

DFlash2 mainly accelerates generation, not fresh prefill.

Expected PP:

| Prompt depth | Expected PP |
|---|---:|
| 2K–10K | ~2500–3000 tok/s |
| ~32K | ~2200–2600 tok/s |
| ~64K | ~1800–2200 tok/s |
| ~90K–105K | ~1500–1900 tok/s |

Expected **uncached** client-visible TTFT for the capability-first profile, with the model already resident:

| New uncached prompt tokens | Expected TTFT |
|---:|---:|
| ~2K | **0.9–1.6 s** |
| ~10K | **3.8–5.8 s** |
| ~32K | **13–18 s** |
| ~64K | **29–37 s** |
| ~90K | **48–62 s** |
| ~105K | **56–72 s** |

### Prefix-cache-hit TTFT

With a retained large prefix:

```text
~1K new suffix: ~0.6–1.5 s
~4K new suffix: ~1.8–3.5 s
```

Report cache-hit and uncached TTFT separately.

## 6. Windows paths

Example:

```text
C:\AI\llama.cpp-dflash2
D:\Models\Qwen38-4090\
```

```powershell
$Llama  = "C:\AI\llama.cpp-dflash2"
$Models = "D:\Models\Qwen38-4090"
```

## 7. Fetch and pin DFlash2 llama.cpp

DFlash2 release instructions currently use llama.cpp PR #27342.

```powershell
New-Item -ItemType Directory -Force -Path "C:\AI" | Out-Null
Set-Location "C:\AI"

if (-not (Test-Path $Llama)) {
    git clone https://github.com/ggml-org/llama.cpp.git $Llama
}

Set-Location $Llama
git fetch origin pull/27342/head:dflash2-pr-27342
git switch dflash2-pr-27342

$Commit = git rev-parse HEAD
$Commit | Set-Content "$Llama\DFLASH2_COMMIT.txt"
Write-Host "Pinned DFlash2 commit: $Commit"
```

Do not update after acceptance without rerunning correctness and performance tests.

## 8. Build for RTX 4090

Use the existing Windows toolchain:

```powershell
Set-Location $Llama

cmake -S . -B build-dflash2 `
  -DGGML_CUDA=ON `
  -DCMAKE_CUDA_ARCHITECTURES=89 `
  -DGGML_CUDA_FA_ALL_QUANTS=ON

cmake --build build-dflash2 --config Release --parallel
```

Locate and verify:

```powershell
$Server = (Get-ChildItem "$Llama\build-dflash2" -Recurse -Filter "llama-server.exe" |
    Select-Object -First 1).FullName

if (-not $Server) { throw "llama-server.exe not found" }

& $Server --version
& $Server --help | Select-String "draft-dflash|spec-draft|cache-type"
```

`draft-dflash` must be present.

## 9. Download models

```powershell
New-Item -ItemType Directory -Force -Path $Models | Out-Null

# Capability-first
hf download unsloth/Qwen3.8-27B-GGUF `
  Qwen3.8-27B-UD-Q4_K_XL.gguf `
  --local-dir $Models

# Context-first
hf download unsloth/Qwen3.8-27B-GGUF `
  Qwen3.8-27B-UD-Q4_K_M.gguf `
  --local-dir $Models

# DFlash2
hf download incoai/Qwen3.8-27B-DFlash2-GGUF `
  Qwen3.8-27B-DFlash2-Q4_K_M.gguf `
  --local-dir $Models
```

Do not load mmproj.

## 10. Launch — capability-first

```powershell
$Target = "$Models\Qwen3.8-27B-UD-Q4_K_XL.gguf"
$Draft  = "$Models\Qwen3.8-27B-DFlash2-Q4_K_M.gguf"

& $Server `
  --model $Target `
  --spec-draft-model $Draft `
  --alias "qwen3.8-27b-dflash2-4090" `
  --host 127.0.0.1 `
  --port 8080 `
  --gpu-layers all `
  --spec-draft-ngl all `
  --ctx-size 110000 `
  --parallel 1 `
  --kv-unified `
  --flash-attn on `
  --cache-type-k q8_0 `
  --cache-type-v q8_0 `
  --spec-type draft-dflash `
  --spec-draft-n-max 5 `
  --spec-draft-type-k f16 `
  --spec-draft-type-v f16 `
  --batch-size 512 `
  --ubatch-size 128 `
  --fit off `
  --no-mmproj `
  --no-context-shift `
  --jinja `
  --reasoning auto `
  --reasoning-preserve `
  --metrics
```

After baseline validation, benchmark `n-max=4`, `5`, and `7` at deep context and keep the best total wall-clock result.

## 11. Launch — context-first 131072

```powershell
$Target = "$Models\Qwen3.8-27B-UD-Q4_K_M.gguf"
$Draft  = "$Models\Qwen3.8-27B-DFlash2-Q4_K_M.gguf"

& $Server `
  --model $Target `
  --spec-draft-model $Draft `
  --alias "qwen3.8-27b-dflash2-4090-131k" `
  --host 127.0.0.1 `
  --port 8080 `
  --gpu-layers all `
  --spec-draft-ngl all `
  --ctx-size 131072 `
  --parallel 1 `
  --kv-unified `
  --flash-attn on `
  --cache-type-k q8_0 `
  --cache-type-v q8_0 `
  --spec-type draft-dflash `
  --spec-draft-n-max 5 `
  --spec-draft-type-k f16 `
  --spec-draft-type-v f16 `
  --batch-size 512 `
  --ubatch-size 128 `
  --fit off `
  --no-mmproj `
  --no-context-shift `
  --jinja `
  --reasoning auto `
  --reasoning-preserve `
  --metrics
```

This profile must survive real traffic without repeated:

```text
failed to find a memory slot
llama_decode returned 1
CUDA OOM
```

Booting successfully is not enough.

## 12. Windows VRAM logging

In another PowerShell window:

```powershell
$Log = "$env:TEMP\qwen4090-vram.csv"
"timestamp,total_mib,used_mib,free_mib" | Set-Content $Log

while ($true) {
    $row = nvidia-smi `
      --query-gpu=memory.total,memory.used,memory.free `
      --format=csv,noheader,nounits

    "$(Get-Date -Format o),$row" | Add-Content $Log
    Start-Sleep -Milliseconds 250
}
```

After the run:

```powershell
$rows = Import-Csv "$env:TEMP\qwen4090-vram.csv"
$minFree = ($rows | ForEach-Object { [int]$_.free_mib } |
    Measure-Object -Minimum).Minimum

Write-Host "Minimum free VRAM: $minFree MiB"
```

Record the result for both profiles. Prefer moving display duties to an iGPU/other adapter where possible.

## 13. TTFT verification

Measure client-visible streaming TTFT.

Capability-first tiers:

```text
~2K uncached
~10K
~32K
~64K
~90K
~105K
```

For every tier:

1. add a unique nonce/new suffix;
2. time from HTTP send to first streamed content/reasoning token;
3. record server prompt-eval tok/s;
4. run at least 3 times;
5. report median TTFT.

Then measure retained-prefix turns with approximately:

```text
1K new suffix
4K new suffix
```

Do not mix cached and uncached TTFT results.

## 14. TG verification

Run:

1. predictable code, >=2048 generated tokens;
2. real coding-agent task, >=1024 generated tokens;
3. reasoning/unpredictable task, >=1024 generated tokens.

For Q4_K_XL:

```text
<10K
80K–95K
95K–105K
```

For Q4_K_M / 131K:

```text
<10K
100K–115K
115K–125K
```

Record:

```text
prompt tokens
PP tok/s
client TTFT
TG tok/s
drafted tokens
accepted draft tokens
mean accepted length
total wall time
minimum free VRAM
draft allocation warnings/failures
```

## 15. Expected vs observed — capability-first

| RTX 4090 / Q4_K_XL | Expected | Observed |
|---|---:|---:|
| Context | ~110000 | |
| PP, short | 2500–3000 tok/s | |
| PP, ~90K–105K | 1500–1900 tok/s | |
| TTFT, ~2K uncached | 0.9–1.6 s | |
| TTFT, ~32K uncached | 13–18 s | |
| TTFT, ~64K uncached | 29–37 s | |
| TTFT, ~90K uncached | 48–62 s | |
| TTFT, ~105K uncached | 56–72 s | |
| Cache-hit TTFT, ~1K suffix | 0.6–1.5 s | |
| TG, mixed coding | 65–85 tok/s | |
| TG, ~90K–105K depth | 58–82 tok/s | |
| TG, reasoning | 50–70 tok/s | |
| Minimum free VRAM | record | |
| Draft allocation failures | none | |
| Long-context retrieval | pass | |
| DFlash2 deterministic parity | pass | |

## 16. Expected vs observed — context-first

| RTX 4090 / Q4_K_M | Expected | Observed |
|---|---:|---:|
| Context | 131072 | |
| Deep-context TG | ~55–75 tok/s | |
| Favorable code TG | ~80–100 tok/s | |
| Minimum free VRAM | likely very small; record | |
| Repeated draft-slot failures | **none allowed** | |
| 115K–125K retrieval | pass | |
| Deterministic parity | pass | |

A current real setup at this profile reported only ~416 MiB free, so local stability decides whether it is acceptable.

## 17. DFlash2 correctness gate

Compare target-only with DFlash2:

```text
temperature = 0
top_k = 1
fixed seed
```

Use at least:

- 5 coding prompts;
- 5 reasoning prompts;
- 5 JSON/tool prompts;
- 1 deep-context retrieval prompt.

Save runtime commit, CUDA/driver versions, target/draft hashes and both outputs.

Do not sign off unexplained deterministic divergence.

## 18. Long-context retrieval gate

Capability-first:

```text
90K–105K actual tokenizer tokens
```

Context-first:

```text
115K–125K actual tokenizer tokens
```

Put unique facts near the beginning, middle and far end and require all of them in the answer. Also generate at least 2048 tokens in a sustained test.

## 19. Troubleshooting order

Capability-first Q4_K_XL:

1. Remove Windows GPU-heavy workloads / move display to iGPU.
2. Lower `ubatch-size` from 128 to 64.
3. Lower context in 4096-token steps.
4. If ~131K is operationally more important, switch to Q4_K_M @ 131072.

Context-first Q4_K_M:

1. Eliminate non-LLM VRAM use.
2. Lower ubatch.
3. If draft-slot failures persist, reduce context modestly.
4. Do **not** lower target KV below Q8.

Never use partial CPU target offload as the normal production profile.

## 20. Final sign-off checklist

- [ ] Windows-native build.
- [ ] Exact DFlash2 llama.cpp commit recorded.
- [ ] CUDA target includes `sm_89`.
- [ ] Chosen target quant/context documented.
- [ ] Target K/V = Q8_0/Q8_0.
- [ ] DFlash2 Q4_K_M active.
- [ ] Draft K/V = F16/F16.
- [ ] Target and drafter fully GPU-resident.
- [ ] `parallel=1`.
- [ ] No mmproj.
- [ ] No context shift.
- [ ] Long-context retrieval passes.
- [ ] Deterministic parity gate passes.
- [ ] 2048-token sustained generation passes.
- [ ] Minimum free VRAM recorded.
- [ ] No repeated draft allocation failures.
- [ ] Uncached TTFT measured.
- [ ] Cache-hit TTFT measured.
- [ ] PP/TG/acceptance recorded.
- [ ] Locally selected `n-max` documented.

## Research references

- DFlash2 release / llama.cpp PR instructions: https://inco.ai/blog/dflash2/
- DFlash2 GGUF sizes and acceptance: https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF
- llama.cpp speculative CLI: https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- DFlash Q8 draft-cache acceptance issue: https://github.com/ggml-org/llama.cpp/issues/25725
- Qwen3.8 Unsloth quant files/sizes: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
- Real RTX 4090 long-context DFlash2 agentic test: https://www.reddit.com/r/LocalLLM/comments/1vsuf77/dflash2_speeds_qwen_38_27b_up_to_4_times/
- RTX 4090 prompt-processing data: https://www.hardware-corner.net/qwen3-8-27b-hardware-tests/
- Additional RTX 4090 Qwen3.8 field data: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/32

All expected TTFT/TG values are planning ranges. The final source of truth is the measured Windows host.
