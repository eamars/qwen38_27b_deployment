# Qwen3.8-27B + DFlash2 — RTX 5090 32 GB Windows Handover

**Purpose:** setup-agent handover.  
**Host:** Windows; CUDA/CMake/MSVC build environment is already established. Do not reinstall it.  
**GPU:** 1× RTX 5090 32 GB.  
**Workload:** single-user coding / agentic inference.  
**Hard constraints:** DFlash2 mandatory; target KV must be Q8_0 or better; context approximately 130K; **at least 1024 MiB free VRAM during the worst accepted long-context workload**.  
**Research date:** 2026-08-20.

## 1. Production recommendation

| Item | Selection |
|---|---|
| Target | Qwen3.8-27B |
| Repository | `unsloth/Qwen3.8-27B-GGUF` |
| **Target quant** | **`UD-Q6_K_M`** |
| File | `Qwen3.8-27B-UD-Q6_K_M.gguf` |
| File size | ~23.1 GB |
| Initial context | **131072** |
| Target K/V cache | **Q8_0 / Q8_0** |
| DFlash2 | `incoai/Qwen3.8-27B-DFlash2-GGUF` |
| DFlash2 quant | **Q4_K_M** |
| DFlash2 file | `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` |
| DFlash2 size | ~1.14 GB |
| DFlash2 K/V cache | **F16 / F16** |
| GPU offload | all target + all draft layers |
| Flash Attention | on |
| Slots | **1** |
| Vision/mmproj | off |
| Context shift | off |

Current Unsloth file sizes are approximately Q6_K=22.0 GB, Q6_K_M=23.1 GB, Q6_K_L=24.2 GB, Q6_K_XL=25.3 GB.

A current 5090 field test can run Q6_K_XL + Q8 target KV + MTP at 131072, but DFlash2 adds an external drafter and draft state. This deployment has a stricter criterion: **1 GiB must still be free under a real 115K–125K prompt plus sustained generation**. Q6_K_M is therefore the production starting point.

### Optional promotion

After Q6_K_M passes all tests, test `UD-Q6_K_L`. Promote it only if it independently keeps **>=1024 MiB free VRAM** throughout the full stress run. Never promote based on startup memory alone.

## 2. Capability/context fallback order

Use this exact order:

```text
1. UD-Q6_K_M @ 131072 + Q8 target KV + DFlash2
2. If minimum free VRAM < 1024 MiB:
      reduce context in 4096-token steps:
      126976 -> 122880 -> 118784
3. If Q6_K_M needs less than ~118K to preserve the reserve:
      use UD-Q6_K and restore 131072
4. Optionally test Q6_K_L only if it meets the same >=1024 MiB reserve
```

Do **not** lower target KV below Q8_0 to make a larger weight quant fit.

## 3. DFlash2 drafter precision

Keep the DFlash2 drafter at `Q4_K_M`.

DFlash2 proposes candidate tokens; the Qwen3.8 target verifies them. The target weights and target KV determine target-model capability. Drafter precision mostly affects acceptance and speed.

Released DFlash2 GGUF measurements:

| Drafter | Approx. size | Acceptance length |
|---|---:|---:|
| BF16 | 3.8 GB | 5.28 |
| Q8_0 | 2.0 GB | 5.13 |
| **Q4_K_M** | **1.1 GB** | **5.39** |

There is no good capability reason to spend another ~0.9–2.7 GB on the drafter instead of the target model / target KV.

## 4. Draft KV stays F16

Use:

```text
--spec-draft-type-k f16
--spec-draft-type-v f16
```

F16 is above the required Q8 floor.

Do not change DFlash live draft KV to Q8 merely to save memory. llama.cpp has a reported DFlash issue where Q8 draft KV can collapse acceptance to nearly zero while F16/BF16 behaves normally.

Target KV remains:

```text
--cache-type-k q8_0
--cache-type-v q8_0
```

## 5. Expected performance before local verification

These are **engineering expectations**, not guarantees. The setup agent must measure the actual Windows host and fill in the result table.

### TG expectation

For `UD-Q6_K_M + DFlash2`:

| Workload | Expected TG |
|---|---:|
| Short / predictable code | **110–160 tok/s sustained** |
| Very favorable code bursts | **up to ~180–200 tok/s** plausible |
| Mixed coding-agent output | **80–115 tok/s** |
| ~100K–120K prompt depth | **75–105 tok/s** |
| Reasoning / unpredictable prose | **65–90 tok/s** |

Basis: a current 5090 `Q5_K_L + DFlash2 + Q8 KV` run reports ~120 tok/s average on code-heavy generation, ~80–90 tok/s during thinking, and short ~200 tok/s code bursts. Current Q6_K_XL + MTP tests at 131K report roughly ~90–115 tok/s in representative runs. Q6_K_M + DFlash2 should be treated as somewhat workload-dependent.

### Prompt processing and uncached TTFT

DFlash2 accelerates decoding; it does not remove bulk prefill.

Expected prompt-processing envelope:

| Prompt depth | Expected PP |
|---|---:|
| 2K–10K | ~2800–3300 tok/s |
| ~32K | ~2600–3100 tok/s |
| ~64K | ~2350–2900 tok/s |
| ~100K–120K | ~2100–2700 tok/s |

Expected **uncached** client TTFT with the model already resident:

| New uncached prompt tokens | Expected TTFT |
|---:|---:|
| ~2K | **0.7–1.3 s** |
| ~10K | **3.2–4.8 s** |
| ~32K | **11–14 s** |
| ~64K | **22–29 s** |
| ~100K | **38–49 s** |
| ~120K | **45–59 s** |

These values do not include model startup/load time.

### Prefix-cache-hit TTFT

For an agent turn with a valid retained prefix, TTFT depends mainly on the **new suffix**, not the total nominal context.

Planning target:

```text
~1K new tokens after cache hit: ~0.5–1.2 s
~4K new tokens after cache hit: ~1.5–3.0 s
```

Report cached and uncached TTFT separately.

## 6. Windows paths

Example layout:

```text
C:\AI\llama.cpp-dflash2
D:\Models\Qwen38-5090\
```

```powershell
$Llama  = "C:\AI\llama.cpp-dflash2"
$Models = "D:\Models\Qwen38-5090"
```

Adapt paths only as required by the host.

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

Do not update the accepted runtime without rerunning correctness, VRAM, TTFT and TG tests.

## 8. Build for RTX 5090

Use the already-established Windows build environment:

```powershell
Set-Location $Llama

cmake -S . -B build-dflash2 `
  -DGGML_CUDA=ON `
  -DCMAKE_CUDA_ARCHITECTURES=120 `
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

`draft-dflash` must appear.

## 9. Download models

```powershell
New-Item -ItemType Directory -Force -Path $Models | Out-Null

hf download unsloth/Qwen3.8-27B-GGUF `
  Qwen3.8-27B-UD-Q6_K_M.gguf `
  --local-dir $Models

hf download unsloth/Qwen3.8-27B-GGUF `
  Qwen3.8-27B-UD-Q6_K.gguf `
  --local-dir $Models

hf download incoai/Qwen3.8-27B-DFlash2-GGUF `
  Qwen3.8-27B-DFlash2-Q4_K_M.gguf `
  --local-dir $Models
```

Optional promotion candidate:

```powershell
hf download unsloth/Qwen3.8-27B-GGUF `
  Qwen3.8-27B-UD-Q6_K_L.gguf `
  --local-dir $Models
```

Do not load mmproj for this text/coding service.

## 10. Production candidate launch

```powershell
$Target = "$Models\Qwen3.8-27B-UD-Q6_K_M.gguf"
$Draft  = "$Models\Qwen3.8-27B-DFlash2-Q4_K_M.gguf"

& $Server `
  --model $Target `
  --spec-draft-model $Draft `
  --alias "qwen3.8-27b-dflash2-5090" `
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
  --batch-size 1024 `
  --ubatch-size 256 `
  --fit off `
  --no-mmproj `
  --no-context-shift `
  --jinja `
  --reasoning auto `
  --reasoning-preserve `
  --metrics
```

Start with `n-max=5`. Once stable, benchmark `4`, `5`, and `7` with the same long-context workload. Choose by **total wall-clock completion time**, not acceptance alone.

## 11. Mandatory 1 GiB VRAM-reserve test

The hard condition is:

```text
minimum free VRAM during the accepted stress run >= 1024 MiB
```

In a second PowerShell window:

```powershell
$Log = "$env:TEMP\qwen5090-vram.csv"
"timestamp,total_mib,used_mib,free_mib" | Set-Content $Log

while ($true) {
    $row = nvidia-smi `
      --query-gpu=memory.total,memory.used,memory.free `
      --format=csv,noheader,nounits

    "$(Get-Date -Format o),$row" | Add-Content $Log
    Start-Sleep -Milliseconds 250
}
```

Stop with `Ctrl+C`, then:

```powershell
$rows = Import-Csv "$env:TEMP\qwen5090-vram.csv"
$minFree = ($rows | ForEach-Object { [int]$_.free_mib } |
    Measure-Object -Minimum).Minimum

Write-Host "Minimum free VRAM: $minFree MiB"

if ($minFree -lt 1024) {
    Write-Error "FAIL: RTX 5090 reserve below 1024 MiB"
}
```

The sampling window must include:

- fully loaded target + DFlash2;
- an actual 115K–125K-token prompt;
- at least 2048 generated tokens.

If the 5090 also drives Windows displays, close GPU-heavy applications during validation; using an iGPU/other adapter for display is preferable.

## 12. TTFT verification procedure

Measure TTFT from the **client** using streaming output.

Run uncached tests near:

```text
2K
10K
32K
64K
100K
120K
```

For each:

1. include a unique nonce/new suffix to prevent full-prefix reuse;
2. start the timer immediately before HTTP send;
3. stop on the first streamed content/reasoning token;
4. record server `prompt eval time` and PP tok/s;
5. run at least three times and use the median.

Then run cache-hit tests with an already-retained large prefix and approximately:

```text
1K new suffix
4K new suffix
```

Do not report a cache-hit 100K conversation as an "uncached 100K TTFT".

## 13. TG verification procedure

Run all three:

### Predictable code
Generate >=2048 tokens.

### Real coding-agent task
Use the actual harness/repository and generate >=1024 tokens.

### Reasoning / unpredictable text
Generate >=1024 tokens.

Test at:

```text
<10K prompt depth
90K–105K
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
```

## 14. Expected vs observed table

The setup agent must fill this in:

| RTX 5090 / Q6_K_M | Expected | Observed |
|---|---:|---:|
| Configured context | 131072 initially | |
| Min free VRAM under 115K–125K stress | **>=1024 MiB required** | |
| PP, short | 2800–3300 tok/s | |
| PP, ~100K–120K | 2100–2700 tok/s | |
| TTFT, ~2K uncached | 0.7–1.3 s | |
| TTFT, ~32K uncached | 11–14 s | |
| TTFT, ~64K uncached | 22–29 s | |
| TTFT, ~100K uncached | 38–49 s | |
| TTFT, ~120K uncached | 45–59 s | |
| Cache-hit TTFT, ~1K suffix | 0.5–1.2 s | |
| TG, mixed coding | 80–115 tok/s | |
| TG, ~100K–120K depth | 75–105 tok/s | |
| TG, reasoning | 65–90 tok/s | |
| Long-context retrieval | pass | |
| DFlash2 deterministic parity | pass | |

A speed result outside the expected band is a diagnostic trigger, not automatically a deployment failure. **VRAM reserve, correctness, no OOM and full GPU residency are hard requirements.**

## 15. DFlash2 correctness gate

Compare target-only and DFlash2 using:

```text
temperature = 0
top_k = 1
fixed seed
```

At minimum:

- 5 coding prompts;
- 5 reasoning prompts;
- 5 JSON/tool prompts;
- 1 prompt above 100K.

Save:

```text
llama.cpp commit
CUDA version
NVIDIA driver
target SHA256
draft SHA256
prompt
target-only output
DFlash2 output
```

DFlash2 is intended to preserve target output/distribution. Do not sign off unexplained deterministic divergence.

## 16. Long-context quality gate

Create a tokenizer-calibrated 115K–125K prompt. Insert unique facts around:

```text
~5K
~50K
~100K+
```

The final task must require all three facts.

Acceptance:

```text
all facts recovered
no OOM
target fully GPU-resident
draft fully GPU-resident
target KV Q8_0/Q8_0
draft KV F16/F16
minimum free VRAM >=1024 MiB
>=2048 output tokens sustained
```

## 17. Troubleshooting order

If Q6_K_M @ 131072 misses the 1 GiB reserve:

1. Reduce context: `126976`, then `122880`, then `118784`.
2. Reduce workspace: `--batch-size 512 --ubatch-size 256`, then `--ubatch-size 128`.
3. Rebenchmark PP/TTFT.
4. If ~130K is more important than remaining on Q6_K_M, switch to `UD-Q6_K` and restore 131072.

Never rescue the setup using target KV below Q8, CPU target-layer spill, CPU draft-layer spill, or multiple parallel slots.

## 18. Final sign-off checklist

- [ ] Windows-native build.
- [ ] Exact DFlash2 llama.cpp commit recorded.
- [ ] CUDA target includes `sm_120`.
- [ ] Target quant/context documented.
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
- [ ] **Minimum free VRAM >=1024 MiB.**
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
- RTX 5090 Qwen3.8 measurements: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/14
- RTX 5090 Qwen3.8 + DFlash2: https://www.reddit.com/r/LocalLLaMA/comments/1vs43av/i_tested_dflash2_for_qwen38_27b_on_a_5090/

All expected TTFT/TG values are planning ranges. The final source of truth is the measured Windows host.
