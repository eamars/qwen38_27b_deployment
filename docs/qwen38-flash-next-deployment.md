# Qwen3.8-Flash-Next deployment and profiling

This workspace keeps the existing Qwen3.8-27B DFlash2 deployment unchanged.
Flash-Next uses a separate Qwen4Exp-compatible llama.cpp build because the
checked-in runtime is pinned to DFlash2 commit `5ecbe1ac17ec0484c5b44af0bd580cdc9c428ed4`.
The separate source is pinned to Qwen4Exp commit
`6c5afc86ae84448ae4d744e357017e2c490ad9c3`, and its four-shard Q4 model set
has been staged and loaded successfully.

The preparation work remains available as a load-free preflight. An explicit
load validation and fixed executor/context matrix completed on
2026-08-28. This path is experimental rather than production sign-off: MTP,
parity, and the requested 250k-token workload remain future work.

## Required layout

The default scripts expect these paths:

```text
runtime/llama.cpp-qwen4exp/build/bin/Release/llama-server.exe
models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf
models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf
models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf
models/Qwen3.8-Flash-Next-MTP-F16.gguf
```

The MTP head is optional for the no-MTP baseline. Adjust `-ModelPath` or
`--model` if the downloaded shard names differ. Do not hash or load the large
weights as part of preparation unless that is an explicit later step.

## No-load preparation

Run this to refresh the no-load preflight before selecting or revalidating a
deployment build:

```powershell
.\scripts\prepare-qwen38-flash-next.ps1
```

It writes [qwen38-flash-next-environment.json](qwen38-flash-next-environment.json) and uses only runtime
`--version`, `--help`, and `--list-devices` plus read-only host/GPU queries.
Use `-Strict` after the separate Qwen4Exp runtime and the four model shards are
staged:

```powershell
.\scripts\prepare-qwen38-flash-next.ps1 -Strict
```

For the accepted deployment manifest, hash every required shard and record a
deterministic model-set identity:

```powershell
.\scripts\prepare-qwen38-flash-next.ps1 -HashModels -Strict
```

Hashing reads the GGUF files but does not load a model or allocate GPU memory.

The manifest records the source/runtime state, GPU UUIDs, CUDA/CMake output,
required CLI options, artifact sizes, and missing items. The later explicit
load validation and measured matrix are recorded under
`benchmarks/qwen38_flash_next/2026-08-28/`.

## Deployment shape

The launcher uses the measured two-executor full-context baseline:

```text
CUDA0 = RTX 5090, CUDA1 = RTX 4090
split-mode = layer
tensor split = 38,10
gpu layers = all
per_layer_token_embd = CPU
n-cpu-moe = 33
load mode = none (CPU-resident weights are allocated in RAM, not mmap-paged from NVMe)
server context = 524288
slot context = 262144
slots = 2
KV = F16/F16
batch/ubatch = 2048/256
fit = off
```

The candidate began with the preserved one-slot measurements and GGUF tensor
byte sizes. A second 262144-token slot adds about 7.79 GiB on the RTX
5090 and 2.55 GiB on the RTX 4090. Each ordinary resident expert layer is
1500 MiB (layer 30 is 1750 MiB). A two-slot load at 35/38:10 measured 6133 MiB
and 3391 MiB idle; keeping layers 33 and 34 on the RTX 5090 subtracts exactly
3000 MiB. In the completed 2026-08-28 matrix, the worst recorded minimum free
VRAM was 1957 MiB on the RTX 5090 and 2321 MiB on the RTX 4090 in the
two-executor mid-context case. This is a measured 128k-prompt result, not a
250k-token validation or a production acceptance claim.
`CUDA_MODULE_LOADING=EAGER` is set for the child process so CUDA initializes
before the non-mmap host allocation reaches its peak.

Preview either profile without starting a process:

```powershell
.\scripts\start-qwen38-flash-next.ps1 -Profile Baseline -DryRun
.\scripts\start-qwen38-flash-next.ps1 -Profile Mtp -MtpModelPath 'models\Qwen3.8-Flash-Next-MTP-F16.gguf' -DryRun
```

The normal launcher invocation starts the server and therefore loads weights;
do not use it while the GPUs are occupied. Stop only the matching managed port
with the same interface as the other starter scripts:

```powershell
.\scripts\start-qwen38-flash-next.ps1 -Stop
```

The stop path matches both the pinned executable and port, so it does not stop
the separate Qwen3.8-27B or Gemma llama.cpp processes.

The pinned Qwen4Exp candidate does not implement the architecture-specific
Flash-Next MTP graph. Its generic llama.cpp MTP CLI flags do not change that.
The MTP profile is therefore a future-gated F16/BF16 harness: use it only after
Qwen4Exp MTP support and a compatible MTP GGUF are available. It puts the
draft head on one selected device and must be re-tuned against the no-MTP
baseline. Q8 MTP is not enabled by the launcher or profiler.

## Staged profiler

The profiler is also dry-run by default. It writes a plan and empty-schema
outputs without starting `llama-server`:

```powershell
python .\scripts\profile-qwen38-flash-next.py --stage context --dry-run
```

Only an explicit `--run` starts a fresh server for each case:

```powershell
python .\scripts\profile-qwen38-flash-next.py `
  --stage context `
  --run `
  --runtime 'runtime\llama.cpp-qwen4exp\build\bin\Release\llama-server.exe' `
  --model 'models\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf' `
  --output 'benchmarks\qwen38_flash_next\2026-08-28\matrix-e1-e2-short-mid-r3.json'
```

The active profiler has one fixed matrix: executor counts 1 and 2, prompt
classes short (about 4k tokens) and mid (about 128k tokens), and exactly three
repetitions of each combination. Other workloads and configuration sweeps have
been removed from its command-line interface.

Every case launches with `--ctx-size 524288 --parallel 2`, yielding two
262144-token slots. The one-executor rows leave one slot idle; the two-executor
rows synchronize one request on each slot. The harness rejects
`--stage-context` or `--context-values` values below 262144 so a smaller KV
cache cannot invalidate VRAM or throughput comparisons. VRAM headroom is
recorded for diagnosis and is not a hard rejection gate in the active run.

Every run records a raw server log, prompt hash, prompt/generation timings,
MTP counters, GPU peak/minimum memory, system RAM, process working set, process
CPU, page faults, and process disk-read counters. Without `psutil`, Windows
uses kernel process counters plus `GlobalMemoryStatusEx`. JSON is
written incrementally after each case, and CSV contains per-run plus median
rows. The active profile records headroom for diagnosis and does not reject a
result solely for falling below 1024 MiB; the completed matrix nevertheless
recorded at least 1957 MiB free on the RTX 5090 and 2321 MiB on the RTX 4090.

Profiler requests set `cache_prompt=false`, and profiler servers use
`--cache-ram 0`, so repeated measurements perform real prompt evaluation
instead of benchmarking prompt-cache hits.

The same run command resumes by default. Completed cases are skipped when the
selected case list and deployment identity match the existing JSON. Use
`--no-resume` only when a fresh repeated matrix is intended.

Do not create `BEST_CONFIG.json` from the starting values. Pass `--best-config`
only after measured runs exist; the script emits per-workload candidates and
leaves the single recommended default null until workload priorities are
chosen.

### Completed baseline — 2026-08-28

The fixed matrix completed with all four executor/workload combinations and
three repetitions per combination. Every measured row reported retrieval
correctness. The medians below are from the committed CSV; `e=2` represents
one request on each of the two full-context slots.

| Executors | Prompt | Median prompt tokens | Median TTFT (s) | Median prompt eval (tok/s) | Median generation (tok/s) | Median wall (s) | Min free VRAM (5090/4090 MiB) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | short | 3989 | 20.832 | 194.19 | 26.53 | 50.707 | 2231 / 3029 |
| 2 | short | 3989 | 41.449 | 95.73 | 13.45 | 75.518 | 2234 / 3001 |
| 1 | mid | 128010 | 798.800 | 160.35 | 18.12 | 842.810 | 2112 / 2422 |
| 2 | mid | 128010 | 1250.932 | 76.28 | 1.11 | 1746.701 | 1957 / 2321 |

The source records are [the JSON result](../benchmarks/qwen38_flash_next/2026-08-28/matrix-e1-e2-short-mid-r3.json),
[the CSV](../benchmarks/qwen38_flash_next/2026-08-28/matrix-e1-e2-short-mid-r3.csv),
and the local prompt/raw-log companion files. The result used Qwen4Exp
`6c5afc86a`, `38,10` layer splitting, `n-cpu-moe=33`, F16/F16 KV, and
2048/256 batch/ubatch. It does not establish MTP support, parity, or the
requested 250k-token workload.

## Build and repinning gate

The completed baseline used the pinned Qwen4Exp source and runtime recorded in
`qwen38-flash-next-build.json` and `qwen38-flash-next-environment.json`. For a
future source change or new accepted profile:

1. Select a Qwen4Exp-compatible llama.cpp commit and record the full SHA.
2. Build natively with CUDA support for both SM89 and SM120.
3. Confirm the build manifest records compiler, CUDA, driver, CMake, flags,
   and generated architectures.
4. Run the no-MTP 8k and 32k correctness checks.
5. Only then spend time on the 245k/250k workload and MTP stages.

The local baseline uses `6c5afc86ae84448ae4d744e357017e2c490ad9c3`, with the
short runtime identity `6c5afc86a`. This is a known-good load and benchmark
baseline for this host, not a general production acceptance claim. MTP remains
outside the current runtime/profile and must be validated separately.

The build helper refuses a non-40-character SHA, a mismatched `HEAD`, and a
dirty source tree unless `-AllowDirtySource` is explicit. It never fetches or
checks out code automatically. After reviewing the source tree, use:

```powershell
.\scripts\build-qwen38-flash-next.ps1 -Commit '<FULL_QWEN4EXP_SHA>' -Configure
.\scripts\build-qwen38-flash-next.ps1 -Commit '<FULL_QWEN4EXP_SHA>' -Build
```

The helper records [qwen38-flash-next-build.json](qwen38-flash-next-build.json), including the exact
SHA, CUDA architectures (`89;120`), tool output, CMake arguments, and runtime
version. It performs no model load. Do not update the Qwen4Exp source during a
profiling matrix. If the source changes, start a new baseline and record a new
SHA.

The native Windows build sets `GGML_CUDA_NCCL=OFF`. NCCL is a Linux library,
and this deployment uses layer split rather than the tensor-parallel all-reduce
path. A separate Linux/WSL build is required for any later NCCL experiment.
