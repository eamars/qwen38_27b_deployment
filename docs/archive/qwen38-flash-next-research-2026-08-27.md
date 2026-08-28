# Qwen3.8-Flash-Next llama.cpp Deployment, MTP Integration & Performance-Profiling Handover

**Status date:** 27 August 2026
**Target:** Qwen3.8-Flash-Next local deployment
**Hardware target:** RTX 5090 32 GB + RTX 4090 24 GB + 128 GB system RAM
**Primary objective:** Maximum practical single-request inference performance at up to ~250k-token working context while maintaining **Q4-or-higher model quality**
**Secondary objective:** Integrate and tune native MTP speculative decoding where it produces a genuine end-to-end performance improvement
**Serving interface:** OpenAI-compatible llama.cpp server

---

# 1. Purpose

This work is the follow-up deployment recipe to the previous Qwen3.8-27B deployment work.

Do **not** treat this as simply:

> “Find enough RAM to load a 111 GB GGUF.”

Qwen3.8-Flash-Next has a very unusual architecture and requires a topology-aware loading strategy.

The implementation must deliberately optimise the placement of:

1. the large PLE / n-gram embedding table;
2. ordinary non-MoE model tensors;
3. sparse MoE expert tensors;
4. long-context state and KV;
5. the MTP head;
6. work split between the RTX 5090 and RTX 4090;
7. CPU memory bandwidth;
8. PCIe traffic between the heterogeneous GPUs.

The final configuration must be selected through **profiling on the actual target machine**, not from theoretical memory calculations alone.

The implementation agent is explicitly instructed to build a repeatable benchmarking harness, conduct systematic profiling, and produce one or more final deployment recipes supported by measured results.

---

# 2. Model architecture relevant to deployment

Qwen3.8-Flash-Next consists of:

- **125B main-model parameters**
- an additional **51B N-gram embedding capacity**
- only approximately **6B parameters activated per token**
- hybrid Gated DeltaNet + Qwen Sparse Attention
- 48 hybrid transformer layers in the current llama.cpp implementation
- 512 MoE experts
- top-10 expert routing
- large per-layer PLE / n-gram embeddings

Qwen explicitly designed the N-gram embedding table so it can be held in **host memory**, with lookups overlapped with accelerator computation. citeturn931920view0turn355201view0

The current llama.cpp implementation describes the PLE hash table as approximately **97.7 GiB in its original representation** and performs host-side row lookup followed by `ggml_get_rows`. citeturn931920view1

This has a major architectural implication:

> **System RAM should not simply be treated as emergency overflow VRAM. It is an intentional part of the Qwen3.8-Flash-Next memory hierarchy.**

---

# 3. Primary deployment architecture

The desired steady-state layout is:

```text
                        Qwen3.8-Flash-Next
                               │
          ┌────────────────────┴─────────────────────┐
          │                                          │
   Host-memory path                            GPU compute path
          │                                          │
          │                            ┌─────────────┴─────────────┐
          │                            │                           │
PLE / N-gram embeddings          RTX 5090 32 GB             RTX 4090 24 GB
CPU-side MoE experts             larger layer share         smaller layer share
OS / model page cache            QSA / GDN / dense          QSA / GDN / dense
          │                      GPU-resident experts        GPU-resident experts
          │                            │                           │
          └────────────────────── PCIe / CPU ─────────────────────┘
```

The design principles are:

- keep **non-expert compute-heavy tensors on GPUs**;
- keep the PLE table on the host;
- selectively CPU-offload **MoE expert weights**, rather than entire transformer layers;
- use **layer splitting** across the 5090 and 4090;
- leave sufficient VRAM for long-context state and, where enabled, the MTP head;
- tune CPU expert residency against GPU expert residency empirically.

---

# 4. Do not use tensor-parallel mode for the initial deployment

Use:

```text
-sm layer
```

Do **not** make this deployment depend on:

```text
-sm tensor
```

As of 27 August 2026, the active Qwen4Exp llama.cpp PR specifically lists tensor split as WIP for Qwen3.8-Flash-Next. citeturn502379view2

The current llama.cpp CLI defines:

- `layer`: split layers and KV across GPUs;
- `row`: row-wise parallelisation;
- `tensor`: weight/KV tensor parallelism, currently experimental.

It also supports explicit GPU proportions through `-ts/--tensor-split`. citeturn786983search1

The 5090 and 4090 are:

- different performance classes;
- different VRAM capacities;
- connected through PCIe rather than a homogeneous high-bandwidth NVLink fabric.

Therefore start with **pipeline/layer placement**, with the 5090 receiving the larger share.

---

# 5. Target quantisation

## Initial production candidate

Use:

```text
unsloth/Qwen3.8-Flash-Next-GGUF
UD-Q4_K_XL
```

The currently published UD-Q4_K_XL is approximately **111 GB across four GGUF shards**. citeturn502379view3

This is the initial quality/performance target.

Do not begin below Q4.

### Quantisation priorities

In order:

1. **UD-Q4_K_XL**
2. equivalent Q4 mixed quant if profiling exposes a better implementation
3. Q5 mixed quant as a secondary quality-first experiment
4. Q6 only if performance remains unexpectedly good

Do not spend early profiling time on IQ2/IQ3 configurations.

The purpose of this deployment is not merely to make Flash-Next run; it is to retain a relatively high-quality local model.

---

# 6. llama.cpp version policy

## 6.1 Critical rule: pin the implementation

At the time of this handover, Qwen's own repository says llama.cpp supports Flash-Next, but the actual Qwen4Exp implementation remains under active development in llama.cpp PR **#27742**. citeturn355201view0turn931920view1

Therefore:

**Do not deploy from an unrecorded moving `master` or PR HEAD.**

The agent shall:

1. identify a Qwen4Exp commit that passes the required validation;
2. record its full Git SHA;
3. build and benchmark that exact SHA;
4. retain it as the known-good baseline;
5. only update after re-running regression tests.

Recent PR fixes have included:

- quantized-KV correctness;
- multi-slot/QSA state handling;
- `--fit` graph sizing;
- QSA cache placement;
- split granularity fixes.

This is evidence that the implementation is moving too quickly for an unpinned production recipe. citeturn931920view2

---

# 7. CUDA build target

The deployment should use a recent CUDA toolchain with native Blackwell support.

Current llama.cpp CMake explicitly notes that Blackwell architecture 120 requires **CUDA 12.8 or newer**, while the current Windows CI includes CUDA 13.3 x64 builds. citeturn502379search0turn502379search1

For this mixed system, the build must support both:

```text
RTX 4090 → Ada / SM89
RTX 5090 → Blackwell / SM120
```

Prefer a native build on the target host.

Generic build:

```powershell
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

The build manifest must record:

```text
llama.cpp Git SHA
compiler
CUDA toolkit version
NVIDIA driver version
CMake version
build flags
CUDA architectures generated
```

Do not optimise specifically for SM120 while accidentally falling back to a poor code path on the 4090.

---

# 8. Hardware discovery before loading

Before any profiling, capture:

```powershell
nvidia-smi
nvidia-smi topo -m
.\llama-server.exe --list-devices
```

Record the exact llama.cpp device mapping.

For example:

```text
CUDA0 = NVIDIA RTX 5090
CUDA1 = NVIDIA RTX 4090
```

is desired for the initial commands below.

If llama.cpp enumerates them differently, modify every `-dev`, `-ts`, and MTP-device parameter accordingly.

Never infer GPU numbering from Windows Task Manager.

---

# 9. Initial non-MTP baseline

The first milestone is a stable, high-performance configuration **without speculative decoding**.

This baseline is essential because all MTP measurements will be compared against it.

## Initial server configuration

Assuming:

```text
CUDA0 = RTX 5090
CUDA1 = RTX 4090
```

begin with:

```powershell
.\llama-server.exe `
  -hf "unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL" `
  -c 262144 `
  -np 1 `
  -dev CUDA0,CUDA1 `
  -ngl all `
  -sm layer `
  -ts 3,2 `
  -ot "per_layer_token_embd=CPU" `
  -ncmoe 28 `
  -fa on `
  -ctk f16 `
  -ctv f16 `
  -b 2048 `
  -ub 1024 `
  -fit off `
  --jinja `
  --host 127.0.0.1 `
  --port 8000
```

If `-hf` is unreliable with the development branch, download all GGUF shards and use the first shard explicitly:

```powershell
-m "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
```

---

# 10. Why `-ngl all` plus `-ncmoe` is important

The desired strategy is **not**:

```text
Put N entire transformer layers on CPU
```

Instead:

```text
-ngl all
-ncmoe N
```

The current llama.cpp CLI supports `-ncmoe/--n-cpu-moe`, which keeps the MoE weights of the first N layers on CPU while allowing the remaining parts of those layers to remain GPU-offloaded. citeturn786983search1

This is substantially better suited to Flash-Next because only a small subset of experts is activated for any token.

The primary memory/performance control knob should therefore be:

```text
-ncmoe
```

not `-ngl`.

---

# 11. PLE / N-gram embedding placement

Explicitly place:

```text
per_layer_token_embd
```

on the CPU:

```text
-ot "per_layer_token_embd=CPU"
```

## Important: NVMe is backing storage, not the intended hot inference path

Do not attempt to achieve low RAM usage by deliberately allowing the PLE table to continuously fault from NVMe.

Current Qwen4Exp development notes report that PLE accesses are highly non-sequential: the implementation performs multiple small gathers per token and ordinary sequential mmap readahead is ineffective. Current work is investigating per-tensor mmap behaviour for precisely this reason. citeturn502379view2turn355201search2

The target should therefore be:

```text
NVMe
  ↓ initial/backing storage

Windows page cache / physical RAM
  ↓

PLE lookups during steady-state inference
```

not:

```text
PLE lookup
  ↓
page fault
  ↓
NVMe read
  ↓
every token
```

### Profiling rejection condition

Once the model is warmed:

> sustained significant NVMe read traffic caused by PLE lookups is a failed configuration.

Monitor:

```text
physical RAM utilisation
model working set
page faults
hard page faults
NVMe read MB/s
NVMe IOPS
```

Do not use whole-model `mlock` blindly on a 128 GB machine with a ~111 GB GGUF.

The machine still requires RAM for:

- OS;
- llama runtime;
- CPU expert tensors;
- PLE hot pages;
- prompt buffers;
- page tables;
- API/runtime processes.

---

# 12. Long-context target

The model's published native deployment context is:

```text
262144 tokens
```

Qwen's own SGLang and vLLM examples use that context length. citeturn355201view0

Use:

```text
-c 262144
```

for the final max-context profile.

However, remember that context contains:

```text
prompt + generated tokens
```

Therefore a 250k-token input does not leave 250k tokens for generation.

Profiling should include:

```text
245k prompt + 4k generation
250k prompt + 2k generation
```

as realistic near-limit cases.

---

# 13. KV strategy

## Stage 1

Start with:

```text
-ctk f16
-ctv f16
```

This establishes the correctness baseline.

## Stage 2

After the F16 baseline is stable, benchmark:

```text
-ctk q8_0
-ctv q8_0
```

This is now worth testing because the current Qwen4Exp PR reports that a recent quantized-KV bug was fixed and that Q8_0 results are close to F16 in its validation. citeturn931920view2

The interesting question is **not merely whether Q8 KV is faster**.

The real optimisation is:

```text
Q8 KV
→ less VRAM consumed by context
→ lower -ncmoe possible
→ more experts resident on GPU
→ potentially higher token generation speed
```

Therefore whenever KV precision changes, the agent must re-tune `-ncmoe`.

Do not declare:

```text
F16 KV beats Q8
```

or vice versa without re-optimising expert placement.

Do not use Q4 KV as a production candidate during the first profiling cycle.

---

# 14. VRAM headroom policy

Do not tune until `nvidia-smi` shows only a few MB free.

A configuration that starts but OOMs on a large prompt is not acceptable.

After:

1. model load;
2. full-context allocation;
3. first warm request;

retain a practical safety margin.

Initial desired margin:

```text
RTX 5090: ~3–5 GiB
RTX 4090: ~2–3 GiB
```

This is a starting engineering target, not a fixed law.

The profiler may reduce it if repeated max-context stress tests demonstrate stability.

---

# 15. Initial GPU split

Start with:

```text
-ts 3,2
```

meaning approximately:

```text
RTX 5090 → 60%
RTX 4090 → 40%
```

of the GPU-split portion.

Do not assume this is optimal.

The 5090 has both:

- more VRAM;
- substantially more compute/memory performance.

The optimal split may therefore be more asymmetric.

Test at minimum:

```text
55 / 45
60 / 40
65 / 35
70 / 30
```

subject to actual VRAM capacity.

The profiling objective is not equal utilisation percentages.

The objective is minimum wall-clock inference time.

---

# 16. MTP integration status

This must be treated separately from the basic Qwen4Exp deployment.

As of this handover:

> llama.cpp PR #27742 explicitly says **“MTP — not in this PR yet.”** citeturn502379view2

Generic llama.cpp already supports:

```text
--spec-type draft-mtp
-md / --spec-draft-model
--spec-draft-n-max
--spec-draft-ngl
--spec-draft-device
--spec-draft-type-k
--spec-draft-type-v
```

but the Qwen4Exp-specific graph still needs a compatible implementation. citeturn502379view4turn786983view0

Therefore maintain **two branches/configurations**.

### Branch A — Known-good baseline

```text
qwen4exp-baseline
```

Pinned from the validated Qwen3.8-Flash-Next llama.cpp implementation.

No MTP dependency.

### Branch B — MTP experimental

```text
qwen4exp-mtp
```

Either:

1. use upstream Qwen4Exp MTP once it lands; or
2. port/rebase a validated Qwen4Exp MTP implementation onto the pinned baseline.

Do not destabilise Branch A while adding speculative decoding.

---

# 17. MTP correctness reference

A same-day downstream Qwen4Exp implementation has already added a routing-safe MTP graph and provides useful validation data.

Its reported behaviour includes:

- real 111 GB checkpoint validation;
- F16/BF16 MTP support;
- **68.3% generated-token acceptance with F16 MTP**;
- explicit rejection of quantized MTP heads because the authors found that quantisation can change expert routing. citeturn931920view5turn931920view6

This code should be treated as:

```text
implementation reference
+
validation oracle
```

rather than automatically becoming the production fork.

If upstream gains equivalent MTP support before implementation is complete, prefer the upstream implementation after validation.

---

# 18. MTP precision policy

## Production correctness baseline

Start with:

```text
BF16 or F16 MTP
```

Do **not** start by assuming a Q8 MTP head is safe.

There is currently conflicting evidence.

One experimental Q8 MTP implementation reports:

```text
MTP size: ~3.85 GiB
UD-IQ4_XS target:
24.2 t/s → 29.3 t/s
acceptance ≈ 0.623
```

on a Radeon 8060S. citeturn355201search0

However, the routing-safe Qwen4Exp implementation specifically reports that quantized MTP weights can change expert routing and therefore rejects quantized MTP export, while F16/BF16 remains supported. citeturn931920view5

Therefore:

### Required order

```text
1. F16/BF16 MTP correctness
2. F16/BF16 performance
3. Q8 MTP experiment
4. Q8 accepted only if routing/output/acceptance validation passes
```

Do not select Q8 merely because it consumes less VRAM.

---

# 19. MTP memory trade-off

A 4B MTP head at BF16/F16 is on the order of ~8 GB of weight storage before runtime overhead.

That creates a central optimisation problem:

```text
more GPU-resident target experts
                versus
GPU-resident MTP head
```

The MTP head may require raising `-ncmoe` so additional target experts move into CPU RAM.

Therefore:

> **MTP must beat the re-optimised no-MTP configuration, not an artificially identical target placement.**

Example:

```text
No MTP:
-ncmoe 22
tg = X

MTP:
-ncmoe 30
MTP entirely GPU resident
tg = Y
```

The relevant comparison is `Y` versus `X`.

Do not compare both configurations at `-ncmoe 30` and conclude MTP is superior.

---

# 20. Initial MTP launch shape

Once a compatible F16/BF16 MTP-only GGUF exists:

```powershell
.\llama-server.exe `
  -m "<MAIN_Q4_GGUF>" `
  -md "<QWEN38_FLASH_NEXT_MTP_F16.gguf>" `
  -c 262144 `
  -np 1 `
  -dev CUDA0,CUDA1 `
  -ngl all `
  -sm layer `
  -ts 3,2 `
  -ot "per_layer_token_embd=CPU" `
  -ncmoe <PROFILED_VALUE> `
  -fa on `
  -ctk f16 `
  -ctv f16 `
  --spec-type draft-mtp `
  --spec-draft-device CUDA0 `
  --spec-draft-ngl all `
  --spec-draft-type-k f16 `
  --spec-draft-type-v f16 `
  --spec-draft-n-max 3 `
  -b 2048 `
  -ub 1024 `
  -fit off `
  --jinja `
  --host 127.0.0.1 `
  --port 8000
```

Do not assume `CUDA0` is automatically optimal for the draft.

Profile:

```text
MTP → RTX 5090
MTP → RTX 4090
```

and, if supported and meaningful:

```text
MTP split
```

I expect a single-device MTP placement to be preferable, but this must be measured.

---

# 21. Mandatory performance-profiling work

The implementation agent is explicitly instructed:

> **Do not stop when the model runs. Perform systematic profiling to determine the best configuration for this exact machine.**

A configuration is not considered complete until profiling results exist.

Create an automated benchmark harness.

Suggested structure:

```text
qwen38_flash_next_deployment/
│
├─ README.md
├─ BUILD.md
├─ configs/
│  ├─ baseline.ps1
│  ├─ max_context.ps1
│  ├─ mtp.ps1
│  └─ fallback.ps1
│
├─ scripts/
│  ├─ profile.ps1
│  ├─ run_benchmark.py
│  ├─ collect_gpu_stats.ps1
│  └─ analyse_results.py
│
├─ benchmark/
│  ├─ prompts/
│  ├─ workload_manifest.json
│  └─ golden_outputs/
│
├─ results/
│  ├─ raw/
│  ├─ results.csv
│  ├─ summary.md
│  └─ BEST_CONFIG.json
│
└─ environment/
   └─ manifest.json
```

---

# 22. Metrics that must be captured

For every benchmark run capture:

## Inference

```text
prompt tokens
generated tokens
prompt-eval time
prompt processing tokens/sec
time to first generated token
generation time
generation tokens/sec
total wall time
```

## Speculative decoding

```text
draft tokens generated
draft tokens accepted
acceptance rate
average accepted draft length
MTP generation time
target verification time
```

llama.cpp already reports speculative-draft statistics including generated and accepted token counts. citeturn931920view4

## GPU 0 and GPU 1

```text
VRAM allocated
peak VRAM
GPU utilisation
memory-controller utilisation if available
power draw
SM clock
memory clock
temperature
PCIe throughput if observable
```

## CPU/system

```text
total RAM
available RAM
llama-server working set
CPU utilisation
per-core utilisation
memory bandwidth if available
page faults
hard page faults
NVMe read throughput
NVMe IOPS
```

## Configuration

```text
Git SHA
model SHA/hash
quant
context
ncmoe
GPU split
batch
ubatch
thread count
KV type
MTP yes/no
MTP type
MTP GPU
MTP n-max
```

Every row in `results.csv` must be reproducible from the recorded parameters.

---

# 23. Benchmark methodology

Each configuration should be:

1. loaded;
2. warmed;
3. benchmarked at least three times;
4. benchmarked using identical prompts and generation settings;
5. tested with deterministic/greedy decoding for correctness runs;
6. separated into prompt-processing and token-generation results.

Discard obviously anomalous warm-up runs.

Report:

```text
median
minimum
maximum
```

for primary timings.

Do not report the fastest single run as the result.

---

# 24. Profiling Stage 0 — establish hardware baseline

Record:

```text
GPU order
PCIe topology
GPU link width/speed
idle VRAM
driver
CUDA
CPU
RAM configuration
RAM speed
storage device
llama.cpp SHA
```

During benchmarking:

- close browsers and GPU-heavy applications;
- disable other local LLM servers;
- prevent display/desktop workloads from dominating either GPU where practical;
- use the same power configuration for every run;
- monitor thermal throttling.

If a GPU progressively clocks down during the profiling sequence, correct the thermal/power problem before comparing results.

---

# 25. Profiling Stage 1 — correctness

Before 250k context, validate at:

```text
8k
32k
```

with:

```text
F16 KV
no MTP
single slot
```

Confirm:

- coherent output;
- no assertions;
- no NaNs;
- no QSA/indexer errors;
- deterministic greedy reproduction is stable enough for regression testing.

Only then proceed to expensive long-context tests.

---

# 26. Profiling Stage 2 — CPU MoE boundary

This is the most important first optimisation.

Start:

```text
-ts 60,40
KV = F16
batch = 2048
ubatch = 1024
MTP = OFF
```

Test approximately:

```text
-ncmoe 40
-ncmoe 36
-ncmoe 32
-ncmoe 28
-ncmoe 24
-ncmoe 20
-ncmoe 16
```

Stop downward exploration when:

- model no longer fits safely;
- long-context allocation fails;
- VRAM margin becomes unacceptable;
- throughput stops improving.

Use a coarse-to-fine search.

For example, if:

```text
32 → 22 t/s
28 → 27 t/s
24 → 31 t/s
20 → OOM
```

then profile:

```text
23
22
21
```

rather than continuing a giant Cartesian search.

---

# 27. Profiling Stage 3 — GPU split

Once an approximate CPU-MoE boundary is known, hold it constant and test:

```text
55,45
60,40
65,35
70,30
```

Then refine around the winner.

Potential final split might be something like:

```text
62,38
64,36
67,33
```

There is no requirement for round numbers.

The output should record:

```text
best decode split
best prefill split
best end-to-end split
```

If they differ significantly, retain this information for later workload-profile selection.

---

# 28. Profiling Stage 4 — CPU threads

CPU expert execution is likely to become memory-bandwidth limited.

Do not assume maximum SMT thread count is optimal.

Profile approximately:

```text
8
12
16
physical-core count
physical + SMT
```

for:

```text
--threads
--threads-batch
```

where appropriate.

Measure CPU-MoE throughput rather than CPU utilisation.

A configuration at:

```text
100% CPU
```

that generates fewer tokens/sec than one at:

```text
60% CPU
```

is worse.

---

# 29. Profiling Stage 5 — batch and ubatch

After placement is approximately tuned, test:

### Batch

```text
1024
2048
4096
```

### Microbatch

```text
256
512
1024
2048
```

subject to memory.

This should be a staged search, not every possible combination.

Prioritise:

```text
prompt-processing speed
TTFT
peak VRAM
```

because generation throughput may barely change.

---

# 30. Profiling Stage 6 — context scaling

Run the winning configuration at:

```text
32k
64k
128k
192k
245k
250k
```

Record:

```text
PP t/s
TG t/s
TTFT
VRAM GPU0
VRAM GPU1
system RAM
disk read traffic
```

The final deployment must demonstrate stable operation close to the requested 250k working context.

A model that benchmarks well at 32k but thrashes or crashes at 250k does not satisfy the task.

---

# 31. Profiling Stage 7 — Q8 KV re-optimisation

Repeat relevant placement tests with:

```text
-ctk q8_0
-ctv q8_0
```

Because this changes available VRAM, repeat:

```text
-ncmoe
GPU split
```

around the new boundary.

The final comparison should look like:

| Profile | KV | CPU-MoE | PP | TG | TTFT | E2E |
|---|---|---:|---:|---:|---:|---:|
| A | F16 | tuned | ... | ... | ... | ... |
| B | Q8 | tuned | ... | ... | ... | ... |

Do not compare KV formats with an untuned identical `-ncmoe`.

---

# 32. Profiling Stage 8 — MTP correctness

Only begin after the target baseline is stable.

Use an F16/BF16 MTP head first.

Run greedy decoding against identical inputs:

```text
target without MTP
target with MTP
```

The target verification process should preserve target-model behaviour.

If implementing or porting MTP code directly, instrument and validate:

```text
router logits
expert IDs
target logits
draft acceptance
```

where practical.

A suspiciously low MTP acceptance rate should be investigated as a correctness problem before being treated as a performance result.

Reference expectation:

```text
F16 MTP acceptance ~0.68
```

has already been demonstrated by a routing-safe Qwen4Exp implementation. citeturn931920view5

Do not make 0.68 a hard requirement for every workload, but persistent acceptance below roughly 40–45% deserves investigation.

---

# 33. Profiling Stage 9 — MTP draft length

Test:

```text
--spec-draft-n-max 1
--spec-draft-n-max 2
--spec-draft-n-max 3
--spec-draft-n-max 4
```

Start with:

```text
3
```

because this is the current llama.cpp default and the configuration used in current Flash-Next MTP experiments. citeturn786983view0turn355201search0

Capture:

```text
acceptance
TG
draft overhead
verification overhead
E2E wall time
```

The highest acceptance rate does not necessarily produce the highest throughput.

---

# 34. Profiling Stage 10 — MTP GPU placement

For each promising MTP configuration profile:

```text
--spec-draft-device CUDA0
```

versus:

```text
--spec-draft-device CUDA1
```

Re-tune target `-ncmoe` as necessary to fit the MTP head.

The profiler must explicitly determine whether:

```text
more target experts on 5090
+
MTP on 4090
```

beats:

```text
MTP on 5090
+
more target expert offload
```

This cannot be determined reliably from FLOPS specifications.

---

# 35. Critical MTP prefill test

There is an active llama.cpp issue showing that, on Qwen3.8-27B and other MTP-capable models, enabling `draft-mtp` under **multi-GPU layer splitting** can dramatically reduce prompt-processing throughput.

In one reported configuration:

```text
~1007 PP t/s → ~555 PP t/s
```

while generation still improved substantially. citeturn553764view0

This is not Flash-Next evidence and should not be assumed to reproduce identically here.

But it is sufficiently relevant that this handover makes PP profiling **mandatory**.

Do not judge MTP using generation tok/s alone.

Measure:

```text
No MTP:
PP
TG
E2E

MTP:
PP
TG
E2E
```

for actual long-context workloads.

---

# 36. Two deployment profiles are acceptable — and may be preferable

Do not force one universal configuration if profiling shows different optima.

It is acceptable for the final result to provide:

## Profile A — Long-context / prefill-heavy

Example:

```text
Q4
250k
Q8/F16 KV
no MTP
maximum PP
```

## Profile B — Agentic / generation-heavy

Example:

```text
Q4
MTP enabled
more CPU MoE
slightly lower PP
substantially higher TG
```

This would be a superior deployment result to selecting a single mediocre compromise.

The implementation should make profile switching simple.

---

# 37. Optional Stage 11 — n-gram speculative decoding

After native MTP is completed, optionally compare llama.cpp's draftless:

```text
ngram-mod
ngram-map-k4v
```

for workloads involving:

- code editing;
- repeated source text;
- summarisation;
- agentic rewrite operations.

llama.cpp specifically identifies these as useful scenarios for n-gram speculation. citeturn502379view0

Do not mix this into the primary MTP investigation until MTP has been independently characterised.

---

# 38. Workload suite

At minimum create these fixed workloads:

### W1 — Short interactive

```text
input: 4k
output: 1k
```

### W2 — Medium coding

```text
input: 32k
output: 2k
```

### W3 — Large repository reasoning

```text
input: 128k
output: 2k
```

### W4 — Near-max context

```text
input: 245k
output: 4k
```

### W5 — Requested 250k case

```text
input: 250k
output: 2k
```

### W6 — Generation-heavy

```text
input: 16k
output: 8k
```

### W7 — Repetitive code editing

Use an actual long source/code workload where n-gram and MTP acceptance can be meaningfully compared.

Use identical tokenised inputs between runs.

---

# 39. Selection criterion

The profiler must not optimise one isolated metric.

Primary deployment objective:

> **Minimum user-visible wall time for representative workloads while retaining stable ~250k context support.**

Evaluate:

```text
TTFT
+
generation time
=
actual request wall time
```

Generation tok/s remains important, but a configuration that increases TG by 20% while doubling a 250k prefill is not automatically superior.

For the final result, separately nominate:

```text
Best PP configuration
Best TG configuration
Best 250k E2E configuration
Best MTP configuration
Best balanced configuration
```

Then select the recommended default.

---

# 40. MTP acceptance criteria

Keep MTP enabled in a production profile only if all of the following hold:

- no correctness regression detected;
- stable long-context operation;
- stable routing;
- useful draft acceptance;
- decode throughput meaningfully improves;
- overall target workload wall time improves.

As an engineering threshold, look for at least approximately:

```text
≥10% meaningful E2E gain
```

before accepting the additional code, memory and operational complexity.

A 2% synthetic benchmark improvement is not enough justification for maintaining a custom MTP branch.

---

# 41. Q8 MTP experimental gate

Only after the F16/BF16 MTP profile is complete:

1. obtain/build Q8 MTP;
2. run identical greedy tests;
3. compare acceptance;
4. compare expert routing if instrumented;
5. compare target output;
6. compare TG;
7. calculate VRAM saved;
8. use saved VRAM to re-tune `-ncmoe`;
9. rerun E2E workloads.

If Q8:

```text
saves ~4 GB
→ allows more target experts onto GPU
→ maintains correct routing
→ maintains acceptance
→ improves total throughput
```

then it can become a candidate.

Otherwise retain F16/BF16.

Do not make model size alone the deciding factor.

---

# 42. Stability test

The winning configurations must undergo extended repeated inference.

Test:

### Test A

```text
245k input
4k output
repeat several times
```

### Test B

```text
32k input
8k generation
```

### Test C

multi-turn context growth toward the limit.

### Test D

server restart / clean load / reproduce the same result.

Monitor:

```text
CUDA errors
QSA/index-cache assertions
VRAM fragmentation
RAM growth
page-cache growth
disk thrashing
thermal throttling
MTP acceptance degradation
generation slowdown over long output
```

---

# 43. Profiling automation requirements

The agent should not manually copy results from terminal output.

Implement scripts which:

1. launch a configuration;
2. wait for health readiness;
3. sample GPU/system telemetry;
4. send the fixed benchmark;
5. parse llama.cpp timings;
6. parse MTP statistics;
7. terminate the server cleanly;
8. write one structured result row;
9. continue to the next configuration.

The profiler should be restartable.

Already-completed valid runs should not need to be repeated after an interruption.

---

# 44. Results schema

At minimum:

```csv
timestamp,
llama_sha,
model_quant,
ctx,
prompt_tokens,
output_tokens,
gpu_split,
ncmoe,
threads,
batch,
ubatch,
kv_k,
kv_v,
mtp,
mtp_precision,
mtp_device,
mtp_nmax,
mtp_acceptance,
pp_tps,
tg_tps,
ttft_ms,
wall_ms,
gpu0_peak_mb,
gpu1_peak_mb,
ram_peak_mb,
disk_read_mb,
status
```

Raw server logs must be retained separately.

---

# 45. BEST_CONFIG.json

Produce a machine-readable final result similar to:

```json
{
  "model": "Qwen3.8-Flash-Next",
  "quant": "UD-Q4_K_XL",
  "llama_cpp_sha": "<PINNED_SHA>",
  "context": 262144,
  "devices": {
    "CUDA0": "RTX 5090",
    "CUDA1": "RTX 4090"
  },
  "split_mode": "layer",
  "tensor_split": "<PROFILED>",
  "n_cpu_moe": "<PROFILED>",
  "kv": "<PROFILED>",
  "batch": "<PROFILED>",
  "ubatch": "<PROFILED>",
  "threads": "<PROFILED>",
  "mtp": {
    "enabled": true,
    "precision": "<PROFILED>",
    "device": "<PROFILED>",
    "n_max": "<PROFILED>"
  }
}
```

Do not fill unknown values with assumptions.

Populate them from the profiling winner.

---

# 46. Final deliverables

The implementation is complete only when all of the following exist:

- [ ] pinned working llama.cpp Git SHA
- [ ] reproducible CUDA build
- [ ] verified Qwen3.8-Flash-Next Q4 model
- [ ] stable non-MTP baseline
- [ ] PLE explicitly host placed
- [ ] no sustained disk-thrashing during normal warmed inference
- [ ] 5090/4090 layer split profiled
- [ ] CPU-MoE boundary profiled
- [ ] CPU thread count profiled
- [ ] batch/ubatch profiled
- [ ] F16 KV benchmarked
- [ ] Q8 KV benchmarked and placement re-tuned
- [ ] 245k/250k context validated
- [ ] profiling harness committed
- [ ] structured results CSV committed
- [ ] raw logs retained
- [ ] F16/BF16 MTP correctness validated
- [ ] MTP draft length profiled
- [ ] MTP device placement profiled
- [ ] MTP prefill penalty measured
- [ ] MTP end-to-end benefit measured
- [ ] Q8 MTP evaluated only after F16/BF16 baseline
- [ ] long-run stability test passed
- [ ] final default recipe documented
- [ ] fallback no-MTP recipe documented
- [ ] `BEST_CONFIG.json` produced

---

# 47. Expected starting hypothesis

The implementation agent should **test**, not assume, the following hypothesis:

```text
Model:
Qwen3.8-Flash-Next UD-Q4_K_XL

Context:
262144

PLE:
CPU / host RAM

GPU:
all non-MoE compute GPU resident

MoE:
hybrid CPU/GPU
-ncmoe likely somewhere around 20–32

GPU split:
roughly 60–70% 5090
roughly 30–40% 4090

Split mode:
layer

KV:
F16 correctness baseline
Q8 likely worth testing for final performance

MTP:
F16/BF16 correctness first
probably single-GPU resident
draft n-max around 3 initially

Slots:
1
```

The final values are expected to differ from these starting values.

That is the purpose of the profiler.

---

# 48. Things the implementation must NOT do

Do not:

```text
- simply use llama.cpp defaults
- simply use -ngl until VRAM fills
- CPU-offload entire layers before testing CPU-MoE offload
- use tensor split while Qwen4Exp tensor mode remains broken/WIP
- assume 50/50 GPU loading is optimal
- optimise using only free-VRAM figures
- depend on constant NVMe page faults for PLE
- enable whole-model mlock without proving enough host-memory headroom
- use MTP without a no-MTP baseline
- judge MTP only from TG tok/s
- assume Q8 MTP is routing-safe
- benchmark only at 8k/32k and extrapolate to 250k
- benchmark only one run
- update llama.cpp halfway through the matrix without restarting the baseline
- publish a recipe without the exact Git SHA
```

---

# 49. Implementation-agent directive

**Implement this deployment as an empirical optimisation project, not a configuration-copying exercise.**

First establish a pinned, reproducible Qwen4Exp llama.cpp build and a stable UD-Q4_K_XL no-MTP baseline.

Then build an automated profiler and determine the optimal:

```text
CPU MoE boundary
GPU layer ratio
CPU thread count
batch
ubatch
KV precision
VRAM reserve
```

for the exact RTX 5090 + RTX 4090 + 128 GB machine.

Validate performance at progressively larger contexts up to the requested ~250k operating point.

Only after the target model is profiled should native Qwen3.8-Flash-Next MTP be integrated.

Use F16/BF16 MTP as the correctness reference. Profile MTP draft length and device placement, and re-optimise target-model expert residency because the MTP head consumes VRAM that would otherwise hold experts.

Do not accept an MTP configuration based only on higher generation tok/s. Measure the prompt-processing penalty and total wall-clock performance on the provided workload suite.

If a single configuration cannot simultaneously maximise long-context prefill and generation throughput, produce two supported deployment profiles:

```text
max-context / prefill profile
generation / MTP profile
```

The final handover must contain measured evidence explaining why each final parameter was selected.

The final outcome is not:

> “Qwen3.8-Flash-Next runs.”

The required outcome is:

> **“This is the fastest reproducible Qwen3.8-Flash-Next configuration we found on this exact hardware under the specified Q4+ and ~250k-context constraints, and these measurements demonstrate why.”**

---

# 50. Current upstream caveat

This implementation is being performed immediately after the Flash-Next release.

As of 27 August 2026:

- Qwen3.8-Flash-Next was released on 26 August;
- Qwen officially lists llama.cpp support;
- the active llama.cpp Qwen4Exp PR remains open;
- tensor split is still WIP in that PR;
- MTP is not yet part of that PR;
- recent commits have fixed QSA, multi-slot, KV and memory-fit issues. citeturn355201view0turn931920view1turn502379view2

Therefore the code-version pin and regression harness are part of the deployment architecture, not administrative extras.

Once upstream stabilises, the implementation should be straightforward to rebase—but only after the benchmark and correctness suite passes again.