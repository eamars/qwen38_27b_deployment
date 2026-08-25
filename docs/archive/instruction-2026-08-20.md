# Qwen3.8-27B Dual-GPU Local Deployment — Top-Level Implementation Instruction

## Objective

Deploy two independent local Qwen3.8-27B inference instances on this Windows host:

- **RTX 5090 32 GB:** capability-first Qwen3.8-27B configuration, approximately 130K context.
- **RTX 4090 24 GB:** highest-capability Qwen3.8-27B configuration that satisfies the same operational constraints; context should be maximized after model capability and VRAM safety are satisfied.

DFlash2 speculative decoding is **mandatory**.

Both models must be usable concurrently and independently. No tensor parallelism is required or desired.

The deployment must be optimized and validated on the **actual local hardware**, rather than assuming published benchmark results transfer directly.

---

# 1. Repository and directory layout

Initialize a local Git repository at the parent/root directory of this deployment.

Use at least:

```text
/
├── .gitignore
├── docs/
├── models/
├── scripts/
├── benchmarks/
└── runtime/          # optional, if locally built runtime artifacts are retained
```

Requirements:

- All documentation, architecture notes, setup instructions, operating instructions, benchmark conclusions, and handover material must live under `docs/`.
- All executable PowerShell/Python/helper scripts must live under `scripts/`.
- All downloaded model files must live under `models/`.
- Raw benchmark data may live under `benchmarks/`.
- Model files must be excluded from Git.
- Large generated binaries, caches, temporary build outputs, benchmark transient files, and other unsuitable artifacts must also be excluded.
- Remove stale, superseded, duplicate, experimental, or abandoned scripts before final handover.
- Do not leave multiple vaguely named versions such as `start2.ps1`, `test-new.ps1`, `final-final.ps1`, etc.
- Keep one clearly documented canonical script for each operational task.
- Record the final deployment state in Git.

If the directory already contains useful material, preserve it and establish a baseline commit before materially changing it.

---

# 2. Host assumptions

The host runs **Windows**.

CUDA, NVIDIA drivers, CMake, MSVC/build tools, and the CUDA-capable build environment are already established.

Do **not** spend time reinstalling or replacing the development environment unless an actual incompatibility is demonstrated.

Before installation, record:

```text
Windows version
CPU
system RAM
GPU 0 name / UUID / PCI bus ID / VRAM
GPU 1 name / UUID / PCI bus ID / VRAM
NVIDIA driver
CUDA runtime/toolkit version
```

Map the RTX 5090 and RTX 4090 explicitly by GPU UUID or PCI identity. Do not rely blindly on an assumed GPU index.

---

# 3. Runtime

`llama.cpp` may be:

- built locally from source; or
- supplied as an appropriate pre-built binary.

Because **DFlash2 is mandatory**, use a runtime revision that actually supports the required Qwen3.8 DFlash2 implementation.

If building from source:

- build natively for Windows;
- enable CUDA;
- build for the relevant GPU architecture;
- record the exact Git commit;
- keep executable and associated libraries from the same build;
- do not mix binaries and DLLs from different revisions.

Pin the accepted runtime revision after profiling.

Do not automatically update the runtime after deployment without rerunning the acceptance suite.

---

# 4. Model storage

All model files must be downloaded into:

```text
models/
```

They must not be checked into Git.

Record for every model:

```text
repository
filename
quantization
file size
SHA256
download date
```

Do the same for the DFlash2 drafter.

---

# 5. Mandatory inference architecture

Run **two separate inference backends**:

```text
RTX 5090 backend
RTX 4090 backend
```

Each backend must:

- load its own complete target model;
- load its own DFlash2 drafter;
- use only its assigned GPU;
- avoid tensor parallelism;
- avoid target-model CPU offload;
- avoid DFlash2 CPU offload;
- keep target KV on GPU;
- use a single inference slot initially.

Do not distribute one model across both GPUs.

---

# 6. API architecture

Prefer one OpenAI-compatible front-door endpoint exposing both models by distinct model names.

For example:

```text
http://127.0.0.1:8080/v1/
```

with models such as:

```text
qwen3.8-27b-5090
qwen3.8-27b-4090
```

Internally, the preferred topology is:

```text
                    OpenAI-compatible endpoint
                              |
                         model router
                         /          \
                        /            \
             RTX 5090 backend     RTX 4090 backend
```

The two GPU backends may listen on separate localhost-only ports.

The router must preserve:

- `/v1/models`
- `/v1/chat/completions`
- streaming responses
- model selection through the `model` field
- normal OpenAI-compatible request/response semantics required by Codex and the DeepSeek harness

Do not introduce a complicated proxy stack merely for aesthetic reasons. If one front-door endpoint proves materially less robust than two direct endpoints, document the reason and preserve the two direct endpoints as an operational fallback.

---

# 7. Target model policy

## RTX 5090

Start with the current capability-first candidate:

```text
Qwen3.8-27B UD-Q6_K_M
context = 131072
target K cache = Q8_0
target V cache = Q8_0
DFlash2 drafter = Q4_K_M
DFlash2 live K/V = F16/F16
parallel = 1
```

The target configuration must retain:

```text
>= 1024 MiB actual free VRAM
```

throughout the worst accepted long-context workload.

If Q6_K_M at 131072 violates the VRAM requirement:

1. reduce context modestly;
2. retest;
3. if substantial context reduction is required, compare against Q6_K at 131072.

After Q6_K_M is stable, `UD-Q6_K_L` may be tested as an optional capability promotion.

Promote Q6_K_L **only** if it independently satisfies every acceptance requirement, including the 1 GiB VRAM reserve.

Do not use startup/free-idle VRAM as the acceptance measurement.

## RTX 4090

Start capability-first with:

```text
Qwen3.8-27B UD-Q4_K_XL
target K cache = Q8_0
target V cache = Q8_0
DFlash2 drafter = Q4_K_M
DFlash2 live K/V = F16/F16
parallel = 1
```

Find the **maximum stable context** that still preserves:

```text
>= 1024 MiB actual free VRAM
```

during the full stress workload.

Research suggests the relevant operating region is approximately 100K–110K, but the local hardware measurement is authoritative.

Also profile a 131072 context candidate using `UD-Q4_K_M` if useful for comparison.

However, do not select the 131K profile for production if it fails the 1 GiB VRAM reserve or exhibits DFlash allocation failures.

Prefer:

```text
better target quant + slightly less context
```

over:

```text
lower-quality target + nominally larger context
```

unless actual application testing proves otherwise.

---

# 8. KV-cache policy

Target KV cache must be:

```text
Q8_0 or higher precision
```

on both GPUs.

Do **not** use Q4/Q5 target KV as a method of fitting a larger target model.

If memory pressure exists, trade:

```text
context
or
target-model quantization
```

before degrading target KV below Q8.

For the DFlash2 live draft cache, use:

```text
F16/F16
```

unless local testing demonstrates a clearly superior supported configuration.

Do not assume Q8 is automatically preferable for the DFlash2 cache; current DFlash implementations have demonstrated cases where quantized draft KV severely damages acceptance.

---

# 9. VRAM requirement

Both GPUs have a hard production requirement:

```text
minimum observed free VRAM >= 1024 MiB
```

This must be measured during the worst accepted workload and includes real Windows/background GPU usage.

Sample VRAM repeatedly during profiling, preferably every approximately 250–500 ms.

Record:

```text
total VRAM
peak used VRAM
minimum free VRAM
idle VRAM
loaded-idle VRAM
prefill peak
generation peak
```

The requirement must be tested while:

- target model is resident;
- DFlash2 is resident;
- a deep-context request is active;
- at least 2048 output tokens are generated.

A model that loads but leaves less than 1 GiB under real workload is **not accepted**.

---

# 10. CPU-offload policy

CPU offload should be avoided and is not an acceptable normal production configuration.

Verify from runtime logs that:

- all target layers are GPU-resident;
- all DFlash2 layers are GPU-resident;
- target KV remains on GPU;
- no unexpected CUDA operation falls back to CPU.

If an apparent high-quality configuration requires target-layer CPU offload, reject it and use the next lower GPU-resident configuration.

Do not hide CPU offload behind automatic fitting.

---

# 11. DFlash2 must be characterized, not merely enabled

For each GPU, measure:

```text
target-only baseline
DFlash2 enabled
```

Where practical, also measure the model's native MTP path as a useful reference.

For DFlash2 record:

```text
drafted tokens
accepted tokens
acceptance ratio
mean accepted tokens per verification step
effective TG
verification-step latency
```

Test at least:

```text
n-max = 4
n-max = 5
n-max = 7
```

using identical prompts.

Select the value producing the best **end-to-end wall-clock performance** on the target workload.

Do not select `n-max` merely because it has the highest draft acceptance.

---

# 12. Mandatory performance characterization

Performance characterization is part of deployment, not optional documentation.

Measure separately:

1. **server/model cold start**
2. **cold/uncached prompt TTFT**
3. **prefix-cache-hit TTFT**
4. **prompt processing throughput**
5. **token generation throughput**
6. **DFlash2 acceptance**
7. **VRAM consumption**
8. **long-context correctness**

Do not combine these into one ambiguous "latency" number.

---

# 13. Cold-start measurement

Measure:

```text
process start -> API ready
```

for each backend independently.

Record:

```text
runtime startup time
model load time
time until health/model endpoint responds
initial loaded-idle VRAM
```

Repeat at least three times and report the median.

Do not include this startup time in normal warm-server TTFT.

---

# 14. Uncached TTFT characterization

TTFT means:

```text
client sends request
        ->
first streamed model token received
```

Measure from the client side.

For the RTX 5090, profile approximately:

```text
2K
10K
32K
64K
100K
120K
```

new uncached prompt tokens.

For the RTX 4090, profile approximately:

```text
2K
10K
32K
64K
90K
maximum accepted production context
```

Use unique prompt content/nonces to ensure tests intended to be cold are not accidentally benefiting from prefix reuse.

Run at least three repetitions per tier and report the median.

Record server-side prompt-processing throughput beside client TTFT.

---

# 15. Prefix-cache-hit TTFT

Agentic coding usage is frequently dominated by a large stable prefix plus a relatively small new suffix.

Explicitly profile this case.

Construct a large retained context and then issue turns with approximately:

```text
+1K new tokens
+4K new tokens
```

Measure client-visible TTFT.

Report:

```text
full nominal context
number of new tokens actually requiring prefill
cache hit/miss status
TTFT
PP throughput
```

Do not report a cache-hit 100K conversation as "100K cold TTFT".

Validate that prefix caching works correctly with Qwen3.8's hybrid attention/Gated-DeltaNet state rather than merely appearing to reuse a textual prefix.

---

# 16. TG characterization

Measure TG separately for at least three workload classes.

## A. Predictable code

Use code generation with significant structural predictability.

Generate at least:

```text
2048 output tokens
```

## B. Real coding-agent workload

Use Codex or the intended coding harness against a representative repository/task.

Generate at least:

```text
1024 output tokens
```

## C. Reasoning / lower-predictability generation

Use a reasoning or free-form task with less token predictability.

Generate at least:

```text
1024 output tokens
```

Run these at shallow and deep context.

Report:

```text
prompt tokens
context position
PP tok/s
TTFT
TG tok/s
wall-clock completion time
DFlash acceptance
minimum free VRAM
```

---

# 17. Initial expected performance envelope

Treat the following as **diagnostic expectations**, not hard pass/fail requirements.

The local host is authoritative.

## RTX 5090 / Q6_K_M + DFlash2

Planning range:

```text
mixed coding TG:             ~80–115 tok/s
deep 100K–120K coding TG:    ~75–105 tok/s
reasoning TG:                ~65–90 tok/s
favorable predictable code:  ~110–160 tok/s
```

Approximate uncached TTFT expectation:

```text
2K:    ~0.7–1.3 s
32K:   ~11–14 s
64K:   ~22–29 s
100K:  ~38–49 s
120K:  ~45–59 s
```

Approximate retained-prefix expectation:

```text
1K new suffix: ~0.5–1.2 s
4K new suffix: ~1.5–3.0 s
```

## RTX 4090 / Q4_K_XL + DFlash2

Planning range:

```text
mixed coding TG:          ~65–85 tok/s
deep-context TG:          ~58–82 tok/s
reasoning TG:             ~50–70 tok/s
favorable short code:     ~80–105 tok/s
```

Approximate uncached TTFT:

```text
2K:    ~0.9–1.6 s
32K:   ~13–18 s
64K:   ~29–37 s
90K:   ~48–62 s
105K:  ~56–72 s
```

Approximate retained-prefix expectation:

```text
1K new suffix: ~0.6–1.5 s
4K new suffix: ~1.8–3.5 s
```

If measured performance is significantly outside these ranges, **profile the cause before simply changing quantization**.

---

# 18. Research-driven bottleneck profiling

Recent Qwen3.8/DFlash2 work demonstrates that runtime implementation can matter as much as GPU class.

If observed performance is materially lower than expected, perform targeted profiling rather than guessing.

## Step 1 — establish target-only baseline

Disable speculation and measure target-only:

```text
TG
PP
VRAM
GPU utilization
memory utilization
```

This determines the cost of one target forward pass.

## Step 2 — measure DFlash2 benefit

Enable DFlash2 and compare:

```text
target-only TG
DFlash2 TG
accepted tokens / verification
draft overhead
verification latency
```

If the target-only baseline is healthy but DFlash2 gain is small, the bottleneck is probably speculative acceptance or verification rather than the target model.

## Step 3 — inspect acceptance

For coding workloads, healthy DFlash2 often derives its speed from producing multiple accepted target tokens per expensive verification pass.

If mean accepted length is unexpectedly low—especially below roughly 2 tokens/verification on predictable coding workloads—investigate:

```text
DFlash2 configuration
n-max
draft model
draft cache precision
context depth
prompt/workload characteristics
```

DFlash2 acceptance can change materially between code, reasoning, and deep free-form contexts.

## Step 4 — profile context-depth degradation

Run identical generation tasks around:

```text
short context
~32K
~64K
~100K
maximum production depth
```

Compare:

```text
target-only TG
DFlash2 TG
acceptance
verification latency
```

Determine whether long-context slowdown comes primarily from:

```text
attention/verification cost
DFlash acceptance decline
KV/cache operations
or target forward-pass cost
```

## Step 5 — inspect GPU utilization

During target-only and DFlash2 decode record:

```text
GPU utilization
memory-controller utilization if available
power
clock
VRAM
```

If useful tools are already installed, deeper profiling may use NVIDIA Nsight Systems/Compute.

Do not install a large profiling stack merely for this task unless required.

Determine whether decode is:

```text
memory-bandwidth bound
compute bound
verification-kernel bound
or suffering from idle/underutilized GPU resources
```

## Step 6 — inspect speculative verification efficiency

Recent high-performance Qwen3.8 work found that multi-token speculative verification can suffer poor GPU utilization even when ordinary one-token decode performs well.

If:

```text
target-only performance is strong
DFlash acceptance is strong
but effective DFlash TG is unexpectedly weak
```

profile the verification path specifically.

Look for:

```text
low SM utilization
attention verification overhead
poor scaling with multi-token verification
unexpected synchronization
kernel-launch overhead
```

Document the finding before attempting runtime modification.

## Step 7 — inspect sampling/logits overhead

Qwen3.8 has a very large vocabulary.

If profiling shows significant time outside model forward/attention execution, determine whether:

```text
logit processing
top-k/top-p sampling
large-vocabulary operations
CPU/GPU synchronization
```

are material contributors.

## Step 8 — verify prefix caching

Recent high-performance agentic setups derive extremely low follow-up TTFT primarily from prefix reuse, not magically faster full prefill.

Verify that:

```text
cold prompt
and
cache-hit follow-up
```

behave as expected.

A cache-hit agent turn should not redo the entire 100K context.

---

# 19. Do not blindly chase published 130+ tok/s figures

Published Qwen3.8 results above 130 tok/s have used combinations such as:

```text
W4A16/Marlin target weights
custom quantized lm_head/embeddings
custom DFlash2 kernels
lookup/context speculative drafting
optimized verification attention
specialized sampling kernels
shorter ~64K fast-context configurations
```

These results prove that runtime architecture has substantial optimization headroom, but they are **not directly comparable** with:

```text
Q6 target
Q8 target KV
~130K context
stock llama.cpp DFlash2
```

Do not lower target quality merely to reproduce a headline benchmark.

Instead, use those results diagnostically:

> If our 5090 is unexpectedly slow despite much stronger hardware, determine whether the limiting factor is runtime implementation rather than immediately blaming the model quant or GPU.

---

# 20. Optional advanced exploration

Only after the production llama.cpp deployment is complete, correct, documented and benchmarked may the agent investigate more aggressive optimizations.

Potential areas include:

```text
lookup/context speculative drafting
more efficient DFlash2 verification
prefix-cache optimizations
sampling/logit-path optimization
alternative DFlash2 runtimes
```

Do not mix experimental optimization work into the production baseline before a clean baseline has been recorded.

Any experimental branch must be benchmarked against the exact same prompts and methodology.

---

# 21. Long-context correctness tests

Performance alone is insufficient.

For each production model, construct a tokenizer-calibrated deep-context test containing unique facts near:

```text
beginning
middle
far end
```

Ask a final task that requires all of them.

Test near the actual production context ceiling.

Record:

```text
context length
retrieval correctness
TTFT
PP
TG
VRAM
DFlash acceptance
```

Do not accept a configuration that is fast but demonstrably degrades long-context retrieval.

---

# 22. DFlash2 correctness check

Compare target-only versus DFlash2 using deterministic decoding:

```text
temperature = 0
top_k = 1
fixed seed
```

Use:

```text
coding prompts
reasoning prompts
structured JSON/tool-call prompts
deep-context prompt
```

Record both outputs.

Unexpected deterministic divergence must be investigated before production sign-off.

---

# 23. Harness validation

Validate both deployed models using:

- direct OpenAI-compatible API calls;
- Codex;
- the DeepSeek harness.

For each harness verify:

```text
model discovery/name
chat completion
streaming
long-running generation
tool/function behavior if used
error handling
cancellation
prefix/cache behavior across realistic agent turns
```

A server benchmark alone is not sufficient. At least one performance characterization must run through the actual intended harness.

---

# 24. Benchmark artifacts

Create a reproducible benchmark output under `benchmarks/`.

At minimum record CSV/JSON data for:

```text
GPU
runtime commit
model
model hash
draft model/hash
context
KV type
draft KV type
DFlash n-max
prompt type
prompt token count
cache state
PP
TTFT
TG
wall time
drafted tokens
accepted tokens
mean acceptance
peak VRAM
minimum free VRAM
result correctness
```

Create a concise Markdown analysis under:

```text
docs/
```

summarizing:

- final selected configuration for each GPU;
- rejected configurations and why;
- performance results;
- VRAM results;
- cold-start results;
- cold TTFT;
- cache-hit TTFT;
- DFlash2 effectiveness;
- long-context behavior;
- any identified runtime bottlenecks.

---

# 25. Canonical scripts

The final `scripts/` folder should contain only useful maintained scripts, for example:

```text
start-all.ps1
stop-all.ps1
start-qwen27b-5090.ps1
start-qwen27b-4090.ps1
profile-vram.ps1
benchmark-ttft.ps1
benchmark-tg.ps1
benchmark-suite.ps1
health-check.ps1
```

Names may differ if there is a better structure.

Scripts must:

- fail clearly on errors;
- use deterministic paths/configuration;
- not contain stale hard-coded experiments;
- log enough information to reproduce the run.

Remove superseded scripts before final handover.

---

# 26. Hard acceptance criteria

A backend is **not production-ready** unless all applicable items pass:

- Target model is loaded entirely on the intended GPU.
- DFlash2 is loaded entirely on the intended GPU.
- No tensor parallelism.
- No normal target CPU offload.
- Target KV is Q8_0 or better.
- DFlash2 is active and verified.
- Actual minimum free VRAM remains **>=1024 MiB** under the stress workload.
- No CUDA OOM.
- No repeated DFlash memory-slot/allocation failures.
- OpenAI-compatible streaming works.
- Codex works.
- DeepSeek harness works.
- Long-context retrieval test passes.
- Deterministic DFlash2 correctness test passes.
- Cold-start performance is measured.
- Cold/uncached TTFT is measured.
- Prefix-cache-hit TTFT is measured.
- PP is measured.
- TG is measured.
- DFlash acceptance is measured.
- Selected `n-max` is justified by local measurements.
- Benchmark results are reproducible.
- Runtime/model versions and hashes are documented.
- Git repository is clean.
- Model files are ignored by Git.
- Stale scripts are removed.
- Final operating instructions are under `docs/`.

---

# 27. Decision principle

When choosing between configurations, optimize in this order:

```text
1. Correctness
2. Full GPU residency
3. >=1 GiB VRAM safety reserve
4. Target KV >= Q8
5. Target-model capability
6. Useful context length
7. DFlash2 effectiveness
8. TTFT / TG performance
```

Do not trade away correctness, GPU residency, VRAM safety, or target KV quality merely to maximize a benchmark number.

The final configuration must be selected from **local empirical evidence on this machine**.
