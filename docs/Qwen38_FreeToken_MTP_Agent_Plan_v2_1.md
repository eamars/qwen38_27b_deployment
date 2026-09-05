# Qwen3.8 Flash Next MTP — Revised Agent Execution Plan

Version: 2.1 — hardware-aware amendment  
Date: 2026-09-05  
Status: hardware specifications/source paths reviewed; local implementation, links under load, and performance NOT verified; full-context feature-stream authorization unresolved  
Baseline: `af71ba43206e124f5ff6419b47ee36c6e9981078`  
Experiment branch: `exp/qwen38-mtp-4090`  
Experiment worktree: `C:/workspace/qwen38_27b/runtime/freetoken-mtp-4090-exp`

This document supersedes **version 2.0** and the original **Qwen3.8 Flash Next MTP Development Plan** for agent execution. It preserves the checkpoint, device roles, offload policy, isolation boundary, default-off behavior, correctness contracts, and G6 review stop. S0 is the original plan; S1 is the user's hardware-bandwidth update. Public hardware sources H1–H9 are indexed in §13. All new execution decisions and arithmetic are proposals/derived estimates, not local measurements.

**Execution boundary introduced by S1:** the user prohibits retaining or transferring the complete approximately 5 GiB, 262144-row feature stream. Version 2.0's full-history MTP initialization would still transfer that many bytes cumulatively, even with bounded chunks. Do not silently interpret chunking as satisfying a prohibition on cumulative traffic. G0–G2 and bounded short-context MTP probes can proceed. Full-context MTP prompt initialization remains `BLOCKED_REQUIREMENT` until the user explicitly permits cumulative chunked transfer or approves and validates another initialization design. Never build an unauthorized full-context sidecar through many small transfers. See §5.3 and §6.7.

## Executive direction

**Prove that the target can profitably verify several tokens with its existing offload machinery before building a speculative-serving framework. Then prove a real, depth-one, two-GPU MTP loop. Expand only after those probes work.**

Required progression:

`local evidence → real offload probe → target-only verification oracle → real MTP alignment → minimal two-GPU loop → measured optimization → acceptance benchmark → approval → serving completion`

Do not treat route overlap, an upstream benchmark, successful imports, unit-test counts, or a high draft-acceptance rate as proof that this design works on this machine.

### Version 2.1 hardware decision

Keep the two-process/two-GPU design. Prioritize **target expert-fetch bytes per emitted token and block-verification latency**, not attempts to make host-to-5090 transfers run at the DDR5 channel rate. Treat decode bridging as a small-message latency problem and prompt preparation as a separate bulk-transfer/compute/contended-memory problem. The hardware figures strengthen this prioritization; they do not establish 48 token/s or a 20% gain.

| New evidence or constraint | Required change from v2.0 |
|---|---|
| DDR5-6000, two channels: 96 GB/s theoretical | Keep channel bandwidth, sustainable CPU bandwidth, and actual offload-gather throughput as separate metrics. |
| 5090 Gen5 x8; 4090 Gen4 x8 expected | Add directional link ceilings and verify generation/width under actual transfer load. The existing installed two-card topology is used for every baseline/off/on comparison. |
| P2P reported unavailable | Retain host staging; no P2P-driver or topology modification work. |
| 26.9 GB/s workload result | Recover benchmark path/units/timing before interpreting it. If it is the current host-to-5090 gather rate, it is already about 85% of that link's encoding-level ceiling. |
| 20–100 KiB decode payloads | Persistent pinned slots and coalesced valid rows from the first real bridge; optimize synchronization before chasing bulk GB/s. |
| Full 5 GiB feature-stream prohibition | Full-history cold initialization conflicts with a literal cumulative-transfer ban; clarify before executing it. Chunking limits peak memory, not aggregate traffic. |
| Shared host memory and WSL | Add process-local pinning validation, VM memory/swap discovery, and loaded contention measurements. |

No hardware change authorizes CPU MoE, a different checkpoint, a quantized drafter, or relaxed performance/correctness gates.

### Material revisions

| Issue in S0 | Revised direction |
|---|---|
| Generic speculative-core port precedes the main execution-path proof | No framework port before the target offload probe. Borrow only the specific helper required by the next runnable experiment. |
| Assumes a new expert-union/cache planner is needed | First exercise the baseline's existing device-side on-demand cache path with multiple rows. Extend verified limitations, not a parallel replacement. |
| One wide hidden seed is treated as the normal round payload | Budget accepted-target-feature catch-up as well. Token acceptance does not establish recursive draft-KV equivalence. |
| Accepted token count and cached token count are underspecified | Use an explicit pending-anchor convention, row-to-logit mapping, and commit table. |
| Correctness gates conflate floating-point agreement and exact output | Exact discrete-state/token checks plus separately declared numerical tolerances; no tolerance may excuse a greedy token mismatch. |
| Route overlap and a sub-1 ms bridge are hard early vetoes | Use a measured whole-cycle budget. These are diagnostics, not independently sufficient success/failure criteria. |
| Final speed gate precedes potentially essential optimization | Separate a feasibility checkpoint from final acceptance; allow only measured, bounded optimization before final acceptance. |
| Prefix support is scheduled after a cache-enabled acceptance benchmark | A minimal synchronized sidecar precedes final cache-enabled acceptance. Full lifecycle hardening remains later. |
| Approximate 40 token/s is the sole relative baseline | Re-measure the protected baseline; require both the absolute 48 token/s floor and a 20% paired median gain. |
| “262K context” does not reserve output space | Freeze exact prompt length, output budget, sequence capacity, and position policy. Never silently truncate or extend the model configuration. |
| Stage 0 is marked complete and effort hours are asserted | Revalidate setup; report evidence and blockers rather than unsupported completion or time estimates. |

## 1. Objective and fixed scope

Build an experimental single-active-request MTP path for the existing local `Qwen3.8-Flash-Next-NVFP4` checkpoint:

- Target execution stays on the RTX 5090; target NVFP4 expert banks remain host-resident with on-demand GPU fetch.
- The checkpoint's MTP component executes on the RTX 4090, with resident BF16 MTP experts and the required shared embedding/head tensors from that same checkpoint.
- The primary target backend remains `offload`, with `moe_cpu_layers=0`. Assert the resolved backend and zero CPU expert execution, rather than trusting the launch arguments alone.
- No alternative model, checkpoint conversion, requantization, target tensor parallelism, or target-expert placement on the 4090.
- Separate target and draft processes, each exposing only its assigned GPU as local `cuda:0`.
- No dependence on P2P, NVLink, CUDA IPC tensors, or a second full target instance.
- Existing baseline source, model files, project launchers, and project configuration remain unchanged.

The experiment is about **user-visible speed at unchanged target behavior**, not fitting a larger model or reporting an isolated draft-kernel speedup.

## 2. Success criteria and evidence levels

### 2.1 Three distinct milestones

**Feasible prototype:** a real two-GPU, depth-one loop works with the existing checkpoint, produces correct greedy tokens, and has a measured path to the speed target. A replay-based diagnostic implementation may reach this milestone, but cannot establish final performance.

**Greedy acceptance candidate:** the no-target-replay path passes numerical/state checks and the full greedy performance gate, including prompt preparation and synchronized prefix-cache cases. This is not yet a completed sampled-serving implementation.

**Completed experimental runtime:** after review approval, sampled decoding, supported API semantics, operational hardening, and final regression reruns pass. Unsupported request features use ordinary target decoding or an explicit unsupported-feature error; never approximate them.

### 2.2 Correctness

1. MTP-off behavior matches the protected baseline on a frozen tokenized corpus.
2. Greedy MTP-on token sequences match ordinary target decoding exactly on that corpus, forced-boundary cases, and long-context validation.
3. Token IDs, positions, lengths, page ownership, n-gram histories, ring cursors, and allocator/refcount invariants are exact. Physical page numbers may differ when allocation order differs; compare the logical mapping and ownership, not incidental addresses.
4. Floating tensors are compared using a predeclared policy established from baseline repeatability and independent references. Report maximum error, relative/RMS error, NaN/Inf checks, and target top-logit margins. Do not invent one blanket BF16 tolerance for every subsystem.
5. A greedy-token mismatch is a failure even when logits are within tolerance. Diagnose numerical ordering, routing, state, or alignment; do not change the corpus or loosen the gate to conceal it.
6. Sampling preserves the target distribution for the explicitly supported processor/sampler stack, subject to measured implementation numerics. Same seed does **not** require the same sampled sequence as the non-speculative implementation, because random-number consumption differs. Independent deterministic sampler-oracle tests are still required. [R8, R9]
7. Every accepted-prefix length `a=0..k`, termination inside a block, and failure boundary leaves the correct subsequent state.

A corpus establishes tested equivalence, not a proof of bitwise identity for every possible prompt. Preserve this distinction in the report.

### 2.3 Performance

Let `B` be the freshly measured median target-only decode throughput for the frozen primary workload.

Final greedy acceptance requires all of:

- Median MTP-on decode throughput at least `max(48 token/s, 1.20 × B)`.
- Matched prefill and TTFT slowdown no more than 3%, with uncertainty reported.
- MTP-off throughput/latency regression no more than 1%, with uncertainty reported.
- Lower end-to-end request latency on the primary workload, not merely a shorter inner decode kernel.
- No correctness failure; no target rollback replay in the measured fast path.
- Measured expert-fetch traffic and a reconciled critical-path profile explaining the result. For the bandwidth-amortization claim, demonstrate lower expert-fetch bytes per emitted token. If a gain instead comes from another mechanism, report it separately; do not silently claim the original bandwidth criterion passed.
- No dropped warmup debt, omitted catch-up, hidden cache rebuild, or selected-only “good” MTP rounds in the headline result.

A result whose noise prevents resolving the 1% or 3% margins is `INCONCLUSIVE`, not `PASS`. Do not replace the margins with looser ones.

### 2.4 Evidence labels

Every important item must be classified as one of:

`SOURCE_REPORTED`, `SOURCE_CHECKED`, `LOCAL_MEASURED`, `DERIVED_ESTIMATE`, `PROPOSED`, `BLOCKED`, `NOT_RUN`.

S0 workload/checkpoint figures and S1 reports of installed/configured hardware remain `SOURCE_REPORTED` until their local evidence is captured. Manufacturer-supported capabilities checked against H1–H9 may be `SOURCE_CHECKED`; arithmetic from those capabilities and the reported topology is `DERIVED_ESTIMATE`. A specification is not a measurement of the configured machine. Preserve separate fields for these provenance levels.

## 3. Non-goals and prohibitions

No model downloads or checkpoint conversion. No new quantization. No CPU expert-compute A/B in the primary experiment. No general multi-request scheduler, speculative tree, generic distributed runtime, or generalized transport library. No unsolicited commit, push, upstream issue, or pull request. Do not merge a donor PR wholesale.

No broad coverage campaign, repository-wide lint cleanup, or speculative refactor before a relevant runtime probe succeeds. Small safety checks and independent correctness oracles are necessary; they are not a substitute for the probe.

Do not suppress tests after the runtime path works. The order is:

`hypothesis → minimal executable probe → evidence → focused hardening/tests → rerun the probe`.

## 4. Isolation, startup, and local discovery

### 4.1 Mutation boundary

| Resource | Policy |
|---|---|
| `runtime/freetoken-a80b4d3` at the full baseline SHA | Read/run only; do not change tracked or untracked source files. |
| `runtime/freetoken-mtp-4090-exp` | All source edits, new scripts, and experiment artifacts. |
| `C:/workspace/qwen38_27b` project outside the experiment | Read existing launch/config/log evidence only. |
| Existing model directory | Read-only; preserve tensor files, configuration, tokenizer, and metadata. |
| Experiment branch | Local only; no commit unless explicitly authorized. |

Revalidate rather than assuming the worktree exists or is clean. Preserve pre-existing changes. Never use `reset --hard`, `clean -fd`, broad process killing, or a forced checkout to repair setup.

Use separate environments, ports, PID records, caches, and logs. Keep dependency versions equal to the baseline initially; no opportunistic upgrades. Direct new Triton, Torch-extension, CUDA, and Python bytecode caches into the experiment. Check the imported `freetoken` path before each run. Force local/offline model resolution and check for automatic weight-export or conversion side effects; no new FTW or other converted checkpoint may be created. Existing weight caches may be reused read-only when they are part of the frozen baseline.

Resolve Windows paths inside the actual WSL distribution with `wslpath`; do not assume `/mnt/c`. Verify real paths and symlinks before writes. No system-wide driver, WSL, power-limit, or CUDA configuration changes without approval.

### 4.2 First commands: read-only discovery

These commands are real discovery commands, not a proposed FreeToken CLI:

```bash
set -euo pipefail
PROJECT="$(wslpath -u 'C:\workspace\qwen38_27b')"
BASE="$PROJECT/runtime/freetoken-a80b4d3"
EXP="$PROJECT/runtime/freetoken-mtp-4090-exp"
EXPECTED=af71ba43206e124f5ff6419b47ee36c6e9981078

test -d "$BASE" && test -d "$EXP"
test "$(git -C "$BASE" rev-parse HEAD)" = "$EXPECTED"
for p in "$PROJECT" "$BASE" "$EXP"; do
  printf '\n=== %s ===\n' "$p"
  git -C "$p" rev-parse --show-toplevel
  git -C "$p" rev-parse HEAD
  git -C "$p" status --short
 done
git -C "$EXP" branch --show-current
nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader
```

If a path, SHA, branch, or existing change contradicts the boundary, record `BLOCKED_SETUP`; do not repair it destructively. An experiment with existing changes may be resumed only after recording and identifying those changes.

Resolve the actual baseline interpreter, resolved launch configuration, model path, and reproducible workload from local project evidence. Never invent them. Read repository `AGENTS.md`/contribution guidance before editing. Fail early with the missing fields when discovery cannot resolve a requirement.

### 4.3 Evidence workspace

Create only inside the verified experiment:

```text
.mtp-exp/
  discovery.json
  execution.json
  commands.jsonl
  status.json
  environment/
  source-audit/
  corpus/
  probes/
  benchmarks/
  reports/
  caches/
```

This layout is proposed. It does not imply the files or runner already exist.

`discovery.json` records paths, SHAs, existing diffs, device UUIDs, model metadata hashes, tokenizer/template identity, dependency versions, backend configuration, baseline command as an argument array, and benchmark availability. Add `hardware.json` with the provenance-separated inventory in §5.4 and `metric-provenance.md` identifying what the historical 26.9 GB/s measured. Missing hardware counters are `null` with an explanation, not invented readings.

Because commits are prohibited without permission, a SHA alone does not identify modified experiment code. Each run must also record a tracked diff hash and hashes of all relevant untracked source files, excluding logs/caches. Save the corresponding patch/source manifest.

Run protected baseline and experiment target processes **sequentially**, not simultaneously. Do not duplicate the large host expert bank to run paired comparisons concurrently. Only terminate processes owned by the experiment supervisor; an occupied GPU/port is a recorded conflict, not permission to kill another service.

## 5. Assumptions that must be measured

### 5.1 Inventory before model construction

S0 reports: RTX 5090 32 GiB, RTX 4090 24 GiB, about 4.86 GiB of BF16 `mtp.*`, about 2.37 GiB for embedding/head duplication, about 0.52 GiB draft KV/index storage at 262K, a 10240-element BF16 wide feature, about 26.9 GB/s gather bandwidth, and about 40 token/s baseline. None was locally re-measured in this review.

Read the local checkpoint index and tensor headers before allocating GPU memory. Produce an exact allowlisted tensor map with shape, dtype, quantization metadata, aliases/tied weights, storage bytes, and checkpoint shard source.

Do not infer that embedding and LM-head tensors are BF16 merely because `mtp.*` is BF16. The pinned target code has an optional quantized LM-head path. Check the actual checkpoint and 4090 kernel compatibility. [R1]

Standard in-memory operations needed to execute the checkpoint's represented weights must be explicit and numerically checked. Do not reconstruct nonexistent original BF16 weights, requantize weights, save a converted checkpoint, or silently substitute a different head. Stop on an unsupported representation rather than guessing.

Do not instantiate the complete target model or deserialize entire expert shards just to load a few shared tensors. Use indexed/selective reads. Confirm peak host RAM, not just final residency.

### 5.2 Memory budget

Record allocated/reserved/peak memory separately on each GPU. Include:

- Weights, target expert cache, KV/index pools, graph pools, attention workspaces, and allocator reserve.
- GDN/PLE rollback boundaries, QSA carry journals, per-row target logits/features, and temporary routing buffers.
- Draft persistent cache, canonical catch-up buffers, recursive scratch, and prefix sidecars.
- Host expert pinning, PLE tables, both processes' staging buffers, shared-memory copies, and transient loader copies.

Report how MTP reduces any target expert-cache capacity. Do not reduce the protected baseline's cache to make the comparison easier. An equal-cache-budget diagnostic is separate from the real deployment comparison.

Define transient headroom from measured peak allocations, attention/graph workspace growth, and allocator behavior before increasing depth. An initial reserve of `max(1 GiB, 5% of usable device memory)` is a conservative planning suggestion, not a universal gate. Record and justify the actual reserve; do not silently shrink target context or expert-cache capacity to recover from OOM.

For `k=4`, rollback storage may require the pre-round boundary plus five verification boundaries, not four. Allocate from discovered tensor shapes and dtypes, reuse buffers across depths, and include capture-pool duplication.

### 5.3 Bridge volume, serialization, and the full-prompt constraint

At the reported width, one BF16 wide feature is `10240 × 2 = 20,480 bytes` (20 KiB). Verify width/dtype from the checkpoint. Under §6.4's nonterminal canonical-catch-up convention, a round with `a` accepted candidates normally transfers `a+1` target feature rows and emits `a+1` new tokens. Excluding bootstrap/termination effects and redundant copies, this is approximately **20 KiB of logical forward feature payload per emitted token**, not 20 KiB multiplied by draft depth for every output token.

At 48 token/s this is 0.98304 MB/s; at 100 token/s it is 2.048 MB/s. These are derived traffic estimates for the greedy feature path, not measured bandwidth or throughput. Token/control replies and future sampled-q payloads must be counted separately. An implementation that exports every verify row before deciding acceptance transfers more bytes; record valid, discarded, and retransmitted row counts instead of claiming the minimal ratio.

Probe 20 KiB, 40 KiB, 60 KiB, and 100 KiB messages: `k=1,2,4` use maximum block widths `2,3,5`. Coalesce contiguous rows needed for one decision into one payload; do not send five tiny messages just because k=4. Do not delay a needed round waiting to batch independent future rounds that do not yet exist.

Using the S1 topology's encoding-level rates, without transaction/software overhead:

```text
B_5090 = 32e9 transfers/s × 8 lanes × 128/130 ÷ 8 = 31.507692... GB/s
B_4090 = 16e9 transfers/s × 8 lanes × 128/130 ÷ 8 = 15.753846... GB/s

Whole-payload store-and-forward serialization:
  t_serial(S) = S/B_5090 + S/B_4090
  20 KiB: 1.95 microseconds
  100 KiB: 9.75 microseconds

Asymptotic one-way pipelined logical feature rate:
  R <= min(B_5090, B_4090) = 15.753846... GB/s
```

These are different models: a payload that is completely staged before forwarding pays both leg times; a stream of independent chunks can overlap legs, approaching the slower-leg rate plus pipeline fill/drain. Neither is the observed two-process latency. Include launch, scheduling, CPU copies, notifications, and GPU completion. The 15.75 GB/s is a **one-direction** logical ceiling; it is not a combined full-duplex aggregate and is not a direct-P2P claim. Reverse control traffic is generally asymmetric.

**Full-prompt volume:** `262144 × 20480 = 5,368,709,120 bytes = 5 GiB`. For §9.2's actual 261120-token input, the feature total is 4.98046875 GiB. Full-history initialization in v2.0 consumes these target feature rows to build the draft's canonical cache. Streaming them avoids a 5 GiB allocation but does not eliminate their cumulative transfer. Sparse/indexed attention alone is not evidence that prompt rows can be skipped.

The ideal link-only time for the 5 GiB reference case is approximately 0.341 s for a deeply pipelined stream, or 0.511 s when both complete legs serialize. Actual initialization also includes target production timing, CPU copies, contended expert fetch, MTP prefill/index work, and synchronization. Neither number is a TTFT forecast. Overlap may hide some or much of it, but cannot be assumed.

**S1 policy, pending clarification:**

```text
full_context_feature_stream_authorized = false
full_context_initialization_status = BLOCKED_REQUIREMENT
short_context_feature_probe_max_rows = 4096  # proposed safe probe scope, not a model limit
```

The agent may transfer bounded short-context traces to prove alignment; it may not export or accumulate the complete near-full-context feature stream. An already valid paired sidecar may be examined/reused with provenance, but must not be created through an unauthorized warmup. A warm-only or fallback result cannot pass the full cold/warm acceptance gate.

The user must resolve whether the restriction bans (1) monolithic allocation/transfer only, allowing cumulative chunked traffic, or (2) the cumulative full-history traffic itself. Under (1), enable bounded streaming only after that clarification. Under (2), the current cold full-history initialization is out of scope: report the blocker, preserve target-only behavior, and seek approval for a different independently validated initialization approach. Do not silently truncate history, recompute the full target on the 4090, or use lossy feature compression.

A possible separate design investigation under a cumulative-volume ban is a **bounded-history MTP proposer**, initialized from a limited suffix of verified target features while the target still uses its full context. This would deliberately change the draft-history contract, not the target context. For greedy verification, arbitrary proposal tokens can still be checked against authoritative target choices; full-history draft-cache equivalence is not itself the proof of final target correctness. However, correct truncated-history positions/indexing, checkpoint compatibility, acceptance, and speed are unverified here. Such a design needs a separately declared bounded-history reference and explicit approval before replacing §6.4's canonical full-history contract. Do not present it as an already working escape from the 5 GiB constraint or silently apply it in the primary benchmark.

### 5.4 Hardware inventory and evidence

| Item | Planning value | Evidence classification / local work |
|---|---|---|
| CPU | Ryzen 9 9950X3D, 16 cores / 32 threads | CPU identity is S1-reported; capabilities checked in H1. |
| Host memory | 128 GiB, 2 × 64 GiB, DDR5-6000 | S1-reported; record installed capacity, configured rate, channels, BIOS version, and VM-visible memory separately. |
| CPU rated memory speed | DDR5-5600 for two 1R or 2R DIMMs | H1 `SOURCE_CHECKED`. DDR5-6000 is above that rated specification; do not assume EXPO specifically rather than manual tuning unless observed. |
| Theoretical memory-channel data rate | 96 GB/s = 89.40697 GiB/s at DDR5-6000 | `DERIVED_ESTIMATE`; a channel ceiling, not sustained CPU or GPU-gather performance. |
| Rated-rate channel ceiling | 89.6 GB/s at DDR5-5600 | `DERIVED_ESTIMATE`, not this machine's current operating rate. |
| Board | Gigabyte B850 AI TOP rev. 1.0 | S1-reported identity; H2 specifies CPU-connected x16/x8 slots and x8/x8 allocation when both are populated. |
| 5090 link | Expected Gen5 x8, 31.507692 GB/s/direction | H2/H3 plus S1 slot configuration; verify live generation and width under transfer load. |
| 4090 link | Expected Gen4 x8, 15.753846 GB/s/direction | H2/H4 plus S1 slot configuration; the Gen5-capable slot does not make the 4090 a Gen5 device. |
| Idle link | x8, Gen1 reported | S1 observation; not evidence of an under-load Gen1 fault. Record idle and loaded states separately. |
| Peer access | NS / unavailable reported, topology label PXB | S1-reported. Record actual API capability results in both directions when available. No dependency on peer access. |
| Historical gather | 26.9 GB/s | `SOURCE_REPORTED` until exact benchmark, byte convention, path, units, context, cache state, and timing are recovered. |
| Historical target speed | About 40 token/s | `SOURCE_REPORTED`; use the fresh protected baseline for decisions. |

PCIe figures exclude packet/request/completion overhead. Do not call theoretical rates achieved DMA throughput. The `.max` capability reported for a device and its current negotiated loaded link are distinct. A PXB label does not establish host-memory throughput, a shared fixed-rate uplink, or CUDA peer support. Preserve the raw topology; do not infer a different physical motherboard wiring from a virtualized label alone. [H2–H6]

Read-only discovery uses the locally available command syntax:

```bash
nvidia-smi --help-query-gpu
nvidia-smi -L
nvidia-smi -q -x
nvidia-smi topo -h
# When supported by the local driver/environment:
nvidia-smi topo -m
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
cat /proc/meminfo
cat /proc/swaps
```

Capture generation/width both idle and during a bounded, active transfer probe. Query only fields supported by the installed driver; host-side Windows read-only telemetry may be needed where WSL hides fields. Record unsupported queries without installing/changing drivers. A small disposable discovery utility may enumerate both GPUs to query `cudaDeviceCanAccessPeer()`/`cuDeviceCanAccessPeer()` in both directions; the target and draft runtime processes remain isolated to one GPU each. Do not enable peer access or change IOMMU/ACS/BIOS/power settings. [H5–H7]

WSL has its own configured memory budget, and CUDA-on-WSL documents limited pinned-memory availability. Installed 128 GiB does not establish usable/pinnable memory for the two processes. Record VM capacity, current available RAM, swap activity, existing expert pinning, registration failures, and transient loader peaks. Read `.wslconfig` when available, but do not edit it or restart WSL. [H7, H8]

### 5.5 Separate the bandwidth experiments

Do not collapse all measurements into a single `memory_bandwidth` number.

| Experiment | Measures | Required controls |
|---|---|---|
| CPU streaming read/copy | Sustainable CPU memory access for the stated access pattern | Compiled/native loop or existing validated tool; working set exceeds cache; checksum; declared read/write byte accounting; record threads, faults, and available memory. |
| CPU indexed gather | CPU access locality and host-copy cost | Actual representative index/block distribution; distinguish CPU-to-CPU gather from GPU-originated reads. Not a CPU-MoE implementation. |
| Pinned H2D/D2H legs | Each GPU's transfer latency/bulk rate | Persistent buffers, direction, payload size, completion-based timing, under-load link state. |
| Actual expert-fetch kernel | The baseline offload data path plus route/cache behavior | Existing quantized banks, true misses, all bank/scales/alignment bytes, selected kernel, cold/warm capacity. |
| Two-process bridge | Consumer-ready latency and completed logical throughput | Actual socket/shared-memory path, registered/pinned behavior, copies, ACKs, valid row counts, and buffer lifetime. |
| Loaded combined run | End-to-end interference | Target expert fetch alone versus the same work plus bridge; target prefill plus permitted bounded prompt transfer; record both producer and consumer slowdowns. |

A quick clean CPU-memory probe is diagnostic, not permission to build a CPU-MoE backend or delay the decisive real offload-layer experiment with an exhaustive memory survey. Use small bounded allocations sized to observed free memory, not a second full expert bank. No swapping/page-fault storm is allowed to masquerade as bandwidth performance. Any stress or external-tool installation outside existing permissions needs approval.

The pinned FreeToken cache source includes GPU kernels that dereference registered host-bank device aliases and dispatch missing-row copies. Therefore an operation named “gather” can already include PCIe traffic rather than measure CPU DRAM alone. Confirm the local resolved path before attaching the historical 26.9 figure to it. [R3, H9]

Conditional interpretation only: if 26.9 GB/s measures sustained uncached bytes across the **current host-to-5090 path**, it is `26.9 / 31.507692 = 85.38%` of the encoding-level limit. Even the physically optimistic bandwidth-only improvement is at most `31.507692 / 26.9 = 1.1713×`, before protocol overhead. This is not a measured utilization or an overall speed forecast. It cannot be applied to a CPU-only gather result or a differently configured run.

### 5.6 Shared-memory traffic and contention model

S1's read/write accounting is a useful first-order model, not a measured DRAM-counter result:

```text
logical host traffic per bridge payload byte ≈ 2 + 2*c
c = additional full host copies
isolated channel-ceiling model: 96/(2+2*c) GB/s
c=0: 48; c=1: 24; c=2: 16; c=3: 12
```

Real DRAM traffic can differ because CPU caches, write allocation, coherence, read/write turnarounds, and driver staging affect the memory bus. In particular, a 20 KiB CPU copy can be cache-served; do not assert that every logical load/store becomes a DRAM transaction. At large streaming sizes the model is useful for copy amplification. With target expert traffic active, the full 96 GB/s is not available to the bridge. Use measured workload-matched sustainable bandwidth and account for other readers/writers, rather than dividing the nominal ceiling and declaring guaranteed spare capacity.

Track these separately: expert host-read bytes; 5090 H2D expert bytes; 5090 D2H features; 4090 H2D features; reverse reply bytes; CPU/socket copy bytes; pinned/registered resident bytes. A direct GPU read of a registered expert bank and a CPU gather-then-copy pipeline have different host traffic factors.

The 5090 expert-input direction and its feature-output direction are opposite, so do not add them as if they share one half-duplex PCIe budget. They still may contend in host memory, the root/I/O path, GPU memory/SM resources, or copy scheduling. A GPU gather kernel is not necessarily an independent copy-engine operation. Establish overlap and interference from the actual loaded trace, not GPU count or stream names.

## 6. Proposed architecture and contracts

### 6.1 Keep two processes; use a bounded, persistent bridge

Target process: 5090 only, authoritative target state and user-visible output.  
Draft process: 4090 only, allowlisted MTP/shared tensors, canonical draft cache plus discardable recursive scratch.

Use a length-prefixed Unix-domain socket inside WSL for control. Before the first real model round, allocate persistent process-local pinned staging and a bounded two-slot ownership ring. Coalesce the round's valid target feature rows into one contiguous payload. The simplest initial data plane may also use the socket, but it must reuse its staging rather than allocate/pin/serialize arbitrary tensor objects every round. Record actual host copies.

Select a shared-memory data plane only when the loaded profile or prompt-bandwidth budget justifies it. First test whether the same shared pages can be registered and used safely by each process's local CUDA context; absence of P2P does not answer that host-registration question. Use local pinned staging plus explicit measured host copies when shared registration fails. Do not assume `multiprocessing.shared_memory` is CUDA-pinned or that CUDA IPC device tensors are the bridge. Do not implement three transports before the first model round. [H5, H7, H9]

Proposed initial capacities: two decode slots, each sized for the largest supported valid block (100 KiB at k=4 and the reported feature width), plus explicit metadata/reply storage. For permitted prompt probes, start with two row-aligned chunks of 256 or 1024 rows (5 or 20 MiB per slot), subject to measured memory headroom. Count every process-local host/GPU copy of each slot in the budget. These are starting points for measurement, not mandatory optimal sizes. No full-context streaming is authorized by these buffer examples.

Use process-local pinned staging unless registration of shared pages has been explicitly tested on both processes. “Shared memory” does not mean “pinned memory.” Track slot ownership:

`FREE → PRODUCER_COPY → HOST_READY → CONSUMER_COPY → REUSABLE`

Publish readiness only after the producing GPU copy completes. Reuse a shared slot only after the consuming H2D finishes; GPU destination scratch has its own lifetime until its compute consumers finish. Use request/round/generation/slot-sequence checks to prevent stale ACKs from releasing a reused slot. No per-round pin/unpin, unbounded queue, busy allocation, or device-wide synchronization in the fast path. Local CUDA events and explicit notification order are required; more streams do not remove data dependencies.

Measure p50/p95/p99 consumer-ready latency, CPU scheduling/queueing, and target slowdown with the bridge active. Idle GB/s alone does not choose the transport. Full-duplex link capability does not imply that same-round drafting, target verification, and canonical catch-up can all run concurrently. Overlap only independent work—such as prompt chunk stages or bookkeeping after state ownership is safe. [H6, H9]

Do not pass CUDA tensors through `multiprocessing.Queue`. Do not assume GPU UUID isolation eliminates all backend-global allocation assumptions: assert device ownership after actual loading and execution.

### 6.2 Target verification: reuse first, decouple second

The pinned target's offload layer already distinguishes on-demand loading from whole-layer prefill and includes a device-side cache-remap/copy path. Probe this existing path with multiple token rows before adding a planner. [R2, R3]

Introduce a narrow forward intent or equivalent typed metadata:

```text
PROMPT: prompt prefill; ordinary prefill loading policy
DECODE: ordinary one-token continuation
VERIFY: one request, k+1 causal rows, on-demand experts, all row logits/features,
        speculative mutable state, no premature scheduler advancement
```

Use separate properties for attention continuation, expert-loading policy, output selection, and speculative ownership. Audit every affected phase-dependent call site, including metadata builders and PLE prefetch, not only `Batch.is_prefill`.

For each verification layer:

1. Produce router decisions for all valid rows. Preserve raw expert IDs before in-place cache-slot remapping.
2. Reuse/extend the existing unique-active-expert and cache-remap machinery; validate duplicate IDs, cross-row reuse, and capacity limits.
3. Fetch missing experts once per required cache plan when feasible. Keep slots live until all consuming kernels complete.
4. Execute the actual NVFP4 backend selected by the baseline, with supported multi-row behavior.
5. Assert no whole-layer materialization and no CPU expert computation.

When the active union cannot fit available slots, use a proven bounded subplan or reduce depth before launching. Do not evict an expert still referenced by a live row. A tiled subplan must be reported; “one plan per layer” is not a reason to make allocation unsafe.

**One target verification forward means one layer-stack traversal, not one CUDA kernel.** Short row loops inside a GDN/PLE layer are permitted initially to preserve canonical recurrence order while MoE remains blockwise. A loop over complete target-model forwards is diagnostic sequential verification and cannot establish final MTP speed.

Do not automatically call the generic GDN prefill/chunk kernel. Test whether its arithmetic/state semantics preserve the required output against ordinary decode; use an ordered per-row recurrence within a single layer traversal when needed. [R4]

QSA must enforce per-row causality for both attention and compressed/index selection. A later speculative row must not make a partially built compressed block visible to an earlier row. Add a “change only future candidates” test: earlier logits must remain unchanged under the declared numerical policy.

### 6.3 Canonical target frontier: pending anchor

Use this convention unless a demonstrably equivalent adapter is documented against the baseline engine:

- `f` = count of tokens already processed into authoritative target state.
- `u` = one committed token not yet processed by the target; its absolute position is `f`.
- The logical token sequence has length `f+1`.
- `h_prev` = target pre-final-mixer wide feature from the last processed token, at `f-1`.
- `k` = number of draft candidates, excluding `u` and excluding the target correction/bonus.

The first generated target token after prompt prefill becomes `u`. Prompt initialization, empty-prefix/BOS behavior, and the first available `h_prev` require their own alignment fixture.

Draft candidates are `d1..dk`. The target executes:

| Verify row | Input | Absolute position | Logits used for |
|---:|---|---:|---|
| 0 | `u` | `f` | validating `d1` |
| 1 | `d1` | `f+1` | validating `d2`, or replacement after `d1` |
| … | … | … | … |
| k | `dk` | `f+k` | bonus token when every candidate is accepted |

Let `a` be the contiguous accepted-candidate count. Emit `d1..da` followed by target correction/bonus `z`, subject to termination truncation. In a normal nonterminal round:

```text
new target processed length = f + a + 1
committed verify rows       = 0 .. a
new pending anchor          = z at position f + a + 1
new h_prev                  = wide target feature of verify row a
new emitted token count     = a + 1
```

The correction/bonus is emitted but has **not** been target-processed. Do not fabricate its hidden state or increment the cache frontier an extra time.

Depth-two examples:

| a | Newly emitted | Target rows retained | Next pending token |
|---:|---|---|---|
| 0 | `z` | `u` | `z` |
| 1 | `d1, z` | `u, d1` | `z` |
| 2 | `d1, d2, z` | `u, d1, d2` | `z` |

If termination truncates the planned output to `e` newly committed tokens, retain `e` verification-input rows: old anchor plus the first `e-1` newly emitted candidates. The newest retained output token remains pending under this convention. For `e=0`, commit no new target rows and preserve the old pending anchor (or terminate and safely release the request). The terminal-cache adapter must match the baseline's finish behavior. Parser-visible bytes and token-level accounting are related but not interchangeable.

### 6.4 Draft math and canonical catch-up

The inspected Qwen4Exp references distinguish a mixed-width representation for the LM head from a wide representation for recursive drafting. The pinned target currently collapses its wide stream before returning LM-head input. Export the correct pre-mixer feature explicitly; do not feed final logits/head input back as the wide state. [R1, R5]

Before fixing daemon messages, write `alignment-contract.md` from the local checkpoint and independent reference. It must specify normalization, weight orientation, layer configuration, token shift, absolute position, rotary/index behavior, bootstrap rows, both output shapes, and cache effects.

The provisional alignment to test is:

```text
(target wide feature at position t, token at position t+1)
    -> MTP input row at position t+1
    -> proposal distribution for position t+2
```

Test this mapping; do not treat it as a substitute for checking the checkpoint/reference convention.

**Token acceptance is not a proof that recursive MTP KV is target-aligned.** A later recursive draft row can use a draft-generated wide feature instead of the corresponding target-generated feature. The conservative proposed implementation is:

1. Keep canonical MTP history separate from discardable recursive rows.
2. The first MTP input for `u` uses `h_prev`. Retain that canonical row when the round commits.
3. Discard recursively conditioned rows after that anchor unless their equivalence has independently been proved.
4. After target verification, rebuild rows for accepted tokens `d1..da` from their verified preceding target features. Do this only on the **draft** GPU.
5. Store the target feature of row `a` to condition the next pending anchor `z`.

This generally transfers target feature rows `0..a`—`a+1` wide vectors—not just one. Catch-up may be combined with the next draft's first causal pass over `d1..da,z`, but its entire cost belongs to the measured round/request.

For `a=0`, retain the canonical old-anchor row and use its target output feature for the new anchor. For all-accepted `a=k`, the final accepted token may never have been consumed as an input during recursive drafting; its canonical MTP row still needs construction.

Compare the resulting draft cache and next-step distributions with a fresh teacher-forced MTP pass over the same committed token sequence and verified target features. A shorter reuse scheme is an optimization, not the default correctness assumption. The upstream MTP work's separate accepted-update machinery is a useful reference, not proof that an abbreviated local protocol is correct. [R6]

### 6.5 Transaction ownership

Create a state inventory with owner, dtype/shape, mutation sites, restore rule, and lifetime:

| State | Required treatment |
|---|---|
| GDN recurrent state | Capture every possible committed row boundary using the declared recurrence semantics. |
| GDN convolution | Capture the matching raw-history window; not just the final recurrent tensor. |
| PLE convolution and n-gram context | Capture each boundary, including short prefixes, sentinel/BOS handling, and reset/boundary tokens. |
| QSA KV and compressed/index state | Speculative tail ownership plus journaled overwrite-prone carry/ring state and per-row visibility. |
| Page tables, allocations, COW/refcounts | Release rejected suffix only after in-flight work drains; never mutate shared prefix pages in place. |
| Target features and logits | Row-addressed scratch; retain only the committed seed/features needed for catch-up. |
| Draft canonical and recursive state | Discard recursive suffix; canonical catch-up and frontier checked independently. |
| Token buffers, scheduler lengths, parser/usage state | Apply only the accepted/terminated output prefix, exactly once. |
| Sampler state and RNG | Record deterministic stream/position policy; no processor side effects from rejected suffixes. |

Do not include the entire immutable expert bank in a transaction. Expert-cache contents/LRU history need not roll back for model correctness when weights and mappings are correct; rejected work's cache effects must remain in performance measurements. Semantic state and performance cache state are different.

Maintain two implementations:

- **Reference oracle:** independent snapshot/restore and ordinary target replay, used outside performance timing. At long context, serialize reference runs or use an explicitly immutable/COW prefix instead of allocating a second full target. Do not share the fast commit-selection implementation with the oracle.
- **Fast transaction:** select captured boundaries and restore bounded overwritten regions, with no target replay. Capturing all boundaries must not require serializing the full state to host every round.

A merged llama.cpp fix specifically addressed incomplete convolution snapshots alongside recurrent rollback; copying only SSM state is insufficient. Treat this as a required test category, not a transferable implementation. [R7]

### 6.6 Minimal protocol and failure handling

Messages carry protocol version, daemon-generation nonce, request ID, round ID, source/destination frontier, valid row count, dtype/shape, payload length, operation ID, and status. Validate sizes before allocation. Use fixed maximum capacities from the manifest.

Operations:

`HELLO/INIT`, `PREFILL_CHUNK`, `DRAFT`, `COMMIT_CATCHUP`, `ABORT_ROUND`, `RESET`, `SHUTDOWN`.

`COMMIT_CATCHUP` can be fused into a following `DRAFT` only after the unfused contract passes. Acknowledgement means required GPU work has completed or is protected by an explicit ownership/event rule—not merely that bytes were received.

State machine:

```text
READY -> DRAFTING -> DRAFT_READY -> VERIFYING -> DECIDED
      -> TARGET_COMMITTED -> DRAFT_SYNCHRONIZED -> READY
```

The target is authoritative. Before sending externally visible output, ensure target commit and output accounting are internally consistent. A failed draft after a valid target commit may be disabled for the remainder of the request; do not undo already published tokens. Before target commit, restore the pre-round transaction before fallback.

Use request/round IDs to deduplicate retryable control messages. Never retry a state mutation blindly after an ACK timeout. An ambiguous draft frontier triggers draft invalidation, not an assumed successful commit. Stale replies and messages from a previous daemon generation are discarded.

Fallback policy:

- Missing compatible prefix, unsupported sampler/parser feature, failed draft initialization, or unsupported depth: choose target-only before drafting.
- Draft failure with verified target state: disable MTP and continue target-only; invalidate affected sidecars.
- Unknown/corrupt target state or an unrecoverable CUDA error: controlled request/process failure, not “best-effort” continuation.
- Never free pages/buffers while an in-flight kernel or copy still references them. Cancellation prevents further publication immediately; reclamation occurs after work drains.

### 6.7 Prefill and prefix cache

**Authorization check first:** §5.3 blocks full-context cold MTP feature streaming pending clarification of S1. The following streaming design is permitted for bounded short probes and is conditional for full-context execution; it does not override that block.

When authorized, cold prompt processing streams bounded target feature/token chunks to build canonical MTP state. Total traffic remains approximately prompt_rows × feature_bytes. The 4090's initialization compute and cache/index work must be measured independently from transfer. Do not block first-token publication merely to hide a slow drafter; time-to-second-token, outstanding catch-up, decode throughput, and full request latency must include the consequences of any lag.

Record target feature production frontier, draft-consumed frontier, queue high-water mark, ring occupancy, and draft-ready time. A finite ring with a slower consumer requires backpressure or disabling MTP; it cannot both preserve every row and remain forever nonblocking. Before its bounded capacity is exceeded, abort draft initialization and preserve target-only generation, or apply an explicitly measured backpressure policy. No spilling an entire feature stream to RAM/disk, silent row loss, hidden prefix replay, or unlimited queued CUDA work. Label this fallback as fallback, not successful MTP performance.

Profile permitted prompt streaming under real target-prefill expert transfers. The 3% TTFT gate remains, and first-to-second-token/full-request timing prevents moving initialization debt just outside TTFT. A measured overlap gain is required before claiming the link-only time is hidden.

Before final acceptance, provide a **minimal synchronized prefix sidecar**, not a general new radix-cache implementation. Attach it to actual resumable target handles/checkpoints. Respect target GDN/PLE checkpoint granularity; do not promise arbitrary-prefix reuse when the target cannot resume there.

A sidecar identifies exact token prefix, target handle/generation, checkpoint and tokenizer identity, position/rotary configuration, draft layout/version, draft canonical frontier, index/carry state, and required boundary seed. Publish it only after both sides reach the declared boundary. Hash keys require a token-identity/collision check.

For a partial prefix hit, restore the deepest compatible **paired** boundary without degrading the target's existing reuse. When no paired boundary can preserve the existing target hit, use target-only for that request. Never rerun the full target prompt simply to enable MTP.

Full/partial hit, sidecar miss, eviction, daemon restart, and COW behavior must be tested. An MTP fallback run is a fallback result, not evidence that warm-prefix MTP is fast.

No performance acceptance run may disable the baseline's prefix cache. Synthetic cache-disabled runs are diagnostics only.

### 6.8 CUDA graphs

Start eager. Reuse the baseline's device-side dynamic offload machinery where compatible; do not move routing/planning to the CPU as an accidental architectural regression. [R2, R3]

After profiling, capture the draft step first or the actual measured bottleneck. Target verification graphs may include an already capture-safe device-side cache path if multi-row safety is proved. When an implementation requires host decisions between layers, capture compatible islands or retain eager orchestration; moving every dynamic copy “outside the graph” is not, by itself, a complete whole-model capture design.

Test graph replay with varying routes, accepted lengths, page/ring crossings, and stale-buffer sentinels. Padded rows must not update state, counters, routing, or caches. Include graph-pool memory in the resource gate.

### 6.9 Exact sampling contract, after greedy acceptance approval

Define `p_i` as the target distribution after the **complete supported target processor stack** at the candidate prefix. Define `q_i` as the actual normalized distribution used to sample draft candidate `d_i` at that prefix. Preserve the actual `q_i`; do not reconstruct it later from a different hidden state.

Use contiguous-prefix rejection sampling:

```text
accept d_i with probability min(1, p_i(d_i) / q_i(d_i))
at the first rejection:
    sample z from normalize(max(p_i - q_i, 0))
when all k candidates survive:
    sample z from target p_(k+1)
```

A sampled proposal has positive `q_i(d_i)`. Invalid/nonfinite probabilities, a zero-mass filtered distribution, or an impossible residual normalizer are errors to diagnose, not permission to substitute an arbitrary sampler. The algorithmic basis is R8; numerical qualifications are R9.

Exact sparse transport is allowed when `q_i` genuinely has finite truncated support: transmit all nonzero support IDs and their **normalized** probabilities. Outside that support, `q_i=0`; the target still needs its complete applicable `p_i` to form the residual. Candidate probability alone is insufficient.

Read the checkpoint/request defaults; S0's `top_k=20` is not locally verified. When top-k is disabled, use an exact supported dense path or target-only. Never silently substitute top-20, infer q from raw logits after different filters, or use greedy matching for sampled target requests.

Temperature, top-k/top-p order, repetition/frequency/presence penalties, EOS masks, logit bias, vocabulary restrictions, minimum length, and structured/grammar constraints must have explicit support status. Per-position processors see the committed prefix plus that row's preceding candidates, never the whole speculative suffix. Stateful processors must be cloned/snapshotted or disable MTP for that request.

Maintain separate deterministic RNG streams for proposal, acceptance, and target replacement/bonus, with a documented request/round/position mapping. Test the sampler independently with tiny probability vectors and chosen random draws, then run empirical distribution tests with predetermined sample counts and acceptance criteria.

## 7. Upstream work: selective references, not dependencies to merge

The public code inspected supports retaining S0's general direction, but does not certify this local combination.

| Reference | Permitted reuse | Boundary |
|---|---|---|
| Baseline model/core/offload code [R1–R4, R11] | Actual local integration seams and existing cache/data-movement primitives | Confirm identical local files and runtime backend before editing. |
| FreeToken DFlash PR #258, S0-reviewed head `9f0a136bdbf2b1f25066dc21ce2fa770b42da78e` [R10] | Small acceptance, multi-token accounting, graph-shape, and test ideas | Different draft architecture. Review the engine's actual offload verification fallback before porting; a sequential full-target fallback is not the required fast path. |
| FreeToken DSpark PR #69, S0-reviewed head `2cf938b7bba362e577aaa4a059188948d1692bcd` [R12] | Frontier ownership, carry journaling, exact-sampler concepts | Its model/TP/hybrid CPU execution must not enter this target-only offload experiment. |
| Qwen4Exp MTP implementation [R5], ik_llama PR #2369 [R6] | Numerical wiring, wide/mixed feature separation, accepted-update alignment | Source inspection only; no alternate model or GGUF conversion. |
| llama.cpp rollback fix #28123 [R7] | Regression categories for coupled recurrent/convolution state | Not a drop-in FreeToken state implementation. |

PR #258 and #69 were publicly open when checked. Do not assume their merge-conflict status or branch heads from S0. Freeze each actually used source blob/SHA and retain license/attribution for copied code. The vLLM documentation URL is moving; record the exact source revision or downloaded source hash before relying on it.

Do not load a full second framework solely to obtain an oracle. A small independent equation-level implementation over the existing local tensors is preferred. Cross-check its equations against the references so two implementations do not merely share the same wiring mistake.

## 8. Development stages: deliverables, probes, gates

Every stage ends with a machine-readable decision and a brief report. Dependencies are strict unless the stage explicitly allows an earlier read-only task. Do not mark a stage complete from a code review, import, or test count alone.

### Stage 0 — Revalidate isolation and freeze the experiment

**Question:** What exactly is running, and what comparison would be valid?

Actions:

1. Execute §4 discovery; inspect existing instructions, interpreter, launch configuration, and actual model metadata. Add §5.4's hardware/WSL inventory, expected-versus-loaded links, and the unresolved full-context initialization requirement. Baseline target-only prefill is allowed; do not enable full-context feature export.
2. Record raw/tied/quantized tensor inventory and a device/host memory budget. Assert no target expert bank will load in the draft process.
3. Record kernel backends, KV/index dtype, prompt chunk sizes, prefix-cache mode, offload-cache size, and graph settings.
4. Recover or construct a frozen benchmark manifest with tokenizer/template hashes and exact input IDs. Preserve the existing meaningful code workload. Synthetic development prompts must be labeled synthetic, not presented as the user's established benchmark.
5. Run protected baseline repeatability at a small context, then the near-full-context primary shape. Do not run a second target alongside it.
6. Recover the original 26.9 GB/s benchmark evidence: CPU-only or host-to-GPU path, exact byte numerator, GB/GiB units, elapsed-time boundaries, cache conditions, link state, and source revision. Missing evidence stays `SOURCE_REPORTED`; do not make it a hard local bandwidth assumption.
7. Save environment and baseline raw results. Verify no protected source/launcher/model changes.

**Required artifacts:** `discovery.json`, `execution.json`, `hardware.json`, `metric-provenance.md`, `source-audit.md`, `checkpoint-map.json`, `memory-budget.json`, corpus manifest, baseline run records. Mark full-context MTP initialization `BLOCKED_REQUIREMENT` without pretending that target-only baseline discovery is blocked by it.

**Gate G0:** exact source/model/device/configuration identity established; baseline loads and generates correctly; benchmark definition is valid. If the historical 40 token/s cannot be reproduced, use the newly measured baseline and explain the difference. Do not fabricate missing historical evidence.

### Stage 1 — Probe existing offload economics and the bridge

**Question:** Does the existing target machinery provide credible room for block verification?

Implement only the minimal probe adapter and counters necessary for:

1. Capturing a bounded real baseline trace: per-token/per-layer raw expert IDs, routes, missing IDs, bank bytes, target latency, and cache capacity.
2. Replaying global-cache access order for ordinary token-major execution versus proposed layer-major block execution at `k=1,2,4`. Include the pending anchor: block widths are `2,3,5` rows.
3. Running representative real NVFP4 offload layers at those row counts using captured real inputs/routes and the existing device-side path. Compare outputs against per-row execution and measure gather plus expert compute, not just set overlap.
4. Separating CPU-memory diagnostics, individual GPU transfer legs, the actual expert-gather kernel, and real two-process handoff (§5.5). Use persistent pinned slots and coalesced 20/40/60/100 KiB payloads from the beginning. Compare target expert-fetch work alone and with the exact bridge path active; measure both bridge latency and target slowdown. Short row-aligned prompt-chunk probes are allowed; the complete full-context feature stream is not. Begin with socket control/data and test shared registration only when a measured limitation warrants it.
5. Estimating full-cycle latency using §9.1, including measured contention and the hardware-aware byte-reduction model. Keep observed acceptance separate from hypothetical acceptance until MTP actually runs. Do not use 96 GB/s as achievable expert-fetch bandwidth or the 1.95 microsecond wire floor as real message latency.

The cache simulator must use the real global cache order/capacity and allocation rules. Summing independent per-layer hit rates can be wrong when execution order changes. Include rejected-suffix cache pollution once real draft traces exist. Route traces along a known correct continuation are an optimistic scenario, not the routes of rejected candidates.

Use real quantized layers; a mock kernel or fabricated bandwidth number cannot pass G1. Confirm whether repeated rows actually share expert transfers, and whether the selected expert kernel repeats HBM work per row.

**Required artifacts:** route trace, validated cache replay, layer output comparisons, layer timings, separately named CPU/PCIe/gather/bridge measurements, loaded target-interference results, host-copy accounting, and the initial cycle-budget table. Unsupported counters remain explicitly unavailable.

**Gate G1:** existing multi-row offload behavior is correct, or a narrowly identified change makes it correct; measured costs leave a credible within-scope route to the target. A bridge above 1 ms or weak route overlap is not an automatic veto. Conversely, an idle bridge below 1 ms is not a pass.

If the current mechanism is clearly uneconomic and no specific measured improvement is identified, stop with `FAIL_ECONOMICS`. Do not build a framework to compensate for a failed premise.

### Stage 2 — Target-only oracle block verification

**Question:** Can one target traversal correctly and efficiently verify a short continuation?

Actions:

1. Add the narrow verification intent and feature/logit export, with ordinary decode unchanged when disabled.
2. Use known target continuations as candidate inputs, not an MTP model yet. Clearly label these **oracle candidates**.
3. Build correct row metadata, causal QSA behavior, ordered GDN/PLE updates, and a slow independent transaction oracle.
4. Compare per-row target logits/features and final logical state to ordinary sequential target execution from the same prefix.
5. Exercise all target boundaries at depth one before expanding to depths two/four.
6. Measure eager target block latency and no-replay ideal acceptance scenarios; keep snapshot/replay work separate and explicitly labeled diagnostic overhead.

An oracle supplies perfect candidates to measure target opportunity; it does not establish attainable acceptance or end-to-end MTP speed. A high measured eager latency is not a mathematical impossibility proof. A profile demonstrating a removable launch bottleneck may justify one narrow capture experiment; otherwise record the limiting component before continuing.

**Required artifacts:** `frontier-contract.md`, row/logit alignment fixture, state inventory, target block probe, sequential-vs-block comparisons, measured target-only budget.

**Gate G2:** one layer-stack traversal, no whole-layer expert transfer, no CPU expert compute, correct row causality, exact discrete frontier state, target greedy choices identical on the probe corpus, and a credible cycle budget. No donor framework or server integration is required to pass this gate.

### Stage 3 — Standalone MTP and the first real two-GPU loop

**Question:** Does the actual checkpoint draft correctly on the 4090, with correct target/draft alignment?

Actions:

1. Implement a selective tensor loader and minimal BF16 resident MTP runner. Freeze the shape/quantization manifest before allocation.
2. Add an independent equation-level MTP reference for small inputs and short contexts. Use a transparent causal attention/index reference rather than the same optimized kernels for every component.
3. Validate the target feature tap, entry fusion, both outputs, prompt token shift, recursive steps, and initial/absolute positions.
4. Run teacher-forced MTP on a real target trace, then recursive `k=1` drafting. Record actual first-step accuracy and latency. Extend to `k=2,4` only after depth one is correct.
5. Implement a minimal independently launched draft process, persistent pinned two-slot bridge, coalesced row payloads, and sequence/generation checks. Use bounded short-context prompt streaming within §5.3's probe scope. Do not construct the full-context draft state while its authorization remains unresolved.
6. Connect a **diagnostic** greedy round using the slow target transaction oracle. Complete multiple consecutive rounds; force `a=0` and `a=1` rather than accepting only favorable drafts.
7. Validate canonical draft catch-up against a fresh teacher-forced cache, including the all-accepted final input row.
8. Measure draft compute, canonical catch-up, bridge, target verify, and initialization lag at authorized contexts. Predicting full-context performance from short-context numbers does not satisfy long-context validation; label the latter blocked/not run until a permitted full-history or independently validated alternate initialization exists.

**Required artifacts:** `alignment-contract.md`, loader inventory, reference comparison, 4090 peak-memory report, real two-GPU token trace, actual acceptance and cycle timings.

**Gate G3:** real two-process generation works; model math/alignment is validated; no target context exists on the 4090; no draft context exists on the 5090; updated measured economics remain credible. No speed claim is accepted while target replay is enabled.

Failure localization:

- Incorrect first proposal: inspect feature tap, tensor layout, normalization, token shift, position/indexing, and loaded head representation.
- Correct first proposal but later divergence: inspect recursive wide output and draft-state alignment.
- Correct local draft but divergent remote loop: inspect protocol frontiers, staging ownership, and catch-up.

### Stage 4 — Fast transaction and a no-replay depth-one loop

**Question:** Can rejected speculative work be discarded without rerunning the target?

Actions:

1. Implement boundary capture/selection for every GDN/PLE mutable state and QSA overwrite-prone region from the inventory.
2. Keep slow replay independent; compare the fast path against it at every `a` and every relevant boundary.
3. Implement depth-one target/draft commits using §6.3 and §6.4.
4. Verify subsequent ordinary decode, not only the immediately following token.
5. Add transaction-failure injection and draft-loss fallback before/after target commit.
6. Run a real no-replay, two-GPU generation smoke benchmark with complete timing.

**Required artifacts:** no-replay evidence/counter, forced-rejection results, long-continuation comparisons, memory lifetime checks, depth-one end-to-end profile.

**Gate G4:** no target replay; no unjournaled semantic state; identical greedy continuation; bounded memory; controlled failure/fallback; measured speed feasibility.

If state correctness fails, stop performance tuning. Debug profiles may identify the fault, but are never performance evidence.

### Stage 5 — Expand only what the profile justifies

**Question:** Can the feasible loop meet the original deployment requirements?

Actions, in this order:

1. Add `k=2`, then `k=4`, with matching forced-boundary tests. Do not assume larger k is faster.
2. Add a narrow default-off engine/scheduler adapter for the existing greedy streaming endpoint. Preserve one active request. Enforce capacity, termination, cancellation, and exactly-once token accounting before timing that endpoint.
3. Add the minimal synchronized prefix sidecar from §6.7, including full/partial hits and target-hit/draft-miss fallback.
4. Resolve §5.3's full-context initialization requirement before primary-context MTP work. When authorized, profile complete cold-prompt and paired-prefix requests, including cumulative feature volume and draft initialization. Otherwise report `BLOCKED_REQUIREMENT`; preserve short-context results but do not stage a hidden full-prefix transfer or substitute a warm-only/fallback result.
5. Select at most two material optimization hypotheses from that profile for this feasibility pass. For each, record the bottleneck, expected gain, changed component, isolated A/B, and post-change correctness probe. Candidate changes include draft graph capture, better prompt overlap, measured transport improvements, or target verify capture.
6. Validate before retaining each optimization. Do not add adaptive depth until fixed-depth results are understood.

If adaptive depth is added, choose from `{0,1,2,4}` using measured cycle cost, acceptance by position, cache behavior, and catch-up cost. Apply hysteresis and include exploration/fallback overhead in the request result. Never truncate q support based on confidence without reflecting the actual proposal distribution.

**Required artifacts:** depth sweep, minimal server smoke results, prefix-sidecar checks, optimization A/B records, frozen candidate configuration.

**Gate G5:** correct no-replay candidate, bounded memory, supported cache behavior, and reproducible near-full-context run. If still short of the goal, report which tested mechanism failed; do not expand into new models, CPU expert execution, or unrelated refactoring.

### Stage 6 — Final greedy acceptance benchmark and mandatory review stop

Run the locked protocol in §9. Compare protected baseline, experiment MTP-off, and experiment MTP-on. Use production-equivalent prefix caching, sampling mode for the **greedy** milestone, context capacity, output lengths, and instrumentation settings.

**Required artifacts:** complete raw results, source/environment hashes, token comparisons, latency distributions, memory and traffic data, run exclusions, and acceptance decision.

**Gate G6:** all §2.3 requirements pass and full-context initialization obeys the resolved user constraint. Record one of `PASS_GREEDY`, `FAIL_CORRECTNESS`, `FAIL_PERFORMANCE`, `INCONCLUSIVE`, `BLOCKED_REQUIREMENT`, or `BLOCKED_ENVIRONMENT`. No cold full-context MTP run means no complete G6 pass.

**Stop for user review regardless of result.** A passing greedy result does not authorize the remaining stages, committing, modifying launchers, or publishing.

### Stage 7 — After explicit approval: exact sampling and supported server semantics

Actions:

1. Implement §6.9 and independent sampler tests before enabling sampled MTP requests.
2. Build a request-feature capability table from the actual server. Unsupported combinations fall back before drafting.
3. Validate the already supported OpenAI-compatible endpoint and any existing Anthropic/Responses adapters that expose the new path. Do not add a new API family merely because S0 named it.
4. Cover reasoning/tool-call parsing, grammar state, stop strings across tokens/blocks, Unicode/UTF-8, usage, finish reasons, and disconnects.
5. Use target logprobs for public outputs. A rejection-sampler residual or draft q is not the target logprob promised by the API.
6. Rebenchmark default sampled workloads separately. Never present the greedy speed gate as the measured speed of sampled serving.

**Gate G7:** exact implemented sampling algorithm, supported request semantics, robust fallback, and passing distribution/streaming tests. Regressions in the greedy/off paths block completion.

### Stage 8 — After approval: hardening and reproducibility

Actions:

1. Harden paired-prefix ownership, eviction, stale handles, long multi-turn reuse, and daemon lifecycle.
2. Run long-context soak, repeated rejection, cancellation, OOM/transport failure, restart, near-limit, and leak tests.
3. Run relevant subsystem tests, then the existing non-slow suite and targeted static analysis. Record unrelated baseline failures separately; do not “fix” them by altering protected code or suppressing tests.
4. Re-run critical runtime probes and the final benchmark after hardening. An earlier speed result does not certify later code.
5. Document exact launch commands, supported features, fallback reasons, memory budgets, rollback state, protocol transitions, and all retained optimizations.
6. Verify source/launcher/model isolation again.

**Gate G8:** reproducible experimental runtime, still default-off and still local-only. No upstream action is implied.

## 9. Economics and benchmark specification

### 9.1 Whole-cycle model

For draft depth `k`, let `A` be accepted-candidate count and `L=A+1` emitted tokens in a nonterminal round. Without assuming independent acceptance:

```text
E[L] = 1 + sum(j=1..k) P(A >= j)
```

Measure the full cycle, including canonical draft catch-up:

```text
T_cycle = host/bridge critical-path work
        + draft proposal work
        + target block verification
        + acceptance/sampling
        + target commit
        + draft canonical catch-up
        + output/accounting work
```

This is a conceptual dependency decomposition, not permission to sum overlapping event durations twice. Use the actual critical path and wall clock. If catch-up is charged to the next round, normalize over complete runs so it is counted once.

For ordinary target cost `t0=1/B` seconds/token:

```text
projected speedup = t0 * E[L] / E[T_cycle]
required cost/token <= min(1/48, t0/1.20)
```

At the source-reported `B=40`, the required cost is at most about `20.83 ms/emitted token`. For illustration only, at `k=1` and acceptance probability `0.8`, `E[L]=1.8`, so the whole cycle must average at most `37.5 ms` to reach 48 token/s. This is an arithmetic example, not an acceptance/latency prediction.

Teacher-forced route overlap and perfect-candidate timings are upper-opportunity diagnostics. Incorporate actual rejected draft routes, target cache pollution, per-depth accepted-length distributions, and catch-up cost before making the feasibility decision.

### 9.1.1 Hardware-aware opportunity model

For the unchanged target offload path, bytes moved per emitted token matter more than the installed DIMM rate. If §5.5 confirms 26.9 GB/s as the current host-to-5090 gather rate, the maximum bandwidth-only gain to an impossible overhead-free 31.507692 GB/s is about 17.13%. Even assuming every millisecond of the 40 token/s baseline scaled with that gain, the resulting ceiling would be only about 46.85 token/s, below 48. Any unchanged compute/synchronization makes the bandwidth-only gain smaller. This is a conditional upper-bound illustration, not a prediction and not applicable to a CPU-only gather measurement.

Do not conclude that MTP cannot reach 48: block reuse changes required bytes and amortizes other work, which a bandwidth-only optimization does not. Conversely, the theoretical 96 GB/s channel rate does not establish a 3.57× expert-fetch opportunity.

A simple sensitivity model, to be replaced by measured critical-path costs:

```text
t0 = measured baseline milliseconds/token
f  = exposed expert-fetch fraction of t0 (not a sum of overlapped profiler durations)
r  = new expert-fetch bytes/emitted-token divided by baseline, at the same effective fetch rate
h  = net new non-fetch overhead milliseconds/emitted-token

t_new ≈ t0 * ((1-f) + f*r) + h

At t0 = 25 ms and target 48 token/s:
  r <= 1 - (1/6 + h/25)/f
```

Illustration only, with h = 2 ms/token and other target work unchanged:

| Baseline exposed fetch fraction f | Maximum remaining byte ratio r | Minimum byte reduction |
|---:|---:|---:|
| 0.50 | 0.5067 | 49.3% |
| 0.70 | 0.6476 | 35.2% |
| 0.90 | 0.7259 | 27.4% |

Neither f nor h has been measured here. Actual verification may also change compute cost and effective fetch rate, so retain the complete §9.1 model. The table demonstrates why a nominal 20% throughput gain can require substantially more than 16.7% expert-byte reduction after draft/catch-up costs. There is no new fixed percentage gate; use measured byte counts and timings.

For every fixed k, record `E[A]`, `E[L]`, `E[T_cycle]`, `E[expert_bytes]/E[L]`, and all feature/copy bytes per emitted token. Use actual candidate routes, not only teacher-forced perfect routes. At B=40 the whole-cycle budget is `E[L] × 20.8333 ms`; changing bridge implementation must be justified by how much of this budget it recovers.

### 9.2 Define “262K” precisely

Freeze `sequence_capacity`, `prompt_tokens`, `max_new_tokens`, actual chat-template/special-token count, and the position interval measured. Read the real model/engine limit; do not change RoPE or increase a limit silently.

Default primary definition when the configured total capacity is 262144:

```text
sequence_capacity = 262144
max_new_tokens    = 1024
prompt_tokens     = 261120, after template/special tokens
```

This is a **262144-capacity, near-full-context** test, not a 262144-input-token test. If the recovered historical baseline used a different definition, preserve that as a separately named run and explain the mismatch. A genuine 262144-input-token test needs independently supported additional capacity for its output.

Before each round, constrain k by remaining output budget, valid verification-input capacity, available page/state scratch, and supported graph shape. With one output token remaining, use ordinary target decoding rather than forcing a speculative bonus past the limit. Padded rows must remain nonsemantic.

Use exact lower context points `4096`, `32768`, and `131072` in diagnostics, reserving output capacity there as well. Report exact lengths rather than mixing 131K and 128K labels.

### 9.3 Benchmark matrix without an unnecessary full cross product

**Tuning set:** a small frozen code/prose/structured subset; short and primary context; depths 0/1/2/4. Use this only to select a fixed candidate or adaptive policy.

**Final primary set:** held-out frozen code prompts at the primary context, one active request, fixed greedy configuration, protected baseline vs experiment-off vs chosen experiment-on configuration, at least five measured paired repeats per primary condition after warmup.

**Required secondary checks:** other context points, prose and structured output, depth-specific correctness, sampled serving after G7, and below cache conditions. Do not run every combination before the first useful feasibility decision.

Separate:

| Condition | Meaning |
|---|---|
| Prefix miss | Neither model has reusable semantic prompt state. Caching remains enabled. |
| Paired full/partial hit | Both target and MTP restore a valid matching prefix state. |
| Target hit / draft sidecar miss | Target-only fallback; preserve target reuse. Not an MTP speed sample. |
| Expert cold | Expert cache starts in the declared controlled state. |
| Expert warm/steady | Expert cache follows the same declared warmup workload. |

Prefix state and expert-weight cache state are different. Do not claim a complete cold/warm matrix if a state cannot actually be controlled; record the limitation and measured initial state.

Include prefix construction cost in cold-request TTFT/end-to-end numbers. A warmed sidecar created by an untimed earlier request must be labeled warm. Where usefulness depends on reuse, report first request plus subsequent requests as a combined workload as well.

### 9.4 Timing and fairness

- Warm each environment/capture shape consistently; compilation and model load are excluded from steady-state decode but reported separately.
- Pair/reorder runs to reduce thermal and temporal bias. Record GPU clocks, power/thermal conditions, CPU load, and competing processes. Do not change hardware power settings without approval.
- Baseline/off/on all use the same two physically installed GPUs and expected x8/x8 lane allocation. An inactive 4090 does not imply that the 5090 becomes x16. Any separately authorized single-card/x16 experiment is a different hardware comparison, not the protected software baseline. Record loaded links and memory configuration for each run.
- Record initialization/bridge bytes, staging high-water marks, CPU-copy amplification, pin/register failures, and actual VM memory/swap conditions. A fast copy-only microbenchmark is not a substitute for consumer-ready latency or contended end-to-end speed.
- Use the same target checkpoint, tokenizer, target KV precision, output budget, and serving route. Record any unavoidable target cache reduction from MTP allocations.
- Measure client-observed request start, first token, last token, and finish separately from engine timings.
- Define decode throughput as `(N-1)/(t_last_token-t_first_token)` for `N>=2`, with tokens counted consistently. Also record total generation and full request latency; do not inflate the denominator/numerator by counting rejected proposals or by discarding early slow rounds.
- Streaming emits bursts. Record chunk-arrival gaps and tokens/chunk, plus engine commit timing. Do not assign invented equally spaced timestamps to tokens in one chunk.
- Use per-device CUDA events only for intervals on that device; use a common host monotonic clock for cross-process wall time. Do not subtract unrelated GPU clocks.
- Avoid `.item()`/host synchronization in hot-path telemetry. Read aggregate device counters after measured windows. Detailed profiling is a separate run with its overhead measured.
- Use actual emitted lengths for early EOS. Predeclare how short runs are classified; do not exclude slow/short cases after inspecting results. Fixed-length synthetic timing that suppresses EOS is explicitly diagnostic, not the only semantic benchmark.

Report per-prompt results and paired ratios, not only a favorable pooled token/s. Use a predeclared paired bootstrap or comparable method. Suggested decision policy: primary median meets the absolute/20% gates and the 95% paired speedup interval excludes slowdown; equivalence intervals must resolve the MTP-off 1% and TTFT 3% margins. Increase repeats under a predeclared rule if noisy; otherwise return `INCONCLUSIVE`.

### 9.5 Required counters and metric provenance

For each run record:

```text
identity: run_id, mode, source SHA + diff/untracked hashes, exact argv/environment,
          GPU UUIDs, checkpoint/tokenizer/template identity, dependencies
workload: prompt/output counts, capacity, positions, sampler/processor settings,
          prefix condition, initial expert-cache state, selected depth/policy
results: output token hash, TTFT, prefill duration/rate, decode throughput,
         client chunk gaps, full request time, accepted-length histogram
costs:   target verify, proposal, canonical catch-up, bridge/copies/queueing,
         commit, sampling/processors, output, unresolved critical-path time
traffic: raw routes, unique experts, misses, bank bytes, target replay count,
         whole-layer materializations, CPU expert executions
memory:  per-process/device peaks, host/pinned peaks, expert-cache capacity,
          live pages/sidecars/staging slots, WSL available memory and swap state
hardware: configured DDR rate/channels and source, idle/loaded PCIe gen/width,
          per-direction capability and observed rates, P2P results or null
bridge:  forward/reverse logical bytes, valid/discarded/retransmitted rows,
          host-copy bytes, CPU/DRAM byte provenance, pin/register count,
          slot/queue high-water marks, consumer-ready p50/p95/p99 latency
init:    full-context stream authorization, cumulative prompt feature bytes,
          target-produced/draft-consumed frontiers, draft-ready time, fallback
fallback: count, reason, request fraction, tokens before/after fallback
```

Distinguish measured hardware DRAM/PCIe counters, logical bytes inferred from actual bank-copy operations, and estimates from route counts. They are not interchangeable. Some existing aggregate fields may be backend-specific; validate counters against known transfers before using them. Include quantization scales, padding/alignment, and all bank components.

When WSL hides hardware bandwidth counters, store `null` plus an explanation. Use validated logical transfer counts and timings as evidence of the mechanism; do not label them measured physical DRAM traffic. If the original literal hardware-counter requirement cannot be established, report that part as unverified rather than manufacturing a pass.

Use a proposed 5% unresolved wall-time budget as a profiling quality check. Exceeding it requires further attribution before claiming the bottleneck is understood. Overlapping work must be reconciled, not forced to add up by double-counting.

## 10. Test strategy and executable handoff contract

### 10.1 Independent oracles

| Property | Oracle |
|---|---|
| MTP equations/wiring | Small direct implementation over the local weights, cross-checked against source references. |
| Route union/cache mapping | CPU simulator reproducing actual cache order and capacity; real-layer comparison. |
| Target row logits | Ordinary sequential target execution from an equivalent prefix. |
| Fast state commit | Independently captured pre-round state plus ordinary prefix replay. |
| Canonical draft cache | Fresh teacher-forced MTP history with verified target feature/token pairs. |
| Exact sampler | Explicit small probability vectors and fixed random draws. |
| Prefix restoration | Fresh prefill of the same exact token sequence/configuration. |
| Server semantics | Existing target-only endpoint behavior for supported features. |

Do not test a function solely against another function that uses its internal helper or the same incorrectly indexed tensor. Preserve regression fixtures for every discovered mismatch.

### 10.2 Escalating validation

**During probes:** depth one; short real sequences; all acceptance outcomes; one page/ring and PLE boundary; multiple consecutive rounds; subsequent target-only continuation. Use enough generation to expose state corruption, not a single successful token.

**Before G6:** depths 1/2/4; all `a=0..k`; long-context cases; at least 4096 subsequent ordinary tokens for forced-boundary continuation checks where capacity allows. Near-capacity tests must reserve that continuation space or be separately identified shorter terminal-boundary tests. Never exceed the model limit to satisfy a test length.

**Before G8:** sampled and greedy, repeated transitions, prefix reuse, restarts, cancellation, memory/resource lifetime, and supported parser/API matrices.

Mandatory adversarial cases:

- Page/index-compression/ring boundaries at `boundary-1`, `boundary`, and `boundary+1`; more than one row touching the same mutable ring region.
- PLE convolution tail, n-gram wrap/reset/sentinel, short prefix, and multiple PLE state owners if configured.
- Mutating a future draft candidate must not change an earlier target row's output.
- Zero/all acceptance; EOS, stop, and length limit at every output position.
- Rejected rows whose experts pollute the weight cache; semantic output remains correct.
- Capacity-limited union, padded graph rows, varying routes at fixed graph shape, and stale scratch-buffer sentinels.
- Draft timeout/crash before DRAFT response, after decision, during catch-up, after ACK, and after daemon generation change.
- Cancellation while target kernels/copies are in flight; no premature resource reuse.
- Paired prefix full/partial hit, stale/missing sidecar, shared-page COW, eviction, and terminal frontier.
- Unsupported sampler/grammar/API option routes to a safe target-only path.

### 10.3 Minimal runner specification — to implement, not pre-existing

Create one small experiment runner, for example `benchmarks/qwen38_mtp_probe.py`. Reuse baseline benchmark helpers where sound. The name below is a proposed interface, not a claim that these commands work before the runner is implemented:

```bash
# AFTER implementing the runner and validating execution.json:
python benchmarks/qwen38_mtp_probe.py inventory --manifest .mtp-exp/execution.json
python benchmarks/qwen38_mtp_probe.py offload --manifest .mtp-exp/execution.json
python benchmarks/qwen38_mtp_probe.py bridge --manifest .mtp-exp/execution.json
python benchmarks/qwen38_mtp_probe.py verify --manifest .mtp-exp/execution.json --depth 1
python benchmarks/qwen38_mtp_probe.py draft --manifest .mtp-exp/execution.json --depth 1
python benchmarks/qwen38_mtp_probe.py loop --manifest .mtp-exp/execution.json --depth 1
python benchmarks/qwen38_mtp_probe.py acceptance --manifest .mtp-exp/execution.json
```

Runner requirements: argparse/help, typed validated configuration, subprocess argument arrays rather than interpolated shell commands, no hidden downloads, source-path/device checks, bounded timeout and owned-process cleanup, JSONL metrics, preserved raw stdout/stderr, explicit `NOT_RUN` on unsupported stages, and nonzero exit on errors. Do not implement fake successful subcommands or replace a hardware probe with a mock silently.

Do not invent FreeToken CLI flags from this plan. Resolve existing flags from the pinned CLI and document new flags only after implementing them. Initially use internal adapters in the probe; expose an experimental server flag only at Stage 5.

Suggested exit meanings: `0=completed/pass`, `2=invalid manifest/setup`, `3=correctness failure`, `4=failed performance gate`, `5=inconclusive`, `6=blocked environment`. Exact codes may follow repository conventions if recorded consistently.

### 10.4 Stage record

Maintain a compact record after every stage and before stopping:

```json
{
  "plan_version": "2.1",
  "stage": "G0",
  "decision": "NOT_RUN",
  "source_identity": null,
  "commands": [],
  "evidence_paths": [],
  "measured": {},
  "unverified_assumptions": [],
  "full_context_feature_stream_authorized": false,
  "full_context_initialization_status": "BLOCKED_REQUIREMENT",
  "failures": [],
  "next_authorized_action": "Run read-only discovery from section 4.2"
}
```

A stage report contains: hypothesis; exact changed files; exact commands; observed output; state/token comparison; measured cost and memory; result; remaining uncertainty; next permitted action. A local agent without access to the GPUs stops as `BLOCKED_ENVIRONMENT` with completed source work and runnable probes—it does not claim runtime verification.

## 11. Risk register

| Risk | Earliest useful detector | Required response |
|---|---|---|
| Wrong local checkpoint or assumed BF16 shared head | G0 tensor/quantization inventory | Stop loading; resolve actual allowlisted representation. |
| Too little actual expert-fetch reuse | G1 real-layer and global-cache replay | Recompute the full-cycle budget; stop if no measured mechanism supports viability. |
| Existing multi-row cache mapping unsafe | G1 duplicate/capacity tests | Fix only the demonstrated issue or reduce depth; do not replace the whole cache by default. |
| Verify enters whole-layer prefill or CPU MoE | G2 counters and assertions | Fail immediately; separate intent from loading policy. |
| Batched recurrence changes target greedy tokens | G2 sequential/block comparison | Preserve canonical recurrence/kernel semantics; no tolerance waiver. |
| Future QSA index data leaks into earlier rows | G2 causal perturbation test | Add per-row frontier/visibility logic before integration. |
| Recursive draft KV incorrectly retained | G3 teacher-forced catch-up comparison | Canonicalize accepted rows and include its cost. |
| Prompt streaming contends with expert fetch | G1/G3/G5 loaded bridge and TTFT probe | Bound staging; measure target slowdown as well as bridge rate; tune only independent overlap. |
| Cumulative full-prompt transfer violates S1 | G0 requirement audit | Block full-context MTP initialization until clarified; chunking is not a volume reduction. |
| 26.9 GB/s misclassified as CPU DRAM or link throughput | G0 benchmark provenance | Keep separate metrics; do not use the 17% conditional bound until the path is confirmed. |
| Idle Gen1 interpreted as loaded link speed | G0/G1 active-transfer telemetry | Record expected, capability-max, idle, and loaded values separately; no unauthorized hardware changes. |
| Shared pages are not CUDA-registered in both processes | G1 minimal registration/copy probe | Fall back to persistent local pinned buffers plus measured host copies. |
| Per-round pinning or unbounded prompt lag | G1/G3 slot/registration/frontier counters | Persistent bounded slots; explicit fallback/backpressure; never accumulate a full stream. |
| VM/pinned-memory limits despite 128 GiB installed | G0/G3 VM and allocation evidence | Bound allocations; report missing capacity; do not change WSL/BIOS/driver settings. |
| Missing GDN/PLE/QSA mutable state | G4 forced rejection and continuation | Block optimization until the inventory/transaction is complete. |
| Graph capture freezes addresses/routes or mutates padding | G5 changing-route/replay tests | Fix capture contract or retain eager execution; include the performance cost. |
| State buffers reduce target expert-cache capacity | G0/G4 peak budget and cache telemetry | Account for real deployment loss; reuse buffers, reduce depth, or stop. |
| Prefix sidecar breaks an otherwise valid target hit | G5 hit/miss/COW comparison | Preserve existing target hit and fall back; never target-prefill solely for MTP. |
| Sampling tests rely on same-seed text equality | G7 independent probability oracle | Test distribution and deterministic algorithm cases separately. |
| Cancellation or daemon loss reuses live memory | G4/G8 failure injection | Drain owned work before reclaiming; fail closed on unknown target state. |
| Benchmark overstates context or excludes startup debt | G0/G6 workload/timeline audit | Correct labels and include full request costs; rerun invalid comparisons. |
| Counters or noise cannot support the stated claim | G6 metric-provenance/uncertainty check | `INCONCLUSIVE`/unverified, never fabricated precision. |

## 12. Stop, review, and completion policy

Within the isolated experiment, the agent may proceed through G0–G5 only while each prerequisite is satisfied. A blocked environment, failed correctness gate, or failed feasibility premise stops dependent work. The unresolved full-context cumulative-transfer restriction blocks the dependent long-context MTP initialization/acceptance tasks, not independent target-only discovery or bounded short-context probes. A passing gate is permission for the next listed stage, not for unrelated refactoring. Do not interpret the user supplying hardware data as authorizing a new CPU backend or a hardware/driver/BIOS change.

At G6, stop regardless of outcome and deliver the evidence package. Stages 7–8 require explicit approval. No stage authorizes downloading another model, changing quantization, converting the checkpoint, modifying existing project launchers, committing, pushing, or contacting upstream.

Remove active-hour estimates from execution decisions. They are not acceptance evidence. Report completed artifacts, observed blockers, and the next executable action.

Final report format:

```text
Status: PASS_GREEDY / COMPLETED_EXPERIMENTAL / FAIL_* / INCONCLUSIVE / BLOCKED_REQUIREMENT / BLOCKED_ENVIRONMENT
What actually ran:
Source/model/configuration identity:
Correctness evidence:
Baseline vs off vs on performance:
TTFT and end-to-end results:
Traffic/memory/critical-path explanation:
Fallback and unsupported features:
Remaining unverified items:
Protected-boundary audit:
Next action requiring review, if any:
```

**Completion means measured runtime behavior at the final recorded source identity—not code written, tests added, or an earlier prototype that once ran.**

## 13. Source and review provenance

### Source boundary

S0: user-supplied `Pasted markdown(6).md`, titled “Qwen3.8 Flash Next MTP Development Plan,” dated 2026-09-05. Its local performance/resource figures, stage-completion claim, and reviewed donor heads were treated as source-reported until checked. This review did not access the user's machine, checkpoint, local repository, or benchmark corpus.

Version 2.0's model/state-contract review is inherited, not claimed as a new complete audit. This 2.1 amendment checked the official AMD/motherboard/NVIDIA hardware specifications, CUDA memory/peer/WSL guidance, WSL configuration documentation, and the pinned FreeToken offload source to clarify the possible gather data path. NVIDIA specification tables were visually inspected. No local repository, checkpoint, or GPU was accessed.

S1: the user's hardware-bandwidth update in this conversation (9950X3D, 2 × 64 GiB DDR5-6000, B850 AI TOP, expected Gen5 x8 / Gen4 x8 links, reported NS/PXB topology, and the full-feature-stream restriction). Local configuration and measurement statements are user-reported. H references support manufacturer capabilities and API behavior, not actual achieved throughput. New state/protocol/gate designs and derived costs must be validated by the local agent.

### Reference index

URLs are provided as a source-retrieval block for the implementation agent. Public documentation/PR pages can move; pin actual blobs before copying code.

```text
R1 — Pinned FreeToken Qwen4Exp target model, feature collapse and head dispatch:
https://raw.githubusercontent.com/FlashML-org/FreeToken/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/models/qwen4_exp/model.py

R2 — Pinned FreeToken MoE layer, on-demand/prefill dispatch and selected kernels:
https://raw.githubusercontent.com/FlashML-org/FreeToken/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/layers/moe.py

R3 — Pinned FreeToken offload cache, mappings, copy paths, and counters:
https://raw.githubusercontent.com/FlashML-org/FreeToken/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/moe/offload_cache.py

R4 — Pinned FreeToken GDN and related state/attention code:
https://raw.githubusercontent.com/FlashML-org/FreeToken/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/models/qwen4_exp/gdn.py
https://raw.githubusercontent.com/FlashML-org/FreeToken/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/models/qwen4_exp/ple.py
https://raw.githubusercontent.com/FlashML-org/FreeToken/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/attention/qsa_sparse.py

R5 — vLLM Qwen4Exp MTP numerical wiring and two-output convention:
https://docs.vllm.ai/en/latest/api/vllm/models/qwen4_exp/amd/mtp/

R6 — ik_llama Qwen4Exp MTP implementation discussion:
https://github.com/ikawrakow/ik_llama.cpp/pull/2369

R7 — llama.cpp recurrent/convolution rollback fix, merged September 1, 2026:
https://github.com/ggml-org/llama.cpp/pull/28123

R8 — Leviathan, Kalman, Matias, Fast Inference from Transformers via Speculative Decoding:
https://proceedings.mlr.press/v202/leviathan23a.html

R9 — vLLM speculative-decoding numerical/losslessness qualifications:
https://docs.vllm.ai/en/latest/features/speculative_decoding/

R10 — FreeToken DFlash donor and S0-pinned engine/worker source:
https://github.com/FlashML-org/FreeToken/pull/258
https://raw.githubusercontent.com/FlashML-org/FreeToken/9f0a136bdbf2b1f25066dc21ce2fa770b42da78e/python/freetoken/engine/engine.py
https://raw.githubusercontent.com/FlashML-org/FreeToken/9f0a136bdbf2b1f25066dc21ce2fa770b42da78e/python/freetoken/speculative/dflash/worker.py

R11 — Pinned FreeToken request, batch, and process-global context contracts:
https://raw.githubusercontent.com/FlashML-org/FreeToken/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/core.py

R12 — FreeToken DSpark donor:
https://github.com/FlashML-org/FreeToken/pull/69
```


### Hardware amendment source index — checked 2026-09-05

```text
H1 — AMD Ryzen 9 9950X3D: cores/threads, two memory channels, two-DIMM DDR5-5600 rating:
https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-9-9950x3d.html

H2 — Gigabyte B850 AI TOP rev. 1.0: CPU-connected x16/x8 slots and lane sharing:
https://www.gigabyte.com/Motherboard/B850-AI-TOP-rev-10/sp

H3 — NVIDIA RTX Blackwell architecture, RTX 5090 specification table (PDF page 15):
https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf

H4 — NVIDIA Ada architecture, RTX 4090 specification table (PDF page 31):
https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf?ncid=no-ncid

H5 — CUDA Driver API: peer capability queries and context access:
https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html

H6 — NVIDIA System Management Interface: topology, link status, unsupported queries:
https://docs.nvidia.com/deploy/nvidia-smi/index.html

H7 — CUDA on WSL: pinned-memory and telemetry limitations:
https://docs.nvidia.com/cuda/wsl-user-guide/index.html

H8 — Microsoft WSL configuration: VM memory budget and settings:
https://learn.microsoft.com/en-us/windows/wsl/wsl-config

H9 — CUDA Best Practices: data-transfer batching, pinned memory, mapped access, overlap:
https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
```

These documents describe capabilities and generic constraints; none supplies this system's sustainable CPU memory, exact expert-gather throughput, bridge latency, MTP acceptance, or end-to-end speed. Those fields remain unmeasured until the local agent produces the required evidence.
