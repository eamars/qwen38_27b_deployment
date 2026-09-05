# Qwen3.8 Flash Next MTP — Revised Agent Execution Plan

Version: 2.0  
Date: 2026-09-05  
Status: reviewed plan; implementation and hardware results NOT verified in this review  
Baseline: `af71ba43206e124f5ff6419b47ee36c6e9981078`  
Experiment branch: `exp/qwen38-mtp-4090`  
Experiment worktree: `C:/workspace/qwen38_27b/runtime/freetoken-mtp-4090-exp`

This document supersedes the attached **Qwen3.8 Flash Next MTP Development Plan** for execution order and implementation contracts. It preserves the checkpoint, devices, offload policy, isolation boundary, default-off behavior, and approval boundary. The original plan is source S0; public sources checked during this review are indexed in §13. New contracts and stage gates below are proposed engineering decisions, not claims that the local runtime already implements them.

## Executive direction

**Prove that the target can profitably verify several tokens with its existing offload machinery before building a speculative-serving framework. Then prove a real, depth-one, two-GPU MTP loop. Expand only after those probes work.**

Required progression:

`local evidence → real offload probe → target-only verification oracle → real MTP alignment → minimal two-GPU loop → measured optimization → acceptance benchmark → approval → serving completion`

Do not treat route overlap, an upstream benchmark, successful imports, unit-test counts, or a high draft-acceptance rate as proof that this design works on this machine.

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

All hardware, checkpoint-size, bridge, and throughput figures from S0 initially remain `SOURCE_REPORTED`.

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

`discovery.json` records paths, SHAs, existing diffs, device UUIDs, model metadata hashes, tokenizer/template identity, dependency versions, backend configuration, baseline command as an argument array, and benchmark availability.

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

### 5.3 Bridge volume

At the reported width, one BF16 wide feature is `10240 × 2 = 20,480 bytes` (20 KiB). A complete 262144-row feature stream is **5 GiB**, before token/control metadata. These are arithmetic estimates, not measured transfer volumes.

Never retain that full prompt-feature stream merely for convenience. Use bounded chunks. Probe 20 KiB, 40 KiB, and up to `(k+1) × 20 KiB` round payloads, plus representative prompt chunks. The larger round payload accounts for canonical draft catch-up (§6.4).

Benchmark the actual device→host→process→host→device path under target expert-fetch load, not only idle copy bandwidth. Account for socket serialization, local pinned staging, host copies, acknowledgements, and copy completion.

## 6. Proposed architecture and contracts

### 6.1 Keep two processes; begin with the simplest transport

Target process: 5090 only, authoritative target state and user-visible output.  
Draft process: 4090 only, allowlisted MTP/shared tensors, canonical draft cache plus discardable recursive scratch.

Start with a length-prefixed Unix-domain socket inside WSL and bounded per-process staging buffers. Use a shared-memory data plane only when the real profile shows a material socket/copy cost. Do not implement three transports before the first model round.

Use process-local pinned staging unless registration of shared pages has been explicitly tested on both processes. “Shared memory” does not mean “pinned memory”; a GPU producer/consumer must finish before a slot is acknowledged reusable.

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

Cold prompt processing streams bounded target feature/token chunks to build canonical MTP state. Do not block first-token publication merely to hide a slow drafter; however, time-to-second-token, outstanding catch-up, decode throughput, and full request latency must include the consequences of any lag.

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

1. Execute §4 discovery; inspect existing instructions, interpreter, launch configuration, and actual model metadata.
2. Record raw/tied/quantized tensor inventory and a device/host memory budget. Assert no target expert bank will load in the draft process.
3. Record kernel backends, KV/index dtype, prompt chunk sizes, prefix-cache mode, offload-cache size, and graph settings.
4. Recover or construct a frozen benchmark manifest with tokenizer/template hashes and exact input IDs. Preserve the existing meaningful code workload. Synthetic development prompts must be labeled synthetic, not presented as the user's established benchmark.
5. Run protected baseline repeatability at a small context, then the near-full-context primary shape. Do not run a second target alongside it.
6. Save environment and baseline raw results. Verify no protected source/launcher/model changes.

**Required artifacts:** `discovery.json`, `execution.json`, `source-audit.md`, `checkpoint-map.json`, `memory-budget.json`, corpus manifest, baseline run records.

**Gate G0:** exact source/model/device/configuration identity established; baseline loads and generates correctly; benchmark definition is valid. If the historical 40 token/s cannot be reproduced, use the newly measured baseline and explain the difference. Do not fabricate missing historical evidence.

### Stage 1 — Probe existing offload economics and the bridge

**Question:** Does the existing target machinery provide credible room for block verification?

Implement only the minimal probe adapter and counters necessary for:

1. Capturing a bounded real baseline trace: per-token/per-layer raw expert IDs, routes, missing IDs, bank bytes, target latency, and cache capacity.
2. Replaying global-cache access order for ordinary token-major execution versus proposed layer-major block execution at `k=1,2,4`. Include the pending anchor: block widths are `2,3,5` rows.
3. Running representative real NVFP4 offload layers at those row counts using captured real inputs/routes and the existing device-side path. Compare outputs against per-row execution and measure gather plus expert compute, not just set overlap.
4. Measuring real two-process GPU handoff for 20–100 KiB payloads, bounded prompt chunks, and under simultaneous target expert-fetch pressure. Start with sockets. Measure an alternative only when a bottleneck is demonstrated.
5. Estimating full-cycle latency using §9.1. Keep observed acceptance separate from hypothetical acceptance until MTP actually runs.

The cache simulator must use the real global cache order/capacity and allocation rules. Summing independent per-layer hit rates can be wrong when execution order changes. Include rejected-suffix cache pollution once real draft traces exist. Route traces along a known correct continuation are an optimistic scenario, not the routes of rejected candidates.

Use real quantized layers; a mock kernel or fabricated bandwidth number cannot pass G1. Confirm whether repeated rows actually share expert transfers, and whether the selected expert kernel repeats HBM work per row.

**Required artifacts:** route trace, validated cache replay, layer output comparisons, layer timings, bridge timings, initial cycle-budget table.

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
5. Implement a minimal independently launched draft process and socket transport, with sequence/generation checks and bounded prompt streaming.
6. Connect a **diagnostic** greedy round using the slow target transaction oracle. Complete multiple consecutive rounds; force `a=0` and `a=1` rather than accepting only favorable drafts.
7. Validate canonical draft catch-up against a fresh teacher-forced cache, including the all-accepted final input row.
8. Measure draft compute, canonical catch-up, bridge, target verify, and initialization lag at representative contexts.

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
4. Profile complete cold-prompt and paired-prefix requests at the primary context.
5. Select at most two material optimization hypotheses from that profile for this feasibility pass. For each, record the bottleneck, expected gain, changed component, isolated A/B, and post-change correctness probe. Candidate changes include draft graph capture, better prompt overlap, measured transport improvements, or target verify capture.
6. Validate before retaining each optimization. Do not add adaptive depth until fixed-depth results are understood.

If adaptive depth is added, choose from `{0,1,2,4}` using measured cycle cost, acceptance by position, cache behavior, and catch-up cost. Apply hysteresis and include exploration/fallback overhead in the request result. Never truncate q support based on confidence without reflecting the actual proposal distribution.

**Required artifacts:** depth sweep, minimal server smoke results, prefix-sidecar checks, optimization A/B records, frozen candidate configuration.

**Gate G5:** correct no-replay candidate, bounded memory, supported cache behavior, and reproducible near-full-context run. If still short of the goal, report which tested mechanism failed; do not expand into new models, CPU expert execution, or unrelated refactoring.

### Stage 6 — Final greedy acceptance benchmark and mandatory review stop

Run the locked protocol in §9. Compare protected baseline, experiment MTP-off, and experiment MTP-on. Use production-equivalent prefix caching, sampling mode for the **greedy** milestone, context capacity, output lengths, and instrumentation settings.

**Required artifacts:** complete raw results, source/environment hashes, token comparisons, latency distributions, memory and traffic data, run exclusions, and acceptance decision.

**Gate G6:** all §2.3 requirements pass. Record one of `PASS_GREEDY`, `FAIL_CORRECTNESS`, `FAIL_PERFORMANCE`, `INCONCLUSIVE`, or `BLOCKED_ENVIRONMENT`.

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
         live pages/sidecars/staging slots
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
  "plan_version": "2.0",
  "stage": "G0",
  "decision": "NOT_RUN",
  "source_identity": null,
  "commands": [],
  "evidence_paths": [],
  "measured": {},
  "unverified_assumptions": [],
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
| Prompt streaming contends with expert fetch | G3/G5 loaded bridge and TTFT probe | Bound staging; tune overlap only from the measured critical path. |
| Missing GDN/PLE/QSA mutable state | G4 forced rejection and continuation | Block optimization until the inventory/transaction is complete. |
| Graph capture freezes addresses/routes or mutates padding | G5 changing-route/replay tests | Fix capture contract or retain eager execution; include the performance cost. |
| State buffers reduce target expert-cache capacity | G0/G4 peak budget and cache telemetry | Account for real deployment loss; reuse buffers, reduce depth, or stop. |
| Prefix sidecar breaks an otherwise valid target hit | G5 hit/miss/COW comparison | Preserve existing target hit and fall back; never target-prefill solely for MTP. |
| Sampling tests rely on same-seed text equality | G7 independent probability oracle | Test distribution and deterministic algorithm cases separately. |
| Cancellation or daemon loss reuses live memory | G4/G8 failure injection | Drain owned work before reclaiming; fail closed on unknown target state. |
| Benchmark overstates context or excludes startup debt | G0/G6 workload/timeline audit | Correct labels and include full request costs; rerun invalid comparisons. |
| Counters or noise cannot support the stated claim | G6 metric-provenance/uncertainty check | `INCONCLUSIVE`/unverified, never fabricated precision. |

## 12. Stop, review, and completion policy

Within the isolated experiment, the agent may proceed through G0–G5 only while each prerequisite is satisfied. A blocked environment, failed correctness gate, or failed feasibility premise stops dependent work. A passing gate is permission for the next listed stage, not for unrelated refactoring.

At G6, stop regardless of outcome and deliver the evidence package. Stages 7–8 require explicit approval. No stage authorizes downloading another model, changing quantization, converting the checkpoint, modifying existing project launchers, committing, pushing, or contacting upstream.

Remove active-hour estimates from execution decisions. They are not acceptance evidence. Report completed artifacts, observed blockers, and the next executable action.

Final report format:

```text
Status: PASS_GREEDY / COMPLETED_EXPERIMENTAL / FAIL_* / INCONCLUSIVE / BLOCKED_*
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

Public-source inspection checked the pinned model/core/offload/GDN code and the specific implementation references below. It was not a complete repository audit or a hardware test. New state/protocol/gate designs in this document are proposals that the local agent must validate.

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
