# Agent kickoff — Qwen3.8 / FreeToken / two-GPU MTP experiment

Version: 2.1, hardware-aware amendment, 2026-09-05

Read `Qwen38_FreeToken_MTP_Agent_Plan_v2_1.md` as the authoritative execution plan. It supersedes version 2.0. This kickoff does not replace its state contracts, numerical requirements, stage gates, or mutation boundaries. Manufacturer capabilities are source-checked; this machine's operating configuration and historical performance are source-reported until locally verified.

## Task and authorization

Determine whether the existing local NVFP4 target on the RTX 5090 can be accelerated by its existing BF16 MTP component on the RTX 4090, with unchanged target behavior, checkpoint, target offload policy, protected baseline, and project launchers.

Execute authorized G0–G6 tasks in dependency order. Stop at G6 regardless of outcome. G7–G8 require explicit approval. Do not commit, push, publish, contact upstream, download another model, convert/requantize weights, move target experts to the 4090, or introduce CPU expert computation. Do not change BIOS, drivers, power settings, IOMMU/ACS, or WSL configuration.

**Unresolved full-context initialization constraint:** the user prohibits retaining or transferring the complete approximately 5 GiB prompt-feature stream. Bounded chunking reduces peak allocation but still transfers the full volume cumulatively. Until the user permits cumulative chunked transfer or approves a different validated initialization design, set `full_context_feature_stream_authorized=false` and mark full-context MTP initialization `BLOCKED_REQUIREMENT`. Do not create an unauthorized warm sidecar through many small transfers. This does not block target-only G0–G2 work or bounded short-context MTP probes (initial proposed maximum 4096 prompt-feature rows). No full cold-context MTP run means no complete G6 pass.

## First executable work

Run the plan's read-only discovery in §4.2. Inspect local repository instructions, existing changes, exact baseline SHA, imported package paths, interpreter, resolved launch configuration, checkpoint tensor metadata, and GPU UUIDs. Preserve existing work; never repair setup with destructive git commands or broad process termination.

Create the evidence workspace only inside the verified experiment worktree. Resolve all paths and actual CLI flags from local evidence, not guesses. Reproduce the protected baseline before changing execution paths. Run only one full target process at a time.

Add `hardware.json` and `metric-provenance.md` to the evidence package. Separate:

- Reported installation: 9950X3D, 2 × 64 GiB DDR5-6000, B850 AI TOP rev. 1.0, 5090 + 4090 in the CPU-connected slots.
- Derived ceilings: dual-channel DDR5-6000 = 96 GB/s; expected 5090 Gen5 x8 = 31.507692 GB/s/direction; expected 4090 Gen4 x8 = 15.753846 GB/s/direction. None is measured sustained throughput.
- Live evidence: idle and loaded generation/width; P2P query results in both directions when supported; guest/host available memory, swap and pinned allocation behavior. Unsupported counters are null with a reason.
- Historical 26.9 GB/s: recover benchmark code/logs, byte numerator, GB/GiB convention, actual CPU-only versus host-to-GPU path, timing boundaries, cache state, and link configuration. Keep SOURCE_REPORTED when evidence is unavailable.

Use the same two physically installed GPUs and lane allocation for baseline, MTP-off, and MTP-on. An idle 4090 is not evidence that the 5090 regains x16.

## Probe priorities

First run the smallest real multi-row NVFP4 offload-layer probe through the baseline's existing device-side cache path. Include k+1 rows: depths 1/2/4 correspond to widths 2/3/5. Measure actual fetched bytes, cache behavior, numerical outputs, and layer latency. Do not start with a donor framework port or an exhaustive CPU-memory tuning project.

Keep CPU streaming, CPU indexed gather, individual pinned PCIe transfers, real expert-gather kernels, and two-process bridge results separate. Run a clean bounded CPU-memory diagnostic when feasible; it does not authorize a CPU-MoE backend. An operation called gather may already read host memory from a GPU across PCIe.

Measure target expert fetch alone and with the actual bridge active. Record both consumer-ready bridge latency and target slowdown. A 20 KiB message's 1.95 microsecond link-only serialization estimate is not a measured IPC latency or a runtime pass criterion.

## Bridge implementation from the first real loop

Use two isolated GPU processes. Start with Unix-domain socket control and, initially, socket data if simplest. Allocate persistent process-local pinned staging and two bounded slots; coalesce the valid feature rows for each round. At the reported width, decode capacity through k=4 is 100 KiB per slot, plus separate metadata/replies. Count all per-process host/GPU buffer copies.

Do not allocate/pin/unpin every round. Do not send tensor objects through multiprocessing.Queue. Use a shared-memory data plane only after a minimal registration probe and the loaded profile justify it; shared memory is not automatically CUDA-pinned.

Enforce slot ownership: `FREE -> PRODUCER_COPY -> HOST_READY -> CONSUMER_COPY -> REUSABLE`. Publish only after producer completion; release shared source memory only after the consumer's H2D copy completes. Device destination scratch remains live through its compute consumers. Validate generation/request/round/slot sequence numbers. No device-wide synchronization in the final fast path.

Record valid, discarded and retransmitted rows; host-copy bytes; registration counts; queue/slot high-water marks; and consumer-ready p50/p95/p99 latency. Overlap only genuinely independent work. Two GPUs do not remove the dependency from drafting to target verification to canonical catch-up.

## Preserve these correctness contracts

A depth-k round verifies the pending anchor plus k candidates. Accepted, emitted, and processed-token counts are distinct. Implement §6.3's exact row-to-logit/frontier contract or prove an equivalent adapter.

Export the correct wide target pre-mixer feature. Keep MTP LM-head input and recursive wide output distinct. Accepted tokens do not automatically make recursive draft KV canonical: validate target-feature catch-up against the declared teacher-forced reference and include its cost. Normally a+1 feature rows accompany a+1 emitted tokens, approximately 20 KiB/emitted token at the reported width, before extra copies/metadata.

Preserve every GDN, convolution, PLE, QSA/index/ring, page and scheduler frontier. Reference target replay is diagnostic only; the accepted performance path must not replay the target. Never tune around known incorrect output.

Keep MTP-off free of draft startup, hidden-feature capture, staging allocations, and synchronization overhead. Missing prefix sidecars preserve existing target reuse and fall back to target-only; fallback is not an MTP speed sample.

## Reporting and stopping

Use hypothesis -> minimal executable probe -> observed evidence -> focused tests/hardening -> rerun. Unit tests support runtime evidence; they cannot replace it. No fake successful commands or measurements.

Identify each run by baseline SHA plus experiment diff/untracked-source hashes, environment, checkpoint identity, exact argv, GPU UUIDs, and hardware state. Preserve raw logs and machine-readable metrics. Update `.mtp-exp/status.json` at each gate with changed files, commands, observed output, correctness comparison, costs/memory, decision, unresolved constraints, and next permitted action.

Proceed only when prerequisites pass. A blocked full-context initialization stops dependent full-context work, not independent allowed probes. When blocked, deliver evidence already obtained and the exact missing requirement; do not claim a local GPU test without executing it.

The final greedy gate remains: decode >= max(48 token/s, 1.20 × fresh protected baseline), prefill/TTFT regression <= 3%, MTP-off regression <= 1%, correct outputs and state, no target replay, lower primary end-to-end latency, and an explained traffic/critical-path result. Include initialization/catch-up debt and real cache-capacity costs. Full-context prompt streaming must obey the resolved user constraint. Stop for review at G6 regardless of outcome; a passing greedy candidate is not a completed sampled-serving runtime.
