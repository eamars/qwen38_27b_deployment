# Documentation index

Read the documents in this order when operating the workspace:

1. [Deployment](deployment.md) — what is installed, how the independent
   launch modes are configured, and the commands to prepare, launch,
   stop, and profile them.
2. [Benchmarks](benchmarks.md) — what has actually been measured, what was
   selected provisionally, and which acceptance work remains.
3. [Models](models.md) — model sources, file sizes, and hashes.
4. [Host inventory](host-inventory.md) — the hardware and toolchain snapshot
   used for the measurements.
5. [History](history.md) — the timeline of deployment stages and configuration
   decisions.
6. [Historical official Flash-Next FreeToken deployment](qwen38-flash-next-freetoken.md) —
   the retired RTX 5090 configuration and retained measured 4K result.
7. [Historical uncensored Flash-Next FreeToken deployment](qwen38-flash-next-uncensored.md) —
   the WSL checkpoint and loader checks, retired 2026-10-10 in favour of Strata.
8. [DSH Qwen3.8 Flash-Next vision runbook](dsh-qwen38-flash-next-vision.md) —
   the complete runtime, public-RPC catalog, compatibility, swap, and
   verification procedure for vision-capable DSH entries.
9. [FreeToken v0.1.3 comparison](freetoken-v013-comparison-2026-09-18.md) —
   matched cold/warm timings, final 4,600-slot configuration, and checkpoint-miss findings.
10. [FreeToken checkpoint handoff fix](freetoken-checkpoint-handoff-fix-2026-09-18.md) —
    applied v0.1.3 fix, direct reuse validation, regression tests, and limitations.
11. [FreeToken fork runtime](freetoken-fork-runtime-2026-09-18.md) —
    published fork commit, active source checkout, local compatibility overlay and rebuild command.
12. [FreeToken parallel prefill reproduction](freetoken-parallel-prefill-reproduction-2026-09-19.md) —
    measured duplicate cold prefill, streaming stalls, cached controls, and state-pool implications.
13. [FreeToken PR #447 local experiment history](freetoken-pr447-local-experiment-2026-09-29.md) —
    RTX 5090 + 4090 TP2+EP2, 262K capacity and a completed 128K cold request:
    31.16 token/s average decode, retained launch evidence, and untested placement proposals.
14. [Native Windows Strata Flash-Next deployment](qwen38-flash-next-strata.md) —
    pinned ModelOpt checkpoint, workspace layout, vision/MTP, long-output and
    multi-GPU measurements, plus two-agent tool/cache tests and Asuna findings.

The [historical Gemma 4 26B RTX 4090 runbook](gemma4-26b-4090-vision-mtp.md)
retains the retired profile and its measured validation. The
[2026-10-03 cleanup record](history.md#2026-10-03--deployment-cleanup) lists
removed deployments, preserved shared resources and pending Windows deletion.

The [script inventory](../scripts/README.md) lists the canonical launch,
profiling, setup, and inventory helpers.

The FreeToken MTP-on-RTX-4090 experiment is closed. Its
[conclusion and evidence](archive/qwen38-mtp-4090-conclusion.md) are retained
for cleanup of the runtime test ground; deletion is pending an execution-policy block.

The material in [archive](archive/) records earlier handover and instruction
stages. It is useful for provenance, but commands there may describe
superseded scripts or pre-measurement assumptions.
