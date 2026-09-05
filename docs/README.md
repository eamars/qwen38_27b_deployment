# Documentation index

Read the documents in this order when operating the workspace:

1. [Deployment](deployment.md) — what is installed, how the independent and
   shared launch modes are configured, and the commands to prepare, launch,
   stop, and profile them.
2. [Benchmarks](benchmarks.md) — what has actually been measured, what was
   selected provisionally, and which acceptance work remains.
3. [Models](models.md) — model sources, file sizes, and hashes.
4. [Host inventory](host-inventory.md) — the hardware and toolchain snapshot
   used for the measurements.
5. [History](history.md) — the timeline of deployment stages and configuration
   decisions.
6. [Qwen3.8-Flash-Next FreeToken deployment](qwen38-flash-next-freetoken.md) —
   the retained RTX 5090 configuration, measured 4K result, and operational
   commands.

The [script inventory](../scripts/README.md) lists the canonical launch,
profiling, setup, and inventory helpers.

The FreeToken MTP-on-RTX-4090 experiment is closed. Its
[conclusion and evidence](archive/qwen38-mtp-4090-conclusion.md) are retained
for cleanup of the runtime test ground; deletion is pending an execution-policy block.

The material in [archive](archive/) records earlier handover and instruction
stages. It is useful for provenance, but commands there may describe
superseded scripts or pre-measurement assumptions.
