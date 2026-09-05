# FreeToken MTP on RTX 4090: experiment closed

Disposition recorded 2026-09-06: **G2 FAIL_ECONOMICS; G2R HARD_STOP_SCOPE**.
The tested path does not meet the current checkpoint, hardware allocation,
exact numerical contract and performance requirements. This is a scoped
negative result, not proof that MTP is impossible on these GPUs.

The experiment used Qwen3.8-Flash-Next native NVFP4 on the RTX 5090, with
the RTX 4090 intended for BF16 MTP. It started from FreeToken commit
`af71ba43206e124f5ff6419b47ee36c6e9981078`. Target expert placement,
precision and the 4068-slot expert cache were retained.

## Evidence and conclusion

- G0 and G1 passed their scoped isolation, repeatability, expert and bridge
  probes. Wider real-trace cache replay showed no useful expert-byte saving.
- The G2 verifier passed the recorded correctness checks, but its later
  near-full-context two-row samples took 43.835, 42.608 and 42.850 ms.
  Even the fastest sample implies only 46.94 tokens/s at perfect acceptance,
  before MTP or commit overhead, below the absolute 48 tokens/s requirement.
  The provisional relative performance screen was higher: 54.62 tokens/s.
  These are target-only probe results, not a completed client benchmark.
- The architect's final bounded G2R-A screen covered 449 real dense call
  sites at widths 1, 2, 3 and 5. All loaded dense sites were BF16; routed
  experts remained NVFP4. Broad batching changed outputs at the first GDN
  projection. Direct router checks also changed expert sets in 7/96,
  12/144 and 24/240 layer/row cases at widths 2, 3 and 5 respectively.
- The locally exact subset saved only 0.723, 2.571 and 2.695 ms at those
  widths in operator replay. These are operator measurements, not measured
  full-target speedups; they did not justify proceeding to integration.

G2 was not completed. G2R-B and G3-G6 were not run. No MTP model, daemon,
two-GPU serving loop or fast rollback was deployed. Full-context MTP
initialization remained an unresolved requirement. The final audit recorded
no owned workers remaining and no protected source or checkpoint metadata
changes; checkpoint payloads were not fully rehashed.

Further implementation would require a newly reviewed experiment with
changed constraints or a materially different approach. The retained
[FreeToken deployment](../qwen38-flash-next-freetoken.md) remains the operating
path; these probes do not establish production 256K validation.

## Cleanup and retained artifacts

At the user's request, `runtime/freetoken-mtp-4090-exp` was prepared for
removal, including its isolated Git repository, virtual environment, caches
and bulk probe captures. The deletion command was rejected by execution
policy before it ran, so removal remains pending. `runtime/freetoken-a80b4d3`,
`runtime/llama.cpp-dflash2`, model assets and launchers were preserved.

The [compact evidence archive](qwen38-mtp-4090-evidence-2026-09-06.zip)
retains 125 files: all final report-directory files, the experiment's new
source/harness files and license, small execution metadata, and selected
G2/G2R raw JSON records. Every archived file was verified against its source
SHA256 before the attempted removal. Large tensors, caches, the virtual environment and
large profiles are omitted, so this is not a complete replay bundle.
Paths inside archived reports describe the original test ground.
The original agent plans remain historical instructions, not authorization
to resume this closed experiment.
