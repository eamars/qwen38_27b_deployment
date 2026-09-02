# Benchmark results

Benchmark artifacts are grouped by model and capture date:

```text
qwen27b/2026-08-21/   initial dual-GPU baseline
qwen27b/2026-08-26/   5090 DFlash2 n-max comparison
gemma4/2026-08-25/    Gemma 4 / Google MTP experiment
qwen38_flash_next/2026-09-02/   retained FreeToken 4K result
```

JSON files contain request and summary metrics. CSV files contain the sampled
GPU memory/utilization series used for the VRAM decisions. Generated `.log`
files are kept beside each profile for local investigation but remain ignored
by Git.

The human-readable conclusions and acceptance caveats are in
[docs/benchmarks.md](../docs/benchmarks.md). New runs should use the same
model/date layout and include the runtime commit, model hashes, configuration,
prompt size, cache state, throughput, wall time, acceptance, VRAM minimum, and
correctness result.
