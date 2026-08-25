# Consolidated benchmark results

The raw measurements are grouped under [benchmarks](../benchmarks/):

```text
benchmarks/
├── qwen27b/2026-08-21/   initial dual-GPU baseline
├── qwen27b/2026-08-26/   Qwen 5090 n-max comparison
└── gemma4/2026-08-25/    Gemma 4 / MTP experiment
```

See [benchmarks/README.md](../benchmarks/README.md) for the file-level index.
The JSON/CSV files are the source data; the tables below are the readable
decision record.

## Acceptance rules

The deployment requires correct retrieval, full GPU residency, DFlash2 active,
target KV `Q8_0` or better, and at least 1024 MiB minimum free VRAM during the
accepted stress workload. Throughput is useful only after those conditions
hold. Measurements from different dates are not directly comparable unless
the prompt, context, runtime, background GPU load, and cache state match.

## Current Qwen evidence

The 2026-08-26 runs used the RTX 5090 Q6_K_M backend at context 126976 with
the same DFlash2 setup and compared `n-max=5` and `n-max=7`.

| Case | Prompt / output | TTFT (s) | TG (tok/s) | Wall (s) | 5090 min free VRAM | Retrieval |
|---|---|---:|---:|---:|---:|---|
| `n-max=5` | 118K / 2048 | 83.94 | 75.65 | 103.30 | 1439 MiB | pass |
| `n-max=7` | 118K / 2048 | 84.29 | 77.96 | 110.56 | 1169 MiB | pass |
| `n-max=5` | 118K / 1024 | 83.42 | 88.00 | 95.06 | captured separately | pass |
| `n-max=7` | 118K / 1024 | 84.47 | 87.09 | 96.23 | captured separately | pass |

The long VRAM files are `k5-vram-long.csv` and `k7-vram-long.csv`. Their
minimum RTX 5090 free-memory samples are 1439 MiB and 1169 MiB respectively.
The 4090 samples in those files represent concurrent background use and are
not a replacement for a dedicated 4090 acceptance run.

The provisional choice is `n-max=5`: it kept the reserve and completed the
2048-token deep run faster. The difference is workload-sensitive, so this is a
selection for the current profile rather than a universal claim that `n-max=5`
is fastest.

Files:

- [k5 API](../benchmarks/qwen27b/2026-08-26/k5-api-sustained.json) and [k7 API](../benchmarks/qwen27b/2026-08-26/k7-api-sustained.json)
- [k5 deep 118K](../benchmarks/qwen27b/2026-08-26/k5-deep-118k.json) and [k7 deep 118K](../benchmarks/qwen27b/2026-08-26/k7-deep-118k.json)
- [k5 deep 1024](../benchmarks/qwen27b/2026-08-26/k5-deep-118k-1024.json) and [k7 deep 1024](../benchmarks/qwen27b/2026-08-26/k7-deep-118k-1024.json)
- [k5 VRAM](../benchmarks/qwen27b/2026-08-26/k5-vram-long.csv) and [k7 VRAM](../benchmarks/qwen27b/2026-08-26/k7-vram-long.csv)

## Initial Qwen baseline — 2026-08-21

| Backend | Configuration | Minimum free VRAM | TTFT (s) | TG (tok/s) | Retrieval |
|---|---|---:|---:|---:|---|
| RTX 5090 | Q6_K_M, context 131072 | **640 MiB — reject** | 0.434 | 114.3 | n/a |
| RTX 5090 | Q6_K_M, context 126976, 118K prompt | 1272 MiB | 85.864 | 84.75 | pass |
| RTX 4090 | Q4_K_XL, context 110000, 100K prompt | 1103 MiB | 71.502 | 65.44 | pass |
| RTX 4090 | Q4_K_XL, context 110000, 105K prompt | 1103 MiB | 77.144 | 67.61 | pass |

The 131072 5090 run failed the hard reserve. Reducing context to 126976
provided the measured reserve used by the current launcher. The 4090 profile
passed narrowly and should be repeated with a dedicated, documented background
load before sign-off.

The baseline source files are in
[benchmarks/qwen27b/2026-08-21](../benchmarks/qwen27b/2026-08-21/).

## Gemma 4 / MTP experiment — 2026-08-25

This was a 4090, 56320-context, short 256-token comparison of target-only and
Google MTP. The best recorded MTP case was target KV `q8_0/q8_0`, draft KV
`q8_0/q8_0`, `n-max=3`:

| Case | Median TTFT (s) | Median TG (tok/s) | Median wall (s) | Mean accepted / verification |
|---|---:|---:|---:|---:|
| Target-only, q8/q8 | 0.228 | 43.27 | 6.07 | — |
| MTP, q8/q8, n=3 | 0.175 | 106.27 | 2.56 | 2.19 |

This result demonstrates a useful short-run MTP signal, not production
readiness. It has no long-context quality or reserve gate. The complete JSON
and ignored per-case logs are in
[benchmarks/gemma4/2026-08-25](../benchmarks/gemma4/2026-08-25/).

## Remaining acceptance work

- Repeat the 4090 run with a clean, explicitly recorded background state.
- Measure cold start, uncached TTFT tiers, and prefix-cache-hit TTFT.
- Verify target and drafter GPU residency from startup logs.
- Run deterministic parity, JSON/tool-call, cancellation, and intended
  harness tests.
- Record a repeatable selection for every production context and `n-max`.
