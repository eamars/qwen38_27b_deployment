# FreeToken parallel prefill reproduction, 2026-09-19

Verified on the actual RTX 5090: two requests decode together, but cold prefill
delays an already-generating request. Two identical prompts submitted together
also both prefetched their entire input when no reusable checkpoint existed at
admission. Once cached, the same pair reused its prefix successfully.

This is a synthetic reproduction through FreeToken's real streaming API, not
a captured DSH conversation. It establishes two runtime mechanisms that can
explain the symptom; it does not identify every miss in the user's DSH traffic.

## Environment and scope

- Model: `orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4`, vision-enabled,
  with text-only requests in this test.
- GPU: RTX 5090, nominal 32 GB; CPU: Ryzen 9 9950X3D; host RAM: 128 GB class;
  runtime under WSL2.
- Source: `eamars/FreeToken`, v0.1.3,
  `ae8b3cfaef74bb3b687dce8e761f6355171c6137`, including the checkpoint handoff fix.
  Existing uncensored loader and CLI overlays were unchanged.
- Two running requests, ratio 2, 12 usable GDN slots, 4,600 expert-cache slots,
  shared 262,144-token KV pool, 8,192-token prefill budget, memory ratio 0.90.
- CPU MoE auto placement selected the same 15 head/tail layers as previously.
  Existing pageable-bank/mlock warnings remained. Other host workloads were
  not isolated, so these are diagnostic single-run timings, not a throughput
  benchmark.
- Production port 1919 was stopped before testing. A temporary instance used
  port 1920 and was stopped after the completed probe; launcher settings and
  runtime code were not changed.

## Procedure

The streaming client timestamped each nonempty content/reasoning SSE event
with a monotonic clock and recorded the final API token usage. First-token
time means first nonempty content or reasoning event. Inter-event gaps are
client-observed delivery pauses, not direct GPU-kernel timings. Scheduler logs
provide independent evidence of prefill and two-request decoding.

An initial 9,264-token request warmed the runtime and is excluded below. It
took 74.61 seconds, so counting it as concurrency cost would be misleading.
Requests used temperature 0 and low reasoning effort. Long prompts contained
36,919 tokens and requested at most 32 output tokens (31 reported). The short
counting prompt contained 83 tokens, requested 256 output tokens, and produced
255 reported tokens in each measured counting run.

## Simultaneous identical prompts

Two client threads started the exact same long prompt behind a barrier. The
second pair repeated those requests after the first pair completed.

| Case | Cached tokens per request | First event A / B | Both complete | Longest A pause |
|---|---:|---:|---:|---:|
| Simultaneous cold pair | 0 / 0 | 33.782 / 53.141 s | 54.422 s | 19.437 s |
| Simultaneous cached pair | 36,864 / 36,864 | 6.391 / 6.391 s | 7.672 s | 0.078 s |

The cold requests both processed the input despite identical prompts. A later
cached pair processed only 55 uncached tokens per request. This is evidence of
duplicated work on overlapping cold admission, not evidence that caching is
globally disabled or that the checkpoint handoff fix failed. Logs show a
two-sequence prefill batch and subsequently `Decode batch, #running-req: 2`.

## Long prefill arriving during generation

The short counting prompt was seeded first. Its measured requests each reused
64 prompt tokens. The incoming long prompt had a distinct early prefix, so it
could not reuse the preceding long-prompt test's checkpoint. Request B was
submitted immediately after request A emitted its 16th nonempty SSE event.

| Scenario | A total time | Longest gap in A | Incoming B first event | B cached tokens |
|---|---:|---:|---:|---:|
| A alone | 9.406 s | 0.078 s | — | — |
| Cold B arrives | 34.922 s | 19.891 s | 25.046 s | 0 |
| Same B, now cached | 13.203 s | 3.078 s | 3.172 s | 36,864 |

The cold arrival added 25.516 seconds to A's completion time. The largest
individual pause was 19.891 seconds; other delays account for the remainder.
The warm case still had a 3.078-second pause. This experiment does not separate
tokenization/request preparation, short prefill, and other shared-resource
costs within that residual delay.

The installed scheduler selects a prefill batch whenever one can be scheduled,
falling back to decode only otherwise (`scheduler/scheduler.py`,
`_schedule_next_batch`). The observations are consistent with that strict
prefill priority. Both sequences subsequently decode in one batch; they are
not independent GPU executors.

## State pool and memory

Minimum sampled free VRAM was 1,153 MiB across 332 samples, with no monitoring
errors or observed OOM. All 11 probe requests completed. The log reached
8/12 non-evictable state slots during two-request work.

That gauge excludes evictable cached snapshots, so 8/12 is not proof that four
slots are unused. Twelve slots allow the worst-case eight working slots plus
four additional retained checkpoints. Eight slots is the two-request working
floor and would save about 441 MiB, but it was not tested here and would leave
no additional checkpoint capacity at peak working occupancy. No slot reduction
was made.

## Implications

The earlier overlapping-request smoke test established admission and successful
completion, not fair scheduling. This reproduction directly establishes the
generation stalls and duplicate cold prefill described above.

Potential remedies address different causes:

1. Interleave bounded decode work with prefill chunks to improve responsiveness.
   This does not eliminate the total prefill work and needs a separate scheduler
   implementation and validation.
2. For genuinely shared cold prefixes, defer duplicate admission until a reusable
   checkpoint is available, or prefill a stable common prefix before fan-out.
   A client-only approach works only when the tokenized prefixes match through
   a retained state boundary; neither remedy was implemented or benchmarked here.
3. Inspect actual DSH tokenized prefixes to distinguish changed prompts from
   missing/evicted checkpoints. This experiment does not prove DSH preserves the
   same prefix as the synthetic identical-prompt case.

Reducing the state pool does not address either reproduced scheduling behavior.
One-request operation might improve responsiveness for this workload, but this
run did not perform a matched one-versus-two configuration benchmark.

## Reproduction artifacts

Directory: `benchmarks/raw/parallel-prefill-20260919-202934/`.

- `probe.py`: exact synthetic prompts, SSE timing, barrier and arrival trigger.
- `result.json`: all event timestamps, text/reasoning deltas, usages and VRAM samples.
- `probe.stdout.log`: per-request summaries and phase boundaries.
- `server.stdout.log` / `server.stderr.log`: admission, prefill, decode and startup logs.
- `launch.json` / `final-state.json`: temporary endpoint/process and final runtime state.

Commands used:

```powershell
pwsh -NoProfile -File scripts\start-qwen38-flash-next-uncensored-freetoken-vision.ps1 -Port 1920
python benchmarks\raw\parallel-prefill-20260919-202934\probe.py
pwsh -NoProfile -File scripts\start-qwen38-flash-next-uncensored-freetoken-vision.ps1 -Port 1920 -Stop
```

The launcher ran in a hidden background PowerShell process with redirected logs.
Raw artifacts are ignored by Git and must be attached separately if sharing an
issue. No upstream PR or runtime patch was created for this reproduction.
