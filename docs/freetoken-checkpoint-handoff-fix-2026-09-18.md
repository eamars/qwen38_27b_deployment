# FreeToken v0.1.3 checkpoint handoff fix

Subsequently committed and pushed to `eamars/FreeToken` main as
`ae8b3cfaef74bb3b687dce8e761f6355171c6137`, then rebuilt from the fork.
See the [fork runtime record](freetoken-fork-runtime-2026-09-18.md).
The measurements below describe the initial local application.

Applied locally to v0.1.3 (`cac247a860e316e06580d05aeb05f2e647bde214`).
The running uncensored vision service was restarted with the existing launcher.
No DSH configuration, checkpoint capacity, expert capacity, concurrency,
prefill size, or memory ratio was changed. The launcher SHA256 remains
`0715a89b214de9cd5d661de3cb3b181f66aa5f09ebe306cad919c3506197eea1`.
The live geometry is 4,600 expert slots, 24 usable recurrent-state slots,
262,144 KV tokens with 64-token pages, one concurrent request, and 8,192-token
prefill chunks. The package and health endpoint still report 0.1.3.

## Change

`PrefillAdder._add_one_req` now accepts the preceding chunk's
`mamba_last_track_seqlen` and carries it alongside the existing ping-pong
buffers and cursor. If a continuation creates a newer snapshot, the existing
metadata builder replaces the marker and advances the cursor as before.
If it cannot create one, the preceding snapshot remains identifiable.

The existing final-prefill/finish cache paths then donate the correct buffer.
No intermediate cache insertion, pool resizing, kernel change, or additional
GPU buffer was introduced. This keeps the current overlap ownership rules.
The production change is five added lines in `scheduler/prefill.py`.

This repairs the missing-marker case described in the
[earlier diagnosis](freetoken-v013-comparison-2026-09-18.md#deeper-checkpoint-diagnosis).
It does not implement periodic retained checkpoints or arbitrary tool-opener
checkpoints, and cannot guarantee reuse when DSH rewrites an earlier prefix.

## Runtime evidence

The direct API reproduction used the same 16,443-token prompt, temperature 0,
low reasoning and a 16-token output cap before and after the patch. Both runs
returned 15 completion tokens.

| Request | Before patch | After patch |
|---|---:|---:|
| First request | 16.065 s; 0 cached | 47.532 s; 0 cached |
| Identical repeat | 14.894 s; 0 cached | 3.711 s; 16,320 cached |

The repeat was about 4.0 times faster in this pair. First-request timings are
not a controlled cold-start comparison: the patched service had just loaded,
and its first prefill was much slower. Do not infer a cold-prefill performance
improvement from this change. Exact returned choices, including the short
reasoning text, matched both within each pair and across the patch.

Logs confirm the fixed repeat prefills only 123 tokens. A final repeat after
the regression checks reused 16,384 tokens, taking 4.005 and 3.726 seconds;
the prior 123-token extend had created the additional later checkpoint.
Its output still exactly matched the unpatched short reference.

A longer-output check of the same boundary shape, with a different date in
the prompt and a 128-token output cap, returned `OK.` in both runs and reused
16,320 tokens. It took 21.466 s cold and 5.392 s warm. However, the reasoning
text differed: 44 versus 69 completion tokens. The strict full-choice equality
assertion failed, and the differing responses are retained. Numerical effects
of different prefill shapes are a possible explanation, not established by
this investigation. This is a remaining generation-parity limitation; the
complete final answer matched, but full reasoning-token equivalence is not
claimed.

The ordinary 36,927-token control creates a new snapshot in its final chunk,
so it does not depend on retaining the earlier marker. It took 27.535 s cold
and 4.962 s warm, reused 36,864 tokens, and returned exactly identical complete
output, including reasoning, with 64 completion tokens and finish reason
`stop`. The patched service remained healthy and idle after these checks;
no OOM or scheduler integrity failure was observed.

## Regression checks

35 targeted tests passed across hybrid cache management, chunked-prefill
accounting, and in-flight abort handling. The added 16 cases use the real
prefill manager, metadata builder, cache manager, radix matcher and cleanup
method, with CPU state tensors standing in for GPU kernel outputs. They cover:

- Final extends of 1, 59, 64 and 65 tokens.
- Two and three preceding chunks, exercising both ping-pong buffer positions.
- Final prefill commit and completion/abort cleanup before that commit.
- Retaining the preceding state versus replacing it with a newer state.
- Exact snapshot position and buffer contents, idempotent cleanup, and complete
  page/slot recovery after cache eviction.

The 59-token-tail regression fails against the unpatched v0.1.3 prefill module
loaded into an isolated test process: it returns a zero-length hit instead of
192 tokens in the small CPU fixture. The running server and source files were
not reverted for this check. Real GPU reuse is established separately by the
live reproduction above; the CPU tests do not validate GPU arithmetic.

`git diff --check`, patch reverse-application checking, and `uv pip check`
passed. Pytest and its two missing helper dependencies were installed for
testing; no existing runtime dependency was upgraded. The CPU-only tests emit
the expected Triton fallback warning when CUDA visibility is disabled.

## Evidence and recovery

Raw requests, responses, scripts, server logs and JUnit results are retained
under `benchmarks/raw/checkpoint-handoff-fix-2026-09-18/`. In particular,
`boundary-full-output.json` preserves the longer-output mismatch;
`validation.json` contains the ordinary control.

The independently applicable
[checkpoint-handoff-v013.patch](../artifacts/freetoken/checkpoint-handoff-v013.patch)
contains only this fix and its tests. Preserve it alongside
`qwen38-local-compat-v013.patch`, which retains the pre-existing loader and CLI
changes. No commit or upstream submission was made.

To remove only this fix, first review `git apply --reverse --check` against
the handoff patch, then reverse-apply it and restart the service at an idle
point using the same launcher. Keep the loader/CLI compatibility patch applied.
