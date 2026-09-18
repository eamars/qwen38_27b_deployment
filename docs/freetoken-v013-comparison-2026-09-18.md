# FreeToken v0.1.3 comparison, 2026-09-18

The subsequent [checkpoint handoff fix](freetoken-checkpoint-handoff-fix-2026-09-18.md)
is now applied to this v0.1.3 runtime with the same configuration. The timings
and source audit below describe the earlier, pre-fix comparison.

The shared runtime is restored to v0.1.3, commit
`cac247a860e316e06580d05aeb05f2e647bde214`, with the uncensored model-loader
and checkpoint-capacity CLI patches preserved. The installed package and
running API both report 0.1.3. Dependencies were unchanged and `uv pip check`
passed. The server remains running with 4,600 GPU expert slots, 24 usable
GDN checkpoint slots, one active request, 262,144 context/KV tokens,
8,192-token prefill chunks and memory ratio 0.90.

## Matched speed probe

The user requested reuse of the existing v0.1.2 baseline rather than another
old-version run. That baseline is commit
`f7dbab7f151df353d70b325a8ac09ce0f7f1c456`, after moving the Windows display
off the RTX 5090. Both versions used the settings above and the same local
checkpoint and dependencies. The Python benchmark scripts are byte-identical
(SHA256 `02e9f22cb4256fb071b359682317760c90cada84c64129b2eef27c3211e11f96`).

Each request has 36,927 prompt tokens, temperature 0, low reasoning effort,
and a 16-token output cap. Both versions returned 15 completion tokens.
Times below measure the complete non-streaming HTTP request, including
tokenization, prefill and generation; they do not isolate sustained decode.

| Case | v0.1.2 | v0.1.3 | Reduction in elapsed time |
|---|---:|---:|---:|
| Cold request | 32.656 s | 29.219 s | 10.52% |
| Identical warm request | 4.234 s | 3.859 s | 8.86% |

Both cold requests reported no cached tokens; both warm requests reused
36,864 tokens. Minimum sampled free VRAM was 712 MiB for v0.1.2 and 786 MiB
for v0.1.3, with no OOM. This is one cold/warm pair per version, not a
statistical performance study. It does not establish performance for images,
long output generation, or a complete DSH tool workflow.

## Checkpoint misses

The user observed more frequent misses around the earlier v0.1.3 upgrade
and v0.1.2 rollback. The matched warm probe above did not reproduce a
version-specific loss of cache reuse, but that does not establish that miss
frequency is unchanged in the actual DSH conversation.

A separate v0.1.3 probe sent the same 16,443-token text request twice. Both
reported zero cached tokens, taking 14.938 and 14.781 seconds respectively.
Its final prefill chunk is only 59 tokens. Intermediate chunk cache commits
are skipped, the continuation request does not carry the pending snapshot
marker, and an extend of at most 64 tokens cannot create a new intermediate
GDN checkpoint. An unaligned final decode state cannot be donated at a
64-token page boundary. Spare checkpoint slots cannot fix that missing state.

The earlier v0.1.2 run reproduced the same failure pattern at 147,519 tokens
(a 63-token final chunk), with no reuse on an identical repeat. This used
4,350 expert slots and 24 checkpoint slots, so its elapsed times are not a
matched speed comparison with the v0.1.3 boundary probe.

Source comparison found no changes in the hybrid radix cache, cache manager,
GDN metadata builder, Qwen3.8 GDN implementation or OpenAI chat API between
these revisions. The intermediate-chunk skip and continuation snapshot
handling are also unchanged. The prefill/scheduler changes chiefly concern
image spans for bidirectional multimodal attention. No scheduler or kernel
fix was applied here. Establishing a higher DSH miss frequency still requires
matched DSH requests and their token-prefix/cache-admission evidence; the
shared boundary failure alone neither proves nor disproves that observation.

## Deeper checkpoint diagnosis

The reproduced failure is a missing reusable checkpoint, not evidence that a
valid checkpoint was unloaded for lack of memory. Qwen3.8's hybrid cache needs
both the attention KV prefix and the recurrent/causal-convolution state at
the same token boundary. `HybridRadixCache.match_prefix` walks back to a node
with a live recurrent snapshot; matching tokens alone cannot provide a hit.
The running QSA backend requires 64-token pages, and the GDN prefill kernel
also uses a 64-token internal chunk size.

The relevant production path is:

1. `attention/linear.py::_build_track_metadata` selects the deepest boundary
   strictly inside each prefill extend: `cached_len + ((extend_len - 1) // 64)
   * 64`. An extend of 64 tokens or fewer produces no new frozen snapshot.
   The Qwen GDN layer copies both recurrent and convolution state into a
   ping-pong slot at that boundary.
2. `scheduler/scheduler.py::_process_last_data` deliberately skips cache
   insertion for every intermediate `ChunkedReq`. Its comment explains the
   ownership hazard: overlap scheduling already created the next chunk with
   the old cache handle, so inserting/freeing against the earlier handle can
   cause duplicate frees. This is intra-request overlap even at concurrency 1.
3. `scheduler/prefill.py::PrefillAdder` builds a new request for each
   continuation, preserving the live slot, ping-pong slots and next slot index,
   but not `mamba_last_track_seqlen`. Thus the previous frozen snapshot can
   still occupy a slot while its boundary is no longer recorded on the final
   request. A short final extend does not replace that missing marker.
4. `scheduler/cache.py::_cache_req_hybrid` cannot commit a prefill snapshot
   without that marker. On completion, it can donate the live state only if
   its actual position is page-aligned. Rounding the position down would
   associate a later recurrent state with an earlier token prefix and produce
   incorrect continuation state.

For the measured 16,443-token case, the chunks are 8,192 + 8,192 + 59.
The first two chunks can freeze states at 8,128 and 16,320, respectively,
but neither is inserted into the prefix tree. The final 59-token chunk
creates no snapshot and has lost the earlier marker. With an unaligned
finish and no older matching checkpoint, the repeat must prefill again.
This explains the observed zero-hit repeat. The successful 36,927-token
case has a 4,159-token final chunk, which produces the observed 36,864-token
checkpoint. These are existing runtime observations; this deeper source
audit did not rerun the benchmark or change the server.

Tool-call reuse has a separate constraint. The optional
`--enable-special-token-ckpt` flag is off in the current launcher. Even when
enabled, `snapshot_toolcall_anchor` requires the tool opener's exact state
position to be page-aligned. It therefore does not provide an arbitrary
tool-boundary checkpoint with this 64-token-page backend. If the next
request's serialized tokens diverge before the only retained checkpoint,
the cache needs an earlier snapshot to fall back to. Tool-call formatting,
reasoning removal, or earlier-message changes are possible triggers, not
verified explanations of the actual DSH requests.

History inspection places the intermediate-chunk skip in the initial
open-source commit `3af9d90`. It is also present at the older local
`a80b4d3` revision. The metadata builder and hybrid radix implementation
have no diff from `a80b4d3` to the installed v0.1.3 revision. The exact
v0.1.2-to-v0.1.3 comparison also shows no changes to the cache manager.
This supports a pre-existing limitation; it does not establish the cause
of the user's observed change in miss frequency.

The existing chunked-prefill regression tests use ordinary radix caching
with page size 1. The hybrid cache-manager tests construct request snapshot
markers directly and do not exercise the full scheduler/engine handoff.
Those tests can validate their local contracts without catching this
64-token-page, short-final-chunk combination.

The narrow repair candidate is to preserve a valid pending snapshot and its
exact boundary across continuations, then commit it when safe. That needs
runtime verification of slot ownership and overwrite ordering. More general
reuse after prompt edits requires retaining checkpoints at multiple earlier
boundaries and handling tool/decode boundaries safely. Simply removing the
intermediate-chunk skip reintroduces the documented ownership hazard.
Increasing slots, disabling overlap alone, or changing the prefill chunk
size does not repair the missing-state handoff; changing chunk size merely
moves the affected prompt lengths. No such repair was applied in this audit.

## Evidence and recovery

Raw scripts, startup logs, geometry, memory samples, version confirmation,
baseline and comparison results are under
`benchmarks/raw/freetoken-v013-comparison-2026-09-18/`. The baseline source
run is in `benchmarks/raw/checkpoint-tuning-2026-09-18-display-offload/`.

The runtime stash `90eb1ce816cbbd7221d2809771e4443984dedc44` preserves all
three pre-restoration local modifications. It was applied, not popped.
`artifacts/freetoken/qwen38-local-compat-v013.patch` preserves all three local
changes relative to v0.1.3; its reverse-apply check passed. The previous
runtime commit remains available for rollback. Launcher syntax and dry-run
checks passed. No user conversation or DSH setting was changed.
