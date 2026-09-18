# FreeToken local compatibility patches

## Which patches to keep

Checked against the active fork checkout on 2026-09-19:

| Patch | Role |
|---|---|
| `qwen38-local-compat-v013.patch` | Required recovery copy of the three uncommitted loader/CLI changes in the active runtime. Reverse-application check passes. |
| `checkpoint-handoff-v013.patch` | Historical recovery copy. Already committed in fork commit `ae8b3cf`; do not reapply to that commit. Reverse-application check passes. |
| `qwen38-uncensored-compat-f7dbab7.patch` | Historical loader patch for the older `f7dbab7` base. |
| `checkpoint-cache-cli-f7dbab7.patch` | Historical CLI patch for the older base; included in the combined v0.1.3 compatibility patch. |

Retain these small files for reproducible rebuilds and rollback. They are not
loaded by the server at runtime. Do not apply every patch together: choose
the patch matching the source revision and check applicability first.

The active Qwen3.8 runtime now uses `runtime/freetoken-eamars`, cloned from
`https://github.com/eamars/FreeToken.git`, branch `main`, commit
`ae8b3cfaef74bb3b687dce8e761f6355171c6137`. That commit contains the checkpoint
handoff fix and tests. The three loader/CLI changes below remain applied
locally; the fork branch stays focused on the checkpoint-fix PR. All Qwen3.8
launchers share the rebuilt WSL environment. The earlier checkout below is
retained for recovery.

The previous runtime at `runtime/freetoken-a80b4d3` is upstream v0.1.3
(`cac247a860e316e06580d05aeb05f2e647bde214`), restored on 2026-09-18 for a
controlled speed comparison. It was first upgraded on 2026-09-17 and rolled
back that day to `f7dbab7f151df353d70b325a8ac09ce0f7f1c456` (v0.1.2) after
the user reported slower performance.
The directory name is historical and does not identify the current commit.

The September 18 restoration preserved all three local modifications in stash
`90eb1ce816cbbd7221d2809771e4443984dedc44` and applied it cleanly to v0.1.3.
`qwen38-local-compat-v013.patch` backs up those three modifications relative to that release.
The real-model speed/cache comparison is recorded in
[the comparison report](../../docs/freetoken-v013-comparison-2026-09-18.md).

`qwen38-uncensored-compat-f7dbab7.patch` preserves the two local modifications
relative to the previous upstream commit
`f7dbab7f151df353d70b325a8ac09ce0f7f1c456`. They were integrated into the shared
runtime on 2026-09-12 and retained during the September 14 and 17 upgrades.

The `orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4` checkpoint requires:

- Compressed-tensors NVFP4 expert-name mapping and reciprocal global scales.
- FP8 per-channel scale handling during dense projection loading and fusion.
- BF16 PLE storage, including two-byte row strides and optional unit scales,
  alongside the official checkpoint's FP8 PLE format.

These changes affect `python/freetoken/models/qwen4_exp/weight.py` and
`python/freetoken/models/qwen4_exp/ple_disk.py`. They remain applied but
uncommitted in the shared runtime. Do not discard them when updating upstream.

An isolated v0.1.3 copy with these patches passed the existing five CPU-only
loader checks before upgrade authorization. All declared core dependencies
were compatible. The actual upgrade used only source, package/version, and
dependency checks; no model loading or benchmarks were run.

Additional recovery copy: runtime stash
`793afb357039f57c7f024ff089d1cab235c876e2`, titled
`preserve uncensored Qwen3.8 compatibility before v0.1.3 upgrade 2026-09-17`.
The stash was applied, not popped, so it remains available.

The September 17 rollback preserved an additional stash of the v0.1.3 local patches:
`df620cec5023d566c947430f44b23cc729eab709`. Both restored patched files
exactly match the original pre-upgrade stash above. The rollback rebuilt the
editable installation without changing dependencies; no model loading or
benchmarks were performed. The v0.1.3 commit remains available in Git.

This patch is already applied to the current runtime. To reuse it on a clean
checkout, check it with `git apply --check` first. Windows/Linux line endings
may require `--ignore-space-change`; the isolated upgrade check needed that
option. Review any actual code conflicts rather than discarding the patch.

## Checkpoint capacity CLI (2026-09-18)

`checkpoint-cache-cli-f7dbab7.patch` additionally exposes the existing
`linear_state_cache_ratio` engine setting as `--linear-state-cache-ratio`.
It changes only CLI argument handling; the default remains 2. The argument
accepts positive integers and sizes retained GPU GDN checkpoints per maximum
running request, with a minimum of four retained slots. The pool also reserves
four working slots per request and one padding slot.

The earlier uncensored vision profile used ratio 20 with one running request:
24 usable slots, plus padding. The current profile uses ratio 2 with two
running requests: 12 usable slots, plus padding. After moving the Windows display off the RTX 5090,
the launcher pins the GPU expert cache to 4,600 slots (the initial checkpoint
tuning used 4,200). Preserve this patch as well as the model-loader
patches when replacing the shared runtime. Verification and memory observations
are recorded in `docs/qwen38-flash-next-uncensored.md`.

## Checkpoint handoff fix (2026-09-18)

`checkpoint-handoff-v013.patch` additionally preserves the pending recurrent
snapshot marker across prefill continuations. It contains the five-line
production fix and regression tests, and is already committed in the active
fork. Keep it as recovery evidence; apply it only to an older base missing
the fix. No launcher or
cache-capacity setting was changed. See the [fix report](../../docs/freetoken-checkpoint-handoff-fix-2026-09-18.md)
for live reuse measurements, test results, and the longer-output reasoning
variation observed during validation.
