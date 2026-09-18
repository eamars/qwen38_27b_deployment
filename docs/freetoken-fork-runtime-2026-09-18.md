# Qwen3.8 runtime from eamars/FreeToken

The active FreeToken source is now the user's fork:

- Repository: <https://github.com/eamars/FreeToken>
- Branch: `main`
- Commit: `ae8b3cfaef74bb3b687dce8e761f6355171c6137`
- Base: upstream v0.1.3, `cac247a860e316e06580d05aeb05f2e647bde214`
- Checkout: `C:\workspace\qwen38_27b\runtime\freetoken-eamars`
- Editable installation: `/home/rba90/.freetoken-qwen38/venv`

The commit was pushed to the fork and contains only the checkpoint handoff
fix and its regression tests. No PR was created; the user will create it.

## Runtime overlay and configuration

The uncensored loader adaptations in `weight.py` and `ple_disk.py`, plus the
`--linear-state-cache-ratio` CLI exposure in `server/args.py`, remain local
changes in the new checkout. They are preserved in
`artifacts/freetoken/qwen38-local-compat-v013.patch`. These three files are
deliberately excluded from the fork commit so the proposed upstream PR stays
focused. All five modified files, including the committed fix and tests,
match the previously tested checkout after normalizing line endings.

The old `runtime/freetoken-a80b4d3` checkout is retained for recovery. Future
source changes and rebuilds should use `runtime/freetoken-eamars`.

All four Qwen3.8 launchers invoke the shared venv's `ft`, so both checkpoints
and both text/vision launch modes now resolve to the fork. The source migration
did not change launcher settings. Later on 2026-09-18, the uncensored vision
launcher was changed to two concurrent requests and ratio 2 (12 usable
recurrent-state slots), retaining 4,600 expert slots, 262,144 shared KV/context
tokens, 8,192-token prefill chunks and memory ratio 0.90. The checkpoint fix
remains installed. The verification below records the earlier one-request,
24-slot source-migration run.

Rebuild, after verifying/preserving the local overlay:

```bash
CUDA_HOME=/usr/local/cuda-13.3 \
PATH=/home/rba90/.freetoken-qwen38/venv/bin:/usr/local/cuda-13.3/bin:/usr/local/bin:/usr/bin:/bin \
/home/rba90/.local/bin/uv pip install \
  --python /home/rba90/.freetoken-qwen38/venv/bin/python \
  --no-deps --no-build-isolation \
  -e /mnt/c/workspace/qwen38_27b/runtime/freetoken-eamars
```

The installed version remains 0.1.3. The rebuild did not change dependencies;
`uv pip check` passed for all 105 installed packages. The three active probe
and benchmark helpers now reference the fork source; the old benchmark's
hard-coded source revision was replaced with the checkout's actual HEAD.

## Verification

The fork checkout passed 35 targeted scheduler tests before the commit was
pushed. The installed fork then passed all five loader checks, including the
compiled PLE store's BF16 and FP8 row handling. These are loader checks, not
full inference coverage for both checkpoints. The updated helper scripts
passed Python syntax parsing.

The live restart and direct cache probe are recorded under
`benchmarks/raw/freetoken-fork-runtime-2026-09-18/`. The 16,443-token probe
passed: 17.300 seconds cold, 3.879 seconds on the identical repeat with
16,320 cached tokens. Both returned exactly the same 15-token output as the
pre-fix reference. The restarted service reports healthy on v0.1.3, with
instance `b737c992-f53e-437b-a267-df228be65687` and the unchanged cache geometry.
This live check exercised the uncensored vision deployment; the official
checkpoint was not separately reloaded during this source migration.
The broader correctness
limits from the [handoff fix report](freetoken-checkpoint-handoff-fix-2026-09-18.md)
still apply, including the longer-output reasoning variation. No DSH
conversation or setting was changed.
