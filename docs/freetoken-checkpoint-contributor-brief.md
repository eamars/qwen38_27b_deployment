# Checkpoint handoff: explanation and contribution evidence

This is a factual review aid, not a statement that the contributor has already
reviewed the code or personally executed the tests. The implementation and
tests were carried out by Codex on the contributor's machine, at their request.
The contributor should review the small diff and the evidence before submitting.

## What broke

Prefill means processing the input prompt before generating an answer. A
checkpoint is a saved model state that lets a later request resume after an
already processed prefix. This is different from the model-weight checkpoint
downloaded from Hugging Face.

The hybrid model needs both its cached attention data and its recurrent state
at the same token position. Think of the latter as a saved calculation with a
label saying how far through the input it was made.

FreeToken creates a new request object for each prefill chunk. It copied the
snapshot buffers and their cursor into the continuation, but omitted the
snapshot position, `mamba_last_track_seqlen`. A normal final chunk generated a
new snapshot and hid the omission. A final chunk of at most 64 tokens could
not generate a new snapshot, leaving the final request unable to save the
existing one into the reusable cache. If no older matching checkpoint or
aligned final-state donation rescued the request, repeating it was a cache miss.

For the reproduction, 16,443 input tokens were split into 8,192 + 8,192 + 59.
The second chunk could freeze a state after token 16,320. The last 59-token
chunk lost its position marker and created no replacement. After the fix,
the repeat resumes at 16,320 and processes only 123 remaining prompt tokens.

## What the five-line production change does

1. Adds an optional `last_track_seqlen` argument, defaulting to `None`.
2. Assigns it to the new request's `mamba_last_track_seqlen`.
3. Passes the preceding chunk's value when creating a continuation.
4. Adds two comment lines explaining why the marker must travel with the cursor.

The existing metadata builder still replaces the marker when a newer snapshot
is made. The existing final-prefill/finish paths still save and release state.
There is no new buffer, larger cache, kernel change, or model-loader change.
Intermediate cache insertion remains disabled because the scheduler already
starts the next chunk before retiring the previous one; changing ownership
there would require a separate fix to avoid duplicate frees.

The added tests exercise short and long final chunks, alternating snapshot
slots, completion before/after prefill commit, and resource recovery. CPU
state tensors represent kernel outputs in these tests; real model behavior
was checked separately through the HTTP API.

## Hardware and software

Hardware/software were rechecked on 2026-09-18. The recorded launch logs
provide the effective runtime configuration used for the measurements.

| Item | Value |
|---|---|
| GPU used for model GPU work | RTX 5090, nominal 32 GB; driver reports 32,607 MiB |
| Other installed GPU | RTX 4090, nominal 24 GB; excluded from model GPU work; Windows display moved off the 5090 |
| CPU | AMD Ryzen 9 9950X3D, 16 cores / 32 threads |
| Host RAM | 128 GB installed class; 127.64 GiB visible to Windows |
| Host OS | Windows 11 Pro, build 26200 |
| Linux environment | Ubuntu 26.04 LTS under WSL2 2.7.12.0 |
| WSL kernel | 6.18.33.2-microsoft-standard-WSL2 |
| NVIDIA driver | 610.74 |
| CUDA toolkit | 13.3, nvcc 13.3.73 |
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu130 |
| Transformers / Triton / FlashInfer | 5.16.1 / 3.6.0 / 0.6.18 |
| Model | `orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4` |
| Model revision | `c1209bda15a6bbc4c68b585e93d40c0d85f50306` |
| Baseline source | v0.1.3, `cac247a860e316e06580d05aeb05f2e647bde214` |
| Fixed source | `eamars/FreeToken:main`, `ae8b3cfaef74bb3b687dce8e761f6355171c6137` |

Upstream `main` was rechecked for this note and still pointed to the baseline
SHA. Package version remains 0.1.3 on both sides; use the SHA to distinguish them.

The test was text-only through a vision-enabled endpoint, directly against
FreeToken's local HTTP API. It did not use DSH, images, tensor parallelism or
a speculative drafter. The checkpoint requires the separate local loader
overlay, and the launch uses a local CLI exposure of `linear_state_cache_ratio`.
Both sides had those same changes. They are not part of the proposed PR.

The runtime uses expert offload and disk-backed PLE. `--moe-cpu-layers auto`
selected 15 layers for CPU decode. Logs also contain an mlock warning and
pageable expert-bank fallback on both sides. This was not a pure GPU-only
execution benchmark, and those warnings should remain in the attached logs.

## Exact launch configuration

The executed Windows launcher was:

```powershell
pwsh -NoProfile -File C:\workspace\qwen38_27b\scripts\start-qwen38-flash-next-uncensored-freetoken-vision.ps1
```

Its expanded WSL command, with the launcher's environment, is:

```bash
export CUDA_HOME=/usr/local/cuda-13.3
export TVM_FFI_CUDA_ARCH_LIST=12.0
export CUDA_VISIBLE_DEVICES=GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0
export PATH="/home/rba90/.freetoken-qwen38/venv/bin:/usr/local/cuda-13.3/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-13.3/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

/home/rba90/.freetoken-qwen38/venv/bin/ft serve \
  --model /home/rba90/models/Qwen3.8-Flash-Next-Uncensored-NVFP4 \
  --served-model-name qwen38-next-uncensored-freetoken-vision \
  --gpu GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0 \
  --host 0.0.0.0 --port 1919 \
  --max-running-requests 1 --dtype bfloat16 --memory-ratio 0.90 \
  --moe-strategy offload --moe-cpu-layers auto --moe-cache-size 4600 \
  --linear-state-cache-ratio 20 --ple-backend disk \
  --max-seq-len-override 262144 --kv-reserve-tokens 262144 --num-tokens 262144 \
  --max-prefill-length 8192 --max-output-tokens 65536 \
  --cache-type radix --enable-cache-report \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --mm-encoder-weights host
```

The effective backend is `qsa_sparse`, cache type `hybrid_radix`, page size 64,
with 24 usable recurrent-state slots. `--linear-state-cache-ratio` is supplied
by the local CLI patch, not stock v0.1.3. Apply the same compatibility overlay
to each side for this particular model/configuration. Its recovery file is
`artifacts/freetoken/qwen38-local-compat-v013.patch`.

## Reproduction and evidence

The exact client script is
`benchmarks/raw/checkpoint-handoff-fix-2026-09-18/probe.py`. It generates:

```python
unit = ' alpha beta gamma delta epsilon zeta eta theta'
content = 'Checkpoint capacity validation 2026-09-18. Read these data:' + unit * 1820 + '\nReply OK.'
```

It sends that user message twice to `/v1/chat/completions`, with the served
model ID above, `temperature=0`, `reasoning_effort="low"`, `max_tokens=16`,
and `stream=False`. The server reports 16,443 prompt tokens and 15 completion
tokens. The script records elapsed HTTP time, cache usage and complete choices.

The actual before/after commands were:

```powershell
wsl.exe /home/rba90/.freetoken-qwen38/venv/bin/python /mnt/c/workspace/qwen38_27b/benchmarks/raw/checkpoint-handoff-fix-2026-09-18/probe.py baseline
# After applying the fix and restarting at the same settings:
wsl.exe /home/rba90/.freetoken-qwen38/venv/bin/python /mnt/c/workspace/qwen38_27b/benchmarks/raw/checkpoint-handoff-fix-2026-09-18/probe.py patched
```

These are historical commands: running the `baseline` case against the currently
fixed runtime would correctly fail its assertion that the repeat has no hit.
Do not overwrite the captured baseline when reproducing in a separate setup.

| Case | Baseline | Fixed local runtime | Rebuilt fork runtime |
|---|---:|---:|---:|
| First request | 16.065 s | 47.532 s | 17.300 s |
| Identical repeat | 14.894 s | 3.711 s | 3.879 s |
| Repeat cached tokens | 0 | 16,320 | 16,320 |

These are individual end-to-end HTTP measurements. The fixed local runtime's
first request followed a model restart and was much slower. They are not a
statistical benchmark or evidence of improved cold-prefill/decode throughput.
TTFT and sustained decode tokens/s were not measured in this probe.

The three targeted test files are `tests/scheduler/test_hybrid_cache_manager.py`,
`tests/scheduler/test_scheduler_chunked_prefill.py`, and
`tests/scheduler/test_abort_inflight_prefill.py`. All 35 tests passed, including
16 added cases. The corresponding 59-token-tail test fails against the
unpatched prefill module loaded in an isolated test process. The fork test run
used the shared venv and its source via `PYTHONPATH`, with CUDA visibility
disabled. From the fork checkout, the equivalent test invocation is:

```bash
CUDA_VISIBLE_DEVICES= CUDA_HOME=/usr/local/cuda-13.3 TVM_FFI_CUDA_ARCH_LIST=12.0 \
PATH=/home/rba90/.freetoken-qwen38/venv/bin:/usr/local/cuda-13.3/bin:/usr/local/bin:/usr/bin:/bin \
PYTHONPATH=/mnt/c/workspace/qwen38_27b/runtime/freetoken-eamars/python \
/home/rba90/.freetoken-qwen38/venv/bin/python -m pytest \
  tests/scheduler/test_hybrid_cache_manager.py \
  tests/scheduler/test_scheduler_chunked_prefill.py \
  tests/scheduler/test_abort_inflight_prefill.py -q --tb=short
```

Full captured evidence to attach, rather than screenshots:

- Baseline stdout/stderr: `benchmarks/raw/freetoken-v013-comparison-2026-09-18/server.stdout.log` and `server.stderr.log`.
- Patched stdout/stderr: `benchmarks/raw/checkpoint-handoff-fix-2026-09-18/server.stdout.log` and `server.stderr.log`.
- Same directory: `probe.py`, `baseline.json`, `patched.json`, `tests.xml`, `verify_regression.py`, and `boundary-full-output.json`.
- Fork rebuild stdout/stderr and results: `benchmarks/raw/freetoken-fork-runtime-2026-09-18/`.

Local filesystem links and ignored raw files are not available to GitHub
reviewers automatically; attach the relevant files to the issue/PR yourself.

## Contribution guide and remaining limits

The [contribution guide](https://github.com/FlashML-org/FreeToken/blob/main/CONTRIBUTING.md)
requires human ownership of AI-assisted work, a focused PR linked to its issue,
hardware/model/command evidence, and a regression test for a bug fix. Performance
claims need appropriate A/B measurements. It also requests checking existing
reports and testing current main.

The source/base comparison and regression checks are complete. Searches found
no exact matching report; no issue has been created or linked yet. The FAQ and
roadmap were also read. The proposed title follows the repository convention:
`fix(scheduler): preserve pending hybrid checkpoint across prefill chunks`.

Disclose these boundaries:

- The short probe's returned choices match exactly before/after and cold/warm.
- A longer boundary probe returned the same final `OK.` answer but different
  reasoning text (44 versus 69 completion tokens). Its strict equality check
  failed. The cause remains unresolved; numerical differences are a hypothesis.
- An ordinary longer-prompt control returned identical complete cold/warm output.
- The fix does not establish the cause of DSH's changing miss frequency, cover
  arbitrary prompt rewrites, add periodic retained snapshots, or validate every
  model using the shared scheduler.
- The human contributor's understanding/review is still their responsibility;
  agent-run hardware tests should be described accurately as such.
