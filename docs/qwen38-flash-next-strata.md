# Qwen3.8 Flash-Next on Strata (RTX 5090, native Windows)

The shared Flash endpoint, port **1919**, serves the OrcaRouter Qwen3.8-Flash-Next uncensored ModelOpt NVFP4
checkpoint through the [sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) fork of Strata. Asuna,
Xiaoman and the DSH host at `192.168.2.10` use it live. This page is the current operating record. The full
per-upgrade narratives from 2026-10-03 to 2026-11-23 are kept verbatim in
[the archived runbook](archive/qwen38-flash-next-strata-2026-10-03-to-11-23.md).

**Current state (2026-11-23):** runtime 0.2.15-nvfp4.7 (engine 0.1.61), MTP with Basalt's drafting, KV streaming,
one request at a time, conversation cache 24 GiB / 16 slots. Chinese / English chat decodes at 215 / 219 tok/s, a
128K prompt reads at 11,056 tok/s. See [Benchmarks per update](#benchmarks-per-update).

## Configuration in force

Written by `scripts/start-qwen38-flash-next-uncensored-strata-vision.ps1` into
`runtime/strata-nvfp4/config/workspace-1919.json` at every start. Snapshot of the 2026-11-23 start:
[strata-0215-config-1919.json](../benchmarks/qwen38_flash_next/2026-11-23/strata-0215-config-1919.json).

| Setting | Value |
|---|---|
| Endpoint | `http://127.0.0.1:1919/v1`, bound to `0.0.0.0`; LAN clients use `http://192.168.2.13:1919/v1` |
| Served model name | `qwen38-next-uncensored-strata-vision`; other names are accepted (Asuna sends its own model ID) |
| GPU | RTX 5090 alone, UUID `GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0`; the 4090 is not used |
| Context | 262,144 tokens (`Native256K`); `-Profile Short4K` gives 8,192 |
| Active requests | 1 (`parallel` 1); further requests queue |
| K/V | INT8, KV streaming: 32,768 cells per attention layer in VRAM, the whole K/V in 3.09 GiB of pinned RAM |
| Experts | 63.28 GiB loaded into RAM, part of it pinned for the GPU; automatic GPU expert cache (8,367 slots); VRAM reserve 2,048 MiB |
| Drafter | MTP: Basalt's draft layer (`mtp/rt-tuned`), `--spec 8 --mtp-head proxy --mtp-cum auto`, CJK draft vocabulary (124,737 tokens), `"drafting": "mtp"` |
| Conversation cache | 24,576 MiB host RAM, 16 parked conversations, physical RAM floor 2,560 MiB |
| Vision | CPU encoder (`mmproj-f32.gguf`), at most 1,024 image tokens per picture |
| RTX 50 kernels | the fork's 33 default speed switches; `STRATA_NVFP4_SM120=0` in the config's `env` turns them off |
| Sampling | the model card's per thinking state when a request names none (since 0.2.0): thinking 1.0 / 0.95 / top-k 20; no thinking 0.7 / 0.8 / 20, presence penalty 1.5 |

Server options the launcher sets, each for a reason:

| Option | Why |
|---|---|
| `"reasoning_close_retry": true` | A reply that ends inside its thinking with no answer and no tool call is closed once and continued, so an agent turn gets an answer instead of empty content (upstream #1053). |
| `"undeclared_tool_calls": "call"` | In a request that declares tools, a well-formed call to an undeclared tool reaches the client as a call (0.2.0's behaviour) instead of text that ends the agent's turn. A request that declares no tools gets such a call back as text, on purpose (owner, 2026-11-12); Asuna owns the race that sent tool-less requests. |
| `"drafting": "mtp"` | Keeps the launcher's drafter arguments as written. The fork's drafting policy misreads this model's GGUF header ([Niko1221/Strata#6695](https://github.com/Niko1221/Strata/issues/6695)); without the key its Auto could rewrite them if that is ever fixed. |
| `"update_check": false` | The daily notice names `update.cmd`, which unzips over `serve\` and drops the local patches. Upgrades go through `setup-strata-runtime.ps1`. |
| `"anthropic_thinking": "on_request"` | The Anthropic API thinks only when a request asks for it. |

Launcher switches: `-Stop`, `-DryRun`, `-NoMtp`, `-Port`, `-BindAddress 127.0.0.1` (local only), `-Parallel N`,
`-ConversationCacheMiB` / `-ConversationCacheSlots`, `-VramReserveMiB` (1,024–16,384), `-RamBudgetGiB` (the fork's
low-RAM mode), `-GpuMode LayerSplit | ExpertHelper` (measured slower; see [Decisions](#decisions-that-hold)). The
launcher runs in the foreground; the owner starts and stops 1919 (Asuna/Xiaoman depend on it). `-Stop` waits up to
30 s while the engine releases its pinned RAM. `STRATA_API_KEY` is optional; when set, clients must send it.

## Pinned inputs

| Input | Pin |
|---|---|
| Runtime | [strata-nvfp4 v0.2.15-nvfp4.7](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.2.15-nvfp4.7), commit `4442f2d53`, engine 0.1.61 |
| Release zip SHA-256 | `84bd404b27b8a3982538294d9547a25fd1c3e483c749856a275a9a8cabed5759` |
| Engine SHA-256 | `9bff87b4ee5a85c479d522fda95d8c48b649df1b74866598377e23adf78fae66` |
| Bundled llama.cpp converter | `3cf03257f219afbe7334045ff7c6a06ac68c627d` |
| Checkpoint | [jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4](https://huggingface.co/jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4) `f24d2b68ff2814f24455ae86717be276619b5664`: 220 files, 135,253,671,102 bytes ([manifest](qwen38-strata-checkpoint-manifest.json)) |
| MTP draft layer | Basalt's self-distilled `dense.bin` from [Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata) `346744eb`, SHA-256 `dfe18647…852d75`, Qwen Community License 1.0 |

Earlier runtimes, each upgraded in place: 0.1.37-nvfp4.2 (`69a60f5`, 10-03), 0.1.39-nvfp4.3 (`b17e5ef`, 10-06),
0.1.40.2-nvfp4.1 (`d147ca4f`, 10-08), 0.1.41-nvfp4.4 (`ec1ea2d0`, 10-10), 0.1.44.1-nvfp4.1 (`3fa66b43`, 10-15),
0.2.0-beta-nvfp4.3 (`57293b8e`, engine 0.1.45.3, 10-22), 0.2.13-nvfp4.1 (`b08a70078`, engine 0.1.59, 11-12). Their
release zips stay in `runtime/` for rollback.

## Files

```text
runtime/strata-nvfp4/                pinned release, upgraded in place (Git-ignored)
  engine/                            strata.exe, strata-vision.exe, CUDA DLLs, BUILD.json
  .venv-convert/  .venv-serve/       converter and server environments
  config/workspace-1919.json         written by the launcher
runtime/strata-nvfp4-v<version>-windows-x64.zip   every pinned release, for rollback
models/qwen38-flash-next-uncensored-strata/       (Git-ignored, 317 GiB with the checkpoint)
  checkpoint/  checkpoint-manifest.json           verified source, kept
  preparation.json                                completed preparation steps and output sizes
  orca-nvfp4.gguf  pack/                          NVFP4 experts, Q8/BF16 dense weights, tokenizer
  ple-fp8.gguf  token-embd-bf16.gguf              original FP8 PLE and BF16 embeddings
  mmproj-f32.gguf                                 CPU vision encoder
  mtp/rt/                                         the checkpoint's MTP layer and the CJK draft vocabulary
  mtp/rt-tuned/                                   Basalt's dense.bin + rt's experts.bin, draft_vocab.bin (hard links)
benchmarks/raw/qwen38-strata/                     raw evidence (Git-ignored); server/engine-1919.log is the live log
benchmarks/qwen38_flash_next/<date>/              curated results, one record per update (tracked)
```

## Local patches to the release

`setup-strata-runtime.ps1` applies two changes to the pinned server and refuses a runtime whose anchors moved.
Reruns skip a change already present.

1. **`thinking_token_budget` alias** (`serve/server.py`). pi-ai/DSH send the thinking budget under that name; the
   server reads only `reasoning_budget_tokens` and llama.cpp's `thinking_budget_tokens`. Without it a step can think
   through all of its output tokens. The launcher refuses to start a server without it. Exit: when upstream reads
   the name.
2. **strata-think-echo v5** (`serve/frontend.py`, `scripts/patches/strata-think-echo.py`). The model writes
   `</think>` as one token whether it ends its thinking or quotes the tag. The stock parser switches to the answer
   at the first one, so a quoted tag leaks the rest of the reasoning and the reply twice into `content` (upstream
   #537, #1053). The model's own close is `\n</think>\n\n` or `text</think>\n\n` (420 of 420 recorded closes followed
   a newline; no recorded quote was followed by a blank line), so the patch takes only that as the close while
   thinking, and decides a `</think>` alone on its line in the answer by what follows. Costs, seen in the fork's
   suite: a thinking reply's answer streams only after 512 characters or its end, a stop string inside that window
   stops late, `reasoning_tokens` counts the close's newline, and an output that ends on, or closes right after a
   quote with, a `</think>` is treated as a quote (this model has not produced either). Check at every upgrade:
   `scripts/patches/test-strata-think-echo.py <runtime> --stock <release.zip>` replays the 16 recorded leak shapes
   (whole, per character, in random chunks): v5 passes 48 of 48; the stock 0.2.15 parser still gets 6 of 16 wrong.
   Exit: when the stock parser gets none wrong.

Retired patches: the keyless LAN-bind change (through 0.2.0; 0.2.13 only warns) and strata-tool-tags (2026-11-12 to
11-23; the fork fixed [#24](https://github.com/sergqwer/strata-nvfp4/issues/24) in 0.2.15-nvfp4.1).
`scripts/patches/test-strata-tool-tags.py <runtime>` still replays those recorded tool calls through the installed
parser at every upgrade.

Known limitation, not patched: the server reads only `enable_thinking` and `reasoning_effort` from a request's
`chat_template_kwargs`. `preserve_thinking` therefore stays at the template's default, on: earlier turns' reasoning
stays in the prompt, which keeps the prefix stable for the conversation cache (89% of prompt tokens reused over 400
Asuna requests, 2026-10-08), and also keeps a stray thought the model may repeat until the conversation is compacted.

## Prepare, start, check

From the workspace root in PowerShell:

```powershell
.\scripts\setup-strata-runtime.ps1
.\runtime\strata-nvfp4\.venv-convert\Scripts\python.exe .\scripts\stage-qwen38-strata.py
python .\scripts\prepare-qwen38-strata.py
.\scripts\start-qwen38-flash-next-uncensored-strata-vision.ps1
```

- The installer downloads the pinned release, checks its SHA-256 and the engine's, and upgrades an earlier pinned
  runtime in place (virtual environments, configs and models stay).
- Staging verifies all 220 checkpoint files against the publisher's hashes and resumes interrupted downloads.
  Preparation converts with CPU PyTorch, records each completed step in `preparation.json`, reruns only missing
  ones, and fetches Basalt's draft layer (`mtp_tuned` step, hash-checked). Allow about 340 GB of disk.
- Smoke check (text, streaming, a tool call and its answer, a picture, 4K retrieval with a cached repeat):
  `python .\scripts\probe-qwen38-strata.py --base-url http://127.0.0.1:1919 --output <file>.json`.
  Long retrieval: add `--only-long --long-tokens 32000`.
- Live state: `GET /metrics` (engine version, drafter, expert slots, the last 500 requests with
  `first_engine_token_s`, conversation-cache counters); engine log `benchmarks/raw/qwen38-strata/server/engine-1919.log`.

## Upgrading

The procedure every update since 10-10 followed:

1. **Read** the fork's release notes and open issues, and upstream's for the same span.
2. **Stage** beside production while 1919 keeps serving: `setup-strata-runtime.ps1 -Stage` with the new pin (or
   unpack the zip to `runtime/strata-nvfp4-<version>`, add a `.venv-serve`, apply the two patches).
3. **Check without the GPU:** diff the converters `prepare-qwen38-strata.py` runs, `data/expert-profile.bin`,
   `data/draft_vocab.bin` and `requirements-serve.txt` (identical or unreachable for this checkpoint means the
   prepared assets are reused); check the patch anchors; run both replay tests with `--stock`; run the fork's `serve`
   suite stock and patched and account for every extra failure.
4. **Measure** with 1919 stopped by the owner:
   `python .\scripts\benchmark-qwen38-strata-ab.py --candidate strata-nvfp4-<version> --candidate-only --output benchmarks/raw/qwen38-strata/<date>/upgrade`.
   It runs on `127.0.0.1:1921` only (nothing on the LAN or on 1919 can reach it) from the production launcher's own
   config, about 6–8 minutes per arm: chat decode in Chinese and English, 4K and 128K prompt + 4K output, one and two
   requests at once, two agents switching 32K histories, smoke, `</think>` tags, 3 long-thinking and 12 exact-answer
   prompts. A plain upgrade is compared with the last record; A/B arms (`--candidate-set`, `--candidate-env`,
   `--candidate-args`, `--candidate-model`) only when a setting or model is being chosen. `--candidate-env` reaches
   the engine only; a variable the server reads must be set in the harness's own environment.
5. **Promote** in place with `setup-strata-runtime.ps1`, confirm the engine and patched `serve/` files are
   byte-identical to what was measured, rerun the harness without `--candidate`, and let the owner start 1919. Smoke
   check 1919.
6. **Record:** a curated JSON and the config snapshot in `benchmarks/qwen38_flash_next/<date>/`, a row in
   [Benchmarks per update](#benchmarks-per-update), the history entry, and every exit condition reached.

Rollback: stop 1919, re-extract the previous release zip over `runtime/strata-nvfp4`, restore the previous installer
pin, launcher and harness from Git, and run the installer (it reapplies that version's patches).

## Benchmarks per update

Production as adopted at each update, RTX 5090 alone, nothing else on the GPU, one trial per case (a few percent is
noise). Decode is the engine's tokens per second over the reply; prompt read is tokens per second.

| Date | Runtime (engine) | Drafter | Chat decode zh / en | 4K read / decode | 128K read / decode | One request at a time | Agent switch, s | Long thinking / exact | GPU expert slots | Record |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 10-03 | 0.1.37-nvfp4.2 | MTP w4, en+Cyrillic vocab | — | 1,905–1,994 / 125–129 | 5,883 / 117.8 | — | — | — | — | [performance](../benchmarks/qwen38_flash_next/2026-10-03/strata-performance.json) |
| 10-06 | 0.1.39-nvfp4.3 | MTP w4, + CJK vocab | 137.9 / 139.2 | 2,029–2,055 / 137 | 6,903 / 136.9 | — | — | — | 8,087 | [upgrade-tuning](../benchmarks/qwen38_flash_next/2026-10-06/strata-upgrade-tuning.json) |
| 10-08 | 0.1.40.2-nvfp4.1 | MTP w4 | 158.2 / 159.8 | 2,118–2,364 / 163–169 | 7,689 / 150.9 | — | — | — | 8,022 | [0140-upgrade](../benchmarks/qwen38_flash_next/2026-10-08/strata-0140-upgrade.json) |
| 10-10 | 0.1.41-nvfp4.4 | MTP w4 | 162.6 / 164.9 | 2,349 / 169.5 | 8,136 / 156.1 | — | 0.90 | 2/3 / 12/12 | 8,066 | [0141-ab](../benchmarks/qwen38_flash_next/2026-10-10/strata-0141-ab.json) |
| 10-15 | 0.1.44.1-nvfp4.1 | MTP w4 | 166.1 / 165.7 | 2,331 / 167.6 | 8,459 / 162.4 | 149.3 | 0.90 | 2/3 / 12/12 | 8,066 | [04411-upgrade-split](../benchmarks/qwen38_flash_next/2026-10-15/strata-04411-upgrade-split.json) |
| 10-22 | 0.2.0-beta-nvfp4.3 (0.1.45.3) | MTP w4 | 171.7 / 172.5 | 2,368 / 172.3 | 8,464 / 164.7 | 155.1 | 0.87 | 2/3 / 12/12 | 8,060 | [020beta3-upgrade](../benchmarks/qwen38_flash_next/2026-10-22/strata-020beta3-upgrade.json) |
| 11-12 | 0.2.13-nvfp4.1 (0.1.59) | MTP w4 | 170.7 / 175.0 | 2,345 / 167.7 | 8,450 / 160.7 | 155.7 | 0.89 | 1/3 / 12/12 | 7,950 | [0213-upgrade](../benchmarks/qwen38_flash_next/2026-11-12/strata-0213-upgrade.json) |
| **11-23** | **0.2.15-nvfp4.7 (0.1.61)** | **MTP + Basalt drafting and layer, w8** | **215.2 / 219.1** | **2,606 / 223.5** | **11,056 / 218.5** | **192.7** | **0.73** | **2/3 / 12/12** | **8,367** | [0215-drafters](../benchmarks/qwen38_flash_next/2026-11-23/strata-0215-drafters.json), [config](../benchmarks/qwen38_flash_next/2026-11-23/strata-0215-config-1919.json) |

How to read it:
- From 10-10 on every row comes from `benchmark-qwen38-strata-ab.py` with the same probes. 10-03 used a 4K / 4K and
  128K / 4K run measured at the client (2 trials); 10-06 and 10-08 used the engine's figures on 4K / 4K and 128K / 4K
  runs, so compare those three rows only roughly. The same runtime measured on different days has differed by up to
  8–12% (0.1.39 on 10-06 and 10-08); each record holds its own same-session comparison.
- Every row passed retrieval, smoke and `</think>` (12/12 since 10-08) and, from 10-10, the cache switch. Long
  thinking fails long-3 (bracket coding) on most rows; from 0.2.0 on, the server's `answer_room` closes a thinking
  that runs past ¾ of `max_tokens` with a wrap-up sentence, and the probe counts that as a fail.
- From 10-03 to 11-23: chat decode +56% (Chinese, against 10-06) and the 128K prompt read +88%. The largest steps:
  the CJK draft vocabulary (Chinese +25%, 10-06), 0.1.40.2's tensor-core kernels (+7–17%, 10-08), and 0.2.15's
  RTX 50 kernels and KV streaming (+14–25% decode, +30% at 128K) with Basalt's drafting (+3–9% more, 11-23).

## Decisions that hold

Each with the evidence that decided it; the archived runbook has the measurements.

- **One GPU, one active request.** Layer split over 5090 + 4090 and the expert helper were slower on every measure
  (10-03; re-measured 10-15 with `parallel` 2: −8% to −40%, two streams at once ~70 tok/s each because pipelined
  batch windows run no drafts). Two independent instances do not fit in RAM (63 GiB pinned arena each). Requests
  are served one at a time; a second request queues. Per-card pinning on this PC left both cards with no free VRAM,
  so both multi-GPU modes set `STRATA_ARENA_PIN_PER_STAGE=0`. `parallel` 2 on the 5090 alone was not measured (the
  fork measured −10% for one dialog).
- **Conversation cache, 24 GiB / 16 slots** (10-06). Two agents alternating histories need it: without parking
  every return reread the full history (2.3–2.5 s per return at 8K; a cold 128K read takes about 22 s); with it
  every return hit (28–58 tokens reread). Eight slots evicted 291 histories in live use while only 7–14 GB were parked, so slots were the limit.
  A history is matched by its exact rendered prefix: a changed system prompt, tool schema or history rewrite misses.
  Snapshots are lost on restart; under the RAM floor parked conversations give RAM back, oldest first.
- **CJK draft vocabulary** (10-06). The fork's subset held 27 of 55,328 Han tokens; with CJK added Chinese drafts
  2.04 tokens a round instead of 1.41 and decodes 25% faster, English unchanged.
- **OrcaRouter ModelOpt pack stays.** The fork's GPTQ of OrcaRouter added 1–2% on chat and failed more long-thinking
  prompts (10-10; the fork withdrew it 10-11). The original-Qwen GPTQ pack with the fork's uncensoring vector was a
  tie on speed and checks (11-12); the owner kept OrcaRouter with Qwen4 expected, and both GPTQ packs were deleted
  (their manifests stay in the raw folders).
- **Drafter: MTP with Basalt's drafting and Basalt's layer** (11-23). Same session, Chinese / English chat: plain
  MTP w4 203 / 200, Basalt drafting over this checkpoint's layer 209 / 214 (repeat 208 / 213), with Basalt's layer
  216 / 216 (77% of drafts accepted against 73%), DFlash2 170 / 196. The fork's suggested `--spec 6 --spec-min-p 0.7`
  was within noise of 4 / 0.5 on 10-10.
- **Pictures on the CPU encoder** (owner, 11-23). The fork's in-engine reader (`"in_engine": true` under `vision`)
  measured 631 against 1,352 ms to the first token for a 329-token picture on its PC; not tested here.
- **Tool calls in requests without tools come back as text** (owner, 11-12). Restoring 0.2.0's leniency would be a
  one-line local change (`declared_only` off when `"undeclared_tool_calls"` is `"call"`).

## Findings worth knowing

- **Broken tool-call arguments, 2026-11-07 to 11-11.** Four of Xiaoman's calls streamed arguments that were not
  JSON. In three the model dropped the `>` of `<parameter=NAME` or `</parameter`; in one it wrote its reply inside
  the `think` argument. With an open value the fork's EOS literal then wrote `<|im_end|>` into it up to 64 times.
  All four ran at 80K–175K tokens of context; why the slips started on 11-07 is not known. Reported as
  [sergqwer/strata-nvfp4#24](https://github.com/sergqwer/strata-nvfp4/issues/24) and fixed in 0.2.15-nvfp4.1 (the
  slipped tag is read as the tag; the EOS literal no longer fires there). The reply-inside-`think` case still fails
  at the client, which owns a retry.
- **First-token stalls, 2026-10-22.** Asuna waited ~69 s for a first token whose engine work took ~7 s: the elastic
  K/V grew from 8,192 to 122,880 cells and moved 529 experts, which the engine does not time. A second wait (28.5 s)
  was queueing behind a 181K-token request from the DSH host. From 10-20 to 10-21, 96 of 3,437 Asuna steps (2.8%)
  waited over 20 s with under 10K new tokens. 0.2.15's KV streaming removes the growth; see [Open items](#open-items).
- **Large pages.** Windows refuses large-page allocation (error 1450); the engine uses 4 KiB pages. The PC has a fixed
  64,000 MB page file, which the engine checks at start (it shrinks the expert cache when commit runs short).
- **LAN.** The launcher binds `0.0.0.0`; the DSH host at `.10` reaches `.13:1919` (its requests appear in the engine
  log). `.232` is the backup adapter's DHCP address, not a deployment address. `scripts/capture-strata-lan.ps1` is the
  ARP capture script from the 10-03 reachability investigation (evidence in `benchmarks/raw/qwen38-strata/network/`).
- **Client notes.** Asuna's providers and DSH send full, separate histories to the one endpoint; matching is by
  rendered tokens, not by client. Keep the system prompt, tool schema and their order stable within a history, keep
  `reasoning_content` and DSH's replay metadata in the history, and expect a full reread after compaction, persona
  edits, tool changes or image offloading.

## Open items

- **Did KV streaming end the long first-token stalls?** A one-off scheduled task, `strata-1919-stall-check`
  (2026-11-25 19:00 NZDT), reads 1919's `/metrics` and engine log. Rule: closed when under 0.5% of requests with
  under 10K new tokens wait over 20 s for the first token (queueing behind another request not counted) and no K/V
  growth moves experts. It records the verdict below and in
  `benchmarks/qwen38_flash_next/2026-11-25/stall-check.json`.
  - Result: not yet run.
- **[Niko1221/Strata#6695](https://github.com/Niko1221/Strata/issues/6695)**, the GGUF header reader (posted
  2026-11-23). Nothing here depends on it (`"drafting": "mtp"`). Exit: closed upstream, or no longer relevant when
  the drafter is next chosen.
