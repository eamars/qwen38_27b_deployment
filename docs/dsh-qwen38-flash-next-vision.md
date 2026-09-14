# DSH: Qwen3.8 Flash-Next vision integration

This is the canonical runbook for adding or switching a vision-capable
Qwen3.8 Flash-Next model in DeepSeek Harness (DSH). It records both sides of
the integration:

1. The FreeToken server must start the vision tower and advertise the correct
   served model ID.
2. DSH must carry the model capabilities that FreeToken cannot expose through
   automatic OpenAI-compatible discovery.

The runtime details and checkpoint preparation remain in the
[official deployment note](qwen38-flash-next-freetoken.md) and the
[uncensored deployment note](qwen38-flash-next-uncensored.md). This document
is the source of truth for the DSH catalog and swap procedure.

## Current topology

| Component | Value |
|---|---|
| DSH host | `192.168.2.10` |
| DSH public RPC | `https://192.168.2.10/api/<channel>/<endpoint>` |
| FreeToken host | `192.168.2.13` |
| FreeToken API | `http://192.168.2.13:1919/v1` |
| DSH provider | `local-qwen38-flash` |
| provider API | `openai-completions` |
| provider settings namespace | `llm-pi-ai` |
| deployment port | `1919` |
| context | `262144` |

The public DSH RPC is used for catalog changes. SSH is not required. The
provider can contain both model IDs even though only one checkpoint is served
on port `1919` at a time.

## Required model entries

The two entries must remain distinct so a harness cannot confuse a
text-only model with a vision-capable model:

| Checkpoint | Launcher | Served model ID |
|---|---|---|
| Official NVFP4 | `start-qwen38-flash-next-freetoken-vision.ps1` | `qwen38-next-freetoken-vision` |
| Uncensored NVFP4 | `start-qwen38-flash-next-uncensored-freetoken-vision.ps1` | `qwen38-next-uncensored-freetoken-vision` |

Each DSH model entry must contain the following effective values:

```json
{
  "id": "qwen38-next-freetoken-vision",
  "name": "qwen38-next-freetoken-vision",
  "contextWindow": 262144,
  "input": ["text", "image"],
  "reasoningEfforts": {
    "low": "low",
    "medium": "medium",
    "xhigh": "xhigh"
  },
  "compat": {
    "supportsDeveloperRole": false,
    "chatTemplateKwargs": {},
    "chatTemplateArgs": {}
  }
}
```

Use the uncensored ID for the uncensored entry. The model-specific
`reasoningEfforts` map is what makes the DSH thinking-effort selector render.
The explicit `supportsDeveloperRole: false` is required because these
FreeToken endpoints reject an OpenAI `developer` message with:

```text
400: could not encode request: Unexpected message role.
```

The same compatibility flag is required for the Gemma4 entries served by the
third-party local providers. A missing flag is not equivalent to an explicit
false value for DSH request adaptation.

## Why the catalog must be edited manually

The FreeToken `/v1/models` response supplies the model ID, context length, and
reasoning efforts, but does not supply DSH's `input` modality or compatibility
metadata. DSH automatic discovery therefore returns only the basic model
identity and context. It cannot infer that the vision tower is enabled, and it
cannot infer that the backend does not accept the `developer` role.

Consequences of omitting a field:

| Missing catalog field | Observable symptom |
|---|---|
| `input: ["text", "image"]` | Image attachment/control is unavailable for the model |
| `reasoningEfforts` | Thinking-effort selector is missing |
| `compat.supportsDeveloperRole: false` | DSH can send `developer` messages and FreeToken returns HTTP 400 |
| `contextWindow: 262144` | DSH uses an incorrect context limit |

Keep the provider-level `defaultInput: ["text"]` unchanged if text-only models
also use the provider. Vision support is declared per model in the `input`
field.

## Runtime actions

The shared FreeToken runtime must include the upstream Qwen3.8 vision support
and the local uncensored checkpoint compatibility patches. The maintained
vision launchers already perform the required runtime change:

- text-only launchers pass `--text-model-only`;
- vision launchers omit `--text-model-only` and pass
  `--mm-encoder-weights host`;
- both retain `--reasoning-parser qwen3` and
  `--tool-call-parser qwen3_coder`;
- both use the same runtime, GPU, port, context settings, and process-group
  lifecycle.

Start exactly one model on port `1919` at a time:

```powershell
# Official vision checkpoint
.\scripts\start-qwen38-flash-next-freetoken-vision.ps1 -Profile Native256K

# Or, after stopping the official process, the uncensored checkpoint
.\scripts\start-qwen38-flash-next-freetoken-vision.ps1 -Stop
.\scripts\start-qwen38-flash-next-uncensored-freetoken-vision.ps1 -Profile Native256K
```

The matching `-Stop` launcher must be used when swapping checkpoints. Do not
run the official and uncensored processes simultaneously on port `1919`.

## DSH public-RPC procedure

### 1. Read the current settings revision

The settings namespace is read first so the mutation can be fenced against a
concurrent change:

```text
POST https://192.168.2.10/api/settings/describe
Content-Type: application/json

{
  "type": "client-request",
  "rpcId": "<new UUID>",
  "method": "settings/describe",
  "payload": { "args": {} }
}
```

Select the namespace `llm-pi-ai`, then inspect:

```text
value.providers.local-qwen38-flash.models
value.providers.local-gemma4-4090-only.models
value.providers.local-kazusa.models
revision
```

Before changing anything, preserve every existing model entry and every
provider field. Add or update only the intended model metadata.

### 2. Build the model entry

When the ID is already present, update its `name`, `contextWindow`, `input`,
`reasoningEfforts`, and `compat` fields. When adding a new ID, clone the
corresponding text entry so its runtime compatibility settings are preserved:

- clone `qwen38-next-freetoken` for the official vision entry;
- clone `qwen38-next-uncensored-freetoken` for the uncensored vision entry.

Then apply the vision values from the model-entry example above. Do not copy a
Gemma entry without adding the explicit `supportsDeveloperRole: false` flag.

### 3. Write the catalog atomically

DSH accepts a settings mutation through the same public RPC transport:

```text
POST https://192.168.2.10/api/settings/mutate
Content-Type: application/json

{
  "type": "client-request",
  "rpcId": "<new UUID>",
  "method": "settings/mutate",
  "payload": {
    "args": {
      "ns": "llm-pi-ai",
      "ops": [
        {
          "op": "set",
          "path": ["providers", "local-qwen38-flash", "models"],
          "value": "<complete read-modify-write model array>"
        }
      ],
      "expectedRevision": "<revision returned by describe>"
    }
  }
}
```

The model list is an array. Use a complete read-modify-write of the provider's
`models` array and the `expectedRevision` fence. Do not rely on an array-index
path such as `models/3/input`; DSH rejects that form for this settings schema.
If both a Qwen and a Gemma provider are being repaired in one operation, use a
second `set` operation for the second provider's complete model array.

### 4. Verify the catalog projection

Query the live model catalog after the mutation:

```text
POST https://192.168.2.10/api/session/modelCatalog
Content-Type: application/json

{
  "type": "client-request",
  "rpcId": "<new UUID>",
  "method": "session/modelCatalog",
  "payload": { "args": {} }
}
```

The `local-qwen38-flash` group must contain both vision IDs. Each must expose:

```text
reasoning.efforts = low, medium, xhigh
```

The default model is independent of the catalog entries. Do not change the
default just to add a model; the harness can select the ID explicitly.

## Verification checklist

Run these checks in order after a catalog change.

### DSH settings

- [ ] `input` is exactly `text,image` for both vision IDs.
- [ ] `contextWindow` is `262144` for both vision IDs.
- [ ] `reasoningEfforts` contains `low`, `medium`, and `xhigh`.
- [ ] `compat.supportsDeveloperRole` is explicitly `false`.
- [ ] The text-only IDs remain text-only and retain their own IDs.
- [ ] The settings revision advanced exactly once for the intended mutation.

### Live backend

With one vision launcher running:

```powershell
curl.exe http://192.168.2.13:1919/health
curl.exe http://192.168.2.13:1919/v1/models
```

Confirm that `/health` and `/v1/models` report the same served model ID as the
launcher. Confirm `context_length: 262144` and the runtime's supported
reasoning efforts.

### Text and image requests

Send a normal text request with `reasoning_effort` set to one of the catalog
values. For a vision probe, send an OpenAI-compatible content array containing
one text block and one `image_url` data URL or uploaded image. A successful
HTTP response proves that the runtime accepts the image payload; it does not
replace the DSH catalog checks.

Do not use a direct backend request containing a `developer` role as a success
test. The backend is expected to reject that role. The DSH compatibility flag
must cause DSH to adapt the request before it reaches FreeToken.

### Harness selection

- [ ] Select `qwen38-next-freetoken-vision` only while the official launcher is
  serving port `1919`.
- [ ] Select `qwen38-next-uncensored-freetoken-vision` only while the uncensored
  launcher is serving port `1919`.
- [ ] Confirm the request's `model` field matches `/v1/models`.
- [ ] Refresh the DSH model selector after catalog changes if the browser holds
  a stale projection.

## Current verified state

As of 2026-09-14, both Qwen vision entries are present in DSH and have been
verified through the public RPC:

| Model | Input | Context | Reasoning | Developer role |
|---|---|---:|---|---|
| `qwen38-next-freetoken-vision` | text, image | 262144 | low, medium, xhigh | disabled |
| `qwen38-next-uncensored-freetoken-vision` | text, image | 262144 | low, medium, xhigh | disabled |

The DSH settings revision at the last verification was `15`. The dedicated
Gemma4 provider entries and its `local-kazusa` Gemma4 entry also explicitly
disable the developer role.

