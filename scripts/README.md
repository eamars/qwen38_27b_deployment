# Maintained scripts

The scripts directory contains only maintained operational helpers. The
canonical launch and profiling entry points are:

| Purpose | Script |
|---|---|
| Qwen RTX 5090 launch/stop | `start-qwen27b-5090.ps1` |
| Qwen RTX 4090 launch/stop | `start-qwen27b-4090.ps1` |
| Shared Qwen + Gemma server | `start-kazusa-models.ps1` |
| Gemma RTX 5090 experiment | `start-gemma4-5090.ps1` |
| Gemma RTX 4090 experiment | `start-gemma4-4090.ps1` |
| Qwen API smoke/sustained checks | `profile-api.py` |
| Qwen tokenizer-calibrated deep context | `profile-deep-context.py` |
| GPU memory sampling | `profile-vram.ps1` |
| Gemma MTP comparison | `profile-gemma4-mtp.py`, `profile-gemma4-4090.py` |
| Gemma short/long combined comparison | `profile-gemma4-combined.py` |
| Runtime/model/GPU preflight | `check-runtime.ps1` |

Setup and inventory helpers are also retained because they produce reproducible
state rather than launch an alternative runtime:

- `download-models.ps1` — fetches the Qwen deployment set.
- `stage-gemma4-assets.ps1` and `download-gemma4-persona-ranged.ps1` — stages
  the isolated Gemma experiment.
- `collect-host-inventory.ps1` — refreshes `docs/host-inventory.md`.
- `record-model-manifest.ps1` — refreshes `docs/models.md`.

`start-kazusa-models.ps1` launches one shared `llama-server` router with the
`qwen27b-5090` and `gemma4-4090` profiles. Their model-specific settings are
hard-coded in the launcher and emitted only to a temporary preset while the
server runs. There is no separate `kazusa-models.ini` file; the temporary
preset is deleted when the server exits.

The unvalidated two-slot Qwen launcher remains removed. Cleanup and restoration
history is recorded in [docs/history.md](../docs/history.md).
