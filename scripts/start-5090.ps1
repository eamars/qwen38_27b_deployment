[CmdletBinding()]
param(
    [int]$Port = 8080,
    [int]$ContextSize = 126976,
    [ValidateSet(4,5,7)][int]$DraftNMax = 5
)

$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$runtime = Join-Path $workspace 'runtime\llama.cpp-dflash2\build-dflash2\bin\Release\llama-server.exe'
$target = Join-Path $workspace 'models\Qwen3.8-27B-UD-Q6_K_M.gguf'
$draft = Join-Path $workspace 'models\Qwen3.8-27B-DFlash2-Q4_K_M.gguf'
$expectedUuid = 'GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0'

if (-not (Test-Path -LiteralPath $runtime)) { throw "Runtime not built: $runtime" }
if (-not (Test-Path -LiteralPath $target)) { throw "Target model is missing: $target" }
if (-not (Test-Path -LiteralPath $draft)) { throw "DFlash2 drafter is missing: $draft" }
$gpu = @(nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader,nounits | Where-Object { $_ -match 'RTX 5090' -and $_ -match $expectedUuid })
if ($gpu.Count -ne 1) { throw "Expected RTX 5090 UUID $expectedUuid exactly once; found $($gpu.Count) matches." }
$oldVisible = $env:CUDA_VISIBLE_DEVICES
$env:CUDA_VISIBLE_DEVICES = $expectedUuid
Write-Host "Validated RTX 5090 UUID $expectedUuid and restricted the child process to that UUID as runtime CUDA0."
Write-Host "Starting only after this script is explicitly run; model load begins with llama-server below."

try {
    & $runtime `
        --model $target `
        --spec-draft-model $draft `
        --alias 'qwen3.8-27b-dflash2-5090' `
        --host 127.0.0.1 `
        --port $Port `
        --device CUDA0 `
        --spec-draft-device CUDA0 `
        --split-mode none `
        --gpu-layers all `
        --spec-draft-ngl all `
        --ctx-size $ContextSize `
        --parallel 1 `
        --kv-unified `
        --flash-attn on `
        --cache-type-k q8_0 `
        --cache-type-v q8_0 `
        --spec-type draft-dflash `
        --spec-draft-n-max $DraftNMax `
        --spec-draft-type-k f16 `
        --spec-draft-type-v f16 `
        --batch-size 1024 `
        --ubatch-size 256 `
        --fit off `
        --no-mmproj `
        --no-context-shift `
        --jinja `
        --reasoning auto `
        --reasoning-preserve `
        --metrics
} finally {
    if ($null -eq $oldVisible) { Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue } else { $env:CUDA_VISIBLE_DEVICES = $oldVisible }
}
