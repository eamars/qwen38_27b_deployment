[CmdletBinding()]
param(
    [int]$Port = 8083,
    [ValidateRange(1024, 262144)][int]$ContextSize = 262144,
    [ValidateRange(1, 16)][int]$MtpNMax = 3,
    [ValidateSet('f32', 'f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0', 'q5_1')]
    [string]$CacheTypeK = 'q8_0',
    [ValidateSet('f32', 'f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0', 'q5_1')]
    [string]$CacheTypeV = 'q8_0',
    [ValidateSet('f32', 'f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0', 'q5_1')]
    [string]$MtpCacheTypeK = 'q8_0',
    [ValidateSet('f32', 'f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0', 'q5_1')]
    [string]$MtpCacheTypeV = 'q8_0',
    [ValidateRange(1, 4096)][int]$BatchSize = 128,
    [ValidateRange(1, 4096)][int]$UbatchSize = 128,
    [string]$BindAddress = '0.0.0.0',
    [ValidateSet('on', 'off', 'auto')]
    [string]$Reasoning = 'on',
    [ValidateRange(-1, 32768)]
    [int]$ReasoningBudget = 64,
    [string]$ReasoningBudgetMessage = '',
    [switch]$NoMtp,
    [switch]$DryRun,
    [switch]$Stop
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runtime = Join-Path $workspace 'runtime\llama.cpp-dflash2\build-dflash2\bin\Release\llama-server.exe'
# Verified at 262144 context with Q8/Q8 KV, vision and MTP on the RTX 4090.
# See docs/gemma4-26b-4090-vision-mtp.md for probes and memory limitations.
# QAT Instruct target and matching MTP drafter from the same repository.
# Source: https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF
$qatModelRoot = Join-Path $workspace 'models\unsloth-gemma4-26b-qat'
$target = Join-Path $qatModelRoot 'gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf'
$mtpHead = Join-Path $qatModelRoot 'MTP\mtp-gemma-4-26B-A4B-it-Q8_0.gguf'
$mmproj = Join-Path $qatModelRoot 'mmproj-F16.gguf'
$expectedUuid = 'GPU-eed52936-813f-8d68-1654-bfb56cb42bc3'

function Get-FullPath([string]$Path) {
    return [System.IO.Path]::GetFullPath($Path)
}

function Quote-CommandArgument([string]$Value) {
    if ($Value -match '[\s"]') {
        return '"' + $Value.Replace('"', '\"') + '"'
    }
    return $Value
}

if ($Stop) {
    $runtimeFullPath = Get-FullPath $runtime
    $processes = @(
        Get-CimInstance Win32_Process -Filter "Name = 'llama-server.exe'" |
            Where-Object {
                $executablePath = if ($_.ExecutablePath) {
                    Get-FullPath $_.ExecutablePath
                } else {
                    $null
                }
                if (-not $executablePath -or $executablePath -ine $runtimeFullPath) {
                    return $false
                }

                $commandLine = [string]$_.CommandLine
                $portMatch = [regex]::Match($commandLine, '(?:^|\s)--port\s+(?<port>\d+)(?=\s|$)')
                $portMatch.Success -and ([int]$portMatch.Groups['port'].Value -eq $Port) -and
                    $commandLine.Contains($target)
            }
    )

    if ($processes.Count -eq 0) {
        Write-Host "No managed Gemma 4 RTX 4090 llama-server process found on port $Port."
        exit 0
    }

    foreach ($process in $processes) {
        Stop-Process -Id $process.ProcessId
        Write-Host "Stopped Gemma 4 RTX 4090 llama-server PID $($process.ProcessId) on port $Port."
    }
    exit 0
}

if (-not (Test-Path -LiteralPath $runtime -PathType Leaf)) {
    throw "Pinned llama.cpp runtime is missing: $runtime"
}
if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
    throw "The Gemma 4 26B-A4B target is missing: $target. Run scripts/stage-gemma4-26b-assets.ps1 first."
}
if (-not $NoMtp -and -not (Test-Path -LiteralPath $mtpHead -PathType Leaf)) {
    throw "The matching Gemma 4 26B-A4B Instruct QAT MTP drafter is missing: $mtpHead"
}
if (-not (Test-Path -LiteralPath $mmproj -PathType Leaf)) {
    throw "The matching vision projector is missing: $mmproj. Run scripts/stage-gemma4-26b-assets.ps1 first."
}
if ($Port -lt 1 -or $Port -gt 65535) { throw 'Port must be between 1 and 65535.' }
if ($ContextSize -lt 1) { throw 'ContextSize must be positive.' }
if ($UbatchSize -ne $BatchSize) { throw 'Gemma vision requires equal BatchSize and UbatchSize for non-causal image attention in this runtime.' }

$mode = if ($NoMtp) { 'target-only' } else { 'qat-mtp' }
$alias = "gemma4-26b-a4b-it-qat-vision-$ContextSize-$mode"

$arguments = @(
    '--model', $target
)
if (-not $NoMtp) {
    $arguments += @(
        '--spec-draft-model', $mtpHead,
        '--spec-type', 'draft-mtp',
        '--spec-draft-n-max', "$MtpNMax",
        '--spec-draft-device', 'CUDA0',
        '--spec-draft-ngl', 'all',
        '--spec-draft-override-tensor', '^token_embd\.weight$=CUDA0',
        '--spec-draft-type-k', $MtpCacheTypeK,
        '--spec-draft-type-v', $MtpCacheTypeV
    )
}
$arguments += @(
    '--alias', $alias,
    '--host', $BindAddress,
    '--port', "$Port",
    '--device', 'CUDA0',
    '--split-mode', 'none',
    '--gpu-layers', 'all',
    # --gpu-layers all alone leaves token embeddings CPU-mapped.
    '--override-tensor', '^token_embd\.weight$=CUDA0',
    '--load-mode', 'mmap',
    '--tensor-read-lazy', 'on',
    '--ctx-checkpoints', '6',
    '--cache-ram', '0',
    '--ctx-size', "$ContextSize",
    '--parallel', '1',
    '--kv-unified',
    '--flash-attn', 'on',
    '--cache-type-k', $CacheTypeK,
    '--cache-type-v', $CacheTypeV,
    '--batch-size', "$BatchSize",
    '--ubatch-size', "$UbatchSize",
    '--fit', 'off',
    '--mmproj', $mmproj,
    '--mmproj-offload',
    '--mmproj-device', 'CUDA0',
    '--image-max-tokens', '1120',
    '--no-context-shift',
    '--jinja',
    '--reasoning', $Reasoning,
    '--verbosity', '4',
    '--metrics'
)
if ($ReasoningBudget -ge 0) {
    $arguments += @('--reasoning-budget', "$ReasoningBudget")
}
if (-not [string]::IsNullOrWhiteSpace($ReasoningBudgetMessage)) {
    if ($ReasoningBudget -lt 0) {
        throw 'ReasoningBudgetMessage requires ReasoningBudget >= 0.'
    }
    $arguments += @('--reasoning-budget-message', $ReasoningBudgetMessage)
}

$displayCommand = ($arguments | ForEach-Object { Quote-CommandArgument ([string]$_) }) -join ' '
Write-Host "Gemma 4 mode: $mode"
Write-Host "Target: $target"
Write-Host "Vision projector: $mmproj"
if (-not $NoMtp) { Write-Host "MTP drafter: $mtpHead" }
Write-Host "Runtime: $runtime"
Write-Host "RTX 4090 UUID: $expectedUuid"
Write-Host "Context: $ContextSize; target KV: K=$CacheTypeK V=$CacheTypeV; draft KV: K=$MtpCacheTypeK V=$MtpCacheTypeV"
Write-Host "Reasoning: $Reasoning; budget: $ReasoningBudget tokens"
Write-Host 'Memory policy: mmap model loading, lazy tensor reads, and six context checkpoints. Checkpoint snapshots and staging still use host RAM.'

if ($DryRun) {
    Write-Host 'Dry run only: no GPU validation, server process, or model load was started.'
    Write-Host "CUDA_VISIBLE_DEVICES=$expectedUuid & `"$runtime`" $displayCommand"
    exit 0
}

$nvidia = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $nvidia) { throw 'nvidia-smi.exe was not found in PATH.' }
$gpu = @(& $nvidia.Source --query-gpu=index,name,uuid --format=csv,noheader,nounits |
    Where-Object { $_ -match 'RTX 4090' -and $_ -match [regex]::Escape($expectedUuid) })
if ($gpu.Count -ne 1) {
    throw "Expected RTX 4090 UUID $expectedUuid exactly once; found $($gpu.Count) matches."
}

$oldVisible = $env:CUDA_VISIBLE_DEVICES
$env:CUDA_VISIBLE_DEVICES = $expectedUuid
Write-Host "Validated RTX 4090 UUID $expectedUuid and restricted the child process to that UUID as runtime CUDA0."
Write-Host "Binding Gemma 4 llama-server to $BindAddress`:$Port."
Write-Host 'Starting llama-server with host-RAM-minimizing model loading; model loading begins with the command below.'

$exitCode = 0
try {
    & $runtime @arguments
    $exitCode = $LASTEXITCODE
} finally {
    if ($null -eq $oldVisible) {
        Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    } else {
        $env:CUDA_VISIBLE_DEVICES = $oldVisible
    }
}

if ($exitCode -ne 0) { exit $exitCode }
