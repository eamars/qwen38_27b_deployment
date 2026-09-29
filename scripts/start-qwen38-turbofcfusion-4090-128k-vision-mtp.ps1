[CmdletBinding()]
param(
    [int]$Port = 8081,
    [ValidateRange(1024, 262144)][int]$ContextSize = 131072,
    [ValidateRange(1, 16)][int]$MtpNMax = 3,
    [ValidateRange(1, 4096)][int]$BatchSize = 128,
    [ValidateRange(1, 4096)][int]$UbatchSize = 128,
    [string]$BindAddress = '0.0.0.0',
    [switch]$DryRun,
    [switch]$Stop
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runtime = Join-Path $workspace 'runtime\llama.cpp-dflash2\build-dflash2\bin\Release\llama-server.exe'
$modelRoot = Join-Path $workspace 'models\qwen38-turbo-fcfusion-mtp'
$target = Join-Path $modelRoot 'Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-LOW-MTP-IQ4_XS.gguf'
$mmproj = Join-Path $modelRoot 'mmproj-F16.gguf'
$expectedUuid = 'GPU-eed52936-813f-8d68-1654-bfb56cb42bc3'
$alias = 'qwen38-turbo-fcfusion-4090-vision-mtp'

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
    $targetFullPath = Get-FullPath $target
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
                    $commandLine.Contains($targetFullPath)
            }
    )

    if ($processes.Count -eq 0) {
        Write-Host "No managed Qwen3.8 Turbo Fable Cold Fusion server found on port $Port."
        exit 0
    }
    foreach ($process in $processes) {
        Stop-Process -Id $process.ProcessId
        Write-Host "Stopped Qwen3.8 Turbo Fable Cold Fusion PID $($process.ProcessId) on port $Port."
    }
    exit 0
}

if (-not (Test-Path -LiteralPath $runtime -PathType Leaf)) {
    throw "Pinned llama.cpp runtime is missing: $runtime"
}
if (-not (Test-Path -LiteralPath $target -PathType Leaf) -or -not (Test-Path -LiteralPath $mmproj -PathType Leaf)) {
    throw "Model assets are missing. Run scripts/stage-qwen38-turbofcfusion-assets.ps1 first."
}
if ($UbatchSize -ne $BatchSize) {
    throw 'Qwen3.8 vision requires equal BatchSize and UbatchSize for image prefill in this runtime.'
}

$arguments = @(
    '--model', $target,
    '--spec-type', 'draft-mtp',
    '--spec-draft-n-max', "$MtpNMax",
    '--spec-draft-device', 'CUDA0',
    '--spec-draft-ngl', 'all',
    '--spec-draft-override-tensor', '^token_embd\.weight$=CUDA0',
    '--spec-draft-type-k', 'f16',
    '--spec-draft-type-v', 'f16',
    '--alias', $alias,
    '--host', $BindAddress,
    '--port', "$Port",
    '--device', 'CUDA0',
    '--split-mode', 'none',
    '--gpu-layers', 'all',
    '--override-tensor', '^token_embd\.weight$=CUDA0',
    '--load-mode', 'mmap',
    '--tensor-read-lazy', 'on',
    '--ctx-checkpoints', '6',
    '--cache-ram', '0',
    '--ctx-size', "$ContextSize",
    '--parallel', '1',
    '--kv-unified',
    '--flash-attn', 'on',
    '--cache-type-k', 'q8_0',
    '--cache-type-v', 'q8_0',
    '--batch-size', "$BatchSize",
    '--ubatch-size', "$UbatchSize",
    '--fit', 'off',
    '--mmproj', $mmproj,
    '--mmproj-offload',
    '--mmproj-device', 'CUDA0',
    '--image-max-tokens', '1120',
    '--no-context-shift',
    '--jinja',
    '--reasoning', 'auto',
    '--reasoning-preserve',
    '--metrics'
)

$displayCommand = ($arguments | ForEach-Object { Quote-CommandArgument ([string]$_) }) -join ' '
Write-Host "Model: $target"
Write-Host "Vision projector: $mmproj"
Write-Host "Runtime: $runtime"
Write-Host "RTX 4090 UUID: $expectedUuid"
Write-Host "Context: $ContextSize; target KV: K=q8_0 V=q8_0; MTP: on (n-max=$MtpNMax); vision: on"
Write-Host "Batch/ubatch: $BatchSize/$UbatchSize; one slot; lazy mmap loading"

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
Write-Host "Binding llama-server to $BindAddress`:$Port."
Write-Host 'This starts model loading on the RTX 4090.'

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
