[CmdletBinding()]
param(
    [int]$Port = 8081,
    [ValidateRange(1024, 262144)][int]$ContextSize = 131072,
    [ValidateRange(1, 16)][int]$MtpNMax = 2,
    [ValidateRange(1, 4096)][int]$BatchSize = 128,
    [ValidateRange(1, 4096)][int]$UbatchSize = 128,
    [ValidatePattern('^(all|[0-9]+)$')][string]$GpuLayers = '14',
    [ValidateRange(1024, 24576)][int]$RequiredFreeVramMiB = 10000,
    [string]$BindAddress = '127.0.0.1',
    [switch]$DryRun,
    [switch]$Stop
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runtime = Join-Path $workspace 'runtime\llama.cpp-dflash2\build-dflash2\bin\Release\llama-server.exe'
$modelRoot = Join-Path $workspace 'models\qwen38-humanlike-chat-vision-mtp'
$target = Join-Path $modelRoot 'Qwen3.8-27B-Humanlike-Chat-IQ4_XS.gguf'
$mtpHead = Join-Path $modelRoot 'MTP\mtp-Qwen3.8-27B-Q4_0.gguf'
$mmproj = Join-Path $modelRoot 'mmproj-F16.gguf'
$expectedUuid = 'GPU-eed52936-813f-8d68-1654-bfb56cb42bc3'
$alias = 'qwen38-humanlike-chat-4090-vision-mtp'

function Get-FullPath([string]$Path) {
    return [System.IO.Path]::GetFullPath($Path)
}

function Quote-CommandArgument([string]$Value) {
    if ($Value -match '[\s"]') {
        return '"' + $Value.Replace('"', '\"') + '"'
    }
    return $Value
}

function Assert-Artifact([string]$Path, [long]$ExpectedBytes, [string]$ExpectedSha256, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label is missing: $Path. Run scripts/stage-qwen38-humanlike-chat-vision-mtp.ps1 first."
    }
    $info = Get-Item -LiteralPath $Path
    if ($info.Length -ne $ExpectedBytes) {
        throw "$Label size mismatch: got $($info.Length), expected $($ExpectedBytes): $Path"
    }
    $hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($hash -ne $ExpectedSha256) {
        throw "$Label SHA256 mismatch: got $hash, expected $($ExpectedSha256): $Path"
    }
}

if ($Stop) {
    $runtimeFullPath = Get-FullPath $runtime
    $targetFullPath = Get-FullPath $target
    $processes = @(
        Get-CimInstance Win32_Process -Filter "Name = 'llama-server.exe'" |
            Where-Object {
                $executablePath = if ($_.ExecutablePath) { Get-FullPath $_.ExecutablePath } else { $null }
                if (-not $executablePath -or $executablePath -ine $runtimeFullPath) { return $false }
                $commandLine = [string]$_.CommandLine
                $portMatch = [regex]::Match($commandLine, '(?:^|\s)--port\s+(?<port>\d+)(?=\s|$)')
                $portMatch.Success -and ([int]$portMatch.Groups['port'].Value -eq $Port) -and
                    $commandLine.Contains($targetFullPath)
            }
    )
    if ($processes.Count -eq 0) {
        Write-Host "No managed Humanlike Qwen3.8 server found on port $Port."
        exit 0
    }
    foreach ($process in $processes) {
        Stop-Process -Id $process.ProcessId
        Write-Host "Stopped Humanlike Qwen3.8 PID $($process.ProcessId) on port $Port."
    }
    exit 0
}

if (-not (Test-Path -LiteralPath $runtime -PathType Leaf)) {
    throw "Pinned llama.cpp runtime is missing: $runtime"
}
if ($Port -lt 1 -or $Port -gt 65535) { throw 'Port must be between 1 and 65535.' }
if ($UbatchSize -gt $BatchSize) { throw 'UbatchSize cannot exceed BatchSize.' }

Assert-Artifact $target 15095040512L '38B417B142E0D87C90FDA9393263D523BEF17EAFC241CE070AB03797396B6850' 'Humanlike IQ4_XS target'
Assert-Artifact $mtpHead 1369590656L '50D9CE5A6DA381BBCFB31061CF73DF94A90E6FAF8EFEDDEE379A9CB8F1501C6E' 'Qwen3.8 base MTP drafter'
Assert-Artifact $mmproj 927607488L 'CBB841A9EE0636B2EC172F5BB8DF2EA8DFEB01E90FE7C6126581D662A0B4E43E' 'Qwen3.8 vision projector'

$arguments = @(
    '--model', $target,
    '--mmproj', $mmproj,
    '--mmproj-offload',
    '--mmproj-device', 'CUDA0',
    '--spec-draft-model', $mtpHead,
    '--spec-type', 'draft-mtp',
    '--spec-draft-n-max', "$MtpNMax",
    '--spec-draft-device', 'CUDA0',
    '--spec-draft-ngl', 'all',
    '--spec-draft-type-k', 'q8_0',
    '--spec-draft-type-v', 'q8_0',
    '--alias', $alias,
    '--host', $BindAddress,
    '--port', "$Port",
    '--device', 'CUDA0',
    '--split-mode', 'none',
    '--gpu-layers', $GpuLayers,
    '--load-mode', 'mmap',
    '--tensor-read-lazy', 'on',
    '--cache-ram', '0',
    '--ctx-checkpoints', '0',
    '--ctx-size', "$ContextSize",
    '--parallel', '1',
    '--kv-unified',
    '--flash-attn', 'on',
    '--cache-type-k', 'q8_0',
    '--cache-type-v', 'q8_0',
    '--batch-size', "$BatchSize",
    '--ubatch-size', "$UbatchSize",
    '--fit', 'off',
    '--image-max-tokens', '1120',
    '--no-context-shift',
    '--jinja',
    '--reasoning', 'auto',
    '--reasoning-preserve',
    '--metrics'
)

$displayCommand = ($arguments | ForEach-Object { Quote-CommandArgument ([string]$_) }) -join ' '
Write-Host "Target: $target"
Write-Host "Base MTP drafter: $mtpHead"
Write-Host "Qwen3.8 vision projector: $mmproj"
Write-Host "Runtime: $runtime"
Write-Host "RTX 4090 UUID: $expectedUuid"
Write-Host "Context: $ContextSize; target KV: K=q8_0 V=q8_0; MTP draft: Q4_0 weights and Q8_0 KV (n-max=$MtpNMax); vision: on"
Write-Host "Batch/ubatch: $BatchSize/$UbatchSize; target GPU layers: $GpuLayers; one slot; initial bind: $BindAddress`:$Port"

if ($DryRun) {
    Write-Host 'Dry run only: no GPU validation, server process, or model load was started.'
    Write-Host "CUDA_VISIBLE_DEVICES=$expectedUuid & `"$runtime`" $displayCommand"
    exit 0
}

$listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
if ($listeners.Count -gt 0) {
    $owners = ($listeners | Select-Object -ExpandProperty OwningProcess -Unique) -join ', '
    throw "Port $Port is already in use by process ID(s): $owners. Stop the current server explicitly before swapping models."
}

$nvidia = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $nvidia) { throw 'nvidia-smi.exe was not found in PATH.' }
$gpu = @(& $nvidia.Source --query-gpu=name,uuid,memory.free --format=csv,noheader,nounits |
    Where-Object { $_ -match 'RTX 4090' -and $_ -match [regex]::Escape($expectedUuid) })
if ($gpu.Count -ne 1) { throw "Expected RTX 4090 UUID $expectedUuid exactly once; found $($gpu.Count) matches." }
$freeMiB = [int](($gpu[0] -split ',')[-1].Trim())
if ($freeMiB -lt $RequiredFreeVramMiB) {
    throw "RTX 4090 has only $freeMiB MiB free before launch; this profile requires at least $RequiredFreeVramMiB MiB."
}

$oldVisible = $env:CUDA_VISIBLE_DEVICES
$env:CUDA_VISIBLE_DEVICES = $expectedUuid
Write-Host "Validated RTX 4090 UUID $expectedUuid with $freeMiB MiB free (minimum $RequiredFreeVramMiB MiB) and restricted the child to this card as CUDA0."
Write-Host "Starting llama-server on $BindAddress`:$Port; watch VRAM during startup and image requests."

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
