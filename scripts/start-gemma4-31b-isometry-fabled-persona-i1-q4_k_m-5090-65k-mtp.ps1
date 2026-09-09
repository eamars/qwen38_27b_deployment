[CmdletBinding()]
param(
    [int]$Port = 8082,
    [int]$ContextSize = 65536,
    [ValidateRange(1, 16)][int]$MtpNMax = 3,
    [ValidateRange(1, 4096)][int]$BatchSize = 512,
    [ValidateRange(1, 4096)][int]$UbatchSize = 128,
    [string]$BindAddress = '0.0.0.0',
    [switch]$NoMtp,
    [switch]$DryRun,
    [switch]$Stop
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runtime = Join-Path $workspace 'runtime\llama.cpp-dflash2\build-dflash2\bin\Release\llama-server.exe'
$target = Join-Path $workspace 'models\Gemma-4-31B-Isometry-Fabled-Persona.i1-Q4_K_M.gguf'
$mtpHead = Join-Path $workspace 'models\mtp-gemma-4-31B-it-Q8_0.gguf'
$expectedUuid = 'GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0'

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
                $portMatch.Success -and ([int]$portMatch.Groups['port'].Value -eq $Port)
            }
    )

    if ($processes.Count -eq 0) {
        Write-Host "No managed Gemma 4 llama-server process found on port $Port."
        exit 0
    }

    foreach ($process in $processes) {
        Stop-Process -Id $process.ProcessId
        Write-Host "Stopped Gemma 4 llama-server PID $($process.ProcessId) on port $Port."
    }
    exit 0
}

if (-not (Test-Path -LiteralPath $runtime -PathType Leaf)) {
    throw "Pinned llama.cpp runtime is missing: $runtime"
}
if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
    throw "The required Isometry-Fabled-Persona target is missing: $target"
}
if (-not $NoMtp -and -not (Test-Path -LiteralPath $mtpHead -PathType Leaf)) {
    throw "The project-local Google MTP drafter is missing: $mtpHead"
}
if ($ContextSize -lt 1) { throw 'ContextSize must be positive.' }
if ($UbatchSize -gt $BatchSize) { throw 'UbatchSize cannot exceed BatchSize.' }

$mode = if ($NoMtp) { 'target-only' } else { 'google-mtp' }
$alias = "gemma4-31b-isometry-fabled-persona-5090-$mode"

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
        '--spec-draft-type-k', 'f16',
        '--spec-draft-type-v', 'f16'
    )
}
$arguments += @(
    '--alias', $alias,
    '--host', $BindAddress,
    '--port', "$Port",
    '--device', 'CUDA0',
    '--split-mode', 'none',
    '--gpu-layers', 'all',
    '--ctx-size', "$ContextSize",
    '--parallel', '1',
    '--kv-unified',
    '--flash-attn', 'on',
    '--cache-type-k', 'q8_0',
    '--cache-type-v', 'q8_0',
    '--batch-size', "$BatchSize",
    '--ubatch-size', "$UbatchSize",
    '--fit', 'off',
    '--no-mmproj',
    '--no-context-shift',
    '--jinja',
    '--reasoning', 'auto',
    '--reasoning-preserve',
    '--metrics'
)

$displayCommand = ($arguments | ForEach-Object { Quote-CommandArgument ([string]$_) }) -join ' '
Write-Host "Gemma 4 mode: $mode"
Write-Host "Target: $target"
if (-not $NoMtp) { Write-Host "MTP head: $mtpHead" }
Write-Host "Runtime: $runtime"
Write-Host "RTX 5090 UUID: $expectedUuid"

if ($DryRun) {
    Write-Host 'Dry run only: no GPU validation, server process, or model load was started.'
    Write-Host "CUDA_VISIBLE_DEVICES=$expectedUuid & `"$runtime`" $displayCommand"
    exit 0
}

$nvidia = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $nvidia) { throw 'nvidia-smi.exe was not found in PATH.' }
$gpu = @(& $nvidia.Source --query-gpu=index,name,uuid --format=csv,noheader,nounits |
    Where-Object { $_ -match 'RTX 5090' -and $_ -match [regex]::Escape($expectedUuid) })
if ($gpu.Count -ne 1) {
    throw "Expected RTX 5090 UUID $expectedUuid exactly once; found $($gpu.Count) matches."
}

$oldVisible = $env:CUDA_VISIBLE_DEVICES
$env:CUDA_VISIBLE_DEVICES = $expectedUuid
Write-Host "Validated RTX 5090 UUID $expectedUuid and restricted the child process to that UUID as runtime CUDA0."
Write-Host "Binding Gemma 4 llama-server to $BindAddress`:$Port."
Write-Host 'Starting llama-server; model loading begins with the command below.'

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
