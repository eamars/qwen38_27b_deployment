[CmdletBinding()]
param(
    [int]$Port = 8083,
    [ValidateRange(1, 1000000)][int]$ContextSize = 65536,
    [ValidateRange(1, 16)][int]$MtpNMax = 3,
    [ValidateSet('f32', 'f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0', 'q5_1')]
    [string]$CacheTypeK = 'q8_0',
    [ValidateSet('f32', 'f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0', 'q5_1')]
    [string]$CacheTypeV = 'q8_0',
    [ValidateSet('f32', 'f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0', 'q5_1')]
    [string]$MtpCacheTypeK = 'q8_0',
    [ValidateSet('f32', 'f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0', 'q5_1')]
    [string]$MtpCacheTypeV = 'q8_0',
    # Gemma 4 vision inputs can exceed the text-only launcher's 128-token ubatch.
    [ValidateRange(1, 4096)][int]$BatchSize = 512,
    [ValidateRange(1, 4096)][int]$UbatchSize = 512,
    [string]$BindAddress = '0.0.0.0',
    [switch]$NoMtp,
    [switch]$NoVision,
    [switch]$DryRun,
    [switch]$Stop
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runtime = Join-Path $workspace 'runtime\llama.cpp-dflash2\build-dflash2\bin\Release\llama-server.exe'
$modelRoot = Join-Path $workspace 'models\hauhaucs-gemma4-qat-uncensored-balanced-mtp'
$target = Join-Path $modelRoot 'Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf'
$mtpHead = Join-Path $modelRoot 'mtp-gemma-4-31B-it.gguf'
$mmproj = Join-Path $modelRoot 'mmproj-Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf'
$expectedUuid = 'GPU-eed52936-813f-8d68-1654-bfb56cb42bc3'
$expectedRevision = '9654466e82d83f5ebfe1518a369bc5900873abb1'
$expectedTargetBytes = 18687062176
$expectedTargetSha256 = '71667F9E601A4B914A98425C59150B731F6E15D260D661DBD1F1EE07469FC7DB'
$expectedMtpBytes = 279954368
$expectedMtpSha256 = 'B5C4E583FC5982439080114BBC1B7EDAEC361F9D4C9193D6BED606A3DE401B62'
$expectedMmprojBytes = 1200726016
$expectedMmprojSha256 = '7BEF0D0FB3E85FC2941EC5F1C375FEBF3742645F158132A43CED557093AEA841'

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
        throw "$Label is missing: $Path"
    }
    $info = Get-Item -LiteralPath $Path
    if ($info.Length -ne $ExpectedBytes) {
        throw "$Label has unexpected size $($info.Length) bytes; expected $ExpectedBytes bytes: $Path"
    }
    $hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($hash -ne $ExpectedSha256) {
        throw "$Label SHA256 mismatch: got $hash, expected $ExpectedSha256"
    }
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
        Write-Host "No managed HauhauCS Gemma 4 llama-server process found on port $Port."
        exit 0
    }

    foreach ($process in $processes) {
        Stop-Process -Id $process.ProcessId
        Write-Host "Stopped HauhauCS Gemma 4 llama-server PID $($process.ProcessId) on port $Port."
    }
    exit 0
}

if (-not (Test-Path -LiteralPath $runtime -PathType Leaf)) {
    throw "Pinned llama.cpp runtime is missing: $runtime"
}
if ($Port -lt 1 -or $Port -gt 65535) { throw 'Port must be between 1 and 65535.' }
if ($ContextSize -lt 1) { throw 'ContextSize must be positive.' }
if ($UbatchSize -gt $BatchSize) { throw 'UbatchSize cannot exceed BatchSize.' }

Assert-Artifact $target $expectedTargetBytes $expectedTargetSha256 'HauhauCS target'
if (-not $NoMtp) {
    Assert-Artifact $mtpHead $expectedMtpBytes $expectedMtpSha256 'HauhauCS MTP drafter'
}
if (-not $NoVision) {
    Assert-Artifact $mmproj $expectedMmprojBytes $expectedMmprojSha256 'HauhauCS vision projector'
}

$mode = if ($NoMtp) { 'target-only' } else { 'qat-mtp' }
$visionMode = if ($NoVision) { 'text-only' } else { 'vision' }
$alias = "gemma4-31b-hauhaucs-balanced-q4-k-m-65k-$mode-$visionMode"

$arguments = @(
    '--model', $target
)
if (-not $NoVision) {
    $arguments += @('--mmproj', $mmproj)
}
if (-not $NoMtp) {
    $arguments += @(
        '--spec-draft-model', $mtpHead,
        '--spec-type', 'draft-mtp',
        '--spec-draft-n-max', "$MtpNMax",
        '--spec-draft-device', 'CUDA0',
        '--spec-draft-ngl', 'all',
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
    '--load-mode', 'mmap',
    '--tensor-read-lazy', 'on',
    '--ctx-checkpoints', '0',
    '--ctx-size', "$ContextSize",
    '--parallel', '1',
    '--kv-unified',
    '--flash-attn', 'on',
    '--cache-type-k', $CacheTypeK,
    '--cache-type-v', $CacheTypeV,
    '--batch-size', "$BatchSize",
    '--ubatch-size', "$UbatchSize",
    '--fit', 'off',
    '--no-context-shift',
    '--jinja',
    '--reasoning', 'auto',
    '--metrics'
)

$displayCommand = ($arguments | ForEach-Object { Quote-CommandArgument ([string]$_) }) -join ' '
Write-Host "HauhauCS Gemma 4 mode: $mode; input mode: $visionMode"
Write-Host "Target: $target"
if (-not $NoMtp) { Write-Host "MTP drafter: $mtpHead" }
if (-not $NoVision) { Write-Host "Vision projector: $mmproj" }
Write-Host "Hugging Face revision: $expectedRevision"
Write-Host "Runtime: $runtime"
Write-Host "RTX 4090 UUID: $expectedUuid"
Write-Host "Context: $ContextSize; target KV: K=$CacheTypeK V=$CacheTypeV; draft KV: K=$MtpCacheTypeK V=$MtpCacheTypeV"
Write-Host 'Memory policy: mmap model loading, lazy tensor reads, and context checkpoints disabled.'

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
$gpu = @(& $nvidia.Source --query-gpu=index,name,uuid --format=csv,noheader,nounits |
    Where-Object { $_ -match 'RTX 4090' -and $_ -match [regex]::Escape($expectedUuid) })
if ($gpu.Count -ne 1) {
    throw "Expected RTX 4090 UUID $expectedUuid exactly once; found $($gpu.Count) matches."
}

$oldVisible = $env:CUDA_VISIBLE_DEVICES
$env:CUDA_VISIBLE_DEVICES = $expectedUuid
Write-Host "Validated RTX 4090 UUID $expectedUuid and restricted the child process to that UUID as runtime CUDA0."
Write-Host "Binding HauhauCS Gemma 4 llama-server to $BindAddress`:$Port."
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
