[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$Port = 8080,
    [string]$BindAddress = '0.0.0.0',
    [switch]$DryRun,
    [switch]$Stop
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runtime = Join-Path $workspace 'runtime\llama.cpp-dflash2\build-dflash2\bin\Release\llama-server.exe'
$modelsDirectory = Join-Path $workspace 'models'

# Physical-to-runtime ordering. The router exposes the 5090 as CUDA0 and the
# 4090 as CUDA1, so each preset below uses the corresponding runtime device.
$qwenUuid = 'GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0'
$gemmaUuid = 'GPU-eed52936-813f-8d68-1654-bfb56cb42bc3'

$gemmaTarget = Join-Path $modelsDirectory 'Gemma-4-31B-Isometry-Fabled-Persona.i1-Q4_K_S.gguf'
$gemmaDraft = Join-Path $modelsDirectory 'mtp-gemma-4-31B-it-Q8_0.gguf'
$qwenTarget = Join-Path $modelsDirectory 'Qwen3.8-27B-UD-Q6_K_M.gguf'
$qwenDraft = Join-Path $modelsDirectory 'Qwen3.8-27B-DFlash2-Q4_K_M.gguf'

function Get-FullPath([string]$Path) {
    return [System.IO.Path]::GetFullPath($Path)
}

function Quote-CommandArgument([string]$Value) {
    if ($Value -match '[\s"]') {
        return '"' + $Value.Replace('"', '\"') + '"'
    }
    return $Value
}

function Get-PresetText {
    return @"
version = 1

[gemma4-4090]
model = models/Gemma-4-31B-Isometry-Fabled-Persona.i1-Q4_K_S.gguf
load-on-startup = true
device = CUDA1
split-mode = none
gpu-layers = all
ctx-size = 56320
parallel = 1
kv-unified = true
flash-attn = on
cache-type-k = q8_0
cache-type-v = q8_0
spec-draft-model = models/mtp-gemma-4-31B-it-Q8_0.gguf
spec-type = draft-mtp
spec-draft-n-max = 3
spec-draft-device = CUDA1
spec-draft-ngl = all
spec-draft-type-k = q8_0
spec-draft-type-v = q8_0
batch-size = 256
ubatch-size = 128
fit = off
no-mmproj = true
no-context-shift = true
jinja = true
reasoning = auto
reasoning-preserve = true

[qwen27b-5090]
model = models/Qwen3.8-27B-UD-Q6_K_M.gguf
load-on-startup = true
device = CUDA0
split-mode = none
gpu-layers = all
ctx-size = 126976
parallel = 1
kv-unified = true
flash-attn = on
cache-type-k = q8_0
cache-type-v = q8_0
spec-draft-model = models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
spec-type = draft-dflash
spec-draft-n-max = 5
spec-draft-device = CUDA0
spec-draft-ngl = all
spec-draft-type-k = f16
spec-draft-type-v = f16
batch-size = 1024
ubatch-size = 256
fit = off
no-mmproj = true
no-context-shift = true
jinja = true
reasoning = auto
reasoning-preserve = true
"@
}

if ($Stop) {
    $runtimeFullPath = Get-FullPath $runtime
    $processes = @(
        Get-CimInstance Win32_Process -Filter "Name = 'llama-server.exe'" |
            Where-Object {
                $executablePath = if ($_.ExecutablePath) { Get-FullPath $_.ExecutablePath } else { $null }
                if (-not $executablePath -or $executablePath -ine $runtimeFullPath) { return $false }
                $commandLine = [string]$_.CommandLine
                $portMatch = [regex]::Match($commandLine, '(?:^|\s)--port\s+(?<port>\d+)(?=\s|$)')
                $portMatch.Success -and ([int]$portMatch.Groups['port'].Value -eq $Port)
            }
    )

    if ($processes.Count -eq 0) {
        Write-Host "No managed Kazusa model server found on port $Port."
        exit 0
    }

    foreach ($process in $processes) {
        Stop-Process -Id $process.ProcessId
        Write-Host "Stopped Kazusa model server PID $($process.ProcessId) on port $Port."
    }
    exit 0
}

if (-not (Test-Path -LiteralPath $runtime -PathType Leaf)) {
    throw "Pinned llama.cpp runtime is missing: $runtime"
}
foreach ($model in @($gemmaTarget, $gemmaDraft, $qwenTarget, $qwenDraft)) {
    if (-not (Test-Path -LiteralPath $model -PathType Leaf)) {
        throw "Required model file is missing: $model"
    }
}

$presetPath = Join-Path ([System.IO.Path]::GetTempPath()) ("kazusa-models-{0}.tmp" -f [guid]::NewGuid())
$arguments = @(
    '--models-preset', $presetPath,
    '--models-max', '2',
    '--host', $BindAddress,
    '--port', "$Port",
    '--metrics'
)
$displayCommand = ($arguments | ForEach-Object { Quote-CommandArgument ([string]$_) }) -join ' '

Write-Host 'Kazusa single-server multi-model launcher'
Write-Host "Runtime: $runtime"
Write-Host "Port: $BindAddress`:$Port"
Write-Host "gemma4-4090 -> RTX 4090 ($gemmaUuid)"
Write-Host "qwen27b-5090 -> RTX 5090 ($qwenUuid)"
Write-Host 'The model preset is hard-coded in this launcher and written only to a temporary file while the server runs.'

if ($DryRun) {
    Write-Host 'Dry run only: no GPU validation, router process, or model load was started.'
    Write-Host "CUDA_VISIBLE_DEVICES=$qwenUuid,$gemmaUuid & `"$runtime`" $displayCommand"
    Write-Host '--- embedded model configuration ---'
    Get-PresetText | Write-Host
    exit 0
}

$nvidia = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $nvidia) { throw 'nvidia-smi.exe was not found in PATH.' }
$gpuRows = @(& $nvidia.Source --query-gpu=index,name,uuid --format=csv,noheader,nounits)
if (@($gpuRows | Where-Object { $_ -match 'RTX 5090' -and $_ -match [regex]::Escape($qwenUuid) }).Count -ne 1) {
    throw "Expected RTX 5090 UUID $qwenUuid exactly once in nvidia-smi output."
}
if (@($gpuRows | Where-Object { $_ -match 'RTX 4090' -and $_ -match [regex]::Escape($gemmaUuid) }).Count -ne 1) {
    throw "Expected RTX 4090 UUID $gemmaUuid exactly once in nvidia-smi output."
}

$listener = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
if ($listener.Count -gt 0) {
    throw "Port $Port is already in use; stop the existing server before starting Kazusa."
}

$oldVisible = $env:CUDA_VISIBLE_DEVICES
$env:CUDA_VISIBLE_DEVICES = "$qwenUuid,$gemmaUuid"
$exitCode = 0
try {
    Set-Content -LiteralPath $presetPath -Value (Get-PresetText) -Encoding UTF8
    Write-Host 'Validated both GPUs. The router exposes them as CUDA0=RTX 5090 and CUDA1=RTX 4090.'
    Write-Host 'Starting llama-server with both model profiles set to load on startup.'
    Push-Location $workspace
    try {
        & $runtime @arguments
        $exitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
} finally {
    if ($null -eq $oldVisible) {
        Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    } else {
        $env:CUDA_VISIBLE_DEVICES = $oldVisible
    }
    Remove-Item -LiteralPath $presetPath -Force -ErrorAction SilentlyContinue
}

if ($exitCode -ne 0) { exit $exitCode }
