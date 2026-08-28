<#
.SYNOPSIS
Starts or stops the managed Qwen3.8-Flash-Next llama.cpp server.

.EXAMPLE
.\scripts\start-qwen38-flash-next.ps1

.EXAMPLE
.\scripts\start-qwen38-flash-next.ps1 -Stop

Stops only the llama-server process whose executable and port match this
launcher. Other Qwen and Gemma server processes are not affected.
#>
[CmdletBinding()]
param(
    [ValidateSet('Baseline', 'Mtp')]
    [string]$Profile = 'Baseline',
    [string]$RuntimePath = 'runtime\llama.cpp-qwen4exp\build\bin\Release\llama-server.exe',
    [string]$ModelPath = 'models\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf',
    [string]$MtpModelPath = 'models\Qwen3.8-Flash-Next-MTP-F16.gguf',
    [int]$ContextSize = 524288,
    [string]$TensorSplit = '38,10',
    [ValidateRange(0, 48)]
    [int]$CpuMoeLayers = 33,
    [ValidateSet('f16', 'bf16', 'q8_0')]
    [string]$CacheTypeK = 'f16',
    [ValidateSet('f16', 'bf16', 'q8_0')]
    [string]$CacheTypeV = 'f16',
    [int]$BatchSize = 2048,
    [int]$UbatchSize = 256,
    [ValidateRange(1, 16)]
    [int]$MtpNMax = 3,
    [ValidateSet('f16', 'bf16')]
    [string]$MtpPrecision = 'f16',
    [ValidateSet('CUDA0', 'CUDA1')]
    [string]$MtpDevice = 'CUDA0',
    [int]$Threads = 0,
    [int]$ThreadsBatch = 0,
    [string]$BindAddress = '0.0.0.0',
    [int]$Port = 8000,
    [string]$Gpu0Uuid = 'GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0',
    [string]$Gpu1Uuid = 'GPU-eed52936-813f-8d68-1654-bfb56cb42bc3',
    [switch]$DryRun,
    [switch]$Stop
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))

function Resolve-WorkspacePath {
    param(
        [Parameter(Mandatory)][string]$Value,
        [Parameter(Mandatory)][string]$Label
    )

    $candidate = if ([System.IO.Path]::IsPathRooted($Value)) {
        [System.IO.Path]::GetFullPath($Value)
    } else {
        [System.IO.Path]::GetFullPath((Join-Path $workspace $Value))
    }
    try {
        $candidateUri = [System.Uri]::new($candidate)
        $workspaceUri = [System.Uri]::new($workspace.TrimEnd('\') + '\')
        if (-not $candidateUri.AbsoluteUri.StartsWith($workspaceUri.AbsoluteUri, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "$Label must be inside the workspace: $candidate"
        }
    } catch [System.UriFormatException] {
        throw "$Label is not a valid path: $Value"
    }
    return $candidate
}

function Quote-CommandArgument {
    param([Parameter(Mandatory)][AllowEmptyString()][string]$Value)
    if ($Value -notmatch '[\s"]') { return $Value }
    return '"' + ($Value -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
}

function Stop-ManagedServer {
    param(
        [Parameter(Mandatory)][string]$Runtime,
        [Parameter(Mandatory)][int]$ServerPort
    )

    $runtimeFullPath = [System.IO.Path]::GetFullPath($Runtime)
    $processes = @(
        Get-CimInstance Win32_Process -Filter "Name = 'llama-server.exe'" |
            Where-Object {
                $executablePath = if ($_.ExecutablePath) {
                    [System.IO.Path]::GetFullPath($_.ExecutablePath)
                } else {
                    $null
                }
                if (-not $executablePath -or $executablePath -ine $runtimeFullPath) {
                    return $false
                }

                $commandLine = [string]$_.CommandLine
                $portMatch = [regex]::Match($commandLine, '(?:^|\s)--port\s+(?<port>\d+)(?=\s|$)')
                $portMatch.Success -and ([int]$portMatch.Groups['port'].Value -eq $ServerPort)
            }
    )

    if ($processes.Count -eq 0) {
        Write-Host "No managed Qwen3.8-Flash-Next llama-server process found on port $ServerPort."
        return
    }

    foreach ($process in $processes) {
        Stop-Process -Id $process.ProcessId
        Write-Host "Stopped Qwen3.8-Flash-Next llama-server PID $($process.ProcessId) on port $ServerPort."
    }
}

function Assert-GpuMapping {
    param(
        [Parameter(Mandatory)][string]$FirstUuid,
        [Parameter(Mandatory)][string]$SecondUuid
    )

    $rows = @(nvidia-smi.exe --query-gpu=index,name,uuid --format=csv,noheader,nounits 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi GPU query failed: $($rows -join "`n")"
    }

    $first = @($rows | Where-Object { $_ -match [regex]::Escape($FirstUuid) })
    $second = @($rows | Where-Object { $_ -match [regex]::Escape($SecondUuid) })
    if ($first.Count -ne 1 -or $first[0] -notmatch 'RTX 5090') {
        throw "Expected one RTX 5090 row with UUID $FirstUuid."
    }
    if ($second.Count -ne 1 -or $second[0] -notmatch 'RTX 4090') {
        throw "Expected one RTX 4090 row with UUID $SecondUuid."
    }
}

function Assert-RuntimeOptions {
    param(
        [Parameter(Mandatory)][string]$Runtime,
        [Parameter(Mandatory)][bool]$RequireMtp
    )

    $baselineOptions = @(
        '--load-mode',
        '--n-cpu-moe',
        '--split-mode',
        '--tensor-split',
        '--override-tensor',
        '--cache-type-k',
        '--cache-type-v',
        '--flash-attn',
        '--fit',
        '--metrics'
    )
    $mtpOptions = @(
        '--spec-type',
        '--spec-draft-model',
        '--spec-draft-device',
        '--spec-draft-ngl',
        '--spec-draft-type-k',
        '--spec-draft-type-v',
        '--spec-draft-n-max'
    )
    $required = if ($RequireMtp) { $baselineOptions + $mtpOptions } else { $baselineOptions }
    $helpText = @(& $Runtime '--help' 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "Runtime --help failed: $($helpText -join "`n")"
    }
    $help = $helpText -join "`n"
    $missing = @($required | Where-Object { $help.IndexOf($_, [System.StringComparison]::Ordinal) -lt 0 })
    if ($missing.Count -gt 0) {
        throw "Runtime is missing required options: $($missing -join ', ')"
    }
}

function Get-ExpectedModelShards {
    param([Parameter(Mandatory)][string]$FirstShard)

    $name = [System.IO.Path]::GetFileName($FirstShard)
    $directory = [System.IO.Path]::GetDirectoryName($FirstShard)
    $match = [regex]::Match($name, '^(?<prefix>.+)-(?<part>\d{5})-of-(?<count>\d{5})\.gguf$', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
    if (-not $match.Success) { return @($FirstShard) }

    $count = [int]$match.Groups['count'].Value
    $prefix = $match.Groups['prefix'].Value
    return @(1..$count | ForEach-Object {
        Join-Path $directory ('{0}-{1:D5}-of-{2:D5}.gguf' -f $prefix, $_, $count)
    })
}

function Assert-ModelShards {
    param([Parameter(Mandatory)][string]$FirstShard)

    $missing = @(Get-ExpectedModelShards -FirstShard $FirstShard | Where-Object {
        -not (Test-Path -LiteralPath $_ -PathType Leaf)
    })
    if ($missing.Count -gt 0) {
        throw "Required model shard(s) are missing: $($missing -join ', ')"
    }
}

$runtime = Resolve-WorkspacePath -Value $RuntimePath -Label 'RuntimePath'
$model = Resolve-WorkspacePath -Value $ModelPath -Label 'ModelPath'
$mtpModel = Resolve-WorkspacePath -Value $MtpModelPath -Label 'MtpModelPath'

if ($Stop) {
    Stop-ManagedServer -Runtime $runtime -ServerPort $Port
    exit 0
}

if ($ContextSize -lt 1) { throw 'ContextSize must be positive.' }
if ($BatchSize -lt 1) { throw 'BatchSize must be positive.' }
if ($UbatchSize -lt 1 -or $UbatchSize -gt $BatchSize) { throw 'UbatchSize must be positive and no larger than BatchSize.' }
if ($Port -lt 1 -or $Port -gt 65535) { throw 'Port must be between 1 and 65535.' }
if ($TensorSplit -notmatch '^\s*[0-9]+(?:\.[0-9]+)?\s*,\s*[0-9]+(?:\.[0-9]+)?\s*$') {
    throw "TensorSplit must contain two positive proportions, for example '3,2'."
}
$splitValues = @($TensorSplit.Split(',') | ForEach-Object { [double]$_.Trim() })
if ($splitValues.Count -ne 2 -or @($splitValues | Where-Object { $_ -le 0 }).Count -gt 0) {
    throw "TensorSplit must contain two positive proportions, for example '3,2'."
}

$arguments = @(
    '--model', $model,
    '--load-mode', 'none',
    '--alias', 'qwen3.8-flash-next',
    '--host', $BindAddress,
    '--port', $Port,
    '--device', 'CUDA0,CUDA1',
    '--split-mode', 'layer',
    '--tensor-split', $TensorSplit,
    '--gpu-layers', 'all',
    '--ctx-size', $ContextSize,
    '--parallel', '2',
    '--override-tensor', 'per_layer_token_embd=CPU',
    '--n-cpu-moe', $CpuMoeLayers,
    '--flash-attn', 'on',
    '--cache-type-k', $CacheTypeK,
    '--cache-type-v', $CacheTypeV,
    '--batch-size', $BatchSize,
    '--ubatch-size', $UbatchSize,
    '--fit', 'off',
    '--no-mmproj',
    '--no-context-shift',
    '--jinja',
    '--metrics'
)

if ($Threads -gt 0) { $arguments += @('--threads', $Threads) }
if ($ThreadsBatch -gt 0) { $arguments += @('--threads-batch', $ThreadsBatch) }

if ($Profile -eq 'Mtp') {
    $arguments += @(
        '--spec-draft-model', $mtpModel,
        '--spec-type', 'draft-mtp',
        '--spec-draft-device', $MtpDevice,
        '--spec-draft-ngl', 'all',
        '--spec-draft-type-k', $MtpPrecision,
        '--spec-draft-type-v', $MtpPrecision,
        '--spec-draft-n-max', $MtpNMax
    )
}

if ($DryRun) {
    Write-Host 'DRY RUN: no llama-server process will start and no model will be loaded.'
    if (-not (Test-Path -LiteralPath $runtime -PathType Leaf)) { Write-Warning "Runtime is missing: $runtime" }
    foreach ($missingShard in @(Get-ExpectedModelShards -FirstShard $model | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })) {
        Write-Warning "Model shard is missing: $missingShard"
    }
    if ($Profile -eq 'Mtp' -and -not (Test-Path -LiteralPath $mtpModel -PathType Leaf)) { Write-Warning "MTP model is missing: $mtpModel" }
    Write-Host "CUDA_VISIBLE_DEVICES=$Gpu0Uuid,$Gpu1Uuid"
    Write-Host "Command: $(Quote-CommandArgument $runtime) $(($arguments | ForEach-Object { Quote-CommandArgument ([string]$_) }) -join ' ')"
    exit 0
}

if (-not (Test-Path -LiteralPath $runtime -PathType Leaf)) { throw "Runtime is missing: $runtime" }
Assert-ModelShards -FirstShard $model
if ($Profile -eq 'Mtp' -and -not (Test-Path -LiteralPath $mtpModel -PathType Leaf)) { throw "MTP model is missing: $mtpModel" }
Assert-RuntimeOptions -Runtime $runtime -RequireMtp ($Profile -eq 'Mtp')
Assert-GpuMapping -FirstUuid $Gpu0Uuid -SecondUuid $Gpu1Uuid

$oldVisible = [Environment]::GetEnvironmentVariable('CUDA_VISIBLE_DEVICES', 'Process')
$oldCudaModuleLoading = [Environment]::GetEnvironmentVariable('CUDA_MODULE_LOADING', 'Process')
$env:CUDA_VISIBLE_DEVICES = "$Gpu0Uuid,$Gpu1Uuid"
$env:CUDA_MODULE_LOADING = 'EAGER'
Write-Host "Validated CUDA0=RTX 5090 ($Gpu0Uuid), CUDA1=RTX 4090 ($Gpu1Uuid)."
Write-Host "Starting Qwen3.8-Flash-Next $Profile profile on $BindAddress`:$Port."

try {
    & $runtime @arguments
} finally {
    if ($null -eq $oldVisible) {
        Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    } else {
        $env:CUDA_VISIBLE_DEVICES = $oldVisible
    }
    if ($null -eq $oldCudaModuleLoading) {
        Remove-Item Env:CUDA_MODULE_LOADING -ErrorAction SilentlyContinue
    } else {
        $env:CUDA_MODULE_LOADING = $oldCudaModuleLoading
    }
}
