[CmdletBinding()]
param(
    [ValidateSet('Short4K', 'Native256K')]
    [string]$Profile = 'Native256K',
    [ValidateRange(1, 8)]
    [int]$MaxRunningRequests = 2,
    [string]$Model = '/home/rba90/models/Qwen3.8-Flash-Next-NVFP4',
    [string]$GpuUuid = 'GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0',
    [int]$Port = 1919,
    [switch]$DryRun,
    [switch]$Stop
)

$ErrorActionPreference = 'Stop'

function ConvertTo-WslMountPath {
    param([Parameter(Mandatory)][string]$Path)

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $match = [regex]::Match($fullPath, '^(?<drive>[A-Za-z]):\\(?<tail>.*)$')
    if (-not $match.Success) {
        throw "Expected a local Windows drive path, got: $fullPath"
    }

    $drive = $match.Groups['drive'].Value.ToLowerInvariant()
    $tail = $match.Groups['tail'].Value.Replace('\', '/')
    return "/mnt/$drive/$tail"
}

$launchScript = ConvertTo-WslMountPath (Join-Path $PSScriptRoot 'launch-freetoken-wsl.sh')
$freeToken = '/home/rba90/.freetoken-qwen38/venv/bin/ft'
$pidFile = "/tmp/qwen38-flash-next-freetoken-$Port.pid"

if ($Stop) {
    $recordedPidOutput = & wsl.exe cat $pidFile 2>$null
    $recordedPid = if ($null -eq $recordedPidOutput) { '' } else { ([string]$recordedPidOutput).Trim() }
    if ($recordedPid -match '^[1-9][0-9]*$') {
        # The launcher records the setsid child, which is normally its own
        # process-group leader. Try the group first, then the exact child.
        & wsl.exe kill -TERM -- "-$recordedPid" 2>$null
        Start-Sleep -Seconds 1
        & wsl.exe kill -TERM -- $recordedPid 2>$null
        Start-Sleep -Seconds 1
        & wsl.exe kill -KILL -- "-$recordedPid" 2>$null
        & wsl.exe kill -KILL -- $recordedPid 2>$null
    }
    & wsl.exe rm -f -- $pidFile 2>$null
    return
}

$gpu = nvidia-smi --query-gpu=uuid,name --format=csv,noheader | Where-Object { $_ -like "$GpuUuid,*" }
if (-not $gpu) {
    throw "RTX 5090 UUID $GpuUuid was not found by nvidia-smi"
}

& wsl.exe test -d $Model
if ($LASTEXITCODE -ne 0) {
    throw "FreeToken checkpoint was not found in WSL: $Model"
}
& wsl.exe test -f "$Model/model.safetensors.index.json"
if ($LASTEXITCODE -ne 0) {
    throw "FreeToken checkpoint index was not found in WSL: $Model/model.safetensors.index.json"
}
& wsl.exe test -x $freeToken
if ($LASTEXITCODE -ne 0) {
    throw "FreeToken executable was not found in WSL: $freeToken"
}

$tokens = if ($Profile -eq 'Native256K') { 262144 } else { 8192 }
$maxOutput = if ($Profile -eq 'Native256K') { 65536 } else { 512 }
$command = @(
    $freeToken, 'serve',
    '--model', $Model,
    '--served-model-name', 'qwen38-next-freetoken',
    '--gpu', $GpuUuid,
    '--host', '0.0.0.0',
    '--port', $Port.ToString(),
    '--max-running-requests', $MaxRunningRequests.ToString(),
    '--dtype', 'bfloat16',
    '--memory-ratio', '0.90',
    '--moe-backend', 'offload',
    '--moe-cpu-layers', '0',
    '--moe-cache-auto',
    '--ple-backend', 'disk',
    '--kv-reserve-tokens', $tokens.ToString(),
    '--num-tokens', $tokens.ToString(),
    '--max-prefill-length', '8192',
    '--max-output-tokens', $maxOutput.ToString(),
    '--cache-type', 'radix',
    '--reasoning-parser', 'qwen3',
    '--tool-call-parser', 'qwen3_coder'
)

$display = @('wsl.exe', 'bash', $launchScript, $pidFile) + $command
if ($DryRun) {
    $display -join ' '
    return
}

Write-Host "Starting $Profile on $gpu"
Write-Host "OpenAI endpoint: http://127.0.0.1:$Port/v1"
& wsl.exe bash $launchScript $pidFile @command
if ($LASTEXITCODE -ne 0) {
    throw "FreeToken exited with code $LASTEXITCODE"
}
