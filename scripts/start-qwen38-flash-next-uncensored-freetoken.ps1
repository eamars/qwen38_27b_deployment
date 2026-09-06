[CmdletBinding()]
param(
    [ValidateSet('Short4K', 'Native256K')]
    [string]$Profile = 'Native256K',
    [ValidateRange(1, 8)]
    [int]$MaxRunningRequests = 2,
    [string]$Model = '/home/rba90/models/Qwen3.8-Flash-Next-Uncensored-NVFP4',
    [string]$GpuUuid = 'GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0',
    [ValidateRange(1, 65535)]
    [int]$Port = 1919,
    [switch]$DryRun,
    [switch]$Check,
    [switch]$Stop
)

$ErrorActionPreference = 'Stop'
if (([int]$DryRun.IsPresent + [int]$Check.IsPresent + [int]$Stop.IsPresent) -gt 1) {
    throw 'Choose only one of -DryRun, -Check, or -Stop'
}

function ConvertTo-WslMountPath {
    param([Parameter(Mandatory)][string]$Path)
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $match = [regex]::Match($fullPath, '^(?<drive>[A-Za-z]):\\(?<tail>.*)$')
    if (-not $match.Success) { throw "Expected a local Windows drive path: $fullPath" }
    return '/mnt/' + $match.Groups['drive'].Value.ToLowerInvariant() + '/' + $match.Groups['tail'].Value.Replace('\', '/')
}

$manager = ConvertTo-WslMountPath (Join-Path $PSScriptRoot 'manage-qwen38-uncensored.py')
$python = '/home/rba90/.freetoken-qwen38/venv/bin/python'
$freeToken = '/home/rba90/.freetoken-qwen38/venv/bin/ft'
$action = if ($Stop) { 'stop' } elseif ($Check) { 'check' } else { 'start' }
$managerArgs = @($manager, '--action', $action, '--model', $Model, '--port', $Port.ToString(), '--gpu', $GpuUuid)

$tokens = if ($Profile -eq 'Native256K') { 262144 } else { 8192 }
$maxOutput = if ($Profile -eq 'Native256K') { 65536 } else { 512 }
$command = @(
    $freeToken, 'serve',
    '--model', $Model,
    '--served-model-name', 'qwen38-next-uncensored-freetoken',
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
if ($action -eq 'start') { $managerArgs += @('--') + $command }
if ($DryRun) {
    $display = @('wsl.exe', '--exec', $python) + $managerArgs | ForEach-Object {
        if ($_ -match '\s') { "'" + $_.Replace("'", "''") + "'" } else { $_ }
    }
    $display -join ' '
    return
}

if ($action -eq 'start') {
    $listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    if ($listener) { throw "Port $Port is occupied on Windows; no model started and no process stopped" }
    Write-Host "Starting uncensored $Profile on $GpuUuid"
    Write-Host "OpenAI endpoint: http://127.0.0.1:$Port/v1"
}
& wsl.exe --exec $python @managerArgs
if ($LASTEXITCODE -ne 0) { throw "Uncensored FreeToken $action failed with code $LASTEXITCODE" }
