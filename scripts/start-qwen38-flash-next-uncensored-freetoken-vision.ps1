[CmdletBinding()]
param(
    [ValidateSet('Short4K', 'Native256K')]
    [string]$Profile = 'Native256K',
    [ValidateRange(1, 8)]
    [int]$MaxRunningRequests = 1,
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
$model = '/home/rba90/models/Qwen3.8-Flash-Next-Uncensored-NVFP4'
$servedModel = 'qwen38-next-uncensored-freetoken-vision'
$pidFile = "/tmp/qwen38-flash-next-uncensored-freetoken-$Port.pid"

if ($Stop) {
    $stopErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $recordedPidOutput = & wsl.exe cat $pidFile 2>$null
    $recordedPid = if ($null -eq $recordedPidOutput) { '' } else { ([string]$recordedPidOutput).Trim() }
    if ($recordedPid -match '^[1-9][0-9]*$') {
        & wsl.exe kill -TERM -- "-$recordedPid" 2>$null
        Start-Sleep -Seconds 1
        & wsl.exe kill -TERM -- $recordedPid 2>$null
        Start-Sleep -Seconds 1
        & wsl.exe kill -KILL -- "-$recordedPid" 2>$null
        & wsl.exe kill -KILL -- $recordedPid 2>$null
    }
    & wsl.exe rm -f -- $pidFile 2>$null
    $ErrorActionPreference = $stopErrorAction
    return
}

$gpu = nvidia-smi --query-gpu=uuid,name --format=csv,noheader | Where-Object { $_ -like "$GpuUuid,*" }
if (-not $gpu) {
    throw "RTX 5090 UUID $GpuUuid was not found by nvidia-smi"
}

& wsl.exe test -d $model
if ($LASTEXITCODE -ne 0) {
    throw "Uncensored FreeToken checkpoint was not found in WSL: $model"
}
& wsl.exe test -f "$model/model.safetensors.index.json"
if ($LASTEXITCODE -ne 0) {
    throw "Uncensored FreeToken checkpoint index was not found in WSL: $model/model.safetensors.index.json"
}
& wsl.exe test -x $freeToken
if ($LASTEXITCODE -ne 0) {
    throw "FreeToken executable was not found in WSL: $freeToken"
}

$tokens = if ($Profile -eq 'Native256K') { 262144 } else { 8192 }
$maxOutput = if ($Profile -eq 'Native256K') { 65536 } else { 512 }
$command = @(
    $freeToken, 'serve',
    '--model', $model,
    '--served-model-name', $servedModel,
    '--gpu', $GpuUuid,
    '--host', '0.0.0.0',
    '--port', $Port.ToString(),
    '--max-running-requests', $MaxRunningRequests.ToString(),
    '--dtype', 'bfloat16',
    '--memory-ratio', '0.90',
    '--moe-strategy', 'offload',
    '--moe-cpu-layers', 'auto',
    # Ratio 2 plus one request gives 8 usable GDN state slots (4 working + 4 cache).
    # Keep the expert cache at 4600 to retain VRAM headroom.
    '--moe-cache-size', '4600',
    '--linear-state-cache-ratio', '2',
    '--ple-backend', 'disk',
    '--max-seq-len-override', $tokens.ToString(),
    '--kv-reserve-tokens', $tokens.ToString(),
    '--num-tokens', $tokens.ToString(),
    '--max-prefill-length', '8192',
    '--max-output-tokens', $maxOutput.ToString(),
    '--cache-type', 'radix',
    '--enable-cache-report',
    '--reasoning-parser', 'qwen3',
    '--tool-call-parser', 'qwen3_coder',
    '--mm-encoder-weights', 'host'
)

$display = @('wsl.exe', 'bash', $launchScript, $pidFile) + $command
if ($DryRun) {
    ($display | ForEach-Object {
        if ($_ -match '\s') { "'" + $_.Replace("'", "''") + "'" } else { $_ }
    }) -join ' '
    return
}

Write-Host "Starting uncensored vision deployment on $gpu"
Write-Host "Served model: $servedModel"
Write-Host "Profile: $Profile ($tokens tokens)"
Write-Host "OpenAI endpoint: http://127.0.0.1:$Port/v1"
& wsl.exe bash $launchScript $pidFile @command
if ($LASTEXITCODE -ne 0) {
    throw "FreeToken exited with code $LASTEXITCODE"
}
