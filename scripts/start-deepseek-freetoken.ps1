[CmdletBinding()]
param(
    [ValidateRange(128, 180000)]
    [int]$Context = 180000,
    [ValidateRange(256, 4096)]
    [int]$ExpertCacheSlots = 1536,
    [ValidateRange(1, 8)]
    [int]$MaxRunningRequests = 1,
    [string]$Model = '/home/rba90/models/DeepSeek-V4-Flash-0731-IQ3_XXS/IQ3_XXS/DeepSeek-V4-Flash-0731-IQ3_XXS-00001-of-00004.gguf',
    [string]$GpuUuid = 'GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0',
    [ValidateRange(1, 1024)]
    [int]$WindowPages = 64,
    [ValidateRange(128, 16384)]
    [int]$MaxPrefillLength = 4096,
    [switch]$DryRun,
    [switch]$Check,
    [switch]$Stop
)

$ErrorActionPreference = 'Stop'
$Port = 1919
if (([int]$DryRun.IsPresent + [int]$Check.IsPresent + [int]$Stop.IsPresent) -gt 1) {
    throw 'Choose only one of -DryRun, -Check, or -Stop'
}

function ConvertTo-WslMountPath {
    param([Parameter(Mandatory)][string]$Path)
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $match = [regex]::Match($fullPath, '^(?<drive>[A-Za-z]):\\(?<tail>.*)$')
    if (-not $match.Success) {
        throw "Expected a local Windows drive path: $fullPath"
    }
    return '/mnt/' + $match.Groups['drive'].Value.ToLowerInvariant() + '/' +
        $match.Groups['tail'].Value.Replace('\', '/')
}

function Invoke-WslTest {
    param([Parameter(Mandatory)][string[]]$Arguments)
    & wsl.exe @Arguments
    if ($LASTEXITCODE -ne 0) { return $false }
    return $true
}

function Test-LocalTcpPort {
    param([Parameter(Mandatory)][int]$PortNumber)
    try {
        return [bool](Test-NetConnection -ComputerName '127.0.0.1' -Port $PortNumber -InformationLevel Quiet -WarningAction SilentlyContinue)
    } catch {
        return $false
    }
}

$launchScript = ConvertTo-WslMountPath (Join-Path $PSScriptRoot 'launch-freetoken-wsl.sh')
$probeScript = ConvertTo-WslMountPath (Join-Path $PSScriptRoot 'start-deepseek-freetoken-probe.sh')
$pidFile = "/tmp/deepseek-freetoken-$Port.pid"
$runtime = ConvertTo-WslMountPath (Join-Path $PSScriptRoot '..\runtime\freetoken-deepseek-spec')
$kvTokens = [int]([math]::Floor($Context / 128) * 128)

if ($Stop) {
    $recordedPidOutput = & wsl.exe cat $pidFile 2>$null
    $recordedPid = if ($null -eq $recordedPidOutput) { '' } else { ([string]$recordedPidOutput).Trim() }
    if ($recordedPid -notmatch '^[1-9][0-9]*$') {
        Write-Host "No DeepSeek PID file found for port $Port. No process was stopped."
        return
    }

    Write-Host "Stopping DeepSeek FreeToken process group $recordedPid"
    & wsl.exe kill -TERM -- "-$recordedPid" 2>$null
    & wsl.exe kill -TERM -- $recordedPid 2>$null
    # launch-freetoken-wsl.sh records its setsid shell. The Python process in the
    # shell's pipeline can have a separate process group, so clean up only the
    # matching DeepSeek server for this port as well.
    $serverRows = & wsl.exe ps -eo pid=,comm=,args= 2>$null | Where-Object {
        $_ -match ('^\s*\d+\s+python\s+.*freetoken\.cli\s+serve\s+.*--served-model-name\s+deepseek-v4-flash-gpu-probe\s+.*--port\s+' + $Port + '(\s|$)')
    }
    foreach ($row in $serverRows) {
        if ($row -match '^\s*(?<serverPid>[1-9][0-9]*)\s+') {
            & wsl.exe kill -TERM -- $Matches['serverPid'] 2>$null
        }
    }
    Start-Sleep -Seconds 2
    & wsl.exe kill -KILL -- "-$recordedPid" 2>$null
    & wsl.exe kill -KILL -- $recordedPid 2>$null
    foreach ($row in $serverRows) {
        if ($row -match '^\s*(?<serverPid>[1-9][0-9]*)\s+') {
            & wsl.exe kill -KILL -- $Matches['serverPid'] 2>$null
        }
    }
    & wsl.exe rm -f -- $pidFile 2>$null
    return
}

if ($Check) {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 10
        if ([string]$health.status -ne 'ok') {
            throw "server status is '$($health.status)'"
        }
        Write-Host "DeepSeek FreeToken is ready on http://127.0.0.1:$Port/v1"
        Write-Output ($health | ConvertTo-Json -Compress)
    } catch {
        throw "DeepSeek FreeToken is not ready on port ${Port}: $($_.Exception.Message)"
    }
    return
}

if ($Context -gt 180000) {
    throw 'Context above the tested 180K maximum is disabled.'
}

$gpu = nvidia-smi --query-gpu=uuid,name --format=csv,noheader | Where-Object {
    $_ -like "$GpuUuid,*"
}
if (-not $gpu) {
    throw "RTX 5090 UUID $GpuUuid was not found by nvidia-smi"
}

if (-not (Invoke-WslTest @('test', '-f', $Model))) {
    throw "DeepSeek GGUF first shard was not found in WSL: $Model"
}
if (-not (Invoke-WslTest @('test', '-f', '/home/rba90/deepseek-freetoken-probe/download-verified.json'))) {
    throw 'The verified DeepSeek download marker is missing. Run scripts/download-deepseek-freetoken.py first.'
}
if (-not (Invoke-WslTest @('test', '-f', '/home/rba90/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062/tokenizer.json'))) {
    throw 'The official DeepSeek tokenizer is missing. Run scripts/download-deepseek-freetoken.py first.'
}

if (Test-LocalTcpPort -PortNumber $Port) {
    throw "Port $Port is already occupied; use -Stop or choose another port."
}

$command = @(
    'env',
    'FREETOKEN_MTP5=0',
    'FREETOKEN_NGRAM=0',
    ('FREETOKEN_PROBE_RUNTIME=' + $runtime),
    'FREETOKEN_LOG_SUFFIX=-baseline',
    'bash', $probeScript,
    $Context.ToString(),
    $ExpertCacheSlots.ToString(),
    '1',
    $WindowPages.ToString(),
    $MaxPrefillLength.ToString()
)
$launchArguments = @('bash', $launchScript, $pidFile) + $command

if ($DryRun) {
    $display = @('wsl.exe') + $launchArguments | ForEach-Object {
        if ($_ -match '\s') { "'" + $_.Replace("'", "''") + "'" } else { $_ }
    }
    $display -join ' '
    return
}

Write-Host "Starting DeepSeek V4 Flash baseline on $gpu"
Write-Host "GPU-only target: RTX 5090; CPU expert layers: 0"
Write-Host "Context: $kvTokens usable tokens; expert-cache slots: $ExpertCacheSlots"
Write-Host "OpenAI endpoint: http://127.0.0.1:$Port/v1"
& wsl.exe @launchArguments
if ($LASTEXITCODE -ne 0) {
    throw "DeepSeek FreeToken exited with code $LASTEXITCODE"
}
