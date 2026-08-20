[CmdletBinding()]
param(
    [int]$DurationSeconds = 120,
    [int]$IntervalMilliseconds = 250,
    [string]$Phase = 'unspecified',
    [string]$OutputPath = (Join-Path (Split-Path -Parent $PSScriptRoot) ('benchmarks\vram-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.csv'))
)

$ErrorActionPreference = 'Stop'
if ($DurationSeconds -lt 1) { throw 'DurationSeconds must be positive.' }
if ($IntervalMilliseconds -lt 100) { throw 'IntervalMilliseconds must be at least 100.' }
$output = [System.IO.Path]::GetFullPath($OutputPath)
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $output) | Out-Null

'timestamp,phase,gpu_index,name,uuid,total_mib,used_mib,free_mib,utilization_gpu' | Set-Content -LiteralPath $output -Encoding UTF8
$query = 'index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu'
$watch = [System.Diagnostics.Stopwatch]::StartNew()
while ($watch.Elapsed.TotalSeconds -lt $DurationSeconds) {
    $rows = @(nvidia-smi.exe --query-gpu=$query --format=csv,noheader,nounits 2>&1)
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi query failed: $($rows -join "`n")" }
    $timestamp = Get-Date -Format 'o'
    foreach ($row in $rows) {
        $fields = $row -split ',\s*'
        if ($fields.Count -lt 7) { continue }
        "$timestamp,$Phase,$($fields[0]),$($fields[1]),$($fields[2]),$($fields[3]),$($fields[4]),$($fields[5]),$($fields[6])" | Add-Content -LiteralPath $output
    }
    Start-Sleep -Milliseconds $IntervalMilliseconds
}
$watch.Stop()

$data = Import-Csv -LiteralPath $output
foreach ($gpu in ($data | Group-Object gpu_index)) {
    $minimum = ($gpu.Group | ForEach-Object { [int]$_.free_mib } | Measure-Object -Minimum).Minimum
    $peak = ($gpu.Group | ForEach-Object { [int]$_.used_mib } | Measure-Object -Maximum).Maximum
    Write-Host "GPU $($gpu.Name): minimum free $minimum MiB; peak used $peak MiB; samples $($gpu.Count)"
}
Write-Host "VRAM profile written: $output"
