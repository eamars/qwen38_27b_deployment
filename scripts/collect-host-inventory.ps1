[CmdletBinding()]
param(
    [string]$OutputPath = (Join-Path (Split-Path -Parent $PSScriptRoot) 'docs\host-inventory.md')
)

$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$output = [System.IO.Path]::GetFullPath($OutputPath)
$outputDirectory = Split-Path -Parent $output
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

function Get-CommandText {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$FilePath, [string[]]$Arguments = @())
    $result = & $FilePath @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $FilePath $($Arguments -join ' ')`n$($result -join "`n")"
    }
    return (($result | Out-String).Trim())
}

$os = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$ramGiB = [math]::Round([double]$os.TotalVisibleMemorySize / 1MB, 1)
$gpuQuery = 'index,name,uuid,pci.bus_id,memory.total,memory.used,memory.free,driver_version,temperature.gpu,power.draw,utilization.gpu'
$gpuRows = @(nvidia-smi.exe --query-gpu=$gpuQuery --format=csv,noheader,nounits 2>&1)
if ($LASTEXITCODE -ne 0) { throw "nvidia-smi GPU query failed: $($gpuRows -join "`n")" }
$cudaText = Get-CommandText -FilePath 'nvcc.exe' -Arguments @('--version')
$cmakePath = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$msvcPath = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe'
$cmakeText = if (Test-Path -LiteralPath $cmakePath) { Get-CommandText -FilePath $cmakePath -Arguments @('--version') } else { 'Not found at expected Visual Studio Build Tools path' }
$msvcText = if (Test-Path -LiteralPath $msvcPath) { (& $msvcPath 2>&1 | Select-Object -First 3 | Out-String).Trim() } else { 'Not found at expected Visual Studio Build Tools path' }

$gpuTable = @(
    '| CUDA index | Name | UUID | PCI bus ID | Total MiB | Used MiB | Free MiB | Driver |'
    '|---:|---|---|---|---:|---:|---:|---|'
)
foreach ($row in $gpuRows) {
    $fields = $row -split ',\s*'
    if ($fields.Count -lt 8) { continue }
    $gpuTable += "| $($fields[0]) | $($fields[1]) | $($fields[2]) | $($fields[3]) | $($fields[4]) | $($fields[5]) | $($fields[6]) | $($fields[7]) |"
}

$content = @"
# Host inventory

Captured: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')

This is the pre-installation host snapshot required by the deployment instruction. GPU assignments are explicit and must be rechecked before starting a backend.

## Operating system and CPU

- Windows: $($os.Caption), version $($os.Version), build $($os.BuildNumber), $($os.OSArchitecture)
- CPU: $($cpu.Name.Trim())
- Cores / logical processors: $($cpu.NumberOfCores) / $($cpu.NumberOfLogicalProcessors)
- System RAM: $ramGiB GiB visible to Windows

## GPUs

$($gpuTable -join "`n")

Deployment mapping:

- RTX 5090 backend: CUDA index 0, UUID GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0, PCI 00000000:01:00.0
- RTX 4090 backend: CUDA index 1, UUID GPU-eed52936-813f-8d68-1654-bfb56cb42bc3, PCI 00000000:03:00.0

The CUDA index is only used after the UUID/name check; the UUID and PCI identity are the authoritative mapping.

## CUDA and build tools

NVIDIA driver: 610.74

CUDA toolkit/compiler:

~~~text
$cudaText
~~~

CMake executable: $cmakePath

~~~text
$cmakeText
~~~

MSVC compiler executable: $msvcPath

~~~text
$msvcText
~~~

## Baseline caveat

At capture time, both GPUs had substantial non-deployment Windows/LM Studio usage. The deployment setup does not terminate those processes.
VRAM acceptance measurements must be repeated with the intended workload and the actual background usage documented.
"@

Set-Content -LiteralPath $output -Value $content -Encoding UTF8
Write-Host "Wrote host inventory: $output"
