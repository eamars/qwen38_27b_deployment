[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$requiredDirectories = @('docs', 'models', 'scripts', 'benchmarks', 'runtime')
foreach ($directory in $requiredDirectories) {
    $path = Join-Path $workspace $directory
    if (-not (Test-Path -LiteralPath $path -PathType Container)) { throw "Missing required directory: $path" }
}

$runtime = Join-Path $workspace 'runtime\llama.cpp-dflash2\build-dflash2\bin\Release\llama-server.exe'
if (-not (Test-Path -LiteralPath $runtime -PathType Leaf)) { throw "Runtime executable is not ready: $runtime" }
$models = @(
    (Join-Path $workspace 'models\Qwen3.8-27B-UD-Q6_K_M.gguf'),
    (Join-Path $workspace 'models\Qwen3.8-27B-UD-Q4_K_XL.gguf'),
    (Join-Path $workspace 'models\Qwen3.8-27B-DFlash2-Q4_K_M.gguf')
)
foreach ($model in $models) {
    if (-not (Test-Path -LiteralPath $model -PathType Leaf)) { throw "Required model artifact is missing: $model" }
}

$version = (& $runtime --version 2>&1 | Out-String).Trim()
$help = (& $runtime --help 2>&1 | Out-String)
if ($help -notmatch 'draft-dflash') { throw 'Runtime help does not advertise draft-dflash support.' }
foreach ($option in @('--spec-draft-model', '--spec-draft-ngl', '--cache-type-k', '--cache-type-v')) {
    if ($help -notmatch [regex]::Escape($option)) { throw "Runtime help is missing required option: $option" }
}

$deviceChecks = @(
    @{ Name = 'RTX 5090'; Uuid = 'GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0' },
    @{ Name = 'RTX 4090'; Uuid = 'GPU-eed52936-813f-8d68-1654-bfb56cb42bc3' }
)
foreach ($check in $deviceChecks) {
    $oldVisible = $env:CUDA_VISIBLE_DEVICES
    try {
        $env:CUDA_VISIBLE_DEVICES = $check.Uuid
        $devices = (& $runtime --list-devices 2>&1 | Out-String)
    } finally {
        if ($null -eq $oldVisible) { Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue } else { $env:CUDA_VISIBLE_DEVICES = $oldVisible }
    }
    if ($devices -notmatch [regex]::Escape($check.Name) -or $devices -notmatch 'CUDA0') {
        throw "UUID isolation check failed for $($check.Name) ($($check.Uuid)).`n$devices"
    }
}

Write-Host 'PRE-LOAD CHECK PASSED'
Write-Host "Runtime: $runtime"
Write-Host $version
Write-Host 'Models are present and runtime DFlash2 options are available.'
Write-Host 'Both GPU UUID isolation checks map the selected physical GPU to runtime CUDA0.'
Write-Host 'No server was started and no model was loaded by this check.'
