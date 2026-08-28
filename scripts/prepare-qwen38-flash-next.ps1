[CmdletBinding()]
param(
    [string]$RuntimePath = 'runtime\llama.cpp-qwen4exp\build\bin\Release\llama-server.exe',
    [string]$SourcePath = 'runtime\llama.cpp-qwen4exp',
    [string]$ModelPath = 'models\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf',
    [string]$MtpModelPath = 'models\Qwen3.8-Flash-Next-MTP-F16.gguf',
    [string]$OutputPath = 'docs\qwen38-flash-next-environment.json',
    [string]$Gpu0Uuid = 'GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0',
    [string]$Gpu1Uuid = 'GPU-eed52936-813f-8d68-1654-bfb56cb42bc3',
    [switch]$HashModels,
    [switch]$Strict
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
    $workspacePrefix = $workspace.TrimEnd('\') + '\'
    if (-not $candidate.StartsWith($workspacePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label must be inside the workspace: $candidate"
    }
    return $candidate
}

function Invoke-OptionalText {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$Arguments = @()
    )

    try {
        $result = & $FilePath @Arguments 2>&1
        $exitCode = $LASTEXITCODE
        return [ordered]@{
            available = $true
            exit_code = $exitCode
            text = (($result | Out-String).Trim())
        }
    } catch {
        return [ordered]@{
            available = $false
            exit_code = $null
            text = $_.Exception.Message
        }
    }
}

function Resolve-OptionalCMake {
    $command = Get-Command 'cmake.exe' -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $visualStudioCmake = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
    if (Test-Path -LiteralPath $visualStudioCmake -PathType Leaf) { return $visualStudioCmake }
    return 'cmake.exe'
}

function Get-GitSnapshot {
    param([Parameter(Mandatory)][string]$Path)

    if (-not (Test-Path -LiteralPath (Join-Path $Path '.git'))) {
        return [ordered]@{
            available = $false
            sha = $null
            dirty = $null
            status = 'source repository not found'
        }
    }

    $shaResult = Invoke-OptionalText -FilePath 'git.exe' -Arguments @('-C', $Path, 'rev-parse', 'HEAD')
    $statusResult = Invoke-OptionalText -FilePath 'git.exe' -Arguments @('-C', $Path, 'status', '--short')
    $statusText = [string]$statusResult.text
    return [ordered]@{
        available = ($shaResult.exit_code -eq 0)
        sha = if ($shaResult.exit_code -eq 0) { [string]$shaResult.text } else { $null }
        dirty = if ($statusResult.exit_code -eq 0) { $statusText.Length -gt 0 } else { $null }
        status = $statusText
    }
}

function Get-ArtifactSnapshot {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Role,
        [switch]$Required,
        [switch]$Hash
    )

    $exists = Test-Path -LiteralPath $Path -PathType Leaf
    if (-not $exists) {
        return [ordered]@{
            role = $Role
            path = $Path
            required = [bool]$Required
            present = $false
            size_bytes = $null
            sha256 = $null
        }
    }

    $file = Get-Item -LiteralPath $Path
    $sha256 = if ($Hash) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
    return [ordered]@{
        role = $Role
        path = $Path
        required = [bool]$Required
        present = $true
        size_bytes = $file.Length
        sha256 = $sha256
    }
}

function Get-MainModelSnapshots {
    param(
        [Parameter(Mandatory)][string]$FirstShard,
        [switch]$Hash
    )

    $name = [System.IO.Path]::GetFileName($FirstShard)
    $directory = [System.IO.Path]::GetDirectoryName($FirstShard)
    $match = [regex]::Match($name, '^(?<prefix>.+)-(?<part>\d{5})-of-(?<count>\d{5})\.gguf$', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
    if (-not $match.Success) {
        return @((Get-ArtifactSnapshot -Path $FirstShard -Role 'main UD-Q4_K_XL model' -Required -Hash:$Hash))
    }

    $count = [int]$match.Groups['count'].Value
    $prefix = $match.Groups['prefix'].Value
    $snapshots = @()
    for ($part = 1; $part -le $count; $part++) {
        $filename = '{0}-{1:D5}-of-{2:D5}.gguf' -f $prefix, $part, $count
        $snapshots += Get-ArtifactSnapshot -Path (Join-Path $directory $filename) -Role ("main UD-Q4_K_XL shard {0}/{1}" -f $part, $count) -Required -Hash:$Hash
    }
    return $snapshots
}

function Get-ModelSetSha256 {
    param([Parameter(Mandatory)][object[]]$Artifacts)

    if ($Artifacts.Count -eq 0 -or @($Artifacts | Where-Object { -not $_.present -or -not $_.sha256 }).Count -gt 0) {
        return $null
    }
    $rows = $Artifacts | ForEach-Object {
        '{0}|{1}|{2}' -f [System.IO.Path]::GetFileName($_.path), $_.size_bytes, $_.sha256
    }
    $canonical = ($rows -join "`n") + "`n"
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        $digest = $hasher.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($canonical))
        return ([System.BitConverter]::ToString($digest) -replace '-', '').ToLowerInvariant()
    } finally {
        $hasher.Dispose()
    }
}

$runtime = Resolve-WorkspacePath -Value $RuntimePath -Label 'RuntimePath'
$source = Resolve-WorkspacePath -Value $SourcePath -Label 'SourcePath'
$model = Resolve-WorkspacePath -Value $ModelPath -Label 'ModelPath'
$mtpModel = Resolve-WorkspacePath -Value $MtpModelPath -Label 'MtpModelPath'
$output = Resolve-WorkspacePath -Value $OutputPath -Label 'OutputPath'

$gpuQuery = 'index,name,uuid,pci.bus_id,memory.total,memory.used,memory.free,driver_version,temperature.gpu,power.draw,utilization.gpu'
$gpuSnapshot = Invoke-OptionalText -FilePath 'nvidia-smi.exe' -Arguments @(( '--query-gpu=' + $gpuQuery ), '--format=csv,noheader,nounits')
$topologySnapshot = Invoke-OptionalText -FilePath 'nvidia-smi.exe' -Arguments @('topo', '-m')
$cudaSnapshot = Invoke-OptionalText -FilePath 'nvcc.exe' -Arguments @('--version')
$cmakeSnapshot = Invoke-OptionalText -FilePath (Resolve-OptionalCMake) -Arguments @('--version')

$runtimeSnapshot = [ordered]@{
    path = $runtime
    present = Test-Path -LiteralPath $runtime -PathType Leaf
    version = $null
    help = $null
    devices = $null
    required_options = [ordered]@{}
    mtp_options = [ordered]@{}
    model_support_validation = 'not run: model loading is intentionally excluded from preparation'
}
$requiredOptions = @(
    '--load-mode',
    '--n-cpu-moe',
    '--split-mode',
    '--tensor-split',
    '--override-tensor',
    '--cache-type-k',
    '--cache-type-v',
    '--flash-attn',
    '--fit',
    '--cache-ram',
    '--metrics',
    '--list-devices'
)
$optionalMtpOptions = @(
    '--spec-type',
    '--spec-draft-model',
    '--spec-draft-device',
    '--spec-draft-ngl',
    '--spec-draft-type-k',
    '--spec-draft-type-v'
)
if ($runtimeSnapshot.present) {
    $versionResult = Invoke-OptionalText -FilePath $runtime -Arguments @('--version')
    $helpResult = Invoke-OptionalText -FilePath $runtime -Arguments @('--help')
    $deviceResult = Invoke-OptionalText -FilePath $runtime -Arguments @('--list-devices')
    $runtimeSnapshot.version = $versionResult.text
    $runtimeSnapshot.help = $helpResult.text
    $runtimeSnapshot.devices = $deviceResult.text
    foreach ($option in $requiredOptions) {
        $runtimeSnapshot.required_options[$option] = ([string]$helpResult.text).IndexOf($option, [System.StringComparison]::Ordinal) -ge 0
    }
    foreach ($option in $optionalMtpOptions) {
        $runtimeSnapshot.mtp_options[$option] = ([string]$helpResult.text).IndexOf($option, [System.StringComparison]::Ordinal) -ge 0
    }
}

$mainArtifacts = @(Get-MainModelSnapshots -FirstShard $model -Hash:$HashModels)
$artifacts = @($mainArtifacts) + @(
    (Get-ArtifactSnapshot -Path $mtpModel -Role 'optional F16/BF16 MTP head' -Hash:$HashModels)
)

$missingRequired = @()
if (-not $runtimeSnapshot.present) { $missingRequired += "runtime: $runtime" }
foreach ($artifact in $mainArtifacts) {
    if (-not $artifact.present) { $missingRequired += "model shard: $($artifact.path)" }
}
if ($runtimeSnapshot.present) {
    foreach ($option in $requiredOptions) {
        if (-not $runtimeSnapshot.required_options[$option]) { $missingRequired += "runtime option: $option" }
    }
}

$manifest = [ordered]@{
    generated_utc = [DateTime]::UtcNow.ToString('o')
    workspace = $workspace
    preparation_only = $true
    model_load_performed = $false
    status = if ($missingRequired.Count -eq 0) { 'ready for explicit load validation' } else { 'incomplete' }
    expected_devices = [ordered]@{
        CUDA0 = [ordered]@{ name = 'RTX 5090'; uuid = $Gpu0Uuid }
        CUDA1 = [ordered]@{ name = 'RTX 4090'; uuid = $Gpu1Uuid }
    }
    gpu_snapshot = $gpuSnapshot
    topology_snapshot = $topologySnapshot
    cuda_toolkit = $cudaSnapshot
    cmake = $cmakeSnapshot
    source = Get-GitSnapshot -Path $source
    runtime = $runtimeSnapshot
    artifacts = $artifacts
    model_set_sha256 = Get-ModelSetSha256 -Artifacts $mainArtifacts
    model_hashing_requested = [bool]$HashModels
    missing_required = $missingRequired
    deployment_constraints = @(
        'split-mode layer only for the initial Flash-Next deployment',
        'tensor split is a layer-placement proportion, not tensor parallelism',
        'per_layer_token_embd is explicitly placed on CPU host memory',
        'target model uses UD-Q4_K_XL or another Q4-or-better candidate',
        'F16/BF16 MTP is the correctness baseline; Q8 MTP is not enabled by this preparation',
        'one server slot and no whole-model mlock until host-memory headroom is proven'
    )
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $output) | Out-Null
$manifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $output -Encoding UTF8
Write-Host "Wrote no-load preparation manifest: $output"
Write-Host 'No model was loaded. Runtime checks used only --version, --help, and --list-devices.'
if ($missingRequired.Count -gt 0) {
    Write-Warning "Preparation is incomplete: $($missingRequired -join '; ')"
    if ($Strict) { throw 'Strict preparation check failed.' }
}
