[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^[0-9a-fA-F]{40}$')]
    [string]$Commit,
    [string]$SourcePath = 'runtime\llama.cpp-qwen4exp',
    [string]$BuildPath = 'runtime\llama.cpp-qwen4exp\build',
    [string]$OutputPath = 'docs\qwen38-flash-next-build.json',
    [string]$CudaArchitectures = '89;120',
    [string]$CmakePath = '',
    [string]$Generator = 'Visual Studio 17 2022',
    [string]$Platform = 'x64',
    [string]$BuildTarget = 'llama-server',
    [switch]$Configure,
    [switch]$Build,
    [switch]$AllowDirtySource
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
    $prefix = $workspace.TrimEnd('\') + '\'
    if (-not $candidate.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label must be inside the workspace: $candidate"
    }
    return $candidate
}

function Invoke-CheckedText {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$Arguments = @()
    )

    $previousErrorAction = $ErrorActionPreference
    try {
        # Native tools such as llama-server write version text to stderr. Capture
        # it without allowing the script-wide Stop policy to turn it into an
        # exception before LASTEXITCODE can be checked.
        $ErrorActionPreference = 'Continue'
        $result = & $FilePath @Arguments 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
    if ($exitCode -ne 0) {
        throw "Command failed ($exitCode): $FilePath $($Arguments -join ' ')`n$($result -join "`n")"
    }
    return (($result | Out-String).Trim())
}

function Get-OptionalCommandText {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$Arguments = @()
    )

    try {
        return Invoke-CheckedText -FilePath $FilePath -Arguments $Arguments
    } catch {
        return "unavailable: $($_.Exception.Message)"
    }
}

function Get-CompilerText {
    param([Parameter(Mandatory)][string]$FilePath)

    try {
        $previousErrorAction = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            $result = & $FilePath 2>&1
            $exitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $previousErrorAction
        }
        $text = (($result | Out-String).Trim())
        if ($text) { return $text }
        return "compiler exited with code $exitCode without output"
    } catch {
        return "unavailable: $($_.Exception.Message)"
    }
}

function Resolve-CMakeExecutable {
    param([string]$RequestedPath)

    if ($RequestedPath) {
        $resolved = [System.IO.Path]::GetFullPath($RequestedPath)
        if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
            throw "CmakePath is missing: $resolved"
        }
        return $resolved
    }
    $command = Get-Command 'cmake.exe' -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $visualStudioCmake = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
    if (Test-Path -LiteralPath $visualStudioCmake -PathType Leaf) { return $visualStudioCmake }
    throw 'cmake.exe was not found in PATH or the Visual Studio Build Tools location.'
}

function Get-CMakeCacheValue {
    param(
        [Parameter(Mandatory)][string]$CachePath,
        [Parameter(Mandatory)][string]$Name
    )

    if (-not (Test-Path -LiteralPath $CachePath -PathType Leaf)) { return $null }
    $match = Select-String -LiteralPath $CachePath -Pattern ('^' + [regex]::Escape($Name) + ':[^=]+=(.*)$') | Select-Object -First 1
    if ($match) { return $match.Matches[0].Groups[1].Value }
    return $null
}

function Get-CMakeCompilerPath {
    param(
        [Parameter(Mandatory)][string]$BuildDirectory,
        [Parameter(Mandatory)][string]$CachePath
    )

    $cached = Get-CMakeCacheValue -CachePath $CachePath -Name 'CMAKE_CXX_COMPILER'
    if ($cached) { return $cached }
    $compilerFile = Get-ChildItem -LiteralPath (Join-Path $BuildDirectory 'CMakeFiles') -Filter 'CMakeCXXCompiler.cmake' -Recurse -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $compilerFile) { return $null }
    $match = Select-String -LiteralPath $compilerFile.FullName -Pattern '^set\(CMAKE_CXX_COMPILER "([^"]+)"\)' | Select-Object -First 1
    if ($match) { return $match.Matches[0].Groups[1].Value }
    return $null
}

$source = Resolve-WorkspacePath -Value $SourcePath -Label 'SourcePath'
$buildDirectory = Resolve-WorkspacePath -Value $BuildPath -Label 'BuildPath'
$output = Resolve-WorkspacePath -Value $OutputPath -Label 'OutputPath'
$cmakeExe = Resolve-CMakeExecutable -RequestedPath $CmakePath
if (-not (Test-Path -LiteralPath $source -PathType Container)) { throw "SourcePath is missing: $source" }

$head = Invoke-CheckedText -FilePath 'git.exe' -Arguments @('-C', $source, 'rev-parse', 'HEAD')
if ($head -ine $Commit) {
    throw "Source HEAD $head does not match the requested pinned commit $Commit. This script never checks out or fetches commits automatically."
}
$sourceStatus = Invoke-CheckedText -FilePath 'git.exe' -Arguments @('-C', $source, 'status', '--short')
if ($sourceStatus -and -not $AllowDirtySource) {
    throw "Source tree is dirty. Commit or stash changes, or pass -AllowDirtySource explicitly."
}

$configureArguments = @(
    '-S', $source,
    '-B', $buildDirectory,
    '-G', $Generator,
    '-A', $Platform,
    '-DGGML_CUDA=ON',
    '-DGGML_CUDA_NCCL=OFF',
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures",
    '-DCMAKE_BUILD_TYPE=Release'
)
$didConfigure = $false
$didBuild = $false
if ($Configure -or $Build) {
    Write-Host "Configuring pinned llama.cpp commit $Commit for CUDA architectures $CudaArchitectures."
    & $cmakeExe @configureArguments
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed with exit code $LASTEXITCODE." }
    $didConfigure = $true
}
if ($Build) {
    & $cmakeExe '--build' $buildDirectory '--config' 'Release' '--target' $BuildTarget '--parallel'
    if ($LASTEXITCODE -ne 0) { throw "CMake build failed with exit code $LASTEXITCODE." }
    $didBuild = $true
}

$runtime = Join-Path $buildDirectory 'bin\Release\llama-server.exe'
$cmakeCache = Join-Path $buildDirectory 'CMakeCache.txt'
$compilerPath = Get-CMakeCompilerPath -BuildDirectory $buildDirectory -CachePath $cmakeCache
$manifest = [ordered]@{
    generated_utc = [DateTime]::UtcNow.ToString('o')
    source = $source
    build = $buildDirectory
    llama_cpp_sha = $head
    source_dirty = [bool]$sourceStatus
    source_status = $sourceStatus
    cuda_architectures = $CudaArchitectures
    cmake_executable = $cmakeExe
    cmake_generator = $Generator
    cmake_platform = $Platform
    build_target = $BuildTarget
    cmake_arguments = $configureArguments
    configure_requested = [bool]$Configure
    build_requested = [bool]$Build
    configure_performed = $didConfigure
    build_performed = $didBuild
    runtime_path = $runtime
    runtime_present = Test-Path -LiteralPath $runtime -PathType Leaf
    compiler_path = $compilerPath
    compiler = if ($compilerPath) { Get-CompilerText -FilePath $compilerPath } else { 'unavailable: compiler path is not present in CMake configuration' }
    cuda = Get-OptionalCommandText -FilePath 'nvcc.exe' -Arguments @('--version')
    cmake = Get-OptionalCommandText -FilePath $cmakeExe -Arguments @('--version')
    driver = Get-OptionalCommandText -FilePath 'nvidia-smi.exe' -Arguments @('--query-gpu=driver_version', '--format=csv,noheader')
    runtime_version = if (Test-Path -LiteralPath $runtime -PathType Leaf) { Get-OptionalCommandText -FilePath $runtime -Arguments @('--version') } else { $null }
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $output) | Out-Null
$manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $output -Encoding UTF8
Write-Host "Wrote build manifest: $output"
if (-not $didConfigure -and -not $didBuild) {
    Write-Host "No build action requested. Use -Configure or -Build after reviewing the pinned source tree."
}
Write-Host 'This script does not load a model.'
