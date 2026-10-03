[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runtimeRoot = Join-Path $workspace 'runtime'
$runtime = Join-Path $runtimeRoot 'strata-nvfp4'
$version = '0.1.37-nvfp4.2'
$archiveName = "strata-nvfp4-v$version-windows-x64.zip"
$archive = Join-Path $runtimeRoot $archiveName
$expectedHash = 'a16d868c4d47dce11b4d1dac806874c8955bcf19bd833e558b34160825d51289'

if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
    Invoke-WebRequest -Uri "https://github.com/sergqwer/strata-nvfp4/releases/download/v$version/$archiveName" -OutFile "$archive.partial"
    Move-Item -LiteralPath "$archive.partial" -Destination $archive
}
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ine $expectedHash) {
    throw "Strata release hash mismatch: $archive"
}
if (-not (Test-Path -LiteralPath (Join-Path $runtime 'engine\BUILD.json'))) {
    Expand-Archive -LiteralPath $archive -DestinationPath $runtimeRoot
}
$build = Get-Content -LiteralPath (Join-Path $runtime 'engine\BUILD.json') -Raw | ConvertFrom-Json
if ($build.version -ne $version -or $build.commit -ne '69a60f5' -or $build.dirty) {
    throw 'The existing runtime differs from the pinned release. Refusing to overwrite it.'
}
if ((Get-FileHash -LiteralPath (Join-Path $runtime 'engine\strata.exe')).Hash -ine $build.engine_sha256) {
    throw 'The installed Strata engine does not match BUILD.json.'
}

foreach ($role in @('convert', 'serve')) {
    $environment = Join-Path $runtime ".venv-$role"
    $python = Join-Path $environment 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        & python -m venv $environment
        if ($LASTEXITCODE -ne 0) { throw "Could not create $environment" }
    }
    & $python -m pip install --disable-pip-version-check -r (Join-Path $runtime "requirements-$role.txt")
    if ($LASTEXITCODE -ne 0) { throw "Strata $role dependency installation failed." }
    & $python -m pip check
    if ($LASTEXITCODE -ne 0) { throw "Strata $role dependency check failed." }
    & $python -m pip freeze | Set-Content -LiteralPath (Join-Path $runtime "requirements-$role-installed.txt")
}
Write-Host "Pinned Strata $version installed: $runtime"
Write-Host 'Next: run scripts/stage-qwen38-strata.py, then scripts/prepare-qwen38-strata.py.'
