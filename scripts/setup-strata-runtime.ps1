[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runtimeRoot = Join-Path $workspace 'runtime'
$runtime = Join-Path $runtimeRoot 'strata-nvfp4'
$version = '0.1.40.2-nvfp4.1'
$commit = 'd147ca4f'
$archiveName = "strata-nvfp4-v$version-windows-x64.zip"
$archive = Join-Path $runtimeRoot $archiveName
$expectedHash = '75a12607946b834d128dd4ef03955ca8209a3c8796e4754e4fc545bd39d9b706'
# Earlier pinned releases upgraded in place, as the release notes direct ("unzip over the previous folder"): the
# virtual environments, configs and models outside the zip stay. Their requirements, data files and the converters
# behind the prepared model assets are identical to this release's (checked 2026-10-06 and 2026-10-08; 0.1.40.2
# only adds PLE key formats to tools/iq_pack.py that this model's BF16 key does not use).
$upgradeFrom = @('0.1.37-nvfp4.2', '0.1.39-nvfp4.3')

if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
    Invoke-WebRequest -Uri "https://github.com/sergqwer/strata-nvfp4/releases/download/v$version/$archiveName" -OutFile "$archive.partial"
    Move-Item -LiteralPath "$archive.partial" -Destination $archive
}
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ine $expectedHash) {
    throw "Strata release hash mismatch: $archive"
}
$buildPath = Join-Path $runtime 'engine\BUILD.json'
if (-not (Test-Path -LiteralPath $buildPath)) {
    Expand-Archive -LiteralPath $archive -DestinationPath $runtimeRoot
} else {
    $installed = Get-Content -LiteralPath $buildPath -Raw | ConvertFrom-Json
    if ($installed.version -ne $version -and $installed.version -in $upgradeFrom -and -not $installed.dirty) {
        if (@(Get-Process -Name strata, strata-vision -ErrorAction SilentlyContinue).Count -gt 0) {
            throw 'Stop the running Strata servers before upgrading the runtime.'
        }
        Write-Host "Upgrading Strata $($installed.version) to $version in place; virtual environments and models stay."
        Expand-Archive -LiteralPath $archive -DestinationPath $runtimeRoot -Force
    }
}
$build = Get-Content -LiteralPath $buildPath -Raw | ConvertFrom-Json
if ($build.version -ne $version -or $build.commit -ne $commit -or $build.dirty) {
    throw 'The existing runtime differs from the pinned release. Refusing to overwrite it.'
}
if ((Get-FileHash -LiteralPath (Join-Path $runtime 'engine\strata.exe')).Hash -ine $build.engine_sha256) {
    throw 'The installed Strata engine does not match BUILD.json.'
}

# Local change to the pinned server: also read the thinking budget from `thinking_token_budget`, the field
# pi-ai/DSH clients send (compat.thinkingTokenBudgetField). Without it their budget is ignored and one step can
# think through all of its output tokens. Reapplied here because runtime/ is ignored, freshly extracted code.
$serverPath = Join-Path $runtime 'serve\server.py'
$server = [System.IO.File]::ReadAllText($serverPath)
if (-not $server.Contains('req.get("thinking_token_budget")')) {
    $anchor = [regex]::Matches($server,
        '(?m)^( *)value = \(req or \{\}\)\.get\("reasoning_budget_tokens"\) if isinstance\(req, dict\) else None(\r?\n)')
    if ($anchor.Count -ne 1) { throw "Cannot apply the thinking_token_budget alias: $serverPath has changed." }
    $indent = $anchor[0].Groups[1].Value
    $newline = $anchor[0].Groups[2].Value
    $alias = "${indent}if value is None and isinstance(req, dict):$newline" +
        "$indent    # The same budget under the name pi-ai/DSH clients send (compat.thinkingTokenBudgetField).$newline" +
        "$indent    value = req.get(`"thinking_token_budget`")$newline"
    $server = $server.Insert($anchor[0].Index + $anchor[0].Length, $alias)
    [System.IO.File]::WriteAllText($serverPath, $server, [System.Text.UTF8Encoding]::new($false))
    Write-Host "Applied the thinking_token_budget alias: $serverPath"
}

# Local change to the pinned server: a </think> the model writes while quoting the tag, or again inside its answer,
# no longer leaks reasoning and a repeated reply into the answer (upstream #537, #1053). Idempotent; an earlier
# version of the patch is replaced from the release archive. See the script.
& python (Join-Path $PSScriptRoot 'patches\strata-think-echo.py') (Join-Path $runtime 'serve\frontend.py') $archive
if ($LASTEXITCODE -ne 0) { throw 'Cannot apply strata-think-echo to the Strata server.' }

foreach ($role in @('convert', 'serve')) {
    $environment = Join-Path $runtime ".venv-$role"
    $python = Join-Path $environment 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        & python -m venv $environment
        if ($LASTEXITCODE -ne 0) { throw "Could not create $environment" }
    }
    & $python -m pip install --disable-pip-version-check -r (Join-Path $runtime "requirements-$role.txt")
    if ($LASTEXITCODE -ne 0) { throw "Strata $role dependency installation failed." }
    if ($role -eq 'serve') {
        # Local addition: the server converts images its vision decoder cannot read (WebP, which DSH sends for
        # transparent pictures) with Pillow, and refuses the whole request without it.
        & $python -m pip install --disable-pip-version-check pillow
        if ($LASTEXITCODE -ne 0) { throw 'Pillow installation for the Strata server failed.' }
    }
    & $python -m pip check
    if ($LASTEXITCODE -ne 0) { throw "Strata $role dependency check failed." }
    & $python -m pip freeze | Set-Content -LiteralPath (Join-Path $runtime "requirements-$role-installed.txt")
}
Write-Host "Pinned Strata $version installed: $runtime"
Write-Host 'Next: run scripts/stage-qwen38-strata.py, then scripts/prepare-qwen38-strata.py.'
