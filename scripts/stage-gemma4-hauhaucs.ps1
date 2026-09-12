[CmdletBinding()]
param(
    [string]$ModelsPath = (Join-Path (Split-Path -Parent $PSScriptRoot) 'models')
)

$ErrorActionPreference = 'Stop'
$repo = 'HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP'
$revision = '9654466e82d83f5ebfe1518a369bc5900873abb1'
$models = [System.IO.Path]::GetFullPath($ModelsPath)
$targetDirectory = Join-Path $models 'hauhaucs-gemma4-qat-uncensored-balanced-mtp'
$files = @(
    @{
        Name = 'Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf'
        Bytes = 18687062176
        Sha256 = '71667F9E601A4B914A98425C59150B731F6E15D260D661DBD1F1EE07469FC7DB'
        Role = 'target'
    }
    @{
        Name = 'mtp-gemma-4-31B-it.gguf'
        Bytes = 279954368
        Sha256 = 'B5C4E583FC5982439080114BBC1B7EDAEC361F9D4C9193D6BED606A3DE401B62'
        Role = 'MTP drafter'
    }
    @{
        Name = 'mmproj-Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf'
        Bytes = 1200726016
        Sha256 = '7BEF0D0FB3E85FC2941EC5F1C375FEBF3742645F158132A43CED557093AEA841'
        Role = 'vision projector'
    }
)

function Get-HuggingFaceCli {
    $command = Get-Command hf.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }

    $candidates = @(
        (Join-Path $env:APPDATA 'Python\Python312\Scripts\hf.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\Scripts\hf.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    throw "Hugging Face CLI 'hf.exe' was not found. Install huggingface_hub, then rerun this script."
}

function Assert-Artifact([hashtable]$Expected) {
    $path = Join-Path $targetDirectory $Expected.Name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "$($Expected.Role) is missing: $path"
    }
    $file = Get-Item -LiteralPath $path
    if ($file.Length -ne $Expected.Bytes) {
        throw "$($Expected.Role) has size $($file.Length) bytes; expected $($Expected.Bytes): $path"
    }
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($hash -ne $Expected.Sha256) {
        throw "$($Expected.Role) SHA256 mismatch: got $hash, expected $($Expected.Sha256)"
    }
    Write-Host "$($Expected.Role) verified: $($Expected.Name) ($($file.Length) bytes)"
}

New-Item -ItemType Directory -Force -Path $targetDirectory | Out-Null
$hf = Get-HuggingFaceCli
Write-Host "Staging $repo at immutable revision $revision"
foreach ($file in $files) {
    & $hf download $repo --revision $revision --include $file.Name --local-dir $targetDirectory
    if ($LASTEXITCODE -ne 0) { throw "Download failed for $($file.Name)" }
}

foreach ($file in $files) { Assert-Artifact $file }
Write-Host "HauhauCS Gemma 4 assets staged under $targetDirectory"
