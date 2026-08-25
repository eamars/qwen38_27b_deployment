[CmdletBinding()]
param(
    [string]$LmStudioTargetPath = 'D:\lm_models\mradermacher\Gemma-4-31B-Isometry-Fabled-Persona-i1-GGUF\Gemma-4-31B-Isometry-Fabled-Persona.i1-Q4_K_M.gguf',
    [string]$ModelsPath = (Join-Path (Split-Path -Parent $PSScriptRoot) 'models')
)

$ErrorActionPreference = 'Stop'
$targetName = 'Gemma-4-31B-Isometry-Fabled-Persona.i1-Q4_K_M.gguf'
$drafterName = 'mtp-gemma-4-31B-it-Q8_0.gguf'
$drafterSha256 = '6B52AB20AF503AEE320DC09E93F886133B18D89FFC9075C7D9DCAF681E20B375'
$models = [System.IO.Path]::GetFullPath($ModelsPath)
$target = Join-Path $models $targetName
$drafter = Join-Path $models $drafterName

New-Item -ItemType Directory -Force -Path $models | Out-Null
if (-not (Test-Path -LiteralPath $LmStudioTargetPath -PathType Leaf)) {
    throw "The exact LM Studio persona target was not found: $LmStudioTargetPath"
}

if (Test-Path -LiteralPath $target -PathType Leaf) {
    $sourceInfo = Get-Item -LiteralPath $LmStudioTargetPath
    $targetInfo = Get-Item -LiteralPath $target
    if ($sourceInfo.Length -ne $targetInfo.Length) {
        throw "A different target already exists at $target; refusing to overwrite it."
    }
    Write-Host "Target already present with matching size: $target"
} else {
    Write-Host "Copying only the requested Isometry-Fabled-Persona target."
    Copy-Item -LiteralPath $LmStudioTargetPath -Destination $target
}

$sourceHash = (Get-FileHash -LiteralPath $LmStudioTargetPath -Algorithm SHA256).Hash
$targetHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
if ($sourceHash -ne $targetHash) { throw "Target SHA256 mismatch after staging: $target" }
Write-Host "Target staged: $target"
Write-Host "Target SHA256: $targetHash"

if (-not (Test-Path -LiteralPath $drafter -PathType Leaf)) {
    $hf = Get-Command hf.exe -ErrorAction SilentlyContinue
    if (-not $hf) {
        $userScripts = Join-Path $env:APPDATA 'Python\Python312\Scripts'
        $candidate = Join-Path $userScripts 'hf.exe'
        if (Test-Path -LiteralPath $candidate) { $hf = Get-Item -LiteralPath $candidate }
    }
    if (-not $hf) {
        throw "Hugging Face CLI 'hf.exe' was not found. Install huggingface_hub, then rerun this script."
    }
    $hfPath = if ($hf.PSObject.Properties.Name -contains 'Source' -and $hf.Source) { $hf.Source } else { $hf.FullName }
    Write-Host "Downloading only the official Google MTP sidecar: $drafterName"
    & $hfPath download ggml-org/gemma-4-31B-it-GGUF $drafterName --local-dir $models
    if ($LASTEXITCODE -ne 0) { throw "MTP drafter download failed." }
}

$drafterHash = (Get-FileHash -LiteralPath $drafter -Algorithm SHA256).Hash
if ($drafterHash -ne $drafterSha256) { throw "MTP drafter SHA256 mismatch: $drafterHash" }
Write-Host "MTP drafter staged: $drafter"
Write-Host "MTP drafter SHA256: $drafterHash"
Write-Host 'No unmodified Gemma target is downloaded by this script.'
