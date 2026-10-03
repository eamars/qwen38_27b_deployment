[CmdletBinding()]
param(
    [string]$ModelsPath = (Join-Path (Split-Path -Parent $PSScriptRoot) 'models')
)

$ErrorActionPreference = 'Stop'
$drafterName = 'mtp-gemma-4-31B-it-Q8_0.gguf'
$drafterSha256 = '6B52AB20AF503AEE320DC09E93F886133B18D89FFC9075C7D9DCAF681E20B375'
$models = [System.IO.Path]::GetFullPath($ModelsPath)
$drafter = Join-Path $models $drafterName

New-Item -ItemType Directory -Force -Path $models | Out-Null
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
Write-Host 'The shared MTP sidecar is used by the retained Gemma Fable-5 Distill launcher.'
