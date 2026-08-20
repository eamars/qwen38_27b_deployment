[CmdletBinding()]
param(
    [switch]$IncludeFallbacks,
    [string]$ModelsPath = (Join-Path (Split-Path -Parent $PSScriptRoot) 'models')
)

$ErrorActionPreference = 'Stop'
$models = [System.IO.Path]::GetFullPath($ModelsPath)
New-Item -ItemType Directory -Force -Path $models | Out-Null

$hf = Get-Command hf.exe -ErrorAction SilentlyContinue
if (-not $hf) {
    $userScripts = Join-Path $env:APPDATA 'Python\Python312\Scripts'
    $candidate = Join-Path $userScripts 'hf.exe'
    if (Test-Path -LiteralPath $candidate) { $hf = Get-Item -LiteralPath $candidate }
}
if (-not $hf) {
    throw "Hugging Face CLI 'hf.exe' was not found. Install the public huggingface_hub package, then rerun this script. No model was loaded."
}
$hfPath = if ($hf.PSObject.Properties.Name -contains 'Source' -and $hf.Source) { $hf.Source } else { $hf.FullName }
if (-not $hfPath) { throw 'Could not resolve the Hugging Face CLI path.' }

$downloads = @(
    @{ Repo = 'unsloth/Qwen3.8-27B-GGUF'; File = 'Qwen3.8-27B-UD-Q6_K_M.gguf' },
    @{ Repo = 'unsloth/Qwen3.8-27B-GGUF'; File = 'Qwen3.8-27B-UD-Q4_K_XL.gguf' },
    @{ Repo = 'incoai/Qwen3.8-27B-DFlash2-GGUF'; File = 'Qwen3.8-27B-DFlash2-Q4_K_M.gguf' }
)
if ($IncludeFallbacks) {
    $downloads += @{ Repo = 'unsloth/Qwen3.8-27B-GGUF'; File = 'Qwen3.8-27B-UD-Q6_K.gguf' }
    $downloads += @{ Repo = 'unsloth/Qwen3.8-27B-GGUF'; File = 'Qwen3.8-27B-UD-Q4_K_M.gguf' }
}

foreach ($item in $downloads) {
    $expected = Join-Path $models $item.File
    if (Test-Path -LiteralPath $expected) {
        Write-Host "Already present: $($item.File)"
        continue
    }
    Write-Host "Downloading $($item.Repo)/$($item.File) -> $models"
    & $hfPath download $item.Repo $item.File --local-dir $models
    if ($LASTEXITCODE -ne 0) { throw "Hugging Face download failed for $($item.File)" }
}

Write-Host 'Model download stage complete. This script only downloads files; it does not start a server or load a model.'
