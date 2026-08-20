[CmdletBinding()]
param(
    [string]$ModelsPath = (Join-Path (Split-Path -Parent $PSScriptRoot) 'models'),
    [string]$OutputPath = (Join-Path (Split-Path -Parent $PSScriptRoot) 'docs\model-manifest.md')
)

$ErrorActionPreference = 'Stop'
$models = [System.IO.Path]::GetFullPath($ModelsPath)
$output = [System.IO.Path]::GetFullPath($OutputPath)
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $output) | Out-Null

$items = @(
    @{ Repo = 'unsloth/Qwen3.8-27B-GGUF'; File = 'Qwen3.8-27B-UD-Q6_K_M.gguf'; Quant = 'UD-Q6_K_M'; Role = 'RTX 5090 primary target' },
    @{ Repo = 'unsloth/Qwen3.8-27B-GGUF'; File = 'Qwen3.8-27B-UD-Q6_K.gguf'; Quant = 'UD-Q6_K'; Role = 'RTX 5090 context fallback' },
    @{ Repo = 'unsloth/Qwen3.8-27B-GGUF'; File = 'Qwen3.8-27B-UD-Q4_K_XL.gguf'; Quant = 'UD-Q4_K_XL'; Role = 'RTX 4090 primary target' },
    @{ Repo = 'unsloth/Qwen3.8-27B-GGUF'; File = 'Qwen3.8-27B-UD-Q4_K_M.gguf'; Quant = 'UD-Q4_K_M'; Role = 'RTX 4090 context fallback' },
    @{ Repo = 'incoai/Qwen3.8-27B-DFlash2-GGUF'; File = 'Qwen3.8-27B-DFlash2-Q4_K_M.gguf'; Quant = 'DFlash2 Q4_K_M'; Role = 'DFlash2 drafter for both backends' }
)

$lines = @(
    '# Model manifest'
    ''
    "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
    ''
    'All model artifacts are local under models/ and are ignored by Git. SHA-256 values below are calculated from the completed files in this workspace.'
    ''
    '| Role | Repository | Filename | Quantization | Size bytes | Size GB | SHA-256 | Download date |'
    '|---|---|---|---|---:|---:|---|---|'
)

foreach ($item in $items) {
    $path = Join-Path $models $item.File
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing model file: $path" }
    Write-Host "Hashing $($item.File)"
    $file = Get-Item -LiteralPath $path
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    $sizeGb = [math]::Round([double]$file.Length / 1GB, 3)
    $downloadDate = $file.LastWriteTime.ToString('yyyy-MM-dd')
    $lines += "| $($item.Role) | $($item.Repo) | $($item.File) | $($item.Quant) | $($file.Length) | $sizeGb | $hash | $downloadDate |"
}

Set-Content -LiteralPath $output -Value ($lines -join "`n") -Encoding UTF8
Write-Host "Wrote model manifest: $output"
