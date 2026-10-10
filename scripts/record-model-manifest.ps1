[CmdletBinding()]
param(
    [string]$ModelsPath = (Join-Path (Split-Path -Parent $PSScriptRoot) 'models'),
    [string]$OutputPath = (Join-Path (Split-Path -Parent $PSScriptRoot) 'docs\models.md')
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

$optionalLocalAssets = @(
    @{ Repo = 'ggml-org/gemma-4-31B-it-GGUF'; File = 'mtp-gemma-4-31B-it-Q8_0.gguf'; Quant = 'MTP Q8_0'; Role = 'Gemma 4 MTP drafter' },
    @{ Repo = 'unsloth/gemma-4-31B-it-qat-GGUF'; Path = 'unsloth-gemma4-qat\gemma-4-31B-it-qat-UD-Q4_K_XL.gguf'; File = 'gemma-4-31B-it-qat-UD-Q4_K_XL.gguf'; Quant = 'QAT UD-Q4_K_XL'; Role = 'Gemma 4 4090 QAT Instruct target' },
    @{ Repo = 'unsloth/gemma-4-31B-it-qat-GGUF'; Path = 'unsloth-gemma4-qat\MTP\mtp-gemma-4-31B-it-Q8_0.gguf'; File = 'MTP/mtp-gemma-4-31B-it-Q8_0.gguf'; Quant = 'MTP Q8_0'; Role = 'Gemma 4 QAT MTP drafter' },
    @{ Repo = 'HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP'; Path = 'hauhaucs-gemma4-qat-uncensored-balanced-mtp\Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf'; File = 'Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf'; Quant = 'QAT Q4_K_M'; Role = 'Gemma 4 4090 HauhauCS uncensored target' },
    @{ Repo = 'HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP'; Path = 'hauhaucs-gemma4-qat-uncensored-balanced-mtp\mtp-gemma-4-31B-it.gguf'; File = 'mtp-gemma-4-31B-it.gguf'; Quant = 'MTP'; Role = 'Gemma 4 HauhauCS MTP drafter' },
    @{ Repo = 'HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP'; Path = 'hauhaucs-gemma4-qat-uncensored-balanced-mtp\mmproj-Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf'; File = 'mmproj-Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf'; Quant = 'BF16 mmproj'; Role = 'Gemma 4 HauhauCS vision projector' },
    @{ Repo = 'DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF'; Path = 'qwen38-turbo-fcfusion-mtp\Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-LOW-MTP-IQ4_XS.gguf'; File = 'Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-LOW-MTP-IQ4_XS.gguf'; Quant = 'LOW-MTP-IQ4_XS'; Role = 'RTX 4090 Turbo Fable Cold Fusion vision/MTP target' },
    @{ Repo = 'unsloth/Qwen3.8-27B-GGUF'; Path = 'qwen38-turbo-fcfusion-mtp\mmproj-F16.gguf'; File = 'mmproj-F16.gguf'; Quant = 'F16 mmproj'; Role = 'Qwen3.8 Turbo Fable Cold Fusion vision projector' }
)
foreach ($item in $optionalLocalAssets) {
    $relativePath = if ($item.Path) { $item.Path } else { $item.File }
    if (Test-Path -LiteralPath (Join-Path $models $relativePath) -PathType Leaf) {
        $items += $item
    }
}

$lines = @(
    '# Model manifest'
    ''
    "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
    ''
    'All model artifacts are local under `models/` and are ignored by Git. SHA-256'
    'values below were calculated from the completed files in this workspace. The'
    'Qwen rows are the deployment set; Gemma rows are experimental and are included'
    'when those files are present. Regenerate this file with'
    '`scripts/record-model-manifest.ps1` after replacing an artifact.'
    ''
    'The native Windows Strata checkpoint and generated assets are under'
    '`models/qwen38-flash-next-uncensored-strata/`; see the'
    '[Strata runbook](qwen38-flash-next-strata.md) and'
    '[checkpoint manifest](qwen38-strata-checkpoint-manifest.json).'
    ''
    '| Role | Repository | Filename | Quantization | Size bytes | Size GB | SHA-256 | Download date |'
    '|---|---|---|---|---:|---:|---|---|'
)

foreach ($item in $items) {
    $relativePath = if ($item.Path) { $item.Path } else { $item.File }
    $path = Join-Path $models $relativePath
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
