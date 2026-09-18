[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$destination = Join-Path (Split-Path -Parent $PSScriptRoot) 'models\unsloth-gemma4-26b-qat'
# Pin all three assets to one revision; verify the publisher's LFS SHA256 values.
$revision = '7b92b5b28818151e8669af2e45e88d6086f490dd'
$files = [ordered]@{
    'gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf' = 'a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891'
    'MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf' = '75f58e20c32281d1adc086c0569d53f147e8d81a7b8493cf4a73a028415f67af'
    'mmproj-F16.gguf' = 'd00f211a7d4f7fb19bd9b75d8e9342eccffb5920b08fd9562167560fdfcc5dd1'
}

foreach ($name in $files.Keys) {
    $path = Join-Path $destination $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        # Requires Python with huggingface_hub; downloads resume via its local cache.
        & python -c 'import sys; from huggingface_hub import hf_hub_download; hf_hub_download("unsloth/gemma-4-26B-A4B-it-qat-GGUF", sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])' $name $revision $destination
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $name. Install Python huggingface_hub if missing." }
    }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ine $files[$name]) {
        throw "SHA256 mismatch: $path. Refusing to use or overwrite this file."
    }
    Write-Host "Verified: $path"
}
