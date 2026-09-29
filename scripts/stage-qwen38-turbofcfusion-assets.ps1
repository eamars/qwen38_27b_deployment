[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$destination = Join-Path $workspace 'models\qwen38-turbo-fcfusion-mtp'

# Pin the target and projector independently so staging remains reproducible.
$assets = @(
    @{
        Repo = 'DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF'
        Revision = 'ceb55042229d17dcf45df52ad84698339e63d4c5'
        File = 'Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-LOW-MTP-IQ4_XS.gguf'
        Sha256 = 'fa92183638b045b01447fd657afd6220f0995ca3a31b528078cefdf26a96c820'
        Size = 15309039136
    },
    @{
        Repo = 'unsloth/Qwen3.8-27B-GGUF'
        Revision = '4ca720788d1e01f1bff70c033e0d0028fd02e502'
        File = 'mmproj-F16.gguf'
        Sha256 = 'cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e'
        Size = 927607488
    }
)

New-Item -ItemType Directory -Force -Path $destination | Out-Null
$remaining = 0L
foreach ($asset in $assets) {
    $path = Join-Path $destination $asset.File
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        $remaining += [long]$asset.Size
    }
}
if ((Get-PSDrive -Name ([System.IO.Path]::GetPathRoot($destination).Substring(0, 1))).Free -lt ($remaining + (5L * 1GB))) {
    throw 'Insufficient free disk space for the remaining assets plus a 5 GiB safety margin.'
}

foreach ($asset in $assets) {
    $path = Join-Path $destination $asset.File
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        Write-Host "Downloading $($asset.Repo)/$($asset.File) at $($asset.Revision)"
        & python -c 'import os,sys; os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "4"); os.environ.setdefault("HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY", "1"); os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0"); from huggingface_hub import hf_hub_download; hf_hub_download(sys.argv[1], sys.argv[2], revision=sys.argv[3], local_dir=sys.argv[4])' $asset.Repo $asset.File $asset.Revision $destination
        if ($LASTEXITCODE -ne 0) {
            throw "Download failed: $($asset.File). Install Python huggingface_hub if missing."
        }
    }

    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Download did not produce the expected file: $path"
    }
    if ((Get-Item -LiteralPath $path).Length -ne [long]$asset.Size) {
        throw "Size mismatch for $path. The file is retained for inspection."
    }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ine $asset.Sha256) {
        throw "SHA256 mismatch for $path. The file is retained for inspection."
    }
    Write-Host "Verified: $path"
}

Write-Host 'Qwen3.8 Turbo Fable Cold Fusion assets staged and SHA256-verified. No model was loaded.'
