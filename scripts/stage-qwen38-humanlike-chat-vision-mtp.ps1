[CmdletBinding()]
param(
    [string]$ModelsPath = (Join-Path (Split-Path -Parent $PSScriptRoot) 'models')
)

$ErrorActionPreference = 'Stop'
$models = [System.IO.Path]::GetFullPath($ModelsPath)
$destination = Join-Path $models 'qwen38-humanlike-chat-vision-mtp'

# The merged Humanlike IQ4_XS target is text-only. Qwen3.8's MTP head and
# vision projector are unchanged by the Humanlike language-only adaptation.
$assets = @(
    @{
        Repo = 'LessThanThreeAI/Qwen3.8-27B-Humanlike-Chat-GGUF'
        Revision = 'ba6d29fb2505241d7dae88df4fda9038999c3ca9'
        File = 'Qwen3.8-27B-Humanlike-Chat-IQ4_XS.gguf'
        Sha256 = '38B417B142E0D87C90FDA9393263D523BEF17EAFC241CE070AB03797396B6850'
        Size = 15095040512L
        Role = 'Humanlike IQ4_XS target'
    },
    @{
        Repo = 'unsloth/Qwen3.8-27B-GGUF'
        Revision = '4ca720788d1e01f1bff70c033e0d0028fd02e502'
        File = 'MTP/mtp-Qwen3.8-27B-Q4_0.gguf'
        Sha256 = '50D9CE5A6DA381BBCFB31061CF73DF94A90E6FAF8EFEDDEE379A9CB8F1501C6E'
        Size = 1369590656L
        Role = 'Qwen3.8 base MTP drafter'
    },
    @{
        Repo = 'unsloth/Qwen3.8-27B-GGUF'
        Revision = '4ca720788d1e01f1bff70c033e0d0028fd02e502'
        File = 'mmproj-F16.gguf'
        Sha256 = 'CBB841A9EE0636B2EC172F5BB8DF2EA8DFEB01E90FE7C6126581D662A0B4E43E'
        Size = 927607488L
        Role = 'Qwen3.8 vision projector'
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

function Assert-Artifact([hashtable]$Asset) {
    $path = Join-Path $destination $Asset.File
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "$($Asset.Role) is missing: $path"
    }
    $info = Get-Item -LiteralPath $path
    if ($info.Length -ne [long]$Asset.Size) {
        throw "$($Asset.Role) size mismatch: got $($info.Length), expected $($Asset.Size): $path"
    }
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($hash -ne $Asset.Sha256) {
        throw "$($Asset.Role) SHA256 mismatch: got $hash, expected $($Asset.Sha256): $path"
    }
    Write-Host "$($Asset.Role) verified ($($info.Length) bytes): $path"
}

New-Item -ItemType Directory -Force -Path $destination | Out-Null
$remaining = [long](($assets | Where-Object {
    -not (Test-Path -LiteralPath (Join-Path $destination $_.File) -PathType Leaf)
} | Measure-Object -Property Size -Sum).Sum)
$driveName = [System.IO.Path]::GetPathRoot($destination).Substring(0, 1)
if ((Get-PSDrive -Name $driveName).Free -lt ($remaining + (5L * 1GB))) {
    throw 'Insufficient free disk space for remaining assets plus a 5 GiB safety margin.'
}

$hf = Get-HuggingFaceCli
$oldRangeGets = $env:HF_XET_NUM_CONCURRENT_RANGE_GETS
$oldSequentialWrites = $env:HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY
$oldChunkCacheSize = $env:HF_XET_CHUNK_CACHE_SIZE_BYTES
$env:HF_XET_NUM_CONCURRENT_RANGE_GETS = '4'
$env:HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY = '1'
$env:HF_XET_CHUNK_CACHE_SIZE_BYTES = '0'
try {
    foreach ($asset in $assets) {
        $path = Join-Path $destination $asset.File
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            Write-Host "Downloading $($asset.Repo)/$($asset.File) at $($asset.Revision)"
            & $hf download $asset.Repo --revision $asset.Revision --include $asset.File --local-dir $destination
            if ($LASTEXITCODE -ne 0) { throw "Download failed for $($asset.File)" }
        }
        Assert-Artifact $asset
    }
} finally {
    if ($null -eq $oldRangeGets) { Remove-Item Env:HF_XET_NUM_CONCURRENT_RANGE_GETS -ErrorAction SilentlyContinue }
    else { $env:HF_XET_NUM_CONCURRENT_RANGE_GETS = $oldRangeGets }
    if ($null -eq $oldSequentialWrites) { Remove-Item Env:HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY -ErrorAction SilentlyContinue }
    else { $env:HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY = $oldSequentialWrites }
    if ($null -eq $oldChunkCacheSize) { Remove-Item Env:HF_XET_CHUNK_CACHE_SIZE_BYTES -ErrorAction SilentlyContinue }
    else { $env:HF_XET_CHUNK_CACHE_SIZE_BYTES = $oldChunkCacheSize }
}

Write-Host "Humanlike Qwen3.8 IQ4_XS + base MTP + vision assets staged under $destination"
