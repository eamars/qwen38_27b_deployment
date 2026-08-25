[CmdletBinding()]
param(
    [string]$OutputPath,
    [ValidateRange(1, 16)][int]$ParallelDownloads = 8,
    [ValidateRange(16777216, 1073741824)][int64]$ChunkBytes = 268435456
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$targetDirectory = Join-Path $workspace 'models'
$expectedBytes = [int64]17763168256
$downloadUrl = 'https://huggingface.co/mradermacher/Gemma-4-31B-Isometry-Fabled-Persona-i1-GGUF/resolve/main/Gemma-4-31B-Isometry-Fabled-Persona.i1-Q4_K_S.gguf?download=true'

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $targetDirectory 'Gemma-4-31B-Isometry-Fabled-Persona.i1-Q4_K_S.gguf'
} else {
    $OutputPath = [System.IO.Path]::GetFullPath($OutputPath)
}

try {
    $outputRelative = [System.IO.Path]::GetRelativePath($workspace, $OutputPath)
} catch {
    throw "OutputPath must be inside the project: $OutputPath"
}
if ($outputRelative.StartsWith('..' + [System.IO.Path]::DirectorySeparatorChar) -or
    [System.IO.Path]::IsPathRooted($outputRelative)) {
    throw "OutputPath must be inside the project: $OutputPath"
}
if (-not $OutputPath.EndsWith('Gemma-4-31B-Isometry-Fabled-Persona.i1-Q4_K_S.gguf', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'This helper is intentionally restricted to the exact i1-Q4_K_S persona GGUF.'
}

New-Item -ItemType Directory -Force -Path $targetDirectory | Out-Null
$downloadStage = Join-Path $targetDirectory '.q4s-download'
$curl = Get-Command curl.exe -ErrorAction SilentlyContinue
if (-not $curl) { throw 'curl.exe was not found in PATH.' }

if (Test-Path -LiteralPath $OutputPath) {
    $existing = Get-Item -LiteralPath $OutputPath
    if ($existing.Length -eq $expectedBytes) {
        Write-Host "The exact target already has the expected size: $($existing.Length) bytes."
        Write-Host "SHA256: $((Get-FileHash -LiteralPath $OutputPath -Algorithm SHA256).Hash)"
        exit 0
    }
    throw "The final target exists with an unexpected size ($($existing.Length) bytes): $OutputPath"
}

New-Item -ItemType Directory -Force -Path $downloadStage | Out-Null
$partCount = [int][math]::Ceiling($expectedBytes / [double]$ChunkBytes)
$activeJobs = @()
$completedParts = 0

try {
    for ($partIndex = 0; $partIndex -lt $partCount; $partIndex++) {
        $partStart = [int64]$partIndex * $ChunkBytes
        $partEnd = [math]::Min($expectedBytes - 1, $partStart + $ChunkBytes - 1)
        $partPath = Join-Path $downloadStage ('part-{0:D4}.bin' -f $partIndex)
        $partExpected = $partEnd - $partStart + 1

        if (Test-Path -LiteralPath $partPath) {
            $partItem = Get-Item -LiteralPath $partPath
            if ($partItem.Length -eq $partExpected) {
                $completedParts++
                Write-Host ("Already complete: part {0}/{1}" -f ($partIndex + 1), $partCount)
                continue
            }
            Remove-Item -LiteralPath $partPath -Force
        }

        $activeJobs += Start-Job -ScriptBlock {
            param($jobUrl, $jobRange, $jobPath, $curlPath)
            & $curlPath --location --fail --retry 10 --retry-delay 5 --retry-all-errors `
                --silent --show-error --range $jobRange --output $jobPath $jobUrl
            if ($LASTEXITCODE -ne 0) {
                throw "curl failed for range $jobRange with exit code $LASTEXITCODE"
            }
        } -ArgumentList @(
            $downloadUrl,
            ('{0}-{1}' -f $partStart, $partEnd),
            $partPath,
            $curl.Source
        )

        while ($activeJobs.Count -ge $ParallelDownloads) {
            $finishedJob = Wait-Job -Job $activeJobs[0]
            Receive-Job -Job $finishedJob -ErrorAction Stop | Out-Host
            if ($finishedJob.State -ne 'Completed') {
                throw "Download job failed: $($finishedJob.State)"
            }
            Remove-Job -Job $finishedJob -Force
            $activeJobs = @($activeJobs | Select-Object -Skip 1)
            $completedParts++
            Write-Host ("Downloaded {0}/{1} parts" -f $completedParts, $partCount)
        }
    }

    foreach ($remainingJob in @($activeJobs)) {
        $finishedJob = Wait-Job -Job $remainingJob
        Receive-Job -Job $finishedJob -ErrorAction Stop | Out-Host
        if ($finishedJob.State -ne 'Completed') {
            throw "Download job failed: $($finishedJob.State)"
        }
        Remove-Job -Job $finishedJob -Force
        $completedParts++
        Write-Host ("Downloaded {0}/{1} parts" -f $completedParts, $partCount)
    }
    $activeJobs = @()

    $partFiles = @(Get-ChildItem -LiteralPath $downloadStage -Filter 'part-*.bin' -File | Sort-Object Name)
    if ($partFiles.Count -ne $partCount) {
        throw "Expected $partCount parts, found $($partFiles.Count)."
    }
    for ($checkIndex = 0; $checkIndex -lt $partCount; $checkIndex++) {
        $checkStart = [int64]$checkIndex * $ChunkBytes
        $checkEnd = [math]::Min($expectedBytes - 1, $checkStart + $ChunkBytes - 1)
        $checkExpected = $checkEnd - $checkStart + 1
        $checkFile = Get-Item -LiteralPath (Join-Path $downloadStage ('part-{0:D4}.bin' -f $checkIndex))
        if ($checkFile.Length -ne $checkExpected) {
            throw "Part size mismatch for $($checkFile.Name): expected $checkExpected, got $($checkFile.Length)."
        }
    }

    $assembledPath = Join-Path $downloadStage 'Gemma-4-31B-Isometry-Fabled-Persona.i1-Q4_K_S.gguf.assembled'
    if (Test-Path -LiteralPath $assembledPath) {
        Remove-Item -LiteralPath $assembledPath -Force
    }
    $outputStream = [System.IO.File]::Open(
        $assembledPath,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None
    )
    try {
        foreach ($partFile in $partFiles) {
            $inputStream = [System.IO.File]::OpenRead($partFile.FullName)
            try {
                $inputStream.CopyTo($outputStream, 4194304)
            } finally {
                $inputStream.Dispose()
            }
        }
    } finally {
        $outputStream.Dispose()
    }

    $assembledItem = Get-Item -LiteralPath $assembledPath
    if ($assembledItem.Length -ne $expectedBytes) {
        throw "Assembled size mismatch: expected $expectedBytes, got $($assembledItem.Length)."
    }
    $sha256 = (Get-FileHash -LiteralPath $assembledPath -Algorithm SHA256).Hash
    Move-Item -LiteralPath $assembledPath -Destination $OutputPath
    Write-Host "Downloaded exact persona quant: $OutputPath"
    Write-Host "Bytes: $expectedBytes"
    Write-Host "SHA256: $sha256"
} catch {
    foreach ($activeJob in @($activeJobs)) {
        Stop-Job -Job $activeJob -ErrorAction SilentlyContinue
        Remove-Job -Job $activeJob -Force -ErrorAction SilentlyContinue
    }
    throw
}

Remove-Item -LiteralPath $downloadStage -Recurse -Force
Write-Host 'Removed temporary ranged-download staging files.'
