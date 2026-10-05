# Capture the ARP failure between the Windows primary NIC and the DSH host.
# Run in an Administrator PowerShell. Does not change networking or Strata.
[CmdletBinding()]
param(
    [ValidateRange(20, 60)]
    [int]$CaptureSeconds = 30
)

$ErrorActionPreference = 'Stop'
$principal = [Security.Principal.WindowsPrincipal]::new(
    [Security.Principal.WindowsIdentity]::GetCurrent()
)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Packet Monitor requires an Administrator PowerShell. No capture or network changes were made.'
}

# Packet Monitor has global capture/filter state. Refuse to replace an existing
# session or filters; unknown/localized status text also fails closed.
$statusText = (& pktmon.exe status 2>&1 | Out-String)
if ($statusText -notmatch '(?i)not running|not started|stopped') {
    throw "Packet Monitor is not confirmed idle. Existing state was left alone:`n$statusText"
}
$filterText = (& pktmon.exe filter list 2>&1 | Out-String)
if ($filterText -notmatch '(?im)^\s*(None|<none>|No (packet )?filters\.?)\s*$') {
    throw "Packet Monitor filters are not confirmed empty. Existing filters were left alone:`n$filterText"
}

$workspaceRoot = Split-Path -Parent $PSScriptRoot
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$captureDir = Join-Path $workspaceRoot "benchmarks\raw\qwen38-strata\network\arp-$stamp"
$null = New-Item -ItemType Directory -Path $captureDir
$etlPath = Join-Path $captureDir 'arp.etl'
$textPath = Join-Path $captureDir 'arp.txt'
$probePath = Join-Path $captureDir 'probes.txt'

function Save-NetworkState([string]$Label) {
    $state = [ordered]@{
        Time = (Get-Date).ToString('o')
        Adapters = @(Get-NetAdapter -Name 'Ethernet', 'Ethernet 2' |
            Select-Object Name, Status, LinkSpeed, MacAddress, ifIndex)
        Addresses = @(Get-NetIPAddress -AddressFamily IPv4 |
            Where-Object { $_.IPAddress -in @('192.168.2.13', '192.168.2.232') } |
            Select-Object InterfaceAlias, InterfaceIndex, IPAddress, PrefixLength, PrefixOrigin, AddressState)
        Neighbours = @(Get-NetNeighbor -AddressFamily IPv4 |
            Where-Object { $_.IPAddress -in @('192.168.2.10', '192.168.2.1') } |
            Select-Object InterfaceAlias, InterfaceIndex, IPAddress, LinkLayerAddress, State)
    }
    $state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $captureDir "$Label.json") -Encoding UTF8
}

Save-NetworkState 'before'
$filterAdded = $false
$captureStarted = $false
try {
    & pktmon.exe filter add "StrataARP-$stamp" -d ARP
    if ($LASTEXITCODE -ne 0) { throw 'Could not add the ARP capture filter.' }
    $filterAdded = $true

    & pktmon.exe start --capture --comp all --pkt-size 64 --file-size 32 --log-mode circular --file-name $etlPath
    if ($LASTEXITCODE -ne 0) { throw 'Could not start Packet Monitor.' }
    $captureStarted = $true
    $timer = [Diagnostics.Stopwatch]::StartNew()
    Write-Host "Capturing ARP for approximately $CaptureSeconds seconds. Optional Linux probe: curl --connect-timeout 5 http://192.168.2.13:1919/v1/models"

    foreach ($sourceAddress in @('192.168.2.13', '192.168.2.232')) {
        foreach ($targetAddress in @('192.168.2.10', '192.168.2.1')) {
            "Source $sourceAddress -> $targetAddress" | Add-Content -LiteralPath $probePath
            & ping.exe -4 -n 2 -w 1000 -S $sourceAddress $targetAddress 2>&1 |
                Tee-Object -FilePath $probePath -Append
        }
    }
    $remainingSeconds = [Math]::Ceiling($CaptureSeconds - $timer.Elapsed.TotalSeconds)
    if ($remainingSeconds -gt 0) { Start-Sleep -Seconds ([int]$remainingSeconds) }
}
finally {
    if ($captureStarted) {
        & pktmon.exe stop
        if ($LASTEXITCODE -ne 0) { throw 'Packet Monitor could not stop. Run pktmon stop in this Administrator terminal.' }
    }
    if ($filterAdded) {
        # The preflight confirmed there were no filters before this capture.
        & pktmon.exe filter remove
        if ($LASTEXITCODE -ne 0) { Write-Warning 'The temporary capture filter could not be removed.' }
    }
}

Save-NetworkState 'after'
& pktmon.exe etl2txt $etlPath --out $textPath --verbose
if ($LASTEXITCODE -ne 0) { throw "Capture saved to $etlPath, but text conversion failed." }
Write-Host "Capture complete: $captureDir"
