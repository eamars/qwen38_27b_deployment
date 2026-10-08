[CmdletBinding()]
param(
    [ValidateSet('Short4K', 'Native256K')]
    [string]$Profile = 'Native256K',
    [string]$GpuUuid = 'GPU-67921d1c-ee8e-304f-b562-d6f87617c5a0',
    [ValidateSet('Single', 'LayerSplit', 'ExpertHelper')]
    [string]$GpuMode = 'Single',
    [string]$SecondaryGpuUuid = 'GPU-eed52936-813f-8d68-1654-bfb56cb42bc3',
    [ValidatePattern('^(auto|[2-9]|[1-3][0-9]|4[0-6])$')]
    [string]$LayerSplit = 'auto',
    [ValidateRange(1, 16000)]
    [int]$HelperExpertSlots = 8000,
    [ValidateRange(1, 65535)]
    [int]$Port = 1919,
    # Match the older Flash launchers: listen on the shared port from the LAN.
    [string]$BindAddress = '0.0.0.0',
    [ValidateRange(0, 120)]
    [int]$RamBudgetGiB = 0,
    # An allocation margin, not a runtime free-VRAM floor. Measured free VRAM differs.
    [ValidateRange(1024, 16384)]
    [int]$VramReserveMiB = 2048,
    # Host RAM snapshots preserve independent histories across sequential requests. Live traffic filled all 8
    # earlier slots with 7-14 GB parked and evicted 291 histories, so slots, not bytes, were the limit.
    [ValidateRange(0, 32768)]
    [int]$ConversationCacheMiB = 24576,
    [ValidateRange(1, 16)]
    [int]$ConversationCacheSlots = 16,
    [switch]$NoMtp,
    [switch]$DryRun,
    [switch]$Stop
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runtime = Join-Path $workspace 'runtime\strata-nvfp4'
$model = Join-Path $workspace 'models\qwen38-flash-next-uncensored-strata'
$python = Join-Path $runtime '.venv-serve\Scripts\python.exe'
$configPath = Join-Path $runtime "config\workspace-$Port.json"
$logDirectory = Join-Path $workspace 'benchmarks\raw\qwen38-strata\server'
$servedModel = 'qwen38-next-uncensored-strata-vision'

function Get-ManagedServer {
    $configPattern = '(?:^|\s)--config\s+"?' + [regex]::Escape($configPath) + '"?(?=\s|$)'
    @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
        $_.CommandLine -match '(?:^|\s)-m\s+serve\.server(?=\s|$)' -and
        $_.CommandLine -match $configPattern
    })
}

if ($Stop) {
    if ($DryRun) { throw 'Use -Stop or -DryRun, not both.' }
    $managed = @(Get-ManagedServer)
    if ($managed.Count -eq 0) {
        Write-Host "No managed Strata server found on port $Port."
    }
    $managedIds = @($managed | ForEach-Object { $_.ProcessId })
    $children = @(Get-CimInstance Win32_Process -Filter "Name = 'strata.exe' OR Name = 'strata-vision.exe'" |
        Where-Object { $_.ParentProcessId -in $managedIds } |
        ForEach-Object { Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue })
    foreach ($process in $managed) {
        # Upstream keeps the engine and CPU vision encoder in a kill-on-close job.
        Stop-Process -Id $process.ProcessId -ErrorAction SilentlyContinue
        Write-Host "Stopped Strata server PID $($process.ProcessId) on port $Port."
    }
    # Releasing the 63 GiB pinned arena takes several seconds even after job termination.
    $stopClock = [System.Diagnostics.Stopwatch]::StartNew()
    foreach ($child in $children) {
        $remainingMs = [math]::Max(0, 30000 - [int]$stopClock.ElapsedMilliseconds)
        if (-not $child.WaitForExit($remainingMs)) {
            throw "Strata child PID $($child.Id) is still releasing resources after 30 seconds."
        }
    }
    return
}

# STRATA_API_KEY remains optional, matching the existing LAN Flash deployments.
# The upstream server reads it directly when the caller sets it.
if ($GpuMode -ne 'Single' -and $RamBudgetGiB -gt 0) {
    throw 'The pinned engine does not support a RAM-budget tier with layer splitting or helper GPUs.'
}
if ($GpuMode -eq 'LayerSplit' -and $ConversationCacheMiB -gt 0) {
    throw 'The pinned engine does not support conversation snapshots with layer splitting. Use -ConversationCacheMiB 0 for that benchmark.'
}
$gpuRows = @(& nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader |
    ConvertFrom-Csv -Header Index, Uuid, Name)
if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi could not enumerate the GPUs.' }
$primaryGpu = @($gpuRows | Where-Object { $_.Uuid.Trim() -eq $GpuUuid })
if ($primaryGpu.Count -ne 1 -or $primaryGpu[0].Name -notmatch 'RTX 5090') {
    throw "Expected the RTX 5090 at UUID $GpuUuid."
}
$visibleGpus = $GpuUuid
$monitorGpus = [int]$primaryGpu[0].Index
if ($GpuMode -ne 'Single') {
    $secondaryGpu = @($gpuRows | Where-Object { $_.Uuid.Trim() -eq $SecondaryGpuUuid })
    if ($secondaryGpu.Count -ne 1 -or $secondaryGpu[0].Name -notmatch 'RTX 4090' -or $SecondaryGpuUuid -eq $GpuUuid) {
        throw "Expected a distinct RTX 4090 at UUID $SecondaryGpuUuid."
    }
    $visibleGpus = "$GpuUuid,$SecondaryGpuUuid"
    # A gpu list makes upstream implicitly add a layer split. A helper must keep
    # this scalar; both devices remain visible through the explicit UUID list.
    if ($GpuMode -eq 'LayerSplit') { $monitorGpus = @([int]$primaryGpu[0].Index, [int]$secondaryGpu[0].Index) }
}
# Like the FreeToken launcher, Short4K has 8192 total tokens for a 4K prompt plus output.
$tokens = if ($Profile -eq 'Native256K') { 262144 } else { 8192 }
$engineArgs = @(
    '--pack', (Join-Path $model 'pack'),
    '--native', (Join-Path $model 'orca-nvfp4.gguf'),
    '--native-dense-gguf', (Join-Path $model 'orca-nvfp4.gguf'),
    '--ple-gguf', (Join-Path $model 'ple-fp8.gguf'),
    '--embd-gguf', (Join-Path $model 'token-embd-bf16.gguf'),
    '--expert-profile', (Join-Path $runtime 'data\expert-profile.bin'),
    '--expert-cache', 'auto', '--prefill', 'auto',
    '--max-context', "$tokens", '--kv', 'int8',
    '--vram-reserve-mib', "$VramReserveMiB", '--vision'
)
if (-not $NoMtp) {
    $engineArgs += @('--mtp', (Join-Path $model 'mtp\rt'), '--spec', '4', '--spec-min-p', '0.5')
}
if ($RamBudgetGiB -gt 0) {
    $engineArgs += @('--low-ram', '--ram-budget', "$RamBudgetGiB")
}
if ($GpuMode -eq 'LayerSplit') { $engineArgs += @('--layer-split', $LayerSplit) }
if ($GpuMode -eq 'ExpertHelper') { $engineArgs += @('--expert-cache-device1', "$HelperExpertSlots") }
if ($ConversationCacheMiB -gt 0) {
    $engineArgs += @('--conversation-cache-mib', "$ConversationCacheMiB",
        '--conversation-cache-slots', "$ConversationCacheSlots",
        '--conversation-cache-min-free-mib', '2560')
}
$config = [ordered]@{
    exe = Join-Path $runtime 'engine\strata.exe'
    args = $engineArgs
    cwd = $runtime
    tokenizer = Join-Path $model 'pack\tokenizer'
    model_name = $servedModel
    anthropic_thinking = 'on_request'
    # A reply that ends inside its thinking with no answer is closed once and continued (upstream #1053, opt-in
    # since 0.1.40.2); without it an agent turn ends with empty content.
    reasoning_close_retry = $true
    host = $BindAddress
    log = Join-Path $logDirectory "engine-$Port.log"
    gpu = $monitorGpus
    env = @{ CUDA_VISIBLE_DEVICES = $visibleGpus; CUDA_DEVICE_ORDER = 'PCI_BUS_ID' }
    vision = @{
        exe = Join-Path $runtime 'engine\strata-vision.exe'
        mmproj = Join-Path $model 'mmproj-f32.gguf'
        model = Join-Path $model 'orca-nvfp4.gguf'
        gpu = $false
        max_tokens = 1024
    }
}

Write-Host "Strata model: $servedModel; profile: $Profile ($tokens tokens); GPU mode: $GpuMode"
Write-Host "Visible GPUs: $visibleGpus; active request slots: 1 (additional requests queue)"
Write-Host "Conversation snapshots: $ConversationCacheMiB MiB host RAM budget, $ConversationCacheSlots parked slots"
Write-Host "OpenAI endpoint: http://127.0.0.1:$Port/v1 (bind $BindAddress)"
Write-Host "Engine log: $($config.log)"
if ($DryRun) {
    $config | ConvertTo-Json -Depth 6
    Write-Host "Working directory: $runtime"
    Write-Host "& '$python' -u -m serve.server --engine strata --config '$configPath' --port $Port"
    return
}

foreach ($required in @($python, $config.exe, $config.vision.exe, $config.vision.mmproj, $config.vision.model,
    (Join-Path $model 'ple-fp8.gguf'), (Join-Path $model 'token-embd-bf16.gguf'),
    (Join-Path $model 'pack\experts.bin'), (Join-Path $model 'pack\native_experts.txt'),
    (Join-Path $config.tokenizer 'vocab.json'), (Join-Path $model 'preparation.json'))) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required Strata asset missing: $required. Run setup-strata-runtime.ps1 and prepare-qwen38-strata.py."
    }
}
if (-not (Select-String -LiteralPath (Join-Path $runtime 'serve\server.py') -SimpleMatch 'req.get("thinking_token_budget")' -Quiet)) {
    throw 'The Strata server lacks the local thinking_token_budget alias. Run setup-strata-runtime.ps1.'
}
$preparation = Get-Content -LiteralPath (Join-Path $model 'preparation.json') -Raw | ConvertFrom-Json
if (-not $preparation.complete) { throw 'Model preparation has not completed successfully.' }
if (-not $NoMtp -and -not (Test-Path -LiteralPath (Join-Path $model 'mtp\rt\draft_vocab.bin'))) {
    throw 'The matching MTP draft head is incomplete.'
}
if (@(Get-ManagedServer).Count -gt 0) { throw "A managed Strata server already uses port $Port." }
# Check before rewriting the per-port config or loading weights.
$probeAddress = if ($BindAddress -eq 'localhost') { [System.Net.IPAddress]::Loopback } else {
    [System.Net.IPAddress]::Parse($BindAddress)
}
$portProbe = [System.Net.Sockets.TcpListener]::new($probeAddress, $Port)
try { $portProbe.Start() } finally { $portProbe.Stop() }
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$config | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $configPath -Encoding UTF8
$oldUtf8 = $env:PYTHONUTF8
$env:PYTHONUTF8 = '1'
Push-Location $runtime
try {
    & $python -u -m serve.server --engine strata --config $configPath --port $Port
    if ($LASTEXITCODE -ne 0) { throw "Strata exited with code $LASTEXITCODE. See $($config.log)." }
} finally {
    Pop-Location
    if ($null -eq $oldUtf8) { Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue } else { $env:PYTHONUTF8 = $oldUtf8 }
}
