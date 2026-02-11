Write-Host "=== WINDOWS SOC LOG COLLECTION STARTED ==="

# ============================================================
# AUTO RUN AS ADMIN
# ============================================================
$currUser = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($currUser)

if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Restarting script as Administrator..."
    Start-Process PowerShell -Verb RunAs -ArgumentList "-ExecutionPolicy Bypass -File `"$PSCommandPath`""
    Exit
}

# ============================================================
# FOLDER SETUP
# ============================================================
$base = "$env:USERPROFILE\Desktop\Cyber_SOC_Enterprise"
$jsonFolder = "$base\Json_Logs"
$evtxFolder = "$base\EVTX_Logs"

New-Item -ItemType Directory -Path $jsonFolder -Force | Out-Null
New-Item -ItemType Directory -Path $evtxFolder -Force | Out-Null

Write-Host "Output folders:"
Write-Host "JSON -> $jsonFolder"
Write-Host "EVTX -> $evtxFolder"

# ============================================================
# TIME FILTER — LAST 2 DAYS
# ============================================================
$days = 2
$startTime = (Get-Date).AddDays(-$days)

# ============================================================
# LOG COLLECTION (FAST MODE)
# ============================================================
Write-Host "Collecting optimized logs (FAST MODE)..."

# Security important Event IDs
$importantSecurityIDs = @(4624, 4625, 4672, 4688, 4720, 4728, 1102)

# --- SECURITY LOG ---
Write-Host "Collecting Security events..."
$securityEvents = Get-WinEvent -FilterHashtable @{
    LogName = "Security"
    StartTime = $startTime
} -MaxEvents 2000 -ErrorAction SilentlyContinue |
Where-Object { $importantSecurityIDs -contains $_.Id } |
Select-Object TimeCreated, Id, LevelDisplayName, ProviderName, Message

$securityJson = "$jsonFolder\Security.json"
$securityEvents | ConvertTo-Json -Depth 4 | Out-File $securityJson -Encoding UTF8
Write-Host "Saved JSON: $securityJson"

$securityEvtx = "$evtxFolder\Security.evtx"
wevtutil epl Security $securityEvtx /q:"*[System[TimeCreated[timediff(@SystemTime) <= $($days*24*60*60*1000)]]]"
Write-Host "Saved EVTX: $securityEvtx"

# --- SYSTEM LOG (Errors/Critical) ---
Write-Host "Collecting System errors..."
$systemEvents = Get-WinEvent -FilterHashtable @{
    LogName = "System"
    StartTime = $startTime
} -MaxEvents 1000 -ErrorAction SilentlyContinue |
Where-Object { $_.LevelDisplayName -in @("Error","Critical") } |
Select-Object TimeCreated, Id, LevelDisplayName, ProviderName, Message

$systemJson = "$jsonFolder\System.json"
$systemEvents | ConvertTo-Json -Depth 4 | Out-File $systemJson -Encoding UTF8
Write-Host "Saved JSON: $systemJson"

$systemEvtx = "$evtxFolder\System.evtx"
wevtutil epl System $systemEvtx /q:"*[System[TimeCreated[timediff(@SystemTime) <= $($days*24*60*60*1000)]]]"
Write-Host "Saved EVTX: $systemEvtx"

# --- APPLICATION LOG (Warnings/Errors) ---
Write-Host "Collecting Application warnings/errors..."
$appEvents = Get-WinEvent -FilterHashtable @{
    LogName = "Application"
    StartTime = $startTime
} -MaxEvents 800 -ErrorAction SilentlyContinue |
Where-Object { $_.LevelDisplayName -in @("Warning","Error") } |
Select-Object TimeCreated, Id, LevelDisplayName, ProviderName, Message

$appJson = "$jsonFolder\Application.json"
$appEvents | ConvertTo-Json -Depth 4 | Out-File $appJson -Encoding UTF8
Write-Host "Saved JSON: $appJson"

$appEvtx = "$evtxFolder\Application.evtx"
wevtutil epl Application $appEvtx /q:"*[System[TimeCreated[timediff(@SystemTime) <= $($days*24*60*60*1000)]]]"
Write-Host "Saved EVTX: $appEvtx"

# ============================================================
Write-Host "=== LOG COLLECTION COMPLETE ==="
Write-Host "JSON Logs -> $jsonFolder"
Write-Host "EVTX Logs -> $evtxFolder"

