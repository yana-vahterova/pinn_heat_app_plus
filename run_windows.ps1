param(
    [int]$Port = 8501,
    [string]$VenvName = "pinn-heat",
    [string]$VenvRoot = "$env:USERPROFILE\venvs",
    [switch]$Headless
)

$ErrorActionPreference = "Stop"
function Die($msg){ Write-Host "[err ] $msg" -ForegroundColor Red; exit 1 }

$VenvPath = Join-Path $VenvRoot $VenvName
$activate = Join-Path $VenvPath "Scripts\Activate.ps1"
if (!(Test-Path $activate)) {
    Die "Venv not found at $VenvPath. Run install_windows.ps1 first."
}
. $activate

# If app.py not in current dir, try the script dir
if (!(Test-Path ".\app.py")) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    if (Test-Path (Join-Path $scriptDir "app.py")) { Set-Location $scriptDir }
}

$args = @("run", "app.py", "--server.port", "$Port")
if ($Headless) { $args += @("--server.headless", "true") }

streamlit @args
