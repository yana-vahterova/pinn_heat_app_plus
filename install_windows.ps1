param(
    [string]$Torch = "cpu",            # cpu | pypi | cu121 | cu118 | cu122 ...
    [string]$VenvName = "pinn-heat",
    [string]$VenvRoot = "$env:USERPROFILE\venvs"
)

$ErrorActionPreference = "Stop"

function Info($msg){ Write-Host "[info] $msg" -ForegroundColor Cyan }
function Warn($msg){ Write-Host "[warn] $msg" -ForegroundColor Yellow }
function Die($msg){ Write-Host "[err ] $msg" -ForegroundColor Red; exit 1 }

# 1) Ensure python exists
try {
    $pyver = & python -V
    Info "Using Python: $pyver"
} catch {
    $py312 = "$env:USERPROFILE\AppData\Local\Programs\Python\Python312\python.exe"
    if (Test-Path $py312) {
        Info "Using Python at $py312"
        Set-Alias -Name python -Value $py312 -Scope Script
    } else {
        Die "Python not found. Install Python 3.11/3.12 from https://python.org and re-run this script."
    }
}

# 2) Create venv outside project
$VenvPath = Join-Path $VenvRoot $VenvName
if (!(Test-Path $VenvRoot)) { New-Item -ItemType Directory -Force $VenvRoot | Out-Null }

if (Test-Path (Join-Path $VenvPath "Scripts\Activate.ps1")) {
    Info "Virtual env already exists: $VenvPath"
} else {
    Info "Creating virtual env at $VenvPath ..."
    python -m venv "$VenvPath" | Out-Null
}

# 3) Activate venv
$activate = Join-Path $VenvPath "Scripts\Activate.ps1"
if (!(Test-Path $activate)) { Die "Activate.ps1 not found at $activate" }
. $activate
Info "Activated venv: $VenvPath"

# 4) Upgrade pip & wheel
python -m pip install --upgrade pip wheel

# 5) Install base deps (from requirements.txt if present)
$projReq = Join-Path (Get-Location) "requirements.txt"
if (Test-Path $projReq) {
    Info "Installing from requirements.txt"
    pip install -r $projReq
} else {
    Info "Installing base Python packages (streamlit, plotly, pandas, numpy)"
    pip install streamlit plotly pandas numpy
}

# 6) Install PyTorch
switch -Regex ($Torch.ToLower()) {
    '^cpu$'    { Info "Installing PyTorch (CPU wheel)"; pip install torch --index-url https://download.pytorch.org/whl/cpu; break }
    '^pypi$'   { Info "Installing PyTorch from PyPI (auto)"; pip install torch; break }
    '^cu\d+$'  { Info "Installing PyTorch CUDA wheel '$Torch'"; pip install torch --index-url ("https://download.pytorch.org/whl/{0}" -f $Torch); break }
    default    { Warn "Unknown Torch option '$Torch'. Using CPU wheel."; pip install torch --index-url https://download.pytorch.org/whl/cpu; break }
}

# 7) Quick check
python -c "import torch, platform; print('Torch:', torch.__version__, '| Python:', platform.python_version()); print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

Info "Done. To run the app next time:"
Write-Host "    & `"$VenvPath\Scripts\Activate.ps1`""
Write-Host "    streamlit run app.py"
