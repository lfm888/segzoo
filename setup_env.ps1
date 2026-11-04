# Usage (PowerShell, run inside this folder):
#   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
# Creates a NEW conda env (default name: segzoo). Existing envs (pytorch, cunex38, ...) are untouched.
# NOTE: this file is intentionally ASCII-only, because Windows PowerShell 5.1 reads
#       BOM-less UTF-8 files as ANSI/GBK and garbles non-ASCII text.
# All parameters have defaults; override only if needed, e.g.  .\setup_env.ps1 -EnvName myenv
param(
    [string]$EnvName = "segzoo",
    [string]$PyVer = "3.10",
    # cu121 PyTorch wheels bundle their own CUDA runtime; any driver supporting CUDA 12.x (yours: 12.3) works
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu121",
    [string]$TorchVer = "2.2.2",
    [string]$TorchvisionVer = "0.17.2"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Run one step; stop on failure unless the step is optional
function Step([string]$msg, [scriptblock]$cmd, [bool]$optional = $false) {
    Write-Host ""
    Write-Host ">>> $msg" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) {
        if ($optional) {
            Write-Host "[warn] optional step failed, skipped: $msg" -ForegroundColor Yellow
        } else {
            throw "Step failed: $msg (exit code $LASTEXITCODE)"
        }
    }
}

# Skip creation if an env with the same name already exists
$exists = (conda env list) -match "^\s*$EnvName\s"
if (-not $exists) {
    Step "Create conda env $EnvName (python=$PyVer)" { conda create -n $EnvName python=$PyVer -y }
} else {
    Write-Host "Env $EnvName already exists, skip creation (to rebuild: conda env remove -n $EnvName)" -ForegroundColor Yellow
}

# pip invoked through 'conda run' so no manual activation is needed
$pip = @("run", "-n", $EnvName, "--no-capture-output", "python", "-m", "pip")
Step "Upgrade pip"                { conda @pip install --upgrade pip }
Step "Install numpy<2"            { conda @pip install "numpy==1.26.4" }
Step "Install PyTorch (cu121)"    { conda @pip install "torch==$TorchVer" "torchvision==$TorchvisionVer" --index-url $TorchIndex }
Step "Install other requirements" { conda @pip install -r requirements.txt }
Step "Install onnxsim (optional)" { conda @pip install onnxsim } $true
Step "Environment self-check"     { conda run -n $EnvName --no-capture-output python check_env.py }

Write-Host ""
Write-Host "Done. Next time use:  conda activate $EnvName" -ForegroundColor Green
