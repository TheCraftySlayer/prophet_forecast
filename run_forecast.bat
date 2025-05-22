@echo off
rem run_forecast.bat — rebuild‑proof runner
setlocal ENABLEDELAYEDEXPANSION

:: ---------------------------------------------------------------------------
:: Resolve script directory and switch to it
set "BASEDIR=%~dp0"
cd /d "%BASEDIR%"

:: ---------------------------------------------------------------------------
:: Create venv once (Python 3.10 assumed in PATH as py -3.10)
if not exist ".venv\\Scripts\\python.exe" (
    py -3.10 -m venv .venv || exit /b 1
)

:: Activate venv
call ".venv\\Scripts\\activate.bat" || exit /b 1

:: ---------------------------------------------------------------------------
:: Install requirements once.  Marker file prevents re‑install on every run.
if not exist ".venv\\.deps_installed" (
    pip install --upgrade pip setuptools wheel >nul || exit /b 1
    pip install --upgrade -r requirements.txt >nul || exit /b 1
    :: Ensure latest compatible cmdstanpy and local CmdStan build
    pip install --upgrade cmdstanpy --upgrade-strategy eager >nul || exit /b 1
    python - <<EOF
import cmdstanpy, pathlib
home = pathlib.Path.home() / ".cmdstan" / "cmdstan-2.36.0"
if not home.exists():
    cmdstanpy.install_cmdstan(version="2.36.0", overwrite=False)
EOF
    > ".venv\\.deps_installed"
)

:: ---------------------------------------------------------------------------
:: Run forecast pipeline
set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=config.yaml"
python pipeline.py "%CONFIG%" || exit /b 1

endlocal
pause
