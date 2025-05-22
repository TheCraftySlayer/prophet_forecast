@echo off
rem run_forecast.bat — rebuild‑proof runner
setlocal ENABLEDELAYEDEXPANSION

:: ---------------------------------------------------------------------------
:: Default to stub libraries unless caller overrides
if not defined USE_STUB_LIBS set "USE_STUB_LIBS=1"

:: ---------------------------------------------------------------------------
:: Resolve script directory and switch to it
set "BASEDIR=%~dp0"
cd /d "%BASEDIR%"

:: ---------------------------------------------------------------------------
:: Determine interpreter.  Prefer caller-supplied PYTHON, then local venv
if defined PYTHON (
    set "_PYTHON=%PYTHON%"
) else if exist ".venv\\Scripts\\python.exe" (
    set "_PYTHON=.venv\\Scripts\\python.exe"
) else (
    set "_PYTHON=python"
)

:: Create and activate venv when using the bundled interpreter
if "%_PYTHON%"==".venv\\Scripts\\python.exe" (
    if not exist "%_PYTHON%" (
        py -3.10 -m venv .venv || exit /b 1
    )
    call ".venv\\Scripts\\activate.bat" || exit /b 1
)

:: ---------------------------------------------------------------------------
:: Install requirements once when using bundled interpreter
if "%_PYTHON%"==".venv\\Scripts\\python.exe" if not exist ".venv\\.deps_installed" (
    pip install --upgrade pip setuptools wheel >nul || exit /b 1
    pip install --upgrade -r requirements.txt >nul || exit /b 1
    :: Ensure latest compatible cmdstanpy and local CmdStan build
    pip install --upgrade cmdstanpy --upgrade-strategy eager >nul || exit /b 1
    "%_PYTHON%" - <<EOF
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
"%_PYTHON%" pipeline.py "%CONFIG%" || exit /b 1

endlocal
pause
