@echo off
:: run_forecast.bat — deterministic, idempotent
setlocal enabledelayedexpansion

:: ---------- move to repo root ----------
pushd "%~dp0" || exit /b 1

:: ---------- verify Python ≥ 3.10 ----------
for /f "tokens=1,2 delims=." %%a in ('python -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do (
    set "MAJOR=%%a"
    set "MINOR=%%b"
)
if not "!MAJOR!"=="3" (
    echo Python 3 required; found !MAJOR!.!MINOR!
    popd & exit /b 1
)
if !MINOR! LSS 10 (
    echo Python >=3.10 required; found 3.!MINOR!
    popd & exit /b 1
)

:: ---------- ensure local venv ----------
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv || (echo venv creation failed & popd & exit /b 1)
)

:: ---------- activate venv ----------
call ".venv\Scripts\activate.bat"

:: ---------- verify C/C++ toolchain ----------
where cl >nul 2>&1 || where g++ >nul 2>&1
if errorlevel 1 (
    echo No C/C++ compiler detected
    popd & exit /b 1
)

:: ---------- dependency stamp ----------
for %%I in ("requirements.txt") do set "REQ_TS=%%~tI"
if exist ".venv\.req_ts" (
    set /p STORED_TS=<".venv\.req_ts"
) else (
    set "STORED_TS="
)
if not "%REQ_TS%"=="%STORED_TS%" (
    pip install -r "requirements.txt" || (echo Dependency install failed & popd & exit /b 1)
    >".venv\.req_ts" echo %REQ_TS%
)

:: ---------- force-reinstall Prophet + CmdStan ----------
pip install --force-reinstall prophet==1.1.5 cmdstanpy==1.2.2 || (
    echo Prophet/CmdStanPy reinstall failed
    popd & exit /b 1
)
python -m cmdstanpy.install_cmdstan --silent || (
    echo CmdStan installation failed
    popd & exit /b 1
)

:: ---------- run pipeline ----------
set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=config.yaml"
python pipeline.py "%CONFIG%" || (echo Pipeline error & popd & exit /b 1)

popd
endlocal
