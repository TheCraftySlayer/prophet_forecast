@echo on
REM ============================================================================
REM  run_forecast.bat – build & run Prophet pipeline in an activated venv
REM  Usage:  run_forecast.bat [config.yaml]
REM ============================================================================

setlocal EnableDelayedExpansion

REM ---------------------------------------------------------------------------
REM 1. Locate a CPython 3.10+ that is **not** MSYS/MinGW
REM ---------------------------------------------------------------------------
set "PYTHON="
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
) else (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        echo %%P| findstr /I /C:"\msys" /C:"\mingw" /C:"pyenv\shims" >nul
        if errorlevel 1 (set "PYTHON=%%P" & goto :python_found)
    )
    for /f "delims=" %%P in ('where python3 2^>nul') do (
        echo %%P| findstr /I /C:"\msys" /C:"\mingw" /C:"pyenv\shims" >nul
        if errorlevel 1 (set "PYTHON=%%P" & goto :python_found)
    )
)
:python_found
if not defined PYTHON (
    echo [ERROR] No usable CPython 3.10+ interpreter found.
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM 2. Jump to repo root (directory where this BAT lives)
REM ---------------------------------------------------------------------------
pushd "%~dp0" || exit /b 2

REM ---------------------------------------------------------------------------
REM 3. Ensure CmdStan temporary directory exists
REM ---------------------------------------------------------------------------
set "CMDSTANPY_TMPDIR=C:\cmdstan_tmp"
if not exist "%CMDSTANPY_TMPDIR%" (
    mkdir "%CMDSTANPY_TMPDIR%"
)

REM ---------------------------------------------------------------------------
REM 4. Create venv if needed
REM ---------------------------------------------------------------------------
if not exist ".venv\Scripts\activate.bat" (
    "%PYTHON%" -m venv .venv || exit /b 3
)

REM ---------------------------------------------------------------------------
REM 5. **Activate** the venv for the remainder of the script
REM ---------------------------------------------------------------------------
call ".venv\Scripts\activate.bat"

REM ---------------------------------------------------------------------------
REM 6. Upgrade tooling & install requirements
REM ---------------------------------------------------------------------------
python -m pip install --upgrade pip setuptools wheel || exit /b 5
if exist requirements.txt (
    python -m pip install -r requirements.txt --disable-pip-version-check || exit /b 6
)

REM ---------------------------------------------------------------------------
REM 7. Optional config file (defaults to config.yaml)
REM ---------------------------------------------------------------------------
set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=config.yaml"

REM ---------------------------------------------------------------------------
REM 8. Run the pipeline (-X dev shows deprecation warnings, -u = unbuffered)
REM ---------------------------------------------------------------------------
python -X dev -u pipeline.py "%CONFIG%"
set "RC=%ERRORLEVEL%"

REM ---------------------------------------------------------------------------
REM 9. Keep the venv activated if launched from Explorer (double-click)
REM ---------------------------------------------------------------------------
if "%cmdcmdline:~0,1%"=="\"" (
    echo.
    echo Pipeline finished. Venv remains active – type ^`exit^` to leave.
    cmd /k
)

popd
exit /b %RC%
