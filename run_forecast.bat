@echo on
setlocal enabledelayedexpansion

REM ---------- interpreter ----------
if not defined PYTHON (
    for /f "delims=" %%I in ('where python 2^>nul') do (
        echo %%I | findstr /I "pyenv\\shims" >nul
        if errorlevel 1 (
            set "PYTHON=%%I"
            goto :int_found
        )
    )
    for /f "delims=" %%I in ('where python3 2^>nul') do (
        echo %%I | findstr /I "pyenv\\shims" >nul
        if errorlevel 1 (
            set "PYTHON=%%I"
            goto :int_found
        )
    )
    if not defined PYTHON if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    if not defined PYTHON if exist "%ProgramFiles%\Python311\python.exe" set "PYTHON=%ProgramFiles%\Python311\python.exe"
)
:int_found
if not defined PYTHON (
    echo [ERROR] No usable Python 3.10+ interpreter found.
    exit /b 1
)

REM ---------- repo root ----------
pushd "%~dp0" || exit /b 2

REM ---------- venv ----------
if not exist ".venv\Scripts\python.exe" (
    "%PYTHON%" -m venv ".venv" || exit /b 3
)
call ".venv\Scripts\activate.bat" || exit /b 4

REM ---------- tooling ----------
python -m pip install --upgrade pip || exit /b 5
if exist requirements.txt (
    python -m pip install -r requirements.txt --disable-pip-version-check || exit /b 6
)

REM ---------- config ----------
set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=config.yaml"

REM ---------- run ----------
python -X dev -u pipeline.py "%CONFIG%"
exit /b %ERRORLEVEL%
