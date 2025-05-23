@echo on
setlocal enabledelayedexpansion

REM ----- interpreter (CPython ≥3.10, not MSYS/MINGW) -----
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
) else (
    for /f "delims=" %%I in ('where python 2^>nul') do (
        echo %%I | findstr /I /C:"\msys" /C:"\mingw" /C:"pyenv\shims" >nul
        if errorlevel 1 set "PYTHON=%%I"& goto :int_found
    )
    for /f "delims=" %%I in ('where python3 2^>nul') do (
        echo %%I | findstr /I /C:"\msys" /C:"\mingw" /C:"pyenv\shims" >nul
        if errorlevel 1 set "PYTHON=%%I"& goto :int_found
    )
)
:int_found
if not defined PYTHON (
    echo [ERROR] No usable CPython 3.10+ interpreter found.
    exit /b 1
)

REM ----- repo root -----
pushd "%~dp0" || exit /b 2

REM ----- venv -----
if not exist ".venv\Scripts\python.exe" (
    "%PYTHON%" -m venv .venv || exit /b 3
)
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] venv creation failed.
    exit /b 3
)
set "VP=.venv\Scripts\python.exe"

REM ----- tooling -----
%VP% -m pip install --upgrade pip || exit /b 5
if exist requirements.txt (
    %VP% -m pip install -r requirements.txt --disable-pip-version-check || exit /b 6
)

REM ----- config -----
set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=config.yaml"

REM ----- run -----
%VP% -X dev -u pipeline.py "%CONFIG%"
exit /b %ERRORLEVEL%
