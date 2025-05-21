@echo off
cd /d "%~dp0"

REM Use stub libraries for unit testing by default
set "USE_STUB_LIBS=1"

REM Run the full forecasting pipeline. Provide a config path as the first
REM argument or default to "config.yaml".
set "CONFIG=%1"
if "%CONFIG%"=="" set "CONFIG=config.yaml"

REM Determine which Python interpreter to use. If "PYTHON" is set we honor
REM it, otherwise prefer a local virtual environment before falling back to
REM the system "python".
if "%PYTHON%"=="" (
    if exist ".venv\Scripts\python.exe" (
        set "PYTHON=.venv\Scripts\python.exe"
    ) else (
        set "PYTHON=python"
    )
)

%PYTHON% pipeline.py "%CONFIG%"

pause
