@echo off
cd /d "%~dp0"

REM Run the full forecasting pipeline. Provide a config path as the first
REM argument or default to "config.yaml". An optional second argument sets
REM the output directory which will be forwarded to the pipeline.

set "CONFIG=%1"
if "%CONFIG%"=="" set "CONFIG=config.yaml"

set "OUT=%2"
set USE_REAL_LIBS=1

if not "%OUT%"=="" (
    python pipeline.py "%CONFIG%" --out "%OUT%"
) else (
    python pipeline.py "%CONFIG%"
)

pause
