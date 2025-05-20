@echo off
cd /d "%~dp0"

REM Run the full forecasting pipeline. Pass a custom config path as the first
REM argument or default to "config.yaml" if none is provided.
set "CONFIG=%1"
if "%CONFIG%"=="" set "CONFIG=config.yaml"

set USE_REAL_LIBS=1
python pipeline.py "%CONFIG%"

pause
