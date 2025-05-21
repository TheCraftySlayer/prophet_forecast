@echo off
cd /d "%~dp0"

REM Run the full forecasting pipeline. Provide a config path as the first
REM argument or default to "config.yaml".

set "CONFIG=%1"
if "%CONFIG%"=="" set "CONFIG=config.yaml"
if "%PYTHON%"=="" set "PYTHON=python"

%PYTHON% pipeline.py "%CONFIG%"

pause
