@echo off
cd /d "%~dp0"

REM Run the full forecasting pipeline using the default configuration
set USE_REAL_LIBS=1
python pipeline.py config.yaml

pause
