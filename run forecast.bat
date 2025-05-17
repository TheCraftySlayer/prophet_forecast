@echo off
REM Navigate to the directory containing this script
cd /d "%~dp0"

cmd /k python prophet_analysis.py calls.csv visitors.csv queries.csv prophet_results --handle-outliers winsorize --use-transformation false
