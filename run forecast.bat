@echo off
cd /d "%~dp0"

python prophet_analysis.py calls.csv visitors.csv queries.csv prophet_results --handle-outliers winsorize --use-transformation false

pause
