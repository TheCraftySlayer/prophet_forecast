@echo off
cd /d "%~dp0"

REM Run Prophet forecasting with outlier handling and cross-validation
set USE_REAL_LIBS=1
python prophet_analysis.py calls.csv visitors.csv queries.csv prophet_results ^
    --handle-outliers winsorize ^
    --use-transformation false ^
    --cross-validate

pause
