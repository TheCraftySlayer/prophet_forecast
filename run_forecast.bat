@echo off
cd /d "%~dp0"
setlocal enableextensions enabledelayedexpansion

REM --- Explicit Removal of Stub Directories (Mandatory) ---
for %%D in (matplotlib seaborn sklearn statsmodels pandas) do (
    if exist "%%D" (
        echo [cleanup] Removing stub directory %%D
        rmdir /S /Q "%%D"
    )
)

REM --- Mandatory Environment Variable ---
set USE_STUB_LIBS=0

REM --- Virtual environment bootstrap ---
if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating Python 3.10 venv .venv
    py -3.10 -m venv .venv 2>NUL || python -m venv .venv
)

REM Activate venv
call ".venv\Scripts\activate.bat"

REM --- Explicit Cleanup of Problematic Packages and Cache ---
pip uninstall numpy matplotlib prophet cmdstanpy scikit-learn statsmodels seaborn -y
pip cache purge
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"

REM --- Dependency sync ---
python -m pip install --upgrade pip
if exist requirements.txt (
    pip install -r requirements.txt
)

REM Explicit Reinstallation of Dependencies (Mandatory)
python -m pip install --upgrade --no-cache-dir ^
    prophet==1.1.6 cmdstanpy==1.2.2 ^
    matplotlib==3.7.1 seaborn==0.12.2 ^
    scikit-learn==1.3.1 statsmodels==0.14.4 pandas>=2.0 numpy==1.23.5

python -m cmdstanpy.install_cmdstan --overwrite

REM --- Diagnostic Information ---
echo Current working directory: %cd%
echo Python executable location:
where python

REM --- Run forecasting pipeline ---
set "CONFIG=%1"
if "%CONFIG%"=="" set "CONFIG=config.yaml"

python pipeline.py "%CONFIG%"
if %errorlevel% neq 0 (
    echo [ERROR] Pipeline execution failed with error level %errorlevel%
    pause
    exit /b %errorlevel%
)

endlocal
pause
