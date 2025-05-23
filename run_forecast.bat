@echo off
setlocal
pushd "%~dp0"
set "PY=%~dp0\.venv\Scripts\python.exe"
if not exist "%PY%" exit /b 1
set "PATH=C:\msys64\mingw64\bin;%PATH%"
set "MAKE=mingw32-make"

"%PY%" - <<EOF
import cmdstanpy, pathlib
if not pathlib.Path(cmdstanpy.cmdstan_path()).exists():
    cmdstanpy.install_cmdstan()
EOF

"%PY%" pipeline.py %*
popd
