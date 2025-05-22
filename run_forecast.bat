@echo off
rem run_forecast.bat – resilient runner
setlocal ENABLEDELAYEDEXPANSION

:: ------------------------------------------------------------------
:: Default to stub libs unless caller overrides
if not defined USE_STUB_LIBS set "USE_STUB_LIBS=1"

:: ------------------------------------------------------------------
:: Resolve script directory and switch to it
set "BASEDIR=%~dp0"
cd /d "%BASEDIR%"

:: ------------------------------------------------------------------
:: Choose interpreter: caller-supplied PYTHON → local venv → system
if defined PYTHON (
    set "_PYTHON=%PYTHON%"
) else if exist ".venv\Scripts\python.exe" (
    set "_PYTHON=.venv\Scripts\python.exe"
) else (
    set "_PYTHON=python"
)

:: ------------------------------------------------------------------
:: Create and activate local venv when using bundled interpreter
if "%_PYTHON%"==".venv\Scripts\python.exe" (
    if not exist "%_PYTHON%" (
        where py >nul 2>&1 || (echo Python launcher missing & exit /b 1)
        py -3.10 -m venv .venv || exit /b 1
    )
    call ".venv\Scripts\activate.bat" || exit /b 1
)

:: ------------------------------------------------------------------
:: One-time dependency bootstrap
if "%_PYTHON%"==".venv\Scripts\python.exe" if not exist ".venv\.deps_installed" (

    rem --- core tooling
    "%_PYTHON%" -m pip install --upgrade pip setuptools wheel || exit /b 1

    rem --- project requirements
    "%_PYTHON%" -m pip install -r requirements.txt || exit /b 1

    rem --- pin cmdstanpy; stop NumPy-2 pull
    "%_PYTHON%" -m pip install --no-deps --upgrade-strategy=only-if-needed cmdstanpy==1.2.5 || exit /b 1

    rem --- verify dependency graph
    "%_PYTHON%" -m pip check || exit /b 1

    rem --- ensure CmdStan 2.36.0 (only if make exists)
    where mingw32-make >nul 2>&1 && (
        set "_TMP=%TEMP%\install_cmdstan_!RANDOM!.py"
        >"!_TMP!" (
            echo import cmdstanpy, pathlib
            echo tgt = pathlib.Path.home^(^) / ".cmdstan" / "cmdstan-2.36.0"
            echo if not tgt.exists^(^):
            echo     cmdstanpy.install_cmdstan^(version="2.36.0", overwrite=False^)
        )
        "%_PYTHON%" "!_TMP!" || exit /b 1
        del "!_TMP!"
    )

    rem --- mark completion
    > ".venv\.deps_installed" echo done
)

:: ------------------------------------------------------------------
:: Run forecast pipeline
set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=config.yaml"
"%_PYTHON%" pipeline.py "%CONFIG%" || exit /b 1

endlocal
