@echo off
REM ==============================================================================
REM Chemical Reaction Optimizer - Launch Script (Windows)
REM ==============================================================================
REM This script sets up and launches the interactive dashboard
REM Usage: run_demo.bat
REM ==============================================================================

setlocal EnableDelayedExpansion

REM Colors (via ANSI escape codes - works in Windows 10+)
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "NC=[0m"

REM ==============================================================================
REM Banner
REM ==============================================================================

echo.
echo %BLUE%
echo ================================================================
echo.
echo     Chemical Reaction Optimizer
echo     Powered by Anaconda
echo.
echo ================================================================
echo %NC%

REM ==============================================================================
REM Check Prerequisites
REM ==============================================================================

echo %YELLOW%Checking prerequisites...%NC%

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo %RED%Error: conda not found%NC%
    echo Please install Anaconda or Miniconda:
    echo   https://www.anaconda.com/download
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('conda --version') do set CONDA_VERSION=%%i
echo %GREEN%[OK]%NC% Conda found: %CONDA_VERSION%

REM Check if environment exists
conda env list | findstr "reaction-optimizer" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo %YELLOW%Environment 'reaction-optimizer' not found%NC%
    echo %YELLOW%Creating environment ^(this may take 5-10 minutes^)...%NC%
    conda env create -f environment.yml

    if %ERRORLEVEL% NEQ 0 (
        echo %RED%Failed to create environment%NC%
        pause
        exit /b 1
    )
    echo %GREEN%[OK]%NC% Environment created successfully
) else (
    echo %GREEN%[OK]%NC% Environment 'reaction-optimizer' exists
)

REM ==============================================================================
REM Activate Environment
REM ==============================================================================

echo.
echo %YELLOW%Activating environment...%NC%

call conda activate reaction-optimizer

if %ERRORLEVEL% NEQ 0 (
    echo %RED%Failed to activate environment%NC%
    pause
    exit /b 1
)
echo %GREEN%[OK]%NC% Environment activated

REM Verify key packages
python -c "import panel, plotly, numpy" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo %RED%Missing required packages%NC%
    echo Try recreating the environment:
    echo   conda env remove -n reaction-optimizer
    echo   conda env create -f environment.yml
    pause
    exit /b 1
)
echo %GREEN%[OK]%NC% All packages available

REM ==============================================================================
REM Launch Dashboard
REM ==============================================================================

echo.
echo %BLUE%================================================================%NC%
echo %BLUE%Starting Dashboard%NC%
echo %BLUE%================================================================%NC%
echo.
echo %GREEN%Dashboard will be available at:%NC%
echo   %BLUE%http://localhost:5006%NC%
echo.
echo %YELLOW%Press Ctrl+C to stop the server%NC%
echo.

REM Launch Panel
panel serve app.py --show --port 5006

REM If user stops server
echo.
echo %YELLOW%Server stopped%NC%
pause
