@echo off
title Ollama GUI

echo ================================
echo    Ollama GUI Launcher
echo ================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.8 or higher.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Create virtual environment if not exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment found.
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install/Update dependencies
echo Installing dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo Dependencies installed.

REM Check Ollama connection
echo.
echo Checking Ollama connection...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo.
    echo ========================================
    echo [WARNING] Ollama is not running!
    echo ========================================
    echo Please start Ollama first:
    echo   - On Windows: Run 'ollama' from Start Menu
    echo   - Or open terminal and run: ollama serve
    echo.
    echo Press any key to continue anyway...
    pause >nul
) else (
    echo Ollama connection successful.
)

echo.
echo ================================
echo     Starting Ollama GUI
echo ================================
echo.

REM Run the application
python app.py

REM Keep window open if app exits
pause