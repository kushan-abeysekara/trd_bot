@echo off
echo Starting Deriv AI Trading Bot...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH!
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo pip is not available!
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment!
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies!
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

REM Create logs directory
if not exist "logs" mkdir logs

REM Initialize database
echo Initializing database...
python database.py

REM Check if .env file exists
if not exist ".env" (
    echo.
    echo WARNING: .env file not found!
    echo Please create a .env file with your configuration.
    echo See .env.example for reference.
    echo.
    pause
)

REM Start the bot
echo.
echo Starting the trading bot...
echo Press Ctrl+C to stop the bot
echo.
python main.py

pause
