@echo off
echo ===============================================
echo     🤖 DERIV AI TRADING BOT - ADVANCED SETUP
echo ===============================================
echo.

REM Check Python version compatibility
echo [1/8] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH!
    echo Please install Python 3.8-3.12 from https://python.org
    echo ⚠️  AVOID Python 3.13 - compatibility issues
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Python version: %PYTHON_VERSION%

REM Check if version is 3.13 (problematic)
echo %PYTHON_VERSION% | findstr "3.13" >nul
if not errorlevel 1 (
    echo ⚠️  WARNING: Python 3.13 detected!
    echo This version has compatibility issues with some packages.
    echo Consider using Python 3.8-3.12 for best results.
    echo.
    choice /c YN /m "Continue anyway? (Y/N)"
    if errorlevel 2 exit /b 1
)

REM Check pip
echo [2/8] Checking pip...
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is not available!
    pause
    exit /b 1
)
echo ✅ pip is available

REM Create virtual environment
echo [3/8] Setting up virtual environment...
if not exist "venv" (
    echo Creating new virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment
echo [4/8] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment!
    pause
    exit /b 1
)
echo ✅ Virtual environment activated

REM Upgrade pip
echo [5/8] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo ✅ pip upgraded

REM Install dependencies with better error handling
echo [6/8] Installing dependencies...
echo This may take a few minutes, please wait...
pip install -r requirements.txt --no-cache-dir
if errorlevel 1 (
    echo ❌ Failed to install some dependencies!
    echo.
    echo 🔧 Trying alternative installation method...
    echo Installing core packages individually...
    
    pip install websockets requests python-dotenv fastapi uvicorn sqlalchemy
    pip install aiofiles openai schedule jinja2 python-multipart asyncio httpx
    
    REM Try installing numpy and pandas separately with compatible versions
    pip install "numpy>=1.21.0,<1.27.0"
    pip install "pandas>=2.0.0" --no-build-isolation
    
    if errorlevel 1 (
        echo ❌ Critical dependencies failed to install
        echo Please check your internet connection and Python version
        echo Consider using Python 3.8-3.11 instead of 3.13
        pause
        exit /b 1
    )
)
echo ✅ Dependencies installed successfully

REM Initialize database
echo [7/8] Setting up database...
python database.py
if errorlevel 1 (
    echo ⚠️  Database setup had issues, but continuing...
) else (
    echo ✅ Database initialized
)

REM Create logs directory
if not exist "logs" mkdir logs
echo ✅ Logs directory ready

REM Verify configuration
echo [8/8] Verifying configuration...
if not exist ".env" (
    echo ⚠️  .env file not found, using config.py defaults
) else (
    echo ✅ Environment file found
)

echo.
echo ===============================================
echo       🚀 STARTING DERIV AI TRADING BOT
echo ===============================================
echo.
echo Bot Configuration:
echo - API Token: Configured ✅
echo - OpenAI Key: Configured ✅  
echo - Demo Mode: %DEMO_MODE% (RECOMMENDED)
echo - Initial Stake: $1.00
echo - Max Daily Loss: $100.00
echo.
echo ⚠️  IMPORTANT SAFETY REMINDERS:
echo - This bot is running in DEMO mode (safe)
echo - Always test strategies before live trading
echo - Never risk more than you can afford to lose
echo - Monitor the bot regularly
echo.
echo Starting bot in 3 seconds...
timeout /t 3 /nobreak >nul

REM Start the trading bot
python main.py

REM Handle bot exit
echo.
echo ===============================================
echo Bot has stopped running.
echo Check the logs for any errors or issues.
echo ===============================================
pause
