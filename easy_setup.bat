@echo off
echo ===============================================
echo     🤖 DERIV AI TRADING BOT - EASY SETUP  
echo ===============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH!
    echo Please install Python 3.8-3.12 from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Upgrade pip first
echo 📦 Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install packages individually with better error handling
echo 📦 Installing core packages...

REM Core web framework packages
python -m pip install "websockets>=11.0.0" --quiet
python -m pip install "requests>=2.31.0" --quiet  
python -m pip install "python-dotenv>=1.0.0" --quiet
python -m pip install "fastapi>=0.100.0" --quiet
python -m pip install "uvicorn>=0.20.0" --quiet
python -m pip install "sqlalchemy>=1.4.0,<2.0.0" --quiet
python -m pip install "aiofiles>=23.0.0" --quiet
python -m pip install "jinja2>=3.1.0" --quiet
python -m pip install "python-multipart>=0.0.6" --quiet
python -m pip install "httpx>=0.27.0" --quiet
python -m pip install "pymysql>=1.1.0" --quiet
python -m pip install "schedule>=1.2.0" --quiet
python -m pip install "openai>=1.3.0" --quiet

echo ✅ Core packages installed

REM Scientific packages with fallbacks
echo 📦 Installing scientific packages...
python -m pip install "numpy>=1.21.0,<1.27.0" --no-build-isolation --quiet
if errorlevel 1 (
    echo ⚠️  Trying fallback numpy installation...
    python -m pip install numpy --quiet
)

python -m pip install "pandas>=2.0.0" --no-build-isolation --quiet  
if errorlevel 1 (
    echo ⚠️  Trying fallback pandas installation...
    python -m pip install pandas --quiet
)

python -m pip install "scikit-learn>=1.3.0" --quiet
if errorlevel 1 (
    echo ⚠️  Scikit-learn installation failed, continuing...
)

echo ✅ Scientific packages installed

REM Optional packages (don't fail if these don't work)
echo 📦 Installing optional packages...
python -m pip install matplotlib seaborn cryptography psutil --quiet 2>nul

echo ✅ Package installation completed!
echo.

REM Create necessary directories
if not exist "logs" mkdir logs
if not exist "static" mkdir static  
if not exist "templates" mkdir templates

REM Check if .env exists
if not exist ".env" (
    echo ⚙️  Creating .env file...
    echo # Environment Configuration > .env
    echo DERIV_TOKEN=your_deriv_token_here >> .env
    echo OPENAI_API_KEY=your_openai_key_here >> .env
    echo. >> .env
    echo # Database Configuration >> .env  
    echo DATABASE_URL=sqlite:///trading_bot.db >> .env
    echo. >> .env
    echo # Trading Configuration >> .env
    echo DEMO_MODE=true >> .env
    echo MAX_DAILY_LOSS=100 >> .env
    echo MAX_CONSECUTIVE_LOSSES=5 >> .env
    echo INITIAL_STAKE=1.0 >> .env
    echo. >> .env
    echo # WebSocket Configuration >> .env
    echo DERIV_WS_URL=wss://ws.binaryws.com/websockets/v3?app_id=1089 >> .env
    
    echo ✅ .env file created
    echo ⚠️  Please edit .env file and add your API tokens!
) else (
    echo ✅ .env file already exists
)

echo.
echo ===============================================
echo        🚀 SETUP COMPLETED SUCCESSFULLY!
echo ===============================================
echo.
echo What would you like to do?
echo 1. Start Trading Bot
echo 2. Start Web Dashboard  
echo 3. Edit configuration
echo 4. Exit
echo.

:choice
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo 🤖 Starting Trading Bot...
    echo Press Ctrl+C to stop the bot
    echo.
    python main.py
    goto end
)

if "%choice%"=="2" (
    echo.
    echo 🌐 Starting Web Dashboard...
    echo Dashboard will be available at: http://localhost:8000
    echo Press Ctrl+C to stop the dashboard
    echo.
    python dashboard.py
    goto end
)

if "%choice%"=="3" (
    echo.
    echo ⚙️  Configuration:
    echo Please edit the .env file with your API tokens:
    echo - DERIV_TOKEN: Get from Deriv.com API settings
    echo - OPENAI_API_KEY: Get from OpenAI API settings
    echo.
    notepad .env
    goto end
)

if "%choice%"=="4" (
    echo.
    echo 👋 Goodbye!
    goto end
)

echo Please enter 1, 2, 3, or 4
goto choice

:end
pause
