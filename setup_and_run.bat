@echo off
setlocal enabledelayedexpansion

REM AI Trading Bot Setup and Run Script for Windows

echo 🤖 AI Trading Bot - Setup and Launch
echo ====================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed. Please install Node.js 16 or higher.
    pause
    exit /b 1
)

echo ✅ Python and Node.js detected

REM Setup Backend
echo.
echo 🔧 Setting up Backend...
cd backend

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install Python dependencies
echo 📥 Installing Python dependencies...
pip install -r requirements.txt

REM Initialize database
echo 🗄️ Initializing database...
python -c "from app import create_app, db; app = create_app(); app.app_context().push(); db.create_all(); print('Database initialized successfully!')"

cd ..

REM Setup Frontend
echo.
echo 🎨 Setting up Frontend...
cd frontend

REM Install Node dependencies
echo 📥 Installing Node.js dependencies...
npm install

cd ..

echo.
echo ✅ Setup completed successfully!
echo.
echo 🚀 Starting AI Trading Bot...
echo.

REM Start backend in background
echo 🔙 Starting Backend server...
cd backend
call venv\Scripts\activate.bat
start /B python app.py

cd ..

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend
echo 🖥️ Starting Frontend server...
cd frontend
start /B npm start

cd ..

echo.
echo 🎉 AI Trading Bot is now running!
echo.
echo 📊 Dashboard: http://localhost:3000
echo 🔗 API: http://localhost:5000
echo.
echo Features:
echo • 🤖 AI-powered trading with ML models for all contract types
echo • 📈 Advanced money management with smart Martingale
echo • 🛡️ Safety limits (daily stop loss, daily target, cooldowns)
echo • 🔄 Dynamic strategy switching (3 modes)
echo • 📊 All 10 Deriv contract types supported
echo • 🧠 ChatGPT integration for auto-training
echo • 💾 Local ML training data saved in SQL database
echo • 📱 Real-time monitoring and control
echo • 🎯 Live Trading Bot with Start/Stop controls
echo • 📋 Real-time open trades display
echo • 📊 Complete trading history with P&L
echo • 💰 Auto stake management with manual override
echo • 🎨 Advanced Mean Reversion Strategy
echo • 🔍 Real-time strategy status display
echo • 🎛️ Real-Time Strategy Monitor Window
echo • 📊 Live Technical Indicators Dashboard
echo • ✅ Strategy Condition Tracker (4/4 indicators)
echo • 🎯 Visual Signal Analysis with live validation
echo • ⚡ Auto-refreshing conditions every 5 seconds
echo.
echo Press any key to stop all services...
pause >nul

REM Kill all related processes
taskkill /f /im python.exe >nul 2>&1
taskkill /f /im node.exe >nul 2>&1

echo 🛑 All services stopped.
pause
