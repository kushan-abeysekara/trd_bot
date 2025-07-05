@echo off
echo 🚀 Starting Deriv Trading Bot - Production Mode...
echo.

REM Set production environment
set FLASK_ENV=production
set FLASK_DEBUG=False

echo 📦 Installing/updating Python dependencies...
cd backend
pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to install backend dependencies!
    pause
    exit /b 1
)

echo 📦 Installing/updating Node.js dependencies...
cd ../frontend
npm install

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to install frontend dependencies!
    pause
    exit /b 1
)

echo 🏗️ Building frontend for production...
npm run build

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to build frontend!
    pause
    exit /b 1
)

echo.
echo ✅ Production build complete!
echo.
echo 🚀 Starting production servers...
echo.

REM Start backend server in production mode
echo Starting backend server...
start "Backend Production Server" cmd /k "cd ../backend && set FLASK_ENV=production && python production_server.py"

timeout /t 3 >nul

REM Start frontend production server
echo Starting frontend production server...
start "Frontend Production Server" cmd /k "cd frontend && npm run start-production"

echo.
echo ✅ Production servers are starting up!
echo.
echo 🌐 Access your application at:
echo Frontend: http://your-server-ip:8080
echo Backend API: http://your-server-ip:5000/api
echo.
echo For external access, make sure ports 5000 and 8080 are open in your firewall.
echo.

pause
