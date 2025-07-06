@echo off
echo Starting Deriv Trading Bot - Full Application...
echo.

echo Opening backend server in new window...
start "Backend Server" cmd /k "cd backend && python app.py"

timeout /t 3 >nul

echo Opening frontend server in new window...
start "Frontend Server" cmd /k "cd frontend && npm start"

echo.
echo ✅ Both servers are starting up!
echo.
echo Backend will be available at: http://localhost:5000
echo Frontend will be available at: http://localhost:8080
echo.
echo 🌐 Production URLs:
echo Frontend: https://tradingbot-4iuxi.ondigitalocean.app
echo Backend API: https://tradingbot-4iuxi.ondigitalocean.app/api
echo.
echo 🔧 For deployment on external server, use start-production.bat instead
echo The application will automatically open in your browser.
echo.
echo 📱 IMPORTANT: When closing the app, please use the "Disconnect" button 
echo    or close this window to ensure the bot stops trading properly.
echo.

REM Capture the process IDs of the backend and frontend
for /f "tokens=2" %%a in ('tasklist /fi "windowtitle eq Backend Server" /fo list /v ^| find /i "PID:"') do set backend_pid=%%a
for /f "tokens=2" %%a in ('tasklist /fi "windowtitle eq Frontend Server" /fo list /v ^| find /i "PID:"') do set frontend_pid=%%a

echo Press any key to stop all servers and exit...
pause

echo Stopping servers...
if defined backend_pid taskkill /F /PID %backend_pid% >nul 2>&1
if defined frontend_pid taskkill /F /PID %frontend_pid% >nul 2>&1

echo All servers stopped.
pause
