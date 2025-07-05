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
echo Frontend will be available at: http://localhost:3000
echo.
echo The application will automatically open in your browser.
echo.

pause
