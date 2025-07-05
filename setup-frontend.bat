@echo off
echo Setting up Deriv Trading Bot Frontend...
echo.

cd frontend

echo Installing Node.js dependencies...
npm install

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Frontend setup completed successfully!
    echo.
    echo To start the frontend server, run:
    echo npm start
) else (
    echo.
    echo ❌ Frontend setup failed!
    echo Please check your Node.js installation and try again.
)

pause
