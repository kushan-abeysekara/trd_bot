@echo off
echo Setting up Deriv Trading Bot Backend...
echo.

cd backend

echo Installing Python dependencies...
pip install -r requirements.txt

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Backend setup completed successfully!
    echo.
    echo To start the backend server, run:
    echo python app.py
) else (
    echo.
    echo ❌ Backend setup failed!
    echo Please check your Python installation and try again.
)

pause
