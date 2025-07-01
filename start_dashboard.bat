@echo off
echo Starting Deriv AI Trading Bot Dashboard...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Start the dashboard
echo Starting web dashboard on http://localhost:8000
echo Press Ctrl+C to stop the dashboard
echo.
python dashboard.py

pause
