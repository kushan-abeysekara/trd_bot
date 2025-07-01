@echo off
echo Checking if virtual environment exists...
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found!
    echo Please run install_dependencies.bat first.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Checking Python version...
python --version

REM Check if Python 3.13 is being used and warn about potential issues
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
if "%PYTHON_VERSION:~0,4%"=="3.13" (
    echo.
    echo WARNING: Running with Python 3.13 - some packages may have compatibility issues
    echo.
)

echo Starting Flask application...
python app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Application failed to start!
    echo Error code: %ERRORLEVEL%
    echo.
    echo Common solutions:
    if "%PYTHON_VERSION:~0,4%"=="3.13" (
        echo PYTHON 3.13 DETECTED - Try these solutions in order:
        echo 1. Delete 'venv' folder and reinstall with updated requirements.txt
        echo 2. Install Python 3.11 or 3.12 from python.org
        echo 3. Create new venv with compatible Python:
        echo    py -3.11 -m venv venv  (if available)
        echo    py -3.12 -m venv venv  (if available)
        echo 4. Run install_dependencies.bat again
    ) else (
        echo 1. Delete 'venv' folder and run install_dependencies.bat again
        echo 2. Check that all environment variables are set in .env file
        echo 3. Verify database connection settings
    )
    echo.
    echo Check the error messages above for specific details.
    echo.
)

pause
