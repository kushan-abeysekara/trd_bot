@echo off
echo Checking Python version...
python --version

REM Check if Python 3.13 is being used
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Detected Python version: %PYTHON_VERSION%

if "%PYTHON_VERSION:~0,4%"=="3.13" (
    echo.
    echo INFO: Python 3.13 detected - using compatible package versions
    echo SQLAlchemy will be installed with Python 3.13 compatibility fixes
    echo.
)

echo.
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies...
if "%PYTHON_VERSION:~0,4%"=="3.13" (
    echo Installing with Python 3.13 optimizations...
    pip install typing-extensions>=4.8.0
    pip install -r requirements.txt
) else (
    pip install -r requirements.txt
)

REM Check if pip install actually failed
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to install dependencies!
    echo Error code: %ERRORLEVEL%
    echo.
    if "%PYTHON_VERSION:~0,4%"=="3.13" (
        echo Python 3.13 compatibility issue detected.
        echo.
        echo TRYING ALTERNATIVE SOLUTION:
        echo Installing packages individually with compatibility fixes...
        pip install --upgrade setuptools wheel
        pip install typing-extensions>=4.8.0
        pip install SQLAlchemy==2.0.25
        pip install -r requirements.txt --no-deps
        pip install Flask Flask-CORS Flask-JWT-Extended PyMySQL bcrypt requests python-dotenv flask-sqlalchemy marshmallow flask-marshmallow marshmallow-sqlalchemy email-validator phonenumbers cryptography
    )
    
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo FINAL RECOMMENDATION:
        echo Consider using Python 3.11 or 3.12 for guaranteed compatibility
        echo Download from: https://python.org/downloads/
        pause
        exit /b 1
    )
)

echo.
echo SUCCESS: All dependencies installed successfully!
echo.
echo Setup complete! 
echo To run the application:
echo 1. Run: run_app.bat
echo OR manually:
echo 1. Activate virtual environment: venv\Scripts\activate.bat
echo 2. Run the app: python app.py
echo.
pause
echo 1. Activate virtual environment: venv\Scripts\activate.bat
echo 2. Run the app: python app.py
echo.
pause
