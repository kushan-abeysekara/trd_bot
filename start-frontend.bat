@echo off
echo Starting Deriv Trading Bot Frontend Server...
echo.

cd frontend
echo Installing/updating dependencies...
npm install
echo.
echo Starting development server on port 8080...
npm start

pause
