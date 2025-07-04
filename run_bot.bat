@echo off
echo Starting Trading Bot...

cd backend
echo Starting backend server...
start powershell -NoProfile -Command "python app.py"
timeout /t 5

cd ../frontend
echo Installing frontend dependencies if needed...
npm install
echo Starting frontend application...
start powershell -NoProfile -Command "npm start"

echo Trading Bot started successfully!
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
