@echo off
echo 🔄 Resetting bot session and risk limits...
curl -X POST http://localhost:5000/api/risk-management/reset -H "Content-Type: application/json" -d "{\"full_reset\": true}"
echo.
echo 🔄 Disabling session trade limit...
curl -X POST http://localhost:5000/api/risk-management/disable-session-limit -H "Content-Type: application/json"
echo.
echo ✅ Reset complete! The bot should now trade continuously.
echo.
pause
