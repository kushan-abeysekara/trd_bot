@echo off
echo 🚀 Setting up production environment...

REM Create production .env file
(
echo # Production Environment Variables
echo FLASK_ENV=production
echo FLASK_DEBUG=False
echo DERIV_APP_ID=1089
echo CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,https://tradingbot-4iuxi.ondigitalocean.app
echo HOST=0.0.0.0
echo PORT=5000
) > .env

echo ✅ Production .env file created

REM Install dependencies
echo 📦 Installing Python dependencies...
pip install -r requirements.txt

echo 🌐 CORS configured for:
echo   - http://localhost:3000 (local development)
echo   - https://tradingbot-4iuxi.ondigitalocean.app (production)

echo 🚀 Ready to deploy!
echo Run with: python app.py

pause
