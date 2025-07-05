import os
from dotenv import load_dotenv

# Load environment variables based on environment
env = os.getenv('FLASK_ENV', 'development')
if env == 'production':
    load_dotenv('.env.production')
else:
    load_dotenv('.env.development')

# Deriv Trading Bot Configuration

# Trading Settings
DEFAULT_TRADE_AMOUNT = 1.0
DEFAULT_DURATION_TICKS = 5
TRADE_INTERVAL_SECONDS = 30

# Simulation Settings (for demo purposes)
SIMULATED_WIN_RATE = 0.6  # 60% win rate
PAYOUT_MULTIPLIER = 1.95  # 95% payout

# API Settings
DERIV_WEBSOCKET_URL = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
TRADING_SYMBOL = "R_10"  # Volatility 100 Index

# Server Settings - Environment specific
FLASK_HOST = os.getenv('FLASK_HOST', "0.0.0.0")
FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'

# CORS Settings
CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:8080').split(',')

# Frontend Settings
REACT_PORT = 8080

# URL Settings
FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:8080')
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:5000/api')
