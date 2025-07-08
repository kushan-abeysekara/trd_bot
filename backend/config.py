# Deriv Trading Bot Configuration
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Trading Settings
DEFAULT_TRADE_AMOUNT = 1.0
MIN_TRADE_AMOUNT = 0.35  # Minimum allowed trade amount in USD
DEFAULT_DURATION_SECONDS = 3  # Changed for 1-tick trading (approx. 3 seconds)
MIN_DURATION_SECONDS = 2
MAX_DURATION_SECONDS = 5
TRADE_INTERVAL_SECONDS = 3  # Reduced to 3 seconds for more manageable trades
ONE_TRADE_AT_A_TIME = False  # Allow multiple trades for testing
FORCE_STRATEGY_SIGNALS = True  # Force more aggressive signal generation
AGGRESSIVE_SIGNAL_MODE = True  # Enable aggressive signal generation
SIGNAL_CONFIDENCE_THRESHOLD = 0.50  # Lowered threshold for more signals

# Risk Management Settings
MAX_CONSECUTIVE_LOSSES = 50  # Increased from 50 to 50 consecutive losses
MAX_SESSION_TRADES = 100  # Cap at 100 trades per session
DEFAULT_RISK_REWARD_RATIO = 2.0  # Default 2:1 risk-reward ratio

# Strategy Optimization Settings
ENABLE_PARAMETER_OPTIMIZATION = True
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
RSI_PERIOD = 7
BOLLINGER_PERIOD = 20
BOLLINGER_STD_DEV = 2.0
ATR_PERIOD = 14
DONCHIAN_PERIOD = 20

# Simulation Settings (for demo purposes)
SIMULATED_WIN_RATE = 0.6  # 60% win rate
PAYOUT_MULTIPLIER = 1.95  # 95% payout

# API Settings
DERIV_WEBSOCKET_URL = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
TRADING_SYMBOL = "R_100"  # Volatility 100 Index

# Environment-based configuration
FLASK_ENV = os.getenv('FLASK_ENV', 'development')
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))

# CORS Settings - Allow multiple origins
CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000,http://localhost:8080,https://tradingbot-4iuxi.ondigitalocean.app').split(',')

# Frontend Settings
REACT_PORT = 8080

# Balance Tracking Settings
SHOW_STARTING_BALANCE = True  # Display starting account balance
TRACK_SESSION_PNL = True      # Track profit/loss by session
ACCURATE_PNL_CALCULATION = True  # Use more accurate PnL calculation
