# Deriv Trading Bot Configuration

# Trading Settings
DEFAULT_TRADE_AMOUNT = 1.0
MIN_TRADE_AMOUNT = 0.35  # Minimum allowed trade amount in USD
DEFAULT_DURATION_SECONDS = 15  # Changed from ticks to seconds-based trading
MIN_DURATION_SECONDS = 10
MAX_DURATION_SECONDS = 30
TRADE_INTERVAL_SECONDS = 5  # Reduced from 30 to 5 seconds for more frequent trades
ONE_TRADE_AT_A_TIME = False  # Allow multiple trades for testing
FORCE_STRATEGY_SIGNALS = True  # Force more aggressive signal generation

# Simulation Settings (for demo purposes)
SIMULATED_WIN_RATE = 0.6  # 60% win rate
PAYOUT_MULTIPLIER = 1.95  # 95% payout

# API Settings
DERIV_WEBSOCKET_URL = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
TRADING_SYMBOL = "R_100"  # Volatility 100 Index

# Server Settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

# Frontend Settings
REACT_PORT = 3000

# Balance Tracking Settings
SHOW_STARTING_BALANCE = True  # Display starting account balance
TRACK_SESSION_PNL = True      # Track profit/loss by session
ACCURATE_PNL_CALCULATION = True  # Use more accurate PnL calculation
