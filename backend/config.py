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
TRADING_SYMBOL = "R_100"  # Volatility 100 Index

# Server Settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

# Frontend Settings
REACT_PORT = 3000
