"""
Configuration settings for the Deriv AI Trading Bot
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
DERIV_TOKEN = os.getenv('DERIV_TOKEN', '6D0lReGW3insnlx')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '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')

# WebSocket Configuration
DERIV_WS_URL = os.getenv('DERIV_WS_URL', 'wss://ws.binaryws.com/websockets/v3?app_id=1089')

# Trading Configuration - Optimized for Fast Index Trading
DEMO_MODE = os.getenv('DEMO_MODE', 'true').lower() == 'true'
INITIAL_STAKE = float(os.getenv('INITIAL_STAKE', 2.0))  # Increased for faster profits
MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', 100.0))
MAX_CONSECUTIVE_LOSSES = int(os.getenv('MAX_CONSECUTIVE_LOSSES', 4))  # Reduced for faster recovery

# Fast Trading Symbols - Optimized for Speed and Volume
TRADING_SYMBOLS = [
    'R_10',   # Volatility 10 Index - Fast, 1-second updates
    'R_25',   # Volatility 25 Index - Medium volatility
    'R_50',   # Volatility 50 Index - Higher volatility
    'R_75',   # Volatility 75 Index - Very volatile
    'R_100',  # Volatility 100 Index - Extreme volatility  
    'BOOM1000',   # Boom 1000 Index - Spike patterns
    'CRASH1000'   # Crash 1000 Index - Drop patterns
]

# Primary trading symbol for speed
DEFAULT_SYMBOL = 'R_10'  # Fastest moving index

# Trading Configuration (Optimized for Speed)
DEMO_MODE = os.getenv('DEMO_MODE', 'true').lower() == 'true'
INITIAL_STAKE = float(os.getenv('INITIAL_STAKE', 0.35))  # Start small and fast
MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', 50.0))  # Conservative daily limit
MAX_CONSECUTIVE_LOSSES = int(os.getenv('MAX_CONSECUTIVE_LOSSES', 3))  # Quick recovery
TRADE_FREQUENCY = int(os.getenv('TRADE_FREQUENCY', 5))  # Seconds between trades

# Smart Martingale Configuration - Optimized for Fast Index Trading
MARTINGALE_CONFIG = {
    'initial_stake': INITIAL_STAKE,
    'max_consecutive_losses': 3,  # Quick martingale activation
    'multiplier': 2.2,  # Slightly higher for faster recovery
    'max_stake': 75.0,  # Increased max stake for profitable trades
    'reset_on_win': True,
    'unpredictability_factor': 0.18,  # Higher randomness for indices
    'smart_scaling': True,
    'volatility_adjustment': True,  # Critical for volatile indices
    'progressive_multiplier': True,
    'pattern_memory_size': 12,  # Shorter memory for faster adaptation
    'time_based_variance': True,
    'session_performance_adjustment': True,
    'adaptive_confidence_threshold': 0.65,  # Lower threshold for more trades
    'max_martingale_level': 6,  # Higher levels for index volatility
    'cool_down_period': 180,  # Shorter cooldown (3 minutes)
    'emergency_brake_losses': 7,  # Higher tolerance for indices
    'balance_protection_ratio': 0.03,  # 3% per trade for faster growth
    'speed_trading_mode': True,  # Enable fast trading features
    'index_volatility_scaling': True,  # Special scaling for indices
    'quick_recovery_enabled': True,  # Faster recovery algorithms
    'profit_acceleration': True,  # Accelerate profits on winning streaks
}

# AI Analysis Configuration - Optimized for Fast Index Trading
AI_CONFIG = {
    'model': 'gpt-4-turbo-preview',  # Best model for analysis
    'temperature': 0.15,  # Very deterministic for index trading
    'max_tokens': 1200,  # Optimized for speed
    'analysis_timeframes': ['1m', '3m', '5m', '15m'],  # Shorter timeframes for speed
    'indicators_to_analyze': [
        'RSI', 'MACD', 'Bollinger Bands', 'Moving Averages',
        'Momentum', 'Volatility', 'Price Action', 'ATR',
        'Stochastic', 'Williams %R', 'Volume'  # Optimized for indices
    ],
    'market_context_analysis': True,
    'sentiment_analysis': True,
    'pattern_recognition': True,
    'confidence_weighting': True,
    'multi_timeframe_analysis': True,
    'adaptive_learning': True,
    'risk_assessment_depth': 'fast',  # Speed over depth for indices
    'index_specific_analysis': True,  # Special analysis for synthetic indices
    'volatility_focus': True,  # Focus on volatility patterns
    'quick_signal_generation': True,  # Faster signal processing
    'momentum_priority': True,  # Prioritize momentum in indices
}

# Technical Analysis Configuration
TECHNICAL_CONFIG = {
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bb_period': 20,
    'bb_std': 2,
    'ma_periods': [10, 20, 50, 100]
}

# Risk Management
RISK_CONFIG = {
    'max_position_size': 5.0,  # Percentage of balance
    'stop_loss_pips': 50,
    'take_profit_pips': 100,
    'max_open_positions': 3,
    'daily_profit_target': 50.0,
    'emergency_stop_loss': 200.0  # Total portfolio loss
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'trading_bot.log',
    'max_file_size': 10485760,  # 10MB
    'backup_count': 5
}

# Database Configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'mysql+pymysql://u626686198_tradingu:Kushan%4020001018@82.197.82.97:3306/u626686198_trading')

# MySQL Database Configuration (if using remote MySQL)
MYSQL_CONFIG = {
    'host': '82.197.82.97',
    'port': 3306,
    'database': 'u626686198_trading',
    'username': 'u626686198_tradingu',
    'password': 'Kushan@20001018',
    'charset': 'utf8mb4'
}

# Trading Symbols (Optimized for Speed and Profit)
TRADING_SYMBOLS = [
    'R_10',    # Volatility 10 Index - Fast, frequent signals
    'R_25',    # Volatility 25 Index - Good balance 
    'R_50',    # Volatility 50 Index - Higher volatility
    'R_75',    # Volatility 75 Index - More aggressive
    'R_100',   # Volatility 100 Index - Highest volatility
    'BOOM1000',   # Boom 1000 Index - Spike patterns
    'CRASH1000'   # Crash 1000 Index - Drop patterns
]

# Default trading symbol (fastest for profits)
DEFAULT_SYMBOL = 'R_10'  # High frequency, good for scalping
