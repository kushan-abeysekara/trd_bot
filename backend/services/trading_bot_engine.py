import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from threading import Thread, Event
import sqlite3
from flask import current_app

from utils.deriv_service import DerivService
from services.ml_strategies import MLStrategyManager
from models.user import User

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Trading signal from strategy analysis"""
    action: str  # 'buy', 'sell', 'call', 'put', etc.
    confidence: float  # 0.0 to 1.0
    strategy_name: str
    signal_strength: float
    metadata: Dict = None

@dataclass
class TradeResult:
    """Result of a trade execution"""
    trade_id: str
    contract_type: str
    action: str
    stake_amount: float
    payout: float
    result: str  # 'win', 'loss', 'pending'
    timestamp: datetime
    metadata: Dict = None

class TradingMode(Enum):
    MODE_A = "MA_RSI_TREND"
    MODE_B = "PRICE_ACTION_BOUNCE"
    MODE_C = "RANDOM_ENTRY_SMART_EXIT"

class ContractType(Enum):
    RISE_FALL = "rise_fall"
    TOUCH_NO_TOUCH = "touch_no_touch"
    IN_OUT = "in_out"
    ASIANS = "asians"
    DIGITS = "digits"
    RESET_CALL_PUT = "reset_call_put"
    HIGH_LOW_TICKS = "high_low_ticks"
    ONLY_UPS_DOWNS = "only_ups_downs"
    MULTIPLIERS = "multipliers"
    ACCUMULATORS = "accumulators"

@dataclass
class TradingSettings:
    daily_stop_loss_percent: float = 10.0
    daily_target_percent: float = 20.0
    base_stake_percent: float = 2.0
    max_stake_percent: float = 10.0
    cool_down_after_loss: int = 3
    strategy_switch_wins: int = 3
    strategy_switch_losses: int = 2
    reevaluate_trades: int = 10
    enable_martingale: bool = True
    martingale_multiplier: float = 1.5
    max_martingale_steps: int = 3

@dataclass
class TradeResult:
    contract_id: str
    contract_type: ContractType
    entry_price: float
    exit_price: float
    stake_amount: float
    profit_loss: float
    duration: int
    strategy_used: TradingMode
    timestamp: datetime
    success: bool

class TradingBotEngine:
    def __init__(self, user_id: int, api_token: str, account_type: str = 'demo'):
        self.user_id = user_id
        self.api_token = api_token
        self.account_type = account_type
        self.deriv_service = DerivService()
        self.ml_strategy_manager = MLStrategyManager()
        
        # Trading state
        self.is_active = False
        self.is_trading = False
        self.stop_event = Event()
        self.trading_thread = None
        
        # Settings and state
        self.settings = TradingSettings()
        self.current_mode = TradingMode.MODE_A
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.trades_since_evaluation = 0
        self.in_cooldown = False
        self.cooldown_trades_left = 0
        self.martingale_step = 0
        
        # Account data
        self.account_balance = 0.0
        self.starting_balance = 0.0
        self.max_daily_loss = 0.0
        self.daily_target = 0.0
        
        # Trade history
        self.trade_history: List[TradeResult] = []
        self.current_trades: Dict[str, Any] = {}
        
        # Strategy performance tracking
        self.strategy_performance = {
            TradingMode.MODE_A: {'wins': 0, 'losses': 0, 'profit': 0.0},
            TradingMode.MODE_B: {'wins': 0, 'losses': 0, 'profit': 0.0},
            TradingMode.MODE_C: {'wins': 0, 'losses': 0, 'profit': 0.0}
        }
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize trading bot database tables"""
        try:
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            
            # Create trading sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_end TIMESTAMP,
                    starting_balance REAL,
                    ending_balance REAL,
                    total_pnl REAL,
                    total_trades INTEGER,
                    win_rate REAL,
                    strategy_used TEXT,
                    account_type TEXT,
                    settings_json TEXT
                )
            ''')
            
            # Create trade results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_id INTEGER,
                    contract_id TEXT,
                    contract_type TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    stake_amount REAL,
                    profit_loss REAL,
                    duration INTEGER,
                    strategy_used TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN,
                    ml_prediction_confidence REAL,
                    market_conditions_json TEXT,
                    FOREIGN KEY (session_id) REFERENCES trading_sessions (id)
                )
            ''')
            
            # Create ML training data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contract_type TEXT,
                    market_data_json TEXT,
                    technical_indicators_json TEXT,
                    price_history_json TEXT,
                    outcome BOOLEAN,
                    profit_loss REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    strategy_used TEXT,
                    market_volatility REAL,
                    trend_direction TEXT
                )
            ''')
            
            # Create strategy performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    strategy_name TEXT,
                    contract_type TEXT,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    total_profit REAL DEFAULT 0.0,
                    win_rate REAL DEFAULT 0.0,
                    avg_profit_per_trade REAL DEFAULT 0.0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Trading bot database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
    
    async def start_trading(self):
        """Start the trading bot"""
        if self.is_active:
            return {"success": False, "message": "Bot is already active"}
        
        try:
            # Validate API token and get account info
            validation_result = self.deriv_service.validate_token(self.api_token)
            if not validation_result['valid']:
                return {"success": False, "message": f"Invalid API token: {validation_result['message']}"}
            
            # Get account balance
            balance_result = self.deriv_service.get_account_info(self.api_token)
            if not balance_result['success']:
                return {"success": False, "message": f"Failed to get account info: {balance_result['message']}"}
            
            self.account_balance = balance_result['data']['balance']
            self.starting_balance = self.account_balance
            self.max_daily_loss = self.starting_balance * (self.settings.daily_stop_loss_percent / 100)
            self.daily_target = self.starting_balance * (self.settings.daily_target_percent / 100)
            
            # Reset daily stats
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.consecutive_wins = 0
            self.consecutive_losses = 0
            self.trades_since_evaluation = 0
            self.in_cooldown = False
            self.martingale_step = 0
            
            # Start trading session in database
            self._start_trading_session()
            
            # Start trading thread
            self.is_active = True
            self.stop_event.clear()
            self.trading_thread = Thread(target=self._trading_loop)
            self.trading_thread.daemon = True
            self.trading_thread.start()
            
            logger.info(f"Trading bot started for user {self.user_id} with balance {self.account_balance}")
            return {
                "success": True, 
                "message": "Trading bot started successfully",
                "data": {
                    "starting_balance": self.starting_balance,
                    "daily_target": self.daily_target,
                    "max_daily_loss": self.max_daily_loss,
                    "current_mode": self.current_mode.value
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to start trading bot: {str(e)}")
            return {"success": False, "message": f"Failed to start bot: {str(e)}"}
    
    async def stop_trading(self):
        """Stop the trading bot"""
        if not self.is_active:
            return {"success": False, "message": "Bot is not active"}
        
        try:
            self.is_active = False
            self.stop_event.set()
            
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=30)
            
            # End trading session in database
            self._end_trading_session()
            
            logger.info(f"Trading bot stopped for user {self.user_id}")
            return {
                "success": True,
                "message": "Trading bot stopped successfully",
                "data": {
                    "final_balance": self.account_balance,
                    "daily_pnl": self.daily_pnl,
                    "total_trades": self.daily_trades,
                    "win_rate": self._calculate_win_rate()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to stop trading bot: {str(e)}")
            return {"success": False, "message": f"Failed to stop bot: {str(e)}"}
    
    def _trading_loop(self):
        """Main trading loop"""
        logger.info("Trading loop started")
        
        while self.is_active and not self.stop_event.is_set():
            try:
                # Check safety limits
                if self._check_safety_limits():
                    logger.info("Safety limits triggered, stopping trading")
                    break
                
                # Check if in cooldown
                if self.in_cooldown:
                    if self.cooldown_trades_left > 0:
                        self.cooldown_trades_left -= 1
                        logger.info(f"In cooldown, skipping signal. {self.cooldown_trades_left} trades left")
                        self.stop_event.wait(30)  # Wait 30 seconds
                        continue
                    else:
                        self.in_cooldown = False
                        logger.info("Cooldown period ended")
                
                # Update account balance
                self._update_account_balance()
                
                # Get market data
                market_data = self._get_market_data()
                if not market_data:
                    self.stop_event.wait(10)
                    continue
                
                # Select best contract type and strategy
                contract_type, strategy_confidence = self._select_best_contract_type(market_data)
                
                # Get trading signal using ML
                signal = self.ml_strategy_manager.get_trading_signal(
                    contract_type, 
                    self.current_mode, 
                    market_data
                )
                
                if signal and signal['confidence'] > 0.6:  # Only trade if confidence > 60%
                    # Calculate stake amount
                    stake_amount = self._calculate_stake_amount()
                    
                    # Execute trade
                    trade_result = self._execute_trade_sync(
                        contract_type, 
                        signal, 
                        stake_amount, 
                        market_data
                    )
                    
                    if trade_result:
                        # Process trade result
                        self._process_trade_result(trade_result)
                        
                        # Update strategy performance
                        self._update_strategy_performance(trade_result)
                        
                        # Check for strategy switching
                        self._check_strategy_switching()
                        
                        # Save ML training data
                        self._save_ml_training_data(market_data, trade_result)
                
                # Wait before next signal
                self.stop_event.wait(10)  # Wait 10 seconds between signals
                
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                self.stop_event.wait(30)  # Wait 30 seconds on error
        
        logger.info("Trading loop ended")
    
    def _check_safety_limits(self) -> bool:
        """Check if safety limits are reached"""
        # Daily loss limit
        if abs(self.daily_pnl) >= self.max_daily_loss and self.daily_pnl < 0:
            logger.warning(f"Daily stop loss reached: {self.daily_pnl}")
            return True
        
        # Daily target reached
        if self.daily_pnl >= self.daily_target:
            logger.info(f"Daily target reached: {self.daily_pnl}")
            return True
        
        # Account balance protection
        current_loss_percent = ((self.starting_balance - self.account_balance) / self.starting_balance) * 100
        if current_loss_percent >= self.settings.daily_stop_loss_percent:
            logger.warning(f"Account loss limit reached: {current_loss_percent}%")
            return True
        
        return False
    
    def _update_account_balance(self):
        """Update current account balance"""
        try:
            balance_result = self.deriv_service.get_real_time_balance(self.api_token)
            if balance_result['success']:
                self.account_balance = balance_result['data']['balance']
        except Exception as e:
            logger.error(f"Failed to update account balance: {str(e)}")
    
    def _get_market_data(self) -> Optional[Dict]:
        """Get current market data for analysis"""
        try:
            # In production, integrate with RealTimeMarketAnalyzer
            # For now, simulate comprehensive market data
            
            import random
            import time
            
            base_price = 1.0123 + (random.random() - 0.5) * 0.001
            volatility = random.uniform(0.005, 0.03)
            
            # Generate price history
            price_history = []
            current_time = datetime.now()
            
            for i in range(100):
                price = base_price + random.gauss(0, volatility) * 0.1
                price_history.append({
                    'price': price,
                    'timestamp': current_time - timedelta(seconds=i),
                    'volume': random.uniform(100, 1000)
                })
            
            # Calculate technical indicators
            prices = [p['price'] for p in price_history]
            
            # Trend calculation
            if len(prices) >= 20:
                short_ma = sum(prices[:10]) / 10
                long_ma = sum(prices[:20]) / 20
                trend = (short_ma - long_ma) / long_ma
            else:
                trend = 0
            
            # RSI calculation (simplified)
            rsi = 50 + random.gauss(0, 15)
            rsi = max(0, min(100, rsi))
            
            # Momentum
            momentum = (prices[0] - prices[9]) / prices[9] if len(prices) > 9 else 0
            
            return {
                'current_price': base_price,
                'price_history': price_history,
                'volatility': volatility,
                'trend': trend,
                'rsi': rsi,
                'momentum': momentum,
                'price_change': (prices[0] - prices[1]) / prices[1] if len(prices) > 1 else 0,
                'timestamp': current_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            return None
    
    def _select_best_contract_type(self, market_data: Dict) -> tuple[ContractType, float]:
        """Select the best contract type based on market conditions and performance"""
        try:
            volatility = market_data.get('volatility', 0.01)
            trend = market_data.get('trend', 0)
            rsi = market_data.get('rsi', 50)
            
            # Get contract performance data
            contract_performance = self.get_contract_performance_summary()
            
            # Score each contract type
            contract_scores = {}
            
            for contract_type in ContractType:
                score = 0.5  # Base score
                
                # Market condition suitability
                if contract_type == ContractType.RISE_FALL:
                    # Good for trending markets
                    if abs(trend) > 0.001:
                        score += 0.2
                    if 30 < rsi < 70:
                        score += 0.15
                
                elif contract_type == ContractType.TOUCH_NO_TOUCH:
                    # Good for volatile markets
                    if volatility > 0.015:
                        score += 0.25
                    if rsi > 70 or rsi < 30:
                        score += 0.1
                
                elif contract_type == ContractType.IN_OUT:
                    # Good for range-bound markets
                    if volatility < 0.01 and abs(trend) < 0.001:
                        score += 0.3
                
                elif contract_type == ContractType.ASIANS:
                    # Good for stable markets
                    if volatility < 0.015:
                        score += 0.2
                    if abs(trend) < 0.001:
                        score += 0.15
                
                elif contract_type == ContractType.DIGITS:
                    # Good for very short-term
                    if volatility > 0.01:
                        score += 0.15
                
                elif contract_type == ContractType.MULTIPLIERS:
                    # Good for strong trending markets
                    if abs(trend) > 0.002 and volatility < 0.02:
                        score += 0.25
                
                # Historical performance adjustment
                performance = contract_performance.get(contract_type.value, {})
                win_rate = performance.get('win_rate', 0.5)
                total_trades = performance.get('total_trades', 0)
                
                if total_trades > 10:  # Enough data
                    performance_score = (win_rate - 0.5) * 0.4  # Max ±20% adjustment
                    score += performance_score
                
                # Recent performance boost
                if total_trades > 0:
                    recent_profit = performance.get('total_profit', 0)
                    if recent_profit > 0:
                        score += 0.1
                
                contract_scores[contract_type] = max(0.1, min(0.9, score))
            
            # Select best contract type
            best_contract = max(contract_scores, key=contract_scores.get)
            best_score = contract_scores[best_contract]
            
            logger.info(f"Selected contract type: {best_contract.value} (score: {best_score:.2f})")
            
            return best_contract, best_score
            
        except Exception as e:
            logger.error(f"Error selecting contract type: {str(e)}")
            return ContractType.RISE_FALL, 0.5
    
    def _calculate_stake_amount(self) -> float:
        """Calculate optimal stake amount using advanced money management"""
        try:
            base_stake = self.account_balance * (self.settings.base_stake_percent / 100)
            
            # Apply Martingale if enabled
            if self.settings.enable_martingale and self.martingale_step > 0:
                martingale_multiplier = self.settings.martingale_multiplier ** self.martingale_step
                stake = base_stake * martingale_multiplier
            else:
                stake = base_stake
            
            # Volatility adjustment
            market_data = self._get_market_data()
            if market_data:
                volatility = market_data.get('volatility', 0.01)
                
                # Higher volatility = lower stake
                volatility_adjustment = 1.0 - (volatility - 0.01) * 5
                volatility_adjustment = max(0.5, min(1.5, volatility_adjustment))
                stake *= volatility_adjustment
            
            # Consecutive losses adjustment
            if self.consecutive_losses > 0:
                # Reduce stake after consecutive losses
                loss_adjustment = 0.9 ** min(self.consecutive_losses, 5)
                stake *= loss_adjustment
            
            # Consecutive wins adjustment  
            if self.consecutive_wins > 2:
                # Slightly increase stake after wins (but cap it)
                win_adjustment = min(1.3, 1 + (self.consecutive_wins - 2) * 0.1)
                stake *= win_adjustment
            
            # Account balance protection
            balance_protection = self.account_balance * 0.02  # Never risk more than 2% of balance
            stake = min(stake, balance_protection)
            
            # Max stake limit
            max_stake = self.account_balance * (self.settings.max_stake_percent / 100)
            stake = min(stake, max_stake)
            
            # Minimum stake
            stake = max(stake, 1.0)
            
            logger.info(f"Calculated stake: ${stake:.2f} (Martingale step: {self.martingale_step})")
            
            return stake
            
        except Exception as e:
            logger.error(f"Error calculating stake amount: {str(e)}")
            return self.account_balance * 0.01  # Fallback to 1%

    def _get_market_data(self) -> Dict:
        """Get enhanced market data with ML features."""
        try:
            # Get basic market analysis
            if hasattr(self, 'market_analyzer') and self.market_analyzer:
                analysis = self.market_analyzer.get_market_analysis(self.symbol)
            else:
                # Fallback data if market analyzer not available
                analysis = {
                    'current_price': 1.0120,
                    'volatility': 0.015,
                    'trend': 0.05,
                    'rsi': 65.0,
                    'macd': {'macd': 0.001, 'signal': 0.0005},
                    'momentum': 0.001,
                    'timestamp': datetime.utcnow(),
                    'price_change_1m': 0.0001,
                    'price_change_5m': 0.0005,
                    'sma_20': 1.0120,
                    'bollinger_position': 0.5,
                    'atr': 0.001,
                    'support_distance': 0.002,
                    'resistance_distance': 0.002,
                    'digit_history': [1, 2, 3, 4, 5, 6, 7, 8, 9, 0] * 2,
                    'tick_history': [1.0120, 1.0121, 1.0122, 1.0123, 1.0124],
                    'volume': 1000,
                    'market_regime': 0.5,
                    'trend_consistency': 0.6
                }
            
            # Enhance analysis data with additional fields needed for ML
            current_price = analysis.get('current_price', 1.0)
            indicators = analysis.get('technical_indicators', {})
            
            enhanced_data = {
                'current_price': current_price,
                'volatility': indicators.get('volatility', 0.01),
                'trend': indicators.get('trend_strength', 0) / 100,  # Normalize
                'rsi': indicators.get('rsi', 50),
                'macd': indicators.get('macd', {'macd': 0, 'signal': 0}),
                'momentum': indicators.get('momentum', 0),
                'timestamp': datetime.utcnow(),
                'price_change_1m': 0.0001,  # Would come from price history
                'price_change_5m': 0.0005,
                'sma_20': current_price * 0.999,  # Approximate
                'bollinger_position': 0.5,
                'atr': indicators.get('volatility', 0.01) * 0.5,
                'support_distance': 0.002,
                'resistance_distance': 0.002,
                'digit_history': getattr(self, 'market_analyzer', None) and getattr(self.market_analyzer, 'digit_history', []) or [1, 2, 3, 4, 5] * 4,
                'tick_history': [current_price + i * 0.0001 for i in range(-2, 3)],
                'volume': 1000,
                'market_regime': 1.0 if abs(indicators.get('trend_strength', 0)) > 50 else 0.0,
                'trend_consistency': 0.6,
                'autocorrelation': 0.5,
                'hurst_exponent': 0.5,
                'time_of_day_effect': 0.5,
                'regime_stability': 0.7,
                'expected_drift': indicators.get('trend_strength', 0) / 10000,
                'tick_momentum': 0.001,
                'microtrend': 0.0001,
                'price_velocity': 0.001,
                'time_to_support': 300,
                'time_to_resistance': 300,
                'volatility_trend': 0.01,
                'bollinger_width': 0.002,
                'price_reversals_1h': 3,
                'avg_candle_size': 0.001,
                'breakout_probability': 0.3,
                'mean_reversion_strength': 0.5,
                'price_drift': 0.0001,
                'volatility_of_volatility': 0.001,
                'market_microstructure': 0.5,
                'trend_persistence': 0.6,
                'volatility_clustering': 0.4,
                'mean_reversion': 0.5,
                'jump_risk': 0.1,
                'barrier_proximity': 0.5,
                'accumulation_rate': 0.01,
                'time_decay': 1.0,
                'knock_out_probability': 0.2,
                'optimal_holding_time': 600,
                'compound_growth_potential': 0.1
            }
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Failed to get market data: {str(e)}")
            return None
    
    def _select_best_contract_type(self, market_data: Dict) -> tuple:
        """Select the best contract type based on comprehensive market analysis"""
        try:
            volatility = market_data.get('volatility', 0.01)
            trend_strength = abs(market_data.get('trend', 0))
            rsi = market_data.get('rsi', 50)
            momentum = market_data.get('momentum', 0)
            market_regime = market_data.get('market_regime', 0.5)
            
            # Score each contract type based on market conditions
            contract_scores = {}
            
            # Rise/Fall - best for trending markets
            rise_fall_score = 0.6  # Base score
            if trend_strength > 0.02:
                rise_fall_score += 0.3
            if 30 <= rsi <= 70:  # Not extreme RSI
                rise_fall_score += 0.2
            if 0.005 < volatility < 0.03:  # Moderate volatility
                rise_fall_score += 0.2
            contract_scores[ContractType.RISE_FALL] = rise_fall_score
            
            # Touch/No Touch - best for volatile markets with clear levels
            touch_score = 0.5
            if volatility > 0.015:
                touch_score += 0.4
            if trend_strength < 0.01:  # Sideways market
                touch_score += 0.2
            if rsi > 70 or rsi < 30:  # Near extremes
                touch_score += 0.2
            contract_scores[ContractType.TOUCH_NO_TOUCH] = touch_score
            
            # In/Out (Boundary) - best for ranging markets
            boundary_score = 0.4
            if volatility < 0.015 and trend_strength < 0.01:
                boundary_score += 0.5
            if 40 <= rsi <= 60:  # Neutral RSI
                boundary_score += 0.3
            if market_regime < 0.3:  # Ranging market
                boundary_score += 0.2
            contract_scores[ContractType.IN_OUT] = boundary_score
            
            # Asian Options - best for low volatility, trending markets
            asian_score = 0.3
            if volatility < 0.01:
                asian_score += 0.4
            if trend_strength > 0.005:
                asian_score += 0.3
            if abs(momentum) > 0.0005:  # Consistent momentum
                asian_score += 0.2
            contract_scores[ContractType.ASIANS] = asian_score
            
            # Digits - specialized for price precision trading
            digits_score = 0.4
            current_digit = int((market_data.get('current_price', 1.0) * 100) % 10)
            digit_history = market_data.get('digit_history', [])
            if digit_history:
                # Check for digit patterns
                recent_digits = digit_history[-10:]
                if len(set(recent_digits)) <= 6:  # Low digit diversity
                    digits_score += 0.3
                if current_digit in [0, 5]:  # Common digits
                    digits_score += 0.2
            if volatility > 0.01:  # Some volatility needed
                digits_score += 0.2
            contract_scores[ContractType.DIGITS] = digits_score
            
            # Reset Call/Put - benefits from volatility
            reset_score = 0.3
            if volatility > 0.02:
                reset_score += 0.4
            if trend_strength > 0.015:
                reset_score += 0.3
            if abs(momentum) > 0.001:
                reset_score += 0.2
            contract_scores[ContractType.RESET_CALL_PUT] = reset_score
            
            # High/Low Ticks - very specialized, high risk
            ticks_score = 0.2
            if volatility > 0.025:  # Very high volatility
                ticks_score += 0.3
            tick_history = market_data.get('tick_history', [])
            if len(tick_history) >= 5:
                tick_range = max(tick_history) - min(tick_history)
                if tick_range > volatility * 2:
                    ticks_score += 0.2
            contract_scores[ContractType.HIGH_LOW_TICKS] = ticks_score
            
            # Only Ups/Downs - best for strong trends
            only_score = 0.3
            if trend_strength > 0.03 and volatility < 0.02:
                only_score += 0.5
            if abs(momentum) > 0.002:
                only_score += 0.2
            if market_regime > 0.7:  # Strong trending
                only_score += 0.2
            contract_scores[ContractType.ONLY_UPS_DOWNS] = only_score
            
            # Multipliers - for experienced strategies
            multiplier_score = 0.4
            if self.consecutive_wins >= 2:  # On winning streak
                multiplier_score += 0.3
            if volatility > 0.01 and trend_strength > 0.01:
                multiplier_score += 0.2
            if self.daily_pnl > 0:  # Profitable day
                multiplier_score += 0.1
            contract_scores[ContractType.MULTIPLIERS] = multiplier_score
            
            # Accumulators - for long-term trends
            accumulator_score = 0.2
            if trend_strength > 0.02 and volatility < 0.015:
                accumulator_score += 0.4
            if self.consecutive_wins >= 3:  # Very confident
                accumulator_score += 0.3
            contract_scores[ContractType.ACCUMULATORS] = accumulator_score
            
            # Factor in bot's current performance for contract selection
            for contract_type in contract_scores:
                # Boost score for contracts we're performing well with
                if contract_type in self.strategy_performance:
                    performance = self.strategy_performance[contract_type]
                    total_trades = performance['wins'] + performance['losses']
                    if total_trades >= 10:
                        win_rate = performance['wins'] / total_trades
                        avg_profit = performance['profit'] / total_trades
                        
                        if win_rate > 0.6:  # Good win rate
                            contract_scores[contract_type] *= 1.2
                        elif win_rate < 0.4:  # Poor win rate
                            contract_scores[contract_type] *= 0.7
                        
                        if avg_profit > 0:  # Profitable
                            contract_scores[contract_type] *= 1.1
            
            # Risk-based adjustments
            current_loss_pct = abs(self.daily_pnl) / max(1.0, self.starting_balance) if self.daily_pnl < 0 else 0
            
            if current_loss_pct > 0.05:  # If losing > 5%, prefer safer contracts
                contract_scores[ContractType.ASIANS] *= 1.3
                contract_scores[ContractType.RISE_FALL] *= 1.2
                contract_scores[ContractType.HIGH_LOW_TICKS] *= 0.5
                contract_scores[ContractType.DIGITS] *= 0.7
            
            # Select best contract type
            best_contract = max(contract_scores, key=contract_scores.get)
            best_score = contract_scores[best_contract]
            
            # Confidence based on score
            confidence = min(0.95, max(0.3, best_score))
            
            logger.info(f"Selected contract: {best_contract.value} with confidence {confidence:.2f}")
            logger.debug(f"Contract scores: {[(ct.value, score) for ct, score in sorted(contract_scores.items(), key=lambda x: x[1], reverse=True)]}")
            
            return best_contract, confidence
            
        except Exception as e:
            logger.error(f"Error selecting contract type: {str(e)}")
            # Fallback to Rise/Fall
            return ContractType.RISE_FALL, 0.6
    
    def _calculate_stake_amount(self) -> float:
        """Calculate stake amount using advanced money management with smart Martingale"""
        try:
            # Base stake calculation
            base_stake = self.account_balance * (self.settings.base_stake_percent / 100)
            
            # Account balance-based dynamic sizing
            balance_factor = min(2.0, self.account_balance / max(1.0, self.starting_balance))
            
            # Win/loss streak adjustments
            if self.consecutive_wins >= 2:
                # Increase stake after wins (compound growth)
                win_multiplier = 1 + (self.consecutive_wins * 0.1)  # 10% per win
                stake = base_stake * min(win_multiplier, 1.5)  # Cap at 50% increase
            elif self.consecutive_losses >= 1 and self.settings.enable_martingale:
                # Smart Martingale - more conservative than traditional
                martingale_multiplier = self.settings.martingale_multiplier
                
                # Reduce multiplier based on current loss
                loss_ratio = abs(self.daily_pnl) / max(1.0, self.starting_balance)
                if loss_ratio > 0.05:  # If losing more than 5%
                    martingale_multiplier = min(martingale_multiplier, 1.2)  # Reduce aggression
                
                stake = base_stake * (martingale_multiplier ** min(self.martingale_step, self.settings.max_martingale_steps))
            else:
                stake = base_stake * balance_factor
            
            # Daily profit/loss adjustments
            daily_profit_ratio = self.daily_pnl / max(1.0, self.starting_balance)
            if daily_profit_ratio > 0.1:  # If up more than 10%
                stake *= 0.8  # Reduce stake when very profitable
            elif daily_profit_ratio < -0.05:  # If down more than 5%
                stake *= 0.7  # Reduce stake when losing
            
            # Time-based adjustments (reduce stake near end of trading day)
            current_hour = datetime.utcnow().hour
            if current_hour >= 20 or current_hour <= 2:  # Late trading hours
                stake *= 0.8
            
            # Volatility-based adjustments
            market_data = self._get_market_data()
            if market_data:
                volatility = market_data.get('volatility', 0.01)
                if volatility > 0.02:  # High volatility
                    stake *= 0.9
                elif volatility < 0.005:  # Low volatility
                    stake *= 1.1
            
            # Apply maximum stake limit
            max_stake = self.account_balance * (self.settings.max_stake_percent / 100)
            stake = min(stake, max_stake)
            
            # Apply minimum stake
            min_stake = max(1.0, self.account_balance * 0.001)  # 0.1% minimum
            stake = max(stake, min_stake)
            
            # Safety check - never risk more than 15% of balance in single trade
            absolute_max = self.account_balance * 0.15
            stake = min(stake, absolute_max)
            
            logger.info(f"Calculated stake: {stake:.2f} (Base: {base_stake:.2f}, Balance: {self.account_balance:.2f})")
            return round(stake, 2)
            
        except Exception as e:
            logger.error(f"Error calculating stake amount: {str(e)}")
            return max(1.0, self.account_balance * 0.02)  # Fallback to 2% of balance
    
    def _execute_trade_sync(self, contract_type: ContractType, signal: dict, stake_amount: float, market_data: Dict) -> Optional[TradeResult]:
        """Synchronous version of _execute_trade for use in the trading loop"""
        try:
            # For demo purposes - in production, use actual Deriv API
            contract_id = f"T{int(datetime.now().timestamp())}"
            entry_price = market_data.get('current_price', 1.0)
            
            # Simulate trade execution
            logger.info(f"Executing {contract_type.value} trade: {signal['action']} with stake ${stake_amount:.2f}")
            
            # Store active trade
            self.current_trades[contract_id] = {
                'contract_type': contract_type,
                'action': signal['action'],
                'entry_price': entry_price,
                'stake_amount': stake_amount,
                'duration': signal.get('duration', 60),
                'entry_time': datetime.now(),
                'signal_confidence': signal['confidence'],
                'market_data': market_data
            }
            
            # Create trade result (in production this would be returned from Deriv API)
            return TradeResult(
                trade_id=contract_id,
                contract_type=contract_type.value,
                action=signal['action'],
                stake_amount=stake_amount,
                payout=0.0,  # Will be updated when trade completes
                result='pending',
                timestamp=datetime.now(),
                metadata={'entry_price': entry_price, 'confidence': signal['confidence']}
            )
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return None

    async def _execute_trade(self, contract_type: ContractType, signal: TradingSignal, stake_amount: float, market_data: Dict) -> Optional[TradeResult]:
        """Execute a trade with the given parameters (async version for API usage)"""
        try:
            # For demo purposes - in production, use actual Deriv API
            contract_id = f"T{int(datetime.now().timestamp())}"
            entry_price = signal.entry_price if hasattr(signal, 'entry_price') else 1.0
            
            # Simulate trade execution
            logger.info(f"Executing {contract_type.value} trade: {signal.action} with stake ${stake_amount:.2f}")
            
            # Store active trade
            self.current_trades[contract_id] = {
                'contract_type': contract_type,
                'action': signal.action,
                'entry_price': entry_price,
                'stake_amount': stake_amount,
                'duration': getattr(signal, 'duration', 60),
                'entry_time': datetime.now(),
                'signal_confidence': signal.confidence,
                'market_data': market_data
            }
            
            # Create trade result (in production this would be returned from Deriv API)
            return TradeResult(
                trade_id=contract_id,
                contract_type=contract_type.value,
                action=signal.action,
                stake_amount=stake_amount,
                payout=0.0,  # Will be updated when trade completes
                result='pending',
                timestamp=datetime.now(),
                metadata={'entry_price': entry_price, 'confidence': signal.confidence}
            )
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return None
    
    def _process_trade_result(self, trade_result: TradeResult):
        """Process trade result and update bot state"""
        try:
            # Update daily P&L
            self.daily_pnl += trade_result.profit_loss
            self.daily_trades += 1
            
            # Update account balance
            self.account_balance += trade_result.profit_loss
            
            # Update consecutive wins/losses
            if trade_result.success:
                self.consecutive_wins += 1
                self.consecutive_losses = 0
                self.martingale_step = 0  # Reset Martingale on win
                
                # Check if we need cooldown after win streak
                if self.consecutive_wins >= self.settings.strategy_switch_wins + 1:
                    self.in_cooldown = True
                    self.cooldown_trades_left = self.settings.cool_down_after_loss
                    logger.info(f"Entering cooldown after {self.consecutive_wins} consecutive wins")
            else:
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                
                # Update Martingale step
                if self.settings.enable_martingale:
                    self.martingale_step = min(
                        self.martingale_step + 1, 
                        self.settings.max_martingale_steps
                    )
            
            # Update trade count for strategy evaluation
            self.trades_since_evaluation += 1
            
            # Add to trade history
            self.trade_history.append(trade_result)
            
            # Save to database
            self._save_trade_to_database(trade_result)
            
            logger.info(f"Trade processed: {trade_result.success} | P&L: ${trade_result.profit_loss:.2f} | Daily P&L: ${self.daily_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error processing trade result: {str(e)}")
    
    def _check_strategy_switching(self):
        """Check if strategy switching is needed"""
        try:
            # Switch after consecutive wins
            if self.consecutive_wins >= self.settings.strategy_switch_wins:
                self.current_mode = TradingMode.MODE_C
                logger.info(f"Switched to MODE_C after {self.consecutive_wins} wins")
                return
            
            # Switch after consecutive losses
            if self.consecutive_losses >= self.settings.strategy_switch_losses:
                self.current_mode = TradingMode.MODE_B
                logger.info(f"Switched to MODE_B after {self.consecutive_losses} losses")
                return
            
            # Re-evaluate best strategy every N trades
            if self.trades_since_evaluation >= self.settings.reevaluate_trades:
                best_strategy = self._find_best_performing_strategy()
                if best_strategy != self.current_mode:
                    self.current_mode = best_strategy
                    logger.info(f"Switched to {best_strategy.value} based on performance evaluation")
                
                self.trades_since_evaluation = 0
                
        except Exception as e:
            logger.error(f"Error checking strategy switching: {str(e)}")
    
    def _find_best_performing_strategy(self) -> TradingMode:
        """Find the best performing strategy based on recent results"""
        try:
            strategy_scores = {}
            
            for strategy in TradingMode:
                performance = self.strategy_performance[strategy]
                total_trades = performance['wins'] + performance['losses']
                
                if total_trades == 0:
                    strategy_scores[strategy] = 0.5
                else:
                    win_rate = performance['wins'] / total_trades
                    avg_profit = performance['profit'] / total_trades if total_trades > 0 else 0
                    
                    # Combined score: win rate + profit factor
                    score = win_rate * 0.7 + (avg_profit > 0) * 0.3
                    strategy_scores[strategy] = score
            
            best_strategy = max(strategy_scores, key=strategy_scores.get)
            
            logger.info(f"Strategy scores: {strategy_scores}")
            
            return best_strategy
            
        except Exception as e:
            logger.error(f"Error finding best strategy: {str(e)}")
            return TradingMode.MODE_A
    
    def _update_strategy_performance(self, trade_result: TradeResult):
        """Update strategy performance tracking"""
        try:
            strategy = trade_result.strategy_used
            
            if trade_result.success:
                self.strategy_performance[strategy]['wins'] += 1
            else:
                self.strategy_performance[strategy]['losses'] += 1
            
            self.strategy_performance[strategy]['profit'] += trade_result.profit_loss
            
        except Exception as e:
            logger.error(f"Error updating strategy performance: {str(e)}")
    
    def _save_ml_training_data(self, market_data: Dict, trade_result: TradeResult):
        """Save training data for ML model improvement"""
        try:
            self.ml_strategy_manager.save_training_data(
                trade_result.contract_type,
                market_data,
                trade_result.success,
                trade_result.profit_loss
            )
            
            # Trigger auto-training periodically
            if self.daily_trades % 10 == 0:  # Every 10 trades
                self.ml_strategy_manager.train_models(trade_result.contract_type)
                
                # Get ChatGPT suggestions for improvement
                self.ml_strategy_manager.auto_train_with_chatgpt(trade_result.contract_type)
            
        except Exception as e:
            logger.error(f"Error saving ML training data: {str(e)}")
    
    def get_contract_performance_summary(self) -> Dict:
        """Get performance summary for all contract types"""
        try:
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT contract_type, 
                       COUNT(*) as total_trades,
                       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as wins,
                       SUM(profit_loss) as total_profit,
                       AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as win_rate
                FROM trade_results 
                WHERE user_id = ? 
                GROUP BY contract_type
            ''', (self.user_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            performance = {}
            for row in rows:
                performance[row[0]] = {
                    'total_trades': row[1],
                    'wins': row[2],
                    'total_profit': row[3],
                    'win_rate': row[4] or 0.0
                }
            
            return performance
            
        except Exception as e:
            logger.error(f"Error getting contract performance: {str(e)}")
            return {}
    
    def auto_retrain_ml_models(self):
        """Automatically retrain ML models based on performance"""
        try:
            # Check if enough new data for retraining
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT contract_type, COUNT(*) as count
                FROM ml_training_data 
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY contract_type
            ''')
            
            recent_data = dict(cursor.fetchall())
            conn.close()
            
            # Retrain models with sufficient new data
            for contract_type_str, count in recent_data.items():
                if count >= 50:  # Minimum samples for retraining
                    try:
                        contract_type = ContractType(contract_type_str)
                        logger.info(f"Auto-retraining {contract_type.value} model with {count} new samples")
                        
                        # Use ChatGPT to optimize training
                        self.ml_strategy_manager.auto_train_with_chatgpt(contract_type)
                        
                        # Regular model training
                        self.ml_strategy_manager.train_models(contract_type, force_retrain=True)
                        
                    except Exception as e:
                        logger.error(f"Error retraining {contract_type_str}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Auto-retrain error: {str(e)}")
    
    def optimize_settings_with_ai(self):
        """Use AI to optimize bot settings based on performance"""
        try:
            # Get recent performance data
            performance = self.get_contract_performance_summary()
            
            # Calculate overall metrics
            total_trades = sum(p['total_trades'] for p in performance.values())
            total_profit = sum(p['total_profit'] for p in performance.values())
            overall_win_rate = sum(p['winning_trades'] for p in performance.values()) / max(1, total_trades)
            
            # Create optimization prompt for ChatGPT
            prompt = f"""
            Analyze this AI trading bot performance and suggest optimizations:
            
            Current Settings:
            - Daily Stop Loss: {self.settings.daily_stop_loss_percent}%
            - Daily Target: {self.settings.daily_target_percent}%
            - Base Stake: {self.settings.base_stake_percent}%
            - Max Stake: {self.settings.max_stake_percent}%
            - Martingale Enabled: {self.settings.enable_martingale}
            - Martingale Multiplier: {self.settings.martingale_multiplier}
            - Max Martingale Steps: {self.settings.max_martingale_steps}
            
            Performance Metrics:
            - Total Trades: {total_trades}
            - Overall Win Rate: {overall_win_rate:.3f}
            - Total Profit: {total_profit:.2f}
            - Current Balance: {self.account_balance:.2f}
            - Daily P&L: {self.daily_pnl:.2f}
            
            Contract Performance:
            {json.dumps(performance, indent=2)}
            
            Suggest optimized settings for:
            1. Risk management (stop loss, targets)
            2. Position sizing (stake percentages)
            3. Martingale parameters
            4. Any other improvements
            
            Format as JSON with exact parameter names.
            """
            
            response = self.ml_strategy_manager.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert algorithmic trading optimization specialist."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.2
            )
            
            suggestions = response.choices[0].message.content
            logger.info(f"AI optimization suggestions: {suggestions}")
            
            # Apply conservative optimizations (would implement parsing logic)
            self._apply_conservative_optimizations(suggestions)
            
        except Exception as e:
            logger.error(f"AI optimization error: {str(e)}")
    
    def _apply_conservative_optimizations(self, suggestions: str):
        """Apply conservative optimizations to bot settings"""
        try:
            # Simple rule-based optimizations based on performance
            overall_profit_ratio = self.daily_pnl / max(1.0, self.starting_balance)
            
            # If losing money, reduce risk
            if overall_profit_ratio < -0.03:  # Down more than 3%
                self.settings.base_stake_percent = max(1.0, self.settings.base_stake_percent * 0.8)
                self.settings.max_stake_percent = max(5.0, self.settings.max_stake_percent * 0.9)
                self.settings.martingale_multiplier = max(1.1, self.settings.martingale_multiplier * 0.9)
                logger.info("Applied risk reduction optimizations")
            
            # If very profitable, slightly increase aggression
            elif overall_profit_ratio > 0.1:  # Up more than 10%
                self.settings.base_stake_percent = min(5.0, self.settings.base_stake_percent * 1.1)
                self.settings.daily_target_percent = min(30.0, self.settings.daily_target_percent * 1.2)
                logger.info("Applied profit optimization")
            
        except Exception as e:
            logger.error(f"Error applying optimizations: {str(e)}")
    
    def get_advanced_bot_status(self) -> Dict:
        """Get comprehensive bot status including ML and AI insights"""
        base_status = self.get_bot_status()
        
        try:
            # Add ML model performance
            ml_performance = self.ml_strategy_manager.get_model_performance()
            
            # Add contract performance
            contract_performance = self.get_contract_performance_summary()
            
            # Calculate advanced metrics
            total_trades = sum(p['total_trades'] for p in contract_performance.values())
            if total_trades > 0:
                sharpe_ratio = self._calculate_sharpe_ratio()
                max_drawdown = self._calculate_max_drawdown()
                profit_factor = self._calculate_profit_factor()
            else:
                sharpe_ratio = 0
                max_drawdown = 0
                profit_factor = 0
            
            advanced_status = {
                **base_status,
                'ml_model_performance': ml_performance,
                'contract_performance': contract_performance,
                'advanced_metrics': {
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'profit_factor': profit_factor,
                    'total_contract_trades': total_trades,
                    'avg_trade_duration': self._calculate_avg_trade_duration(),
                    'best_performing_contract': self._get_best_contract_type(),
                    'worst_performing_contract': self._get_worst_contract_type()
                },
                'ai_insights': {
                    'recommended_optimization': 'Risk reduction' if self.daily_pnl < 0 else 'Continue current strategy',
                    'next_retrain_due': self._get_next_retrain_time(),
                    'model_confidence_avg': self._calculate_avg_model_confidence()
                }
            }
            
            return advanced_status
            
        except Exception as e:
            logger.error(f"Error getting advanced status: {str(e)}")
            return base_status
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of trading performance"""
        try:
            if len(self.trade_history) < 10:
                return 0.0
            
            returns = [trade.profit_loss / trade.stake_amount for trade in self.trade_history]
            if not returns:
                return 0.0
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            # Annualized Sharpe ratio (assuming 252 trading days)
            sharpe = (mean_return / std_return) * np.sqrt(252)
            return round(sharpe, 3)
            
        except Exception as e:
            logger.error(f"Sharpe ratio calculation error: {str(e)}")
            return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        try:
            if not self.trade_history:
                return 0.0
            
            cumulative_pnl = []
            running_total = 0
            
            for trade in self.trade_history:
                running_total += trade.profit_loss
                cumulative_pnl.append(running_total)
            
            peak = cumulative_pnl[0]
            max_dd = 0
            
            for pnl in cumulative_pnl:
                if pnl > peak:
                    peak = pnl
                else:
                    drawdown = (peak - pnl) / max(1.0, abs(peak))
                    max_dd = max(max_dd, drawdown)
            
            return round(max_dd * 100, 2)  # Return as percentage
            
        except Exception as e:
            logger.error(f"Max drawdown calculation error: {str(e)}")
            return 0.0
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        try:
            if not self.trade_history:
                return 0.0
            
            gross_profit = sum(trade.profit_loss for trade in self.trade_history if trade.profit_loss > 0)
            gross_loss = abs(sum(trade.profit_loss for trade in self.trade_history if trade.profit_loss < 0))
            
            if gross_loss == 0:
                return float('inf') if gross_profit > 0 else 0.0
            
            return round(gross_profit / gross_loss, 3)
            
        except Exception as e:
            logger.error(f"Profit factor calculation error: {str(e)}")
            return 0.0
    
    def _calculate_avg_trade_duration(self) -> float:
        """Calculate average trade duration in minutes"""
        try:
            if not self.trade_history:
                return 0.0
            
            durations = [trade.duration for trade in self.trade_history]
            return round(np.mean(durations) / 60, 2)  # Convert to minutes
            
        except Exception as e:
            return 0.0
    
    def _get_best_contract_type(self) -> str:
        """Get best performing contract type"""
        try:
            performance = self.get_contract_performance_summary()
            if not performance:
                return "None"
            
            best = max(performance.items(), key=lambda x: x[1]['total_profit'])
            return best[0]
            
        except Exception as e:
            return "Unknown"
    
    def _get_worst_contract_type(self) -> str:
        """Get worst performing contract type"""
        try:
            performance = self.get_contract_performance_summary()
            if not performance:
                return "None"
            
            worst = min(performance.items(), key=lambda x: x[1]['total_profit'])
            return worst[0]
            
        except Exception as e:
            return "Unknown"
    
    def _get_next_retrain_time(self) -> str:
        """Get next scheduled ML model retrain time"""
        try:
            # Check last training time from ML manager
            last_train = None
            for contract_type in ContractType:
                perf = self.ml_strategy_manager.model_performance.get(contract_type, {})
                train_time = perf.get('last_trained')
                if train_time:
                    if last_train is None or train_time > last_train:
                        last_train = train_time
            
            if last_train:
                next_train = last_train + timedelta(hours=24)  # Retrain every 24 hours
                return next_train.strftime('%Y-%m-%d %H:%M UTC')
            else:
                return "Pending initial training"
                
        except Exception as e:
            return "Unknown"
    
    def _calculate_avg_model_confidence(self) -> float:
        """Calculate average ML model confidence"""
        try:
            if not self.trade_history:
                return 0.0
            
            # Would extract from trade metadata in real implementation
            return 0.75  # Placeholder
            
        except Exception as e:
            return 0.0
    
    def get_bot_status(self) -> Dict:
        """Get current bot status and statistics"""
        return {
            'is_active': self.is_active,
            'current_mode': self.current_mode.value,
            'account_balance': self.account_balance,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'win_rate': self._calculate_win_rate(),
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'in_cooldown': self.in_cooldown,
            'cooldown_trades_left': self.cooldown_trades_left,
            'martingale_step': self.martingale_step,
            'strategy_performance': self.strategy_performance,
            'daily_target': self.daily_target,
            'max_daily_loss': self.max_daily_loss,
            'safety_limits_reached': self._check_safety_limits()
        }
    
    def update_settings(self, new_settings: Dict):
        """Update bot settings"""
        for key, value in new_settings.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
        
        # Recalculate limits if balance percentages changed
        self.max_daily_loss = self.starting_balance * (self.settings.daily_stop_loss_percent / 100)
        self.daily_target = self.starting_balance * (self.settings.daily_target_percent / 100)
    
    def _save_trade_to_database(self, trade_result: TradeResult):
        """Save trade result to database"""
        try:
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trade_results (
                    user_id, contract_id, contract_type, entry_price, exit_price,
                    stake_amount, profit_loss, duration, strategy_used, success,
                    ml_prediction_confidence, market_conditions_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.user_id,
                trade_result.contract_id,
                trade_result.contract_type.value,
                trade_result.entry_price,
                trade_result.exit_price,
                trade_result.stake_amount,
                trade_result.profit_loss,
                trade_result.duration,
                trade_result.strategy_used.value,
                trade_result.success,
                0.0,  # ML confidence - to be filled by signal
                json.dumps({})  # Market conditions
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving trade to database: {str(e)}")
    
    def _start_trading_session(self):
        """Start a new trading session in database"""
        try:
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trading_sessions (
                    user_id, starting_balance, strategy_used, account_type, settings_json
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                self.user_id,
                self.starting_balance,
                self.current_mode.value,
                self.account_type,
                json.dumps(self.settings.__dict__)
            ))
            
            self.session_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error starting trading session: {str(e)}")
    
    def _end_trading_session(self):
        """End the current trading session"""
        try:
            if not hasattr(self, 'session_id'):
                return
            
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            
            win_rate = self._calculate_win_rate()
            
            cursor.execute('''
                UPDATE trading_sessions 
                SET session_end = CURRENT_TIMESTAMP,
                    ending_balance = ?,
                    total_pnl = ?,
                    total_trades = ?,
                    win_rate = ?
                WHERE id = ?
            ''', (
                self.account_balance,
                self.daily_pnl,
                self.daily_trades,
                win_rate,
                self.session_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error ending trading session: {str(e)}")
    
    def _calculate_win_rate(self) -> float:
        """Calculate current session win rate"""
        if len(self.trade_history) == 0:
            return 0.0
        
        wins = sum(1 for trade in self.trade_history if trade.success)
        return wins / len(self.trade_history)
    
    def get_status(self) -> Dict:
        """Get current bot status"""
        return {
            'is_active': self.is_active,
            'is_trading': self.is_trading,
            'current_mode': self.current_mode.value,
            'account_balance': self.account_balance,
            'starting_balance': self.starting_balance,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'martingale_step': self.martingale_step,
            'in_cooldown': self.in_cooldown,
            'cooldown_trades_left': self.cooldown_trades_left,
            'trades_since_evaluation': self.trades_since_evaluation,
            'current_trades': len(self.current_trades),
            'win_rate': self._calculate_win_rate(),
            'strategy_performance': {
                mode.value: {
                    'wins': perf['wins'],
                    'losses': perf['losses'],
                    'profit': perf['profit'],
                    'win_rate': perf['wins'] / (perf['wins'] + perf['losses']) if (perf['wins'] + perf['losses']) > 0 else 0
                }
                for mode, perf in self.strategy_performance.items()
            }
        }
    
    def update_settings(self, new_settings: Dict):
        """Update bot settings"""
        try:
            for key, value in new_settings.items():
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)
            
            logger.info(f"Settings updated: {new_settings}")
            
        except Exception as e:
            logger.error(f"Error updating settings: {str(e)}")
    
    def get_recent_trades(self, limit: int = 50) -> List[Dict]:
        """Get recent trades"""
        try:
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT contract_id, contract_type, entry_price, exit_price, 
                       stake_amount, profit_loss, duration, strategy_used, 
                       timestamp, success
                FROM trade_results 
                WHERE user_id = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (self.user_id, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            trades = []
            for row in rows:
                trades.append({
                    'contract_id': row[0],
                    'contract_type': row[1],
                    'entry_price': row[2],
                    'exit_price': row[3],
                    'stake_amount': row[4],
                    'profit_loss': row[5],
                    'duration': row[6],
                    'strategy_used': row[7],
                    'timestamp': row[8],
                    'success': bool(row[9])
                })
            
            return trades
            
        except Exception as e:
            logger.error(f"Error getting recent trades: {str(e)}")
            return []
