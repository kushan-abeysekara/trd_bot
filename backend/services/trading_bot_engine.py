import asyncio
import json
import logging
import random
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
        """Main trading loop for AI-powered trading bot"""
        logger.info("AI Trading Bot loop started")
        
        # Initialize trading stats
        trade_interval = 5  # Start with 5 second intervals
        last_market_update = datetime.now()
        active_predictions = {}
        last_strategy_optimization = datetime.now()
        
        while self.is_active and not self.stop_event.is_set():
            try:
                # Check safety limits
                if self._check_safety_limits():
                    logger.info("Safety limits triggered, stopping trading")
                    self.is_active = False
                    break
                
                # Check if in cooldown
                if self.in_cooldown:
                    if self.cooldown_trades_left > 0:
                        self.cooldown_trades_left -= 1
                        logger.info(f"In cooldown, skipping signal. {self.cooldown_trades_left} trades left")
                        self.stop_event.wait(5)  # Wait 5 seconds
                        continue
                    else:
                        self.in_cooldown = False
                        logger.info("Cooldown period ended")
                
                # Update account balance
                self._update_account_balance()
                
                # Real-time market data analysis (multi-timeframe)
                market_data = self._get_enhanced_market_data()
                if not market_data:
                    self.stop_event.wait(3)
                    continue
                
                # Update trading status for frontend display
                self._update_live_trading_status(market_data)
                
                # Check for completed trades and process results
                self._check_and_process_completed_trades()
                
                # Select best contract type and strategy based on market conditions
                contract_type, strategy_confidence = self._select_best_contract_type(market_data)
                
                # Dynamic interval adjustment based on market volatility
                trade_interval = self._calculate_optimal_trading_interval(market_data)
                
                # Get trading signal using ML with predictions
                signal = self.ml_strategy_manager.get_trading_signal_with_predictions(
                    contract_type, 
                    self.current_mode, 
                    market_data,
                    include_future_predictions=True
                )
                
                # Log ML predictions for frontend display
                self._log_ml_predictions(signal, contract_type)
                
                # Check if signal meets confidence threshold and trade filtering criteria
                if (signal and 
                    signal['confidence'] > 0.65 and  # Higher confidence requirement
                    self._validate_trading_signal(signal, market_data)):
                    
                    # Calculate optimal stake amount using advanced money management
                    stake_amount = self._calculate_stake_amount(contract_specific=contract_type)
                    
                    # Execute trade with ML-enhanced parameters
                    trade_result = self._execute_trade_sync(
                        contract_type, 
                        signal, 
                        stake_amount, 
                        market_data
                    )
                    
                    if trade_result:
                        # Process trade result and update ML models
                        self._process_trade_result(trade_result)
                        
                        # Update strategy performance metrics
                        self._update_strategy_performance(trade_result)
                        
                        # Check for strategy switching based on performance
                        self._check_strategy_switching()
                        
                        # Save ML training data for self-learning
                        self._save_ml_training_data(market_data, trade_result)
                        
                        # Auto-optimize trading settings periodically
                        if (datetime.now() - last_strategy_optimization).total_seconds() > 3600:  # Every hour
                            self.optimize_settings_with_ai()
                            last_strategy_optimization = datetime.now()
                
                # Run ML self-improvement if needed
                if len(self.trade_history) % 5 == 0 and len(self.trade_history) > 0:
                    self._run_ml_self_improvement()
                
                # Wait before next signal - adaptive based on market conditions
                wait_time = max(1, min(10, trade_interval))  # Between 1-10 seconds
                self.stop_event.wait(wait_time)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                self.stop_event.wait(5)  # Wait 5 seconds on error
        
        logger.info("AI Trading Bot loop ended")
    
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
    
    def _select_best_contract_type(self, market_data: Dict) -> Tuple[ContractType, float]:
        """Select best contract type based on market conditions and ML analysis"""
        try:
            contract_scores = {}
            
            for contract_type in ContractType:
                # Get ML prediction for this contract type
                signal = self.ml_strategy_manager.get_trading_signal_with_predictions(
                    contract_type, self.current_mode, market_data
                )
                
                if signal and signal.get('confidence', 0) > 0.5:
                    # Calculate contract score based on:
                    # 1. ML confidence
                    # 2. Market conditions suitability
                    # 3. Historical performance
                    base_score = signal['confidence']
                    
                    # Market conditions bonus
                    volatility = market_data.get('volatility', 0.01)
                    trend_strength = abs(market_data.get('trend', 0))
                    
                    if contract_type == ContractType.DIGITS:
                        # Digits work better in low volatility
                        market_bonus = 0.1 if volatility < 0.015 else -0.05
                    elif contract_type == ContractType.RISE_FALL:
                        # Rise/Fall works better with clear trends
                        market_bonus = 0.1 if trend_strength > 0.001 else -0.02
                    elif contract_type == ContractType.TOUCH_NO_TOUCH:
                        # Touch/No Touch works better in high volatility
                        market_bonus = 0.1 if volatility > 0.02 else -0.03
                    elif contract_type == ContractType.IN_OUT:
                        # In/Out works better in sideways markets
                        market_bonus = 0.1 if volatility > 0.01 and trend_strength < 0.001 else -0.02
                    else:
                        market_bonus = 0
                    
                    # Historical performance bonus
                    performance = self._get_contract_performance(contract_type)
                    performance_bonus = min(0.1, performance.get('win_rate', 0.5) - 0.5)
                    
                    total_score = base_score + market_bonus + performance_bonus
                    contract_scores[contract_type] = max(0.1, min(0.95, total_score))
                else:
                    contract_scores[contract_type] = 0.1
            
            # Select best contract type
            best_contract = max(contract_scores, key=contract_scores.get)
            best_score = contract_scores[best_contract]
            
            logger.info(f"Contract selection scores: {[(ct.value, f'{score:.3f}') for ct, score in contract_scores.items()]}")
            logger.info(f"Selected: {best_contract.value} with score {best_score:.3f}")
            
            return best_contract, best_score
            
        except Exception as e:
            logger.error(f"Error selecting contract type: {str(e)}")
            return ContractType.RISE_FALL, 0.6

    def _get_contract_performance(self, contract_type: ContractType) -> Dict:
        """Get historical performance for specific contract type"""
        try:
            # Get recent trades for this contract type
            recent_trades = [t for t in self.trade_history[-50:] if hasattr(t, 'contract_type') and t.contract_type == contract_type]
            
            if not recent_trades:
                return {'win_rate': 0.5, 'avg_profit': 0, 'total_trades': 0}
            
            wins = sum(1 for t in recent_trades if t.success)
            total_profit = sum(t.profit_loss for t in recent_trades)
            
            return {
                'win_rate': wins / len(recent_trades),
                'avg_profit': total_profit / len(recent_trades),
                'total_trades': len(recent_trades),
                'recent_performance': wins / min(10, len(recent_trades)) if len(recent_trades) >= 5 else 0.5
            }
            
        except Exception as e:
            logger.error(f"Error getting contract performance: {str(e)}")
            return {'win_rate': 0.5, 'avg_profit': 0, 'total_trades': 0}

    def _execute_trade_sync(self, contract_type: ContractType, signal: Dict, stake_amount: float, market_data: Dict) -> Optional[TradeResult]:
        """Execute trade synchronously with enhanced contract-specific logic"""
        try:
            current_price = market_data.get('current_price', 1.0)
            
            # Contract-specific trade parameters
            trade_params = self._get_contract_specific_params(contract_type, signal, market_data)
            
            # Execute trade via Deriv API
            trade_response = self._execute_deriv_trade(contract_type, trade_params, stake_amount, current_price)
            
            if trade_response and trade_response.get('success'):
                # Create trade result
                trade_result = TradeResult(
                    contract_id=trade_response['contract_id'],
                    contract_type=contract_type,
                    entry_price=current_price,
                    exit_price=0,  # Will be updated when trade completes
                    stake_amount=stake_amount,
                    profit_loss=0,  # Will be updated when trade completes
                    duration=trade_params.get('duration', 300),
                    strategy_used=self.current_mode,
                    timestamp=datetime.now(),
                    success=False  # Will be updated when trade completes
                )
                
                # Add to current trades for monitoring
                self.current_trades[trade_response['contract_id']] = {
                    'trade_result': trade_result,
                    'signal': signal,
                    'market_data': market_data,
                    'start_time': datetime.now(),
                    'contract_type': contract_type.value,
                    'action': signal.get('action', 'call'),
                    'confidence': signal.get('confidence', 0.5)
                }
                
                logger.info(f"Trade executed: {contract_type.value} - {signal.get('action')} - ${stake_amount:.2f}")
                return trade_result
            else:
                logger.error(f"Trade execution failed: {trade_response}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return None

    def _get_contract_specific_params(self, contract_type: ContractType, signal: Dict, market_data: Dict) -> Dict:
        """Get contract-specific parameters for trade execution"""
        current_price = market_data.get('current_price', 1.0)
        volatility = market_data.get('volatility', 0.01)
        
        if contract_type == ContractType.RISE_FALL:
            return {
                'contract_type': 'CALL' if signal.get('action') == 'call' else 'PUT',
                'duration': 300,  # 5 minutes
                'basis': 'stake'
            }
            
        elif contract_type == ContractType.TOUCH_NO_TOUCH:
            # Calculate barrier based on volatility
            barrier_distance = volatility * 20 * current_price
            barrier = current_price + barrier_distance if signal.get('action') == 'touch' else current_price - barrier_distance
            
            return {
                'contract_type': 'ONETOUCH' if signal.get('action') == 'touch' else 'NOTOUCH',
                'duration': 600,  # 10 minutes
                'barrier': barrier,
                'basis': 'stake'
            }
            
        elif contract_type == ContractType.IN_OUT:
            # Calculate boundaries
            boundary_distance = volatility * 15 * current_price
            
            return {
                'contract_type': 'EXPIRYRANGE' if signal.get('action') == 'in' else 'EXPIRYMISS',
                'duration': 600,
                'barrier': current_price + boundary_distance,
                'barrier2': current_price - boundary_distance,
                'basis': 'stake'
            }
            
        elif contract_type == ContractType.DIGITS:
            # Predict next digit
            predicted_digit = signal.get('predicted_digit', 5)
            
            return {
                'contract_type': 'DIGITMATCH',
                'duration': 30,  # 30 seconds
                'digit': predicted_digit,
                'basis': 'stake'
            }
            
        elif contract_type == ContractType.HIGH_LOW_TICKS:
            return {
                'contract_type': 'TICKHIGH' if signal.get('action') == 'high' else 'TICKLOW',
                'duration': 50,  # 5 ticks
                'basis': 'stake'
            }
            
        else:
            # Default parameters
            return {
                'contract_type': 'CALL' if signal.get('action') == 'call' else 'PUT',
                'duration': 300,
                'basis': 'stake'
            }

    def _execute_deriv_trade(self, contract_type: ContractType, trade_params: Dict, stake_amount: float, current_price: float) -> Dict:
        """Execute trade via Deriv API"""
        try:
            # In demo mode, simulate trade execution
            if self.account_type == 'demo':
                return self._simulate_trade_execution(contract_type, trade_params, stake_amount, current_price)
            
            # Real trade execution would go here
            trade_request = {
                'buy': 1,
                'price': stake_amount,
                'parameters': trade_params
            }
            
            response = self.deriv_service.place_trade(self.api_token, trade_request)
            
            if response and response.get('buy'):
                return {
                    'success': True,
                    'contract_id': response['buy']['contract_id'],
                    'buy_price': response['buy']['buy_price'],
                    'payout': response['buy']['payout']
                }
            else:
                return {'success': False, 'error': 'Trade placement failed'}
                
        except Exception as e:
            logger.error(f"Error executing Deriv trade: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _simulate_trade_execution(self, contract_type: ContractType, trade_params: Dict, stake_amount: float, current_price: float) -> Dict:
        """Simulate trade execution for demo/testing"""
        # Generate realistic contract ID
        contract_id = f"CONTRACT_{int(datetime.now().timestamp())}_{contract_type.value[:4].upper()}"
        
        # Calculate payout based on contract type
        if contract_type == ContractType.DIGITS:
            payout = stake_amount * 9.5  # Digits typically have high payout
        elif contract_type == ContractType.TOUCH_NO_TOUCH:
            payout = stake_amount * 3.5
        elif contract_type == ContractType.IN_OUT:
            payout = stake_amount * 2.8
        else:
            payout = stake_amount * 1.85  # Standard Rise/Fall payout
        
        return {
            'success': True,
            'contract_id': contract_id,
            'buy_price': stake_amount,
            'payout': payout,
            'simulated': True
        }

    def _update_live_trading_status(self, market_data: Dict):
        """Update real-time trading status for frontend display"""
        try:
            # Update live status in database for frontend consumption
            live_status = {
                'timestamp': datetime.now().isoformat(),
                'is_active': self.is_active,
                'current_mode': self.current_mode.value,
                'account_balance': self.account_balance,
                'daily_pnl': self.daily_pnl,
                'daily_trades': self.daily_trades,
                'current_price': market_data.get('current_price', 0),
                'volatility': market_data.get('volatility', 0),
                'trend': market_data.get('trend', 0),
                'rsi': market_data.get('rsi', 50),
                'active_trades': len(self.current_trades),
                'consecutive_wins': self.consecutive_wins,
                'consecutive_losses': self.consecutive_losses,
                'martingale_step': self.martingale_step,
                'in_cooldown': self.in_cooldown,
                'strategy_performance': self._get_strategy_performance_summary()
            }
            
            # Store in database for frontend polling
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO live_trading_status (
                    user_id, status_json, timestamp
                ) VALUES (?, ?, ?)
            ''', (
                self.user_id,
                json.dumps(live_status),
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating live trading status: {str(e)}")

    def get_live_trading_status(self) -> Dict:
        """Get current live trading status"""
        try:
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT status_json FROM live_trading_status 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            ''', (self.user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return json.loads(result[0])
            else:
                return self._get_default_status()
                
        except Exception as e:
            logger.error(f"Error getting live trading status: {str(e)}")
            return self._get_default_status()

    def _get_default_status(self) -> Dict:
        """Get default status structure"""
        return {
            'timestamp': datetime.now().isoformat(),
            'is_active': False,
            'current_mode': TradingMode.MODE_A.value,
            'account_balance': 0.0,
            'daily_pnl': 0.0,
            'daily_trades': 0,
            'current_price': 0.0,
            'volatility': 0.0,
            'trend': 0.0,
            'rsi': 50,
            'active_trades': 0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'martingale_step': 0,
            'in_cooldown': False,
            'strategy_performance': {}
        }
