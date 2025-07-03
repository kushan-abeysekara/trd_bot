import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from threading import Thread, Event
import websocket
import ssl

from utils.deriv_service import DerivService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TechnicalIndicators:
    """Technical indicators data class"""
    rsi: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    ema_50: float = 0.0
    volatility: float = 0.0
    momentum: float = 0.0
    tick_velocity: float = 0.0
    williams_r: float = -50.0
    stochastic_k: float = 50.0
    stochastic_d: float = 50.0
    volume_price_trend: float = 0.0

@dataclass
class TradeSignal:
    """Trade signal data class"""
    strategy_name: str
    direction: str  # 'BUY_RISE' or 'BUY_FALL'
    confidence: float
    entry_price: float
    stake_amount: float
    contract_type: str = 'CALL'
    duration: int = 5  # seconds
    indicators: Optional[TechnicalIndicators] = None
    timestamp: datetime = None

@dataclass
class TradeResult:
    """Trade result data class"""
    trade_id: str
    strategy_name: str
    direction: str
    entry_price: float
    exit_price: float
    stake_amount: float
    profit_loss: float
    success: bool
    duration: int
    entry_time: datetime
    exit_time: datetime
    contract_id: Optional[str] = None

class TechnicalTradingStrategy:
    """Base class for technical trading strategies"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.win_count = 0
        self.loss_count = 0
        self.total_profit = 0.0
        self.last_signal_time = None
        
    def should_trade(self, indicators: TechnicalIndicators, price_data: List[float]) -> bool:
        """Override in subclasses"""
        return False
        
    def generate_signal(self, indicators: TechnicalIndicators, price_data: List[float], current_price: float) -> Optional[TradeSignal]:
        """Override in subclasses"""
        return None
        
    def get_win_rate(self) -> float:
        total_trades = self.win_count + self.loss_count
        return (self.win_count / total_trades * 100) if total_trades > 0 else 0.0

class AdaptiveMeanReversionStrategy(TechnicalTradingStrategy):
    """Strategy 1: Adaptive Mean Reversion Rebound"""
    
    def __init__(self):
        super().__init__(
            "Adaptive Mean Reversion Rebound",
            "Profit from short-term reversals during neutral RSI and moderate volatility"
        )
    
    def should_trade(self, indicators: TechnicalIndicators, price_data: List[float]) -> bool:
        # RSI between 48-52
        rsi_neutral = 48 <= indicators.rsi <= 52
        
        # Volatility 1-1.5%
        moderate_volatility = 1.0 <= indicators.volatility <= 1.5
        
        # MACD should be flat (-0.1 to +0.1)
        macd_flat = -0.1 <= indicators.macd_histogram <= 0.1
        
        # Momentum < ±0.2%
        low_momentum = abs(indicators.momentum) < 0.2
        
        return rsi_neutral and moderate_volatility and macd_flat and low_momentum
    
    def generate_signal(self, indicators: TechnicalIndicators, price_data: List[float], current_price: float) -> Optional[TradeSignal]:
        if not self.should_trade(indicators, price_data):
            return None
        
        # Check Bollinger Band touches
        if current_price <= indicators.bb_lower:
            # Price touched lower band - expect bounce up
            return TradeSignal(
                strategy_name=self.name,
                direction='BUY_RISE',
                confidence=0.75,
                entry_price=current_price,
                stake_amount=0.0,  # Will be calculated by bot
                duration=6,
                indicators=indicators,
                timestamp=datetime.utcnow()
            )
        elif current_price >= indicators.bb_upper:
            # Price touched upper band - expect bounce down
            return TradeSignal(
                strategy_name=self.name,
                direction='BUY_FALL',
                confidence=0.75,
                entry_price=current_price,
                stake_amount=0.0,
                duration=6,
                indicators=indicators,
                timestamp=datetime.utcnow()
            )
        
        return None

class RSIMomentumBreakoutStrategy(TechnicalTradingStrategy):
    """Strategy 2: RSI Momentum Breakout"""
    
    def __init__(self):
        super().__init__(
            "RSI Momentum Breakout",
            "Capitalizes on momentum breakouts with RSI confirmation"
        )
    
    def should_trade(self, indicators: TechnicalIndicators, price_data: List[float]) -> bool:
        # RSI breakout conditions
        rsi_breakout = indicators.rsi > 70 or indicators.rsi < 30
        
        # Strong momentum
        strong_momentum = abs(indicators.momentum) > 0.5
        
        # Positive MACD histogram
        macd_positive = indicators.macd_histogram > 0.05
        
        return rsi_breakout and strong_momentum and macd_positive
    
    def generate_signal(self, indicators: TechnicalIndicators, price_data: List[float], current_price: float) -> Optional[TradeSignal]:
        if not self.should_trade(indicators, price_data):
            return None
        
        if indicators.rsi > 70 and indicators.momentum > 0.5:
            return TradeSignal(
                strategy_name=self.name,
                direction='BUY_RISE',
                confidence=0.8,
                entry_price=current_price,
                stake_amount=0.0,
                duration=8,
                indicators=indicators,
                timestamp=datetime.utcnow()
            )
        elif indicators.rsi < 30 and indicators.momentum < -0.5:
            return TradeSignal(
                strategy_name=self.name,
                direction='BUY_FALL',
                confidence=0.8,
                entry_price=current_price,
                stake_amount=0.0,
                duration=8,
                indicators=indicators,
                timestamp=datetime.utcnow()
            )
        
        return None

class BollingerBandSqueezeStrategy(TechnicalTradingStrategy):
    """Strategy 3: Bollinger Band Squeeze"""
    
    def __init__(self):
        super().__init__(
            "Bollinger Band Squeeze",
            "Exploits volatility expansion after band compression"
        )
    
    def should_trade(self, indicators: TechnicalIndicators, price_data: List[float]) -> bool:
        # Low volatility (band squeeze)
        band_width = (indicators.bb_upper - indicators.bb_lower) / indicators.bb_middle
        squeeze_condition = band_width < 0.02
        
        # Price near middle band
        price_centered = abs(price_data[-1] - indicators.bb_middle) / indicators.bb_middle < 0.005
        
        return squeeze_condition and price_centered
    
    def generate_signal(self, indicators: TechnicalIndicators, price_data: List[float], current_price: float) -> Optional[TradeSignal]:
        if not self.should_trade(indicators, price_data):
            return None
        
        # Wait for breakout direction
        if indicators.momentum > 0.3:
            return TradeSignal(
                strategy_name=self.name,
                direction='BUY_RISE',
                confidence=0.85,
                entry_price=current_price,
                stake_amount=0.0,
                duration=10,
                indicators=indicators,
                timestamp=datetime.utcnow()
            )
        elif indicators.momentum < -0.3:
            return TradeSignal(
                strategy_name=self.name,
                direction='BUY_FALL',
                confidence=0.85,
                entry_price=current_price,
                stake_amount=0.0,
                duration=10,
                indicators=indicators,
                timestamp=datetime.utcnow()
            )
        
        return None

class TechnicalTradingEngine:
    """Main technical trading engine with 15 strategies"""
    
    def __init__(self, user_id: int, api_token: str, account_type: str = 'demo'):
        self.user_id = user_id
        self.api_token = api_token
        self.account_type = account_type
        self.deriv_service = DerivService()
        
        # Trading state
        self.is_active = False
        self.stop_event = Event()
        self.trading_thread = None
        
        # Initialize strategies
        self.strategies = {
            'adaptive_mean_reversion': AdaptiveMeanReversionStrategy(),
            'rsi_momentum_breakout': RSIMomentumBreakoutStrategy(),
            'bollinger_band_squeeze': BollingerBandSqueezeStrategy(),
            # Add more strategies here...
        }
        
        self.current_strategy = 'adaptive_mean_reversion'
        
        # Trading settings
        self.settings = {
            'base_stake_amount': 1.0,
            'max_stake_amount': 10.0,
            'daily_stop_loss': 100.0,
            'daily_target': 200.0,
            'auto_stake': True,
            'risk_per_trade': 2.0,
            'max_trades_per_hour': 20,
            'cooldown_after_loss': 3,
            'strategy_rotation': True
        }
        
        # Performance tracking
        self.account_balance = 0.0
        self.starting_balance = 0.0
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.open_trades = []
        self.trade_history = []
        
        # Technical indicators
        self.current_indicators = TechnicalIndicators()
        self.price_data = []
        self.tick_data = []
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize trading database tables"""
        try:
            conn = sqlite3.connect('technical_trading.db')
            cursor = conn.cursor()
            
            # Create technical trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS technical_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    trade_id TEXT UNIQUE NOT NULL,
                    strategy_name TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    stake_amount REAL NOT NULL,
                    profit_loss REAL DEFAULT 0,
                    success BOOLEAN DEFAULT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP,
                    duration INTEGER,
                    contract_id TEXT,
                    indicators TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create technical strategies performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS technical_strategy_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    strategy_name TEXT NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    total_profit REAL DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    last_used TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, strategy_name)
                )
            ''')
            
            # Create technical sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS technical_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_start TIMESTAMP NOT NULL,
                    session_end TIMESTAMP,
                    starting_balance REAL NOT NULL,
                    ending_balance REAL,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    net_pnl REAL DEFAULT 0,
                    strategies_used TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Technical trading database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize technical trading database: {str(e)}")
    
    def calculate_technical_indicators(self, price_data: List[float]) -> TechnicalIndicators:
        """Calculate all technical indicators"""
        if len(price_data) < 50:
            return self.current_indicators
        
        # Convert to pandas for easier calculation
        df = pd.DataFrame({'price': price_data})
        
        # RSI calculation
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD calculation
        ema_12 = df['price'].ewm(span=12).mean()
        ema_26 = df['price'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9).mean()
        macd_histogram = macd_line - macd_signal
        
        # Bollinger Bands
        sma_20 = df['price'].rolling(window=20).mean()
        std_20 = df['price'].rolling(window=20).std()
        bb_upper = sma_20 + (std_20 * 2)
        bb_lower = sma_20 - (std_20 * 2)
        
        # EMA calculations
        ema_50 = df['price'].ewm(span=50).mean()
        
        # Volatility (20-period standard deviation)
        volatility = df['price'].rolling(window=20).std() / df['price'].rolling(window=20).mean() * 100
        
        # Momentum (rate of change)
        momentum = (df['price'] - df['price'].shift(10)) / df['price'].shift(10) * 100
        
        # Williams %R
        high_14 = df['price'].rolling(window=14).max()
        low_14 = df['price'].rolling(window=14).min()
        williams_r = -100 * (high_14 - df['price']) / (high_14 - low_14)
        
        # Stochastic Oscillator
        stoch_k = 100 * (df['price'] - low_14) / (high_14 - low_14)
        stoch_d = stoch_k.rolling(window=3).mean()
        
        # Get latest values
        latest_idx = len(df) - 1
        
        return TechnicalIndicators(
            rsi=float(rsi.iloc[latest_idx]) if not pd.isna(rsi.iloc[latest_idx]) else 50.0,
            macd=float(macd_line.iloc[latest_idx]) if not pd.isna(macd_line.iloc[latest_idx]) else 0.0,
            macd_signal=float(macd_signal.iloc[latest_idx]) if not pd.isna(macd_signal.iloc[latest_idx]) else 0.0,
            macd_histogram=float(macd_histogram.iloc[latest_idx]) if not pd.isna(macd_histogram.iloc[latest_idx]) else 0.0,
            bb_upper=float(bb_upper.iloc[latest_idx]) if not pd.isna(bb_upper.iloc[latest_idx]) else price_data[-1],
            bb_middle=float(sma_20.iloc[latest_idx]) if not pd.isna(sma_20.iloc[latest_idx]) else price_data[-1],
            bb_lower=float(bb_lower.iloc[latest_idx]) if not pd.isna(bb_lower.iloc[latest_idx]) else price_data[-1],
            ema_12=float(ema_12.iloc[latest_idx]) if not pd.isna(ema_12.iloc[latest_idx]) else price_data[-1],
            ema_26=float(ema_26.iloc[latest_idx]) if not pd.isna(ema_26.iloc[latest_idx]) else price_data[-1],
            ema_50=float(ema_50.iloc[latest_idx]) if not pd.isna(ema_50.iloc[latest_idx]) else price_data[-1],
            volatility=float(volatility.iloc[latest_idx]) if not pd.isna(volatility.iloc[latest_idx]) else 0.0,
            momentum=float(momentum.iloc[latest_idx]) if not pd.isna(momentum.iloc[latest_idx]) else 0.0,
            williams_r=float(williams_r.iloc[latest_idx]) if not pd.isna(williams_r.iloc[latest_idx]) else -50.0,
            stochastic_k=float(stoch_k.iloc[latest_idx]) if not pd.isna(stoch_k.iloc[latest_idx]) else 50.0,
            stochastic_d=float(stoch_d.iloc[latest_idx]) if not pd.isna(stoch_d.iloc[latest_idx]) else 50.0,
            tick_velocity=self.calculate_tick_velocity(),
            volume_price_trend=0.0
        )
    
    def calculate_tick_velocity(self) -> float:
        """Calculate tick velocity (price change per second)"""
        if len(self.tick_data) < 2:
            return 0.0
        
        recent_ticks = self.tick_data[-10:]  # Last 10 ticks
        if len(recent_ticks) < 2:
            return 0.0
        
        price_changes = []
        for i in range(1, len(recent_ticks)):
            price_change = abs(recent_ticks[i]['price'] - recent_ticks[i-1]['price'])
            time_diff = (recent_ticks[i]['timestamp'] - recent_ticks[i-1]['timestamp']).total_seconds()
            if time_diff > 0:
                price_changes.append(price_change / time_diff)
        
        return np.mean(price_changes) if price_changes else 0.0
    
    def calculate_stake_amount(self, signal: TradeSignal) -> float:
        """Calculate dynamic stake amount based on confidence and risk management"""
        if self.settings['auto_stake']:
            base_stake = self.account_balance * (self.settings['risk_per_trade'] / 100)
            confidence_multiplier = signal.confidence
            
            # Apply confidence adjustment
            stake = base_stake * confidence_multiplier
            
            # Ensure within limits
            stake = max(self.settings['base_stake_amount'], stake)
            stake = min(self.settings['max_stake_amount'], stake)
            
            return round(stake, 2)
        else:
            return self.settings['base_stake_amount']
    
    def start_trading(self) -> Dict:
        """Start the technical trading bot"""
        try:
            if self.is_active:
                return {'success': False, 'message': 'Trading bot is already active'}
            
            # Get account balance
            balance_result = self.deriv_service.get_real_time_balance(self.api_token)
            if not balance_result['success']:
                return {'success': False, 'message': f'Failed to get account balance: {balance_result["message"]}'}
            
            self.account_balance = balance_result['data']['balance']
            self.starting_balance = self.account_balance
            
            # Start trading session in database
            self._start_trading_session()
            
            # Start trading thread
            self.is_active = True
            self.stop_event.clear()
            self.trading_thread = Thread(target=self._trading_loop, daemon=True)
            self.trading_thread.start()
            
            logger.info(f"Technical trading bot started for user {self.user_id}")
            return {
                'success': True,
                'message': 'Technical trading bot started successfully',
                'account_balance': self.account_balance,
                'current_strategy': self.current_strategy
            }
            
        except Exception as e:
            logger.error(f"Failed to start technical trading bot: {str(e)}")
            return {'success': False, 'message': f'Failed to start trading bot: {str(e)}'}
    
    def stop_trading(self) -> Dict:
        """Stop the technical trading bot"""
        try:
            if not self.is_active:
                return {'success': False, 'message': 'Trading bot is not active'}
            
            self.is_active = False
            self.stop_event.set()
            
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=5)
            
            # End trading session in database
            self._end_trading_session()
            
            logger.info(f"Technical trading bot stopped for user {self.user_id}")
            return {
                'success': True,
                'message': 'Technical trading bot stopped successfully',
                'session_pnl': self.daily_pnl,
                'total_trades': self.total_trades
            }
            
        except Exception as e:
            logger.error(f"Failed to stop technical trading bot: {str(e)}")
            return {'success': False, 'message': f'Failed to stop trading bot: {str(e)}'}
    
    def _trading_loop(self):
        """Main trading loop"""
        logger.info("Technical trading loop started")
        
        while self.is_active and not self.stop_event.is_set():
            try:
                # Get market data
                self._update_market_data()
                
                # Calculate technical indicators
                if len(self.price_data) >= 50:
                    self.current_indicators = self.calculate_technical_indicators(self.price_data)
                
                # Check for trading signals
                signal = self._get_trading_signal()
                
                if signal:
                    self._execute_trade(signal)
                
                # Check stop conditions
                if self._should_stop_trading():
                    logger.info("Stop conditions met, stopping trading")
                    break
                
                # Sleep between iterations
                time.sleep(1)  # 1 second intervals for technical analysis
                
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(5)
        
        logger.info("Technical trading loop ended")
    
    def _update_market_data(self):
        """Update market data and price history"""
        try:
            # Get latest tick data (this would be implemented with WebSocket)
            # For now, simulate with a simple price update
            current_time = datetime.utcnow()
            
            # Add current price to history
            if hasattr(self, '_last_price_update'):
                time_diff = (current_time - self._last_price_update).total_seconds()
                if time_diff >= 1:  # Update every second
                    # Simulate price movement (replace with real data)
                    last_price = self.price_data[-1] if self.price_data else 100.0
                    new_price = last_price + np.random.normal(0, 0.01)
                    
                    self.price_data.append(new_price)
                    self.tick_data.append({
                        'price': new_price,
                        'timestamp': current_time
                    })
                    
                    # Keep only last 200 data points
                    if len(self.price_data) > 200:
                        self.price_data = self.price_data[-200:]
                    
                    if len(self.tick_data) > 100:
                        self.tick_data = self.tick_data[-100:]
                    
                    self._last_price_update = current_time
            else:
                # Initialize with first price
                self.price_data = [100.0]  # Starting price
                self.tick_data = [{'price': 100.0, 'timestamp': current_time}]
                self._last_price_update = current_time
                
        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")
    
    def _get_trading_signal(self) -> Optional[TradeSignal]:
        """Get trading signal from current strategy"""
        try:
            if len(self.price_data) < 50:
                return None
            
            current_strategy = self.strategies.get(self.current_strategy)
            if not current_strategy:
                return None
            
            current_price = self.price_data[-1]
            signal = current_strategy.generate_signal(
                self.current_indicators,
                self.price_data,
                current_price
            )
            
            if signal:
                signal.stake_amount = self.calculate_stake_amount(signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error getting trading signal: {str(e)}")
            return None
    
    def _execute_trade(self, signal: TradeSignal):
        """Execute a trade based on signal"""
        try:
            # Generate unique trade ID
            trade_id = f"tech_{int(time.time() * 1000)}"
            
            # Create trade record
            trade = {
                'trade_id': trade_id,
                'strategy_name': signal.strategy_name,
                'direction': signal.direction,
                'entry_price': signal.entry_price,
                'stake_amount': signal.stake_amount,
                'entry_time': datetime.utcnow(),
                'duration': signal.duration,
                'indicators': asdict(signal.indicators) if signal.indicators else {}
            }
            
            # Add to open trades
            self.open_trades.append(trade)
            
            # Save to database
            self._save_trade_to_db(trade)
            
            logger.info(f"Executed trade {trade_id}: {signal.strategy_name} {signal.direction} @ {signal.entry_price}")
            
            # Simulate trade completion after duration
            Thread(target=self._complete_trade, args=(trade,), daemon=True).start()
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
    
    def _complete_trade(self, trade: Dict):
        """Complete a trade after duration"""
        try:
            # Wait for trade duration
            time.sleep(trade['duration'])
            
            # Get exit price
            exit_price = self.price_data[-1] if self.price_data else trade['entry_price']
            
            # Calculate profit/loss
            if trade['direction'] == 'BUY_RISE':
                success = exit_price > trade['entry_price']
            else:  # BUY_FALL
                success = exit_price < trade['entry_price']
            
            # Calculate P&L (simplified - 80% payout for wins)
            if success:
                profit_loss = trade['stake_amount'] * 0.8
                self.winning_trades += 1
            else:
                profit_loss = -trade['stake_amount']
            
            # Update trade
            trade.update({
                'exit_price': exit_price,
                'exit_time': datetime.utcnow(),
                'profit_loss': profit_loss,
                'success': success
            })
            
            # Update performance
            self.total_trades += 1
            self.daily_pnl += profit_loss
            self.account_balance += profit_loss
            
            # Update strategy performance
            strategy = self.strategies.get(trade['strategy_name'].lower().replace(' ', '_'))
            if strategy:
                if success:
                    strategy.win_count += 1
                else:
                    strategy.loss_count += 1
                strategy.total_profit += profit_loss
            
            # Remove from open trades
            self.open_trades = [t for t in self.open_trades if t['trade_id'] != trade['trade_id']]
            
            # Add to history
            self.trade_history.append(trade)
            
            # Update database
            self._update_trade_in_db(trade)
            
            logger.info(f"Completed trade {trade['trade_id']}: {'WIN' if success else 'LOSS'} P&L: {profit_loss}")
            
        except Exception as e:
            logger.error(f"Error completing trade: {str(e)}")
    
    def _should_stop_trading(self) -> bool:
        """Check if trading should stop based on risk management"""
        # Daily stop loss
        if self.daily_pnl <= -self.settings['daily_stop_loss']:
            logger.info(f"Daily stop loss reached: {self.daily_pnl}")
            return True
        
        # Daily target
        if self.daily_pnl >= self.settings['daily_target']:
            logger.info(f"Daily target reached: {self.daily_pnl}")
            return True
        
        # Maximum trades per hour
        recent_trades = [t for t in self.trade_history 
                        if (datetime.utcnow() - t['entry_time']).total_seconds() < 3600]
        if len(recent_trades) >= self.settings['max_trades_per_hour']:
            logger.info("Maximum trades per hour reached")
            return True
        
        return False
    
    def _save_trade_to_db(self, trade: Dict):
        """Save trade to database"""
        try:
            conn = sqlite3.connect('technical_trading.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO technical_trades 
                (user_id, trade_id, strategy_name, direction, entry_price, 
                 stake_amount, entry_time, duration, indicators)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.user_id,
                trade['trade_id'],
                trade['strategy_name'],
                trade['direction'],
                trade['entry_price'],
                trade['stake_amount'],
                trade['entry_time'],
                trade['duration'],
                json.dumps(trade['indicators'])
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving trade to database: {str(e)}")
    
    def _update_trade_in_db(self, trade: Dict):
        """Update completed trade in database"""
        try:
            conn = sqlite3.connect('technical_trading.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE technical_trades 
                SET exit_price = ?, exit_time = ?, profit_loss = ?, success = ?
                WHERE trade_id = ?
            ''', (
                trade['exit_price'],
                trade['exit_time'],
                trade['profit_loss'],
                trade['success'],
                trade['trade_id']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating trade in database: {str(e)}")
    
    def _start_trading_session(self):
        """Start a new trading session in database"""
        try:
            conn = sqlite3.connect('technical_trading.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO technical_sessions 
                (user_id, session_start, starting_balance, strategies_used)
                VALUES (?, ?, ?, ?)
            ''', (
                self.user_id,
                datetime.utcnow(),
                self.starting_balance,
                json.dumps(list(self.strategies.keys()))
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error starting trading session: {str(e)}")
    
    def _end_trading_session(self):
        """End current trading session in database"""
        try:
            conn = sqlite3.connect('technical_trading.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE technical_sessions 
                SET session_end = ?, ending_balance = ?, total_trades = ?, 
                    winning_trades = ?, net_pnl = ?
                WHERE user_id = ? AND session_end IS NULL
                ORDER BY session_start DESC LIMIT 1
            ''', (
                datetime.utcnow(),
                self.account_balance,
                self.total_trades,
                self.winning_trades,
                self.daily_pnl,
                self.user_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error ending trading session: {str(e)}")
    
    def get_status(self) -> Dict:
        """Get current bot status"""
        return {
            'is_active': self.is_active,
            'current_strategy': self.current_strategy,
            'account_balance': self.account_balance,
            'daily_pnl': self.daily_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': (self.winning_trades / max(self.total_trades, 1)) * 100,
            'open_trades': len(self.open_trades),
            'open_trades_list': self.open_trades,
            'current_indicators': asdict(self.current_indicators),
            'strategies_performance': {
                name: {
                    'win_count': strategy.win_count,
                    'loss_count': strategy.loss_count,
                    'total_profit': strategy.total_profit,
                    'win_rate': strategy.get_win_rate()
                }
                for name, strategy in self.strategies.items()
            }
        }
    
    def get_trading_history(self, limit: int = 50) -> List[Dict]:
        """Get trading history"""
        try:
            conn = sqlite3.connect('technical_trading.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM technical_trades 
                WHERE user_id = ? 
                ORDER BY entry_time DESC 
                LIMIT ?
            ''', (self.user_id, limit))
            
            trades = cursor.fetchall()
            conn.close()
            
            # Convert to list of dictionaries
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, trade)) for trade in trades]
            
        except Exception as e:
            logger.error(f"Error getting trading history: {str(e)}")
            return []
    
    def update_settings(self, new_settings: Dict) -> Dict:
        """Update trading settings"""
        try:
            self.settings.update(new_settings)
            
            return {
                'success': True,
                'message': 'Settings updated successfully',
                'settings': self.settings
            }
            
        except Exception as e:
            logger.error(f"Error updating settings: {str(e)}")
            return {'success': False, 'message': f'Failed to update settings: {str(e)}'}
