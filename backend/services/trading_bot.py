import asyncio
import websocket
import json
import threading
import time
import logging
from datetime import datetime, timedelta
from collections import deque
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import random

from .market_analyzer import market_analyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BotSettings:
    selected_market: str = 'volatility_10_1s'
    auto_stake_enabled: bool = True
    auto_stake: float = 1.0
    manual_stake: float = 1.0
    max_stake: float = 10.0
    min_stake: float = 0.35
    stake_percentage: float = 10.0
    daily_stop_loss: float = 50.0
    daily_target: float = 20.0
    max_concurrent_trades: int = 3
    cooldown_period: int = 5
    strategy_mode: str = 'ADAPTIVE'  # ADAPTIVE, AGGRESSIVE, CONSERVATIVE

class TradeDirection(Enum):
    RISE = "RISE"
    FALL = "FALL"

class TradeStatus(Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    WON = "WON"
    LOST = "LOST"
    CANCELLED = "CANCELLED"

class StrategyType(Enum):
    ADAPTIVE_MEAN_REVERSION = "ADAPTIVE_MEAN_REVERSION"
    TREND_FOLLOWING = "TREND_FOLLOWING"
    MOMENTUM_BREAKOUT = "MOMENTUM_BREAKOUT"

@dataclass
class Trade:
    id: str
    strategy: str
    symbol: str
    direction: TradeDirection
    stake: float
    entry_price: float
    entry_time: datetime
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    payout: float = 0.0
    profit_loss: float = 0.0
    status: TradeStatus = TradeStatus.PENDING
    duration: int = 5  # seconds
    contract_id: Optional[str] = None

class TradingBot:
    def __init__(self, api_token: str = None):
        self.api_token = api_token
        self.is_running = False
        self.is_connected = False
        
        # Trading data
        self.active_trades: Dict[str, Trade] = {}
        self.trade_history: deque = deque(maxlen=1000)
        self.settings = BotSettings()
        
        # Market data
        self.current_price = 0.0
        self.price_history = deque(maxlen=500)
        self.tick_data = deque(maxlen=100)
        
        # Technical indicators
        self.rsi = 50.0
        self.macd = {"macd": 0, "signal": 0, "histogram": 0}
        self.bollinger_bands = {"upper": 0, "middle": 0, "lower": 0}
        self.momentum = 0.0
        self.volatility = 0.0
        
        # Strategy state
        self.current_strategy = "Adaptive Mean Reversion Rebound"
        self.strategy_status = "Monitoring"
        self.last_signal_time = None
        
        # Performance tracking
        self.daily_profit = 0.0
        self.daily_loss = 0.0
        self.total_trades_today = 0
        self.win_rate = 0.0
        self.account_balance = 1000.0
        
        # WebSocket and threading
        self.ws = None
        self.ws_thread = None
        self.analysis_thread = None
        self.subscribers = []
        
        # Trade execution
        self.last_trade_time = None
        self.trade_cooldown = 2  # seconds between trades

    def start_trading(self):
        """Start the trading bot"""
        if self.is_running:
            logger.warning("Trading bot is already running")
            return {'success': False, 'message': 'Trading bot is already running'}
        
        logger.info("Starting trading bot...")
        self.is_running = True
        
        # Start market data feed
        self._start_market_data_feed()
        
        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self._analysis_loop)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
        # Notify subscribers
        self._notify_subscribers({
            'type': 'bot_status',
            'data': {
                'is_running': True,
                'message': 'Trading bot started successfully'
            }
        })
        
        logger.info("Trading bot started successfully")
        return {'success': True, 'message': 'Trading bot started successfully'}

    def stop_trading(self):
        """Stop the trading bot"""
        if not self.is_running:
            logger.warning("Trading bot is not running")
            return {'success': False, 'message': 'Trading bot is not running'}
        
        logger.info("Stopping trading bot...")
        self.is_running = False
        
        # Close all active trades
        for trade_id in list(self.active_trades.keys()):
            self._close_trade(trade_id, force=True)
        
        # Stop WebSocket connection
        if self.ws:
            self.ws.close()
        
        # Notify subscribers
        self._notify_subscribers({
            'type': 'bot_status',
            'data': {
                'is_running': False,
                'message': 'Trading bot stopped'
            }
        })
        
        logger.info("Trading bot stopped")
        return {'success': True, 'message': 'Trading bot stopped'}

    # Alias methods for API compatibility
    def start(self):
        """Alias for start_trading"""
        return self.start_trading()
    
    def stop(self):
        """Alias for stop_trading"""
        return self.stop_trading()

    def _start_market_data_feed(self):
        """Start real-time market data feed"""
        if not self.api_token:
            # Use mock data for development
            self._start_mock_data_feed()
            return
        
        # Start WebSocket connection to Deriv API
        self.ws_thread = threading.Thread(target=self._websocket_connection)
        self.ws_thread.daemon = True
        self.ws_thread.start()

    def _start_mock_data_feed(self):
        """Start mock data feed for development/testing"""
        def mock_data_generator():
            base_price = 1.12345
            while self.is_running:
                # Generate realistic price movement
                change = np.random.normal(0, 0.0001)
                base_price += change
                
                tick_data = {
                    'price': base_price,
                    'timestamp': datetime.utcnow(),
                    'symbol': 'R_10'
                }
                
                self._process_tick_data(tick_data)
                time.sleep(0.1)  # 10 ticks per second
        
        mock_thread = threading.Thread(target=mock_data_generator)
        mock_thread.daemon = True
        mock_thread.start()

    def _websocket_connection(self):
        """WebSocket connection to Deriv API"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'tick' in data:
                    tick_data = {
                        'price': data['tick']['quote'],
                        'timestamp': datetime.utcnow(),
                        'symbol': data['tick']['symbol']
                    }
                    self._process_tick_data(tick_data)
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info("WebSocket connection closed")
            self.is_connected = False
        
        def on_open(ws):
            logger.info("WebSocket connection opened")
            self.is_connected = True
            
            # Subscribe to tick data
            subscribe_msg = {
                "ticks": "R_10",
                "subscribe": 1
            }
            ws.send(json.dumps(subscribe_msg))
        
        # Create WebSocket connection
        websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(
            "wss://ws.binaryws.com/websockets/v3?app_id=1089",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        self.ws.run_forever()

    def _process_tick_data(self, tick_data):
        """Process incoming tick data"""
        self.current_price = tick_data['price']
        self.tick_data.append(tick_data)
        self.price_history.append(tick_data['price'])
        
        # Add to market analyzer
        market_analyzer.add_price_data(tick_data['price'], tick_data['timestamp'])
        
        # Update technical indicators
        self._update_technical_indicators()
        
        # Notify subscribers of price update
        self._notify_subscribers({
            'type': 'price_update',
            'data': {
                'price': self.current_price,
                'timestamp': tick_data['timestamp'].isoformat(),
                'symbol': tick_data.get('symbol', 'R_10')
            }
        })

    def _update_technical_indicators(self):
        """Update technical indicators from price history"""
        if len(self.price_history) < 20:
            return
        
        prices = np.array(list(self.price_history))
        
        # RSI
        self.rsi = self._calculate_rsi(prices, 14)
        
        # MACD
        self.macd = self._calculate_macd(prices)
        
        # Bollinger Bands
        self.bollinger_bands = self._calculate_bollinger_bands(prices, 20, 2)
        
        # Momentum
        self.momentum = self._calculate_momentum(prices, 10)
        
        # Volatility
        self.volatility = self._calculate_volatility(prices)

    def _analysis_loop(self):
        """Main analysis and trading loop"""
        while self.is_running:
            try:
                if len(self.price_history) >= 50:
                    # Check for trading signals
                    signal = self._check_trading_signals()
                    
                    if signal and self._can_place_trade():
                        self._execute_trade(signal)
                
                # Update strategy status
                self._update_strategy_status()
                
                time.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                logger.error(f"Analysis loop error: {e}")
                time.sleep(1)

    def _check_trading_signals(self):
        """Check for trading signals using Adaptive Mean Reversion strategy"""
        if self.current_strategy != "Adaptive Mean Reversion Rebound":
            return None
        
        # Adaptive Mean Reversion Rebound Strategy
        # Conditions: RSI 48-52, Volatility 1-1.5%, Price touches Bollinger Band, Momentum < ±0.2%
        
        # Check RSI condition
        if not (48 <= self.rsi <= 52):
            return None
        
        # Check volatility condition (1-1.5%)
        volatility_percent = self.volatility * 100
        if not (1.0 <= volatility_percent <= 1.5):
            return None
        
        # Check momentum condition (< ±0.2%)
        momentum_percent = abs(self.momentum * 100)
        if momentum_percent >= 0.2:
            return None
        
        # Check MACD flat condition (-0.1 to +0.1)
        if not (-0.1 <= self.macd['macd'] <= 0.1):
            return None
        
        # Check Bollinger Band touches
        price_distance_upper = abs(self.current_price - self.bollinger_bands['upper'])
        price_distance_lower = abs(self.current_price - self.bollinger_bands['lower'])
        
        band_threshold = (self.bollinger_bands['upper'] - self.bollinger_bands['lower']) * 0.02
        
        signal = None
        
        # Touch lower band → Buy Rise (after 1 green tick)
        if price_distance_lower <= band_threshold:
            if self._detect_green_tick():
                signal = {
                    'direction': TradeDirection.RISE,
                    'confidence': 75,
                    'reason': 'Lower BB touch + Green tick',
                    'entry_price': self.current_price
                }
        
        # Touch upper band → Buy Fall (after 1 red tick)
        elif price_distance_upper <= band_threshold:
            if self._detect_red_tick():
                signal = {
                    'direction': TradeDirection.FALL,
                    'confidence': 75,
                    'reason': 'Upper BB touch + Red tick',
                    'entry_price': self.current_price
                }
        
        return signal

    def _detect_green_tick(self):
        """Detect green tick (price increase)"""
        if len(self.tick_data) < 2:
            return False
        
        current_tick = self.tick_data[-1]
        previous_tick = self.tick_data[-2]
        
        return current_tick['price'] > previous_tick['price']

    def _detect_red_tick(self):
        """Detect red tick (price decrease)"""
        if len(self.tick_data) < 2:
            return False
        
        current_tick = self.tick_data[-1]
        previous_tick = self.tick_data[-2]
        
        return current_tick['price'] < previous_tick['price']

    def _can_place_trade(self):
        """Check if we can place a new trade"""
        # Check daily limits
        if abs(self.daily_loss) >= self.settings.daily_stop_loss:
            return False
        
        if self.daily_profit >= self.settings.daily_target:
            return False
        
        # Check concurrent trades limit
        if len(self.active_trades) >= self.settings.max_concurrent_trades:
            return False
        
        # Check trade cooldown
        if self.last_trade_time:
            time_since_last = (datetime.utcnow() - self.last_trade_time).total_seconds()
            if time_since_last < self.trade_cooldown:
                return False
        
        return True

    def _execute_trade(self, signal):
        """Execute a trade based on signal"""
        try:
            # Calculate stake
            stake = self._calculate_stake()
            
            # Create trade
            trade = Trade(
                id=f"trade_{int(time.time() * 1000)}",
                strategy=self.current_strategy,
                symbol="R_10",
                direction=signal['direction'],
                stake=stake,
                entry_price=signal['entry_price'],
                entry_time=datetime.utcnow(),
                duration=6,  # 5-7 seconds for mean reversion
                status=TradeStatus.ACTIVE
            )
            
            # Add to active trades
            self.active_trades[trade.id] = trade
            
            # Schedule trade closure
            threading.Timer(trade.duration, self._close_trade, args=[trade.id]).start()
            
            # Update last trade time
            self.last_trade_time = datetime.utcnow()
            
            # Notify subscribers
            self._notify_subscribers({
                'type': 'new_trade',
                'data': asdict(trade)
            })
            
            logger.info(f"Trade executed: {trade.id} - {trade.direction.value} - ${trade.stake}")
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")

    def _close_trade(self, trade_id, force=False):
        """Close a trade and calculate P&L"""
        if trade_id not in self.active_trades:
            return
        
        trade = self.active_trades[trade_id]
        trade.exit_time = datetime.utcnow()
        trade.exit_price = self.current_price
        
        # Determine win/loss (simplified logic for demo)
        if force:
            trade.status = TradeStatus.CANCELLED
            trade.profit_loss = 0
        else:
            # Calculate if trade won based on direction and price movement
            price_change = trade.exit_price - trade.entry_price
            
            if trade.direction == TradeDirection.RISE:
                won = price_change > 0
            else:  # FALL
                won = price_change < 0
            
            if won:
                trade.status = TradeStatus.WON
                trade.payout = trade.stake * 1.8  # 80% payout
                trade.profit_loss = trade.payout - trade.stake
                self.daily_profit += trade.profit_loss
            else:
                trade.status = TradeStatus.LOST
                trade.profit_loss = -trade.stake
                self.daily_loss += abs(trade.profit_loss)
        
        # Move to history
        self.trade_history.appendleft(trade)
        del self.active_trades[trade_id]
        
        # Update statistics
        self._update_statistics()
        
        # Notify subscribers
        self._notify_subscribers({
            'type': 'trade_closed',
            'data': asdict(trade)
        })
        
        logger.info(f"Trade closed: {trade.id} - {trade.status.value} - P&L: ${trade.profit_loss}")

    def _calculate_stake(self):
        """Calculate stake amount"""
        if not self.settings.auto_stake_enabled:
            return min(self.settings.manual_stake, self.settings.max_stake)
        
        # Auto stake based on balance percentage
        auto_stake = self.account_balance * (self.settings.stake_percentage / 100)
        auto_stake = max(self.settings.min_stake, min(auto_stake, self.settings.max_stake))
        
        return round(auto_stake, 2)

    def _update_statistics(self):
        """Update trading statistics"""
        if not self.trade_history:
            return
        
        # Calculate win rate
        recent_trades = list(self.trade_history)[:50]  # Last 50 trades
        wins = sum(1 for trade in recent_trades if trade.status == TradeStatus.WON)
        total = len(recent_trades)
        
        self.win_rate = (wins / total * 100) if total > 0 else 0
        
        # Update account balance (simplified)
        net_profit = self.daily_profit - abs(self.daily_loss)
        self.account_balance = 1000.0 + net_profit  # Starting balance + net profit

    def _update_strategy_status(self):
        """Update strategy status"""
        if not self.is_running:
            self.strategy_status = "Stopped"
        elif len(self.price_history) < 50:
            self.strategy_status = "Gathering Data"
        elif not self._can_place_trade():
            if abs(self.daily_loss) >= self.settings.daily_stop_loss:
                self.strategy_status = "Daily Stop Loss Hit"
            elif self.daily_profit >= self.settings.daily_target:
                self.strategy_status = "Daily Target Reached"
            elif len(self.active_trades) >= self.settings.max_concurrent_trades:
                self.strategy_status = "Max Trades Active"
            else:
                self.strategy_status = "Cooldown"
        else:
            # Check strategy conditions
            if 48 <= self.rsi <= 52:
                volatility_ok = 1.0 <= (self.volatility * 100) <= 1.5
                momentum_ok = abs(self.momentum * 100) < 0.2
                macd_ok = -0.1 <= self.macd['macd'] <= 0.1
                
                if volatility_ok and momentum_ok and macd_ok:
                    self.strategy_status = "Waiting for BB Touch"
                else:
                    self.strategy_status = "Monitoring Conditions"
            else:
                self.strategy_status = "RSI Out of Range"

    # Technical indicator calculations
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)

    def _calculate_macd(self, prices):
        """Calculate MACD"""
        if len(prices) < 26:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
        
        exp1 = self._ema(prices, 12)
        exp2 = self._ema(prices, 26)
        macd_line = exp1 - exp2
        signal_line = self._ema(np.array([macd_line]), 9)
        
        return {
            'macd': float(macd_line),
            'signal': float(signal_line),
            'histogram': float(macd_line - signal_line)
        }

    def _ema(self, prices, period):
        """Calculate EMA"""
        if len(prices) < period:
            return np.mean(prices)
        
        alpha = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema

    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            middle = np.mean(prices)
            return {'upper': middle, 'middle': middle, 'lower': middle}
        
        recent_prices = prices[-period:]
        middle = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return {
            'upper': float(upper),
            'middle': float(middle),
            'lower': float(lower)
        }

    def _calculate_momentum(self, prices, period=10):
        """Calculate momentum"""
        if len(prices) < period + 1:
            return 0.0
        
        return float((prices[-1] - prices[-period-1]) / prices[-period-1])

    def _calculate_volatility(self, prices):
        """Calculate volatility"""
        if len(prices) < 2:
            return 0.0
        
        returns = np.diff(prices) / prices[:-1]
        return float(np.std(returns))

    # Subscription and notification system
    def subscribe(self, callback):
        """Subscribe to trading bot updates"""
        self.subscribers.append(callback)

    def unsubscribe(self, callback):
        """Unsubscribe from trading bot updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)

    def _notify_subscribers(self, message):
        """Notify all subscribers"""
        for callback in self.subscribers:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Subscriber notification error: {e}")

    # Public API methods
    def get_status(self):
        """Get current bot status"""
        return {
            'is_running': self.is_running,
            'is_connected': self.is_connected,
            'current_strategy': self.current_strategy,
            'strategy_status': self.strategy_status,
            'active_trades_count': len(self.active_trades),
            'current_price': self.current_price,
            'account_balance': self.account_balance,
            'daily_profit': self.daily_profit,
            'daily_loss': self.daily_loss,
            'win_rate': self.win_rate,
            'settings': asdict(self.settings)
        }

    def get_active_trades(self):
        """Get all active trades"""
        return [asdict(trade) for trade in self.active_trades.values()]

    def get_trade_history(self, limit=50):
        """Get trade history"""
        trades = list(self.trade_history)[:limit]
        return [asdict(trade) for trade in trades]

    def update_settings(self, new_settings):
        """Update trading settings"""
        try:
            for key, value in new_settings.items():
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)
            
            # Notify subscribers
            self._notify_subscribers({
                'type': 'settings_updated',
                'data': asdict(self.settings)
            })
            
            return {'success': True, 'message': 'Settings updated successfully'}
        except Exception as e:
            logger.error(f"Failed to update settings: {e}")
            return {'success': False, 'message': f'Failed to update settings: {str(e)}'}

    def get_statistics(self):
        """Get comprehensive trading statistics"""
        try:
            trade_history = list(self.trade_history)
            total_trades = len(trade_history)
            
            if total_trades == 0:
                return {
                    'total_trades': 0,
                    'won_trades': 0,
                    'lost_trades': 0,
                    'win_rate': 0,
                    'total_profit': 0,
                    'total_loss': 0,
                    'net_profit': 0,
                    'account_balance': self.account_balance,
                    'daily_profit': self.daily_profit,
                    'daily_loss': self.daily_loss
                }
            
            won_trades = len([t for t in trade_history if t.status == TradeStatus.WON])
            lost_trades = len([t for t in trade_history if t.status == TradeStatus.LOST])
            
            total_profit = sum(t.profit_loss for t in trade_history if t.profit_loss > 0)
            total_loss = abs(sum(t.profit_loss for t in trade_history if t.profit_loss < 0))
            net_profit = total_profit - total_loss
            
            return {
                'total_trades': total_trades,
                'won_trades': won_trades,
                'lost_trades': lost_trades,
                'win_rate': self.win_rate,
                'total_profit': total_profit,
                'total_loss': total_loss,
                'net_profit': net_profit,
                'account_balance': self.account_balance,
                'daily_profit': self.daily_profit,
                'daily_loss': self.daily_loss
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    def force_close_trade(self, trade_id):
        """Force close a specific trade"""
        try:
            if trade_id in self.active_trades:
                self._close_trade(trade_id, force=True)
                return {'success': True, 'message': f'Trade {trade_id} closed successfully'}
            else:
                return {'success': False, 'message': 'Trade not found'}
        except Exception as e:
            logger.error(f"Failed to force close trade {trade_id}: {e}")
            return {'success': False, 'message': f'Failed to close trade: {str(e)}'}

    def update_market_data(self, market, price):
        """Update market data for the bot"""
        try:
            # Update current price
            self.current_price = float(price)
            
            # Create tick data
            tick_data = {
                'price': float(price),
                'timestamp': datetime.utcnow(),
                'symbol': market
            }
            
            # Process the tick data
            self._process_tick_data(tick_data)
            
            logger.info(f"Market data updated: {market} = ${price}")
            return {'success': True, 'message': 'Market data updated successfully'}
        except Exception as e:
            logger.error(f"Failed to update market data: {e}")
            return {'success': False, 'message': f'Failed to update market data: {str(e)}'}

    def set_api_token(self, api_token):
        """Set the API token for real trading"""
        try:
            self.api_token = api_token
            logger.info("API token updated for trading bot")
            return {'success': True, 'message': 'API token updated successfully'}
        except Exception as e:
            logger.error(f"Failed to set API token: {e}")
            return {'success': False, 'message': f'Failed to set API token: {str(e)}'}

# Global trading bot instance
trading_bot = TradingBot()
