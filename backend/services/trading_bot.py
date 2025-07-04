import websocket
import json
import threading
import time
import logging
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import random
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BotSettings:
    selected_market: str = 'volatility_10_1s'
    stake: float = 1.0
    trade_interval: int = 7  # seconds between trades (random between 5-10)
    daily_stop_loss: float = 50.0
    daily_target: float = 20.0

class TradeDirection(Enum):
    RISE = "RISE"
    FALL = "FALL"

class TradeStatus(Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    WON = "WON"
    LOST = "LOST"
    CANCELLED = "CANCELLED"

@dataclass
class Trade:
    id: str
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
        self.price_history = deque(maxlen=100)
        self.tick_data = deque(maxlen=20)
        
        # Performance tracking
        self.daily_profit = 0.0
        self.daily_loss = 0.0
        self.total_trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        self.win_rate = 0.0
        self.account_balance = 1000.0
        
        # WebSocket and threading
        self.ws = None
        self.ws_thread = None
        self.trading_thread = None
        self.subscribers = []
        
        # Trade execution
        self.last_trade_time = None
        self.last_direction = None
        
        # Initialize technical indicators
        self.rsi = 50.0
        self.macd = {'macd': 0, 'signal': 0, 'histogram': 0}
        self.macd_history = deque(maxlen=100)
        self.bollinger_bands = {'upper': 0, 'middle': 0, 'lower': 0}
        self.momentum = 0.0
        self.volatility = 0.0
        
        # Strategy tracking - Fix for missing attribute
        self.active_strategies = {}
        self.strategy_stats = {}
        self.strategy_trades = {}
        
        # Initialize strategies
        self._initialize_strategies()

    def start_trading(self):
        """Start the trading bot"""
        if self.is_running:
            logger.warning("Trading bot is already running")
            return {'success': False, 'message': 'Trading bot is already running'}
        
        logger.info("Starting trading bot...")
        self.is_running = True
        
        # Start market data feed
        self._start_market_data_feed()
        
        # Start trading thread
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
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
        
        # Reset state
        self.last_trade_time = None
        self.last_direction = None
        
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
                # Generate realistic price movement (simple random walk)
                change = (random.random() - 0.5) * 0.0002
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
        """Establish WebSocket connection to Deriv API"""
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
                elif 'buy' in data:
                    # Handle buy response (trade confirmation)
                    logger.info(f"Received buy confirmation: {data}")
                    
                    # Extract trade_id from passthrough
                    if 'passthrough' in data and 'trade_id' in data['passthrough']:
                        trade_id = data['passthrough']['trade_id']
                        
                        # Update the trade with contract details
                        if trade_id in self.active_trades:
                            trade = self.active_trades[trade_id]
                            trade.contract_id = data['buy']['contract_id']
                            trade.status = TradeStatus.ACTIVE
                            
                            logger.info(f"Trade {trade_id} confirmed with contract ID: {trade.contract_id}")
                    else:
                        logger.warning("Received buy response without trade_id")
                elif 'proposal_open_contract' in data:
                    # Handle contract updates
                    contract = data['proposal_open_contract']
                    contract_id = str(contract['contract_id'])
                    
                    # Find the trade by contract_id
                    for trade_id, trade in self.active_trades.items():
                        if trade.contract_id == contract_id:
                            # Update trade details
                            if contract['is_sold'] == 1:
                                # Contract is finished
                                profit = float(contract['profit'])
                                trade.profit_loss = profit
                                trade.exit_price = float(contract['exit_tick'] or 0)
                                trade.exit_time = datetime.utcnow()
                                trade.status = TradeStatus.WON if profit > 0 else TradeStatus.LOST
                                
                                # Update stats
                                if profit > 0:
                                    self.daily_profit += profit
                                    self.wins_today += 1
                                else:
                                    self.daily_loss += abs(profit)
                                    self.losses_today += 1
                                
                                self.total_trades_today += 1
                                self._update_win_rate()
                                
                                # Move to history
                                self._move_trade_to_history(trade_id)
                                
                                logger.info(f"Trade {trade_id} completed with P&L: ${profit}")
                            

                            break
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
            
            # Authenticate with API token
            if self.api_token:
                auth_msg = {
                    "authorize": self.api_token
                }
                ws.send(json.dumps(auth_msg))
                logger.info("Authentication message sent")
            
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
        self.macd_history.append(self.macd.copy())
        
        # Bollinger Bands
        self.bollinger_bands = self._calculate_bollinger_bands(prices, 20, 2)
        
        # Momentum
        self.momentum = self._calculate_momentum(prices, 10)
        
        # Volatility
        self.volatility = self._calculate_volatility(prices)

    def _trading_loop(self):
        """Main fast tick trading loop - executes trades every 5-10 seconds"""
        while self.is_running:
            try:
                # Wait until we have price data
                if len(self.price_history) >= 5:
                    current_time = time.time()
                    
                    # Only place a trade after the interval has passed (5-10 seconds)
                    if (self.last_trade_time is None or 
                        (current_time - self.last_trade_time) >= random.randint(5, 10)):
                        
                        self._place_next_trade()
                        self.last_trade_time = current_time
                
                time.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                time.sleep(1)

    def _place_next_trade(self):
        """Place the next trade based on simple tick analysis"""
        try:
            if len(self.price_history) < 5:
                logger.warning("Not enough price data to make a trade decision")
                return
                
            # Get the latest price data
            current_price = self.price_history[-1]
            previous_price = self.price_history[-2]
            
            # Alternate between RISE and FALL for consistent trading
            direction = None
            
            # Simple tick analysis: alternating pattern for fast trading
            if self.last_direction == TradeDirection.RISE:
                direction = TradeDirection.FALL
            elif self.last_direction == TradeDirection.FALL:
                direction = TradeDirection.RISE
            else:
                # First trade or after reset - decide based on current tick direction
                direction = TradeDirection.RISE if current_price > previous_price else TradeDirection.FALL
            
            # Store the direction for next trade
            self.last_direction = direction
            
            # Calculate stake
            stake = self.settings.stake
            
            # Create trade
            trade_id = f"trade_{int(time.time() * 1000)}"
            
            trade = Trade(
                id=trade_id,
                symbol=self.settings.selected_market,
                direction=direction,
                stake=stake,
                entry_price=current_price,
                entry_time=datetime.utcnow(),
                duration=5,  # 5-second trades for fast trading
                status=TradeStatus.ACTIVE
            )
            
            # Add to active trades
            self.active_trades[trade.id] = trade
            
            # Update total trades count
            self.total_trades_today += 1
            
            # Notify subscribers
            self._notify_subscribers({
                'type': 'new_trade',
                'data': asdict(trade)
            })
            
            logger.info(f"Fast trade executed: {trade.id} - {direction.value} - ${stake}")
            
            # Schedule trade closure
            threading.Timer(trade.duration, self._close_trade, args=[trade.id]).start()
            
        except Exception as e:
            logger.error(f"Error placing next trade: {e}")

    def _execute_trade(self, signal):
        """Execute a trade based on signal"""
        try:
            # Check if we have a valid API token
            if not self.api_token:
                logger.error("Cannot execute trade: No API token available")
                return
                
            # Calculate stake
            stake = self._calculate_stake()
            
            # Create trade object locally
            trade_id = f"trade_{int(time.time() * 1000)}"
            trade = Trade(
                id=trade_id,
                symbol="R_10",
                direction=signal['direction'],
                stake=stake,
                entry_price=signal['entry_price'],
                entry_time=datetime.utcnow(),
                duration=signal.get('duration', 6),  # Use signal duration or default to 6 seconds
                status=TradeStatus.PENDING
            )
            
            # Add to active trades
            self.active_trades[trade.id] = trade
            
            # IMPORTANT: Send the trade to Deriv API
            if self.ws and self.is_connected:
                # Prepare the contract request
                contract_type = "CALL" if trade.direction == TradeDirection.RISE else "PUT"
                duration_unit = "s"  # seconds
                
                # Create the buy request
                buy_request = {
                    "buy": 1,
                    "price": stake,
                    "parameters": {
                        "contract_type": contract_type,
                        "currency": "USD",  # Should use account currency
                        "duration": trade.duration,
                        "duration_unit": duration_unit,
                        "symbol": trade.symbol,
                    },
                    "passthrough": {
                        "trade_id": trade.id
                    }
                }
                
                # Send buy request
                try:
                    self.ws.send(json.dumps(buy_request))
                    logger.info(f"Buy request sent to Deriv API for trade: {trade.id}")
                    
                    # Update trade status to ACTIVE
                    trade.status = TradeStatus.ACTIVE
                except Exception as e:
                    logger.error(f"Failed to send buy request to Deriv API: {e}")
                    trade.status = TradeStatus.CANCELLED
            else:
                # If we're not connected to Deriv API, mark as simulated
                logger.warning(f"Simulated trade only (no API connection): {trade.id}")
            
            # Update last trade time
            self.last_trade_time = datetime.utcnow()
            
            # Notify subscribers
            self._notify_subscribers({
                'type': 'new_trade',
                'data': asdict(trade)
            })
            
            logger.info(f"Trade executed: {trade.id} - {trade.direction.value} - ${trade.stake}")
            
            # If trade is active, schedule closure
            if trade.status == TradeStatus.ACTIVE:
                threading.Timer(trade.duration, self._close_trade, args=[trade.id]).start()
            
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
                self.wins_today += 1
            else:
                trade.status = TradeStatus.LOST
                trade.profit_loss = -trade.stake
                self.daily_loss += abs(trade.profit_loss)
                self.losses_today += 1
        
        # Move to history
        self.trade_history.appendleft(trade)
        del self.active_trades[trade_id]
        
        # Update statistics
        self._update_statistics()
        
        # Notify subscribers about the closed trade
        self._notify_subscribers({
            'type': 'trade_closed',
            'data': asdict(trade)
        })
        
        logger.info(f"Trade closed: {trade.id} - Status: {trade.status.value} - P&L: ${trade.profit_loss:.2f}")
        
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
        """Update trading statistics for fast trading"""
        if not self.trade_history:
            return
        
        # Calculate win rate from today's trades
        total_today = self.wins_today + self.losses_today
        self.win_rate = (self.wins_today / total_today * 100) if total_today > 0 else 0
        
        # Update account balance (simplified)
        net_profit = self.daily_profit - abs(self.daily_loss)
        self.account_balance = 1000.0 + net_profit  # Starting balance + net profit
        
        # Check daily stop loss or target
        if abs(self.daily_loss) >= self.settings.daily_stop_loss:
            logger.warning("Daily stop loss reached. Stopping trading.")
            self.stop_trading()
            
        if self.daily_profit >= self.settings.daily_target:
            logger.info("Daily profit target reached. Stopping trading.")
            self.stop_trading()

    def _can_place_trade(self):
        """Check if we can place a new trade based on settings and conditions"""
        # Check if bot is running
        if not self.is_running:
            return False

        # Check if we have enough data
        if len(self.price_history) < 50:
            return False
            
        # Check if we've reached maximum concurrent trades
        if len(self.active_trades) >= self.settings.max_concurrent_trades:
            return False
            
        # Check if we're in cooldown period
        if self.last_trade_time and (datetime.utcnow() - self.last_trade_time).total_seconds() < self.trade_cooldown:
            return False
            
        # Check daily loss limit
        if self.daily_loss >= self.settings.daily_stop_loss:
            return False
            
        # Check daily target reached
        if self.daily_profit >= self.settings.daily_target:
            return False
            
        return True

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
        """Notify all subscribers with a message"""
        # Add strategy stats to the message if it doesn't already include them
        if message['type'] in ['bot_status', 'price_update'] and 'strategy_stats' not in message['data']:
            message['data']['strategy_stats'] = self.get_strategy_stats()
            message['data']['active_strategies'] = self.active_strategies
        
        for subscriber in self.subscribers:
            try:
                subscriber(message)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")

    # Public API methods
    def get_status(self):
        """Get current bot status for fast trading bot"""
        return {
            'is_running': self.is_running,
            'is_connected': self.is_connected,
            'active_trades_count': len(self.active_trades),
            'current_price': self.current_price,
            'account_balance': self.account_balance,
            'daily_profit': self.daily_profit,
            'daily_loss': self.daily_loss,
            'win_rate': self.win_rate,
            'total_trades_today': self.total_trades_today,
            'wins_today': self.wins_today,
            'losses_today': self.losses_today,
            'settings': asdict(self.settings),
            'last_trade_time': self.last_trade_time,
            'last_direction': self.last_direction.value if self.last_direction else None
        }

    def get_active_trades(self):
        """Get all active trades"""
        return [asdict(trade) for trade in self.active_trades.values()]

    def get_trade_history(self, limit=50):
        """Get trade history"""
        try:
            # Check if trade history is initialized
            if not hasattr(self, 'trade_history') or self.trade_history is None:
                logger.warning("Trade history not initialized, returning empty list")
                return []
            
            # Convert deque to list and limit the number of records
            trades = list(self.trade_history)[:limit]
            
            # Convert each trade object to a dictionary for JSON serialization
            trade_dicts = []
            for trade in trades:
                try:
                    # Handle different types of trade objects
                    if hasattr(trade, '__dict__'):
                        # For class instances, use asdict if possible
                        trade_dict = asdict(trade)
                    elif isinstance(trade, dict):
                        # Already a dict
                        trade_dict = trade
                    else:
                        # Convert to string representation as fallback
                        logger.warning(f"Unknown trade object type: {type(trade)}")
                        trade_dict = {"id": str(id(trade)), "data": str(trade)}
                    
                    trade_dicts.append(trade_dict)
                except Exception as e:
                    logger.error(f"Error processing trade history record: {e}")
            
            logger.info(f"Returning {len(trade_dicts)} trade history records")
            return trade_dicts
            
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return []

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

    def set_strategy(self, strategy_id):
        """Set the current trading strategy by ID"""
        strategies = self.get_available_strategies()
        for strategy in strategies:
            if strategy['id'] == strategy_id:
                self.current_strategy = strategy['name']
                return {'success': True, 'message': f'Strategy set to {strategy["name"]}'}
                
        return {'success': False, 'message': f'Strategy with ID {strategy_id} not found'}

    def get_available_strategies(self):
        """Get a list of all available trading strategies"""
        return [
            {
                'id': 1,
                'name': 'Adaptive Mean Reversion Rebound',
                'description': 'Trade reversions from Bollinger Bands with strict filters',
                'risk_level': 'low',
                'timeframe': '5-7 seconds'
            },
            {
                'id': 2,
                'name': 'Micro-Trend Momentum Tracker',
                'description': 'Ride micro-trends driven by consistent direction and minor momentum',
                'risk_level': 'medium',
                'timeframe': '6-10 seconds'
            },
            {
                'id': 3,
                'name': 'RSI-Tick Divergence Detector',
                'description': 'Spot reversals when indicator and price ticks diverge',
                'risk_level': 'medium',
                'timeframe': '5 seconds'
            },
            {
                'id': 4,
                'name': 'Volatility Spike Fader',
                'description': 'Fade short bursts of volatility after deviation from the mean',
                'risk_level': 'medium',
                'timeframe': '5-8 seconds'
            },
            {
                'id': 5,
                'name': 'Tick Flow Strength Pulse',
                'description': 'Exploit strong directional tick flow for micro breakout trades',
                'risk_level': 'medium-high',
                'timeframe': '5 seconds'
            },
            {
                'id': 6,
                'name': 'Double Confirmation Breakout',
                'description': 'Filter breakout trades using MACD crossover + EMA slope',
                'risk_level': 'medium',
                'timeframe': '6-8 seconds'
            },
            {
                'id': 7,
                'name': 'RSI Overextension Fade',
                'description': 'Trade reversals from RSI overbought or oversold zones',
                'risk_level': 'medium',
                'timeframe': '5-7 seconds'
            },
            {
                'id': 8,
                'name': 'Multi-Tick Pivot Bounce',
                'description': 'Detect reversal from localized price pivots',
                'risk_level': 'medium',
                'timeframe': '5 seconds'
            },
            {
                'id': 9,
                'name': 'MACD-Momentum Sync Engine',
                'description': 'Trade only when MACD and momentum confirm each other',
                'risk_level': 'medium',
                'timeframe': '6-10 seconds'
            },
            {
                'id': 10,
                'name': 'Time-of-Tick Scalper',
                'description': 'Use timing between ticks to detect fast or slow market surges',
                'risk_level': 'high',
                'timeframe': '6 seconds'
            },
            {
                'id': 11,
                'name': 'Volatility Collapse Compression',
                'description': 'Anticipate breakouts after tight price compression',
                'risk_level': 'medium-high',
                'timeframe': '7 seconds'
            },
            {
                'id': 12,
                'name': 'Two-Step Confirmation Model',
                'description': 'Require sequential indicator agreement before trade',
                'risk_level': 'low',
                'timeframe': '6-10 seconds'
            },
            {
                'id': 13,
                'name': 'Inverted Divergence Flip',
                'description': 'Use inverse divergence signals when MACD and price disagree',
                'risk_level': 'high',
                'timeframe': '5 seconds'
            },
            {
                'id': 14,
                'name': 'Cumulative Strength Index Pullback',
                'description': 'Trade minor pullbacks in strong micro-trends',
                'risk_level': 'medium',
                'timeframe': '6-8 seconds'
            },
            {
                'id': 15,
                'name': 'Tri-Indicator Confluence Strategy',
                'description': 'Enter only when RSI, MACD, and Momentum all align',
                'risk_level': 'low',
                'timeframe': '7-10 seconds'
            }
        ]
        
    def get_strategy_details(self, strategy_id):
        """Get details for a specific strategy by ID"""
        strategies = self.get_available_strategies()
        for strategy in strategies:
            if strategy['id'] == strategy_id:
                return strategy
        return None

    def set_strategy_status(self, strategy_id, active=True):
        """Enable or disable a specific strategy"""
        try:
            if strategy_id in self.active_strategies:
                self.active_strategies[strategy_id] = active
                status = "Active" if active else "Inactive"
                self.strategy_stats[strategy_id]['status'] = status
                
                logger.info(f"Strategy {strategy_id} set to {status}")
                return {
                    'success': True, 
                    'message': f'Strategy {strategy_id} is now {status.lower()}'
                }
            else:
                return {
                    'success': False, 
                    'message': f'Strategy with ID {strategy_id} not found'
                }
        except Exception as e:
            logger.error(f"Error setting strategy status: {e}")
            return {'success': False, 'message': f'Error: {str(e)}'}

    def get_strategy_stats(self):
        """Get statistics for all strategies"""
        # Ensure strategy_stats is initialized
        if not hasattr(self, 'strategy_stats') or self.strategy_stats is None:
            self.strategy_stats = {}
            strategies = self.get_available_strategies()
            for strategy in strategies:
                strategy_id = strategy['id']
                self.strategy_stats[strategy_id] = {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0.0,
                    'profit': 0.0,
                    'status': 'Inactive',
                    'last_signal_time': None
                }
        
        return self.strategy_stats

    def get_strategy_trades(self, strategy_id=None):
        """Get trades for a specific strategy or all strategies"""
        if strategy_id is not None:
            # Return trades for this specific strategy
            return self.strategy_trades.get(strategy_id, [])
        else:
            # Return all trades grouped by strategy
            return self.strategy_trades

    def _initialize_strategies(self):
        """Initialize available strategies and their tracking data"""
        # Get all available strategies
        strategies = self.get_available_strategies()
        
        # Initialize active_strategies dictionary if not already present
        if not hasattr(self, 'active_strategies') or self.active_strategies is None:
            self.active_strategies = {}
        
        # Initialize strategy_stats dictionary if not already present
        if not hasattr(self, 'strategy_stats') or self.strategy_stats is None:
            self.strategy_stats = {}
            
        # Initialize strategy_trades dictionary if not already present
        if not hasattr(self, 'strategy_trades') or self.strategy_trades is None:
            self.strategy_trades = {}
            
        # Setup tracking for each available strategy
        for strategy in strategies:
            strategy_id = strategy['id']
            
            # Set default activation status (only first strategy active by default)
            if strategy_id not in self.active_strategies:
                self.active_strategies[strategy_id] = (strategy_id == 1)
                
            # Initialize statistics tracking
            if strategy_id not in self.strategy_stats:
                self.strategy_stats[strategy_id] = {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0.0,
                    'profit': 0.0,
                    'status': 'Inactive',
                    'last_signal_time': None
                }
                
            # Initialize trade history tracking
            if strategy_id not in self.strategy_trades:
                self.strategy_trades[strategy_id] = []
                
        logger.info(f"Initialized {len(strategies)} trading strategies")

# Create a singleton instance of the TradingBot
trading_bot = TradingBot()
