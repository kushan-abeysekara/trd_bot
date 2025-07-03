from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime, timedelta
import json
import logging
import sqlite3
import asyncio
from threading import Lock, Thread
import time
import numpy as np
import pandas as pd

from models import User, db
from utils.deriv_service import DerivService
from services.technical_analyzer import TechnicalAnalyzer
from services.strategy_engine import StrategyEngine

logger = logging.getLogger(__name__)

# Blueprint for technical trading bot
technical_bot_bp = Blueprint('technical_bot', __name__, url_prefix='/api/technical-bot')

# Global bot instances
active_technical_bots = {}
bot_lock = Lock()

class TechnicalTradingBot:
    def __init__(self, user_id, api_token, account_type='demo'):
        self.user_id = user_id
        self.api_token = api_token
        self.account_type = account_type
        self.deriv_service = DerivService()
        self.technical_analyzer = TechnicalAnalyzer()
        self.strategy_engine = StrategyEngine()
        
        # Bot state
        self.is_active = False
        self.current_strategy = "Adaptive Mean Reversion Rebound"
        self.start_time = None
        self.stop_time = None
        
        # Trading settings
        self.base_stake = 1.0
        self.max_stake = 10.0
        self.daily_stop_loss = 100.0
        self.daily_target = 200.0
        self.risk_per_trade = 2.0  # 2% of balance
        
        # Performance tracking
        self.account_balance = 0.0
        self.starting_balance = 0.0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.net_pnl = 0.0
        self.open_trades = []
        self.trade_history = []
        self.daily_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Strategy metrics
        self.current_rsi = 50.0
        self.current_volatility = 0.0
        self.current_momentum = 0.0
        self.bollinger_upper = 0.0
        self.bollinger_lower = 0.0
        self.macd_value = 0.0
        self.last_tick_direction = None
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize technical bot database tables"""
        try:
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            
            # Technical bot sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS technical_bot_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_end TIMESTAMP,
                    starting_balance REAL,
                    ending_balance REAL,
                    total_profit REAL,
                    total_loss REAL,
                    net_pnl REAL,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    win_rate REAL,
                    account_type TEXT,
                    strategies_used TEXT
                )
            ''')
            
            # Technical trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS technical_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_id INTEGER,
                    contract_id TEXT,
                    strategy_name TEXT,
                    entry_time TIMESTAMP,
                    exit_time TIMESTAMP,
                    entry_price REAL,
                    exit_price REAL,
                    stake_amount REAL,
                    profit_loss REAL,
                    contract_type TEXT,
                    duration INTEGER,
                    success BOOLEAN,
                    rsi_value REAL,
                    volatility REAL,
                    momentum REAL,
                    macd_value REAL,
                    bollinger_position TEXT,
                    market_conditions TEXT,
                    FOREIGN KEY (session_id) REFERENCES technical_bot_sessions (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Technical bot database initialized")
            
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
    
    async def start_bot(self):
        """Start the technical trading bot"""
        try:
            # Validate API token
            validation_result = self.deriv_service.validate_token(self.api_token)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'message': f'Invalid API token: {validation_result["message"]}'
                }
            
            # Get account balance
            balance_result = self.deriv_service.get_account_info(self.api_token)
            if not balance_result['success']:
                return {
                    'success': False,
                    'message': f'Failed to get account info: {balance_result["message"]}'
                }
            
            self.account_balance = balance_result['data']['balance']
            self.starting_balance = self.account_balance
            self.is_active = True
            self.start_time = datetime.now()
            
            # Calculate automatic stake based on balance
            self.base_stake = max(1.0, (self.account_balance * self.risk_per_trade / 100))
            
            # Start trading loop in background
            self.trading_thread = Thread(target=self._trading_loop)
            self.trading_thread.daemon = True
            self.trading_thread.start()
            
            return {
                'success': True,
                'message': 'Technical trading bot started successfully',
                'data': {
                    'account_balance': self.account_balance,
                    'base_stake': self.base_stake,
                    'current_strategy': self.current_strategy,
                    'start_time': self.start_time.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error starting technical bot: {str(e)}")
            return {
                'success': False,
                'message': f'Failed to start bot: {str(e)}'
            }
    
    async def stop_bot(self):
        """Stop the technical trading bot"""
        try:
            self.is_active = False
            self.stop_time = datetime.now()
            
            # Close any open trades
            for trade in self.open_trades:
                try:
                    # Close trade logic here
                    pass
                except Exception as e:
                    logger.error(f"Error closing trade {trade.get('contract_id')}: {str(e)}")
            
            # Save session to database
            self._save_session()
            
            return {
                'success': True,
                'message': 'Technical trading bot stopped successfully',
                'data': {
                    'session_duration': str(self.stop_time - self.start_time),
                    'total_trades': self.daily_trades,
                    'net_pnl': self.net_pnl,
                    'win_rate': (self.winning_trades / max(1, self.daily_trades)) * 100
                }
            }
            
        except Exception as e:
            logger.error(f"Error stopping technical bot: {str(e)}")
            return {
                'success': False,
                'message': f'Failed to stop bot: {str(e)}'
            }
    
    def _trading_loop(self):
        """Main trading loop for technical strategies"""
        while self.is_active:
            try:
                # Get latest market data
                market_data = self.deriv_service.get_latest_ticks('volatility_10_1s', 100)
                
                if market_data and len(market_data) >= 50:
                    # Update technical indicators
                    self._update_indicators(market_data)
                    
                    # Run strategy analysis with current strategy
                    signal = self._analyze_with_current_strategy(market_data)
                    
                    # Execute trade if signal found
                    if signal and signal['action'] != 'HOLD':
                        self._execute_trade(signal, market_data[-1])
                
                # Wait before next analysis
                time.sleep(1)  # Check every second for 5-10 second trades
                
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(5)
    
    def _update_indicators(self, market_data):
        """Update technical indicators from market data"""
        try:
            prices = [tick['quote'] for tick in market_data]
            self.current_rsi = self.technical_analyzer.calculate_rsi(prices, 14)
            self.current_volatility = self.technical_analyzer.calculate_volatility(prices, 20)
            self.current_momentum = self.technical_analyzer.calculate_momentum(prices, 10)
            
            bollinger_bands = self.technical_analyzer.calculate_bollinger_bands(prices, 20, 2)
            self.bollinger_upper = bollinger_bands['upper'][-1]
            self.bollinger_lower = bollinger_bands['lower'][-1]
            
            macd_data = self.technical_analyzer.calculate_macd(prices, 12, 26, 9)
            self.macd_value = macd_data['macd'][-1]
            
        except Exception as e:
            logger.error(f"Error updating indicators: {str(e)}")
    
    def _execute_trade(self, signal, market_tick):
        """Execute a trade based on signal"""
        try:
            # Check daily limits
            if abs(self.net_pnl) >= self.daily_stop_loss:
                return
            
            if self.net_pnl >= self.daily_target:
                return
            
            # Calculate stake amount
            stake = self._calculate_stake()
            
            # Create trade proposal
            proposal = {
                'contract_type': signal['contract_type'],
                'symbol': 'volatility_10_1s',
                'amount': stake,
                'duration': signal.get('duration', 6),  # 6 seconds default
                'barrier': signal.get('barrier'),
                'basis': 'stake'
            }
            
            # Execute trade
            trade_result = self.deriv_service.place_trade(self.api_token, proposal)
            
            if trade_result.get('success'):
                trade_data = {
                    'contract_id': trade_result['contract_id'],
                    'strategy_name': self.current_strategy,
                    'entry_time': datetime.now(),
                    'entry_price': market_tick['quote'],
                    'stake_amount': stake,
                    'contract_type': signal['contract_type'],
                    'duration': proposal['duration'],
                    'rsi_value': self.current_rsi,
                    'volatility': self.current_volatility,
                    'momentum': self.current_momentum,
                    'macd_value': self.macd_value,
                    'market_conditions': json.dumps({
                        'bollinger_position': self._get_bollinger_position(market_tick['quote']),
                        'signal_strength': signal.get('confidence', 0.5)
                    })
                }
                
                self.open_trades.append(trade_data)
                self.daily_trades += 1
                
                logger.info(f"Trade executed: {signal['contract_type']} with stake {stake}")
                
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
    
    def _calculate_stake(self):
        """Calculate stake amount based on balance and risk management"""
        base_stake = self.account_balance * (self.risk_per_trade / 100)
        return min(max(base_stake, 1.0), self.max_stake)
    
    def _get_bollinger_position(self, price):
        """Determine price position relative to Bollinger Bands"""
        if price >= self.bollinger_upper:
            return "above_upper"
        elif price <= self.bollinger_lower:
            return "below_lower"
        else:
            return "middle"
    
    def _save_session(self):
        """Save trading session to database"""
        try:
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            
            win_rate = (self.winning_trades / max(1, self.daily_trades)) * 100
            
            cursor.execute('''
                INSERT INTO technical_bot_sessions 
                (user_id, session_start, session_end, starting_balance, ending_balance,
                 total_profit, total_loss, net_pnl, total_trades, winning_trades, 
                 losing_trades, win_rate, account_type, strategies_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.user_id, self.start_time, self.stop_time, self.starting_balance,
                self.account_balance, self.total_profit, self.total_loss, self.net_pnl,
                self.daily_trades, self.winning_trades, self.losing_trades, win_rate,
                self.account_type, self.current_strategy
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving session: {str(e)}")

    def _analyze_with_current_strategy(self, market_data):
        """Analyze market data using the current strategy"""
        try:
            prices = [tick['quote'] for tick in market_data]
            timestamps = [tick['epoch'] for tick in market_data]
            
            # Update current strategy if needed (rotate every 10 trades)
            if self.daily_trades > 0 and self.daily_trades % 10 == 0:
                self.current_strategy = self.strategy_engine.rotate_strategy()
            
            # Strategy-specific analysis
            if self.current_strategy == "Adaptive Mean Reversion Rebound":
                return self.strategy_engine.analyze_adaptive_mean_reversion(
                    rsi=self.current_rsi,
                    volatility=self.current_volatility,
                    momentum=self.current_momentum,
                    bollinger_upper=self.bollinger_upper,
                    bollinger_lower=self.bollinger_lower,
                    macd=self.macd_value,
                    current_price=prices[-1]
                )
            
            elif self.current_strategy == "RSI Momentum Breakout":
                return self.strategy_engine.analyze_rsi_momentum_breakout(
                    rsi=self.current_rsi,
                    momentum=self.current_momentum,
                    volatility=self.current_volatility,
                    current_price=prices[-1],
                    price_history=prices
                )
            
            elif self.current_strategy == "Bollinger Band Squeeze":
                return self.strategy_engine.analyze_bollinger_squeeze(
                    bollinger_upper=self.bollinger_upper,
                    bollinger_lower=self.bollinger_lower,
                    volatility=self.current_volatility,
                    rsi=self.current_rsi,
                    price_history=prices
                )
            
            elif self.current_strategy == "MACD Histogram Divergence":
                return self.strategy_engine.analyze_macd_histogram_divergence(
                    price_history=prices,
                    macd_data={
                        'macd': [self.macd_value],
                        'signal': [self.macd_value * 0.9],
                        'histogram': [self.macd_value * 0.1]
                    }
                )
            
            elif self.current_strategy == "Volatility Expansion Scalp":
                volatility_history = [self.current_volatility] * 10  # Simplified
                return self.strategy_engine.analyze_volatility_expansion(
                    volatility=self.current_volatility,
                    volatility_history=volatility_history,
                    rsi=self.current_rsi,
                    current_price=prices[-1],
                    bollinger_upper=self.bollinger_upper,
                    bollinger_lower=self.bollinger_lower
                )
            
            elif self.current_strategy == "Tick Velocity Momentum":
                return self.strategy_engine.analyze_tick_velocity_momentum(
                    price_history=prices,
                    timestamps=timestamps,
                    rsi=self.current_rsi
                )
            
            elif self.current_strategy == "Support Resistance Bounce":
                # Calculate support/resistance levels
                support_levels = [min(prices[-20:]), min(prices[-40:-20])]
                resistance_levels = [max(prices[-20:]), max(prices[-40:-20])]
                return self.strategy_engine.analyze_support_resistance_bounce(
                    current_price=prices[-1],
                    support_levels=support_levels,
                    resistance_levels=resistance_levels,
                    rsi=self.current_rsi,
                    momentum=self.current_momentum
                )
            
            elif self.current_strategy == "EMA Crossover Micro":
                return self.strategy_engine.analyze_ema_crossover_micro(
                    price_history=prices
                )
            
            elif self.current_strategy == "Williams R Extreme":
                return self.strategy_engine.analyze_williams_r_extreme(
                    price_history=prices,
                    rsi=self.current_rsi,
                    volatility=self.current_volatility
                )
            
            elif self.current_strategy == "Stochastic Divergence":
                return self.strategy_engine.analyze_stochastic_divergence(
                    price_history=prices,
                    rsi=self.current_rsi,
                    momentum=self.current_momentum
                )
            
            elif self.current_strategy == "Volume Price Trend":
                volume_proxy = [abs(p1 - p2) for p1, p2 in zip(prices[:-1], prices[1:])]
                return self.strategy_engine.analyze_volume_price_trend(
                    price_history=prices,
                    volume_proxy=volume_proxy,
                    rsi=self.current_rsi
                )
            
            elif self.current_strategy == "Microtrend Reversal":
                return self.strategy_engine.analyze_microtrend_reversal(
                    price_history=prices,
                    rsi=self.current_rsi,
                    momentum=self.current_momentum,
                    volatility=self.current_volatility
                )
            
            elif self.current_strategy == "High Frequency Scalp":
                return self.strategy_engine.analyze_high_frequency_scalp(
                    price_history=prices,
                    tick_timestamps=timestamps,
                    rsi=self.current_rsi
                )
            
            elif self.current_strategy == "Neural Pattern Recognition":
                return self.strategy_engine.analyze_neural_pattern_recognition(
                    price_history=prices,
                    rsi=self.current_rsi,
                    volatility=self.current_volatility,
                    momentum=self.current_momentum
                )
            
            elif self.current_strategy == "Adaptive Multi-Timeframe":
                # Create 5-minute data by sampling every 5th tick
                price_history_5m = prices[::5] if len(prices) >= 25 else prices
                return self.strategy_engine.analyze_adaptive_multi_timeframe(
                    price_history_1m=prices,
                    price_history_5m=price_history_5m,
                    rsi=self.current_rsi,
                    macd=self.macd_value,
                    momentum=self.current_momentum
                )
            
            # Default fallback
            return {'action': 'HOLD', 'confidence': 0.0}
            
        except Exception as e:
            logger.error(f"Error in strategy analysis: {str(e)}")
            return {'action': 'HOLD', 'confidence': 0.0}
    
    def get_performance_metrics(self):
        """Calculate and return performance metrics"""
        try:
            # Get trading history from database
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            
            # Get all trades for this user
            cursor.execute('''
                SELECT profit_loss, result, created_at, strategy_used
                FROM technical_bot_trades 
                WHERE user_id = ? 
                ORDER BY created_at ASC
            ''', (self.user_id,))
            
            trades = cursor.fetchall()
            conn.close()
            
            if not trades:
                return {
                    'total_profit': 0.0,
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'today_profit': 0.0,
                    'today_trades': 0,
                    'average_trade_duration': 0.0,
                    'max_consecutive_wins': 0,
                    'max_consecutive_losses': 0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0
                }
            
            # Calculate metrics
            total_profit = sum(trade[0] for trade in trades)
            total_trades = len(trades)
            winning_trades = sum(1 for trade in trades if trade[1] == 'WIN')
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # Today's performance
            today = datetime.now().date()
            today_trades = [trade for trade in trades if datetime.fromisoformat(trade[2]).date() == today]
            today_profit = sum(trade[0] for trade in today_trades)
            today_count = len(today_trades)
            
            # Consecutive wins/losses
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            current_wins = 0
            current_losses = 0
            
            for trade in trades:
                if trade[1] == 'WIN':
                    current_wins += 1
                    current_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, current_wins)
                else:
                    current_losses += 1
                    current_wins = 0
                    max_consecutive_losses = max(max_consecutive_losses, current_losses)
            
            # Calculate drawdown
            running_profit = 0
            peak_profit = 0
            max_drawdown = 0
            
            for trade in trades:
                running_profit += trade[0]
                peak_profit = max(peak_profit, running_profit)
                drawdown = peak_profit - running_profit
                max_drawdown = max(max_drawdown, drawdown)
            
            # Simple Sharpe ratio calculation (assuming risk-free rate of 0)
            if total_trades > 1:
                returns = [trade[0] for trade in trades]
                avg_return = sum(returns) / len(returns)
                std_return = (sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)) ** 0.5
                sharpe_ratio = avg_return / std_return if std_return > 0 else 0.0
            else:
                sharpe_ratio = 0.0
            
            return {
                'total_profit': round(total_profit, 2),
                'total_trades': total_trades,
                'win_rate': round(win_rate, 4),
                'today_profit': round(today_profit, 2),
                'today_trades': today_count,
                'average_trade_duration': 60.0,  # Default 1 minute for binary options
                'max_consecutive_wins': max_consecutive_wins,
                'max_consecutive_losses': max_consecutive_losses,
                'sharpe_ratio': round(sharpe_ratio, 4),
                'max_drawdown': round(max_drawdown, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {
                'total_profit': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'today_profit': 0.0,
                'today_trades': 0,
                'average_trade_duration': 0.0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }

@technical_bot_bp.route('/start', methods=['POST'])
@jwt_required()
def start_technical_bot():
    """Start the technical trading bot"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        # Get user and API token
        user = User.query.get(user_id)
        if not user or not user.deriv_api_token:
            return jsonify({
                'success': False,
                'error': 'API token not configured'
            }), 400
        
        account_type = data.get('account_type', 'demo')
        
        with bot_lock:
            # Check if bot is already running
            if user_id in active_technical_bots and active_technical_bots[user_id].is_active:
                return jsonify({
                    'success': False,
                    'error': 'Technical bot is already active'
                }), 400
            
            # Create and start bot
            bot = TechnicalTradingBot(user_id, user.deriv_api_token, account_type)
            
            # Update settings if provided
            if 'base_stake' in data:
                bot.base_stake = float(data['base_stake'])
            if 'daily_stop_loss' in data:
                bot.daily_stop_loss = float(data['daily_stop_loss'])
            if 'daily_target' in data:
                bot.daily_target = float(data['daily_target'])
            
            # Start bot
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(bot.start_bot())
            
            if result['success']:
                active_technical_bots[user_id] = bot
                return jsonify(result), 200
            else:
                return jsonify(result), 400
        
    except Exception as e:
        logger.error(f"Error starting technical bot: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to start technical bot: {str(e)}'
        }), 500

@technical_bot_bp.route('/stop', methods=['POST'])
@jwt_required()
def stop_technical_bot():
    """Stop the technical trading bot"""
    try:
        user_id = get_jwt_identity()
        
        with bot_lock:
            if user_id not in active_technical_bots:
                return jsonify({
                    'success': False,
                    'error': 'No active technical bot found'
                }), 404
            
            bot = active_technical_bots[user_id]
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(bot.stop_bot())
            
            if result['success']:
                del active_technical_bots[user_id]
                return jsonify(result), 200
            else:
                return jsonify(result), 400
        
    except Exception as e:
        logger.error(f"Error stopping technical bot: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to stop technical bot: {str(e)}'
        }), 500

@technical_bot_bp.route('/status', methods=['GET'])
@jwt_required()
def get_technical_bot_status():
    """Get current technical bot status"""
    try:
        user_id = get_jwt_identity()
        
        if user_id not in active_technical_bots:
            return jsonify({
                'success': True,
                'data': {
                    'is_active': False,
                    'message': 'No active technical bot'
                }
            }), 200
        
        bot = active_technical_bots[user_id]
        
        # Get recent balance update
        try:
            balance_result = bot.deriv_service.get_real_time_balance(bot.api_token)
            if balance_result['success']:
                bot.account_balance = balance_result['data']['balance']
        except:
            pass
        
        status_data = {
            'is_active': bot.is_active,
            'current_strategy': bot.current_strategy,
            'account_balance': bot.account_balance,
            'starting_balance': bot.starting_balance,
            'total_profit': bot.total_profit,
            'total_loss': bot.total_loss,
            'net_pnl': bot.net_pnl,
            'daily_trades': bot.daily_trades,
            'winning_trades': bot.winning_trades,
            'losing_trades': bot.losing_trades,
            'win_rate': (bot.winning_trades / max(1, bot.daily_trades)) * 100,
            'open_trades_count': len(bot.open_trades),
            'open_trades': bot.open_trades[-10:],  # Last 10 open trades
            'base_stake': bot.base_stake,
            'daily_stop_loss': bot.daily_stop_loss,
            'daily_target': bot.daily_target,
            'current_indicators': {
                'rsi': round(bot.current_rsi, 2),
                'volatility': round(bot.current_volatility * 100, 2),
                'momentum': round(bot.current_momentum * 100, 2),
                'macd': round(bot.macd_value, 4),
                'bollinger_upper': round(bot.bollinger_upper, 4),
                'bollinger_lower': round(bot.bollinger_lower, 4)
            },
            'session_duration': str(datetime.now() - bot.start_time) if bot.start_time else None
        }
        
        return jsonify({
            'success': True,
            'data': status_data
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting technical bot status: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get status: {str(e)}'
        }), 500

@technical_bot_bp.route('/trade-history', methods=['GET'])
@jwt_required()
def get_trade_history():
    """Get technical bot trade history"""
    try:
        user_id = get_jwt_identity()
        limit = request.args.get('limit', 50, type=int)
        
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM technical_trades 
            WHERE user_id = ? 
            ORDER BY entry_time DESC 
            LIMIT ?
        ''', (user_id, limit))
        
        trades = []
        for row in cursor.fetchall():
            trades.append({
                'id': row[0],
                'contract_id': row[3],
                'strategy_name': row[4],
                'entry_time': row[5],
                'exit_time': row[6],
                'entry_price': row[7],
                'exit_price': row[8],
                'stake_amount': row[9],
                'profit_loss': row[10],
                'contract_type': row[11],
                'duration': row[12],
                'success': bool(row[13]),
                'rsi_value': row[14],
                'volatility': row[15],
                'momentum': row[16],
                'macd_value': row[17],
                'bollinger_position': row[18],
                'market_conditions': row[19]
            })
        
        conn.close()
        
        return jsonify({
            'success': True,
            'data': {
                'trades': trades,
                'total_count': len(trades)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting trade history: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get trade history: {str(e)}'
        }), 500

@technical_bot_bp.route('/update-settings', methods=['PUT'])
@jwt_required()
def update_technical_bot_settings():
    """Update technical bot settings"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if user_id not in active_technical_bots:
            return jsonify({
                'success': False,
                'error': 'No active technical bot found'
            }), 404
        
        bot = active_technical_bots[user_id]
        
        # Update settings
        if 'base_stake' in data:
            bot.base_stake = max(1.0, float(data['base_stake']))
        if 'daily_stop_loss' in data:
            bot.daily_stop_loss = float(data['daily_stop_loss'])
        if 'daily_target' in data:
            bot.daily_target = float(data['daily_target'])
        if 'risk_per_trade' in data:
            bot.risk_per_trade = float(data['risk_per_trade'])
        
        return jsonify({
            'success': True,
            'message': 'Settings updated successfully',
            'data': {
                'base_stake': bot.base_stake,
                'daily_stop_loss': bot.daily_stop_loss,
                'daily_target': bot.daily_target,
                'risk_per_trade': bot.risk_per_trade
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to update settings: {str(e)}'
        }), 500

@technical_bot_bp.route('/performance', methods=['GET'])
@jwt_required()
def get_performance():
    """Get performance metrics for the technical trading bot"""
    try:
        current_user_id = get_jwt_identity()
        
        with bot_lock:
            bot = active_technical_bots.get(current_user_id)
            
            if not bot:
                # Return default performance data if bot is not active
                return jsonify({
                    'success': True,
                    'data': {
                        'total_profit': 0.0,
                        'total_trades': 0,
                        'win_rate': 0.0,
                        'today_profit': 0.0,
                        'today_trades': 0,
                        'average_trade_duration': 0.0,
                        'max_consecutive_wins': 0,
                        'max_consecutive_losses': 0,
                        'sharpe_ratio': 0.0,
                        'max_drawdown': 0.0
                    }
                }), 200
            
            # Calculate performance metrics
            performance = bot.get_performance_metrics()
            
            return jsonify({
                'success': True,
                'data': performance
            }), 200
            
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get performance metrics: {str(e)}'
        }), 500

@technical_bot_bp.route('/strategy', methods=['POST'])
@jwt_required()
def change_strategy():
    """Change the trading strategy for the technical bot"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data or 'strategy' not in data:
            return jsonify({
                'success': False,
                'error': 'Strategy name is required'
            }), 400
            
        strategy_name = data['strategy']
        
        # List of available strategies
        available_strategies = [
            'adaptive_mean_reversion',
            'rsi_momentum_breakout',
            'bollinger_band_squeeze',
            'macd_histogram_divergence',
            'volatility_expansion_scalp',
            'tick_velocity_momentum',
            'support_resistance_bounce',
            'ema_crossover_micro',
            'williams_r_extreme',
            'stochastic_divergence',
            'volume_price_trend',
            'microtrend_reversal',
            'high_frequency_scalp',
            'neural_pattern_recognition',
            'adaptive_multi_timeframe'
        ]
        
        if strategy_name not in available_strategies:
            return jsonify({
                'success': False,
                'error': f'Invalid strategy. Available strategies: {", ".join(available_strategies)}'
            }), 400
        
        with bot_lock:
            bot = active_technical_bots.get(current_user_id)
            
            if not bot:
                return jsonify({
                    'success': False,
                    'error': 'Technical trading bot is not active'
                }), 400
                
            # Change strategy
            old_strategy = bot.current_strategy
            bot.current_strategy = strategy_name
            
            # Log strategy change
            logger.info(f"Strategy changed from {old_strategy} to {strategy_name} for user {current_user_id}")
            
            return jsonify({
                'success': True,
                'message': f'Strategy changed to {strategy_name}',
                'data': {
                    'old_strategy': old_strategy,
                    'new_strategy': strategy_name,
                    'changed_at': datetime.now().isoformat()
                }
            }), 200
            
    except Exception as e:
        logger.error(f"Error changing strategy: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to change strategy: {str(e)}'
        }), 500

@technical_bot_bp.route('/strategies', methods=['GET'])
def get_available_strategies():
    """Get list of available trading strategies"""
    try:
        strategies = [
            {
                'value': 'adaptive_mean_reversion',
                'name': 'Adaptive Mean Reversion',
                'description': 'Uses adaptive thresholds for mean reversion trading',
                'timeframe': '1-5 minutes',
                'risk_level': 'Medium'
            },
            {
                'value': 'rsi_momentum_breakout',
                'name': 'RSI Momentum Breakout',
                'description': 'Combines RSI signals with momentum indicators',
                'timeframe': '1-3 minutes',
                'risk_level': 'Medium-High'
            },
            {
                'value': 'bollinger_band_squeeze',
                'name': 'Bollinger Band Squeeze',
                'description': 'Detects low volatility periods before breakouts',
                'timeframe': '2-10 minutes',
                'risk_level': 'Medium'
            },
            {
                'value': 'macd_histogram_divergence',
                'name': 'MACD Histogram Divergence',
                'description': 'Uses MACD histogram divergences for entries',
                'timeframe': '3-15 minutes',
                'risk_level': 'Medium-Low'
            },
            {
                'value': 'volatility_expansion_scalp',
                'name': 'Volatility Expansion Scalp',
                'description': 'Scalps during volatility expansion phases',
                'timeframe': '30 seconds - 2 minutes',
                'risk_level': 'High'
            },
            {
                'value': 'tick_velocity_momentum',
                'name': 'Tick Velocity Momentum',
                'description': 'Uses tick velocity for momentum detection',
                'timeframe': '30 seconds - 1 minute',
                'risk_level': 'High'
            },
            {
                'value': 'support_resistance_bounce',
                'name': 'Support Resistance Bounce',
                'description': 'Trades bounces off key support/resistance levels',
                'timeframe': '2-10 minutes',
                'risk_level': 'Medium'
            },
            {
                'value': 'ema_crossover_micro',
                'name': 'EMA Crossover Micro',
                'description': 'Micro-timeframe EMA crossover strategy',
                'timeframe': '1-3 minutes',
                'risk_level': 'Medium'
            },
            {
                'value': 'williams_r_extreme',
                'name': 'Williams R Extreme',
                'description': 'Uses Williams %R extreme readings',
                'timeframe': '1-5 minutes',
                'risk_level': 'Medium-High'
            },
            {
                'value': 'stochastic_divergence',
                'name': 'Stochastic Divergence',
                'description': 'Detects divergences between price and stochastic oscillator',
                'timeframe': '3-10 minutes',
                'risk_level': 'Medium'
            },
            {
                'value': 'volume_price_trend',
                'name': 'Volume Price Trend',
                'description': 'Uses volume-price relationships for signal confirmation',
                'timeframe': '2-8 minutes',
                'risk_level': 'Medium-Low'
            },
            {
                'value': 'microtrend_reversal',
                'name': 'Microtrend Reversal',
                'description': 'Captures very short-term trend reversals',
                'timeframe': '30 seconds - 2 minutes',
                'risk_level': 'High'
            },
            {
                'value': 'high_frequency_scalp',
                'name': 'High Frequency Scalp',
                'description': 'Exploits very short-term price inefficiencies',
                'timeframe': '10-60 seconds',
                'risk_level': 'Very High'
            },
            {
                'value': 'neural_pattern_recognition',
                'name': 'Neural Pattern Recognition',
                'description': 'AI-based pattern recognition for complex signals',
                'timeframe': '1-5 minutes',
                'risk_level': 'Medium-High'
            },
            {
                'value': 'adaptive_multi_timeframe',
                'name': 'Adaptive Multi-Timeframe',
                'description': 'Combines signals from multiple timeframes',
                'timeframe': '1-15 minutes',
                'risk_level': 'Medium'
            }
        ]
        
        return jsonify({
            'success': True,
            'data': strategies
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting strategies: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get strategies: {str(e)}'
        }), 500
