from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
import logging
import asyncio
from datetime import datetime
import sqlite3
import json
from threading import Lock

from services.trading_bot_engine import TradingBotEngine, TradingSettings, ContractType
from services.ml_strategies import MLStrategyManager
from models.user import User

logger = logging.getLogger(__name__)

# Blueprint for trading bot routes
trading_bot_bp = Blueprint('trading_bot', __name__, url_prefix='/api/trading-bot')

# Global bot instances (one per user)
active_bots = {}
bot_lock = Lock()

@trading_bot_bp.route('/start', methods=['POST'])
@jwt_required()
def start_trading_bot():
    """Start the AI trading bot"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}  # Ensure data is not None
        
        # Validate required parameters
        api_token = data.get('api_token')
        if not api_token:
            # Try to get from user profile
            try:
                conn = sqlite3.connect('trading_bot.db')
                cursor = conn.cursor()
                cursor.execute("SELECT deriv_api_token FROM users WHERE id = ?", (user_id,))
                user_data = cursor.fetchone()
                conn.close()
                
                if not user_data or not user_data[0]:
                    return jsonify({
                        'error': 'API token is required. Please configure your Deriv API token first.',
                        'code': 'MISSING_API_TOKEN'
                    }), 400
                api_token = user_data[0]
            except Exception as db_error:
                logger.error(f"Database error while fetching API token: {str(db_error)}")
                return jsonify({
                    'error': 'Failed to retrieve API token. Please configure your Deriv API token.',
                    'code': 'API_TOKEN_RETRIEVAL_ERROR'
                }), 400
        
        account_type = data.get('account_type', 'demo')
        if account_type not in ['demo', 'real']:
            return jsonify({
                'error': 'Invalid account type. Must be "demo" or "real".',
                'code': 'INVALID_ACCOUNT_TYPE'
            }), 400
        
        with bot_lock:
            # Check if bot is already running
            if user_id in active_bots and active_bots[user_id].is_active:
                return jsonify({'error': 'Trading bot is already active'}), 400
            
            # Get custom settings if provided
            settings_data = data.get('settings', {})
            settings = TradingSettings()
            
            # Update settings with user preferences
            for key, value in settings_data.items():
                if hasattr(settings, key):
                    setattr(settings, key, value)
            
            # Create and start bot
            bot = TradingBotEngine(user_id, api_token, account_type)
            bot.settings = settings
            
            # Start the bot
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(bot.start_trading())
            
            if result['success']:
                active_bots[user_id] = bot
                logger.info(f"Trading bot started for user {user_id}")
                
                return jsonify({
                    'success': True,
                    'message': 'AI Trading Bot started successfully',
                    'data': {
                        **result['data'],
                        'settings': settings.__dict__,
                        'bot_id': user_id
                    }
                }), 200
            else:
                return jsonify(result), 400
        
    except Exception as e:
        logger.error(f"Error starting trading bot: {str(e)}")
        return jsonify({
            'error': f'Failed to start trading bot: {str(e)}',
            'code': 'TRADING_BOT_START_ERROR'
        }), 500

@trading_bot_bp.route('/stop', methods=['POST'])
@jwt_required()
def stop_trading_bot():
    """Stop the AI trading bot"""
    try:
        user_id = get_jwt_identity()
        
        with bot_lock:
            if user_id not in active_bots:
                return jsonify({'error': 'No active trading bot found'}), 400
            
            bot = active_bots[user_id]
            
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(bot.stop_trading())
            
            if result['success']:
                del active_bots[user_id]
                logger.info(f"Trading bot stopped for user {user_id}")
                
                return jsonify({
                    'success': True,
                    'message': 'AI Trading Bot stopped successfully',
                    'data': result['data']
                }), 200
            else:
                return jsonify(result), 400
        
    except Exception as e:
        logger.error(f"Error stopping trading bot: {str(e)}")
        return jsonify({'error': f'Failed to stop trading bot: {str(e)}'}), 500

@trading_bot_bp.route('/status', methods=['GET'])
@jwt_required()
def get_bot_status():
    """Get current bot status and performance metrics"""
    try:
        user_id = get_jwt_identity()
        
        with bot_lock:
            if user_id not in active_bots:
                return jsonify({
                    'success': True,
                    'is_active': False,
                    'message': 'No active trading bot'
                }), 200
            
            bot = active_bots[user_id]
            status = bot.get_advanced_bot_status()
            
            return jsonify({
                'success': True,
                'data': status
            }), 200
        
    except Exception as e:
        logger.error(f"Error getting bot status: {str(e)}")
        return jsonify({'error': f'Failed to get bot status: {str(e)}'}), 500



@trading_bot_bp.route('/settings', methods=['GET'])
@jwt_required()
def get_bot_settings():
    """Get current bot settings"""
    try:
        user_id = get_jwt_identity()
        
        with bot_lock:
            if user_id not in active_bots:
                # Return default settings
                default_settings = TradingSettings()
                return jsonify({
                    'success': True,
                    'data': default_settings.__dict__
                }), 200
            
            bot = active_bots[user_id]
            
            return jsonify({
                'success': True,
                'data': bot.settings.__dict__
            }), 200
        
    except Exception as e:
        logger.error(f"Error getting bot settings: {str(e)}")
        return jsonify({'error': f'Failed to get bot settings: {str(e)}'}), 500

@trading_bot_bp.route('/settings', methods=['PUT'])
@jwt_required()
def update_bot_settings():
    """Update bot settings"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        with bot_lock:
            if user_id not in active_bots:
                return jsonify({'error': 'No active trading bot found'}), 400
            
            bot = active_bots[user_id]
            bot.update_settings(data)
            
            return jsonify({
                'success': True,
                'message': 'Bot settings updated successfully',
                'data': bot.settings.__dict__
            }), 200
        
    except Exception as e:
        logger.error(f"Error updating bot settings: {str(e)}")
        return jsonify({'error': f'Failed to update bot settings: {str(e)}'}), 500

@trading_bot_bp.route('/history', methods=['GET'])
@jwt_required()
def get_trading_history():
    """Get trading history for the bot"""
    try:
        user_id = get_jwt_identity()
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        contract_type = request.args.get('contract_type')
        
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()
        
        # Build query
        query = '''
            SELECT contract_id, contract_type, entry_price, exit_price, stake_amount, 
                   profit_loss, duration, strategy_used, timestamp, success,
                   ml_prediction_confidence, market_conditions_json
            FROM trade_results 
            WHERE user_id = ?
        '''
        params = [user_id]
        
        if contract_type:
            query += ' AND contract_type = ?'
            params.append(contract_type)
        
        query += ' ORDER BY timestamp DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Get total count
        count_query = 'SELECT COUNT(*) FROM trade_results WHERE user_id = ?'
        count_params = [user_id]
        if contract_type:
            count_query += ' AND contract_type = ?'
            count_params.append(contract_type)
        
        cursor.execute(count_query, count_params)
        total_count = cursor.fetchone()[0]
        
        conn.close()
        
        # Format results
        trades = []
        for row in rows:
            trade = {
                'contract_id': row[0],
                'contract_type': row[1],
                'entry_price': row[2],
                'exit_price': row[3],
                'stake_amount': row[4],
                'profit_loss': row[5],
                'duration': row[6],
                'strategy_used': row[7],
                'timestamp': row[8],
                'success': bool(row[9]),
                'ml_confidence': row[10],
                'market_conditions': json.loads(row[11]) if row[11] else {}
            }
            trades.append(trade)
        
        return jsonify({
            'success': True,
            'data': {
                'trades': trades,
                'total_count': total_count,
                'limit': limit,
                'offset': offset
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting trading history: {str(e)}")
        return jsonify({'error': f'Failed to get trading history: {str(e)}'}), 500

@trading_bot_bp.route('/sessions', methods=['GET'])
@jwt_required()
def get_trading_sessions():
    """Get trading sessions history"""
    try:
        user_id = get_jwt_identity()
        limit = request.args.get('limit', 50, type=int)
        
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, session_start, session_end, starting_balance, ending_balance,
                   total_pnl, total_trades, win_rate, strategy_used, account_type
            FROM trading_sessions 
            WHERE user_id = ?
            ORDER BY session_start DESC 
            LIMIT ?
        ''', (user_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        sessions = []
        for row in rows:
            session = {
                'id': row[0],
                'session_start': row[1],
                'session_end': row[2],
                'starting_balance': row[3],
                'ending_balance': row[4],
                'total_pnl': row[5],
                'total_trades': row[6],
                'win_rate': row[7],
                'strategy_used': row[8],
                'account_type': row[9],
                'session_duration': None  # Would calculate from start/end times
            }
            sessions.append(session)
        
        return jsonify({
            'success': True,
            'data': sessions
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting trading sessions: {str(e)}")
        return jsonify({'error': f'Failed to get trading sessions: {str(e)}'}), 500

@trading_bot_bp.route('/performance', methods=['GET'])
@jwt_required()
def get_performance_analytics():
    """Get detailed performance analytics"""
    try:
        user_id = get_jwt_identity()
        
        with bot_lock:
            if user_id in active_bots:
                bot = active_bots[user_id]
                contract_performance = bot.get_contract_performance_summary()
                ml_performance = bot.ml_strategy_manager.get_model_performance()
            else:
                # Get from database for inactive bots
                conn = sqlite3.connect('trading_bot.db')
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT contract_type, total_trades, winning_trades, total_profit, win_rate, avg_profit_per_trade
                    FROM strategy_performance 
                    WHERE user_id = ?
                ''', (user_id,))
                
                rows = cursor.fetchall()
                conn.close()
                
                contract_performance = {}
                for row in rows:
                    contract_performance[row[0]] = {
                        'total_trades': row[1],
                        'winning_trades': row[2],
                        'total_profit': row[3],
                        'win_rate': row[4],
                        'avg_profit_per_trade': row[5]
                    }
                
                ml_performance = {}
        
        # Calculate additional analytics
        analytics = {
            'contract_performance': contract_performance,
            'ml_model_performance': ml_performance,
            'summary': {
                'total_trades': sum(p['total_trades'] for p in contract_performance.values()),
                'total_profit': sum(p['total_profit'] for p in contract_performance.values()),
                'overall_win_rate': 0,
                'best_contract': None,
                'worst_contract': None
            }
        }
        
        # Calculate overall win rate
        total_trades = analytics['summary']['total_trades']
        total_wins = sum(p['winning_trades'] for p in contract_performance.values())
        if total_trades > 0:
            analytics['summary']['overall_win_rate'] = total_wins / total_trades
        
        # Find best and worst performing contracts
        if contract_performance:
            best = max(contract_performance.items(), key=lambda x: x[1]['total_profit'])
            worst = min(contract_performance.items(), key=lambda x: x[1]['total_profit'])
            analytics['summary']['best_contract'] = best[0]
            analytics['summary']['worst_contract'] = worst[0]
        
        return jsonify({
            'success': True,
            'data': analytics
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting performance analytics: {str(e)}")
        return jsonify({'error': f'Failed to get performance analytics: {str(e)}'}), 500

@trading_bot_bp.route('/ml-models/retrain', methods=['POST'])
@jwt_required()
def retrain_ml_models():
    """Manually trigger ML model retraining"""
    try:
        user_id = get_jwt_identity()
        
        with bot_lock:
            if user_id in active_bots:
                bot = active_bots[user_id]
                bot.ml_strategy_manager.train_models(force_retrain=True)
                message = 'ML models retrained for active bot'
            else:
                # Create temporary ML manager for training
                from services.ml_strategies import MLStrategyManager
                temp_ml_manager = MLStrategyManager()
                temp_ml_manager.train_models(force_retrain=True)
                message = 'All ML models retrained'
            
            return jsonify({
                'success': True,
                'message': message
            }), 200
        
    except Exception as e:
        logger.error(f"Error retraining ML models: {str(e)}")
        return jsonify({'error': f'Failed to retrain ML models: {str(e)}'}), 500

@trading_bot_bp.route('/ml-models/performance', methods=['GET'])
@jwt_required()
def get_ml_model_performance():
    """Get ML model performance metrics"""
    try:
        user_id = get_jwt_identity()
        
        with bot_lock:
            if user_id not in active_bots:
                return jsonify({'error': 'No active trading bot found'}), 400
            
            bot = active_bots[user_id]
            performance = bot.ml_strategy_manager.get_model_performance()
            
            return jsonify({
                'success': True,
                'data': performance
            }), 200
        
    except Exception as e:
        logger.error(f"Error getting ML model performance: {str(e)}")
        return jsonify({'error': f'Failed to get ML model performance: {str(e)}'}), 500

@trading_bot_bp.route('/test-signal', methods=['POST'])
@jwt_required()
def test_trading_signal():
    """Test trading signal generation without executing trade"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        with bot_lock:
            if user_id not in active_bots:
                return jsonify({'error': 'No active trading bot found'}), 400
            
            bot = active_bots[user_id]
            
            # Get current market data
            market_data = bot._get_market_data()
            if not market_data:
                return jsonify({'error': 'Unable to get market data'}), 500
            
            # Select contract type
            contract_type_str = data.get('contract_type')
            if contract_type_str:
                try:
                    contract_type = ContractType(contract_type_str)
                except ValueError:
                    return jsonify({'error': 'Invalid contract type'}), 400
            else:
                contract_type, _ = bot._select_best_contract_type(market_data)
            
            # Get trading signal
            signal = bot.ml_strategy_manager.get_trading_signal(
                contract_type, 
                bot.current_mode, 
                market_data
            )
            
            if signal:
                signal_data = {
                    'contract_type': contract_type.value,
                    'direction': signal.direction,
                    'confidence': signal.confidence,
                    'duration': signal.duration,
                    'entry_price': signal.entry_price,
                    'target_price': signal.target_price,
                    'stop_loss': signal.stop_loss,
                    'contract_params': signal.contract_specific_params,
                    'market_data': market_data,
                    'recommended_stake': bot._calculate_stake_amount()
                }
            else:
                signal_data = {
                    'error': 'No trading signal generated',
                    'market_data': market_data
                }
            
            return jsonify({
                'success': True,
                'data': signal_data
            }), 200
        
    except Exception as e:
        logger.error(f"Error testing trading signal: {str(e)}")
        return jsonify({'error': f'Failed to test trading signal: {str(e)}'}), 500

@trading_bot_bp.route('/bot-configs', methods=['GET'])
@jwt_required()
def get_bot_configurations():
    """Get available bot configurations and contract types"""
    try:
        configurations = {
            'contract_types': [
                {
                    'value': ct.value,
                    'name': ct.value.replace('_', ' ').title(),
                    'description': get_contract_description(ct)
                }
                for ct in ContractType
            ],
            'trading_modes': [
                {
                    'value': 'MODE_A',
                    'name': 'MA-RSI Trend Bot',
                    'description': 'Moving average and RSI trend following strategy'
                },
                {
                    'value': 'MODE_B',
                    'name': 'Price Action Bounce',
                    'description': 'Reversal strategy looking for price bounces'
                },
                {
                    'value': 'MODE_C',
                    'name': 'Random Entry Smart Exit',
                    'description': 'Random entry with intelligent exit management'
                }
            ],
            'risk_levels': ['low', 'medium', 'high'],
            'default_settings': TradingSettings().__dict__
        }
        
        return jsonify({
            'success': True,
            'data': configurations
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting bot configurations: {str(e)}")
        return jsonify({'error': f'Failed to get bot configurations: {str(e)}'}), 500

def get_contract_description(contract_type: ContractType) -> str:
    """Get description for contract type"""
    descriptions = {
        ContractType.RISE_FALL: "Predict if market will rise or fall at expiry",
        ContractType.TOUCH_NO_TOUCH: "Predict if market will touch or not touch a barrier",
        ContractType.IN_OUT: "Predict if market stays within or breaks out of boundaries",
        ContractType.ASIANS: "Predict if average price will be higher or lower",
        ContractType.DIGITS: "Predict the last digit of the market price",
        ContractType.RESET_CALL_PUT: "Call/Put with barrier reset functionality",
        ContractType.HIGH_LOW_TICKS: "Predict highest or lowest tick in sequence",
        ContractType.ONLY_UPS_DOWNS: "Predict consistent directional movement",
        ContractType.MULTIPLIERS: "Leveraged trading with stop loss control",
        ContractType.ACCUMULATORS: "Accumulate profits based on favorable movement"
    }
    return descriptions.get(contract_type, "Advanced derivative contract")
