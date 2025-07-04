from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime
import json
import time
from dataclasses import asdict
import logging
import time

logger = logging.getLogger(__name__)

from services.trading_bot import trading_bot
from models import User, db

trading_bot_bp = Blueprint('trading_bot', __name__, url_prefix='/api/trading-bot')

# Fast Trading Bot API Routes

@trading_bot_bp.route('/status', methods=['GET'])
@jwt_required()
def get_bot_status():
    """Get current trading bot status"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get bot status
        status = trading_bot.get_status()
        
        return jsonify({
            'success': True,
            'status': status
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to get bot status: {str(e)}'}), 500

@trading_bot_bp.route('/start', methods=['POST'])
@jwt_required()
def start_bot():
    """Start the trading bot"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Check if user has API token configured
        if not user.deriv_api_token:
            return jsonify({
                'success': False,
                'error': 'Please configure your Deriv API token first'
            }), 400
        
        # Initialize the required attributes if they don't exist
        if not hasattr(trading_bot, 'strategy_stats') or trading_bot.strategy_stats is None:
            trading_bot.strategy_stats = {}
            
            # Initialize with default values for all available strategies
            strategies = trading_bot.get_available_strategies()
            for strategy in strategies:
                strategy_id = strategy['id']
                trading_bot.strategy_stats[strategy_id] = {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0.0,
                    'profit': 0.0,
                    'status': 'Inactive',
                    'last_signal_time': None
                }
        
        if not hasattr(trading_bot, 'active_strategies') or trading_bot.active_strategies is None:
            trading_bot.active_strategies = {}
            
            # Initialize with default values (all disabled except for the first one)
            strategies = trading_bot.get_available_strategies()
            for i, strategy in enumerate(strategies):
                trading_bot.active_strategies[strategy['id']] = (i == 0)  # Enable only the first one
                
        # Check if the bot is already running
        status = trading_bot.get_status()
        if status.get('is_running'):
            return jsonify({
                'success': True,
                'message': 'Trading bot is already running'
            }), 200
            
        # Set the API token for the trading bot
        logger.info(f"Setting API token for trading bot (token length: {len(user.deriv_api_token)})")
        trading_bot.set_api_token(user.deriv_api_token)
        
        # Make sure bot is fully stopped before attempting to start again
        # This prevents state management issues
        trading_bot.stop()
        time.sleep(0.5)  # Brief delay to ensure cleanup
        
        # Start the bot
        result = trading_bot.start()
        
        if result['success']:
            logger.info("Trading bot started successfully")
        else:
            logger.error(f"Failed to start trading bot: {result['message']}")
        
        return jsonify({
            'success': result['success'],
            'message': result['message']
        }), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"Failed to start bot: {str(e)}")
        return jsonify({'success': False, 'error': f'Failed to start bot: {str(e)}'}), 500

@trading_bot_bp.route('/stop', methods=['POST'])
@jwt_required()
def stop_bot():
    """Stop the trading bot"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Check if bot is actually running
        status = trading_bot.get_status()
        if not status.get('is_running'):
            return jsonify({
                'success': True,
                'message': 'Trading bot is already stopped'
            }), 200
        
        # Stop the bot
        result = trading_bot.stop()
        
        return jsonify({
            'success': result['success'],
            'message': result['message']
        }), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"Failed to stop bot: {str(e)}")
        return jsonify({'success': False, 'error': f'Failed to stop bot: {str(e)}'}), 500

@trading_bot_bp.route('/active-trades', methods=['GET'])
@jwt_required()
def get_active_trades():
    """Get all active trades"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get active trades
        trades = trading_bot.get_active_trades()
        
        return jsonify({
            'success': True,
            'trades': trades
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to get active trades: {str(e)}'}), 500

@trading_bot_bp.route('/trade-history', methods=['GET'])
@jwt_required()
def get_trade_history():
    """Get trade history"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get limit from query parameters
        limit = request.args.get('limit', 50, type=int)
        
        # Log the request for debugging
        logger.info(f"Trade history requested by user {user_id}, limit={limit}")
        
        # Ensure trading_bot is properly initialized
        if not hasattr(trading_bot, 'trade_history'):
            logger.warning("Trade history not initialized in trading bot")
            return jsonify({
                'success': True,
                'trades': []
            }), 200
            
        # Get trade history with error handling
        try:
            trades = trading_bot.get_trade_history(limit)
            # Ensure we're returning a list even if the service returns None
            if trades is None:
                trades = []
                
            # Additional validation to ensure each trade has required fields
            validated_trades = []
            for trade in trades:
                # Ensure each trade has an ID
                if not trade.get('id'):
                    trade['id'] = f"trade_{int(time.time() * 1000)}_{len(validated_trades)}"
                
                # Ensure each trade has timestamps
                if not trade.get('entry_time'):
                    trade['entry_time'] = datetime.utcnow().isoformat()
                    
                validated_trades.append(trade)
                
            logger.info(f"Returning {len(validated_trades)} trade history records")
            return jsonify({
                'success': True,
                'trades': validated_trades
            }), 200
        except AttributeError as e:
            logger.error(f"AttributeError in trade history: {str(e)}")
            return jsonify({
                'success': True,
                'trades': []
            }), 200
            
    except Exception as e:
        logger.error(f"Error getting trade history: {str(e)}")
        return jsonify({
            'success': True,  # Return success even on error to prevent UI crashes
            'trades': [],
            'error': f'Failed to get trade history: {str(e)}'
        }), 200  # Return 200 instead of 500 to avoid UI errors
        return jsonify({'error': f'Failed to get trade history: {str(e)}'}), 500

@trading_bot_bp.route('/statistics', methods=['GET'])
@jwt_required()
def get_bot_statistics():
    """Get comprehensive trading bot statistics"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get statistics
        stats = trading_bot.get_statistics()
        
        return jsonify({
            'success': True,
            'statistics': stats
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to get statistics: {str(e)}'}), 500

@trading_bot_bp.route('/settings', methods=['GET'])
@jwt_required()
def get_bot_settings():
    """Get current bot settings"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get bot status which includes settings
        status = trading_bot.get_status()
        
        return jsonify({
            'success': True,
            'settings': status.get('settings', {})
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to get settings: {str(e)}'}), 500

@trading_bot_bp.route('/settings', methods=['PUT'])
@jwt_required()
def update_bot_settings():
    """Update bot settings"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get new settings from request
        new_settings = request.get_json()
        
        if not new_settings:
            return jsonify({'error': 'Settings data is required'}), 400
        
        # Update settings
        result = trading_bot.update_settings(new_settings)
        
        return jsonify({
            'success': result['success'],
            'message': result['message']
        }), 200 if result['success'] else 400
        
    except Exception as e:
        return jsonify({'error': f'Failed to update settings: {str(e)}'}), 500

@trading_bot_bp.route('/update-market-data', methods=['POST'])
@jwt_required()
def update_market_data():
    """Update market data for the bot"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get market data from request
        data = request.get_json()
        
        if not data or 'market' not in data or 'price' not in data:
            return jsonify({'error': 'Market and price data are required'}), 400
        
        # Update market data
        result = trading_bot.update_market_data(data['market'], data['price'])
        
        return jsonify({
            'success': result['success'],
            'message': result['message']
        }), 200 if result['success'] else 400
        
    except Exception as e:
        return jsonify({'error': f'Failed to update market data: {str(e)}'}), 500

@trading_bot_bp.route('/force-close/<trade_id>', methods=['POST'])
@jwt_required()
def force_close_trade(trade_id):
    """Force close a specific trade"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Force close the trade
        result = trading_bot.force_close_trade(trade_id)
        
        return jsonify({
            'success': result['success'],
            'message': result['message']
        }), 200 if result['success'] else 400
        
    except Exception as e:
        return jsonify({'error': f'Failed to force close trade: {str(e)}'}), 500

@trading_bot_bp.route('/strategies', methods=['GET'])
@jwt_required()
def get_strategies():
    """Get available trading strategies"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get available strategies
        strategies = trading_bot.get_available_strategies()
        
        return jsonify({
            'success': True,
            'strategies': strategies
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to get strategies: {str(e)}'}), 500
        
@trading_bot_bp.route('/strategies/<int:strategy_id>', methods=['GET'])
@jwt_required()
def get_strategy_details(strategy_id):
    """Get details for a specific strategy"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get strategy details
        strategy = trading_bot.get_strategy_details(strategy_id)
        
        if not strategy:
            return jsonify({'error': f'Strategy with ID {strategy_id} not found'}), 404
        
        return jsonify({
            'success': True,
            'strategy': strategy
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to get strategy details: {str(e)}'}), 500

@trading_bot_bp.route('/set-strategy/<int:strategy_id>', methods=['POST'])
@jwt_required()
def set_strategy(strategy_id):
    """Set the active trading strategy"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Set the strategy
        result = trading_bot.set_strategy(strategy_id)
        
        return jsonify({
            'success': result['success'],
            'message': result['message']
        }), 200 if result['success'] else 400
        
    except Exception as e:
        return jsonify({'error': f'Failed to set strategy: {str(e)}'}), 500

@trading_bot_bp.route('/strategies-status', methods=['GET'])
@jwt_required()
def get_strategies_status():
    """Get status of all strategies"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get active strategies and their stats
        strategies_status = {
            'active_strategies': trading_bot.active_strategies,
            'strategy_stats': trading_bot.get_strategy_stats(),
            'available_strategies': trading_bot.get_available_strategies()
        }
        
        # Add active trades by strategy
        strategy_active_trades = {}
        for trade_id, trade in trading_bot.active_trades.items():
            # Extract strategy_id from the trade (assuming it's in the strategy field format "Strategy Name: Reason")
            strategy_name = trade.strategy.split(':')[0].strip() if ':' in trade.strategy else trade.strategy
            # Find strategy id by name
            strategy_id = None
            for s in trading_bot.get_available_strategies():
                if s['name'] == strategy_name:
                    strategy_id = s['id']
                    break
            
            if strategy_id:
                if strategy_id not in strategy_active_trades:
                    strategy_active_trades[strategy_id] = []
                strategy_active_trades[strategy_id].append(asdict(trade))
        
        strategies_status['strategy_active_trades'] = strategy_active_trades
        
        return jsonify({
            'success': True,
            'data': strategies_status
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to get strategy status: {str(e)}'}), 500

@trading_bot_bp.route('/strategy/<int:strategy_id>/toggle', methods=['POST'])
@jwt_required()
def toggle_strategy(strategy_id):
    """Enable or disable a specific strategy"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get active status from request
        data = request.get_json()
        active = data.get('active', True)  # Default to enabling strategy
        
        # Toggle strategy
        result = trading_bot.set_strategy_status(strategy_id, active)
        
        return jsonify({
            'success': result['success'],
            'message': result['message']
        }), 200 if result['success'] else 400
        
    except Exception as e:
        return jsonify({'error': f'Failed to toggle strategy: {str(e)}'}), 500

@trading_bot_bp.route('/strategy-trades/<int:strategy_id>', methods=['GET'])
@jwt_required()
def get_strategy_trades(strategy_id):
    """Get trades for a specific strategy"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get trades for this strategy
        trades = trading_bot.get_strategy_trades(strategy_id)
        
        return jsonify({
            'success': True,
            'trades': trades
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to get strategy trades: {str(e)}'}), 500

@trading_bot_bp.route('/all-strategy-trades', methods=['GET'])
@jwt_required()
def get_all_strategy_trades():
    """Get trades for all strategies"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get all trades grouped by strategy
        trades = trading_bot.get_strategy_trades()
        
        return jsonify({
            'success': True,
            'strategy_trades': trades
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to get all strategy trades: {str(e)}'}), 500

@trading_bot_bp.route('/strategy-performance', methods=['GET'])
@jwt_required()
def get_strategy_performance():
    """Get performance summary for all strategies"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
                
        # Get strategy statistics
        strategy_stats = trading_bot.get_strategy_stats()
        
        # Get all available strategies for reference
        all_strategies = {s['id']: s for s in trading_bot.get_available_strategies()}
        
        # Build performance data
        performance = []
        for strategy_id, stats in strategy_stats.items():
            if strategy_id in all_strategies:
                strategy_info = all_strategies[strategy_id]
                performance.append({
                    'id': strategy_id,
                    'name': strategy_info['name'],
                    'description': strategy_info['description'],
                    'risk_level': strategy_info['risk_level'],
                    'timeframe': strategy_info['timeframe'],
                    'active': trading_bot.active_strategies.get(strategy_id, False),
                    'status': stats['status'],
                    'trades': stats['trades'],
                    'wins': stats['wins'],
                    'losses': stats['losses'],
                    'win_rate': stats['win_rate'],
                    'profit': stats['profit'],
                    'last_signal_time': stats['last_signal_time'].isoformat() if stats['last_signal_time'] else None
                })
        
        # Sort by win rate (descending)
        performance.sort(key=lambda x: x['win_rate'], reverse=True)
                
        return jsonify({
            'success': True,
            'performance': performance,
            'bot_status': trading_bot.get_status()
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to get strategy performance: {str(e)}'}), 500

@trading_bot_bp.route('/test', methods=['GET'])
def test():
    """Test endpoint for trading bot routes"""
    return jsonify({
        'message': 'Trading bot API routes working',
        'bot_status': trading_bot.get_status()
    }), 200
