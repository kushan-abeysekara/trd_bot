from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime
import json

from services.trading_bot import trading_bot
from models import User, db

trading_bot_bp = Blueprint('trading_bot', __name__, url_prefix='/api/trading-bot')

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
            return jsonify({'error': 'Please configure your Deriv API token first'}), 400
        
        # Set the API token for the trading bot
        trading_bot.set_api_token(user.deriv_api_token)
        
        # Start the bot
        result = trading_bot.start()
        
        return jsonify({
            'success': result['success'],
            'message': result['message']
        }), 200 if result['success'] else 400
        
    except Exception as e:
        return jsonify({'error': f'Failed to start bot: {str(e)}'}), 500

@trading_bot_bp.route('/stop', methods=['POST'])
@jwt_required()
def stop_bot():
    """Stop the trading bot"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Stop the bot
        result = trading_bot.stop()
        
        return jsonify({
            'success': result['success'],
            'message': result['message']
        }), 200 if result['success'] else 400
        
    except Exception as e:
        return jsonify({'error': f'Failed to stop bot: {str(e)}'}), 500

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
        
        # Get trade history
        trades = trading_bot.get_trade_history(limit)
        
        return jsonify({
            'success': True,
            'trades': trades
        }), 200
        
    except Exception as e:
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

@trading_bot_bp.route('/test', methods=['GET'])
def test():
    """Test endpoint for trading bot routes"""
    return jsonify({
        'message': 'Trading bot API routes working',
        'bot_status': trading_bot.get_status()
    }), 200
