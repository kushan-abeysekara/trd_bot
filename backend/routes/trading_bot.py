from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from services.trading_bot_engine import TradingBotEngine
from services.self_learning_engine import SelfLearningEngine
import logging

logger = logging.getLogger(__name__)
bot_bp = Blueprint('bot', __name__, url_prefix='/api/bot')

# Store bot instances per user
bot_instances = {}
learning_engine = SelfLearningEngine()

@bot_bp.route('/start', methods=['POST'])
@jwt_required()
async def start_bot():
    """Start the trading bot"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        api_token = data.get('api_token')
        account_type = data.get('account_type', 'demo')
        
        if not api_token:
            return jsonify({'success': False, 'message': 'API token required'}), 400
        
        # Create or get bot instance
        if user_id not in bot_instances:
            bot_instances[user_id] = TradingBotEngine(user_id, api_token, account_type)
        
        bot = bot_instances[user_id]
        result = await bot.start_trading()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error starting bot: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@bot_bp.route('/stop', methods=['POST'])
@jwt_required()
async def stop_bot():
    """Stop the trading bot"""
    try:
        user_id = get_jwt_identity()
        
        if user_id in bot_instances:
            bot = bot_instances[user_id]
            result = await bot.stop_trading()
            return jsonify(result)
        else:
            return jsonify({'success': False, 'message': 'Bot not found'}), 404
            
    except Exception as e:
        logger.error(f"Error stopping bot: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@bot_bp.route('/status', methods=['GET'])
@jwt_required()
def get_bot_status():
    """Get current bot status"""
    try:
        user_id = get_jwt_identity()
        
        if user_id in bot_instances:
            bot = bot_instances[user_id]
            status = bot.get_advanced_bot_status()
            return jsonify({'success': True, 'data': status})
        else:
            return jsonify({'success': True, 'data': {'is_active': False}})
            
    except Exception as e:
        logger.error(f"Error getting bot status: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@bot_bp.route('/live-status', methods=['GET'])
@jwt_required()
def get_live_trading_status():
    """Get real-time trading status"""
    try:
        user_id = get_jwt_identity()
        
        if user_id in bot_instances:
            bot = bot_instances[user_id]
            status = bot.get_live_trading_status()
            return jsonify({'success': True, 'data': status})
        else:
            return jsonify({'success': True, 'data': {'is_active': False}})
            
    except Exception as e:
        logger.error(f"Error getting live status: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@bot_bp.route('/active-trades', methods=['GET'])
@jwt_required()
def get_active_trades():
    """Get currently active trades"""
    try:
        user_id = get_jwt_identity()
        
        if user_id in bot_instances:
            bot = bot_instances[user_id]
            trades = []
            
            for contract_id, trade_info in bot.current_trades.items():
                trades.append({
                    'contract_id': contract_id,
                    'contract_type': trade_info['contract_type'],
                    'action': trade_info['action'],
                    'confidence': trade_info['confidence'],
                    'stake_amount': trade_info['trade_result'].stake_amount,
                    'start_time': trade_info['start_time'].isoformat(),
                    'duration': trade_info['trade_result'].duration
                })
            
            return jsonify({'success': True, 'data': trades})
        else:
            return jsonify({'success': True, 'data': []})
            
    except Exception as e:
        logger.error(f"Error getting active trades: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@bot_bp.route('/market-data', methods=['GET'])
@jwt_required()
def get_current_market_data():
    """Get current market data"""
    try:
        user_id = get_jwt_identity()
        
        if user_id in bot_instances:
            bot = bot_instances[user_id]
            market_data = bot._get_enhanced_market_data()
            return jsonify({'success': True, 'data': market_data})
        else:
            return jsonify({'success': False, 'message': 'Bot not initialized'}), 404
            
    except Exception as e:
        logger.error(f"Error getting market data: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@bot_bp.route('/ai-predictions', methods=['GET'])
@jwt_required()
def get_ai_predictions():
    """Get AI predictions for all contract types"""
    try:
        user_id = get_jwt_identity()
        
        if user_id in bot_instances:
            bot = bot_instances[user_id]
            market_data = bot._get_enhanced_market_data()
            
            if not market_data:
                return jsonify({'success': False, 'message': 'Market data unavailable'}), 503
            
            predictions = {}
            
            # Get predictions for each contract type
            from services.ml_strategies import ContractType, TradingMode
            
            for contract_type in ContractType:
                try:
                    signal = bot.ml_strategy_manager.get_trading_signal_with_predictions(
                        contract_type, TradingMode.MODE_A, market_data
                    )
                    
                    if signal:
                        predictions[contract_type.value] = {
                            'direction': signal.get('action', 'neutral'),
                            'confidence': signal.get('confidence', 0.5),
                            'predicted_movement': signal.get('predicted_movement'),
                            'future_predictions': signal.get('future_predictions')
                        }
                    else:
                        predictions[contract_type.value] = {
                            'direction': 'neutral',
                            'confidence': 0.5
                        }
                        
                except Exception as e:
                    logger.error(f"Error getting prediction for {contract_type.value}: {str(e)}")
                    predictions[contract_type.value] = {
                        'direction': 'neutral',
                        'confidence': 0.5,
                        'error': str(e)
                    }
            
            return jsonify({'success': True, 'data': predictions})
        else:
            return jsonify({'success': False, 'message': 'Bot not initialized'}), 404
            
    except Exception as e:
        logger.error(f"Error getting AI predictions: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@bot_bp.route('/ml/retrain', methods=['POST'])
@jwt_required()
def retrain_ml_models():
    """Trigger ML model retraining"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        contract_type = data.get('contract_type')  # Optional: retrain specific contract
        
        if contract_type:
            # Retrain specific contract type
            result = learning_engine.continuous_learning_cycle(contract_type)
        else:
            # Retrain all contract types
            from services.ml_strategies import ContractType
            results = {}
            
            for ct in ContractType:
                results[ct.value] = learning_engine.continuous_learning_cycle(ct.value)
            
            result = {
                'status': 'success',
                'retrained_contracts': list(results.keys()),
                'results': results
            }
        
        return jsonify({'success': True, 'data': result})
        
    except Exception as e:
        logger.error(f"Error retraining models: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@bot_bp.route('/ml/status', methods=['GET'])
@jwt_required()
def get_ml_status():
    """Get ML learning system status"""
    try:
        status = learning_engine.get_learning_status()
        return jsonify({'success': True, 'data': status})
        
    except Exception as e:
        logger.error(f"Error getting ML status: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500
