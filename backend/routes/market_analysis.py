from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime
import json
import asyncio
from threading import Thread
import time

from services.market_analyzer import market_analyzer

market_analysis_bp = Blueprint('market_analysis', __name__, url_prefix='/api/market-analysis')

@market_analysis_bp.route('/analyze', methods=['POST'])
@jwt_required()
def analyze_market_data():
    """Analyze market data in real-time"""
    try:
        data = request.get_json()
        
        if not data or 'price_data' not in data:
            return jsonify({'error': 'Price data is required'}), 400
        
        price_data = data['price_data']
        symbol = data.get('symbol', 'default')
        
        # Add price data to analyzer
        for price_point in price_data:
            if isinstance(price_point, dict) and 'price' in price_point:
                timestamp = datetime.fromisoformat(price_point.get('timestamp', datetime.utcnow().isoformat()))
                market_analyzer.add_price_data(price_point['price'], timestamp)
            elif isinstance(price_point, (int, float)):
                market_analyzer.add_price_data(float(price_point))
        
        # Get analysis results
        analysis = market_analyzer.get_latest_analysis()
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@market_analysis_bp.route('/real-time/<symbol>', methods=['POST'])
@jwt_required()
def start_real_time_analysis(symbol):
    """Start real-time analysis for a symbol"""
    try:
        data = request.get_json()
        current_price = data.get('current_price')
        
        if current_price:
            market_analyzer.add_price_data(float(current_price))
        
        analysis = market_analyzer.get_latest_analysis()
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'real_time': True,
            'symbol': symbol
        }, 200)
        
    except Exception as e:
        return jsonify({'error': f'Real-time analysis failed: {str(e)}'}), 500

@market_analysis_bp.route('/predictions', methods=['GET'])
@jwt_required()
def get_predictions():
    """Get current market predictions"""
    try:
        analysis = market_analyzer.get_latest_analysis()
        
        if not analysis:
            return jsonify({'error': 'No analysis data available'}), 404
        
        return jsonify({
            'success': True,
            'predictions': analysis.get('predictions', {}),
            'technical_indicators': analysis.get('technical_indicators', {}),
            'market_sentiment': analysis.get('market_sentiment', 'neutral'),
            'timestamp': analysis.get('timestamp')
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to get predictions: {str(e)}'}), 500

@market_analysis_bp.route('/digit-analysis', methods=['GET'])
@jwt_required()
def get_digit_analysis():
    """Get digit-specific analysis and predictions"""
    try:
        analysis = market_analyzer.get_latest_analysis()
        
        if not analysis:
            return jsonify({'error': 'No analysis data available'}), 404
        
        predictions = analysis.get('predictions', {})
        
        return jsonify({
            'success': True,
            'current_digit': analysis.get('current_digit'),
            'future_digits': predictions.get('future_digits', []),
            'pattern_analysis': predictions.get('pattern_analysis', {}),
            'frequency_analysis': predictions.get('frequency_analysis', {}),
            'timestamp': analysis.get('timestamp')
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to get digit analysis: {str(e)}'}), 500

@market_analysis_bp.route('/chatgpt-analysis', methods=['GET'])
@jwt_required()
def get_chatgpt_analysis():
    """Get ChatGPT market analysis"""
    try:
        analysis = market_analyzer.get_latest_analysis()
        
        if not analysis:
            return jsonify({'error': 'No analysis data available'}), 404
        
        return jsonify({
            'success': True,
            'chatgpt_analysis': analysis.get('chatgpt_analysis', 'Analysis in progress...'),
            'market_sentiment': analysis.get('market_sentiment'),
            'timestamp': analysis.get('timestamp')
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to get ChatGPT analysis: {str(e)}'}), 500

@market_analysis_bp.route('/trading-recommendation', methods=['POST'])
@jwt_required()
def get_trading_recommendation():
    """Get AI trading recommendations"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'default')
        
        analysis = market_analyzer.get_latest_analysis()
        
        if not analysis:
            return jsonify({'error': 'No analysis data available'}), 404
        
        # Generate trading recommendation based on analysis
        recommendation = generate_trading_recommendation(analysis)
        
        # Add trading bot status
        from services.trading_bot import trading_bot
        bot_status = trading_bot.get_status()
        
        return jsonify({
            'success': True,
            'recommendation': recommendation,
            'symbol': symbol,
            'bot_status': {
                'is_running': bot_status['is_running'],
                'strategy_status': bot_status['strategy_status'],
                'current_strategy': bot_status['current_strategy']
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate recommendation: {str(e)}'}), 500

def generate_trading_recommendation(analysis):
    """Generate trading recommendation from analysis"""
    predictions = analysis.get('predictions', {})
    indicators = analysis.get('technical_indicators', {})
    sentiment = analysis.get('market_sentiment', 'neutral')
    
    # Contract type recommendation
    volatility = indicators.get('volatility', 0)
    rsi = indicators.get('rsi', 50)
    
    if volatility > 0.02:
        contract_type = 'touch_no_touch'
        risk_level = 'medium'
    elif volatility < 0.005:
        contract_type = 'asians'
        risk_level = 'low'
    else:
        contract_type = 'rise_fall'
        risk_level = 'low'
    
    # Direction recommendation
    price_movement = predictions.get('price_movement', {})
    direction = price_movement.get('direction', 'neutral')
    
    if direction == 'neutral':
        if rsi > 70:
            direction = 'fall'
        elif rsi < 30:
            direction = 'rise'
    
    # Confidence calculation
    confidence = price_movement.get('confidence', 50)
    
    # Add Adaptive Mean Reversion strategy conditions
    strategy_conditions = {
        'rsi_in_range': 48 <= rsi <= 52,
        'volatility_optimal': 1.0 <= (volatility * 100) <= 1.5,
        'low_momentum': abs(indicators.get('momentum', 0) * 100) < 0.2,
        'flat_macd': abs(indicators.get('macd', {}).get('macd', 0)) <= 0.1
    }
    
    # Check if conditions are met for Adaptive Mean Reversion
    if all(strategy_conditions.values()):
        recommended_strategy = "Adaptive Mean Reversion Rebound"
        strategy_confidence = 85
    else:
        recommended_strategy = "Standard Technical Analysis"
        strategy_confidence = confidence
    
    return {
        'contract_type': contract_type,
        'direction': direction,
        'confidence': strategy_confidence,
        'risk_level': risk_level,
        'duration': '5-7 seconds' if recommended_strategy == "Adaptive Mean Reversion Rebound" else '2-5 minutes',
        'reasoning': f"Based on {sentiment} market sentiment and {recommended_strategy}",
        'strategy': recommended_strategy,
        'strategy_conditions': strategy_conditions,
        'future_digits': predictions.get('future_digits', [])[:3],
        'pattern_insights': predictions.get('pattern_analysis', {})
    }

@market_analysis_bp.route('/status', methods=['GET'])
def get_analysis_status():
    """Get analysis service status"""
    return jsonify({
        'status': 'active',
        'analyzer_ready': True,
        'data_points': len(market_analyzer.price_data),
        'last_analysis': market_analyzer.last_analysis.get('timestamp') if market_analyzer.last_analysis else None
    }), 200
