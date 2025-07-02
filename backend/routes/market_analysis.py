from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime
import json
import asyncio
from threading import Thread
import time
import logging

from services.market_analyzer import market_analyzer

# Set up logging
logger = logging.getLogger(__name__)

market_analysis_bp = Blueprint('market_analysis', __name__, url_prefix='/api/market-analysis')

@market_analysis_bp.route('/analyze', methods=['POST'])
@jwt_required()
def analyze_market_data():
    """Analyze market data in real-time"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Request body is required',
                'code': 'MISSING_REQUEST_BODY'
            }), 400
        
        if 'price_data' not in data:
            return jsonify({
                'error': 'Price data is required in request body',
                'code': 'MISSING_PRICE_DATA',
                'expected_format': {
                    'price_data': [{'price': 123.45, 'timestamp': 'ISO_DATE'}],
                    'symbol': 'optional_symbol'
                }
            }), 400
        
        price_data = data['price_data']
        if not isinstance(price_data, list) or len(price_data) == 0:
            return jsonify({
                'error': 'Price data must be a non-empty array',
                'code': 'INVALID_PRICE_DATA_FORMAT'
            }), 400
        
        symbol = data.get('symbol', 'default')
        
        # Validate and process price data
        valid_prices = []
        for i, price_point in enumerate(price_data):
            try:
                if isinstance(price_point, dict) and 'price' in price_point:
                    price = float(price_point['price'])
                    timestamp = datetime.fromisoformat(price_point.get('timestamp', datetime.utcnow().isoformat()))
                    valid_prices.append({'price': price, 'timestamp': timestamp})
                    market_analyzer.add_price_data(price, timestamp)
                elif isinstance(price_point, (int, float)):
                    price = float(price_point)
                    valid_prices.append({'price': price, 'timestamp': datetime.utcnow()})
                    market_analyzer.add_price_data(price)
                else:
                    logger.warning(f"Invalid price point at index {i}: {price_point}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing price point at index {i}: {e}")
                continue
        
        if len(valid_prices) == 0:
            return jsonify({
                'error': 'No valid price data found',
                'code': 'NO_VALID_PRICE_DATA'
            }), 400
        
        # Get analysis results
        analysis = market_analyzer.get_latest_analysis()
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'symbol': symbol,
            'processed_points': len(valid_prices),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Market analysis error: {str(e)}")
        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'code': 'ANALYSIS_PROCESSING_ERROR'
        }), 500

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
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Real-time analysis failed: {str(e)}'}), 500

@market_analysis_bp.route('/predictions', methods=['GET'])
@jwt_required()
def get_predictions():
    """Get market predictions and forecasts"""
    try:
        analysis = market_analyzer.get_latest_analysis()
        
        if not analysis:
            return jsonify({'error': 'No analysis data available'}), 404
        
        predictions = analysis.get('predictions', {})
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'price_movement': predictions.get('price_movement', {}),
            'volatility_forecast': predictions.get('volatility_forecast', {}),
            'trend_analysis': predictions.get('trend_analysis', {}),
            'timestamp': analysis.get('timestamp')
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get predictions: {str(e)}")
        return jsonify({'error': f'Failed to get predictions: {str(e)}'}), 500

@market_analysis_bp.route('/digit-analysis', methods=['GET'])
@jwt_required()
def get_digit_analysis():
    """Get digit-specific analysis for digit trading"""
    try:
        analysis = market_analyzer.get_latest_analysis()
        
        if not analysis:
            return jsonify({'error': 'No analysis data available'}), 404
        
        digit_predictions = analysis.get('predictions', {}).get('future_digits', [])
        
        return jsonify({
            'success': True,
            'digit_predictions': digit_predictions[:10],  # Next 10 predictions
            'digit_patterns': analysis.get('digit_patterns', {}),
            'timestamp': analysis.get('timestamp')
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get digit analysis: {str(e)}")
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
        
        return jsonify({
            'success': True,
            'recommendation': recommendation,
            'symbol': symbol,
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
    
    return {
        'contract_type': contract_type,
        'direction': direction,
        'confidence': confidence,
        'risk_level': risk_level,
        'duration': '2-5 minutes' if volatility > 0.01 else '5-10 minutes',
        'reasoning': f"Based on {sentiment} market sentiment and technical analysis",
        'future_digits': predictions.get('future_digits', [])[:3],  # Next 3 predictions
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
