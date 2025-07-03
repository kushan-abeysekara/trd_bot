from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import openai
import numpy as np
from datetime import datetime
import json
import os

ai_bp = Blueprint('ai', __name__, url_prefix='/api/ai')

# Initialize OpenAI
openai.api_key = "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"

@ai_bp.route('/analyze-market', methods=['POST'])
@jwt_required()
def analyze_market():
    """AI-powered market analysis using ChatGPT"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        data_points = data.get('dataPoints', [])
        indicators = data.get('indicators', {})
        market_condition = data.get('marketCondition', 'neutral')
        
        if len(data_points) < 50:
            return jsonify({'error': 'Insufficient data points for analysis'}), 400
        
        # Prepare data for AI analysis
        analysis_data = prepare_market_data(data_points, indicators, market_condition)
        
        # Get AI analysis from ChatGPT
        ai_response = get_chatgpt_analysis(symbol, analysis_data)
        
        return jsonify({
            'analysis': ai_response,
            'timestamp': datetime.utcnow().isoformat(),
            'data_points_analyzed': len(data_points)
        }), 200
        
    except Exception as e:
        print(f"AI Analysis error: {str(e)}")
        return jsonify({'error': 'AI analysis failed', 'details': str(e)}), 500

@ai_bp.route('/trading-recommendation', methods=['POST'])
@jwt_required()
def get_trading_recommendation():
    """Get AI trading recommendations"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        data_points = data.get('dataPoints', [])
        
        if len(data_points) < 100:
            return jsonify({'error': 'Need at least 100 data points for reliable recommendations'}), 400
        
        # Analyze market patterns
        patterns = analyze_market_patterns(data_points)
        
        # Get AI recommendation
        recommendation = get_ai_recommendation(symbol, patterns, data_points)
        
        return jsonify(recommendation), 200
        
    except Exception as e:
        print(f"Trading recommendation error: {str(e)}")
        return jsonify({'error': 'Failed to generate recommendation', 'details': str(e)}), 500

def prepare_market_data(data_points, indicators, market_condition):
    """Prepare market data for AI analysis"""
    prices = [point['price'] for point in data_points[-100:]]  # Last 100 points
    
    # Calculate additional technical indicators
    price_changes = []
    for i in range(1, len(prices)):
        change = (prices[i] - prices[i-1]) / prices[i-1] * 100
        price_changes.append(change)
    
    volatility = np.std(price_changes) if price_changes else 0
    avg_change = np.mean(price_changes) if price_changes else 0
    
    return {
        'current_price': prices[-1] if prices else 0,
        'price_range': {'min': min(prices), 'max': max(prices)} if prices else {},
        'volatility': volatility,
        'average_change': avg_change,
        'trend_direction': 'up' if avg_change > 0 else 'down',
        'market_condition': market_condition,
        'indicators': indicators,
        'recent_prices': prices[-20:]  # Last 20 prices for pattern analysis
    }

def get_chatgpt_analysis(symbol, analysis_data):
    """Get market analysis from ChatGPT"""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=openai.api_key)
        
        prompt = f"""
        As an expert financial analyst, analyze the following market data for {symbol}:

        Current Price: {analysis_data['current_price']:.2f}
        Price Range: {analysis_data['price_range']['min']:.2f} - {analysis_data['price_range']['max']:.2f}
        Volatility: {analysis_data['volatility']:.4f}
        Average Change: {analysis_data['average_change']:.4f}%
        Market Condition: {analysis_data['market_condition']}
        Trend Direction: {analysis_data['trend_direction']}

        Technical Indicators:
        - RSI: {analysis_data['indicators'].get('rsi', 'N/A')}
        - SMA20: {analysis_data['indicators'].get('sma20', 'N/A')}
        - SMA50: {analysis_data['indicators'].get('sma50', 'N/A')}
        - Volatility: {analysis_data['indicators'].get('volatility', 'N/A')}

        Recent Price Pattern: {analysis_data['recent_prices'][-10:]}

        Please provide:
        1. Market sentiment analysis
        2. Key support and resistance levels
        3. Risk assessment (Low/Medium/High)
        4. Best trading strategy for current conditions
        5. Recommended contract types and timeframes
        6. Confidence level (1-100%)

        Keep the response concise and actionable.
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert financial market analyst specializing in binary options and derivative trading."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"ChatGPT API error: {str(e)}")
        return generate_fallback_analysis(analysis_data)

def generate_fallback_analysis(analysis_data):
    """Generate fallback analysis when ChatGPT is unavailable"""
    volatility = analysis_data['volatility']
    avg_change = analysis_data['average_change']
    
    if volatility > 2.0:
        market_sentiment = "High volatility detected - expect significant price movements"
        risk_level = "High"
        strategy = "Consider boundary options or short-term contracts"
    elif volatility < 0.5:
        market_sentiment = "Low volatility - market in consolidation phase"
        risk_level = "Low"
        strategy = "Asian options or longer-term rise/fall contracts recommended"
    else:
        market_sentiment = "Moderate volatility - balanced market conditions"
        risk_level = "Medium"
        strategy = "Standard rise/fall contracts with medium timeframes"
    
    confidence = min(85, max(60, 70 + abs(avg_change) * 10))
    
    return f"""
    Market Sentiment: {market_sentiment}
    Risk Level: {risk_level}
    Recommended Strategy: {strategy}
    Trend Bias: {'Bullish' if avg_change > 0 else 'Bearish'}
    Confidence Level: {confidence:.0f}%
    
    Technical Summary: Based on recent price action, the market shows {analysis_data['trend_direction']} bias with {analysis_data['market_condition']} conditions.
    """

def analyze_market_patterns(data_points):
    """Analyze market patterns for trading recommendations"""
    prices = [point['price'] for point in data_points]
    
    # Pattern recognition
    patterns = {
        'trend_strength': calculate_trend_strength(prices),
        'support_resistance': find_support_resistance(prices),
        'momentum': calculate_momentum(prices),
        'volatility_clusters': detect_volatility_clusters(prices)
    }
    
    return patterns

def calculate_trend_strength(prices):
    """Calculate trend strength"""
    if len(prices) < 20:
        return 0
    
    recent_prices = prices[-20:]
    slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
    correlation = np.corrcoef(range(len(recent_prices)), recent_prices)[0, 1]
    
    return abs(slope * correlation)

def find_support_resistance(prices):
    """Find support and resistance levels"""
    if len(prices) < 50:
        return {'support': min(prices), 'resistance': max(prices)}
    
    # Simple approach: use recent highs and lows
    recent_prices = prices[-50:]
    support = np.percentile(recent_prices, 20)
    resistance = np.percentile(recent_prices, 80)
    
    return {'support': support, 'resistance': resistance}

def calculate_momentum(prices):
    """Calculate price momentum"""
    if len(prices) < 10:
        return 0
    
    short_term = np.mean(prices[-5:])
    medium_term = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
    
    return (short_term - medium_term) / medium_term

def detect_volatility_clusters(prices):
    """Detect volatility clustering"""
    if len(prices) < 20:
        return False
    
    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    volatilities = [abs(r) for r in returns]
    
    # Check if recent volatility is higher than average
    recent_vol = np.mean(volatilities[-10:])
    avg_vol = np.mean(volatilities)
    
    return recent_vol > avg_vol * 1.5

def get_ai_recommendation(symbol, patterns, data_points):
    """Generate AI-based trading recommendation"""
    trend_strength = patterns['trend_strength']
    momentum = patterns['momentum']
    support_resistance = patterns['support_resistance']
    
    current_price = data_points[-1]['price']
    
    # Determine best contract type
    if trend_strength > 0.5:
        if momentum > 0:
            contract_type = 'rise_fall'
            direction = 'rise'
            confidence = min(90, 70 + trend_strength * 20)
        else:
            contract_type = 'rise_fall'
            direction = 'fall'
            confidence = min(90, 70 + trend_strength * 20)
    elif abs(momentum) < 0.01:
        contract_type = 'asians'
        direction = 'neutral'
        confidence = 75
    else:
        contract_type = 'touch_no_touch'
        direction = 'touch' if abs(current_price - support_resistance['resistance']) < abs(current_price - support_resistance['support']) else 'no_touch'
        confidence = 65
    
    return {
        'contract_type': contract_type,
        'direction': direction,
        'confidence': confidence,
        'entry_price': current_price,
        'support_level': support_resistance['support'],
        'resistance_level': support_resistance['resistance'],
        'risk_level': 'low' if confidence > 80 else 'medium' if confidence > 60 else 'high',
        'recommended_duration': '3-5 minutes' if trend_strength > 0.3 else '5-10 minutes',
        'analysis_summary': f"Trend strength: {trend_strength:.2f}, Momentum: {momentum:.4f}"
    }

@ai_bp.route('/market-prediction', methods=['POST'])
@jwt_required()
def get_market_prediction():
    """Get AI market predictions"""
    try:
        data = request.get_json()
        # Implement market prediction logic here
        return jsonify({'prediction': 'Feature under development'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
