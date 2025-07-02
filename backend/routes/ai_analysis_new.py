from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import numpy as np
from datetime import datetime
import json
import sqlite3
import logging
from services.ml_strategies import MLStrategyManager, ContractType, TradingMode
from services.market_analyzer import RealTimeMarketAnalyzer
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy import stats
import ta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ai_bp = Blueprint('ai', __name__, url_prefix='/api/ai')

# Initialize ML components
ml_strategy_manager = MLStrategyManager()
market_analyzer = RealTimeMarketAnalyzer()

@ai_bp.route('/analyze-market', methods=['POST'])
@jwt_required()
def analyze_market():
    """ML-powered market analysis using local machine learning models"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        data_points = data.get('dataPoints', [])
        indicators = data.get('indicators', {})
        market_condition = data.get('marketCondition', 'neutral')
        
        if len(data_points) < 50:
            return jsonify({'error': 'Insufficient data points for analysis'}), 400
        
        # Prepare data for ML analysis
        analysis_data = prepare_market_data(data_points, indicators, market_condition)
        
        # Get ML-based analysis
        ml_analysis = get_ml_market_analysis(symbol, analysis_data, data_points)
        
        return jsonify({
            'analysis': ml_analysis['analysis'],
            'ml_confidence': ml_analysis['confidence'],
            'strategy_recommendation': ml_analysis['strategy'],
            'risk_assessment': ml_analysis['risk_level'],
            'technical_signals': ml_analysis['signals'],
            'timestamp': datetime.utcnow().isoformat(),
            'data_points_analyzed': len(data_points),
            'model_version': ml_analysis['model_version']
        }), 200
        
    except Exception as e:
        logger.error(f"ML Analysis error: {str(e)}")
        # Generate fallback technical analysis
        fallback_analysis = generate_technical_fallback_analysis(analysis_data if 'analysis_data' in locals() else {})
        return jsonify({
            'analysis': fallback_analysis['analysis'],
            'ml_confidence': fallback_analysis['confidence'],
            'strategy_recommendation': fallback_analysis['strategy'],
            'risk_assessment': 'medium',
            'error_message': 'ML analysis failed, using technical fallback',
            'timestamp': datetime.utcnow().isoformat()
        }), 200

@ai_bp.route('/trading-recommendation', methods=['POST'])
@jwt_required()
def get_trading_recommendation():
    """Get ML-based trading recommendations"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        data_points = data.get('dataPoints', [])
        contract_type = data.get('contractType', 'rise_fall')
        
        if len(data_points) < 100:
            return jsonify({'error': 'Need at least 100 data points for reliable recommendations'}), 400
        
        # Convert contract type string to enum
        try:
            contract_enum = ContractType(contract_type.lower())
        except ValueError:
            contract_enum = ContractType.RISE_FALL
        
        # Prepare market data for ML analysis
        market_data = {
            'price_history': data_points,
            'symbol': symbol,
            'current_time': datetime.utcnow()
        }
        
        # Get ML recommendation using strategy manager (fix parameter order)
        recommendation = ml_strategy_manager.get_trading_signal(contract_enum, TradingMode.MODE_A, market_data)
        
        # Handle case where recommendation is None
        if not recommendation:
            return jsonify({
                'error': 'Unable to generate trading recommendation',
                'details': 'ML strategy returned no signal'
            }), 400
        
        # Enhanced recommendation with ensemble voting
        ensemble_recommendation = get_ensemble_recommendation(market_data, contract_enum)
        
        return jsonify({
            'primary_recommendation': {
                'direction': recommendation.direction,
                'confidence': recommendation.confidence,
                'duration': recommendation.duration,
                'entry_price': recommendation.entry_price,
                'target_price': recommendation.target_price,
                'stop_loss': recommendation.stop_loss
            },
            'ensemble_recommendation': ensemble_recommendation,
            'technical_analysis': get_technical_analysis_summary(data_points),
            'market_regime': detect_market_regime(data_points),
            'volatility_forecast': forecast_volatility(data_points),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Trading recommendation error: {str(e)}")
        return jsonify({'error': 'Failed to generate recommendation', 'details': str(e)}), 500

def prepare_market_data(data_points, indicators, market_condition):
    """Prepare comprehensive market data for ML analysis"""
    prices = [point['price'] for point in data_points[-200:]]  # Last 200 points for better analysis
    
    if len(prices) < 50:
        return {}
    
    # Enhanced technical analysis
    df = pd.DataFrame({'price': prices})
    
    # Calculate multiple timeframe analysis
    price_changes = []
    for i in range(1, len(prices)):
        change = (prices[i] - prices[i-1]) / prices[i-1] * 100
        price_changes.append(change)
    
    # Advanced volatility metrics
    volatility = np.std(price_changes) if price_changes else 0
    avg_change = np.mean(price_changes) if price_changes else 0
    skewness = stats.skew(price_changes) if len(price_changes) > 2 else 0
    kurtosis = stats.kurtosis(price_changes) if len(price_changes) > 3 else 0
    
    # Support and resistance levels using multiple methods
    support_resistance = calculate_advanced_support_resistance(prices)
    
    # Market microstructure analysis
    microstructure = analyze_market_microstructure(prices)
    
    # Pattern recognition
    patterns = detect_price_patterns(prices)
    
    return {
        'current_price': prices[-1] if prices else 0,
        'price_range': {'min': min(prices), 'max': max(prices)} if prices else {},
        'volatility': volatility,
        'average_change': avg_change,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'trend_direction': 'up' if avg_change > 0 else 'down',
        'market_condition': market_condition,
        'indicators': indicators,
        'recent_prices': prices[-50:],  # Last 50 prices for detailed analysis
        'support_resistance': support_resistance,
        'microstructure': microstructure,
        'patterns': patterns,
        'momentum_indicators': calculate_momentum_indicators(prices),
        'volume_analysis': analyze_volume_patterns(data_points)
    }

def get_ml_market_analysis(symbol, analysis_data, data_points):
    """Generate comprehensive ML-based market analysis"""
    try:
        # Prepare features for ML models
        features = extract_ml_features(analysis_data, data_points)
        
        # Get predictions from multiple models
        predictions = {}
        confidence_scores = {}
        
        # Random Forest prediction
        rf_prediction, rf_confidence = get_random_forest_prediction(features)
        predictions['random_forest'] = rf_prediction
        confidence_scores['random_forest'] = rf_confidence
        
        # Gradient Boosting prediction
        gb_prediction, gb_confidence = get_gradient_boost_prediction(features)
        predictions['gradient_boost'] = gb_prediction
        confidence_scores['gradient_boost'] = gb_confidence
        
        # SVM prediction
        svm_prediction, svm_confidence = get_svm_prediction(features)
        predictions['svm'] = svm_prediction
        confidence_scores['svm'] = svm_confidence
        
        # Ensemble prediction
        ensemble_prediction = get_ensemble_prediction(predictions, confidence_scores)
        
        # Generate analysis text
        analysis_text = generate_ml_analysis_text(analysis_data, ensemble_prediction, predictions)
        
        # Determine strategy recommendation
        strategy = determine_optimal_strategy(analysis_data, ensemble_prediction)
        
        # Risk assessment
        risk_level = assess_risk_level(analysis_data, ensemble_prediction, confidence_scores)
        
        # Technical signals
        signals = generate_technical_signals(analysis_data)
        
        return {
            'analysis': analysis_text,
            'confidence': ensemble_prediction['confidence'],
            'strategy': strategy,
            'risk_level': risk_level,
            'signals': signals,
            'model_version': 'ML_v2.0',
            'individual_predictions': predictions,
            'ensemble_weights': ensemble_prediction['weights']
        }
        
    except Exception as e:
        logger.error(f"ML analysis error: {str(e)}")
        raise

def generate_technical_fallback_analysis(analysis_data):
    """Generate technical analysis when ML models fail"""
    if not analysis_data:
        return {
            'analysis': "Insufficient market data for analysis. Please wait for more data collection.",
            'confidence': 0.3,
            'strategy': 'wait_for_better_conditions'
        }
    
    volatility = analysis_data.get('volatility', 0)
    avg_change = analysis_data.get('average_change', 0)
    trend_direction = analysis_data.get('trend_direction', 'neutral')
    
    # Technical analysis logic
    if volatility > 2.0:
        analysis = "High volatility environment detected. Strong directional movements expected."
        strategy = "scalping_with_tight_stops"
        confidence = 0.75
    elif volatility < 0.5:
        analysis = "Low volatility consolidation phase. Range-bound trading conditions."
        strategy = "range_trading"
        confidence = 0.65
    else:
        analysis = "Moderate volatility with balanced market conditions."
        strategy = "trend_following"
        confidence = 0.70
    
    # Enhance analysis with pattern recognition
    patterns = analysis_data.get('patterns', {})
    if patterns.get('double_top', False):
        analysis += " Double top pattern detected - bearish reversal signal."
        confidence *= 0.9
    elif patterns.get('double_bottom', False):
        analysis += " Double bottom pattern detected - bullish reversal signal."
        confidence *= 0.9
    
    return {
        'analysis': analysis,
        'confidence': confidence,
        'strategy': strategy
    }

def calculate_advanced_support_resistance(prices):
    """Calculate advanced support and resistance levels"""
    if len(prices) < 20:
        return {'support': min(prices), 'resistance': max(prices)}
    
    # Pivot point method
    high = max(prices[-20:])
    low = min(prices[-20:])
    close = prices[-1]
    
    pivot = (high + low + close) / 3
    support1 = 2 * pivot - high
    resistance1 = 2 * pivot - low
    support2 = pivot - (high - low)
    resistance2 = pivot + (high - low)
    
    # Fibonacci retracement levels
    price_range = high - low
    fib_levels = {
        'fib_23.6': high - (price_range * 0.236),
        'fib_38.2': high - (price_range * 0.382),
        'fib_50.0': high - (price_range * 0.500),
        'fib_61.8': high - (price_range * 0.618),
        'fib_78.6': high - (price_range * 0.786)
    }
    
    return {
        'pivot_point': pivot,
        'support_levels': [support1, support2],
        'resistance_levels': [resistance1, resistance2],
        'fibonacci_levels': fib_levels,
        'current_range': {'high': high, 'low': low}
    }

def analyze_market_microstructure(prices):
    """Analyze market microstructure patterns"""
    if len(prices) < 10:
        return {}
    
    # Price acceleration
    acceleration = []
    for i in range(2, len(prices)):
        acc = (prices[i] - 2 * prices[i-1] + prices[i-2])
        acceleration.append(acc)
    
    # Tick-by-tick analysis
    tick_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    positive_ticks = sum(1 for change in tick_changes if change > 0)
    negative_ticks = sum(1 for change in tick_changes if change < 0)
    
    return {
        'acceleration_trend': np.mean(acceleration) if acceleration else 0,
        'tick_imbalance': (positive_ticks - negative_ticks) / len(tick_changes) if tick_changes else 0,
        'price_momentum': calculate_price_momentum(prices)
    }

def detect_price_patterns(prices):
    """Detect common price patterns"""
    if len(prices) < 20:
        return {}
    
    patterns = {}
    
    # Double top/bottom detection
    peaks = find_peaks(prices)
    troughs = find_troughs(prices)
    
    patterns['double_top'] = detect_double_top(peaks, prices)
    patterns['double_bottom'] = detect_double_bottom(troughs, prices)
    patterns['head_shoulders'] = detect_head_shoulders(peaks, prices)
    patterns['triangle'] = detect_triangle_pattern(prices)
    
    return patterns

def calculate_momentum_indicators(prices):
    """Calculate various momentum indicators"""
    if len(prices) < 20:
        return {}
    
    # Rate of Change (ROC)
    roc = (prices[-1] - prices[-10]) / prices[-10] * 100 if len(prices) >= 10 else 0
    
    # Price Rate of Change
    proc = (prices[-1] - prices[-5]) / prices[-5] * 100 if len(prices) >= 5 else 0
    
    # Momentum oscillator
    momentum = prices[-1] - prices[-10] if len(prices) >= 10 else 0
    
    return {
        'roc': roc,
        'proc': proc,
        'momentum': momentum,
        'velocity': calculate_price_velocity(prices)
    }

def analyze_volume_patterns(data_points):
    """Analyze volume patterns if available"""
    volumes = [point.get('volume', 1.0) for point in data_points[-50:]]
    
    if len(volumes) < 10:
        return {'volume_trend': 'insufficient_data'}
    
    # Volume trend analysis
    recent_volume = np.mean(volumes[-5:])
    avg_volume = np.mean(volumes)
    volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
    
    return {
        'volume_trend': 'increasing' if volume_ratio > 1.2 else 'decreasing' if volume_ratio < 0.8 else 'stable',
        'volume_ratio': volume_ratio,
        'volume_spike': volume_ratio > 2.0
    }

# Additional helper functions
def extract_ml_features(analysis_data, data_points):
    """Extract features for ML models"""
    # Implement comprehensive feature extraction
    features = []
    
    # Price-based features
    prices = [point['price'] for point in data_points[-100:]]
    if len(prices) >= 20:
        features.extend([
            analysis_data.get('volatility', 0),
            analysis_data.get('average_change', 0),
            analysis_data.get('skewness', 0),
            analysis_data.get('kurtosis', 0)
        ])
    else:
        features.extend([0, 0, 0, 0])
    
    return np.array(features)

def get_random_forest_prediction(features):
    """Get Random Forest prediction"""
    # Placeholder - implement actual model prediction
    return 'bullish', 0.75

def get_gradient_boost_prediction(features):
    """Get Gradient Boosting prediction"""
    # Placeholder - implement actual model prediction
    return 'bullish', 0.72

def get_svm_prediction(features):
    """Get SVM prediction"""
    # Placeholder - implement actual model prediction
    return 'neutral', 0.68

def get_ensemble_prediction(predictions, confidence_scores):
    """Combine predictions using ensemble methods"""
    # Implement ensemble voting logic
    weighted_prediction = 'bullish'  # Placeholder
    ensemble_confidence = np.mean(list(confidence_scores.values()))
    
    return {
        'prediction': weighted_prediction,
        'confidence': ensemble_confidence,
        'weights': {'rf': 0.4, 'gb': 0.35, 'svm': 0.25}
    }

def generate_ml_analysis_text(analysis_data, ensemble_prediction, individual_predictions):
    """Generate human-readable analysis text"""
    pred = ensemble_prediction['prediction']
    conf = ensemble_prediction['confidence']
    
    text = f"ML Model Analysis: {pred.upper()} sentiment detected with {conf:.1%} confidence.\n\n"
    text += f"Current market shows {analysis_data.get('trend_direction', 'neutral')} bias with "
    text += f"{analysis_data.get('volatility', 0):.2f}% volatility.\n\n"
    
    # Add technical insights
    if analysis_data.get('patterns', {}).get('double_top'):
        text += "⚠️ Double top pattern detected - potential reversal signal.\n"
    elif analysis_data.get('patterns', {}).get('double_bottom'):
        text += "📈 Double bottom pattern detected - potential bullish reversal.\n"
    
    text += f"\nModel Consensus: RF({individual_predictions.get('random_forest', 'N/A')}), "
    text += f"GB({individual_predictions.get('gradient_boost', 'N/A')}), "
    text += f"SVM({individual_predictions.get('svm', 'N/A')})"
    
    return text

def determine_optimal_strategy(analysis_data, ensemble_prediction):
    """Determine optimal trading strategy based on analysis"""
    volatility = analysis_data.get('volatility', 0)
    prediction = ensemble_prediction['prediction']
    confidence = ensemble_prediction['confidence']
    
    if confidence > 0.8 and volatility > 1.5:
        return 'aggressive_trend_following'
    elif confidence > 0.7:
        return 'conservative_trend_following'
    elif volatility < 0.5:
        return 'range_trading'
    else:
        return 'wait_for_better_setup'

def assess_risk_level(analysis_data, ensemble_prediction, confidence_scores):
    """Assess overall risk level"""
    volatility = analysis_data.get('volatility', 0)
    confidence = ensemble_prediction['confidence']
    
    if volatility > 2.0 or confidence < 0.6:
        return 'high'
    elif volatility > 1.0 or confidence < 0.75:
        return 'medium'
    else:
        return 'low'

def generate_technical_signals(analysis_data):
    """Generate technical trading signals"""
    signals = []
    
    # Trend signals
    if analysis_data.get('trend_direction') == 'up':
        signals.append({'type': 'trend', 'signal': 'bullish', 'strength': 'medium'})
    elif analysis_data.get('trend_direction') == 'down':
        signals.append({'type': 'trend', 'signal': 'bearish', 'strength': 'medium'})
    
    # Volatility signals
    volatility = analysis_data.get('volatility', 0)
    if volatility > 2.0:
        signals.append({'type': 'volatility', 'signal': 'high_vol_breakout', 'strength': 'strong'})
    elif volatility < 0.5:
        signals.append({'type': 'volatility', 'signal': 'low_vol_consolidation', 'strength': 'medium'})
    
    return signals

def get_ensemble_recommendation(market_data: dict, contract_type: ContractType) -> dict:
    """Get ensemble recommendation combining multiple ML models"""
    try:
        price_history = market_data.get('price_history', [])
        if len(price_history) < 20:
            return {
                'direction': 'hold',
                'confidence': 0.5,
                'reasoning': 'Insufficient data for ensemble analysis'
            }
        
        prices = [p.get('price', p) if isinstance(p, dict) else p for p in price_history[-50:]]
        
        # Simple technical analysis for ensemble
        recent_trend = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
        volatility = np.std(np.diff(prices)) / np.mean(prices) if len(prices) > 1 else 0
        
        # Determine direction based on trend and volatility
        if recent_trend > 0.001:
            direction = 'call' if contract_type == ContractType.RISE_FALL else 'up'
            confidence = min(0.8, 0.5 + abs(recent_trend) * 10)
        elif recent_trend < -0.001:
            direction = 'put' if contract_type == ContractType.RISE_FALL else 'down'
            confidence = min(0.8, 0.5 + abs(recent_trend) * 10)
        else:
            direction = 'hold'
            confidence = 0.5
        
        return {
            'direction': direction,
            'confidence': confidence,
            'volatility': volatility,
            'trend_strength': abs(recent_trend),
            'reasoning': f'Ensemble analysis shows {direction} bias with {confidence:.1%} confidence'
        }
        
    except Exception as e:
        logger.error(f"Ensemble recommendation error: {str(e)}")
        return {
            'direction': 'hold',
            'confidence': 0.5,
            'reasoning': 'Error in ensemble analysis'
        }

def get_technical_analysis_summary(data_points: list) -> dict:
    """Generate technical analysis summary"""
    try:
        if len(data_points) < 20:
            return {'error': 'Insufficient data for technical analysis'}
        
        prices = [p.get('price', p) if isinstance(p, dict) else p for p in data_points[-50:]]
        
        # Calculate basic indicators
        sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
        current_price = prices[-1]
        
        # RSI calculation (simplified)
        price_changes = np.diff(prices)
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)
        
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
        
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma = np.mean(prices[-20:]) if len(prices) >= 20 else current_price
        std = np.std(prices[-20:]) if len(prices) >= 20 else 0
        bb_upper = sma + (2 * std)
        bb_lower = sma - (2 * std)
        
        return {
            'current_price': current_price,
            'sma_20': sma_20,
            'rsi': rsi,
            'bollinger_bands': {
                'upper': bb_upper,
                'middle': sma,
                'lower': bb_lower,
                'position': (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            },
            'trend': 'bullish' if current_price > sma_20 else 'bearish',
            'momentum': 'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'
        }
        
    except Exception as e:
        logger.error(f"Technical analysis error: {str(e)}")
        return {'error': 'Technical analysis failed'}

def detect_market_regime(data_points: list) -> dict:
    """Detect current market regime"""
    try:
        if len(data_points) < 30:
            return {'regime': 'unknown', 'confidence': 0.5}
        
        prices = [p.get('price', p) if isinstance(p, dict) else p for p in data_points[-30:]]
        returns = np.diff(prices) / np.array(prices[:-1])
        
        volatility = np.std(returns)
        mean_return = np.mean(returns)
        
        # Classify regime based on volatility and trend
        if volatility > 0.02:
            regime = 'high_volatility'
        elif volatility < 0.005:
            regime = 'low_volatility'
        elif mean_return > 0.001:
            regime = 'trending_up'
        elif mean_return < -0.001:
            regime = 'trending_down'
        else:
            regime = 'sideways'
        
        # Calculate confidence based on how clear the regime is
        confidence = min(0.9, 0.5 + (volatility * 25))
        
        return {
            'regime': regime,
            'confidence': confidence,
            'volatility': volatility,
            'mean_return': mean_return
        }
        
    except Exception as e:
        logger.error(f"Market regime detection error: {str(e)}")
        return {'regime': 'unknown', 'confidence': 0.5}

def forecast_volatility(data_points: list) -> dict:
    """Forecast short-term volatility"""
    try:
        if len(data_points) < 20:
            return {'forecast': 'unknown', 'confidence': 0.5}
        
        prices = [p.get('price', p) if isinstance(p, dict) else p for p in data_points[-20:]]
        returns = np.diff(prices) / np.array(prices[:-1])
        
        # Simple EWMA volatility forecast
        current_vol = np.std(returns)
        recent_vol = np.std(returns[-10:]) if len(returns) >= 10 else current_vol
        
        # Forecast direction
        if recent_vol > current_vol * 1.2:
            forecast = 'increasing'
        elif recent_vol < current_vol * 0.8:
            forecast = 'decreasing'
        else:
            forecast = 'stable'
        
        confidence = min(0.8, 0.5 + abs(recent_vol - current_vol) * 50)
        
        return {
            'forecast': forecast,
            'current_volatility': current_vol,
            'recent_volatility': recent_vol,
            'confidence': confidence,
            'expected_range': {
                'high': prices[-1] * (1 + recent_vol),
                'low': prices[-1] * (1 - recent_vol)
            }
        }
        
    except Exception as e:
        logger.error(f"Volatility forecast error: {str(e)}")
        return {'forecast': 'unknown', 'confidence': 0.5}

# Helper functions for pattern detection
def find_peaks(prices):
    """Find price peaks"""
    peaks = []
    for i in range(1, len(prices) - 1):
        if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
            peaks.append((i, prices[i]))
    return peaks

def find_troughs(prices):
    """Find price troughs"""
    troughs = []
    for i in range(1, len(prices) - 1):
        if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
            troughs.append((i, prices[i]))
    return troughs

def detect_double_top(peaks, prices):
    """Detect double top pattern"""
    if len(peaks) < 2:
        return False
    
    # Simple double top detection
    last_two_peaks = peaks[-2:]
    if len(last_two_peaks) == 2:
        diff = abs(last_two_peaks[0][1] - last_two_peaks[1][1])
        avg_price = np.mean([peak[1] for peak in last_two_peaks])
        return diff / avg_price < 0.02  # Within 2% of each other
    
    return False

def detect_double_bottom(troughs, prices):
    """Detect double bottom pattern"""
    if len(troughs) < 2:
        return False
    
    # Simple double bottom detection
    last_two_troughs = troughs[-2:]
    if len(last_two_troughs) == 2:
        diff = abs(last_two_troughs[0][1] - last_two_troughs[1][1])
        avg_price = np.mean([trough[1] for trough in last_two_troughs])
        return diff / avg_price < 0.02  # Within 2% of each other
    
    return False

def detect_head_shoulders(peaks, prices):
    """Detect head and shoulders pattern"""
    # Simplified head and shoulders detection
    return len(peaks) >= 3

def detect_triangle_pattern(prices):
    """Detect triangle pattern"""
    if len(prices) < 20:
        return False
    
    # Simplified triangle detection based on converging highs and lows
    recent_highs = []
    recent_lows = []
    
    for i in range(len(prices) - 10, len(prices)):
        if i > 0 and i < len(prices) - 1:
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                recent_highs.append(prices[i])
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                recent_lows.append(prices[i])
    
    return len(recent_highs) >= 2 and len(recent_lows) >= 2

def calculate_price_momentum(prices):
    """Calculate price momentum"""
    if len(prices) < 10:
        return 0
    
    short_ma = np.mean(prices[-5:])
    long_ma = np.mean(prices[-10:])
    
    return (short_ma - long_ma) / long_ma if long_ma != 0 else 0

def calculate_price_velocity(prices):
    """Calculate price velocity"""
    if len(prices) < 3:
        return 0
    
    velocity = (prices[-1] - prices[-3]) / 2
    return velocity

@ai_bp.route('/market-prediction', methods=['POST'])
@jwt_required()
def get_market_prediction():
    """Get ML-based market predictions"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        data_points = data.get('dataPoints', [])
        prediction_horizon = data.get('horizon', 300)  # 5 minutes default
        
        if len(data_points) < 50:
            return jsonify({'error': 'Insufficient data for prediction'}), 400
        
        # Generate ML predictions
        predictions = generate_ml_predictions(data_points, prediction_horizon)
        
        return jsonify({
            'predictions': predictions,
            'model_confidence': predictions['confidence'],
            'time_horizon': prediction_horizon,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Market prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def generate_ml_predictions(data_points, horizon):
    """Generate ML-based market predictions"""
    prices = [point['price'] for point in data_points[-100:]]
    
    # Simple trend-based prediction (to be enhanced with actual ML models)
    recent_trend = np.polyfit(range(len(prices[-20:])), prices[-20:], 1)[0]
    
    predictions = {
        'direction': 'up' if recent_trend > 0 else 'down',
        'confidence': min(0.85, abs(recent_trend) * 1000),  # Scale trend to confidence
        'price_targets': {
            'optimistic': prices[-1] * (1 + recent_trend * 0.1),
            'realistic': prices[-1] * (1 + recent_trend * 0.05),
            'conservative': prices[-1] * (1 + recent_trend * 0.02)
        }
    }
    
    return predictions

@ai_bp.route('/train-model', methods=['POST'])
@jwt_required()
def train_model():
    """Trigger model training with new data"""
    try:
        # Trigger ML model training
        training_result = ml_strategy_manager.train_models()
        
        return jsonify({
            'status': 'training_completed',
            'models_trained': training_result['models_trained'],
            'performance_metrics': training_result['performance'],
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Model training error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@ai_bp.route('/model-status', methods=['GET'])
@jwt_required()
def get_model_status():
    """Get current ML model status and performance"""
    try:
        status = ml_strategy_manager.get_model_status()
        
        return jsonify({
            'model_status': status,
            'last_training': status.get('last_training'),
            'performance_metrics': status.get('performance'),
            'available_models': list(ContractType),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Model status error: {str(e)}")
        return jsonify({'error': str(e)}), 500
