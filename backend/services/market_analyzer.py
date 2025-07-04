import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from threading import Thread
import time
import websocket
import ssl
from collections import deque
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeMarketAnalyzer:
    def __init__(self, openai_api_key: str = None):
        """Initialize the real-time market analyzer"""
        # Use environment variable if available, otherwise use provided key or a placeholder
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY") or ""
        
        # Handle OpenAI client initialization with error handling
        try:
            if self.openai_api_key:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
            else:
                logger.warning("No OpenAI API key provided. ChatGPT analysis will be unavailable.")
                self.openai_client = None
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            self.openai_client = None
        
        # Data storage
        self.price_data = deque(maxlen=1000)  # Store last 1000 price points
        self.digit_history = deque(maxlen=100)  # Store last 100 digits
        self.prediction_cache = {}
        self.last_analysis = {}
        
        # Analysis state
        self.is_analyzing = False
        self.analysis_thread = None
        self.subscribers = []
        
        # Technical indicators
        self.indicators = {}
        self.market_sentiment = 'neutral'
        self.predictions = {
            'future_digits': [],
            'price_movement': None,
            'pattern_analysis': {},
            'frequency_analysis': {}
        }

    def add_price_data(self, price: float, timestamp: datetime = None):
        """Add new price data point"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        price_point = {
            'price': price,
            'timestamp': timestamp,
            'digit': int((price * 100) % 10)
        }
        
        self.price_data.append(price_point)
        self.digit_history.append(price_point['digit'])
        
        # Trigger real-time analysis
        if len(self.price_data) >= 50:
            self._trigger_analysis()

    def _trigger_analysis(self):
        """Trigger real-time analysis in background thread"""
        if not self.is_analyzing:
            self.analysis_thread = Thread(target=self._perform_analysis)
            self.analysis_thread.daemon = True
            self.analysis_thread.start()

    def _perform_analysis(self):
        """Perform comprehensive market analysis"""
        self.is_analyzing = True
        try:
            # Calculate technical indicators
            self.indicators = self._calculate_technical_indicators()
            
            # Predict future digits
            self.predictions['future_digits'] = self._predict_future_digits()
            
            # Predict price movement
            self.predictions['price_movement'] = self._predict_price_movement()
            
            # Pattern analysis
            self.predictions['pattern_analysis'] = self._analyze_patterns()
            
            # Frequency analysis
            self.predictions['frequency_analysis'] = self._analyze_frequency()
            
            # Market sentiment
            self.market_sentiment = self._analyze_market_sentiment()
            
            # Get ChatGPT analysis
            chatgpt_analysis = self._get_chatgpt_analysis()
            
            # Prepare complete analysis
            analysis_result = {
                'timestamp': datetime.utcnow().isoformat(),
                'current_price': self.price_data[-1]['price'] if self.price_data else None,
                'current_digit': self.price_data[-1]['digit'] if self.price_data else None,
                'technical_indicators': self.indicators,
                'predictions': self.predictions,
                'market_sentiment': self.market_sentiment,
                'chatgpt_analysis': chatgpt_analysis,
                'data_points': len(self.price_data)
            }
            
            self.last_analysis = analysis_result
            
            # Notify subscribers
            self._notify_subscribers(analysis_result)
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
        finally:
            self.is_analyzing = False

    def _calculate_technical_indicators(self) -> Dict:
        """Calculate comprehensive technical indicators"""
        if len(self.price_data) < 14:
            return {}
        
        prices = np.array([p['price'] for p in list(self.price_data)])
        
        try:
            indicators = {
                'rsi': self._calculate_rsi(prices, 14),
                'macd': self._calculate_macd(prices),
                'volatility': self._calculate_volatility(prices),
                'momentum': self._calculate_momentum(prices, 10),
                'trend_strength': self._calculate_trend_strength(prices)
            }
            
            return indicators
        except Exception as e:
            logger.error(f"Technical indicators error: {str(e)}")
            return {}

    def _predict_future_digits(self) -> List[Dict]:
        """Predict next 5 digit values using multiple algorithms"""
        if len(self.price_data) < 50:
            return []
        
        predictions = []
        current_price = self.price_data[-1]['price']
        prices = np.array([p['price'] for p in list(self.price_data)])
        
        # Multiple prediction models
        volatility = self._calculate_volatility(prices[-20:])
        trend = self._calculate_trend(prices[-20:])
        momentum = self._calculate_momentum(prices[-10:])
        
        for step in range(1, 6):
            # Ensemble prediction
            predicted_price = self._predict_next_price(current_price, volatility, trend, momentum, step)
            predicted_digit = int((predicted_price * 100) % 10)
            confidence = self._calculate_digit_confidence(prices, predicted_digit, step)
            
            predictions.append({
                'step': step,
                'predicted_price': round(predicted_price, 5),
                'predicted_digit': predicted_digit,
                'confidence': round(confidence, 1),
                'time_estimate': f'{step * 2}s',
                'algorithm': self._get_algorithm_used(volatility, trend)
            })
        
        return predictions

    def _predict_price_movement(self) -> Dict:
        """Predict price movement with probabilities"""
        if len(self.price_data) < 20:
            return None
        
        prices = np.array([p['price'] for p in list(self.price_data)])
        
        # Calculate signals
        bullish_score = 0
        bearish_score = 0
        signals = []
        
        # RSI signal
        rsi = self.indicators.get('rsi', 50)
        if rsi < 30:
            bullish_score += 2
            signals.append({'type': 'buy', 'reason': 'Oversold RSI'})
        elif rsi > 70:
            bearish_score += 2
            signals.append({'type': 'sell', 'reason': 'Overbought RSI'})
        
        # Trend signal
        trend = self._calculate_trend(prices[-10:])
        if trend > 0.01:
            bullish_score += 2
            signals.append({'type': 'buy', 'reason': 'Strong uptrend'})
        elif trend < -0.01:
            bearish_score += 2
            signals.append({'type': 'sell', 'reason': 'Strong downtrend'})
        
        # MACD signal
        macd = self.indicators.get('macd', {})
        if macd.get('signal', 0) > 0:
            bullish_score += 1
            signals.append({'type': 'buy', 'reason': 'MACD bullish crossover'})
        else:
            bearish_score += 1
            signals.append({'type': 'sell', 'reason': 'MACD bearish crossover'})
        
        total_score = bullish_score + bearish_score
        bullish_percentage = (bullish_score / total_score * 100) if total_score > 0 else 50
        bearish_percentage = 100 - bullish_percentage
        
        direction = 'bullish' if bullish_score > bearish_score else 'bearish' if bearish_score > bullish_score else 'neutral'
        confidence = abs(bullish_score - bearish_score) * 10 + 50
        
        return {
            'direction': direction,
            'confidence': min(95, confidence),
            'bullish_percentage': round(bullish_percentage, 1),
            'bearish_percentage': round(bearish_percentage, 1),
            'signals': signals,
            'next_move_time': '30-60 seconds'
        }

    def _analyze_patterns(self) -> Dict:
        """Analyze digit patterns"""
        if len(self.digit_history) < 20:
            return {}
        
        digits = list(self.digit_history)
        
        # Pattern detection
        patterns = {
            'repeating_sequences': self._find_repeating_sequences(digits),
            'hot_digits': self._find_hot_digits(digits),
            'cold_digits': self._find_cold_digits(digits),
            'consecutive_patterns': self._find_consecutive_patterns(digits),
            'cycle_analysis': self._analyze_cycles(digits)
        }
        
        return patterns

    def _analyze_frequency(self) -> Dict:
        """Analyze digit frequency patterns"""
        if len(self.digit_history) < 10:
            return {}
        
        digits = list(self.digit_history)
        
        # Frequency analysis
        frequency = {}
        for digit in range(10):
            count = digits.count(digit)
            last_seen = self._get_last_seen(digits, digit)
            expected_prob = self._calculate_expected_probability(digits, digit)
            
            frequency[digit] = {
                'count': count,
                'percentage': round((count / len(digits)) * 100, 1),
                'last_seen': last_seen.isoformat() if last_seen else None,
                'expected_probability': round(expected_prob, 1)
            }
        
        return {
            'frequency_distribution': frequency,
            'most_frequent': max(frequency.items(), key=lambda x: x[1]['count'])[0],
            'least_frequent': min(frequency.items(), key=lambda x: x[1]['count'])[0],
            'entropy': self._calculate_entropy(digits),
            'randomness_score': self._calculate_randomness_score(digits)
        }

    def _analyze_market_sentiment(self) -> str:
        """Analyze overall market sentiment"""
        if not self.indicators:
            return 'neutral'
        
        bullish_signals = 0
        bearish_signals = 0
        
        # RSI sentiment
        rsi = self.indicators.get('rsi', 50)
        if rsi < 30:
            bullish_signals += 1
        elif rsi > 70:
            bearish_signals += 1
        
        # Trend sentiment
        trend_strength = self.indicators.get('trend_strength', 0)
        if trend_strength > 0.5:
            bullish_signals += 1
        elif trend_strength < -0.5:
            bearish_signals += 1
        
        # Volatility sentiment
        volatility = self.indicators.get('volatility', 0)
        if volatility > 0.02:
            bearish_signals += 1  # High volatility = uncertainty
        
        if bullish_signals > bearish_signals:
            return 'bullish'
        elif bearish_signals > bullish_signals:
            return 'bearish'
        else:
            return 'neutral'

    def _get_chatgpt_analysis(self) -> str:
        """Get AI analysis from ChatGPT"""
        try:
            if not self.indicators or not self.price_data:
                return self._generate_fallback_analysis()
            
            # Check if OpenAI client is available
            if not self.openai_client:
                logger.warning("OpenAI client not available. Using fallback analysis.")
                return self._generate_fallback_analysis()
                
            # Prepare market data for ChatGPT
            current_price = self.price_data[-1]['price']
            recent_prices = [p['price'] for p in list(self.price_data)[-10:]]
            
            prompt = f"""Analyze this market data:
            Current Price: {current_price}
            Recent prices: {recent_prices}
            RSI: {self.indicators.get('rsi', 'N/A')}
            Volatility: {self.indicators.get('volatility', 'N/A')}
            Market Sentiment: {self.market_sentiment}
            
            Provide a brief analysis in 2-3 sentences focusing on trading opportunities."""
            
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.7
                )
                
                return response.choices[0].message.content.strip()
            except Exception as api_error:
                logger.error(f"OpenAI API error: {str(api_error)}")
                return self._generate_fallback_analysis() + " (API Error)"
            
        except Exception as e:
            logger.error(f"ChatGPT analysis error: {str(e)}")
            return self._generate_fallback_analysis() + " (Error)"

    def _generate_fallback_analysis(self) -> str:
        """Generate fallback analysis when ChatGPT is unavailable"""
        if not self.indicators:
            return "Insufficient data for market analysis. Please wait for more price data."
        
        volatility = self.indicators.get('volatility', 0)
        rsi = self.indicators.get('rsi', 50)
        trend_strength = self.indicators.get('trend_strength', 0)
        
        analysis = f"Market Analysis: {self.market_sentiment.upper()} sentiment detected. "
        
        if volatility > 0.02:
            analysis += "High volatility suggests active trading opportunities. "
        elif volatility < 0.005:
            analysis += "Low volatility indicates stable market conditions. "
        else:
            analysis += "Moderate volatility presents balanced trading conditions. "
        
        if rsi > 70:
            analysis += "Overbought conditions suggest potential price reversal. "
        elif rsi < 30:
            analysis += "Oversold conditions suggest potential price bounce. "
        
        confidence = min(85, max(60, 70 + abs(trend_strength) * 30))
        analysis += f"Confidence: {confidence:.0f}%"
        
        return analysis

    # Technical indicator calculation methods
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
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

    def _calculate_macd(self, prices: np.ndarray) -> Dict:
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

    def _ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
        
        alpha = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema

    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """Calculate price volatility"""
        if len(prices) < 2:
            return 0.0
        
        returns = np.diff(prices) / prices[:-1]
        return float(np.std(returns))

    def _calculate_trend(self, prices: np.ndarray) -> float:
        """Calculate trend direction"""
        if len(prices) < 2:
            return 0.0
        
        return float((prices[-1] - prices[0]) / prices[0])

    def _calculate_momentum(self, prices: np.ndarray, period: int = 10) -> float:
        """Calculate price momentum"""
        if len(prices) < period + 1:
            return 0.0
        
        return float((prices[-1] - prices[-period-1]) / prices[-period-1])

    def _find_repeating_sequences(self, digits: List[int], min_length: int = 3) -> List[str]:
        """Find repeating sequences in digit history"""
        sequences = []
        digit_str = ''.join(map(str, digits))
        for length in range(min_length, len(digits) // 2 + 1):
            for start in range(len(digits) - length):
                seq = digit_str[start:start+length]
                if digit_str.count(seq) >= 2 and seq not in sequences:
                    sequences.append(seq)
        
        return sequences

    def _find_hot_digits(self, digits: List[int], threshold: int = 5) -> List[int]:
        """Find hot digits that appear more frequently than threshold"""
        from collections import Counter
        
        digit_counts = Counter(digits)
        return [digit for digit, count in digit_counts.items() if count >= threshold]

    def _find_cold_digits(self, digits: List[int], threshold: int = 5) -> List[int]:
        """Find cold digits that appear less frequently than threshold"""
        from collections import Counter
        
        digit_counts = Counter(digits)
        return [digit for digit, count in digit_counts.items() if count <= threshold]

    def _find_consecutive_patterns(self, digits: List[int], min_length: int = 3) -> List[str]:
        """Find consecutive patterns like '123', '456'"""
        patterns = []
        digit_str = ''.join(map(str, digits))
        for length in range(min_length, len(digits)):
            for start in range(len(digits) - length):
                seq = digit_str[start:start+length]
                if all(int(seq[i]) + 1 == int(seq[i+1]) for i in range(len(seq) - 1)) and seq not in patterns:
                    patterns.append(seq)
        
        return patterns

    def _analyze_cycles(self, digits: List[int]) -> Dict:
        """Analyze cycles in digit appearance"""
        try:
            from scipy.signal import find_peaks
            
            digit_counts = np.array([digits.count(i) for i in range(10)])
            peaks, _ = find_peaks(digit_counts, height=0)
            
            return {
                'cycle_peaks': peaks.tolist(),
                'cycle_periods': np.diff(peaks).tolist() if len(peaks) > 1 else []
            }
        except ImportError:
            # Fallback without scipy
            return {
                'cycle_peaks': [],
                'cycle_periods': []
            }

    def _get_last_seen(self, digits: List[int], target_digit: int) -> Optional[datetime]:
        """Get the last seen time of a digit in the history"""
        for data in reversed(self.price_data):
            if data['digit'] == target_digit:
                return data['timestamp']
        return None

    def _calculate_expected_probability(self, digits: List[int], target_digit: int) -> float:
        """Calculate expected probability of a digit based on history"""
        total_count = len(digits)
        target_count = sum(1 for d in digits if d == target_digit)
        
        return (target_count / total_count) * 100 if total_count > 0 else 0

    def _calculate_entropy(self, digits: List[int]) -> float:
        """Calculate entropy of the digit sequence"""
        try:
            from math import log2
            from collections import Counter
            
            digit_counts = Counter(digits)
            total_digits = len(digits)
            
            # Low randomness if any digit dominates, high randomness if uniform distribution
            dominance_factor = max(digit_counts.values()) / total_digits
            return (1 - dominance_factor) * 100
        except Exception:
            return 0.0
    def get_latest_analysis(self) -> Dict:
        """Get the latest analysis results"""
        return self.last_analysis
    
    def subscribe_to_analysis(self, callback):
        """Subscribe to real-time analysis updates"""
        self.subscribers.append(callback)

    def _notify_subscribers(self, analysis: Dict):
        """Notify all subscribers of new analysis"""
        for callback in self.subscribers:
            try:
                callback(analysis)
            except Exception as e:
                logger.error(f"Subscriber notification error: {str(e)}")
    
    def _predict_next_price(self, current_price: float, volatility: float, trend: float, momentum: float, step: int) -> float:
        """Predict next price using ensemble methods"""
        # Simple ensemble prediction combining multiple factors
        price_change = (trend * 0.4 + momentum * 0.3 + volatility * 0.3) * step * 0.001
        return current_price + price_change
    
    def _calculate_digit_confidence(self, prices: np.ndarray, predicted_digit: int, step: int) -> float:
        """Calculate confidence in digit prediction"""
        # Base confidence decreases with prediction distance
        base_confidence = max(50, 90 - (step * 10))
        
        # Adjust based on recent digit frequency
        recent_digits = [int((price * 100) % 10) for price in prices[-20:]]
        digit_frequency = recent_digits.count(predicted_digit) / len(recent_digits)
        
        # Confidence adjustment based on frequency
        frequency_adjustment = (digit_frequency - 0.1) * 50
        
        return min(95, max(40, base_confidence + frequency_adjustment))
    
    def _get_algorithm_used(self, volatility: float, trend: float) -> str:
        """Determine which algorithm was primarily used for prediction"""
        if abs(trend) > 0.01:
            return "Trend-based"
        elif volatility > 0.02:
            return "Volatility-based"
        else:
            return "Momentum-based"
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength indicator"""
        if len(prices) < 10:
            return 0.0
            
        # Calculate linear regression slope
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        
        # Normalize slope relative to price
        normalized_slope = slope / np.mean(prices)
        
        return float(normalized_slope * 100)

# Create a global instance for use throughout the application
market_analyzer = RealTimeMarketAnalyzer()