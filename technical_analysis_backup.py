"""
Technical Analysis module for trading signals
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime, timedelta
import config

logger = logging.getLogger(__name__)

# Native implementations of technical indicators (no TA-Lib dependency)
def calculate_rsi(prices, period=14):
    """Calculate RSI without TA-Lib"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD without TA-Lib"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, std=2):
    """Calculate Bollinger Bands without TA-Lib"""
    sma = prices.rolling(window=period).mean()
    std_dev = prices.rolling(window=period).std()
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    return upper_band, sma, lower_band

class TechnicalAnalyzer:
    def __init__(self):
        self.config = config.TECHNICAL_CONFIG
        
    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate various technical indicators"""
        indicators = {}
        
        try:
            # Ensure we have OHLCV data
            if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                logger.error("Missing OHLCV data in DataFrame")
                return indicators
                
            # RSI
            indicators['rsi'] = self._calculate_rsi(df)
            
            # MACD
            indicators['macd'] = self._calculate_macd(df)
            
            # Bollinger Bands
            indicators['bollinger'] = self._calculate_bollinger_bands(df)
            
            # Moving Averages
            indicators['moving_averages'] = self._calculate_moving_averages(df)
            
            # Support and Resistance
            indicators['support_resistance'] = self._calculate_support_resistance(df)
            
            # Volume indicators
            if 'volume' in df.columns:
                indicators['volume'] = self._calculate_volume_indicators(df)
                
            # Candlestick patterns
            indicators['candlestick_patterns'] = self._identify_candlestick_patterns(df)
            
            # Trend analysis
            indicators['trend'] = self._analyze_trend(df)
            
            # Volatility
            indicators['volatility'] = self._calculate_volatility(df)
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            
        return indicators
        
    def _calculate_rsi(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate RSI indicator"""
        try:
            rsi_series = calculate_rsi(df['close'], self.config['rsi_period'])
            current_rsi = rsi_series.iloc[-1] if len(rsi_series) > 0 and not np.isnan(rsi_series.iloc[-1]) else 50
            
            return {
                'value': current_rsi,
                'signal': self._get_rsi_signal(current_rsi),
                'overbought': current_rsi > self.config['rsi_overbought'],
                'oversold': current_rsi < self.config['rsi_oversold']
            }
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return {'value': 50, 'signal': 'NEUTRAL', 'overbought': False, 'oversold': False}
            
    def _get_rsi_signal(self, rsi: float) -> str:
        """Get RSI signal"""
        if rsi > self.config['rsi_overbought']:
            return 'SELL'
        elif rsi < self.config['rsi_oversold']:
            return 'BUY'
        else:
            return 'NEUTRAL'
            
    def _calculate_macd(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate MACD indicator"""
        try:
            macd, macd_signal, macd_hist = calculate_macd(
                df['close'],
                fast=self.config['macd_fast'],
                slow=self.config['macd_slow'],
                signal=self.config['macd_signal']
            )
            
            current_macd = macd[-1] if len(macd) > 0 and not np.isnan(macd[-1]) else 0
            current_signal = macd_signal[-1] if len(macd_signal) > 0 and not np.isnan(macd_signal[-1]) else 0
            current_hist = macd_hist[-1] if len(macd_hist) > 0 and not np.isnan(macd_hist[-1]) else 0
            
            return {
                'macd': current_macd,
                'signal': current_signal,
                'histogram': current_hist,
                'signal_type': 'BUY' if current_macd > current_signal else 'SELL',
                'divergence': self._check_macd_divergence(macd_hist)
            }
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return {'macd': 0, 'signal': 0, 'histogram': 0, 'signal_type': 'NEUTRAL', 'divergence': False}
            
    def _check_macd_divergence(self, macd_hist: np.ndarray) -> bool:
        """Check for MACD divergence"""
        if len(macd_hist) < 10:
            return False
            
        recent_hist = macd_hist[-10:]
        return abs(recent_hist[-1] - recent_hist[0]) > np.std(recent_hist)
        
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        try:
            upper, middle, lower = calculate_bollinger_bands(
                df['close'],
                period=self.config['bb_period'],
                std=self.config['bb_std']
            )
            
            current_price = df['close'].iloc[-1]
            current_upper = upper.iloc[-1] if len(upper) > 0 and not np.isnan(upper.iloc[-1]) else current_price
            current_middle = middle.iloc[-1] if len(middle) > 0 and not np.isnan(middle.iloc[-1]) else current_price
            current_lower = lower.iloc[-1] if len(lower) > 0 and not np.isnan(lower.iloc[-1]) else current_price
            
            return {
                'upper': current_upper,
                'middle': current_middle,
                'lower': current_lower,
                'position': self._get_bb_position(current_price, current_upper, current_middle, current_lower),
                'squeeze': self._check_bb_squeeze(upper, lower)
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return {'upper': 0, 'middle': 0, 'lower': 0, 'position': 'MIDDLE', 'squeeze': False}
            
    def _get_bb_position(self, price: float, upper: float, middle: float, lower: float) -> str:
        """Get Bollinger Bands position"""
        if price > upper:
            return 'ABOVE_UPPER'
        elif price < lower:
            return 'BELOW_LOWER'
        elif price > middle:
            return 'UPPER_HALF'
        else:
            return 'LOWER_HALF'
            
    def _check_bb_squeeze(self, upper: np.ndarray, lower: np.ndarray) -> bool:
        """Check for Bollinger Bands squeeze"""
        if len(upper) < 20 or len(lower) < 20:
            return False
            
        current_width = upper[-1] - lower[-1]
        avg_width = np.mean(upper[-20:] - lower[-20:])
        
        return current_width < avg_width * 0.8
        
    def _calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate moving averages"""
        mas = {}
        
        try:
            for period in self.config['ma_periods']:
                ma = talib.SMA(df['close'].values, timeperiod=period)
                current_ma = ma[-1] if len(ma) > 0 and not np.isnan(ma[-1]) else df['close'].iloc[-1]
                mas[f'ma_{period}'] = current_ma
                
            current_price = df['close'].iloc[-1]
            
            # Golden Cross / Death Cross
            ma_50 = mas.get('ma_50', current_price)
            ma_100 = mas.get('ma_100', current_price)
            
            return {
                'values': mas,
                'golden_cross': ma_50 > ma_100,
                'price_above_ma20': current_price > mas.get('ma_20', current_price),
                'price_above_ma50': current_price > ma_50,
                'trend_signal': self._get_ma_trend_signal(current_price, mas)
            }
        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
            return {'values': {}, 'golden_cross': False, 'price_above_ma20': True, 'price_above_ma50': True, 'trend_signal': 'NEUTRAL'}
            
    def _get_ma_trend_signal(self, price: float, mas: Dict[str, float]) -> str:
        """Get moving average trend signal"""
        if not mas:
            return 'NEUTRAL'
            
        ma_20 = mas.get('ma_20', price)
        ma_50 = mas.get('ma_50', price)
        
        if price > ma_20 > ma_50:
            return 'STRONG_BUY'
        elif price > ma_20:
            return 'BUY'
        elif price < ma_20 < ma_50:
            return 'STRONG_SELL'
        elif price < ma_20:
            return 'SELL'
        else:
            return 'NEUTRAL'
            
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate support and resistance levels"""
        try:
            highs = df['high'].values
            lows = df['low'].values
            
            # Find local maxima and minima
            resistance_levels = []
            support_levels = []
            
            window = 5
            for i in range(window, len(highs) - window):
                if highs[i] == max(highs[i-window:i+window+1]):
                    resistance_levels.append(highs[i])
                if lows[i] == min(lows[i-window:i+window+1]):
                    support_levels.append(lows[i])
                    
            # Get most recent levels
            resistance_levels = sorted(resistance_levels, reverse=True)[:3]
            support_levels = sorted(support_levels, reverse=True)[:3]
            
            return {
                'resistance': resistance_levels,
                'support': support_levels,
                'current_price': df['close'].iloc[-1]
            }
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return {'resistance': [], 'support': [], 'current_price': 0}
            
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume-based indicators"""
        try:
            # On-Balance Volume
            obv = talib.OBV(df['close'].values, df['volume'].values)
            
            # Volume Rate of Change
            volume_roc = talib.ROC(df['volume'].values, timeperiod=10)
            
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
            
            return {
                'obv': obv[-1] if len(obv) > 0 and not np.isnan(obv[-1]) else 0,
                'volume_roc': volume_roc[-1] if len(volume_roc) > 0 and not np.isnan(volume_roc[-1]) else 0,
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'volume_spike': current_volume > avg_volume * 1.5
            }
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
            return {'obv': 0, 'volume_roc': 0, 'current_volume': 0, 'avg_volume': 0, 'volume_spike': False}
            
    def _identify_candlestick_patterns(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Identify candlestick patterns"""
        try:
            patterns = {}
            
            # Doji
            patterns['doji'] = bool(talib.CDLDOJI(df['open'].values, df['high'].values, df['low'].values, df['close'].values)[-1])
            
            # Hammer
            patterns['hammer'] = bool(talib.CDLHAMMER(df['open'].values, df['high'].values, df['low'].values, df['close'].values)[-1])
            
            # Engulfing patterns
            patterns['bullish_engulfing'] = bool(talib.CDLENGULFING(df['open'].values, df['high'].values, df['low'].values, df['close'].values)[-1] > 0)
            patterns['bearish_engulfing'] = bool(talib.CDLENGULFING(df['open'].values, df['high'].values, df['low'].values, df['close'].values)[-1] < 0)
            
            # Morning/Evening Star
            patterns['morning_star'] = bool(talib.CDLMORNINGSTAR(df['open'].values, df['high'].values, df['low'].values, df['close'].values)[-1])
            patterns['evening_star'] = bool(talib.CDLEVENINGSTAR(df['open'].values, df['high'].values, df['low'].values, df['close'].values)[-1])
            
            return patterns
        except Exception as e:
            logger.error(f"Error identifying candlestick patterns: {e}")
            return {}
            
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall trend"""
        try:
            # Calculate trend strength using ADX
            adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            current_adx = adx[-1] if len(adx) > 0 and not np.isnan(adx[-1]) else 25
            
            # Calculate trend direction
            prices = df['close'].values
            if len(prices) >= 20:
                recent_trend = np.polyfit(range(20), prices[-20:], 1)[0]
                trend_direction = 'UPTREND' if recent_trend > 0 else 'DOWNTREND'
            else:
                trend_direction = 'NEUTRAL'
                
            return {
                'direction': trend_direction,
                'strength': current_adx,
                'strong_trend': current_adx > 25,
                'trend_score': self._calculate_trend_score(df)
            }
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return {'direction': 'NEUTRAL', 'strength': 25, 'strong_trend': False, 'trend_score': 0}
            
    def _calculate_trend_score(self, df: pd.DataFrame) -> float:
        """Calculate composite trend score"""
        try:
            score = 0
            
            # Price above moving averages
            current_price = df['close'].iloc[-1]
            ma_20 = talib.SMA(df['close'].values, timeperiod=20)[-1]
            ma_50 = talib.SMA(df['close'].values, timeperiod=50)[-1]
            
            if not np.isnan(ma_20) and current_price > ma_20:
                score += 1
            if not np.isnan(ma_50) and current_price > ma_50:
                score += 1
                
            # RSI in favorable range
            rsi = talib.RSI(df['close'].values, timeperiod=14)[-1]
            if not np.isnan(rsi) and 30 < rsi < 70:
                score += 1
                
            # MACD bullish
            macd, signal, _ = talib.MACD(df['close'].values)
            if len(macd) > 0 and not np.isnan(macd[-1]) and not np.isnan(signal[-1]):
                if macd[-1] > signal[-1]:
                    score += 1
                    
            return score / 4.0  # Normalize to 0-1
        except Exception as e:
            logger.error(f"Error calculating trend score: {e}")
            return 0.5
            
    def _calculate_volatility(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility measures"""
        try:
            # Average True Range
            atr = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            current_atr = atr[-1] if len(atr) > 0 and not np.isnan(atr[-1]) else 0
            
            # Historical volatility
            returns = df['close'].pct_change().dropna()
            hist_vol = returns.std() * np.sqrt(252)  # Annualized
            
            return {
                'atr': current_atr,
                'historical_volatility': hist_vol,
                'high_volatility': current_atr > np.mean(atr[-20:]) * 1.2 if len(atr) >= 20 else False
            }
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return {'atr': 0, 'historical_volatility': 0, 'high_volatility': False}
            
    def get_trading_signal(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal based on technical indicators"""
        try:
            signals = []
            confidence = 0
            
            # RSI signal
            rsi_data = indicators.get('rsi', {})
            if rsi_data.get('signal') == 'BUY':
                signals.append('BUY')
                confidence += 0.2
            elif rsi_data.get('signal') == 'SELL':
                signals.append('SELL')
                confidence += 0.2
                
            # MACD signal
            macd_data = indicators.get('macd', {})
            if macd_data.get('signal_type') == 'BUY':
                signals.append('BUY')
                confidence += 0.2
            elif macd_data.get('signal_type') == 'SELL':
                signals.append('SELL')
                confidence += 0.2
                
            # Bollinger Bands signal
            bb_data = indicators.get('bollinger', {})
            if bb_data.get('position') == 'BELOW_LOWER':
                signals.append('BUY')
                confidence += 0.15
            elif bb_data.get('position') == 'ABOVE_UPPER':
                signals.append('SELL')
                confidence += 0.15
                
            # Moving Average signal
            ma_data = indicators.get('moving_averages', {})
            ma_signal = ma_data.get('trend_signal', 'NEUTRAL')
            if 'BUY' in ma_signal:
                signals.append('BUY')
                confidence += 0.25
            elif 'SELL' in ma_signal:
                signals.append('SELL')
                confidence += 0.25
                
            # Trend signal
            trend_data = indicators.get('trend', {})
            if trend_data.get('direction') == 'UPTREND' and trend_data.get('strong_trend'):
                signals.append('BUY')
                confidence += 0.2
            elif trend_data.get('direction') == 'DOWNTREND' and trend_data.get('strong_trend'):
                signals.append('SELL')
                confidence += 0.2
                
            # Determine final signal
            buy_signals = signals.count('BUY')
            sell_signals = signals.count('SELL')
            
            if buy_signals > sell_signals:
                final_signal = 'BUY'
            elif sell_signals > buy_signals:
                final_signal = 'SELL'
            else:
                final_signal = 'NEUTRAL'
                
            return {
                'signal': final_signal,
                'confidence': min(confidence, 1.0),
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'total_signals': len(signals)
            }
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0, 'buy_signals': 0, 'sell_signals': 0, 'total_signals': 0}
