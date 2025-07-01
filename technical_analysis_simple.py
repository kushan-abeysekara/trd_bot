"""
Simplified Technical Analysis module for trading signals (no TA-Lib dependency)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime, timedelta
import config

logger = logging.getLogger(__name__)

# Native implementations of technical indicators
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

def calculate_sma(prices, period):
    """Calculate Simple Moving Average"""
    return prices.rolling(window=period).mean()

def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    return prices.ewm(span=period).mean()

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()

class TechnicalAnalyzer:
    def __init__(self):
        self.config = config.TECHNICAL_CONFIG
        
    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate various technical indicators"""
        indicators = {}
        
        try:
            # Ensure we have enough data
            if len(df) < 50:
                logger.warning("Insufficient data for technical analysis")
                return self._get_default_indicators()
            
            # Basic indicators
            indicators['rsi'] = self._calculate_rsi(df)
            indicators['macd'] = self._calculate_macd(df)
            indicators['bollinger'] = self._calculate_bollinger_bands(df)
            indicators['moving_averages'] = self._calculate_moving_averages(df)
            indicators['volatility'] = self._calculate_volatility(df)
            indicators['trend'] = self._calculate_trend(df)
            indicators['support_resistance'] = self._calculate_support_resistance(df)
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return self._get_default_indicators()
            
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
            
            current_macd = macd.iloc[-1] if len(macd) > 0 and not np.isnan(macd.iloc[-1]) else 0
            current_signal = macd_signal.iloc[-1] if len(macd_signal) > 0 and not np.isnan(macd_signal.iloc[-1]) else 0
            current_histogram = macd_hist.iloc[-1] if len(macd_hist) > 0 and not np.isnan(macd_hist.iloc[-1]) else 0
            
            return {
                'macd': current_macd,
                'signal': current_signal,
                'histogram': current_histogram,
                'signal_type': 'BUY' if current_macd > current_signal else 'SELL',
                'divergence': self._check_macd_divergence(macd_hist)
            }
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return {'macd': 0, 'signal': 0, 'histogram': 0, 'signal_type': 'NEUTRAL', 'divergence': False}
            
    def _check_macd_divergence(self, macd_hist: pd.Series) -> bool:
        """Check for MACD divergence"""
        try:
            if len(macd_hist) < 10:
                return False
            recent_hist = macd_hist.iloc[-10:]
            return abs(recent_hist.iloc[-1] - recent_hist.iloc[0]) > recent_hist.std()
        except:
            return False
            
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
                'width': current_upper - current_lower,
                'position': self._get_bb_position(current_price, current_upper, current_middle, current_lower),
                'squeeze': self._check_bb_squeeze(upper, lower)
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            price = df['close'].iloc[-1] if len(df) > 0 else 100
            return {
                'upper': price * 1.02,
                'middle': price,
                'lower': price * 0.98,
                'width': price * 0.04,
                'position': 'MIDDLE',
                'squeeze': False
            }
            
    def _get_bb_position(self, price: float, upper: float, middle: float, lower: float) -> str:
        """Get Bollinger Band position"""
        if price > upper:
            return 'ABOVE_UPPER'
        elif price < lower:
            return 'BELOW_LOWER'
        elif price > middle:
            return 'UPPER_HALF'
        else:
            return 'LOWER_HALF'
            
    def _check_bb_squeeze(self, upper: pd.Series, lower: pd.Series) -> bool:
        """Check for Bollinger Band squeeze"""
        try:
            if len(upper) < 20:
                return False
            current_width = upper.iloc[-1] - lower.iloc[-1]
            avg_width = (upper.iloc[-20:] - lower.iloc[-20:]).mean()
            return current_width < avg_width * 0.8
        except:
            return False
            
    def _calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate various moving averages"""
        try:
            mas = {}
            for period in self.config['ma_periods']:
                ma = calculate_sma(df['close'], period)
                mas[f'sma_{period}'] = ma.iloc[-1] if len(ma) > 0 and not np.isnan(ma.iloc[-1]) else df['close'].iloc[-1]
                
            # Calculate trend signal based on MA alignment
            current_price = df['close'].iloc[-1]
            ma_10 = mas.get('sma_10', current_price)
            ma_20 = mas.get('sma_20', current_price)
            ma_50 = mas.get('sma_50', current_price)
            
            if ma_10 > ma_20 > ma_50 and current_price > ma_10:
                trend_signal = 'STRONG_BUY'
            elif ma_10 > ma_20 and current_price > ma_10:
                trend_signal = 'BUY'
            elif ma_10 < ma_20 < ma_50 and current_price < ma_10:
                trend_signal = 'STRONG_SELL'
            elif ma_10 < ma_20 and current_price < ma_10:
                trend_signal = 'SELL'
            else:
                trend_signal = 'NEUTRAL'
                
            mas['trend_signal'] = trend_signal
            return mas
            
        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
            return {'sma_10': df['close'].iloc[-1], 'sma_20': df['close'].iloc[-1], 
                   'sma_50': df['close'].iloc[-1], 'trend_signal': 'NEUTRAL'}
            
    def _calculate_volatility(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility metrics"""
        try:
            # ATR
            atr = calculate_atr(df['high'], df['low'], df['close'])
            current_atr = atr.iloc[-1] if len(atr) > 0 and not np.isnan(atr.iloc[-1]) else 0
            
            # Historical volatility
            returns = df['close'].pct_change()
            hist_vol = returns.std() * np.sqrt(252)  # Annualized
            
            return {
                'atr': current_atr,
                'historical_volatility': hist_vol,
                'volatility_percentile': self._get_volatility_percentile(returns)
            }
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return {'atr': 1.0, 'historical_volatility': 0.1, 'volatility_percentile': 50}
            
    def _get_volatility_percentile(self, returns: pd.Series) -> float:
        """Get volatility percentile"""
        try:
            if len(returns) < 20:
                return 50
            rolling_vol = returns.rolling(window=20).std()
            current_vol = rolling_vol.iloc[-1]
            return (rolling_vol < current_vol).mean() * 100
        except:
            return 50
            
    def _calculate_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trend indicators"""
        try:
            # Simple trend based on price direction
            if len(df) < 10:
                return {'direction': 'NEUTRAL', 'strength': 0, 'strong_trend': False}
                
            recent_prices = df['close'].iloc[-10:]
            trend_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            
            # Normalize trend strength
            price_range = df['close'].iloc[-20:].max() - df['close'].iloc[-20:].min()
            trend_strength = abs(trend_slope) / price_range if price_range > 0 else 0
            
            if trend_slope > 0:
                direction = 'UPTREND'
            elif trend_slope < 0:
                direction = 'DOWNTREND'
            else:
                direction = 'NEUTRAL'
                
            return {
                'direction': direction,
                'strength': trend_strength,
                'strong_trend': trend_strength > 0.1
            }
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return {'direction': 'NEUTRAL', 'strength': 0, 'strong_trend': False}
            
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate support and resistance levels"""
        try:
            if len(df) < 20:
                current_price = df['close'].iloc[-1]
                return {
                    'support_levels': [current_price * 0.99],
                    'resistance_levels': [current_price * 1.01]
                }
                
            # Find local minima and maxima
            highs = df['high'].iloc[-50:]
            lows = df['low'].iloc[-50:]
            
            # Simple support/resistance calculation
            support_levels = []
            resistance_levels = []
            
            # Recent lows as support
            for i in range(2, len(lows) - 2):
                if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                    support_levels.append(lows.iloc[i])
                    
            # Recent highs as resistance  
            for i in range(2, len(highs) - 2):
                if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                    resistance_levels.append(highs.iloc[i])
                    
            # Keep only the most relevant levels
            support_levels = sorted(support_levels, reverse=True)[:3]
            resistance_levels = sorted(resistance_levels)[:3]
            
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels
            }
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            current_price = df['close'].iloc[-1] if len(df) > 0 else 100
            return {
                'support_levels': [current_price * 0.99],
                'resistance_levels': [current_price * 1.01]
            }
            
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
            return {
                'signal': 'NEUTRAL',
                'confidence': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'total_signals': 0
            }
            
    def _get_default_indicators(self) -> Dict[str, Any]:
        """Get default indicators when calculation fails"""
        return {
            'rsi': {'value': 50, 'signal': 'NEUTRAL', 'overbought': False, 'oversold': False},
            'macd': {'macd': 0, 'signal': 0, 'histogram': 0, 'signal_type': 'NEUTRAL', 'divergence': False},
            'bollinger': {'upper': 100, 'middle': 100, 'lower': 100, 'width': 0, 'position': 'MIDDLE', 'squeeze': False},
            'moving_averages': {'sma_10': 100, 'sma_20': 100, 'sma_50': 100, 'trend_signal': 'NEUTRAL'},
            'volatility': {'atr': 1.0, 'historical_volatility': 0.1, 'volatility_percentile': 50},
            'trend': {'direction': 'NEUTRAL', 'strength': 0, 'strong_trend': False},
            'support_resistance': {'support_levels': [99], 'resistance_levels': [101]}
        }
