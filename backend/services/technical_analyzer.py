import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """Advanced technical analysis for high-frequency trading strategies"""
    
    def __init__(self):
        self.indicators_cache = {}
        self.last_update = None
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            prices = np.array(prices)
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            for i in range(period, len(deltas)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi)
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return 50.0
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            if len(prices) < slow:
                return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
            
            prices = np.array(prices)
            
            # Calculate EMAs
            ema_fast = self._calculate_ema(prices, fast)
            ema_slow = self._calculate_ema(prices, slow)
            
            # MACD line
            macd_line = ema_fast - ema_slow
            
            # Signal line
            signal_line = self._calculate_ema(macd_line, signal)
            
            # Histogram
            histogram = macd_line - signal_line
            
            return {
                'macd': float(macd_line[-1]),
                'signal': float(signal_line[-1]),
                'histogram': float(histogram[-1])
            }
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        try:
            if len(prices) < period:
                return {'upper': prices, 'middle': prices, 'lower': prices}
            
            prices = np.array(prices)
            
            # Calculate moving average and standard deviation
            sma = self._calculate_sma(prices, period)
            std = self._calculate_rolling_std(prices, period)
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            return {
                'upper': upper_band,
                'middle': sma,
                'lower': lower_band
            }
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return {'upper': prices, 'middle': prices, 'lower': prices}
    
    def calculate_momentum(self, prices, period=10):
        """Calculate price momentum"""
        try:
            if len(prices) < period + 1:
                return 0.0
            
            prices = np.array(prices)
            momentum = (prices[-1] / prices[-period-1] - 1) * 100
            
            return float(momentum)
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {str(e)}")
            return 0.0
    
    def calculate_volatility(self, prices, period=20):
        """Calculate price volatility (standard deviation of returns)"""
        try:
            if len(prices) < period:
                return 0.0
            
            prices = np.array(prices)
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns[-period:])
            
            return float(volatility)
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return 0.0
    
    def calculate_tick_velocity(self, prices, timestamps, period=10):
        """Calculate tick velocity (price change per time unit)"""
        try:
            if len(prices) < period or len(timestamps) < period:
                return 0.0
            
            prices = np.array(prices[-period:])
            timestamps = np.array(timestamps[-period:])
            
            # Convert timestamps to seconds
            time_diffs = np.diff(timestamps)
            price_diffs = np.diff(prices)
            
            # Calculate average velocity
            velocities = price_diffs / time_diffs
            avg_velocity = np.mean(velocities)
            
            return float(avg_velocity)
            
        except Exception as e:
            logger.error(f"Error calculating tick velocity: {str(e)}")
            return 0.0
    
    def calculate_support_resistance(self, prices, period=50, min_touches=3):
        """Calculate support and resistance levels"""
        try:
            if len(prices) < period:
                return {'support': [], 'resistance': []}
            
            prices = np.array(prices[-period:])
            
            # Find local minima and maxima
            minima = []
            maxima = []
            
            for i in range(2, len(prices) - 2):
                # Local minimum
                if (prices[i] < prices[i-1] and prices[i] < prices[i+1] and
                    prices[i] < prices[i-2] and prices[i] < prices[i+2]):
                    minima.append(prices[i])
                
                # Local maximum
                if (prices[i] > prices[i-1] and prices[i] > prices[i+1] and
                    prices[i] > prices[i-2] and prices[i] > prices[i+2]):
                    maxima.append(prices[i])
            
            # Cluster similar levels
            support_levels = self._cluster_levels(minima, tolerance=0.001)
            resistance_levels = self._cluster_levels(maxima, tolerance=0.001)
            
            # Filter by minimum touches
            support = [level for level, count in support_levels if count >= min_touches]
            resistance = [level for level, count in resistance_levels if count >= min_touches]
            
            return {
                'support': support,
                'resistance': resistance
            }
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {str(e)}")
            return {'support': [], 'resistance': []}
    
    def calculate_williams_r(self, highs, lows, closes, period=14):
        """Calculate Williams %R oscillator"""
        try:
            if len(highs) < period or len(lows) < period or len(closes) < period:
                return -50.0
            
            highs = np.array(highs[-period:])
            lows = np.array(lows[-period:])
            closes = np.array(closes)
            
            highest_high = np.max(highs)
            lowest_low = np.min(lows)
            current_close = closes[-1]
            
            williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100
            
            return float(williams_r)
            
        except Exception as e:
            logger.error(f"Error calculating Williams %R: {str(e)}")
            return -50.0
    
    def calculate_stochastic(self, highs, lows, closes, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        try:
            if len(highs) < k_period or len(lows) < k_period or len(closes) < k_period:
                return {'k': 50.0, 'd': 50.0}
            
            highs = np.array(highs)
            lows = np.array(lows)
            closes = np.array(closes)
            
            k_values = []
            
            for i in range(k_period - 1, len(closes)):
                highest_high = np.max(highs[i-k_period+1:i+1])
                lowest_low = np.min(lows[i-k_period+1:i+1])
                current_close = closes[i]
                
                if highest_high == lowest_low:
                    k = 50.0
                else:
                    k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
                
                k_values.append(k)
            
            # Calculate %D (moving average of %K)
            if len(k_values) >= d_period:
                d = np.mean(k_values[-d_period:])
            else:
                d = k_values[-1] if k_values else 50.0
            
            return {
                'k': float(k_values[-1]) if k_values else 50.0,
                'd': float(d)
            }
            
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {str(e)}")
            return {'k': 50.0, 'd': 50.0}
    
    def _calculate_ema(self, prices, period):
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.array(prices)
        
        prices = np.array(prices)
        multiplier = 2.0 / (period + 1)
        ema = [prices[0]]
        
        for i in range(1, len(prices)):
            ema.append((prices[i] * multiplier) + (ema[-1] * (1 - multiplier)))
        
        return np.array(ema)
    
    def _calculate_sma(self, prices, period):
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return np.array(prices)
        
        prices = np.array(prices)
        sma = []
        
        for i in range(period - 1, len(prices)):
            sma.append(np.mean(prices[i-period+1:i+1]))
        
        # Pad the beginning with the first calculated value
        first_sma = sma[0] if sma else prices[0]
        result = [first_sma] * (period - 1) + sma
        
        return np.array(result)
    
    def _calculate_rolling_std(self, prices, period):
        """Calculate rolling standard deviation"""
        if len(prices) < period:
            return np.zeros(len(prices))
        
        prices = np.array(prices)
        std_values = []
        
        for i in range(period - 1, len(prices)):
            std_values.append(np.std(prices[i-period+1:i+1]))
        
        # Pad the beginning with the first calculated value
        first_std = std_values[0] if std_values else 0.0
        result = [first_std] * (period - 1) + std_values
        
        return np.array(result)
    
    def _cluster_levels(self, levels, tolerance=0.001):
        """Cluster similar price levels"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) <= tolerance * current_cluster[-1]:
                current_cluster.append(level)
            else:
                # Finalize current cluster
                cluster_avg = np.mean(current_cluster)
                clusters.append((cluster_avg, len(current_cluster)))
                current_cluster = [level]
        
        # Add the last cluster
        if current_cluster:
            cluster_avg = np.mean(current_cluster)
            clusters.append((cluster_avg, len(current_cluster)))
        
        return clusters
