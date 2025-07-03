import numpy as np
import logging
from datetime import datetime
from .technical_analyzer import TechnicalAnalyzer

logger = logging.getLogger(__name__)

class StrategyEngine:
    """Advanced strategy engine implementing 15 sophisticated trading strategies"""
    
    def __init__(self):
        self.technical_analyzer = TechnicalAnalyzer()
        self.current_strategy = "Adaptive Mean Reversion Rebound"
        self.strategy_rotation = 0
        
        # Strategy performance tracking
        self.strategy_performance = {
            "Adaptive Mean Reversion Rebound": {"wins": 0, "losses": 0, "profit": 0.0},
            "RSI Momentum Breakout": {"wins": 0, "losses": 0, "profit": 0.0},
            "Bollinger Band Squeeze": {"wins": 0, "losses": 0, "profit": 0.0},
            "MACD Histogram Divergence": {"wins": 0, "losses": 0, "profit": 0.0},
            "Volatility Expansion Scalp": {"wins": 0, "losses": 0, "profit": 0.0},
            "Tick Velocity Momentum": {"wins": 0, "losses": 0, "profit": 0.0},
            "Support Resistance Bounce": {"wins": 0, "losses": 0, "profit": 0.0},
            "EMA Crossover Micro": {"wins": 0, "losses": 0, "profit": 0.0},
            "Williams R Extreme": {"wins": 0, "losses": 0, "profit": 0.0},
            "Stochastic Divergence": {"wins": 0, "losses": 0, "profit": 0.0},
            "Volume Price Trend": {"wins": 0, "losses": 0, "profit": 0.0},
            "Microtrend Reversal": {"wins": 0, "losses": 0, "profit": 0.0},
            "High Frequency Scalp": {"wins": 0, "losses": 0, "profit": 0.0},
            "Neural Pattern Recognition": {"wins": 0, "losses": 0, "profit": 0.0},
            "Adaptive Multi-Timeframe": {"wins": 0, "losses": 0, "profit": 0.0}
        }
    
    def analyze_adaptive_mean_reversion(self, rsi, volatility, momentum, bollinger_upper, bollinger_lower, macd, current_price):
        """
        Strategy 1: Adaptive Mean Reversion Rebound
        Objective: Profit from short-term reversals during neutral RSI and moderate volatility
        """
        try:
            # Strategy conditions
            rsi_neutral = 48 <= rsi <= 52
            volatility_moderate = 0.01 <= volatility <= 0.015  # 1-1.5%
            momentum_low = abs(momentum) < 0.2
            macd_flat = abs(macd) <= 0.1
            
            # Price position relative to Bollinger Bands
            bb_middle = (bollinger_upper + bollinger_lower) / 2
            
            signal = {'action': 'HOLD', 'confidence': 0.0}
            
            # Check for touch conditions
            if current_price <= bollinger_lower and rsi_neutral and volatility_moderate and momentum_low and macd_flat:
                # Touch lower band → Buy Rise after 1 green tick
                signal = {
                    'action': 'BUY_RISE',
                    'contract_type': 'rise_fall',
                    'duration': 6,  # 6 seconds
                    'confidence': 0.75,
                    'strategy': 'Adaptive Mean Reversion Rebound',
                    'entry_reason': 'Lower BB touch with neutral conditions'
                }
            
            elif current_price >= bollinger_upper and rsi_neutral and volatility_moderate and momentum_low and macd_flat:
                # Touch upper band → Buy Fall after 1 red tick
                signal = {
                    'action': 'BUY_FALL',
                    'contract_type': 'rise_fall',
                    'duration': 6,  # 6 seconds
                    'confidence': 0.75,
                    'strategy': 'Adaptive Mean Reversion Rebound',
                    'entry_reason': 'Upper BB touch with neutral conditions'
                }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in adaptive mean reversion strategy: {str(e)}")
            return {'action': 'HOLD', 'confidence': 0.0}
    
    def analyze_rsi_momentum_breakout(self, rsi, momentum, volatility, current_price, price_history):
        """
        Strategy 2: RSI Momentum Breakout
        Objective: Catch momentum when RSI breaks extreme levels with price confirmation
        """
        try:
            # Strategy conditions
            rsi_oversold = rsi < 25
            rsi_overbought = rsi > 75
            momentum_strong = abs(momentum) > 0.5
            volatility_active = volatility > 0.008  # Above 0.8%
            
            # Price momentum confirmation
            recent_prices = price_history[-5:] if len(price_history) >= 5 else price_history
            price_momentum = (recent_prices[-1] / recent_prices[0] - 1) * 100 if len(recent_prices) > 1 else 0
            
            signal = {'action': 'HOLD', 'confidence': 0.0}
            
            if rsi_oversold and momentum > 0.5 and volatility_active and price_momentum > 0.1:
                signal = {
                    'action': 'BUY_RISE',
                    'contract_type': 'rise_fall',
                    'duration': 8,
                    'confidence': 0.8,
                    'strategy': 'RSI Momentum Breakout',
                    'entry_reason': 'RSI oversold with bullish momentum'
                }
            
            elif rsi_overbought and momentum < -0.5 and volatility_active and price_momentum < -0.1:
                signal = {
                    'action': 'BUY_FALL',
                    'contract_type': 'rise_fall',
                    'duration': 8,
                    'confidence': 0.8,
                    'strategy': 'RSI Momentum Breakout',
                    'entry_reason': 'RSI overbought with bearish momentum'
                }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in RSI momentum breakout strategy: {str(e)}")
            return {'action': 'HOLD', 'confidence': 0.0}
    
    def analyze_bollinger_squeeze(self, bollinger_upper, bollinger_lower, volatility, rsi, price_history):
        """
        Strategy 3: Bollinger Band Squeeze
        Objective: Detect volatility compression and trade the breakout
        """
        try:
            # Calculate band width
            bb_width = (bollinger_upper - bollinger_lower) / ((bollinger_upper + bollinger_lower) / 2)
            
            # Strategy conditions
            squeeze_condition = bb_width < 0.02  # Tight bands
            low_volatility = volatility < 0.005  # Very low volatility
            rsi_neutral = 40 <= rsi <= 60
            
            # Price breakout detection
            current_price = price_history[-1] if price_history else 0
            bb_middle = (bollinger_upper + bollinger_lower) / 2
            
            signal = {'action': 'HOLD', 'confidence': 0.0}
            
            if squeeze_condition and low_volatility:
                # Wait for breakout
                if current_price > bollinger_upper:
                    signal = {
                        'action': 'BUY_RISE',
                        'contract_type': 'rise_fall',
                        'duration': 10,
                        'confidence': 0.85,
                        'strategy': 'Bollinger Band Squeeze',
                        'entry_reason': 'Upward breakout from squeeze'
                    }
                elif current_price < bollinger_lower:
                    signal = {
                        'action': 'BUY_FALL',
                        'contract_type': 'rise_fall',
                        'duration': 10,
                        'confidence': 0.85,
                        'strategy': 'Bollinger Band Squeeze',
                        'entry_reason': 'Downward breakout from squeeze'
                    }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in Bollinger squeeze strategy: {str(e)}")
            return {'action': 'HOLD', 'confidence': 0.0}
    
    def analyze_macd_histogram_divergence(self, price_history, macd_data):
        """
        Strategy 4: MACD Histogram Divergence
        Objective: Detect momentum divergence for early reversal signals
        """
        try:
            if len(price_history) < 10:
                return {'action': 'HOLD', 'confidence': 0.0}
            
            current_price = price_history[-1]
            prev_price = price_history[-5]
            
            macd_histogram = macd_data.get('histogram', 0)
            macd_line = macd_data.get('macd', 0)
            signal_line = macd_data.get('signal', 0)
            
            # Price trend
            price_trend = (current_price - prev_price) / prev_price
            
            # MACD conditions
            bullish_cross = macd_line > signal_line and macd_histogram > 0
            bearish_cross = macd_line < signal_line and macd_histogram < 0
            
            signal = {'action': 'HOLD', 'confidence': 0.0}
            
            # Bullish divergence: Price down, MACD up
            if price_trend < -0.001 and bullish_cross and macd_histogram > 0.001:
                signal = {
                    'action': 'BUY_RISE',
                    'contract_type': 'rise_fall',
                    'duration': 7,
                    'confidence': 0.78,
                    'strategy': 'MACD Histogram Divergence',
                    'entry_reason': 'Bullish MACD divergence'
                }
            
            # Bearish divergence: Price up, MACD down
            elif price_trend > 0.001 and bearish_cross and macd_histogram < -0.001:
                signal = {
                    'action': 'BUY_FALL',
                    'contract_type': 'rise_fall',
                    'duration': 7,
                    'confidence': 0.78,
                    'strategy': 'MACD Histogram Divergence',
                    'entry_reason': 'Bearish MACD divergence'
                }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in MACD divergence strategy: {str(e)}")
            return {'action': 'HOLD', 'confidence': 0.0}
    
    def analyze_volatility_expansion(self, volatility, volatility_history, rsi, current_price, bollinger_upper, bollinger_lower):
        """
        Strategy 5: Volatility Expansion Scalp
        Objective: Trade immediate moves when volatility suddenly expands
        """
        try:
            # Calculate volatility expansion
            if len(volatility_history) < 10:
                return {'action': 'HOLD', 'confidence': 0.0}
            
            avg_volatility = np.mean(volatility_history[-10:])
            volatility_expansion = volatility / avg_volatility if avg_volatility > 0 else 1
            
            # Strategy conditions
            expansion_threshold = volatility_expansion > 1.5  # 50% increase in volatility
            rsi_not_extreme = 30 < rsi < 70
            
            # Price position
            bb_middle = (bollinger_upper + bollinger_lower) / 2
            price_above_middle = current_price > bb_middle
            price_below_middle = current_price < bb_middle
            
            signal = {'action': 'HOLD', 'confidence': 0.0}
            
            if expansion_threshold and rsi_not_extreme:
                if price_above_middle and rsi < 60:
                    signal = {
                        'action': 'BUY_RISE',
                        'contract_type': 'rise_fall',
                        'duration': 5,
                        'confidence': 0.72,
                        'strategy': 'Volatility Expansion Scalp',
                        'entry_reason': 'Volatility expansion with upward bias'
                    }
                elif price_below_middle and rsi > 40:
                    signal = {
                        'action': 'BUY_FALL',
                        'contract_type': 'rise_fall',
                        'duration': 5,
                        'confidence': 0.72,
                        'strategy': 'Volatility Expansion Scalp',
                        'entry_reason': 'Volatility expansion with downward bias'
                    }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in volatility expansion strategy: {str(e)}")
            return {'action': 'HOLD', 'confidence': 0.0}
    
    def analyze_tick_velocity_momentum(self, price_history, timestamps, rsi):
        """
        Strategy 6: Tick Velocity Momentum
        Objective: Use tick-by-tick velocity to predict short-term direction
        """
        try:
            if len(price_history) < 15 or len(timestamps) < 15:
                return {'action': 'HOLD', 'confidence': 0.0}
            
            # Calculate tick velocities
            velocities = []
            for i in range(1, len(price_history)):
                time_diff = timestamps[i] - timestamps[i-1]
                price_diff = price_history[i] - price_history[i-1]
                if time_diff > 0:
                    velocity = price_diff / time_diff
                    velocities.append(velocity)
            
            if len(velocities) < 10:
                return {'action': 'HOLD', 'confidence': 0.0}
            
            # Calculate acceleration
            recent_velocity = np.mean(velocities[-5:])
            previous_velocity = np.mean(velocities[-10:-5])
            acceleration = recent_velocity - previous_velocity
            
            # Strategy conditions
            high_velocity = abs(recent_velocity) > 0.001
            positive_acceleration = acceleration > 0.0001
            negative_acceleration = acceleration < -0.0001
            rsi_supportive = 35 < rsi < 65
            
            signal = {'action': 'HOLD', 'confidence': 0.0}
            
            if high_velocity and rsi_supportive:
                if recent_velocity > 0 and positive_acceleration:
                    signal = {
                        'action': 'BUY_RISE',
                        'contract_type': 'rise_fall',
                        'duration': 6,
                        'confidence': 0.74,
                        'strategy': 'Tick Velocity Momentum',
                        'entry_reason': 'Positive velocity with acceleration'
                    }
                elif recent_velocity < 0 and negative_acceleration:
                    signal = {
                        'action': 'BUY_FALL',
                        'contract_type': 'rise_fall',
                        'duration': 6,
                        'confidence': 0.74,
                        'strategy': 'Tick Velocity Momentum',
                        'entry_reason': 'Negative velocity with acceleration'
                    }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in tick velocity strategy: {str(e)}")
            return {'action': 'HOLD', 'confidence': 0.0}
    
    def analyze_support_resistance_bounce(self, current_price, support_levels, resistance_levels, rsi, momentum):
        """
        Strategy 7: Support Resistance Bounce
        Objective: Trade bounces off established support/resistance levels
        """
        try:
            signal = {'action': 'HOLD', 'confidence': 0.0}
            
            # Find nearest support and resistance
            nearest_support = max([s for s in support_levels if s < current_price], default=None)
            nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)
            
            # Calculate distance to levels
            support_distance = abs(current_price - nearest_support) / current_price if nearest_support else float('inf')
            resistance_distance = abs(current_price - nearest_resistance) / current_price if nearest_resistance else float('inf')
            
            # Strategy conditions
            near_support = support_distance < 0.001  # Within 0.1%
            near_resistance = resistance_distance < 0.001  # Within 0.1%
            rsi_oversold = rsi < 35
            rsi_overbought = rsi > 65
            momentum_weak = abs(momentum) < 0.3
            
            # Support bounce
            if near_support and rsi_oversold and momentum_weak:
                signal = {
                    'action': 'BUY_RISE',
                    'contract_type': 'rise_fall',
                    'duration': 8,
                    'confidence': 0.82,
                    'strategy': 'Support Resistance Bounce',
                    'entry_reason': f'Bounce off support at {nearest_support}'
                }
            
            # Resistance bounce
            elif near_resistance and rsi_overbought and momentum_weak:
                signal = {
                    'action': 'BUY_FALL',
                    'contract_type': 'rise_fall',
                    'duration': 8,
                    'confidence': 0.82,
                    'strategy': 'Support Resistance Bounce',
                    'entry_reason': f'Bounce off resistance at {nearest_resistance}'
                }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in support resistance strategy: {str(e)}")
            return {'action': 'HOLD', 'confidence': 0.0}
    
    def analyze_ema_crossover_micro(self, price_history):
        """
        Strategy 8: EMA Crossover Micro
        Objective: Ultra-short EMA crossovers for 5-second trades
        """
        try:
            if len(price_history) < 20:
                return {'action': 'HOLD', 'confidence': 0.0}
            
            # Calculate short EMAs
            ema_3 = self.technical_analyzer._calculate_ema(price_history, 3)
            ema_8 = self.technical_analyzer._calculate_ema(price_history, 8)
            ema_13 = self.technical_analyzer._calculate_ema(price_history, 13)
            
            # Current and previous values
            current_ema3 = ema_3[-1]
            current_ema8 = ema_8[-1]
            current_ema13 = ema_13[-1]
            
            prev_ema3 = ema_3[-2]
            prev_ema8 = ema_8[-2]
            
            signal = {'action': 'HOLD', 'confidence': 0.0}
            
            # Bullish crossover: EMA3 crosses above EMA8 with EMA13 support
            if (current_ema3 > current_ema8 and prev_ema3 <= prev_ema8 and 
                current_ema8 > current_ema13):
                signal = {
                    'action': 'BUY_RISE',
                    'contract_type': 'rise_fall',
                    'duration': 5,
                    'confidence': 0.69,
                    'strategy': 'EMA Crossover Micro',
                    'entry_reason': 'Bullish EMA crossover'
                }
            
            # Bearish crossover: EMA3 crosses below EMA8 with EMA13 resistance
            elif (current_ema3 < current_ema8 and prev_ema3 >= prev_ema8 and 
                  current_ema8 < current_ema13):
                signal = {
                    'action': 'BUY_FALL',
                    'contract_type': 'rise_fall',
                    'duration': 5,
                    'confidence': 0.69,
                    'strategy': 'EMA Crossover Micro',
                    'entry_reason': 'Bearish EMA crossover'
                }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in EMA crossover strategy: {str(e)}")
            return {'action': 'HOLD', 'confidence': 0.0}
    
    def analyze_williams_r_extreme(self, price_history, rsi, volatility):
        """
        Strategy 9: Williams %R Extreme
        Objective: Capitalize on extreme oversold/overbought conditions
        """
        try:
            if len(price_history) < 14:
                return {'action': 'HOLD', 'confidence': 0.0}
            
            # Calculate Williams %R
            high_14 = max(price_history[-14:])
            low_14 = min(price_history[-14:])
            current_price = price_history[-1]
            
            williams_r = ((high_14 - current_price) / (high_14 - low_14)) * -100
            
            signal = {'action': 'HOLD', 'confidence': 0.0}
            
            # Extremely oversold conditions
            if williams_r <= -90 and rsi < 25 and volatility > 0.012:
                signal = {
                    'action': 'BUY_RISE',
                    'contract_type': 'rise_fall',
                    'duration': 3,
                    'confidence': 0.74,
                    'strategy': 'Williams R Extreme',
                    'entry_reason': f'Extreme oversold: Williams %R = {williams_r:.1f}'
                }
            
            # Extremely overbought conditions
            elif williams_r >= -10 and rsi > 75 and volatility > 0.012:
                signal = {
                    'action': 'BUY_FALL',
                    'contract_type': 'rise_fall',
                    'duration': 3,
                    'confidence': 0.74,
                    'strategy': 'Williams R Extreme',
                    'entry_reason': f'Extreme overbought: Williams %R = {williams_r:.1f}'
                }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in Williams R extreme strategy: {str(e)}")
            return {'action': 'HOLD', 'confidence': 0.0}
    
    def analyze_stochastic_divergence(self, price_history, rsi, momentum):
        """
        Strategy 10: Stochastic Divergence
        Objective: Detect divergences between price and stochastic oscillator
        """
        try:
            if len(price_history) < 20:
                return {'action': 'HOLD', 'confidence': 0.0}
            
            # Calculate Stochastic %K
            high_14 = [max(price_history[i-14:i]) for i in range(14, len(price_history))]
            low_14 = [min(price_history[i-14:i]) for i in range(14, len(price_history))]
            close_prices = price_history[14:]
            
            stoch_k = [((close - low) / (high - low)) * 100 
                      for close, high, low in zip(close_prices, high_14, low_14)]
            
            if len(stoch_k) < 6:
                return {'action': 'HOLD', 'confidence': 0.0}
            
            # Calculate %D (3-period SMA of %K)
            stoch_d = []
            for i in range(2, len(stoch_k)):
                stoch_d.append(np.mean(stoch_k[i-2:i+1]))
            
            if len(stoch_d) < 3:
                return {'action': 'HOLD', 'confidence': 0.0}
            
            signal = {'action': 'HOLD', 'confidence': 0.0}
            
            # Bullish divergence: Price making lower lows, stochastic making higher lows
            recent_prices = price_history[-5:]
            recent_stoch = stoch_k[-5:]
            
            price_trend = recent_prices[-1] - recent_prices[0]
            stoch_trend = recent_stoch[-1] - recent_stoch[0]
            
            # Stochastic crossover signals
            current_k = stoch_k[-1]
            current_d = stoch_d[-1]
            prev_k = stoch_k[-2]
            prev_d = stoch_d[-2]
            
            # Bullish: %K crosses above %D in oversold territory
            if (current_k > current_d and prev_k <= prev_d and 
                current_k < 30 and rsi < 40):
                signal = {
                    'action': 'BUY_RISE',
                    'contract_type': 'rise_fall',
                    'duration': 4,
                    'confidence': 0.71,
                    'strategy': 'Stochastic Divergence',
                    'entry_reason': 'Bullish stochastic crossover in oversold'
                }
            
            # Bearish: %K crosses below %D in overbought territory
            elif (current_k < current_d and prev_k >= prev_d and 
                  current_k > 70 and rsi > 60):
                signal = {
                    'action': 'BUY_FALL',
                    'contract_type': 'rise_fall',
                    'duration': 4,
                    'confidence': 0.71,
                    'strategy': 'Stochastic Divergence',
                    'entry_reason': 'Bearish stochastic crossover in overbought'
                }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in stochastic divergence strategy: {str(e)}")
            return {'action': 'HOLD', 'confidence': 0.0}
    
    def analyze_volume_price_trend(self, price_history, volume_proxy, rsi):
        """
        Strategy 11: Volume Price Trend
        Objective: Use volume-price relationships for signal confirmation
        """
        try:
            if len(price_history) < 10:
                return {'action': 'HOLD', 'confidence': 0.0}
            
            # Use price movement velocity as volume proxy
            price_changes = np.diff(price_history[-10:])
            volume_estimates = [abs(change) * 1000 for change in price_changes]
            
            if len(volume_estimates) < 5:
                return {'action': 'HOLD', 'confidence': 0.0}
            
            # Calculate Volume Price Trend
            vpt = 0
            vpt_values = []
            
            for i in range(len(price_changes)):
                if i > 0:
                    vpt += volume_estimates[i] * (price_changes[i] / price_history[-10+i])
                vpt_values.append(vpt)
            
            signal = {'action': 'HOLD', 'confidence': 0.0}
            
            # VPT trend analysis
            if len(vpt_values) >= 3:
                vpt_trend = vpt_values[-1] - vpt_values[-3]
                price_trend = price_history[-1] - price_history[-3]
                
                # Bullish: Rising VPT with rising price and oversold RSI
                if vpt_trend > 0 and price_trend > 0 and rsi < 35:
                    signal = {
                        'action': 'BUY_RISE',
                        'contract_type': 'rise_fall',
                        'duration': 5,
                        'confidence': 0.67,
                        'strategy': 'Volume Price Trend',
                        'entry_reason': 'Bullish VPT confirmation with oversold RSI'
                    }
                
                # Bearish: Falling VPT with falling price and overbought RSI
                elif vpt_trend < 0 and price_trend < 0 and rsi > 65:
                    signal = {
                        'action': 'BUY_FALL',
                        'contract_type': 'rise_fall',
                        'duration': 5,
                        'confidence': 0.67,
                        'strategy': 'Volume Price Trend',
                        'entry_reason': 'Bearish VPT confirmation with overbought RSI'
                    }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in volume price trend strategy: {str(e)}")
            return {'action': 'HOLD', 'confidence': 0.0}
    
    def analyze_microtrend_reversal(self, price_history, rsi, momentum, volatility):
        """
        Strategy 12: Microtrend Reversal
        Objective: Capture very short-term trend reversals using micro patterns
        """
        try:
            if len(price_history) < 8:
                return {'action': 'HOLD', 'confidence': 0.0}
            
            recent_prices = price_history[-8:]
            
            # Identify micro patterns
            def is_micro_uptrend(prices):
                return all(prices[i] >= prices[i-1] for i in range(1, min(4, len(prices))))
            
            def is_micro_downtrend(prices):
                return all(prices[i] <= prices[i-1] for i in range(1, min(4, len(prices))))
            
            signal = {'action': 'HOLD', 'confidence': 0.0}
            
            # Check for micro trend exhaustion
            if len(recent_prices) >= 6:
                first_half = recent_prices[:3]
                second_half = recent_prices[3:6]
                
                # Micro uptrend exhaustion
                if (is_micro_uptrend(first_half) and 
                    recent_prices[-1] < recent_prices[-2] and 
                    rsi > 55 and momentum < 0 and volatility > 0.008):
                    
                    signal = {
                        'action': 'BUY_FALL',
                        'contract_type': 'rise_fall',
                        'duration': 3,
                        'confidence': 0.73,
                        'strategy': 'Microtrend Reversal',
                        'entry_reason': 'Micro uptrend exhaustion detected'
                    }
                
                # Micro downtrend exhaustion
                elif (is_micro_downtrend(first_half) and 
                      recent_prices[-1] > recent_prices[-2] and 
                      rsi < 45 and momentum > 0 and volatility > 0.008):
                    
                    signal = {
                        'action': 'BUY_RISE',
                        'contract_type': 'rise_fall',
                        'duration': 3,
                        'confidence': 0.73,
                        'strategy': 'Microtrend Reversal',
                        'entry_reason': 'Micro downtrend exhaustion detected'
                    }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in microtrend reversal strategy: {str(e)}")
            return {'action': 'HOLD', 'confidence': 0.0}
    
    def analyze_high_frequency_scalp(self, price_history, tick_timestamps, rsi):
        """
        Strategy 13: High Frequency Scalp
        Objective: Exploit very short-term price inefficiencies
        """
        try:
            if len(price_history) < 6 or len(tick_timestamps) < 6:
                return {'action': 'HOLD', 'confidence': 0.0}
            
            recent_prices = price_history[-6:]
            recent_times = tick_timestamps[-6:]
            
            # Calculate tick-to-tick movements
            price_moves = np.diff(recent_prices)
            time_intervals = np.diff(recent_times)
            
            if len(price_moves) < 3:
                return {'action': 'HOLD', 'confidence': 0.0}
            
            # Calculate price acceleration
            price_velocity = price_moves / np.maximum(time_intervals, 0.001)
            price_acceleration = np.diff(price_velocity)
            
            signal = {'action': 'HOLD', 'confidence': 0.0}
            
            # High frequency scalping conditions
            if len(price_acceleration) >= 2:
                current_accel = price_acceleration[-1]
                current_velocity = price_velocity[-1]
                
                # Bullish scalp: Negative acceleration after positive velocity (deceleration)
                if (current_velocity > 0 and current_accel < -0.001 and 
                    rsi < 50 and abs(price_moves[-1]) > 0.0001):
                    
                    signal = {
                        'action': 'BUY_RISE',
                        'contract_type': 'rise_fall',
                        'duration': 2,
                        'confidence': 0.68,
                        'strategy': 'High Frequency Scalp',
                        'entry_reason': 'Bullish momentum deceleration scalp'
                    }
                
                # Bearish scalp: Positive acceleration after negative velocity
                elif (current_velocity < 0 and current_accel > 0.001 and 
                      rsi > 50 and abs(price_moves[-1]) > 0.0001):
                    
                    signal = {
                        'action': 'BUY_FALL',
                        'contract_type': 'rise_fall',
                        'duration': 2,
                        'confidence': 0.68,
                        'strategy': 'High Frequency Scalp',
                        'entry_reason': 'Bearish momentum deceleration scalp'
                    }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in high frequency scalp strategy: {str(e)}")
            return {'action': 'HOLD', 'confidence': 0.0}
    
    def analyze_neural_pattern_recognition(self, price_history, rsi, volatility, momentum):
        """
        Strategy 14: Neural Pattern Recognition
        Objective: Identify complex patterns using neural network-like logic
        """
        try:
            if len(price_history) < 12:
                return {'action': 'HOLD', 'confidence': 0.0}
            
            recent_prices = np.array(price_history[-12:])
            
            # Create pattern features
            features = []
            
            # Price normalization
            price_mean = np.mean(recent_prices)
            price_std = np.std(recent_prices)
            normalized_prices = (recent_prices - price_mean) / max(price_std, 0.0001)
            
            # Pattern recognition features
            # 1. Trend strength
            trend_strength = np.corrcoef(np.arange(len(normalized_prices)), normalized_prices)[0, 1]
            
            # 2. Volatility pattern
            price_changes = np.diff(normalized_prices)
            volatility_pattern = np.std(price_changes)
            
            # 3. Momentum pattern
            momentum_pattern = np.mean(price_changes[-3:]) - np.mean(price_changes[:3])
            
            # 4. Oscillation pattern
            peaks = sum(1 for i in range(1, len(normalized_prices)-1) 
                       if normalized_prices[i] > normalized_prices[i-1] and 
                          normalized_prices[i] > normalized_prices[i+1])
            valleys = sum(1 for i in range(1, len(normalized_prices)-1) 
                         if normalized_prices[i] < normalized_prices[i-1] and 
                            normalized_prices[i] < normalized_prices[i+1])
            
            oscillation_pattern = (peaks + valleys) / len(normalized_prices)
            
            features = [trend_strength, volatility_pattern, momentum_pattern, oscillation_pattern, rsi/100, volatility*100, momentum]
            
            # Simple neural network-like decision making
            signal = {'action': 'HOLD', 'confidence': 0.0}
            
            # Weighted decision function
            weights_bull = [0.3, -0.2, 0.4, -0.1, -0.3, 0.2, 0.3]
            weights_bear = [-0.3, -0.2, -0.4, -0.1, 0.3, 0.2, -0.3]
            
            bull_score = sum(w * f for w, f in zip(weights_bull, features))
            bear_score = sum(w * f for w, f in zip(weights_bear, features))
            
            # Apply activation function (sigmoid-like)
            def activation(x):
                return 1 / (1 + np.exp(-x * 5))
            
            bull_prob = activation(bull_score)
            bear_prob = activation(bear_score)
            
            # Decision threshold
            if bull_prob > 0.75 and bull_prob > bear_prob + 0.1:
                signal = {
                    'action': 'BUY_RISE',
                    'contract_type': 'rise_fall',
                    'duration': 4,
                    'confidence': min(0.75, bull_prob),
                    'strategy': 'Neural Pattern Recognition',
                    'entry_reason': f'Neural bullish pattern (score: {bull_prob:.2f})'
                }
            
            elif bear_prob > 0.75 and bear_prob > bull_prob + 0.1:
                signal = {
                    'action': 'BUY_FALL',
                    'contract_type': 'rise_fall',
                    'duration': 4,
                    'confidence': min(0.75, bear_prob),
                    'strategy': 'Neural Pattern Recognition',
                    'entry_reason': f'Neural bearish pattern (score: {bear_prob:.2f})'
                }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in neural pattern recognition strategy: {str(e)}")
            return {'action': 'HOLD', 'confidence': 0.0}
    
    def analyze_adaptive_multi_timeframe(self, price_history_1m, price_history_5m, rsi, macd, momentum):
        """
        Strategy 15: Adaptive Multi-Timeframe
        Objective: Combine signals from multiple timeframes for robust decisions
        """
        try:
            if len(price_history_1m) < 10 or len(price_history_5m) < 5:
                return {'action': 'HOLD', 'confidence': 0.0}
            
            # Short-term analysis (1-minute)
            short_trend = price_history_1m[-1] - price_history_1m[-5]
            short_volatility = np.std(np.diff(price_history_1m[-10:]))
            
            # Medium-term analysis (5-minute simulated)
            medium_trend = price_history_5m[-1] - price_history_5m[-3]
            medium_volatility = np.std(np.diff(price_history_5m[-5:]))
            
            signal = {'action': 'HOLD', 'confidence': 0.0}
            
            # Multi-timeframe confluence
            # Bullish confluence
            bullish_signals = 0
            bearish_signals = 0
            
            # Short-term signals
            if short_trend > 0 and momentum > 0:
                bullish_signals += 1
            elif short_trend < 0 and momentum < 0:
                bearish_signals += 1
            
            # Medium-term signals
            if medium_trend > 0:
                bullish_signals += 1
            elif medium_trend < 0:
                bearish_signals += 1
            
            # RSI signals
            if rsi < 35:
                bullish_signals += 1
            elif rsi > 65:
                bearish_signals += 1
            
            # MACD signals
            if macd > 0 and momentum > 0:
                bullish_signals += 1
            elif macd < 0 and momentum < 0:
                bearish_signals += 1
            
            # Volatility filter
            volatility_favorable = (short_volatility > 0.008 and 
                                  medium_volatility > 0.005 and 
                                  short_volatility < 0.02)
            
            # Decision based on confluence
            if bullish_signals >= 3 and bearish_signals <= 1 and volatility_favorable:
                confidence = min(0.78, 0.6 + (bullish_signals * 0.05))
                signal = {
                    'action': 'BUY_RISE',
                    'contract_type': 'rise_fall',
                    'duration': 6,
                    'confidence': confidence,
                    'strategy': 'Adaptive Multi-Timeframe',
                    'entry_reason': f'Multi-timeframe bullish confluence ({bullish_signals} signals)'
                }
            
            elif bearish_signals >= 3 and bullish_signals <= 1 and volatility_favorable:
                confidence = min(0.78, 0.6 + (bearish_signals * 0.05))
                signal = {
                    'action': 'BUY_FALL',
                    'contract_type': 'rise_fall',
                    'duration': 6,
                    'confidence': confidence,
                    'strategy': 'Adaptive Multi-Timeframe',
                    'entry_reason': f'Multi-timeframe bearish confluence ({bearish_signals} signals)'
                }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in adaptive multi-timeframe strategy: {str(e)}")
            return {'action': 'HOLD', 'confidence': 0.0}
    
    def get_best_strategy(self):
        """Get the best performing strategy"""
        best_strategy = max(self.strategy_performance.items(), 
                          key=lambda x: x[1]['profit'])
        return best_strategy[0]
    
    def update_strategy_performance(self, strategy_name, profit_loss, success):
        """Update strategy performance metrics"""
        if strategy_name in self.strategy_performance:
            self.strategy_performance[strategy_name]['profit'] += profit_loss
            if success:
                self.strategy_performance[strategy_name]['wins'] += 1
            else:
                self.strategy_performance[strategy_name]['losses'] += 1
    
    def rotate_strategy(self):
        """Rotate to next strategy for testing"""
        strategies = list(self.strategy_performance.keys())
        current_index = strategies.index(self.current_strategy)
        next_index = (current_index + 1) % len(strategies)
        self.current_strategy = strategies[next_index]
        return self.current_strategy
