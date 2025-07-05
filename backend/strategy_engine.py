import numpy as np
import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class TickData:
    """Single tick data structure"""
    price: float
    timestamp: float
    color: str  # 'green' or 'red'
    volume: float = 0.0


@dataclass
class TechnicalIndicators:
    """Technical indicators for current market state"""
    rsi: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    momentum: float = 0.0
    volatility: float = 1.0
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_middle: float = 0.0
    ema3: float = 0.0
    ema5: float = 0.0
    ema20: float = 0.0
    rsi_previous: float = 50.0  # For tracking RSI changes


@dataclass
class TradeSignal:
    """Trade signal from strategy"""
    strategy_name: str
    direction: str  # 'CALL' or 'PUT'
    confidence: float  # 0.0 to 1.0
    hold_time: int  # seconds
    entry_reason: str
    conditions_met: List[str]


class StrategyEngine:
    """Advanced strategy engine with 35 binary trading strategies"""
    
    def __init__(self, max_ticks: int = 200):
        self.max_ticks = max_ticks
        self.tick_history = deque(maxlen=max_ticks)
        self.indicators = TechnicalIndicators()
        self.last_update = 0
        self.is_running = False
        self.signal_callback = None
        self.last_strategy_scan = {}
        self.strategy_performance = {}
        self.total_scans = 0
        self.signals_generated = 0
        
        # Strategy configurations - All 35 strategies
        self.strategies = {
            1: "Adaptive Mean Reversion Rebound",
            2: "Micro-Trend Momentum Tracker", 
            3: "RSI-Tick Divergence Detector",
            4: "Volatility Spike Fader",
            5: "Tick Flow Strength Pulse",
            6: "Double Confirmation Breakout",
            7: "RSI Overextension Fade",
            8: "Multi-Tick Pivot Bounce",
            9: "MACD-Momentum Sync Engine",
            10: "Time-of-Tick Scalper",
            11: "Volatility Collapse Compression",
            12: "Two-Step Confirmation Model",
            13: "Inverted Divergence Flip",
            14: "Cumulative Strength Index Pullback",
            15: "Tri-Indicator Confluence Strategy",
            16: "RSI Stall Reversal",
            17: "Tick Flow Momentum Ride",
            18: "Divergence Snapback",
            19: "Volatility Breakout Tick Rejection",
            20: "Triple Confirmation Flow",
            21: "Tick Trap Reversal",
            22: "Bollinger Bounce Magnet",
            23: "EMA Compression Breakout",
            24: "Tick RSI Bounce",
            25: "Tick Pulse Sync",
            26: "EMA Magnet Pullback",
            27: "RSI Mirror Flip",
            28: "MACD Crossover Trigger",
            29: "Volatility Expansion Ride",
            30: "RSI Gradient Tilt",
            31: "Rebound from Flat Session",
            32: "Opposite Color Flush",
            33: "RSI Ghost Divergence",
            34: "Triple Tick Momentum Snap",
            35: "Hybrid Confluence Gate"
        }
        
    def start_scanning(self, signal_callback):
        """Start real-time strategy scanning"""
        self.signal_callback = signal_callback
        self.is_running = True
        
        scan_thread = threading.Thread(target=self._scan_loop)
        scan_thread.daemon = True
        scan_thread.start()
        
    def stop_scanning(self):
        """Stop strategy scanning"""
        self.is_running = False
        
    def add_tick(self, price: float, timestamp: float = None):
        """Add new tick data and trigger analysis"""
        if timestamp is None:
            timestamp = time.time()
            
        # Determine tick color
        color = 'green'
        if len(self.tick_history) > 0:
            last_price = self.tick_history[-1].price
            color = 'green' if price > last_price else 'red'
            
        tick = TickData(price, timestamp, color)
        self.tick_history.append(tick)
        
        # Update indicators
        self._update_indicators()
        
        # Scan for signals
        if len(self.tick_history) >= 20:  # Need minimum ticks for analysis
            self._scan_strategies()
            
    def _update_indicators(self):
        """Calculate technical indicators from tick history"""
        if len(self.tick_history) < 20:
            return
            
        prices = np.array([tick.price for tick in self.tick_history])
        
        # Store previous RSI for change tracking
        self.indicators.rsi_previous = self.indicators.rsi
        
        # RSI calculation
        self.indicators.rsi = self._calculate_rsi(prices, 14)
        
        # MACD calculation
        self.indicators.macd, self.indicators.macd_signal = self._calculate_macd(prices)
        
        # Momentum calculation
        self.indicators.momentum = self._calculate_momentum(prices, 10)
        
        # Volatility calculation
        self.indicators.volatility = self._calculate_volatility(prices, 20)
        
        # Bollinger Bands
        self.indicators.bb_upper, self.indicators.bb_middle, self.indicators.bb_lower = self._calculate_bollinger_bands(prices, 20)
        
        # EMAs
        self.indicators.ema3 = self._calculate_ema(prices, 3)
        self.indicators.ema5 = self._calculate_ema(prices, 5)
        self.indicators.ema20 = self._calculate_ema(prices, 20)
        
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.mean(gains[-period:])
        avg_losses = np.mean(losses[-period:])
        
        if avg_losses == 0:
            return 100.0
            
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_macd(self, prices: np.ndarray) -> Tuple[float, float]:
        """Calculate MACD and signal line"""
        if len(prices) < 26:
            return 0.0, 0.0
            
        ema12 = self._calculate_ema(prices, 12)
        ema26 = self._calculate_ema(prices, 26)
        macd = ema12 - ema26
        
        # Simple signal line (could be improved with EMA of MACD)
        signal = macd * 0.9  # Simplified
        
        return macd, signal
        
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
            
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
            
        return ema
        
    def _calculate_momentum(self, prices: np.ndarray, period: int = 10) -> float:
        """Calculate price momentum as percentage change"""
        if len(prices) < period + 1:
            return 0.0
            
        current_price = prices[-1]
        past_price = prices[-period-1]
        
        momentum = ((current_price - past_price) / past_price) * 100
        return momentum
        
    def _calculate_volatility(self, prices: np.ndarray, period: int = 20) -> float:
        """Calculate volatility as standard deviation percentage"""
        if len(prices) < period:
            return 1.0
            
        recent_prices = prices[-period:]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns) * 100
        
        return volatility
        
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            mean_price = np.mean(prices)
            return mean_price * 1.02, mean_price, mean_price * 0.98
            
        recent_prices = prices[-period:]
        middle = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper = middle + (2 * std)
        lower = middle - (2 * std)
        
        return upper, middle, lower
        
    def _scan_loop(self):
        """Main scanning loop for real-time strategy analysis - OPTIMIZED"""
        while self.is_running:
            try:
                if len(self.tick_history) >= 20:
                    self.total_scans += 1
                    self._scan_strategies()
                time.sleep(0.03)  # Very fast scanning - every 30ms for responsive trading
            except Exception as e:
                print(f"⚠️  Strategy scan error: {e}")
                time.sleep(0.05)  # Brief pause on error
            
    def _scan_strategies(self):
        """Scan all 35 strategies for trade signals"""
        signals = []
        strategy_status = {}
        
        # Add force signal mechanism if enabled in config
        try:
            import config
            if hasattr(config, 'FORCE_STRATEGY_SIGNALS') and config.FORCE_STRATEGY_SIGNALS:
                # Force at least one signal every 10 scans if no signals generated
                if self.total_scans % 10 == 0 and len(signals) == 0:
                    force_signal = self._generate_force_signal()
                    if force_signal:
                        signals.append(force_signal)
                        print(f"🔧 FORCE SIGNAL GENERATED: {force_signal.strategy_name}")
        except:
            pass
        
        # Scan all 35 strategies in real-time
        for i in range(1, 36):
            try:
                signal = getattr(self, f'_strategy_{i}')()
                strategy_status[i] = {
                    'name': self.strategies[i],
                    'active': signal is not None,
                    'signal': signal
                }
                if signal:
                    signals.append(signal)
            except Exception as e:
                strategy_status[i] = {
                    'name': self.strategies[i],
                    'active': False,
                    'error': str(e)
                }
        
        # Store strategy status for monitoring
        self.last_strategy_scan = strategy_status
        
        # Handle multiple signals - send all high-confidence signals
        if signals and self.signal_callback:
            self.signals_generated += len(signals)
            
            # Sort by confidence (highest first)
            signals.sort(key=lambda x: x.confidence, reverse=True)
            
            # Send signals with confidence > 0.55 (reduced threshold for more trading)
            high_confidence_signals = [s for s in signals if s.confidence > 0.55]
            
            if high_confidence_signals:
                # Send the best signal immediately
                best_signal = high_confidence_signals[0]
                self.signal_callback(best_signal)
                
                # Log all active strategies for monitoring
                active_strategies = [f"{s.strategy_name} ({s.confidence:.2f})" 
                                   for s in high_confidence_signals[:3]]
                print(f"[STRATEGY ENGINE] 🎯 Sending signal: {best_signal.strategy_name} ({best_signal.confidence:.2f})")
                print(f"[STRATEGY ENGINE] Other active: {', '.join(active_strategies[1:])}")
            
            # Also send strategy scan summary (less frequent logging)
            if self.total_scans % 100 == 0:  # Every 100 scans
                active_count = len([s for s in strategy_status.values() if s['active']])
                print(f"[STRATEGY ENGINE] Scan #{self.total_scans}: {active_count}/35 strategies active, {len(signals)} signals")
            
    def _strategy_1(self) -> Optional[TradeSignal]:
        """Adaptive Mean Reversion Rebound"""
        if len(self.tick_history) < 5:
            return None
            
        # Conditions: RSI 48-52, Volatility 1-1.5%, Price touches BB, Momentum < ±0.2%
        conditions_met = []
        
        if 48 <= self.indicators.rsi <= 52:
            conditions_met.append("RSI neutral (48-52)")
            
        if 1.0 <= self.indicators.volatility <= 1.5:
            conditions_met.append("Volatility moderate (1-1.5%)")
            
        if abs(self.indicators.momentum) < 0.2:
            conditions_met.append("Low momentum")
            
        if -0.1 <= self.indicators.macd <= 0.1:
            conditions_met.append("MACD flat")
            
        current_price = self.tick_history[-1].price
        last_tick = self.tick_history[-1].color
        
        # Check Bollinger Band touches
        if current_price <= self.indicators.bb_lower and last_tick == 'green':
            conditions_met.append("Lower BB touch + green tick")
            if len(conditions_met) >= 3:
                return TradeSignal(
                    strategy_name=self.strategies[1],
                    direction='CALL',
                    confidence=0.75,
                    hold_time=6,
                    entry_reason="Mean reversion from lower band",
                    conditions_met=conditions_met
                )
                
        elif current_price >= self.indicators.bb_upper and last_tick == 'red':
            conditions_met.append("Upper BB touch + red tick")
            if len(conditions_met) >= 3:
                return TradeSignal(
                    strategy_name=self.strategies[1],
                    direction='PUT',
                    confidence=0.75,
                    hold_time=6,
                    entry_reason="Mean reversion from upper band",
                    conditions_met=conditions_met
                )
                
        return None
        
    def _strategy_2(self) -> Optional[TradeSignal]:
        """Micro-Trend Momentum Tracker"""
        if len(self.tick_history) < 5:
            return None
            
        conditions_met = []
        
        # MACD > 0.25 or < -0.25
        strong_macd = abs(self.indicators.macd) > 0.25
        if strong_macd:
            conditions_met.append(f"Strong MACD ({self.indicators.macd:.3f})")
            
        # Momentum > ±0.1%
        strong_momentum = abs(self.indicators.momentum) > 0.1
        if strong_momentum:
            conditions_met.append(f"Strong momentum ({self.indicators.momentum:.2f}%)")
            
        # RSI within 35-65
        rsi_range = 35 <= self.indicators.rsi <= 65
        if rsi_range:
            conditions_met.append("RSI in trend range (35-65)")
            
        # Check last 4 ticks for direction alignment
        if len(self.tick_history) >= 4:
            last_4_ticks = [tick.color for tick in list(self.tick_history)[-4:]]
            green_count = last_4_ticks.count('green')
            red_count = last_4_ticks.count('red')
            
            macd_direction = 'up' if self.indicators.macd > 0 else 'down'
            
            if green_count >= 3 and macd_direction == 'up' and strong_macd and strong_momentum and rsi_range:
                conditions_met.append("3/4 green ticks align with MACD")
                return TradeSignal(
                    strategy_name=self.strategies[2],
                    direction='CALL',
                    confidence=0.8,
                    hold_time=8,
                    entry_reason="Micro-trend momentum upward",
                    conditions_met=conditions_met
                )
                
            elif red_count >= 3 and macd_direction == 'down' and strong_macd and strong_momentum and rsi_range:
                conditions_met.append("3/4 red ticks align with MACD")
                return TradeSignal(
                    strategy_name=self.strategies[2],
                    direction='PUT',
                    confidence=0.8,
                    hold_time=8,
                    entry_reason="Micro-trend momentum downward",
                    conditions_met=conditions_met
                )
                
        return None
        
    def _strategy_3(self) -> Optional[TradeSignal]:
        """RSI-Tick Divergence Detector"""
        if len(self.tick_history) < 5:
            return None
            
        conditions_met = []
        
        # Check for divergence patterns
        last_3_ticks = [tick.color for tick in list(self.tick_history)[-3:]]
        
        # MACD near 0
        macd_flat = abs(self.indicators.macd) < 0.1
        if macd_flat:
            conditions_met.append("MACD near zero")
            
        # Volatility 0.8-1.5%
        vol_range = 0.8 <= self.indicators.volatility <= 1.5
        if vol_range:
            conditions_met.append("Volatility in range (0.8-1.5%)")
            
        # Momentum near zero
        low_momentum = abs(self.indicators.momentum) < 0.1
        if low_momentum:
            conditions_met.append("Low momentum")
            
        # RSI rising but 3 red ticks
        if len(self.tick_history) >= 6:
            older_rsi = self._calculate_rsi(np.array([tick.price for tick in list(self.tick_history)[:-3]]), 14)
            rsi_rising = self.indicators.rsi > older_rsi
            
            if rsi_rising and all(color == 'red' for color in last_3_ticks):
                conditions_met.append("RSI rising but 3 red ticks (divergence)")
                if macd_flat and vol_range:
                    return TradeSignal(
                        strategy_name=self.strategies[3],
                        direction='CALL',
                        confidence=0.7,
                        hold_time=5,
                        entry_reason="RSI-tick divergence suggests reversal up",
                        conditions_met=conditions_met
                    )
                    
            # RSI falling but 3 green ticks
            elif not rsi_rising and all(color == 'green' for color in last_3_ticks):
                conditions_met.append("RSI falling but 3 green ticks (divergence)")
                if macd_flat and vol_range:
                    return TradeSignal(
                        strategy_name=self.strategies[3],
                        direction='PUT',
                        confidence=0.7,
                        hold_time=5,
                        entry_reason="RSI-tick divergence suggests reversal down",
                        conditions_met=conditions_met
                    )
                    
        return None
        
    def _strategy_4(self) -> Optional[TradeSignal]:
        """Volatility Spike Fader"""
        if len(self.tick_history) < 3:
            return None
            
        conditions_met = []
        
        # Volatility > 1.5%
        high_vol = self.indicators.volatility > 1.5
        if high_vol:
            conditions_met.append(f"High volatility ({self.indicators.volatility:.2f}%)")
            
        # RSI in 45-55 range
        rsi_neutral = 45 <= self.indicators.rsi <= 55
        if rsi_neutral:
            conditions_met.append("RSI neutral (45-55)")
            
        # MACD flat
        macd_flat = abs(self.indicators.macd) < 0.1
        if macd_flat:
            conditions_met.append("MACD flat")
            
        current_price = self.tick_history[-1].price
        last_tick = self.tick_history[-1].color
        
        # Price breaks outside Bollinger Band
        if current_price > self.indicators.bb_upper and last_tick == 'red':
            conditions_met.append("Price above upper BB + red tick")
            if high_vol and rsi_neutral and macd_flat:
                return TradeSignal(
                    strategy_name=self.strategies[4],
                    direction='PUT',
                    confidence=0.75,
                    hold_time=7,
                    entry_reason="Fading volatility spike upward",
                    conditions_met=conditions_met
                )
                
        elif current_price < self.indicators.bb_lower and last_tick == 'green':
            conditions_met.append("Price below lower BB + green tick")
            if high_vol and rsi_neutral and macd_flat:
                return TradeSignal(
                    strategy_name=self.strategies[4],
                    direction='CALL',
                    confidence=0.75,
                    hold_time=7,
                    entry_reason="Fading volatility spike downward",
                    conditions_met=conditions_met
                )
                
        return None
        
    def _strategy_5(self) -> Optional[TradeSignal]:
        """Tick Flow Strength Pulse"""
        if len(self.tick_history) < 5:
            return None
            
        conditions_met = []
        
        # 4 consecutive same color ticks
        last_4_ticks = [tick.color for tick in list(self.tick_history)[-4:]]
        
        # MACD trending in same direction
        macd_up = self.indicators.macd > 0
        macd_down = self.indicators.macd < 0
        
        # Momentum > ±0.15%
        strong_momentum = abs(self.indicators.momentum) > 0.15
        if strong_momentum:
            conditions_met.append(f"Strong momentum ({self.indicators.momentum:.2f}%)")
            
        # Avoid if RSI > 70 or < 30
        rsi_safe = 30 < self.indicators.rsi < 70
        if rsi_safe:
            conditions_met.append("RSI in safe range (30-70)")
            
        if all(color == 'green' for color in last_4_ticks) and macd_up and strong_momentum and rsi_safe:
            conditions_met.append("4 consecutive green ticks + MACD up")
            return TradeSignal(
                strategy_name=self.strategies[5],
                direction='CALL',
                confidence=0.85,
                hold_time=5,
                entry_reason="Strong green tick flow with MACD confirmation",
                conditions_met=conditions_met
            )
            
        elif all(color == 'red' for color in last_4_ticks) and macd_down and strong_momentum and rsi_safe:
            conditions_met.append("4 consecutive red ticks + MACD down")
            return TradeSignal(
                strategy_name=self.strategies[5],
                direction='PUT',
                confidence=0.85,
                hold_time=5,
                entry_reason="Strong red tick flow with MACD confirmation",
                conditions_met=conditions_met
            )
            
        return None
        
    def _strategy_6(self) -> Optional[TradeSignal]:
        """Double Confirmation Breakout"""
        if len(self.tick_history) < 5:
            return None
            
        conditions_met = []
        
        # MACD crossover (simplified - signal line crossed)
        macd_crossover = abs(self.indicators.macd - self.indicators.macd_signal) > 0.05
        if macd_crossover:
            conditions_met.append("MACD crossover detected")
            
        # EMA trending
        current_price = self.tick_history[-1].price
        ema_rising = current_price > self.indicators.ema5
        ema_falling = current_price < self.indicators.ema5
        
        last_tick = self.tick_history[-1].color
        
        # Check crossover direction and confirmations
        if self.indicators.macd > self.indicators.macd_signal and ema_rising and last_tick == 'green':
            conditions_met.extend(["MACD crossover up", "EMA rising", "Green tick confirmation"])
            return TradeSignal(
                strategy_name=self.strategies[6],
                direction='CALL',
                confidence=0.8,
                hold_time=7,
                entry_reason="Double confirmation breakout up",
                conditions_met=conditions_met
            )
            
        elif self.indicators.macd < self.indicators.macd_signal and ema_falling and last_tick == 'red':
            conditions_met.extend(["MACD crossover down", "EMA falling", "Red tick confirmation"])
            return TradeSignal(
                strategy_name=self.strategies[6],
                direction='PUT',
                confidence=0.8,
                hold_time=7,
                entry_reason="Double confirmation breakout down",
                conditions_met=conditions_met
            )
            
        return None
        
    def _strategy_7(self) -> Optional[TradeSignal]:
        """RSI Overextension Fade"""
        if len(self.tick_history) < 3:
            return None
            
        conditions_met = []
        
        # RSI > 70 or RSI < 30
        rsi_overbought = self.indicators.rsi > 70
        rsi_oversold = self.indicators.rsi < 30
        
        # Volatility between 1.2-2.0%
        vol_range = 1.2 <= self.indicators.volatility <= 2.0
        if vol_range:
            conditions_met.append("Volatility in range (1.2-2.0%)")
            
        current_price = self.tick_history[-1].price
        last_tick = self.tick_history[-1].color
        
        # RSI > 70 + upper band + red tick
        if (rsi_overbought and current_price >= self.indicators.bb_upper and 
            last_tick == 'red' and vol_range):
            conditions_met.extend(["RSI overbought (>70)", "Price at upper BB", "Red tick"])
            return TradeSignal(
                strategy_name=self.strategies[7],
                direction='PUT',
                confidence=0.8,
                hold_time=6,
                entry_reason="RSI overextension fade from top",
                conditions_met=conditions_met
            )
            
        # RSI < 30 + lower band + green tick
        elif (rsi_oversold and current_price <= self.indicators.bb_lower and 
              last_tick == 'green' and vol_range):
            conditions_met.extend(["RSI oversold (<30)", "Price at lower BB", "Green tick"])
            return TradeSignal(
                strategy_name=self.strategies[7],
                direction='CALL',
                confidence=0.8,
                hold_time=6,
                entry_reason="RSI overextension fade from bottom",
                conditions_met=conditions_met
            )
            
        return None
        
    def _strategy_8(self) -> Optional[TradeSignal]:
        """Multi-Tick Pivot Bounce"""
        if len(self.tick_history) < 5:
            return None
            
        conditions_met = []
        
        # Momentum flat
        low_momentum = abs(self.indicators.momentum) < 0.1
        if low_momentum:
            conditions_met.append("Low momentum")
            
        # Volatility < 1.5%
        low_vol = self.indicators.volatility < 1.5
        if low_vol:
            conditions_met.append("Low volatility (<1.5%)")
            
        # Check for pivot patterns in last 4 ticks
        if len(self.tick_history) >= 4:
            last_4_prices = [tick.price for tick in list(self.tick_history)[-4:]]
            last_tick = self.tick_history[-1]
            
            # Local high followed by reversal
            if (last_4_prices[2] == max(last_4_prices[:3]) and 
                last_tick.color == 'red' and low_momentum and low_vol):
                conditions_met.append("Local high + red reversal tick")
                return TradeSignal(
                    strategy_name=self.strategies[8],
                    direction='PUT',
                    confidence=0.7,
                    hold_time=5,
                    entry_reason="Pivot bounce from local high",
                    conditions_met=conditions_met
                )
                
            # Local low followed by reversal
            elif (last_4_prices[2] == min(last_4_prices[:3]) and 
                  last_tick.color == 'green' and low_momentum and low_vol):
                conditions_met.append("Local low + green reversal tick")
                return TradeSignal(
                    strategy_name=self.strategies[8],
                    direction='CALL',
                    confidence=0.7,
                    hold_time=5,
                    entry_reason="Pivot bounce from local low",
                    conditions_met=conditions_met
                )
                
        return None
        
    def _strategy_9(self) -> Optional[TradeSignal]:
        """MACD-Momentum Sync Engine"""
        if len(self.tick_history) < 3:
            return None
            
        conditions_met = []
        
        # RSI in 40-60 range
        rsi_range = 40 <= self.indicators.rsi <= 60
        if rsi_range:
            conditions_met.append("RSI in range (40-60)")
            
        # Check last 2 ticks
        if len(self.tick_history) >= 2:
            last_2_ticks = [tick.color for tick in list(self.tick_history)[-2:]]
            
            # MACD > 0.25 and Momentum > 0.15% + 2 green ticks
            if (self.indicators.macd > 0.25 and self.indicators.momentum > 0.15 and 
                all(color == 'green' for color in last_2_ticks) and rsi_range):
                conditions_met.extend([
                    f"Strong bullish MACD ({self.indicators.macd:.3f})",
                    f"Strong bullish momentum ({self.indicators.momentum:.2f}%)",
                    "2 green ticks"
                ])
                return TradeSignal(
                    strategy_name=self.strategies[9],
                    direction='CALL',
                    confidence=0.85,
                    hold_time=8,
                    entry_reason="MACD-Momentum sync bullish",
                    conditions_met=conditions_met
                )
                
            # MACD < -0.25 and Momentum < -0.15% + 2 red ticks
            elif (self.indicators.macd < -0.25 and self.indicators.momentum < -0.15 and 
                  all(color == 'red' for color in last_2_ticks) and rsi_range):
                conditions_met.extend([
                    f"Strong bearish MACD ({self.indicators.macd:.3f})",
                    f"Strong bearish momentum ({self.indicators.momentum:.2f}%)",
                    "2 red ticks"
                ])
                return TradeSignal(
                    strategy_name=self.strategies[9],
                    direction='PUT',
                    confidence=0.85,
                    hold_time=8,
                    entry_reason="MACD-Momentum sync bearish",
                    conditions_met=conditions_met
                )
                
        return None
        
    def _strategy_10(self) -> Optional[TradeSignal]:
        """Time-of-Tick Scalper"""
        if len(self.tick_history) < 4:
            return None
            
        conditions_met = []
        
        # Check tick timing (speed analysis)
        recent_ticks = list(self.tick_history)[-4:]
        time_intervals = []
        
        for i in range(1, len(recent_ticks)):
            interval = recent_ticks[i].timestamp - recent_ticks[i-1].timestamp
            time_intervals.append(interval)
            
        # Detect if ticks are speeding up
        if len(time_intervals) >= 2:
            avg_recent = sum(time_intervals[-2:]) / 2
            avg_older = sum(time_intervals[:-2]) / max(1, len(time_intervals)-2)
            
            speeding_up = avg_recent < avg_older * 0.8  # 20% faster
            
            if speeding_up:
                conditions_met.append("Tick speed increasing")
                
                # EMA sharply moving
                current_price = self.tick_history[-1].price
                ema_diff = abs(current_price - self.indicators.ema5) / self.indicators.ema5
                sharp_ema = ema_diff > 0.001  # 0.1% difference
                
                if sharp_ema:
                    conditions_met.append("EMA moving sharply")
                    
                    # Tick color matches MACD
                    last_tick = self.tick_history[-1].color
                    macd_up = self.indicators.macd > 0
                    macd_down = self.indicators.macd < 0
                    
                    if last_tick == 'green' and macd_up:
                        conditions_met.append("Green tick matches bullish MACD")
                        return TradeSignal(
                            strategy_name=self.strategies[10],
                            direction='CALL',
                            confidence=0.75,
                            hold_time=6,
                            entry_reason="Time-based scalp signal up",
                            conditions_met=conditions_met
                        )
                        
                    elif last_tick == 'red' and macd_down:
                        conditions_met.append("Red tick matches bearish MACD")
                        return TradeSignal(
                            strategy_name=self.strategies[10],
                            direction='PUT',
                            confidence=0.75,
                            hold_time=6,
                            entry_reason="Time-based scalp signal down",
                            conditions_met=conditions_met
                        )
                        
        return None
        
    def _strategy_11(self) -> Optional[TradeSignal]:
        """Volatility Collapse Compression"""
        if len(self.tick_history) < 5:
            return None
            
        conditions_met = []
        
        # Volatility < 0.6%
        very_low_vol = self.indicators.volatility < 0.6
        if very_low_vol:
            conditions_met.append(f"Very low volatility ({self.indicators.volatility:.2f}%)")
            
        # RSI hovering near 50
        rsi_neutral = 48 <= self.indicators.rsi <= 52
        if rsi_neutral:
            conditions_met.append("RSI near 50 (neutral)")
            
        # Bollinger Bands narrow (squeeze)
        bb_width = (self.indicators.bb_upper - self.indicators.bb_lower) / self.indicators.bb_middle
        narrow_bands = bb_width < 0.02  # 2% width
        if narrow_bands:
            conditions_met.append("Bollinger Bands squeeze")
            
        # First strong tick + MACD spike
        last_tick = self.tick_history[-1]
        current_price = last_tick.price
        
        # Check for price breaking out of compression
        breakout_up = current_price > self.indicators.bb_upper
        breakout_down = current_price < self.indicators.bb_lower
        
        macd_spike = abs(self.indicators.macd) > 0.1
        
        if (very_low_vol and rsi_neutral and narrow_bands and macd_spike):
            if breakout_up and last_tick.color == 'green':
                conditions_met.append("Upward breakout from compression")
                return TradeSignal(
                    strategy_name=self.strategies[11],
                    direction='CALL',
                    confidence=0.8,
                    hold_time=7,
                    entry_reason="Volatility collapse breakout up",
                    conditions_met=conditions_met
                )
                
            elif breakout_down and last_tick.color == 'red':
                conditions_met.append("Downward breakout from compression")
                return TradeSignal(
                    strategy_name=self.strategies[11],
                    direction='PUT',
                    confidence=0.8,
                    hold_time=7,
                    entry_reason="Volatility collapse breakout down",
                    conditions_met=conditions_met
                )
                
        return None
        
    def _strategy_12(self) -> Optional[TradeSignal]:
        """Two-Step Confirmation Model"""
        if len(self.tick_history) < 3:
            return None
            
        conditions_met = []
        
        # Step 1: MACD crossover
        macd_crossover = abs(self.indicators.macd - self.indicators.macd_signal) > 0.05
        crossover_up = self.indicators.macd > self.indicators.macd_signal
        crossover_down = self.indicators.macd < self.indicators.macd_signal
        
        if macd_crossover:
            conditions_met.append("Step 1: MACD crossover")
            
            # Step 2: RSI breaks through 50
            rsi_break_up = self.indicators.rsi > 50
            rsi_break_down = self.indicators.rsi < 50
            
            # Step 3: Momentum confirms with tick direction
            last_tick = self.tick_history[-1].color
            momentum_up = self.indicators.momentum > 0
            momentum_down = self.indicators.momentum < 0
            
            # All 3 steps for bullish signal
            if (crossover_up and rsi_break_up and momentum_up and last_tick == 'green'):
                conditions_met.extend([
                    "Step 2: RSI above 50",
                    "Step 3: Momentum + tick alignment bullish"
                ])
                return TradeSignal(
                    strategy_name=self.strategies[12],
                    direction='CALL',
                    confidence=0.9,
                    hold_time=8,
                    entry_reason="Three-step confirmation bullish",
                    conditions_met=conditions_met
                )
                
            # All 3 steps for bearish signal
            elif (crossover_down and rsi_break_down and momentum_down and last_tick == 'red'):
                conditions_met.extend([
                    "Step 2: RSI below 50",
                    "Step 3: Momentum + tick alignment bearish"
                ])
                return TradeSignal(
                    strategy_name=self.strategies[12],
                    direction='PUT',
                    confidence=0.9,
                    hold_time=8,
                    entry_reason="Three-step confirmation bearish",
                    conditions_met=conditions_met
                )
                
        return None
        
    def _strategy_13(self) -> Optional[TradeSignal]:
        """Inverted Divergence Flip"""
        if len(self.tick_history) < 10:
            return None
            
        conditions_met = []
        
        # Analyze price and MACD patterns over recent history
        recent_prices = [tick.price for tick in list(self.tick_history)[-10:]]
        
        # Check for price making new lows/highs vs MACD
        current_price = recent_prices[-1]
        price_low = min(recent_prices)
        price_high = max(recent_prices)
        
        # Simplified divergence detection
        if current_price == price_low:  # At new low
            # Check if MACD is making higher lows (bullish divergence)
            if self.indicators.macd > -0.2:  # MACD not as low
                conditions_met.append("Price new low, MACD higher low (bullish divergence)")
                
                # Confirm with tick color change
                last_tick = self.tick_history[-1].color
                if last_tick == 'green':
                    conditions_met.append("Green tick confirms reversal")
                    return TradeSignal(
                        strategy_name=self.strategies[13],
                        direction='CALL',
                        confidence=0.75,
                        hold_time=5,
                        entry_reason="Inverted divergence bullish flip",
                        conditions_met=conditions_met
                    )
                    
        elif current_price == price_high:  # At new high
            # Check if MACD is making lower highs (bearish divergence)
            if self.indicators.macd < 0.2:  # MACD not as high
                conditions_met.append("Price new high, MACD lower high (bearish divergence)")
                
                # Confirm with tick color change
                last_tick = self.tick_history[-1].color
                if last_tick == 'red':
                    conditions_met.append("Red tick confirms reversal")
                    return TradeSignal(
                        strategy_name=self.strategies[13],
                        direction='PUT',
                        confidence=0.75,
                        hold_time=5,
                        entry_reason="Inverted divergence bearish flip",
                        conditions_met=conditions_met
                    )
                    
        return None
        
    def _strategy_14(self) -> Optional[TradeSignal]:
        """Cumulative Strength Index Pullback"""
        if len(self.tick_history) < 4:
            return None
            
        conditions_met = []
        
        # Momentum > 0.25%
        strong_momentum = abs(self.indicators.momentum) > 0.25
        if strong_momentum:
            conditions_met.append(f"Strong momentum ({self.indicators.momentum:.2f}%)")
            
        # MACD trending
        macd_trending = abs(self.indicators.macd) > 0.15
        if macd_trending:
            conditions_met.append("MACD trending")
            
        # Check for pullback patterns
        last_2_ticks = [tick.color for tick in list(self.tick_history)[-2:]]
        
        # Uptrend conditions
        if (self.indicators.momentum > 0.25 and self.indicators.macd > 0.15 and 
            self.indicators.rsi > 55):
            conditions_met.append("Strong uptrend (RSI > 55)")
            
            # 1-2 ticks against trend (red ticks in uptrend)
            red_count = last_2_ticks.count('red')
            if red_count >= 1:
                conditions_met.append(f"{red_count} red pullback ticks in uptrend")
                return TradeSignal(
                    strategy_name=self.strategies[14],
                    direction='CALL',
                    confidence=0.8,
                    hold_time=7,
                    entry_reason="Pullback entry in strong uptrend",
                    conditions_met=conditions_met
                )
                
        # Downtrend conditions
        elif (self.indicators.momentum < -0.25 and self.indicators.macd < -0.15 and 
              self.indicators.rsi < 45):
            conditions_met.append("Strong downtrend (RSI < 45)")
            
            # 1-2 ticks against trend (green ticks in downtrend)
            green_count = last_2_ticks.count('green')
            if green_count >= 1:
                conditions_met.append(f"{green_count} green pullback ticks in downtrend")
                return TradeSignal(
                    strategy_name=self.strategies[14],
                    direction='PUT',
                    confidence=0.8,
                    hold_time=7,
                    entry_reason="Pullback entry in strong downtrend",
                    conditions_met=conditions_met
                )
                
        return None
        
    def _strategy_15(self) -> Optional[TradeSignal]:
        """Tri-Indicator Confluence Strategy"""
        if len(self.tick_history) < 3:
            return None
            
        conditions_met = []
        
        # Volatility window (1%-1.5%)
        vol_window = 1.0 <= self.indicators.volatility <= 1.5
        if vol_window:
            conditions_met.append("Volatility in optimal window (1-1.5%)")
            
        last_tick = self.tick_history[-1]
        
        # All bullish conditions
        rsi_rising = self.indicators.rsi > 50
        macd_bullish = self.indicators.macd > 0.2
        momentum_bullish = self.indicators.momentum > 0.1
        tick_green = last_tick.color == 'green'
        
        if rsi_rising and macd_bullish and momentum_bullish and tick_green and vol_window:
            conditions_met.extend([
                "RSI rising through 50",
                f"MACD bullish ({self.indicators.macd:.3f})",
                f"Momentum bullish ({self.indicators.momentum:.2f}%)",
                "Green tick confirmation"
            ])
            return TradeSignal(
                strategy_name=self.strategies[15],
                direction='CALL',
                confidence=0.95,
                hold_time=9,
                entry_reason="Perfect tri-indicator confluence bullish",
                conditions_met=conditions_met
            )
            
        # All bearish conditions
        rsi_falling = self.indicators.rsi < 50
        macd_bearish = self.indicators.macd < -0.2
        momentum_bearish = self.indicators.momentum < -0.1
        tick_red = last_tick.color == 'red'
        
        if rsi_falling and macd_bearish and momentum_bearish and tick_red and vol_window:
            conditions_met.extend([
                "RSI falling through 50",
                f"MACD bearish ({self.indicators.macd:.3f})",
                f"Momentum bearish ({self.indicators.momentum:.2f}%)",
                "Red tick confirmation"
            ])
            return TradeSignal(
                strategy_name=self.strategies[15],
                direction='PUT',
                confidence=0.95,
                hold_time=9,
                entry_reason="Perfect tri-indicator confluence bearish",
                conditions_met=conditions_met
            )
            
        return None
        
    def _strategy_16(self) -> Optional[TradeSignal]:
        """RSI Stall Reversal - Reversal from RSI stalling near center"""
        if len(self.tick_history) < 3:
            return None
            
        conditions_met = []
        
        # Check RSI between 48-52 for last 3 ticks
        if len(self.tick_history) >= 3:
            last_3_rsi = []
            for i in range(-3, 0):
                if abs(i) <= len(self.tick_history):
                    prices_subset = np.array([tick.price for tick in list(self.tick_history)[:i or None]])
                    if len(prices_subset) >= 14:
                        rsi_val = self._calculate_rsi(prices_subset, 14)
                        last_3_rsi.append(rsi_val)
            
            # Check if RSI has been stable between 48-52
            rsi_stable = all(48 <= rsi <= 52 for rsi in last_3_rsi) if len(last_3_rsi) >= 2 else False
            if rsi_stable:
                conditions_met.append("RSI stable 48-52 for 3 ticks")
        
        current_price = self.tick_history[-1].price
        
        # Check Bollinger Band touches
        if current_price <= self.indicators.bb_lower and rsi_stable:
            conditions_met.append("Price touches lower BB")
            return TradeSignal(
                strategy_name=self.strategies[16],
                direction='CALL',
                confidence=0.78,
                hold_time=5,
                entry_reason="RSI stall reversal from lower band",
                conditions_met=conditions_met
            )
            
        elif current_price >= self.indicators.bb_upper and rsi_stable:
            conditions_met.append("Price touches upper BB")
            return TradeSignal(
                strategy_name=self.strategies[16],
                direction='PUT',
                confidence=0.78,
                hold_time=5,
                entry_reason="RSI stall reversal from upper band",
                conditions_met=conditions_met
            )
            
        return None
        
    def _strategy_17(self) -> Optional[TradeSignal]:
        """Tick Flow Momentum Ride - Ride strong tick flow if confirmed by momentum"""
        if len(self.tick_history) < 5:
            return None
            
        conditions_met = []
        
        # Check 4 out of last 5 ticks same color
        last_5_ticks = [tick.color for tick in list(self.tick_history)[-5:]]
        green_count = last_5_ticks.count('green')
        red_count = last_5_ticks.count('red')
        
        # Momentum > ±0.15%
        strong_momentum = abs(self.indicators.momentum) > 0.15
        if strong_momentum:
            conditions_met.append(f"Strong momentum ({self.indicators.momentum:.2f}%)")
            
        # MACD trending in same direction
        macd_trending = abs(self.indicators.macd) > 0.1
        if macd_trending:
            conditions_met.append(f"MACD trending ({self.indicators.macd:.3f})")
        
        if green_count >= 4 and self.indicators.momentum > 0.15 and self.indicators.macd > 0:
            conditions_met.append("4/5 green ticks with upward momentum")
            return TradeSignal(
                strategy_name=self.strategies[17],
                direction='CALL',
                confidence=0.82,
                hold_time=6,
                entry_reason="Strong green tick flow momentum",
                conditions_met=conditions_met
            )
            
        elif red_count >= 4 and self.indicators.momentum < -0.15 and self.indicators.macd < 0:
            conditions_met.append("4/5 red ticks with downward momentum")
            return TradeSignal(
                strategy_name=self.strategies[17],
                direction='PUT',
                confidence=0.82,
                hold_time=6,
                entry_reason="Strong red tick flow momentum",
                conditions_met=conditions_met
            )
            
        return None
        
    def _strategy_18(self) -> Optional[TradeSignal]:
        """Divergence Snapback - Detect price-tick mismatch to fade move"""
        if len(self.tick_history) < 5:
            return None
            
        conditions_met = []
        
        # Check last 3 ticks
        last_3_ticks = [tick.color for tick in list(self.tick_history)[-3:]]
        
        # MACD flat
        macd_flat = abs(self.indicators.macd) < 0.1
        if macd_flat:
            conditions_met.append("MACD flat")
            
        # RSI rising but 3 red ticks
        rsi_rising = self.indicators.rsi > self.indicators.rsi_previous
        rsi_falling = self.indicators.rsi < self.indicators.rsi_previous
        
        if rsi_rising and all(color == 'red' for color in last_3_ticks) and macd_flat:
            conditions_met.extend(["RSI rising", "3 red ticks", "Divergence detected"])
            return TradeSignal(
                strategy_name=self.strategies[18],
                direction='CALL',
                confidence=0.76,
                hold_time=5,
                entry_reason="Divergence snapback - RSI up, ticks red",
                conditions_met=conditions_met
            )
            
        elif rsi_falling and all(color == 'green' for color in last_3_ticks) and macd_flat:
            conditions_met.extend(["RSI falling", "3 green ticks", "Divergence detected"])
            return TradeSignal(
                strategy_name=self.strategies[18],
                direction='PUT',
                confidence=0.76,
                hold_time=5,
                entry_reason="Divergence snapback - RSI down, ticks green",
                conditions_met=conditions_met
            )
            
        return None
        
    def _strategy_19(self) -> Optional[TradeSignal]:
        """Volatility Breakout Tick Rejection - Play reversion after failed spike"""
        if len(self.tick_history) < 3:
            return None
            
        conditions_met = []
        
        # Volatility > 1.6%
        high_vol = self.indicators.volatility > 1.6
        if high_vol:
            conditions_met.append(f"High volatility ({self.indicators.volatility:.2f}%)")
            
        current_price = self.tick_history[-1].price
        last_tick = self.tick_history[-1].color
        
        # Check for rejection at Bollinger Bands
        pierces_upper = current_price > self.indicators.bb_upper
        pierces_lower = current_price < self.indicators.bb_lower
        
        if pierces_upper and last_tick == 'red' and high_vol:
            conditions_met.extend(["Price pierces upper BB", "Red rejection tick"])
            return TradeSignal(
                strategy_name=self.strategies[19],
                direction='PUT',
                confidence=0.8,
                hold_time=8,
                entry_reason="Volatility spike rejection from top",
                conditions_met=conditions_met
            )
            
        elif pierces_lower and last_tick == 'green' and high_vol:
            conditions_met.extend(["Price pierces lower BB", "Green rejection tick"])
            return TradeSignal(
                strategy_name=self.strategies[19],
                direction='CALL',
                confidence=0.8,
                hold_time=8,
                entry_reason="Volatility spike rejection from bottom",
                conditions_met=conditions_met
            )
            
        return None
        
    def _strategy_20(self) -> Optional[TradeSignal]:
        """Triple Confirmation Flow - Trade only when RSI + MACD + Ticks align"""
        if len(self.tick_history) < 4:
            return None
            
        conditions_met = []
        
        # RSI trend for 3 ticks (simplified as current vs previous)
        rsi_rising = self.indicators.rsi > self.indicators.rsi_previous
        rsi_falling = self.indicators.rsi < self.indicators.rsi_previous
        
        # MACD > 0.2 or < -0.2
        macd_strong = abs(self.indicators.macd) > 0.2
        macd_up = self.indicators.macd > 0.2
        macd_down = self.indicators.macd < -0.2
        
        # 3 of last 4 ticks in same direction
        last_4_ticks = [tick.color for tick in list(self.tick_history)[-4:]]
        green_count = last_4_ticks.count('green')
        red_count = last_4_ticks.count('red')
        
        if rsi_rising and macd_up and green_count >= 3:
            conditions_met.extend(["RSI rising", "MACD strong up", "3/4 green ticks"])
            return TradeSignal(
                strategy_name=self.strategies[20],
                direction='CALL',
                confidence=0.85,
                hold_time=10,
                entry_reason="Triple confirmation bullish alignment",
                conditions_met=conditions_met
            )
            
        elif rsi_falling and macd_down and red_count >= 3:
            conditions_met.extend(["RSI falling", "MACD strong down", "3/4 red ticks"])
            return TradeSignal(
                strategy_name=self.strategies[20],
                direction='PUT',
                confidence=0.85,
                hold_time=10,
                entry_reason="Triple confirmation bearish alignment",
                conditions_met=conditions_met
            )
            
        return None
        
    def _strategy_21(self) -> Optional[TradeSignal]:
        """Tick Trap Reversal - Spot overextended micro-trend and reverse"""
        if len(self.tick_history) < 6:
            return None
            
        conditions_met = []
        
        # Check for 5 same-colored ticks in a row
        last_6_ticks = [tick.color for tick in list(self.tick_history)[-6:]]
        last_5_ticks = last_6_ticks[:-1]
        current_tick = last_6_ticks[-1]
        
        # RSI near overbought/oversold
        rsi_overbought = self.indicators.rsi > 65
        rsi_oversold = self.indicators.rsi < 35
        
        if (all(color == 'green' for color in last_5_ticks) and 
            current_tick == 'red' and rsi_overbought):
            conditions_met.extend(["5 green ticks", "Red reversal tick", "RSI overbought"])
            return TradeSignal(
                strategy_name=self.strategies[21],
                direction='PUT',
                confidence=0.83,
                hold_time=6,
                entry_reason="Tick trap reversal from overbought",
                conditions_met=conditions_met
            )
            
        elif (all(color == 'red' for color in last_5_ticks) and 
              current_tick == 'green' and rsi_oversold):
            conditions_met.extend(["5 red ticks", "Green reversal tick", "RSI oversold"])
            return TradeSignal(
                strategy_name=self.strategies[21],
                direction='CALL',
                confidence=0.83,
                hold_time=6,
                entry_reason="Tick trap reversal from oversold",
                conditions_met=conditions_met
            )
            
        return None
        
    def _strategy_22(self) -> Optional[TradeSignal]:
        """Bollinger Bounce Magnet - Rebound from Bollinger outer band"""
        if len(self.tick_history) < 3:
            return None
            
        conditions_met = []
        
        current_price = self.tick_history[-1].price
        current_tick = self.tick_history[-1].color
        
        # Price touches outer Bollinger Band
        touches_upper = current_price >= self.indicators.bb_upper
        touches_lower = current_price <= self.indicators.bb_lower
        
        # RSI between 45-55
        rsi_neutral = 45 <= self.indicators.rsi <= 55
        if rsi_neutral:
            conditions_met.append("RSI neutral (45-55)")
        
        if touches_lower and current_tick == 'green' and rsi_neutral:
            conditions_met.extend(["Price touches lower BB", "Green confirmation tick"])
            return TradeSignal(
                strategy_name=self.strategies[22],
                direction='CALL',
                confidence=0.79,
                hold_time=5,
                entry_reason="Bollinger bounce from lower band",
                conditions_met=conditions_met
            )
            
        elif touches_upper and current_tick == 'red' and rsi_neutral:
            conditions_met.extend(["Price touches upper BB", "Red confirmation tick"])
            return TradeSignal(
                strategy_name=self.strategies[22],
                direction='PUT',
                confidence=0.79,
                hold_time=5,
                entry_reason="Bollinger bounce from upper band",
                conditions_met=conditions_met
            )
            
        return None
        
    def _strategy_23(self) -> Optional[TradeSignal]:
        """EMA Compression Breakout - Play EMA squeeze breakout"""
        if len(self.tick_history) < 3:
            return None
            
        conditions_met = []
        
        # EMA(3) and EMA(5) converge tightly
        ema_convergence = abs(self.indicators.ema3 - self.indicators.ema5) / self.indicators.ema5 < 0.001  # 0.1% difference
        if ema_convergence:
            conditions_met.append("EMA3 and EMA5 converged")
            
        # Check for sudden expansion + 2 ticks same color
        last_2_ticks = [tick.color for tick in list(self.tick_history)[-2:]]
        
        current_price = self.tick_history[-1].price
        price_above_ema = current_price > self.indicators.ema5
        price_below_ema = current_price < self.indicators.ema5
        
        if (ema_convergence and all(color == 'green' for color in last_2_ticks) and 
            price_above_ema):
            conditions_met.extend(["2 green ticks", "Price above EMA5"])
            return TradeSignal(
                strategy_name=self.strategies[23],
                direction='CALL',
                confidence=0.81,
                hold_time=6,
                entry_reason="EMA compression breakout upward",
                conditions_met=conditions_met
            )
            
        elif (ema_convergence and all(color == 'red' for color in last_2_ticks) and 
              price_below_ema):
            conditions_met.extend(["2 red ticks", "Price below EMA5"])
            return TradeSignal(
                strategy_name=self.strategies[23],
                direction='PUT',
                confidence=0.81,
                hold_time=6,
                entry_reason="EMA compression breakout downward",
                conditions_met=conditions_met
            )
            
        return None
        
    def _strategy_24(self) -> Optional[TradeSignal]:
        """Tick RSI Bounce - RSI bounce from 30/70 zone with tick confirmation"""
        if len(self.tick_history) < 2:
            return None
            
        conditions_met = []
        
        # RSI < 30 or > 70
        rsi_extreme = self.indicators.rsi < 30 or self.indicators.rsi > 70
        
        # First tick back toward center
        current_tick = self.tick_history[-1].color
        
        if self.indicators.rsi < 30 and current_tick == 'green':
            conditions_met.extend(["RSI oversold (<30)", "Green reversal tick"])
            return TradeSignal(
                strategy_name=self.strategies[24],
                direction='CALL',
                confidence=0.8,
                hold_time=5,
                entry_reason="RSI bounce from oversold with green tick",
                conditions_met=conditions_met
            )
            
        elif self.indicators.rsi > 70 and current_tick == 'red':
            conditions_met.extend(["RSI overbought (>70)", "Red reversal tick"])
            return TradeSignal(
                strategy_name=self.strategies[24],
                direction='PUT',
                confidence=0.8,
                hold_time=5,
                entry_reason="RSI bounce from overbought with red tick",
                conditions_met=conditions_met
            )
            
        return None
        
    def _strategy_25(self) -> Optional[TradeSignal]:
        """Tick Pulse Sync - Match pulse of momentum + tick velocity"""
        if len(self.tick_history) < 3:
            return None
            
        conditions_met = []
        
        # 3 green/red ticks
        last_3_ticks = [tick.color for tick in list(self.tick_history)[-3:]]
        all_green = all(color == 'green' for color in last_3_ticks)
        all_red = all(color == 'red' for color in last_3_ticks)
        
        # Momentum > 0.2%
        strong_momentum = abs(self.indicators.momentum) > 0.2
        if strong_momentum:
            conditions_met.append(f"Strong momentum ({self.indicators.momentum:.2f}%)")
            
        # RSI rising/falling
        rsi_rising = self.indicators.rsi > self.indicators.rsi_previous
        rsi_falling = self.indicators.rsi < self.indicators.rsi_previous
        
        if all_green and self.indicators.momentum > 0.2 and rsi_rising:
            conditions_met.extend(["3 green ticks", "RSI rising"])
            return TradeSignal(
                strategy_name=self.strategies[25],
                direction='CALL',
                confidence=0.82,
                hold_time=7,
                entry_reason="Bullish tick pulse sync",
                conditions_met=conditions_met
            )
            
        elif all_red and self.indicators.momentum < -0.2 and rsi_falling:
            conditions_met.extend(["3 red ticks", "RSI falling"])
            return TradeSignal(
                strategy_name=self.strategies[25],
                direction='PUT',
                confidence=0.82,
                hold_time=7,
                entry_reason="Bearish tick pulse sync",
                conditions_met=conditions_met
            )
            
        return None
        
    def _strategy_26(self) -> Optional[TradeSignal]:
        """EMA Magnet Pullback - Snapback toward EMA(3)"""
        if len(self.tick_history) < 2:
            return None
            
        conditions_met = []
        
        current_price = self.tick_history[-1].price
        current_tick = self.tick_history[-1].color
        
        # Price > 1% from EMA(3)
        price_distance = abs(current_price - self.indicators.ema3) / self.indicators.ema3
        far_from_ema = price_distance > 0.01  # 1%
        
        if far_from_ema:
            conditions_met.append(f"Price {price_distance*100:.1f}% from EMA3")
            
        # Check if tick direction reverses toward EMA
        price_above_ema = current_price > self.indicators.ema3
        price_below_ema = current_price < self.indicators.ema3
        
        if price_above_ema and current_tick == 'red' and far_from_ema:
            conditions_met.extend(["Price above EMA3", "Red tick toward EMA"])
            return TradeSignal(
                strategy_name=self.strategies[26],
                direction='PUT',
                confidence=0.77,
                hold_time=5,
                entry_reason="EMA magnet pullback from above",
                conditions_met=conditions_met
            )
            
        elif price_below_ema and current_tick == 'green' and far_from_ema:
            conditions_met.extend(["Price below EMA3", "Green tick toward EMA"])
            return TradeSignal(
                strategy_name=self.strategies[26],
                direction='CALL',
                confidence=0.77,
                hold_time=5,
                entry_reason="EMA magnet pullback from below",
                conditions_met=conditions_met
            )
            
        return None
        
    def _strategy_27(self) -> Optional[TradeSignal]:
        """RSI Mirror Flip - Mirror RSI direction after exhaustion"""
        if len(self.tick_history) < 2:
            return None
            
        conditions_met = []
        
        # RSI hits 70 then drops 5+ pts
        rsi_dropped_from_high = (self.indicators.rsi_previous >= 70 and 
                                 self.indicators.rsi < self.indicators.rsi_previous - 5)
        
        # RSI hits 30 then rises 5+ pts  
        rsi_rose_from_low = (self.indicators.rsi_previous <= 30 and 
                            self.indicators.rsi > self.indicators.rsi_previous + 5)
        
        last_2_ticks = [tick.color for tick in list(self.tick_history)[-2:]]
        
        if rsi_dropped_from_high and all(color == 'red' for color in last_2_ticks):
            conditions_met.extend(["RSI dropped 5+ from 70", "2 red ticks"])
            return TradeSignal(
                strategy_name=self.strategies[27],
                direction='PUT',
                confidence=0.84,
                hold_time=6,
                entry_reason="RSI mirror flip from overbought",
                conditions_met=conditions_met
            )
            
        elif rsi_rose_from_low and all(color == 'green' for color in last_2_ticks):
            conditions_met.extend(["RSI rose 5+ from 30", "2 green ticks"])
            return TradeSignal(
                strategy_name=self.strategies[27],
                direction='CALL',
                confidence=0.84,
                hold_time=6,
                entry_reason="RSI mirror flip from oversold",
                conditions_met=conditions_met
            )
            
        return None
        
    def _strategy_28(self) -> Optional[TradeSignal]:
        """MACD Crossover Trigger - Entry on real-time MACD crossover"""
        if len(self.tick_history) < 3:
            return None
            
        conditions_met = []
        
        # MACD crosses signal line (simplified detection)
        macd_bullish_cross = (self.indicators.macd > self.indicators.macd_signal and 
                             abs(self.indicators.macd - self.indicators.macd_signal) > 0.02)
        macd_bearish_cross = (self.indicators.macd < self.indicators.macd_signal and 
                             abs(self.indicators.macd - self.indicators.macd_signal) > 0.02)
        
        # Confirm with 2 ticks in same direction
        last_2_ticks = [tick.color for tick in list(self.tick_history)[-2:]]
        
        if macd_bullish_cross and all(color == 'green' for color in last_2_ticks):
            conditions_met.extend(["MACD bullish crossover", "2 green confirmation ticks"])
            return TradeSignal(
                strategy_name=self.strategies[28],
                direction='CALL',
                confidence=0.83,
                hold_time=8,
                entry_reason="MACD crossover bullish trigger",
                conditions_met=conditions_met
            )
            
        elif macd_bearish_cross and all(color == 'red' for color in last_2_ticks):
            conditions_met.extend(["MACD bearish crossover", "2 red confirmation ticks"])
            return TradeSignal(
                strategy_name=self.strategies[28],
                direction='PUT',
                confidence=0.83,
                hold_time=8,
                entry_reason="MACD crossover bearish trigger",
                conditions_met=conditions_met
            )
            
        return None
        
    def _strategy_29(self) -> Optional[TradeSignal]:
        """Volatility Expansion Ride - Ride spike as volatility expands"""
        if len(self.tick_history) < 3:
            return None
            
        conditions_met = []
        
        # Volatility > 1.8%
        high_vol = self.indicators.volatility > 1.8
        if high_vol:
            conditions_met.append(f"High volatility ({self.indicators.volatility:.2f}%)")
            
        # 3 consecutive large tick bodies (simplified as same color)
        last_3_ticks = [tick.color for tick in list(self.tick_history)[-3:]]
        consistent_direction = (all(color == 'green' for color in last_3_ticks) or 
                               all(color == 'red' for color in last_3_ticks))
        
        # Momentum > ±0.3%
        strong_momentum = abs(self.indicators.momentum) > 0.3
        if strong_momentum:
            conditions_met.append(f"Strong momentum ({self.indicators.momentum:.2f}%)")
            
        if (high_vol and consistent_direction and strong_momentum):
            if all(color == 'green' for color in last_3_ticks):
                conditions_met.append("3 consecutive green ticks")
                return TradeSignal(
                    strategy_name=self.strategies[29],
                    direction='CALL',
                    confidence=0.86,
                    hold_time=7,
                    entry_reason="Volatility expansion ride upward",
                    conditions_met=conditions_met
                )
            else:
                conditions_met.append("3 consecutive red ticks")
                return TradeSignal(
                    strategy_name=self.strategies[29],
                    direction='PUT',
                    confidence=0.86,
                    hold_time=7,
                    entry_reason="Volatility expansion ride downward",
                    conditions_met=conditions_met
                )
                
        return None
        
    def _strategy_30(self) -> Optional[TradeSignal]:
        """RSI Gradient Tilt - Use RSI slope to detect tilt"""
        if len(self.tick_history) < 5:
            return None
            
        conditions_met = []
        
        # RSI rises or falls 10+ pts over 5 ticks (simplified)
        rsi_change = abs(self.indicators.rsi - self.indicators.rsi_previous)
        significant_rsi_change = rsi_change > 5  # Simplified from 10+ over 5 ticks
        
        if significant_rsi_change:
            conditions_met.append(f"RSI changed {rsi_change:.1f} points")
            
        # Momentum aligns
        momentum_aligns = ((self.indicators.rsi > self.indicators.rsi_previous and self.indicators.momentum > 0) or
                          (self.indicators.rsi < self.indicators.rsi_previous and self.indicators.momentum < 0))
        if momentum_aligns:
            conditions_met.append("Momentum aligns with RSI")
            
        # Tick flow 3 of 5 confirm
        last_5_ticks = [tick.color for tick in list(self.tick_history)[-5:]]
        green_count = last_5_ticks.count('green')
        red_count = last_5_ticks.count('red')
        
        if (significant_rsi_change and momentum_aligns and 
            self.indicators.rsi > self.indicators.rsi_previous and green_count >= 3):
            conditions_met.append("3/5 green ticks confirm RSI tilt up")
            return TradeSignal(
                strategy_name=self.strategies[30],
                direction='CALL',
                confidence=0.85,
                hold_time=9,
                entry_reason="RSI gradient tilt upward",
                conditions_met=conditions_met
            )
            
        elif (significant_rsi_change and momentum_aligns and 
              self.indicators.rsi < self.indicators.rsi_previous and red_count >= 3):
            conditions_met.append("3/5 red ticks confirm RSI tilt down")
            return TradeSignal(
                strategy_name=self.strategies[30],
                direction='PUT',
                confidence=0.85,
                hold_time=9,
                entry_reason="RSI gradient tilt downward",
                conditions_met=conditions_met
            )
            
        return None
        
    def _strategy_31(self) -> Optional[TradeSignal]:
        """Rebound from Flat Session - Play fakeout from low-volatility zone"""
        if len(self.tick_history) < 3:
            return None
            
        conditions_met = []
        
        # Volatility < 0.5%
        very_low_vol = self.indicators.volatility < 0.5
        if very_low_vol:
            conditions_met.append(f"Very low volatility ({self.indicators.volatility:.2f}%)")
            
        current_price = self.tick_history[-1].price
        
        # Sudden tick expansion + Bollinger band cross
        breaks_upper = current_price > self.indicators.bb_upper
        breaks_lower = current_price < self.indicators.bb_lower
        
        if very_low_vol and breaks_upper:
            conditions_met.extend(["Sudden expansion", "Upper BB break"])
            return TradeSignal(
                strategy_name=self.strategies[31],
                direction='CALL',
                confidence=0.78,
                hold_time=6,
                entry_reason="Rebound from flat session - upward expansion",
                conditions_met=conditions_met
            )
            
        elif very_low_vol and breaks_lower:
            conditions_met.extend(["Sudden expansion", "Lower BB break"])
            return TradeSignal(
                strategy_name=self.strategies[31],
                direction='PUT',
                confidence=0.78,
                hold_time=6,
                entry_reason="Rebound from flat session - downward expansion",
                conditions_met=conditions_met
            )
            
        return None
        
    def _strategy_32(self) -> Optional[TradeSignal]:
        """Opposite Color Flush - Trade when majority ticks flip direction"""
        if len(self.tick_history) < 5:
            return None
            
        conditions_met = []
        
        # Check for pattern: 3 red ticks → 2 green ticks
        last_5_ticks = [tick.color for tick in list(self.tick_history)[-5:]]
        first_3 = last_5_ticks[:3]
        last_2 = last_5_ticks[3:]
        
        # RSI also rises
        rsi_rising = self.indicators.rsi > self.indicators.rsi_previous
        rsi_falling = self.indicators.rsi < self.indicators.rsi_previous
        
        if (all(color == 'red' for color in first_3) and 
            all(color == 'green' for color in last_2) and rsi_rising):
            conditions_met.extend(["3 red → 2 green flip", "RSI rising"])
            return TradeSignal(
                strategy_name=self.strategies[32],
                direction='CALL',
                confidence=0.79,
                hold_time=5,
                entry_reason="Opposite color flush - reversal to green",
                conditions_met=conditions_met
            )
            
        elif (all(color == 'green' for color in first_3) and 
              all(color == 'red' for color in last_2) and rsi_falling):
            conditions_met.extend(["3 green → 2 red flip", "RSI falling"])
            return TradeSignal(
                strategy_name=self.strategies[32],
                direction='PUT',
                confidence=0.79,
                hold_time=5,
                entry_reason="Opposite color flush - reversal to red",
                conditions_met=conditions_met
            )
            
        return None
        
    def _strategy_33(self) -> Optional[TradeSignal]:
        """RSI Ghost Divergence - Hidden divergence entry"""
        if len(self.tick_history) < 10:
            return None
            
        conditions_met = []
        
        # Simplified divergence detection using recent price action
        recent_prices = [tick.price for tick in list(self.tick_history)[-10:]]
        current_price = recent_prices[-1]
        
        # Find local highs and lows
        price_low = min(recent_prices)
        price_high = max(recent_prices)
        
        # Simplified divergence conditions
        if current_price == price_low and self.indicators.rsi > 40:
            # Price makes lower low, RSI makes higher low
            conditions_met.extend(["Price at recent low", "RSI above 40 (higher low)"])
            
            # Confirm with 2 ticks
            last_2_ticks = [tick.color for tick in list(self.tick_history)[-2:]]
            if last_2_ticks.count('green') >= 1:
                conditions_met.append("Green tick confirmation")
                return TradeSignal(
                    strategy_name=self.strategies[33],
                    direction='CALL',
                    confidence=0.81,
                    hold_time=7,
                    entry_reason="RSI ghost divergence - bullish",
                    conditions_met=conditions_met
                )
                
        elif current_price == price_high and self.indicators.rsi < 60:
            # Price makes higher high, RSI makes lower high
            conditions_met.extend(["Price at recent high", "RSI below 60 (lower high)"])
            
            # Confirm with 2 ticks
            last_2_ticks = [tick.color for tick in list(self.tick_history)[-2:]]
            if last_2_ticks.count('red') >= 1:
                conditions_met.append("Red tick confirmation")
                return TradeSignal(
                    strategy_name=self.strategies[33],
                    direction='PUT',
                    confidence=0.81,
                    hold_time=7,
                    entry_reason="RSI ghost divergence - bearish",
                    conditions_met=conditions_met
                )
                
        return None
        
    def _strategy_34(self) -> Optional[TradeSignal]:
        """Triple Tick Momentum Snap - Burst of 3 heavy momentum ticks"""
        if len(self.tick_history) < 3:
            return None
            
        conditions_met = []
        
        # 3 fast ticks in 1 direction
        last_3_ticks = [tick.color for tick in list(self.tick_history)[-3:]]
        all_green = all(color == 'green' for color in last_3_ticks)
        all_red = all(color == 'red' for color in last_3_ticks)
        
        # MACD + Momentum confirm
        strong_momentum = abs(self.indicators.momentum) > 0.25
        macd_confirms = ((self.indicators.macd > 0.15 and all_green) or 
                        (self.indicators.macd < -0.15 and all_red))
        
        if strong_momentum:
            conditions_met.append(f"Strong momentum ({self.indicators.momentum:.2f}%)")
        if macd_confirms:
            conditions_met.append("MACD confirms direction")
            
        # Check tick timing for "fast" characteristic (simplified)
        if len(self.tick_history) >= 3:
            recent_ticks = list(self.tick_history)[-3:]
            time_intervals = []
            for i in range(1, len(recent_ticks)):
                interval = recent_ticks[i].timestamp - recent_ticks[i-1].timestamp
                time_intervals.append(interval)
            
            fast_ticks = all(interval < 1.5 for interval in time_intervals)  # Fast ticks
            if fast_ticks:
                conditions_met.append("Fast tick sequence")
        
        if all_green and strong_momentum and macd_confirms:
            conditions_met.append("3 fast green ticks")
            return TradeSignal(
                strategy_name=self.strategies[34],
                direction='CALL',
                confidence=0.87,
                hold_time=6,
                entry_reason="Triple tick momentum snap upward",
                conditions_met=conditions_met
            )
            
        elif all_red and strong_momentum and macd_confirms:
            conditions_met.append("3 fast red ticks")
            return TradeSignal(
                strategy_name=self.strategies[34],
                direction='PUT',
                confidence=0.87,
                hold_time=6,
                entry_reason="Triple tick momentum snap downward",
                conditions_met=conditions_met
            )
            
        return None
        
    def _strategy_35(self) -> Optional[TradeSignal]:
        """Hybrid Confluence Gate - Composite system, enter only if all match"""
        if len(self.tick_history) < 4:
            return None
            
        conditions_met = []
        
        # Add volatility filter to prevent excessive signals
        volatility_acceptable = 0.5 <= self.indicators.volatility <= 2.5
        if not volatility_acceptable:
            return None
            
        # Entry Gate requirements:
        # RSI > 52 (Rise) or < 48 (Fall) - more restrictive
        rsi_bullish = self.indicators.rsi > 52
        rsi_bearish = self.indicators.rsi < 48
        
        # MACD > 0.3 or < -0.3 - more restrictive
        macd_strong_bull = self.indicators.macd > 0.3
        macd_strong_bear = self.indicators.macd < -0.3
        
        # Momentum > ±0.25% - more restrictive
        momentum_strong = abs(self.indicators.momentum) > 0.25
        momentum_bull = self.indicators.momentum > 0.25
        momentum_bear = self.indicators.momentum < -0.25
        
        # Tick flow 3 of 4 aligned
        last_4_ticks = [tick.color for tick in list(self.tick_history)[-4:]]
        green_count = last_4_ticks.count('green')
        red_count = last_4_ticks.count('red')
        tick_flow_bull = green_count >= 3
        tick_flow_bear = red_count >= 3
        
        # Additional filter: Price should be trending in the same direction
        current_price = self.tick_history[-1].price
        ema_trend_bull = current_price > self.indicators.ema5 > self.indicators.ema20
        ema_trend_bear = current_price < self.indicators.ema5 < self.indicators.ema20
        
        # Check all bullish conditions (more restrictive)
        if (rsi_bullish and macd_strong_bull and momentum_bull and tick_flow_bull and ema_trend_bull):
            conditions_met.extend([
                "RSI > 52", "MACD > 0.3", "Momentum > 0.25%", "3/4 green ticks", "EMA trend up"
            ])
            return TradeSignal(
                strategy_name=self.strategies[35],
                direction='CALL',
                confidence=0.88,  # Reduced from 0.92 to be less aggressive
                hold_time=8,  # Reduced from 10 to be less blocking
                entry_reason="Hybrid confluence gate - all bullish signals aligned",
                conditions_met=conditions_met
            )
            
        # Check all bearish conditions (more restrictive)
        elif (rsi_bearish and macd_strong_bear and momentum_bear and tick_flow_bear and ema_trend_bear):
            conditions_met.extend([
                "RSI < 48", "MACD < -0.3", "Momentum < -0.25%", "3/4 red ticks", "EMA trend down"
            ])
            return TradeSignal(
                strategy_name=self.strategies[35],
                direction='PUT',
                confidence=0.88,  # Reduced from 0.92 to be less aggressive
                hold_time=8,  # Reduced from 10 to be less blocking
                entry_reason="Hybrid confluence gate - all bearish signals aligned",
                conditions_met=conditions_met
            )
            
        return None
        
    def get_current_indicators(self) -> Dict:
        """Get current technical indicators for display"""
        return {
            'rsi': round(self.indicators.rsi, 2),
            'rsi_previous': round(self.indicators.rsi_previous, 2),
            'macd': round(self.indicators.macd, 4),
            'macd_signal': round(self.indicators.macd_signal, 4),
            'momentum': round(self.indicators.momentum, 3),
            'volatility': round(self.indicators.volatility, 2),
            'bb_upper': round(self.indicators.bb_upper, 5),
            'bb_middle': round(self.indicators.bb_middle, 5),
            'bb_lower': round(self.indicators.bb_lower, 5),
            'ema3': round(self.indicators.ema3, 5),
            'ema5': round(self.indicators.ema5, 5),
            'ema20': round(self.indicators.ema20, 5),
            'tick_count': len(self.tick_history),
            'total_strategies': len(self.strategies)
        }
        
    def get_strategy_status(self) -> Dict:
        """Get current status of all 35 strategies"""
        return {
            'last_scan': self.last_strategy_scan,
            'total_scans': self.total_scans,
            'signals_generated': self.signals_generated,
            'strategies_count': len(self.strategies),
            'scan_frequency': '30ms',
            'active_strategies': len([s for s in self.last_strategy_scan.values() if s.get('active', False)])
        }
        
    def get_performance_summary(self) -> Dict:
        """Get performance summary of all strategies"""
        if not self.last_strategy_scan:
            return {'status': 'No scans completed yet'}
            
        active_strategies = []
        inactive_strategies = []
        
        for strategy_id, status in self.last_strategy_scan.items():
            strategy_info = {
                'id': strategy_id,
                'name': status['name'],
                'active': status.get('active', False)
            }
            
            if status.get('active', False):
                signal = status.get('signal')
                if signal:
                    strategy_info.update({
                        'direction': signal.direction,
                        'confidence': signal.confidence,
                        'hold_time': signal.hold_time,
                        'conditions_met': len(signal.conditions_met)
                    })
                active_strategies.append(strategy_info)
            else:
                inactive_strategies.append(strategy_info)
                
        return {
            'total_strategies': 35,
            'active_count': len(active_strategies),
            'inactive_count': len(inactive_strategies),
            'active_strategies': active_strategies,
            'scan_stats': {
                'total_scans': self.total_scans,
                'signals_generated': self.signals_generated,
                'signal_rate': f"{(self.signals_generated/max(1, self.total_scans)*100):.1f}%"
            }
        }
        
    def reset_performance_stats(self):
        """Reset performance tracking statistics"""
        self.strategy_performance = {}
        self.total_scans = 0
        self.signals_generated = 0
        self.last_strategy_scan = {}
        
    def _generate_force_signal(self) -> Optional[TradeSignal]:
        """Generate a force signal for testing purposes when no natural signals occur"""
        if len(self.tick_history) < 5:
            return None
            
        # Create a simple force signal based on basic conditions
        current_price = self.tick_history[-1].price
        last_tick = self.tick_history[-1].color
        
        # Simple signal based on last tick color and basic momentum
        direction = 'CALL' if last_tick == 'red' else 'PUT'  # Contrarian approach
        confidence = 0.60  # Moderate confidence for force signals
        
        conditions = ["Force signal for testing", f"Last tick: {last_tick}", f"Price: {current_price:.2f}"]
        
        return TradeSignal(
            strategy_name="Force Signal Generator",
            direction=direction,
            confidence=confidence,
            hold_time=12,
            entry_reason="Force signal to ensure trading activity",
            conditions_met=conditions
        )
