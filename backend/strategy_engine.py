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
    rsi7: float = 50.0  # Faster 7-period RSI for early reversal detection
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0  # Added MACD histogram for clearer crossover detection
    momentum: float = 0.0
    volatility: float = 1.0
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_middle: float = 0.0
    ema3: float = 0.0
    ema5: float = 0.0
    ema20: float = 0.0
    rsi_previous: float = 50.0  # For tracking RSI changes
    atr: float = 0.0  # Average True Range for volatility measurement
    donchian_upper: float = 0.0  # Donchian channel upper band
    donchian_lower: float = 0.0  # Donchian channel lower band
    tick_speed: float = 1.0  # Average time between ticks
    tick_acceleration: float = 0.0  # Rate of change in tick speed
    volume_trend: float = 0.0  # Simulated volume trend indicator


@dataclass
class TradeSignal:
    """Trade signal from strategy"""
    strategy_name: str
    direction: str  # 'CALL' or 'PUT'
    confidence: float  # 0.0 to 1.0
    hold_time: int  # seconds
    entry_reason: str
    conditions_met: List[str]
    risk_ratio: float = 2.0  # Risk-reward ratio target (default 2:1)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class StrategyOptimizer:
    """Handles strategy parameter optimization"""
    
    def __init__(self):
        self.strategy_performance = {}
        self.best_parameters = {}
        
    def record_trade_result(self, strategy_name: str, params: Dict, result: bool, profit: float):
        """Record trade result for parameter optimization"""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {
                'total_trades': 0,
                'winning_trades': 0,
                'total_profit': 0.0,
                'params_performance': {}
            }
        
        self.strategy_performance[strategy_name]['total_trades'] += 1
        if result:
            self.strategy_performance[strategy_name]['winning_trades'] += 1
        self.strategy_performance[strategy_name]['total_profit'] += profit
        
        # Track performance by parameter combinations
        param_key = str(params)
        if param_key not in self.strategy_performance[strategy_name]['params_performance']:
            self.strategy_performance[strategy_name]['params_performance'][param_key] = {
                'params': params,
                'total_trades': 0,
                'winning_trades': 0,
                'total_profit': 0.0
            }
        
        self.strategy_performance[strategy_name]['params_performance'][param_key]['total_trades'] += 1
        if result:
            self.strategy_performance[strategy_name]['params_performance'][param_key]['winning_trades'] += 1
        self.strategy_performance[strategy_name]['params_performance'][param_key]['total_profit'] += profit
        
        # Update best parameters if we have enough data
        self._update_best_parameters(strategy_name)
    
    def _update_best_parameters(self, strategy_name: str):
        """Find best parameters for a strategy based on profit and win rate"""
        if strategy_name not in self.strategy_performance:
            return
        
        performance = self.strategy_performance[strategy_name]['params_performance']
        best_profit = -float('inf')
        best_params = None
        
        for param_key, stats in performance.items():
            # Only consider parameter sets with at least 5 trades
            if stats['total_trades'] < 5:
                continue
                
            # Calculate metrics
            win_rate = stats['winning_trades'] / stats['total_trades'] if stats['total_trades'] > 0 else 0
            profit = stats['total_profit']
            
            # Scoring function: balance win rate and profit
            score = profit * (0.5 + 0.5 * win_rate)  # Weight profit by win rate
            
            if score > best_profit:
                best_profit = score
                best_params = stats['params']
        
        if best_params:
            self.best_parameters[strategy_name] = best_params
    
    def get_best_parameters(self, strategy_name: str, default_params: Dict) -> Dict:
        """Get best parameters for a strategy or return defaults"""
        return self.best_parameters.get(strategy_name, default_params)


class StrategyEngine:
    """Advanced strategy engine with optimized binary trading strategies"""
    
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
        
        # Strategy configurations - 38 total strategies (3 new added)
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
            # ... existing strategies 17-35 ...
            36: "MACD-RSI-Bollinger Triple Confirmation", # NEW
            37: "Tick Acceleration Momentum Entry", # NEW
            38: "Donchian Breakout after Consolidation" # NEW
        }
        
        # Initialize strategy optimizer
        self.optimizer = StrategyOptimizer()
        
        # Risk management tracking
        self.consecutive_losses = 0
        self.max_consecutive_losses = 2  # Stop after 2 consecutive losses
        self.session_trades = 0
        self.max_session_trades = 20  # Cap at 20 trades per session
        self.consecutive_losses_warning_shown = False
        self.session_trades_warning_shown = False
        
        # Tick speed analysis
        self.tick_intervals = deque(maxlen=20)
        self.last_tick_time = 0
        
        # ATR calculation data
        self.true_ranges = deque(maxlen=14)  # 14-period ATR
        self.last_close = None
        
    def start_scanning(self, signal_callback):
        """Start real-time strategy scanning"""
        self.signal_callback = signal_callback
        self.is_running = True
        self.session_trades = 0  # Reset session trade counter
        self.consecutive_losses_warning_shown = False  # Reset warning flags
        self.session_trades_warning_shown = False      # Reset warning flags
        
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
            
        # Calculate tick interval for speed analysis
        if self.last_tick_time > 0:
            interval = timestamp - self.last_tick_time
            self.tick_intervals.append(interval)
        self.last_tick_time = timestamp
        
        # Determine tick color
        color = 'green'
        if len(self.tick_history) > 0:
            last_price = self.tick_history[-1].price
            color = 'green' if price > last_price else 'red'
            
        # Update ATR calculation data
        if self.last_close is not None:
            # Calculate true range
            high = max(price, self.last_close)
            low = min(price, self.last_close)
            tr = high - low
            self.true_ranges.append(tr)
        self.last_close = price
            
        tick = TickData(price, timestamp, color)
        self.tick_history.append(tick)
        
        # Update indicators
        self._update_indicators()
        
        # Scan for signals
        if len(self.tick_history) >= 20:  # Need minimum ticks for analysis
            self._scan_strategies()
            
    def _update_indicators(self):
        """Calculate technical indicators from tick history with advanced metrics"""
        if len(self.tick_history) < 20:
            return
            
        prices = np.array([tick.price for tick in self.tick_history])
        
        # Store previous RSI for change tracking
        self.indicators.rsi_previous = self.indicators.rsi
        
        # Standard RSI calculation (14 period)
        self.indicators.rsi = self._calculate_rsi(prices, 14)
        
        # Fast RSI calculation (7 period) for earlier reversal detection
        self.indicators.rsi7 = self._calculate_rsi(prices, 7)
        
        # MACD calculation with improved params (12,26,9)
        self.indicators.macd, self.indicators.macd_signal, self.indicators.macd_histogram = self._calculate_macd(prices, 12, 26, 9)
        
        # Momentum calculation
        self.indicators.momentum = self._calculate_momentum(prices, 10)
        
        # Volatility calculation
        self.indicators.volatility = self._calculate_volatility(prices, 20)
        
        # Bollinger Bands (20,2) - 20 period SMA with 2 standard deviations
        self.indicators.bb_upper, self.indicators.bb_middle, self.indicators.bb_lower = self._calculate_bollinger_bands(prices, 20, 2)
        
        # EMAs
        self.indicators.ema3 = self._calculate_ema(prices, 3)
        self.indicators.ema5 = self._calculate_ema(prices, 5)
        self.indicators.ema20 = self._calculate_ema(prices, 20)
        
        # ATR calculation (14 period)
        self.indicators.atr = np.mean(self.true_ranges) if len(self.true_ranges) > 0 else 0
        
        # Donchian Channels (20 period)
        self.indicators.donchian_upper, self.indicators.donchian_lower = self._calculate_donchian_channels(prices, 20)
        
        # Tick speed analysis
        if len(self.tick_intervals) >= 5:
            # Calculate average interval
            avg_interval = np.mean(self.tick_intervals)
            # Calculate recent vs older tick speed
            recent_intervals = list(self.tick_intervals)[-3:]
            older_intervals = list(self.tick_intervals)[:-3]
            if older_intervals:
                recent_avg = np.mean(recent_intervals)
                older_avg = np.mean(older_intervals)
                # Tick acceleration: ratio of recent to older intervals
                # Values < 1 mean ticks are getting faster (accelerating)
                if older_avg > 0:
                    self.indicators.tick_acceleration = recent_avg / older_avg
                else:
                    self.indicators.tick_acceleration = 1.0
            else:
                self.indicators.tick_acceleration = 1.0
            
            self.indicators.tick_speed = avg_interval
        
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
        
    def _calculate_macd(self, prices: np.ndarray, fast_period=12, slow_period=26, signal_period=9) -> Tuple[float, float, float]:
        """Calculate MACD and signal line with precise parameters"""
        if len(prices) < slow_period:
            return 0.0, 0.0, 0.0
            
        # Calculate EMAs with proper parameters
        ema_fast = self._calculate_ema(prices, fast_period)
        ema_slow = self._calculate_ema(prices, slow_period)
        
        # Calculate MACD line
        macd = ema_fast - ema_slow
        
        # Calculate Signal line (9-period EMA of MACD)
        if len(prices) >= slow_period + signal_period:
            # Generate MACD history to calculate its EMA
            macd_history = []
            for i in range(signal_period):
                if len(prices) > slow_period + i:
                    hist_fast = self._calculate_ema(prices[:-i] if i > 0 else prices, fast_period)
                    hist_slow = self._calculate_ema(prices[:-i] if i > 0 else prices, slow_period)
                    macd_history.append(hist_fast - hist_slow)
            
            if macd_history:
                signal = self._calculate_ema(np.array(macd_history), signal_period)
            else:
                signal = macd * 0.9  # Fallback if history is too short
        else:
            signal = macd * 0.9  # Fallback if not enough data
            
        # Calculate histogram (MACD - Signal)
        histogram = macd - signal
        
        return macd, signal, histogram
        
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
            
        # Calculate SMA first
        sma = np.mean(prices[-period:])
        
        # Calculate smoothing factor
        multiplier = 2 / (period + 1)
        
        # Calculate EMA
        ema = sma
        
        # If we have enough data, refine the EMA calculation
        if len(prices) > period:
            for price in prices[-period:]:
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
        
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands with configurable std deviation"""
        if len(prices) < period:
            mean_price = np.mean(prices)
            return mean_price * 1.02, mean_price, mean_price * 0.98
            
        recent_prices = prices[-period:]
        middle = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return upper, middle, lower
        
    def _calculate_donchian_channels(self, prices: np.ndarray, period: int = 20) -> Tuple[float, float]:
        """Calculate Donchian Channels for breakout detection"""
        if len(prices) < period:
            return np.max(prices), np.min(prices)
            
        recent_prices = prices[-period:]
        upper = np.max(recent_prices)
        lower = np.min(recent_prices)
        
        return upper, lower
        
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
        """Scan all strategies for trade signals with risk management"""
        # Check risk management rules first
        if self.consecutive_losses >= self.max_consecutive_losses and self.max_consecutive_losses > 0:
            # Only show warning once
            if not hasattr(self, 'consecutive_losses_warning_shown') or not self.consecutive_losses_warning_shown:
                print(f"🛑 Risk management: {self.consecutive_losses} consecutive losses reached. Trading paused.")
                self.consecutive_losses_warning_shown = True
            print(f"DEBUG: Strategy engine stopping due to consecutive losses: {self.consecutive_losses}/{self.max_consecutive_losses}")
            return
            
        if self.session_trades >= self.max_session_trades and self.max_session_trades > 0:
            # Only show warning once
            if not hasattr(self, 'session_trades_warning_shown') or not self.session_trades_warning_shown:
                print(f"🛑 Risk management: Maximum {self.max_session_trades} trades for this session reached.")
                self.session_trades_warning_shown = True
            print(f"DEBUG: Strategy engine stopping due to session trades limit: {self.session_trades}/{self.max_session_trades}")
            return
            
        signals = []
        strategy_status = {}
        
        # Scan all strategies in real-time FIRST
        # (36-38 are the new strategies)
        for i in range(1, 39):
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
                    'name': self.strategies.get(i, f"Strategy {i}"),
                    'active': False,
                    'error': str(e)
                }
        
        # Add force signal mechanism if enabled in config AND no natural signals found
        try:
            import config
            if hasattr(config, 'FORCE_STRATEGY_SIGNALS') and config.FORCE_STRATEGY_SIGNALS:
                # Force at least one signal every 3 scans if no signals generated (reduced for more trading)
                if self.total_scans % 3 == 0 and len(signals) == 0:
                    force_signal = self._generate_force_signal()
                    if force_signal:
                        signals.append(force_signal)
                        print(f"🔧 FORCE SIGNAL GENERATED: {force_signal.strategy_name}")
                        
                # EMERGENCY: Generate signal on every 10th scan regardless
                elif self.total_scans % 10 == 0:
                    force_signal = self._generate_force_signal()
                    if force_signal:
                        signals.append(force_signal)
                        print(f"🚨 EMERGENCY FORCE SIGNAL: {force_signal.strategy_name}")
        except Exception as e:
            print(f"⚠️ Force signal error: {e}")
        
        # Store strategy status for monitoring
        self.last_strategy_scan = strategy_status
        
        # Handle multiple signals - send all high-confidence signals
        if signals and self.signal_callback:
            self.signals_generated += len(signals)
            
            # Sort by confidence (highest first)
            signals.sort(key=lambda x: x.confidence, reverse=True)
            
            # Send signals with confidence > 0.50 (further reduced threshold for more trading)
            try:
                import config
                threshold = getattr(config, 'SIGNAL_CONFIDENCE_THRESHOLD', 0.50)
            except:
                threshold = 0.50
                
            high_confidence_signals = [s for s in signals if s.confidence > threshold]
            
            if high_confidence_signals:
                # Send the best signal immediately
                best_signal = high_confidence_signals[0]
                
                # Add risk-reward ratio and stop-loss/take-profit levels
                current_price = self.tick_history[-1].price if self.tick_history else 0
                risk = current_price * 0.005  # 0.5% risk
                
                # Set stop-loss and take-profit based on risk-reward ratio (2:1)
                if best_signal.direction == 'CALL':
                    best_signal.stop_loss = current_price - risk
                    best_signal.take_profit = current_price + (risk * 2)  # 2:1 ratio
                else:  # PUT
                    best_signal.stop_loss = current_price + risk
                    best_signal.take_profit = current_price - (risk * 2)  # 2:1 ratio
                
                self.signal_callback(best_signal)
                # Don't increment session trades here - this is now handled by trading_bot after trade completion
                # The trading_bot manages the session trade counter
                
                # Log all active strategies for monitoring
                active_strategies = [f"{s.strategy_name} ({s.confidence:.2f})" 
                                   for s in high_confidence_signals[:3]]
                print(f"[STRATEGY ENGINE] 🎯 Sending signal: {best_signal.strategy_name} ({best_signal.confidence:.2f})")
                print(f"[STRATEGY ENGINE] Session trades: {self.session_trades}/{self.max_session_trades}")
                if len(active_strategies) > 1:
                    print(f"[STRATEGY ENGINE] Other active: {', '.join(active_strategies[1:])}")
            
            # Also send strategy scan summary (less frequent logging)
            if self.total_scans % 50 == 0:  # Every 50 scans (reduced from 100)
                active_count = len([s for s in strategy_status.values() if s['active']])
                print(f"[STRATEGY ENGINE] Scan #{self.total_scans}: {active_count}/38 strategies active, {len(signals)} signals")
                print(f"[STRATEGY ENGINE] Session trades: {self.session_trades}/{self.max_session_trades}, Consecutive losses: {self.consecutive_losses}")

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
                    direction='PUT',
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
                    direction='CALL',
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
                    direction='PUT',
                    confidence=0.8,
                    hold_time=8,
                    entry_reason="Micro-trend momentum upward",
                    conditions_met=conditions_met
                )
                
            elif red_count >= 3 and macd_direction == 'down' and strong_macd and strong_momentum and rsi_range:
                conditions_met.append("3/4 red ticks align with MACD")
                return TradeSignal(
                    strategy_name=self.strategies[2],
                    direction='CALL',
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
                        direction='PUT',
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
                        direction='CALL',
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
                    direction='CALL',
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
                    direction='PUT',
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
                direction='CALL',
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
                direction='PUT',
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
                    direction='CALL',
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
                    direction='PUT',
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
                    direction='PUT',
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
                    direction='CALL',
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
                
                # Check direction and momentum
                last_tick = self.tick_history[-1].color
                momentum_strong = abs(self.indicators.momentum) > 0.1
                
                if last_tick == 'green' and momentum_strong and self.indicators.momentum > 0:
                    conditions_met.extend(["Green tick", "Positive momentum"])
                    return TradeSignal(
                        strategy_name=self.strategies[10],
                        direction='PUT',
                        confidence=0.72,
                        hold_time=4,
                        entry_reason="Fast tick speed with bullish momentum",
                        conditions_met=conditions_met
                    )
                    
                elif last_tick == 'red' and momentum_strong and self.indicators.momentum < 0:
                    conditions_met.extend(["Red tick", "Negative momentum"])
                    return TradeSignal(
                        strategy_name=self.strategies[10],
                        direction='CALL',
                        confidence=0.72,
                        hold_time=4,
                        entry_reason="Fast tick speed with bearish momentum",
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
                    direction='PUT',
                    confidence=0.8,
                    hold_time=7,
                    entry_reason="Volatility collapse breakout up",
                    conditions_met=conditions_met
                )
                
            elif breakout_down and last_tick.color == 'red':
                conditions_met.append("Downward breakout from compression")
                return TradeSignal(
                    strategy_name=self.strategies[11],
                    direction='CALL',
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
                    direction='PUT',
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
                    direction='CALL',
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
        
    def _strategy_36(self) -> Optional[TradeSignal]:
        """MACD-RSI-Bollinger Triple Confirmation Strategy"""
        if len(self.tick_history) < 20:
            return None
            
        conditions_met = []
        
        # 1. Check MACD signal line crossover (MACD crosses Signal)
        macd_bullish_cross = (self.indicators.macd > self.indicators.macd_signal and 
                            self.indicators.macd_histogram > 0)
        macd_bearish_cross = (self.indicators.macd < self.indicators.macd_signal and 
                            self.indicators.macd_histogram < 0)
        
        # 2. Check fast RSI(7) for early reversal detection
        rsi_rising = self.indicators.rsi7 > 30 and self.indicators.rsi7 < 50  # Bullish zone
        rsi_falling = self.indicators.rsi7 < 70 and self.indicators.rsi7 > 50  # Bearish zone
        
        # 3. Check Bollinger Bands for price confirmation
        current_price = self.tick_history[-1].price
        near_lower_band = current_price < (self.indicators.bb_lower * 1.005)  # Within 0.5% of lower band
        near_upper_band = current_price > (self.indicators.bb_upper * 0.995)  # Within 0.5% of upper band
        
        # 4. Check tick speed for momentum confirmation
        tick_accelerating = self.indicators.tick_acceleration < 0.9  # Ticks getting 10% faster
        
        # 5. EMA trend alignment
        ema_uptrend = self.indicators.ema5 > self.indicators.ema20
        ema_downtrend = self.indicators.ema5 < self.indicators.ema20
        
        # Check for bullish setup
        if (macd_bullish_cross and rsi_rising and near_lower_band and 
            tick_accelerating and ema_uptrend):
            conditions_met.extend([
                "MACD bullish crossover",
                f"Fast RSI(7) rising from oversold ({self.indicators.rsi7:.1f})",
                "Price near lower Bollinger Band",
                f"Tick acceleration detected ({self.indicators.tick_acceleration:.2f})",
                "EMA5 > EMA20 (uptrend)"
            ])
            return TradeSignal(
                strategy_name=self.strategies[36],
                direction='CALL',
                confidence=0.88,
                hold_time=10,
                entry_reason="Triple confirmation bullish setup",
                conditions_met=conditions_met,
                risk_ratio=2.0  # 2:1 reward-to-risk ratio
            )
            
        # Check for bearish setup
        if (macd_bearish_cross and rsi_falling and near_upper_band and 
            tick_accelerating and ema_downtrend):
            conditions_met.extend([
                "MACD bearish crossover",
                f"Fast RSI(7) falling from overbought ({self.indicators.rsi7:.1f})",
                "Price near upper Bollinger Band",
                f"Tick acceleration detected ({self.indicators.tick_acceleration:.2f})",
                "EMA5 < EMA20 (downtrend)"
            ])
            return TradeSignal(
                strategy_name=self.strategies[36],
                direction='PUT',
                confidence=0.88,
                hold_time=10,
                entry_reason="Triple confirmation bearish setup",
                conditions_met=conditions_met,
                risk_ratio=2.0  # 2:1 reward-to-risk ratio
            )
            
        return None

    def _strategy_37(self) -> Optional[TradeSignal]:
        """Tick Acceleration Momentum Entry Strategy"""
        if len(self.tick_history) < 10:
            return None
            
        conditions_met = []
        
        # Check for significant tick acceleration (ticks getting much faster)
        strong_acceleration = self.indicators.tick_acceleration < 0.7  # 30% faster ticks
        if strong_acceleration:
            conditions_met.append(f"Strong tick acceleration ({self.indicators.tick_acceleration:.2f})")
        else:
            # Skip if no significant acceleration
            return None
        
        # Check tick direction consistency
        last_5_ticks = [tick.color for tick in list(self.tick_history)[-5:]]
        green_count = last_5_ticks.count('green')
        red_count = last_5_ticks.count('red')
        
        # Need at least 3 of 5 ticks in same direction
        direction_consistency = green_count >= 3 or red_count >= 3
        
        # Volatility must be sufficient but not extreme
        normal_volatility = 0.8 <= self.indicators.volatility <= 2.0
        if normal_volatility:
            conditions_met.append(f"Appropriate volatility ({self.indicators.volatility:.2f}%)")
        else:
            # Skip if volatility is too low or too high
            return None
            
        # RSI confirmation
        rsi_confirming_up = 40 <= self.indicators.rsi7 <= 60 and self.indicators.rsi7 > self.indicators.rsi_previous
        rsi_confirming_down = 40 <= self.indicators.rsi7 <= 60 and self.indicators.rsi7 < self.indicators.rsi_previous
        
        # Volume/momentum confirmation
        momentum_confirming_up = self.indicators.momentum > 0.1
        momentum_confirming_down = self.indicators.momentum < -0.1
        
        if green_count >= 3 and direction_consistency and rsi_confirming_up and momentum_confirming_up:
            conditions_met.extend([
                f"{green_count}/5 green ticks",
                "RSI(7) rising in neutral zone",
                f"Positive momentum ({self.indicators.momentum:.2f}%)"
            ])
            return TradeSignal(
                strategy_name=self.strategies[37],
                direction='CALL',
                confidence=0.85,
                hold_time=8,
                entry_reason="Accelerating tick momentum entry (bullish)",
                conditions_met=conditions_met,
                risk_ratio=2.0
            )
            
        elif red_count >= 3 and direction_consistency and rsi_confirming_down and momentum_confirming_down:
            conditions_met.extend([
                f"{red_count}/5 red ticks",
                "RSI(7) falling in neutral zone",
                f"Negative momentum ({self.indicators.momentum:.2f}%)"
            ])
            return TradeSignal(
                strategy_name=self.strategies[37],
                direction='PUT',
                confidence=0.85,
                hold_time=8,
                entry_reason="Accelerating tick momentum entry (bearish)",
                conditions_met=conditions_met,
                risk_ratio=2.0
            )
            
        return None

    def _strategy_38(self) -> Optional[TradeSignal]:
        """Donchian Breakout after Consolidation"""
        if len(self.tick_history) < 20:
            return None
            
        conditions_met = []
        
        # Get current price and last few ticks
        current_price = self.tick_history[-1].price
        last_tick = self.tick_history[-1].color
        
        # 1. Check for low volatility period first (consolidation)
        low_volatility = self.indicators.volatility < 0.8
        if not low_volatility:
            # Skip if we're not in consolidation mode
            return None
            
        conditions_met.append(f"Low volatility consolidation ({self.indicators.volatility:.2f}%)")
        
        # 2. Check ATR for expanding volatility (breakout confirmation)
        atr_rising = False
        if len(self.true_ranges) >= 14:
            recent_atr = np.mean(list(self.true_ranges)[-7:])
            older_atr = np.mean(list(self.true_ranges)[-14:-7])
            atr_rising = recent_atr > older_atr * 1.2  # 20% increase in ATR
        
        if atr_rising:
            conditions_met.append("ATR expanding (volatility breakout)")
        else:
            # Skip if ATR isn't expanding
            return None
        
        # 3. Check for Donchian Channel breakout
        breaking_upper = current_price > self.indicators.donchian_upper * 0.998  # Within 0.2% of upper
        breaking_lower = current_price < self.indicators.donchian_lower * 1.002  # Within 0.2% of lower
        
        # 4. Tick confirmation
        if breaking_upper and last_tick == 'green':
            conditions_met.extend([
                "Breaking upper Donchian Channel",
                "Green tick confirmation"
            ])
            
            # 5. Multi-step confirmation checklist
            confirmation_count = 0
            
            # Trend check (EMA alignment)
            if self.indicators.ema5 > self.indicators.ema20:
                confirmation_count += 1
                conditions_met.append("EMA trend alignment")
                
            # Momentum check (MACD)
            if self.indicators.macd > 0:
                confirmation_count += 1
                conditions_met.append("MACD positive")
                
            # RSI check
            if 40 < self.indicators.rsi < 70:
                confirmation_count += 1
                conditions_met.append("RSI in optimal range")
                
            # Enough confirmations?
            if confirmation_count >= 2:  # Need at least 2 of 3 confirming factors
                return TradeSignal(
                    strategy_name=self.strategies[38],
                    direction='CALL',
                    confidence=0.86,
                    hold_time=9,
                    entry_reason="Donchian breakout after consolidation (bullish)",
                    conditions_met=conditions_met,
                    risk_ratio=2.0
                )
            
        elif breaking_lower and last_tick == 'red':
            conditions_met.extend([
                "Breaking lower Donchian Channel",
                "Red tick confirmation"
            ])
            
            # 5. Multi-step confirmation checklist
            confirmation_count = 0
            
            # Trend check (EMA alignment)
            if self.indicators.ema5 < self.indicators.ema20:
                confirmation_count += 1
                conditions_met.append("EMA trend alignment")
                
            # Momentum check (MACD)
            if self.indicators.macd < 0:
                confirmation_count += 1
                conditions_met.append("MACD negative")
                
            # RSI check
            if 30 < self.indicators.rsi < 60:
                confirmation_count += 1
                conditions_met.append("RSI in optimal range")
                
            # Enough confirmations?
            if confirmation_count >= 2:  # Need at least 2 of 3 confirming factors
                return TradeSignal(
                    strategy_name=self.strategies[38],
                    direction='PUT',
                    confidence=0.86,
                    hold_time=9,
                    entry_reason="Donchian breakout after consolidation (bearish)",
                    conditions_met=conditions_met,
                    risk_ratio=2.0
                )
            
        return None
        
    def get_current_indicators(self) -> Dict:
        """Get current technical indicators for display"""
        return {
            'rsi': round(self.indicators.rsi, 2),
            'rsi7': round(self.indicators.rsi7, 2),  # Added fast RSI
            'rsi_previous': round(self.indicators.rsi_previous, 2),
            'macd': round(self.indicators.macd, 4),
            'macd_signal': round(self.indicators.macd_signal, 4),
            'macd_histogram': round(self.indicators.macd_histogram, 4),  # Added histogram
            'momentum': round(self.indicators.momentum, 3),
            'volatility': round(self.indicators.volatility, 2),
            'bb_upper': round(self.indicators.bb_upper, 5),
            'bb_middle': round(self.indicators.bb_middle, 5),
            'bb_lower': round(self.indicators.bb_lower, 5),
            'ema3': round(self.indicators.ema3, 5),
            'ema5': round(self.indicators.ema5, 5),
            'ema20': round(self.indicators.ema20, 5),
            'atr': round(self.indicators.atr, 5),  # Added ATR
            'donchian_upper': round(self.indicators.donchian_upper, 5),  # Added Donchian
            'donchian_lower': round(self.indicators.donchian_lower, 5),  # Added Donchian
            'tick_speed': round(self.indicators.tick_speed, 3),  # Added tick metrics
            'tick_acceleration': round(self.indicators.tick_acceleration, 3),  # Added tick metrics
            'tick_count': len(self.tick_history),
            'total_strategies': len(self.strategies)
        }
        
    def register_trade_result(self, strategy_name: str, direction: str, result: bool, profit: float):
        """Register trade result for performance tracking and risk management"""
        # Update consecutive loss counter for risk management
        if not result:
            self.consecutive_losses += 1
            print(f"⚠️ Consecutive losses: {self.consecutive_losses}/{self.max_consecutive_losses}")
        else:
            # Reset consecutive losses on a win
            self.consecutive_losses = 0
        
        # Record result for parameter optimization
        # Get current strategy parameters (simplified example)
        params = {
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'rsi_period': 7,
            'bb_period': 20,
            'bb_dev': 2
        }
        
        # Pass to optimizer
        self.optimizer.record_trade_result(strategy_name, params, result, profit)
    
    def reset_warnings(self):
        """Reset all warning flags"""
        self.consecutive_losses_warning_shown = False
        self.session_trades_warning_shown = False
        print("Strategy engine warning flags reset")
    
    def reset_session_trades(self):
        """Reset session trades counter and related warnings"""
        previous = self.session_trades
        self.session_trades = 0
        self.session_trades_warning_shown = False
        print(f"Strategy engine session trades reset from {previous} to 0")
        
    def sync_session_trades(self, count):
        """Sync session trades count with trading bot"""
        self.session_trades = count
        print(f"Strategy engine session trades synced to {count}")
    
    # ... other existing methods ...
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
        
        last_5_ticks = [tick.color for tick in list(self.tick_history)[-5:]]
        
        # Check if first 3 are red and last 2 are green
        first_3_red = all(color == 'red' for color in last_5_ticks[:3])
        last_2_green = all(color == 'green' for color in last_5_ticks[3:])
        
        # Check if first 3 are green and last 2 are red
        first_3_green = all(color == 'green' for color in last_5_ticks[:3])
        last_2_red = all(color == 'red' for color in last_5_ticks[3:])
        
        # RSI also rises
        rsi_rising = self.indicators.rsi > self.indicators.rsi_previous
        rsi_falling = self.indicators.rsi < self.indicators.rsi_previous
        
        if (first_3_red and last_2_green and rsi_rising):
            conditions_met.extend(["3 red → 2 green flip", "RSI rising"])
            return TradeSignal(
                strategy_name=self.strategies[32],
                direction='CALL',
                confidence=0.79,
                hold_time=5,
                entry_reason="Opposite color flush - reversal to green",
                conditions_met=conditions_met
            )
            
        elif (first_3_green and last_2_red and rsi_falling):
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
        """Generate a force signal for testing purposes when no natural signals occur - ENHANCED"""
        if len(self.tick_history) < 3:
            return None
            
        # Create a more dynamic force signal based on current market conditions
        current_price = self.tick_history[-1].price
        last_tick = self.tick_history[-1].color
        
        # Get some basic indicators if available
        rsi = getattr(self.indicators, 'rsi', 50.0)
        volatility = getattr(self.indicators, 'volatility', 1.0)
        
        # More intelligent direction selection
        if rsi > 60:
            direction = 'PUT'
            reason = f"Force signal: RSI overbought ({rsi:.1f})"
        elif rsi < 40:
            direction = 'CALL'
            reason = f"Force signal: RSI oversold ({rsi:.1f})"
        else:
            # Random direction for neutral conditions
            import random
            direction = random.choice(['CALL', 'PUT'])
            reason = f"Force signal: Random direction ({direction})"
        
        # Dynamic confidence based on market conditions
        if volatility > 1.5:
            confidence = 0.65  # Higher confidence in volatile markets
        else:
            confidence = 0.55  # Lower confidence in calm markets
        
        conditions = [
            "Force signal for testing", 
            f"Last tick: {last_tick}", 
            f"Price: {current_price:.2f}",
            f"RSI: {rsi:.1f}",
            f"Volatility: {volatility:.2f}%"
        ]
        
        print(f"🔧 GENERATING FORCE SIGNAL: {direction} with confidence {confidence:.2f}")
        
        return TradeSignal(
            strategy_name="Force Signal Generator",
            direction=direction,
            confidence=confidence,
            hold_time=15,  # 15 seconds
            entry_reason=reason,
            conditions_met=conditions
        )
