"""
Advanced Smart Martingale System with AI-driven unpredictability and comprehensive risk management
"""
import random
import logging
import hashlib
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import config

logger = logging.getLogger(__name__)

class SmartMartingaleSystem:
    def __init__(self):
        self.config = config.MARTINGALE_CONFIG
        self.current_stake = self.config['initial_stake']
        self.consecutive_losses = 0
        self.session_trades = 0
        self.session_profit = 0.0
        self.last_win_time = datetime.now()
        self.pattern_memory = []  # To avoid predictable patterns
        self.stake_history = []  # Historical stakes for analysis
        self.performance_metrics = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'streak_data': [],
            'ai_confidence_scores': [],
            'market_volatility_data': []
        }
        self.martingale_level = 0
        self.cool_down_until = None
        self.emergency_brake_active = False
        self.adaptive_multiplier = self.config['multiplier']
        self.risk_adjusted_multiplier = self.config['multiplier']
        self.session_start_time = datetime.now()
        self.last_trade_time = datetime.now()
        
        # Advanced unpredictability factors
        self.fibonacci_sequence = [1, 1, 2, 3, 5, 8, 13, 21]
        self.chaos_seed = self._generate_chaos_seed()
        self.market_regime_detector = {
            'current_regime': 'NEUTRAL',
            'regime_changes': 0,
            'regime_history': []
        }
        
    def _generate_chaos_seed(self) -> int:
        """Generate a chaos seed based on multiple entropy sources"""
        current_time = datetime.now()
        time_hash = hashlib.md5(str(current_time.microsecond).encode()).hexdigest()
        return int(time_hash[:8], 16) % 1000000
        
    def calculate_next_stake(self, 
                            last_trade_result: str, 
                            account_balance: float,
                            market_volatility: float = 1.0,
                            ai_confidence: float = 0.5,
                            market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Calculate next stake using advanced smart martingale with AI-driven unpredictability
        """
        try:
            # Check emergency brake
            if self._check_emergency_brake():
                return self._emergency_response()
                
            # Check cool-down period
            if self._is_in_cooldown():
                return self._cooldown_response()
                
            # Update performance metrics
            self._update_performance_metrics(last_trade_result, ai_confidence, market_volatility)
            
            # Update consecutive losses counter with advanced logic
            self._update_loss_counter(last_trade_result)
                    
            # Detect market regime changes
            self._update_market_regime(market_volatility, ai_confidence, market_data)
            
            # Smart Martingale Logic with AI enhancement
            stake_calculation = self._calculate_advanced_smart_stake(
                account_balance, 
                market_volatility, 
                ai_confidence,
                market_data
            )
            
            # Apply advanced unpredictability factors
            final_stake = self._apply_advanced_unpredictability(
                stake_calculation['base_stake'],
                ai_confidence,
                market_volatility
            )
            
            # Apply comprehensive risk management
            final_stake = self._apply_comprehensive_risk_management(
                final_stake, 
                account_balance,
                ai_confidence
            )
            
            # Update system state
            self.current_stake = final_stake
            self.session_trades += 1
            self.last_trade_time = datetime.now()
            
            # Store pattern for future unpredictability analysis
            self._update_advanced_pattern_memory(final_stake, ai_confidence, market_volatility)
            
            return {
                'stake': final_stake,
                'consecutive_losses': self.consecutive_losses,
                'martingale_level': self.martingale_level,
                'risk_level': stake_calculation['risk_level'],
                'confidence_adjusted': stake_calculation['confidence_adjusted'],
                'volatility_adjusted': stake_calculation['volatility_adjusted'],
                'chaos_factor_applied': stake_calculation.get('chaos_applied', False),
                'ai_confidence': ai_confidence,
                'market_volatility': market_volatility,
                'unpredictability_score': self._calculate_unpredictability_score(),
                'reasoning': stake_calculation['reasoning'],
                'regime_detected': self.market_regime_detector['current_regime'],
                'emergency_brake_status': self.emergency_brake_active,
                'cool_down_active': self.cool_down_until is not None,
                'adaptive_multiplier': self.adaptive_multiplier
            }
            
        except Exception as e:
            logger.error(f"Error calculating advanced martingale stake: {e}")
            return self._get_safe_fallback_response()
            
    def _check_emergency_brake(self) -> bool:
        """Check if emergency brake should be activated"""
        return (self.consecutive_losses >= self.config['emergency_brake_losses'] or 
                self.emergency_brake_active)
    
    def _is_in_cooldown(self) -> bool:
        """Check if system is in cool-down period"""
        return (self.cool_down_until is not None and 
                datetime.now() < self.cool_down_until)
    
    def _emergency_response(self) -> Dict[str, Any]:
        """Emergency response with minimal stake"""
        self.emergency_brake_active = True
        return {
            'stake': self.config['initial_stake'] * 0.1,  # Ultra-conservative
            'consecutive_losses': self.consecutive_losses,
            'martingale_level': 0,
            'risk_level': 'EMERGENCY',
            'reasoning': 'Emergency brake activated - using minimal stake'
        }
    
    def _cooldown_response(self) -> Dict[str, Any]:
        """Cool-down response"""
        time_remaining = (self.cool_down_until - datetime.now()).seconds
        return {
            'stake': 0,  # No trading during cooldown
            'consecutive_losses': self.consecutive_losses,
            'martingale_level': self.martingale_level,
            'risk_level': 'COOLDOWN',
            'reasoning': f'In cool-down period. {time_remaining} seconds remaining'
        }
    
    def _update_performance_metrics(self, result: str, ai_confidence: float, volatility: float):
        """Update performance tracking metrics"""
        self.performance_metrics['total_trades'] += 1
        if result == 'WON':
            self.performance_metrics['wins'] += 1
        elif result == 'LOST':
            self.performance_metrics['losses'] += 1
            
        self.performance_metrics['ai_confidence_scores'].append(ai_confidence)
        self.performance_metrics['market_volatility_data'].append(volatility)
        
        # Keep only recent data (last 100 trades)
        for key in ['ai_confidence_scores', 'market_volatility_data']:
            if len(self.performance_metrics[key]) > 100:
                self.performance_metrics[key] = self.performance_metrics[key][-100:]
    
    def _update_loss_counter(self, result: str):
        """Update consecutive losses with advanced logic"""
        if result == 'LOST':
            self.consecutive_losses += 1
            # Activate cool-down after significant losses
            if self.consecutive_losses >= 5:
                cooldown_duration = min(300 * (self.consecutive_losses - 4), 1800)  # Max 30 min
                self.cool_down_until = datetime.now() + timedelta(seconds=cooldown_duration)
        elif result == 'WON':
            self.session_profit += self.current_stake
            self.last_win_time = datetime.now()
            if self.config['reset_on_win']:
                self.consecutive_losses = 0
                self.martingale_level = 0
                self.emergency_brake_active = False
                self.cool_down_until = None
    
    def _update_market_regime(self, volatility: float, ai_confidence: float, market_data: Dict[str, Any]):
        """Detect and update current market regime"""
        # Simple regime detection based on volatility and confidence
        if volatility > 1.5 and ai_confidence < 0.5:
            new_regime = 'HIGH_VOLATILITY_LOW_CONFIDENCE'
        elif volatility < 0.7 and ai_confidence > 0.8:
            new_regime = 'LOW_VOLATILITY_HIGH_CONFIDENCE'
        elif volatility > 1.2:
            new_regime = 'HIGH_VOLATILITY'
        elif ai_confidence < 0.4:
            new_regime = 'LOW_CONFIDENCE'
        else:
            new_regime = 'NEUTRAL'
            
        if new_regime != self.market_regime_detector['current_regime']:
            self.market_regime_detector['regime_changes'] += 1
            self.market_regime_detector['regime_history'].append({
                'timestamp': datetime.now(),
                'old_regime': self.market_regime_detector['current_regime'],
                'new_regime': new_regime
            })
            self.market_regime_detector['current_regime'] = new_regime
            
    def _calculate_smart_stake(self, 
                              balance: float, 
                              volatility: float, 
                              ai_confidence: float) -> Dict[str, Any]:
        """Calculate base stake using smart logic"""
        
        reasoning = []
        base_stake = self.config['initial_stake']
        risk_level = 'LOW'
        
        # Martingale progression with smart adjustments
        if self.consecutive_losses > 0:
            if self.consecutive_losses >= self.config['max_consecutive_losses']:
                # After 3 losses, apply multiplier with smart adjustments
                multiplier = self._calculate_dynamic_multiplier(volatility, ai_confidence)
                base_stake = self.current_stake * multiplier
                risk_level = 'HIGH'
                reasoning.append(f"Applied {multiplier}x multiplier after {self.consecutive_losses} losses")
            else:
                # Keep same stake for first 3 losses
                base_stake = self.current_stake
                risk_level = 'MEDIUM'
                reasoning.append(f"Maintaining stake level, loss #{self.consecutive_losses}")
        else:
            # Reset to initial stake after win
            base_stake = self.config['initial_stake']
            risk_level = 'LOW'
            reasoning.append("Reset to initial stake after win")
            
        # AI Confidence adjustment
        confidence_adjusted = False
        if ai_confidence > 0.8:
            base_stake *= 1.2  # Increase stake with high confidence
            confidence_adjusted = True
            reasoning.append("Increased stake due to high AI confidence")
        elif ai_confidence < 0.4:
            base_stake *= 0.8  # Decrease stake with low confidence
            confidence_adjusted = True
            reasoning.append("Decreased stake due to low AI confidence")
            
        # Volatility adjustment
        volatility_adjusted = False
        if volatility > 1.5:
            base_stake *= 0.8  # Reduce stake in high volatility
            volatility_adjusted = True
            risk_level = 'HIGH'
            reasoning.append("Reduced stake due to high market volatility")
        elif volatility < 0.5:
            base_stake *= 1.1  # Slightly increase in low volatility
            volatility_adjusted = True
            reasoning.append("Increased stake due to low market volatility")
            
        return {
            'base_stake': base_stake,
            'risk_level': risk_level,
            'confidence_adjusted': confidence_adjusted,
            'volatility_adjusted': volatility_adjusted,
            'reasoning': '; '.join(reasoning)
        }
        
    def _calculate_dynamic_multiplier(self, volatility: float, ai_confidence: float) -> float:
        """Calculate dynamic multiplier based on market conditions"""
        base_multiplier = self.config['multiplier']
        
        # Adjust multiplier based on AI confidence
        if ai_confidence > 0.7:
            confidence_factor = 1.1  # Slightly more aggressive with high confidence
        elif ai_confidence < 0.4:
            confidence_factor = 0.9  # More conservative with low confidence
        else:
            confidence_factor = 1.0
            
        # Adjust for volatility
        if volatility > 1.5:
            volatility_factor = 0.8  # Less aggressive in high volatility
        elif volatility < 0.7:
            volatility_factor = 1.1  # Slightly more aggressive in low volatility
        else:
            volatility_factor = 1.0
            
        # Adjust for consecutive losses (progressive caution)
        if self.consecutive_losses > 5:
            loss_factor = 0.9  # More conservative after many losses
        else:
            loss_factor = 1.0
            
        dynamic_multiplier = base_multiplier * confidence_factor * volatility_factor * loss_factor
        
        # Ensure multiplier stays within reasonable bounds
        return max(1.5, min(3.0, dynamic_multiplier))
        
    def _apply_unpredictability(self, base_stake: float) -> float:
        """Apply unpredictability factor to make patterns less predictable"""
        
        unpredictability = self.config['unpredictability_factor']
        
        # Generate unpredictable adjustment
        random_factors = [
            random.uniform(0.95, 1.05),  # Small random variation
            self._get_time_based_factor(),  # Time-based variation
            self._get_pattern_based_factor(),  # Pattern avoidance
            self._get_session_based_factor()  # Session performance based
        ]
        
        # Apply random selection of factors
        selected_factors = random.sample(random_factors, k=random.randint(1, 3))
        adjustment = 1.0
        
        for factor in selected_factors:
            adjustment *= factor
            
        # Apply unpredictability within bounds
        min_adjustment = 1.0 - unpredictability
        max_adjustment = 1.0 + unpredictability
        
        final_adjustment = max(min_adjustment, min(max_adjustment, adjustment))
        
        return base_stake * final_adjustment
        
    def _get_time_based_factor(self) -> float:
        """Generate time-based unpredictability factor"""
        now = datetime.now()
        
        # Use minute and second for variability
        minute_factor = (now.minute % 7) / 10.0  # 0 to 0.6
        second_factor = (now.second % 13) / 20.0  # 0 to 0.65
        
        return 0.95 + minute_factor + second_factor
        
    def _get_pattern_based_factor(self) -> float:
        """Avoid predictable patterns"""
        if len(self.pattern_memory) < 3:
            return 1.0
            
        # Check if we're in a predictable pattern
        recent_stakes = self.pattern_memory[-3:]
        
        # If stakes are too similar, add variation
        if max(recent_stakes) - min(recent_stakes) < 0.1:
            return random.uniform(0.9, 1.1)
            
        return 1.0
        
    def _get_session_based_factor(self) -> float:
        """Adjust based on session performance"""
        if self.session_trades == 0:
            return 1.0
            
        # If session is going well, be slightly more aggressive
        if self.session_profit > 0 and self.consecutive_losses < 2:
            return random.uniform(1.0, 1.05)
            
        # If session is going poorly, be more conservative
        if self.session_profit < -10 or self.consecutive_losses > 3:
            return random.uniform(0.95, 1.0)
            
        return 1.0
        
    def _apply_risk_management(self, stake: float, balance: float) -> float:
        """Apply comprehensive risk management"""
        
        # Maximum stake limit
        if stake > self.config['max_stake']:
            logger.warning(f"Stake {stake} exceeds max limit {self.config['max_stake']}")
            stake = self.config['max_stake']
            
        # Balance percentage limit (never risk more than 5% of balance)
        max_balance_risk = balance * 0.05
        if stake > max_balance_risk:
            logger.warning(f"Stake {stake} exceeds balance risk limit {max_balance_risk}")
            stake = max_balance_risk
            
        # Minimum stake
        min_stake = 0.1  # Minimum meaningful stake
        if stake < min_stake:
            stake = min_stake
            
        # Session loss limit
        session_loss_limit = balance * 0.1  # Max 10% loss per session
        if abs(self.session_profit) > session_loss_limit and self.session_profit < 0:
            logger.warning("Session loss limit reached, reducing stake")
            stake = min_stake
            
        # Time-based risk management
        time_since_last_win = datetime.now() - self.last_win_time
        if time_since_last_win > timedelta(hours=1):
            logger.warning("Long time since last win, reducing stake")
            stake *= 0.8
            
        return round(stake, 2)
        
    def _get_martingale_level(self) -> int:
        """Get current martingale level"""
        if self.consecutive_losses <= self.config['max_consecutive_losses']:
            return 0  # Pre-martingale phase
        else:
            return self.consecutive_losses - self.config['max_consecutive_losses']
            
    def _update_pattern_memory(self, stake: float):
        """Update pattern memory for unpredictability"""
        self.pattern_memory.append(stake)
        
        # Keep only recent patterns
        if len(self.pattern_memory) > 10:
            self.pattern_memory = self.pattern_memory[-10:]
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'current_stake': self.current_stake,
            'consecutive_losses': self.consecutive_losses,
            'martingale_level': self._get_martingale_level(),
            'session_trades': self.session_trades,
            'session_profit': self.session_profit,
            'last_win_time': self.last_win_time.isoformat(),
            'pattern_memory_size': len(self.pattern_memory),
            'risk_multiplier': self.risk_adjusted_multiplier
        }
        
    def reset_session(self):
        """Reset session data"""
        self.session_trades = 0
        self.session_profit = 0.0
        self.consecutive_losses = 0
        self.current_stake = self.config['initial_stake']
        self.last_win_time = datetime.now()
        self.pattern_memory.clear()
        logger.info("Martingale system session reset")
        
    def should_stop_trading(self, balance: float, daily_loss_limit: float) -> Dict[str, Any]:
        """Determine if trading should stop based on risk management"""
        reasons = []
        should_stop = False
        
        # Check consecutive losses
        if self.consecutive_losses >= 8:  # Hard limit
            should_stop = True
            reasons.append(f"Too many consecutive losses: {self.consecutive_losses}")
            
        # Check session loss
        if self.session_profit < -daily_loss_limit:
            should_stop = True
            reasons.append(f"Daily loss limit reached: {self.session_profit}")
            
        # Check if balance is too low for meaningful trading
        if balance < self.config['initial_stake'] * 10:
            should_stop = True
            reasons.append(f"Balance too low for safe trading: {balance}")
            
        # Check time since last win
        time_since_win = datetime.now() - self.last_win_time
        if time_since_win > timedelta(hours=2):
            should_stop = True
            reasons.append(f"Too long since last win: {time_since_win}")
            
        return {
            'should_stop': should_stop,
            'reasons': reasons,
            'consecutive_losses': self.consecutive_losses,
            'session_profit': self.session_profit,
            'time_since_last_win': str(time_since_win)
        }
        
    def get_recommended_duration(self, market_volatility: float, ai_confidence: float) -> int:
        """Get recommended trade duration based on conditions"""
        base_duration = 5  # Base 5 ticks
        
        # Adjust for volatility
        if market_volatility > 1.5:
            base_duration = 3  # Shorter duration in high volatility
        elif market_volatility < 0.5:
            base_duration = 7  # Longer duration in low volatility
            
        # Adjust for AI confidence
        if ai_confidence > 0.8:
            base_duration += 2  # Longer duration with high confidence
        elif ai_confidence < 0.4:
            base_duration = max(2, base_duration - 2)  # Shorter with low confidence
            
        # Add some unpredictability
        duration_variation = random.choice([-1, 0, 1])
        final_duration = max(1, base_duration + duration_variation)
        
        return final_duration
    
    def _calculate_advanced_smart_stake(self, 
                                      balance: float, 
                                      volatility: float, 
                                      ai_confidence: float,
                                      market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate base stake using advanced smart logic with AI enhancement"""
        
        reasoning = []
        base_stake = self.config['initial_stake']
        risk_level = 'LOW'
        chaos_applied = False
        
        # Advanced Martingale progression with intelligent scaling
        if self.consecutive_losses > 0:
            if self.consecutive_losses >= self.config['max_consecutive_losses']:
                # Apply smart progressive scaling
                self.martingale_level = self.consecutive_losses - self.config['max_consecutive_losses'] + 1
                
                # Use adaptive multiplier based on performance
                multiplier = self._calculate_adaptive_multiplier(volatility, ai_confidence)
                
                # Apply fibonacci-based progression for unpredictability
                if self.martingale_level <= len(self.fibonacci_sequence):
                    fibonacci_factor = self.fibonacci_sequence[self.martingale_level - 1] / 8.0
                    base_stake = self.current_stake * (multiplier * (1 + fibonacci_factor))
                else:
                    base_stake = self.current_stake * multiplier
                    
                risk_level = 'HIGH'
                reasoning.append(f"Applied martingale level {self.martingale_level} with adaptive multiplier {multiplier:.2f}")
            else:
                # Maintain stake for first few losses with slight variations
                variation_factor = self._get_chaos_variation_factor()
                base_stake = self.current_stake * variation_factor
                risk_level = 'MEDIUM'
                chaos_applied = True
                reasoning.append(f"Maintaining stake level with chaos variation, loss #{self.consecutive_losses}")
        else:
            # Reset to initial stake after win with AI confidence boost
            confidence_boost = 1.0 + (ai_confidence - 0.5) * 0.3  # ±15% based on confidence
            base_stake = self.config['initial_stake'] * confidence_boost
            risk_level = 'LOW'
            reasoning.append("Reset to initial stake with AI confidence adjustment")
            
        # AI Confidence enhancement
        confidence_adjusted = False
        if ai_confidence > self.config['adaptive_confidence_threshold']:
            confidence_multiplier = 1.0 + (ai_confidence - 0.5) * 0.4  # Up to 20% increase
            base_stake *= confidence_multiplier
            confidence_adjusted = True
            reasoning.append(f"Increased stake by {(confidence_multiplier-1)*100:.1f}% due to high AI confidence")
        elif ai_confidence < 0.4:
            confidence_multiplier = 0.7 + ai_confidence * 0.5  # 20-95% of base
            base_stake *= confidence_multiplier
            confidence_adjusted = True
            reasoning.append(f"Reduced stake by {(1-confidence_multiplier)*100:.1f}% due to low AI confidence")
            
        # Volatility adjustment with regime awareness
        volatility_adjusted = False
        regime = self.market_regime_detector['current_regime']
        
        if regime == 'HIGH_VOLATILITY_LOW_CONFIDENCE':
            base_stake *= 0.6  # Very conservative
            volatility_adjusted = True
            reasoning.append("Significantly reduced stake due to high volatility and low confidence")
        elif volatility > 1.5:
            volatility_factor = max(0.7, 1.0 - (volatility - 1.0) * 0.2)
            base_stake *= volatility_factor
            volatility_adjusted = True
            reasoning.append(f"Reduced stake by {(1-volatility_factor)*100:.1f}% due to high volatility")
        elif volatility < 0.5 and ai_confidence > 0.7:
            base_stake *= 1.15  # Slightly more aggressive in stable, confident conditions
            volatility_adjusted = True
            reasoning.append("Increased stake due to low volatility and high confidence")
            
        return {
            'base_stake': base_stake,
            'risk_level': risk_level,
            'confidence_adjusted': confidence_adjusted,
            'volatility_adjusted': volatility_adjusted,
            'chaos_applied': chaos_applied,
            'reasoning': '; '.join(reasoning),
            'martingale_level': self.martingale_level,
            'regime': regime
        }
        
    def _calculate_adaptive_multiplier(self, volatility: float, ai_confidence: float) -> float:
        """Calculate adaptive multiplier based on current conditions and historical performance"""
        base_multiplier = self.config['multiplier']
        
        # Adjust based on AI confidence
        if ai_confidence > 0.8:
            confidence_factor = 1.1  # Slightly more aggressive with high confidence
        elif ai_confidence < 0.3:
            confidence_factor = 0.8  # More conservative with very low confidence
        else:
            confidence_factor = 1.0
            
        # Adjust for volatility
        if volatility > 2.0:
            volatility_factor = 0.7  # Much less aggressive in extreme volatility
        elif volatility > 1.5:
            volatility_factor = 0.85
        elif volatility < 0.6:
            volatility_factor = 1.05  # Slightly more aggressive in low volatility
        else:
            volatility_factor = 1.0
            
        # Adjust based on recent performance
        if len(self.performance_metrics['ai_confidence_scores']) > 10:
            recent_confidence = np.mean(self.performance_metrics['ai_confidence_scores'][-10:])
            if recent_confidence > 0.75:
                performance_factor = 1.05
            elif recent_confidence < 0.4:
                performance_factor = 0.9
            else:
                performance_factor = 1.0
        else:
            performance_factor = 1.0
            
        # Progressive caution for extended losing streaks
        if self.consecutive_losses > 6:
            loss_factor = max(0.7, 1.0 - (self.consecutive_losses - 6) * 0.05)
        else:
            loss_factor = 1.0
            
        # Time-based adjustment (reduce aggressiveness over time)
        session_duration = (datetime.now() - self.session_start_time).total_seconds() / 3600
        if session_duration > 2:  # After 2 hours
            time_factor = max(0.9, 1.0 - (session_duration - 2) * 0.02)
        else:
            time_factor = 1.0
            
        adaptive_multiplier = (base_multiplier * confidence_factor * volatility_factor * 
                             performance_factor * loss_factor * time_factor)
        
        # Ensure multiplier stays within bounds
        self.adaptive_multiplier = max(1.3, min(3.5, adaptive_multiplier))
        return self.adaptive_multiplier
        
    def _get_chaos_variation_factor(self) -> float:
        """Generate chaos-based variation factor for unpredictability"""
        # Use multiple entropy sources
        time_factor = (datetime.now().microsecond % 1000) / 1000.0
        chaos_factor = (self.chaos_seed % 1000) / 1000.0
        random_factor = random.random()
        
        # Combine factors with sine wave for smoothness
        combined_entropy = (time_factor + chaos_factor + random_factor) / 3.0
        variation = 0.95 + 0.1 * np.sin(combined_entropy * 2 * np.pi)  # 95% to 105%
        
        # Update chaos seed for next iteration
        self.chaos_seed = (self.chaos_seed * 1103515245 + 12345) % (2**31)
        
        return variation
    
    def _apply_advanced_unpredictability(self, 
                                        base_stake: float, 
                                        ai_confidence: float,
                                        market_volatility: float) -> float:
        """Apply advanced unpredictability factors to make patterns truly unpredictable"""
        
        unpredictability = self.config['unpredictability_factor']
        
        # Generate multiple unpredictable factors
        random_factors = []
        
        # 1. Multi-dimensional chaos factor
        chaos_dimension_1 = self._get_chaos_variation_factor()
        chaos_dimension_2 = self._get_time_entropy_factor()
        chaos_dimension_3 = self._get_market_noise_factor(market_volatility)
        random_factors.extend([chaos_dimension_1, chaos_dimension_2, chaos_dimension_3])
        
        # 2. Pattern avoidance factor
        pattern_factor = self._get_advanced_pattern_avoidance_factor()
        random_factors.append(pattern_factor)
        
        # 3. Session performance-based variation
        session_factor = self._get_enhanced_session_factor()
        random_factors.append(session_factor)
        
        # 4. AI confidence-based entropy
        confidence_entropy = self._get_confidence_entropy_factor(ai_confidence)
        random_factors.append(confidence_entropy)
        
        # 5. Fibonacci-golden ratio unpredictability
        fibonacci_factor = self._get_fibonacci_unpredictability_factor()
        random_factors.append(fibonacci_factor)
        
        # Randomly select and combine factors for maximum unpredictability
        num_factors = random.randint(2, len(random_factors))
        selected_factors = random.sample(random_factors, k=num_factors)
        
        # Apply weighted combination
        weights = [random.uniform(0.5, 1.5) for _ in selected_factors]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        combined_adjustment = sum(factor * weight for factor, weight in zip(selected_factors, normalized_weights))
        
        # Apply bounded unpredictability
        min_adjustment = 1.0 - unpredictability
        max_adjustment = 1.0 + unpredictability
        
        final_adjustment = max(min_adjustment, min(max_adjustment, combined_adjustment))
        
        # Store adjustment for pattern memory
        self.pattern_memory.append({
            'adjustment': final_adjustment,
            'factors_used': len(selected_factors),
            'timestamp': datetime.now(),
            'ai_confidence': ai_confidence,
            'market_volatility': market_volatility
        })
        
        return base_stake * final_adjustment
    
    def _get_time_entropy_factor(self) -> float:
        """Generate time-based entropy factor"""
        now = datetime.now()
        
        # Use multiple time components for entropy
        microsecond_entropy = (now.microsecond % 100000) / 100000.0
        second_entropy = (now.second % 37) / 37.0  # Prime number for better distribution
        minute_entropy = (now.minute % 41) / 41.0  # Another prime
        
        # Combine with trigonometric functions for smoothness
        time_entropy = (np.sin(microsecond_entropy * 2 * np.pi) * 0.4 +
                       np.cos(second_entropy * 3 * np.pi) * 0.4 +
                       np.sin(minute_entropy * np.pi) * 0.2)
        
        return 1.0 + time_entropy * 0.08  # ±8% variation
    
    def _get_market_noise_factor(self, volatility: float) -> float:
        """Generate market noise-based factor"""
        # Use volatility as entropy source
        volatility_hash = hashlib.md5(str(volatility * 1000000).encode()).hexdigest()
        volatility_entropy = int(volatility_hash[:6], 16) / 16777215.0  # Normalize to 0-1
        
        # Apply non-linear transformation
        noise_factor = 1.0 + (volatility_entropy - 0.5) * 0.12  # ±6% variation
        
        return noise_factor
    
    def _get_advanced_pattern_avoidance_factor(self) -> float:
        """Advanced pattern avoidance using statistical analysis"""
        if len(self.pattern_memory) < 3:
            return random.uniform(0.97, 1.03)
            
        # Analyze recent patterns
        recent_adjustments = [p['adjustment'] for p in self.pattern_memory[-5:]]
        
        # Calculate statistical measures
        mean_adjustment = np.mean(recent_adjustments)
        std_adjustment = np.std(recent_adjustments)
        
        # If patterns are too similar, force variation
        if std_adjustment < 0.02:  # Very low variation
            return random.uniform(0.92, 1.08)  # Force higher variation
        
        # If trending in one direction, reverse
        if len(recent_adjustments) >= 3:
            trend = np.polyfit(range(len(recent_adjustments)), recent_adjustments, 1)[0]
            if abs(trend) > 0.01:  # Significant trend detected
                return mean_adjustment - trend * 2  # Counter-trend
                
        return random.uniform(0.96, 1.04)
    
    def _get_enhanced_session_factor(self) -> float:
        """Enhanced session performance-based factor"""
        if self.session_trades == 0:
            return 1.0
            
        # Calculate session win rate
        session_win_rate = self.performance_metrics['wins'] / max(1, self.performance_metrics['total_trades'])
        
        # Time since session start
        session_hours = (datetime.now() - self.session_start_time).total_seconds() / 3600
        
        # Performance-based adjustment
        if session_win_rate > 0.6 and self.session_profit > 0:
            performance_factor = random.uniform(1.02, 1.06)  # Slightly more aggressive
        elif session_win_rate < 0.4 or self.session_profit < -10:
            performance_factor = random.uniform(0.94, 0.98)  # More conservative
        else:
            performance_factor = random.uniform(0.98, 1.02)
            
        # Time fatigue factor
        if session_hours > 3:
            fatigue_factor = max(0.95, 1.0 - (session_hours - 3) * 0.01)
            performance_factor *= fatigue_factor
            
        return performance_factor
    
    def _get_confidence_entropy_factor(self, ai_confidence: float) -> float:
        """Generate AI confidence-based entropy"""
        # Use confidence value as seed for entropy
        confidence_seed = int(ai_confidence * 1000000) % 997  # Prime modulo
        random.seed(confidence_seed)
        
        # Generate confidence-dependent variation
        if ai_confidence > 0.8:
            # High confidence - allow slightly more variation
            entropy_factor = random.uniform(0.98, 1.04)
        elif ai_confidence < 0.4:
            # Low confidence - more conservative variation
            entropy_factor = random.uniform(0.96, 1.02)
        else:
            # Medium confidence - moderate variation
            entropy_factor = random.uniform(0.97, 1.03)
            
        # Reset random seed
        random.seed()
        
        return entropy_factor
    
    def _get_fibonacci_unpredictability_factor(self) -> float:
        """Use Fibonacci and golden ratio for natural unpredictability"""
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        # Use current trade count in fibonacci sequence
        fib_index = self.session_trades % len(self.fibonacci_sequence)
        fib_value = self.fibonacci_sequence[fib_index]
        
        # Apply golden ratio transformation
        phi_factor = (fib_value / golden_ratio) % 1.0
        
        # Convert to variation factor
        variation = 0.96 + 0.08 * phi_factor  # 96% to 104%
        
        return variation
