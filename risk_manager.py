"""
Risk Management Module for Advanced Trading Bot
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import config

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self):
        self.config = config.RISK_CONFIG
        self.daily_loss_tracker = 0.0
        self.session_start_balance = 0.0
        self.max_drawdown_today = 0.0
        self.consecutive_losses = 0
        self.last_reset_date = datetime.now().date()
        
    def evaluate_trade_risk(self, 
                           proposed_trade: Dict[str, Any], 
                           account_balance: float,
                           current_positions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate the risk of a proposed trade"""
        
        try:
            risk_assessment = {
                'approved': True,
                'risk_score': 0.0,
                'warnings': [],
                'recommended_adjustments': {},
                'max_allowed_stake': 0.0
            }
            
            current_positions = current_positions or []
            
            # Check daily loss limit
            if self._check_daily_loss_limit():
                risk_assessment['approved'] = False
                risk_assessment['warnings'].append("Daily loss limit exceeded")
                return risk_assessment
                
            # Check position size limits
            stake = proposed_trade.get('stake', 0)
            max_position_size = account_balance * (self.config['max_position_size'] / 100)
            
            if stake > max_position_size:
                risk_assessment['warnings'].append(f"Stake exceeds position size limit: ${max_position_size:.2f}")
                risk_assessment['recommended_adjustments']['stake'] = max_position_size
                risk_assessment['risk_score'] += 0.3
                
            # Check maximum open positions
            if len(current_positions) >= self.config['max_open_positions']:
                risk_assessment['approved'] = False
                risk_assessment['warnings'].append("Maximum open positions reached")
                return risk_assessment
                
            # Check correlation risk (if multiple positions on same symbol)
            symbol = proposed_trade.get('symbol', '')
            same_symbol_positions = [p for p in current_positions if p.get('symbol') == symbol]
            
            if len(same_symbol_positions) >= 2:
                risk_assessment['warnings'].append("High correlation risk: multiple positions on same symbol")
                risk_assessment['risk_score'] += 0.2
                
            # Check consecutive losses
            if self.consecutive_losses >= 5:
                risk_assessment['warnings'].append("High consecutive losses detected")
                risk_assessment['recommended_adjustments']['stake'] = stake * 0.5
                risk_assessment['risk_score'] += 0.4
                
            # Check volatility risk
            market_volatility = proposed_trade.get('market_volatility', 1.0)
            if market_volatility > 2.0:
                risk_assessment['warnings'].append("High market volatility detected")
                risk_assessment['recommended_adjustments']['duration'] = max(3, proposed_trade.get('duration', 5) - 2)
                risk_assessment['risk_score'] += 0.2
                
            # Check AI confidence
            ai_confidence = proposed_trade.get('ai_confidence', 0.5)
            if ai_confidence < 0.6:
                risk_assessment['warnings'].append("Low AI confidence")
                risk_assessment['recommended_adjustments']['stake'] = stake * 0.7
                risk_assessment['risk_score'] += 0.3
                
            # Calculate maximum allowed stake
            risk_assessment['max_allowed_stake'] = min(
                max_position_size,
                account_balance * 0.02,  # Never risk more than 2% per trade
                stake * (1.0 - risk_assessment['risk_score'])
            )
            
            # Final approval decision
            if risk_assessment['risk_score'] > 0.8:
                risk_assessment['approved'] = False
                risk_assessment['warnings'].append("Risk score too high")
                
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Error evaluating trade risk: {e}")
            return {
                'approved': False,
                'risk_score': 1.0,
                'warnings': ['Risk evaluation error'],
                'recommended_adjustments': {},
                'max_allowed_stake': 0.0
            }
            
    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been exceeded"""
        # Reset daily tracker if new day
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_loss_tracker = 0.0
            self.max_drawdown_today = 0.0
            self.last_reset_date = current_date
            
        return abs(self.daily_loss_tracker) >= self.config.get('daily_loss_limit', config.MAX_DAILY_LOSS)
        
    def update_trade_result(self, trade_result: Dict[str, Any]):
        """Update risk metrics based on trade result"""
        try:
            profit_loss = trade_result.get('profit_loss', 0)
            
            # Update daily loss tracker
            self.daily_loss_tracker += profit_loss
            
            # Update max drawdown
            if profit_loss < 0:
                current_drawdown = abs(self.daily_loss_tracker)
                self.max_drawdown_today = max(self.max_drawdown_today, current_drawdown)
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
                
            logger.info(f"Risk metrics updated - Daily P&L: ${self.daily_loss_tracker:.2f}, "
                       f"Max Drawdown: ${self.max_drawdown_today:.2f}, "
                       f"Consecutive Losses: {self.consecutive_losses}")
                       
        except Exception as e:
            logger.error(f"Error updating trade result: {e}")
            
    def get_portfolio_risk_metrics(self, 
                                  account_balance: float,
                                  active_positions: List[Dict[str, Any]],
                                  recent_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive portfolio risk metrics"""
        
        try:
            # Calculate current exposure
            total_exposure = sum(pos.get('stake', 0) for pos in active_positions)
            exposure_percentage = (total_exposure / account_balance * 100) if account_balance > 0 else 0
            
            # Calculate recent performance metrics
            recent_30_trades = recent_trades[-30:] if len(recent_trades) > 30 else recent_trades
            
            if recent_30_trades:
                recent_profit = sum(t.get('profit_loss', 0) for t in recent_30_trades)
                win_rate = len([t for t in recent_30_trades if t.get('profit_loss', 0) > 0]) / len(recent_30_trades) * 100
                avg_profit_per_trade = recent_profit / len(recent_30_trades)
            else:
                recent_profit = 0
                win_rate = 0
                avg_profit_per_trade = 0
                
            # Calculate volatility of returns
            if len(recent_30_trades) > 5:
                returns = [t.get('profit_loss', 0) for t in recent_30_trades]
                mean_return = sum(returns) / len(returns)
                variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
                volatility = variance ** 0.5
            else:
                volatility = 0
                
            # Risk score calculation
            risk_factors = []
            
            if exposure_percentage > 20:
                risk_factors.append("High portfolio exposure")
                
            if self.consecutive_losses > 3:
                risk_factors.append("High consecutive losses")
                
            if win_rate < 40 and len(recent_30_trades) > 10:
                risk_factors.append("Low win rate")
                
            if self.max_drawdown_today > account_balance * 0.1:
                risk_factors.append("High daily drawdown")
                
            if volatility > account_balance * 0.05:
                risk_factors.append("High return volatility")
                
            # Overall risk level
            if len(risk_factors) >= 3:
                overall_risk = "HIGH"
            elif len(risk_factors) >= 1:
                overall_risk = "MEDIUM"
            else:
                overall_risk = "LOW"
                
            return {
                'overall_risk_level': overall_risk,
                'risk_factors': risk_factors,
                'portfolio_exposure': exposure_percentage,
                'daily_pnl': self.daily_loss_tracker,
                'max_drawdown_today': self.max_drawdown_today,
                'consecutive_losses': self.consecutive_losses,
                'recent_win_rate': win_rate,
                'recent_avg_profit': avg_profit_per_trade,
                'return_volatility': volatility,
                'total_active_positions': len(active_positions),
                'recommendation': self._get_risk_recommendation(overall_risk, risk_factors)
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk metrics: {e}")
            return {
                'overall_risk_level': 'HIGH',
                'risk_factors': ['Calculation error'],
                'recommendation': 'Stop trading and review system'
            }
            
    def _get_risk_recommendation(self, risk_level: str, risk_factors: List[str]) -> str:
        """Get recommendation based on risk assessment"""
        
        if risk_level == "HIGH":
            return "STOP TRADING - Multiple high-risk factors detected. Review strategy and wait for better conditions."
            
        elif risk_level == "MEDIUM":
            recommendations = []
            
            if "High portfolio exposure" in risk_factors:
                recommendations.append("Reduce position sizes")
                
            if "High consecutive losses" in risk_factors:
                recommendations.append("Take a break and review strategy")
                
            if "Low win rate" in risk_factors:
                recommendations.append("Reassess entry criteria")
                
            if "High daily drawdown" in risk_factors:
                recommendations.append("Implement stricter stop losses")
                
            if "High return volatility" in risk_factors:
                recommendations.append("Focus on more stable instruments")
                
            return "REDUCE RISK - " + "; ".join(recommendations)
            
        else:
            return "CONTINUE TRADING - Risk levels are acceptable"
            
    def should_stop_trading(self, account_balance: float) -> Dict[str, Any]:
        """Determine if trading should be stopped based on risk management"""
        
        stop_reasons = []
        should_stop = False
        
        # Daily loss limit
        if self._check_daily_loss_limit():
            stop_reasons.append("Daily loss limit exceeded")
            should_stop = True
            
        # Account balance too low
        min_balance = config.INITIAL_STAKE * 20
        if account_balance < min_balance:
            stop_reasons.append(f"Account balance below minimum: ${min_balance}")
            should_stop = True
            
        # Too many consecutive losses
        if self.consecutive_losses >= 8:
            stop_reasons.append("Too many consecutive losses")
            should_stop = True
            
        # Emergency stop loss
        emergency_loss = self.config.get('emergency_stop_loss', 200.0)
        if abs(self.daily_loss_tracker) >= emergency_loss:
            stop_reasons.append("Emergency stop loss triggered")
            should_stop = True
            
        # Maximum drawdown
        if self.max_drawdown_today >= account_balance * 0.2:
            stop_reasons.append("Maximum drawdown exceeded")
            should_stop = True
            
        return {
            'should_stop': should_stop,
            'reasons': stop_reasons,
            'daily_loss': self.daily_loss_tracker,
            'max_drawdown': self.max_drawdown_today,
            'consecutive_losses': self.consecutive_losses
        }
        
    def get_position_sizing_recommendation(self, 
                                         account_balance: float,
                                         trade_confidence: float,
                                         market_volatility: float) -> Dict[str, Any]:
        """Get position sizing recommendation based on Kelly Criterion and risk factors"""
        
        try:
            # Base position size (1-2% of account)
            base_position_pct = 0.015  # 1.5%
            
            # Adjust for confidence
            confidence_multiplier = max(0.5, min(1.5, trade_confidence * 2))
            
            # Adjust for volatility
            volatility_multiplier = max(0.5, min(1.2, 1 / market_volatility))
            
            # Adjust for consecutive losses
            loss_multiplier = max(0.3, 1.0 - (self.consecutive_losses * 0.1))
            
            # Calculate final position size
            adjusted_position_pct = base_position_pct * confidence_multiplier * volatility_multiplier * loss_multiplier
            
            # Apply hard limits
            max_position_pct = self.config['max_position_size'] / 100
            final_position_pct = min(adjusted_position_pct, max_position_pct)
            
            recommended_stake = account_balance * final_position_pct
            
            return {
                'recommended_stake': recommended_stake,
                'position_percentage': final_position_pct * 100,
                'confidence_adjustment': confidence_multiplier,
                'volatility_adjustment': volatility_multiplier,
                'loss_adjustment': loss_multiplier,
                'reasoning': f"Base: {base_position_pct*100:.1f}%, "
                           f"Confidence: {confidence_multiplier:.2f}x, "
                           f"Volatility: {volatility_multiplier:.2f}x, "
                           f"Loss: {loss_multiplier:.2f}x"
            }
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {
                'recommended_stake': account_balance * 0.01,  # Conservative 1%
                'position_percentage': 1.0,
                'reasoning': 'Error in calculation, using conservative 1%'
            }
            
    def reset_daily_metrics(self):
        """Reset daily risk metrics (called at start of new day)"""
        self.daily_loss_tracker = 0.0
        self.max_drawdown_today = 0.0
        self.last_reset_date = datetime.now().date()
        logger.info("Daily risk metrics reset")
        
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get summary of current risk status"""
        return {
            'daily_pnl': self.daily_loss_tracker,
            'max_drawdown_today': self.max_drawdown_today,
            'consecutive_losses': self.consecutive_losses,
            'last_reset_date': self.last_reset_date.isoformat(),
            'daily_loss_limit': config.MAX_DAILY_LOSS,
            'daily_limit_remaining': config.MAX_DAILY_LOSS - abs(self.daily_loss_tracker)
        }
