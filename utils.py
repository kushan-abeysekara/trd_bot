"""
Utility functions for the trading bot
"""
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import asyncio
import aiofiles
import config

logger = logging.getLogger(__name__)

class DataProcessor:
    """Utility class for data processing and formatting"""
    
    @staticmethod
    def convert_candle_data_to_df(candle_data: List[Dict]) -> pd.DataFrame:
        """Convert candle data from API to pandas DataFrame"""
        try:
            if not candle_data:
                return pd.DataFrame()
                
            df = pd.DataFrame(candle_data)
            
            # Convert string values to numeric
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            # Convert epoch to datetime
            if 'epoch' in df.columns:
                df['datetime'] = pd.to_datetime(df['epoch'], unit='s')
                df.set_index('datetime', inplace=True)
                
            return df
            
        except Exception as e:
            logger.error(f"Error converting candle data to DataFrame: {e}")
            return pd.DataFrame()
            
    @staticmethod
    def calculate_returns(prices: Union[List[float], pd.Series]) -> List[float]:
        """Calculate returns from price series"""
        try:
            if isinstance(prices, pd.Series):
                prices = prices.tolist()
                
            if len(prices) < 2:
                return []
                
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] != 0:
                    return_pct = (prices[i] - prices[i-1]) / prices[i-1]
                    returns.append(return_pct)
                else:
                    returns.append(0.0)
                    
            return returns
            
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return []
            
    @staticmethod
    def calculate_volatility(returns: List[float], periods: int = 252) -> float:
        """Calculate annualized volatility from returns"""
        try:
            if len(returns) < 2:
                return 0.0
                
            return np.std(returns) * np.sqrt(periods)
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0
            
    @staticmethod
    def detect_outliers(data: List[float], threshold: float = 2.0) -> List[int]:
        """Detect outliers using z-score method"""
        try:
            if len(data) < 3:
                return []
                
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            if std_val == 0:
                return []
                
            outliers = []
            for i, value in enumerate(data):
                z_score = abs((value - mean_val) / std_val)
                if z_score > threshold:
                    outliers.append(i)
                    
            return outliers
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
            return []

class Logger:
    """Enhanced logging utility"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        
    async def log_trade(self, trade_data: Dict[str, Any]):
        """Log trade information"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'TRADE',
                'data': trade_data
            }
            
            await self._write_to_file('trades.log', log_entry)
            
        except Exception as e:
            self.logger.error(f"Error logging trade: {e}")
            
    async def log_analysis(self, analysis_data: Dict[str, Any]):
        """Log analysis information"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'ANALYSIS',
                'data': analysis_data
            }
            
            await self._write_to_file('analysis.log', log_entry)
            
        except Exception as e:
            self.logger.error(f"Error logging analysis: {e}")
            
    async def log_error(self, error_data: Dict[str, Any]):
        """Log error information"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'ERROR',
                'data': error_data
            }
            
            await self._write_to_file('errors.log', log_entry)
            
        except Exception as e:
            self.logger.error(f"Error logging error: {e}")
            
    async def _write_to_file(self, filename: str, data: Dict[str, Any]):
        """Write log entry to file"""
        try:
            async with aiofiles.open(f'logs/{filename}', 'a') as f:
                await f.write(json.dumps(data) + '\n')
                
        except Exception as e:
            self.logger.error(f"Error writing to log file: {e}")

class PerformanceCalculator:
    """Calculate trading performance metrics"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns) < 2:
                return 0.0
                
            excess_returns = [r - risk_free_rate/252 for r in returns]  # Daily risk-free rate
            
            if np.std(excess_returns) == 0:
                return 0.0
                
            return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
            
    @staticmethod
    def calculate_max_drawdown(equity_curve: List[float]) -> Dict[str, float]:
        """Calculate maximum drawdown"""
        try:
            if len(equity_curve) < 2:
                return {'max_drawdown': 0.0, 'drawdown_duration': 0}
                
            peak = equity_curve[0]
            max_drawdown = 0.0
            current_drawdown = 0.0
            drawdown_start = 0
            max_drawdown_duration = 0
            current_duration = 0
            
            for i, value in enumerate(equity_curve):
                if value > peak:
                    peak = value
                    if current_drawdown > 0:
                        max_drawdown_duration = max(max_drawdown_duration, current_duration)
                        current_drawdown = 0
                        current_duration = 0
                else:
                    current_drawdown = (peak - value) / peak
                    current_duration = i - drawdown_start
                    
                    if current_drawdown > max_drawdown:
                        max_drawdown = current_drawdown
                        
            return {
                'max_drawdown': max_drawdown,
                'drawdown_duration': max_drawdown_duration
            }
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return {'max_drawdown': 0.0, 'drawdown_duration': 0}
            
    @staticmethod
    def calculate_win_rate(trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate win rate and related metrics"""
        try:
            if not trades:
                return {
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'largest_win': 0.0,
                    'largest_loss': 0.0
                }
                
            winning_trades = [t for t in trades if t.get('profit_loss', 0) > 0]
            losing_trades = [t for t in trades if t.get('profit_loss', 0) < 0]
            
            win_rate = len(winning_trades) / len(trades) * 100
            
            total_wins = sum(t.get('profit_loss', 0) for t in winning_trades)
            total_losses = abs(sum(t.get('profit_loss', 0) for t in losing_trades))
            
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            avg_win = total_wins / len(winning_trades) if winning_trades else 0
            avg_loss = total_losses / len(losing_trades) if losing_trades else 0
            
            largest_win = max((t.get('profit_loss', 0) for t in winning_trades), default=0)
            largest_loss = min((t.get('profit_loss', 0) for t in losing_trades), default=0)
            
            return {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'largest_win': largest_win,
                'largest_loss': largest_loss
            }
            
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }

class MarketDataValidator:
    """Validate market data quality"""
    
    @staticmethod
    def validate_candle_data(candles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate candle data quality"""
        try:
            if not candles:
                return {'valid': False, 'errors': ['No candle data provided']}
                
            errors = []
            warnings = []
            
            for i, candle in enumerate(candles):
                # Check required fields
                required_fields = ['open', 'high', 'low', 'close', 'epoch']
                for field in required_fields:
                    if field not in candle:
                        errors.append(f"Missing field '{field}' in candle {i}")
                        
                # Validate OHLC logic
                try:
                    o, h, l, c = float(candle.get('open', 0)), float(candle.get('high', 0)), \
                                float(candle.get('low', 0)), float(candle.get('close', 0))
                    
                    if h < max(o, c) or l > min(o, c):
                        warnings.append(f"Invalid OHLC relationship in candle {i}")
                        
                    if h < l:
                        errors.append(f"High < Low in candle {i}")
                        
                except (ValueError, TypeError):
                    errors.append(f"Invalid numeric values in candle {i}")
                    
            # Check for gaps in time series
            if len(candles) > 1:
                epochs = [int(candle.get('epoch', 0)) for candle in candles]
                epochs.sort()
                
                expected_interval = epochs[1] - epochs[0] if len(epochs) > 1 else 60
                
                for i in range(1, len(epochs)):
                    actual_interval = epochs[i] - epochs[i-1]
                    if abs(actual_interval - expected_interval) > expected_interval * 0.1:
                        warnings.append(f"Time gap detected between candles {i-1} and {i}")
                        
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'candle_count': len(candles)
            }
            
        except Exception as e:
            logger.error(f"Error validating candle data: {e}")
            return {'valid': False, 'errors': [f'Validation error: {str(e)}']}
            
    @staticmethod
    def detect_data_anomalies(prices: List[float], threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Detect price anomalies"""
        try:
            if len(prices) < 3:
                return []
                
            anomalies = []
            
            for i in range(1, len(prices) - 1):
                prev_price = prices[i-1]
                current_price = prices[i]
                next_price = prices[i+1]
                
                if prev_price == 0 or next_price == 0:
                    continue
                    
                # Check for price spikes
                change_1 = abs(current_price - prev_price) / prev_price
                change_2 = abs(next_price - current_price) / current_price
                
                if change_1 > threshold and change_2 > threshold:
                    anomalies.append({
                        'index': i,
                        'type': 'spike',
                        'severity': max(change_1, change_2),
                        'description': f'Price spike detected at index {i}'
                    })
                    
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting data anomalies: {e}")
            return []

class ConfigManager:
    """Configuration management utility"""
    
    @staticmethod
    def update_config(updates: Dict[str, Any]) -> bool:
        """Update configuration dynamically"""
        try:
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    logger.info(f"Updated config: {key} = {value}")
                else:
                    logger.warning(f"Unknown config key: {key}")
                    
            return True
            
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            return False
            
    @staticmethod
    def get_config_summary() -> Dict[str, Any]:
        """Get current configuration summary"""
        try:
            return {
                'trading': {
                    'demo_mode': config.DEMO_MODE,
                    'initial_stake': config.INITIAL_STAKE,
                    'max_daily_loss': config.MAX_DAILY_LOSS,
                    'max_consecutive_losses': config.MAX_CONSECUTIVE_LOSSES
                },
                'martingale': config.MARTINGALE_CONFIG,
                'ai': config.AI_CONFIG,
                'technical': config.TECHNICAL_CONFIG,
                'risk': config.RISK_CONFIG
            }
            
        except Exception as e:
            logger.error(f"Error getting config summary: {e}")
            return {}

class AsyncTaskManager:
    """Manage asynchronous tasks"""
    
    def __init__(self):
        self.tasks = {}
        
    async def create_task(self, name: str, coro, **kwargs):
        """Create and track async task"""
        try:
            task = asyncio.create_task(coro, **kwargs)
            self.tasks[name] = {
                'task': task,
                'created_at': datetime.now(),
                'status': 'running'
            }
            
            logger.info(f"Created async task: {name}")
            return task
            
        except Exception as e:
            logger.error(f"Error creating task {name}: {e}")
            return None
            
    async def cancel_task(self, name: str):
        """Cancel a running task"""
        try:
            if name in self.tasks:
                task_info = self.tasks[name]
                task_info['task'].cancel()
                task_info['status'] = 'cancelled'
                logger.info(f"Cancelled task: {name}")
                
        except Exception as e:
            logger.error(f"Error cancelling task {name}: {e}")
            
    def get_task_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        try:
            if name in self.tasks:
                task_info = self.tasks[name]
                task = task_info['task']
                
                if task.done():
                    if task.cancelled():
                        status = 'cancelled'
                    elif task.exception():
                        status = 'failed'
                    else:
                        status = 'completed'
                else:
                    status = 'running'
                    
                return {
                    'name': name,
                    'status': status,
                    'created_at': task_info['created_at'].isoformat(),
                    'exception': str(task.exception()) if task.done() and task.exception() else None
                }
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting task status {name}: {e}")
            return None
            
    def cleanup_completed_tasks(self):
        """Remove completed tasks from tracking"""
        try:
            completed_tasks = []
            
            for name, task_info in self.tasks.items():
                if task_info['task'].done():
                    completed_tasks.append(name)
                    
            for name in completed_tasks:
                del self.tasks[name]
                
            if completed_tasks:
                logger.info(f"Cleaned up {len(completed_tasks)} completed tasks")
                
        except Exception as e:
            logger.error(f"Error cleaning up tasks: {e}")

# Utility functions
def format_currency(amount: float, currency: str = 'USD') -> str:
    """Format currency amount"""
    try:
        if currency == 'USD':
            return f"${amount:.2f}"
        else:
            return f"{amount:.2f} {currency}"
    except:
        return str(amount)

def format_percentage(value: float) -> str:
    """Format percentage value"""
    try:
        return f"{value:.2f}%"
    except:
        return str(value)

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value"""
    try:
        return numerator / denominator if denominator != 0 else default
    except:
        return default

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))

def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """Calculate exponential backoff delay"""
    delay = base_delay * (2 ** attempt)
    return min(delay, max_delay)
