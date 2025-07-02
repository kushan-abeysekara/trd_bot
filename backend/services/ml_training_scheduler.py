import schedule
import time
import logging
from datetime import datetime, timedelta
from threading import Thread
import sqlite3
import json
from services.ml_strategies import MLStrategyManager

logger = logging.getLogger(__name__)

class MLTrainingScheduler:
    def __init__(self):
        self.ml_manager = MLStrategyManager()
        self.is_running = False
        self.scheduler_thread = None
        
    def start_scheduler(self):
        """Start the ML training scheduler"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Schedule training tasks
        schedule.every(6).hours.do(self._incremental_training)
        schedule.every().day.at("02:00").do(self._full_model_retraining)
        schedule.every().week.do(self._model_performance_analysis)
        schedule.every().hour.do(self._check_data_quality)
        
        # Start scheduler thread
        self.scheduler_thread = Thread(target=self._run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger.info("ML Training Scheduler started")
    
    def stop_scheduler(self):
        """Stop the ML training scheduler"""
        self.is_running = False
        schedule.clear()
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=30)
        
        logger.info("ML Training Scheduler stopped")
    
    def _run_scheduler(self):
        """Run the scheduler loop"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _incremental_training(self):
        """Perform incremental training with new data"""
        try:
            logger.info("Starting incremental ML training...")
            
            # Get new training data since last training
            new_data_count = self._get_new_training_data_count()
            
            if new_data_count < 50:
                logger.info(f"Insufficient new data for training: {new_data_count} samples")
                return
            
            # Retrain models with new data
            self.ml_manager.retrain_models()
            
            # Log training completion
            self._log_training_event("incremental", new_data_count)
            
            logger.info(f"Incremental training completed with {new_data_count} new samples")
            
        except Exception as e:
            logger.error(f"Incremental training failed: {str(e)}")
    
    def _full_model_retraining(self):
        """Perform full model retraining"""
        try:
            logger.info("Starting full ML model retraining...")
            
            # Get total training data
            total_data_count = self._get_total_training_data_count()
            
            if total_data_count < 100:
                logger.info(f"Insufficient total data for full retraining: {total_data_count} samples")
                return
            
            # Full model retraining
            self.ml_manager.retrain_models()
            
            # Clean old training data (keep last 6 months)
            self._cleanup_old_training_data()
            
            # Log training completion
            self._log_training_event("full", total_data_count)
            
            logger.info(f"Full retraining completed with {total_data_count} total samples")
            
        except Exception as e:
            logger.error(f"Full retraining failed: {str(e)}")
    
    def _model_performance_analysis(self):
        """Analyze model performance and optimize"""
        try:
            logger.info("Starting model performance analysis...")
            
            # Get model performance metrics
            performance = self.ml_manager.get_model_performance()
            
            # Analyze and log performance
            for contract_type, metrics in performance.items():
                accuracy = metrics.get('accuracy', 0)
                
                if accuracy < 0.6:  # Less than 60% accuracy
                    logger.warning(f"Low accuracy for {contract_type}: {accuracy:.2f}")
                    # Trigger additional training for this contract type
                    self._trigger_contract_specific_training(contract_type)
                else:
                    logger.info(f"Good accuracy for {contract_type}: {accuracy:.2f}")
            
            # Log performance analysis
            self._log_performance_analysis(performance)
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {str(e)}")
    
    def _check_data_quality(self):
        """Check training data quality and consistency"""
        try:
            # Get recent training data statistics
            stats = self._get_data_quality_stats()
            
            # Check for data quality issues
            issues = []
            
            if stats['win_rate'] > 0.95 or stats['win_rate'] < 0.05:
                issues.append(f"Suspicious win rate: {stats['win_rate']:.2%}")
            
            if stats['recent_trades'] < 10:
                issues.append(f"Low recent trade volume: {stats['recent_trades']}")
            
            if stats['data_variance'] < 0.001:
                issues.append(f"Low data variance: {stats['data_variance']}")
            
            if issues:
                logger.warning(f"Data quality issues detected: {', '.join(issues)}")
                self._log_data_quality_issues(issues)
            else:
                logger.info("Data quality check passed")
            
        except Exception as e:
            logger.error(f"Data quality check failed: {str(e)}")
    
    def _get_new_training_data_count(self) -> int:
        """Get count of new training data since last training"""
        try:
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            
            # Get data from last 6 hours
            cursor.execute('''
                SELECT COUNT(*) FROM ml_training_data 
                WHERE timestamp > datetime('now', '-6 hours')
            ''')
            
            count = cursor.fetchone()[0]
            conn.close()
            
            return count
            
        except Exception as e:
            logger.error(f"Error getting new training data count: {str(e)}")
            return 0
    
    def _get_total_training_data_count(self) -> int:
        """Get total training data count"""
        try:
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM ml_training_data')
            count = cursor.fetchone()[0]
            conn.close()
            
            return count
            
        except Exception as e:
            logger.error(f"Error getting total training data count: {str(e)}")
            return 0
    
    def _cleanup_old_training_data(self):
        """Clean up old training data (keep last 6 months)"""
        try:
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            
            # Delete data older than 6 months
            cursor.execute('''
                DELETE FROM ml_training_data 
                WHERE timestamp < datetime('now', '-6 months')
            ''')
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old training records")
            
        except Exception as e:
            logger.error(f"Error cleaning up old training data: {str(e)}")
    
    def _trigger_contract_specific_training(self, contract_type: str):
        """Trigger additional training for specific contract type"""
        try:
            # This could be implemented to focus training on specific contract types
            logger.info(f"Triggering additional training for {contract_type}")
            
            # For now, just trigger general retraining
            # In future, implement contract-specific training
            
        except Exception as e:
            logger.error(f"Error in contract-specific training: {str(e)}")
    
    def _get_data_quality_stats(self) -> dict:
        """Get data quality statistics"""
        try:
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            
            # Get recent statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_trades,
                    AVG(CAST(outcome AS FLOAT)) as win_rate,
                    COUNT(CASE WHEN timestamp > datetime('now', '-1 day') THEN 1 END) as recent_trades,
                    AVG(profit_loss) as avg_profit,
                    STDEV(profit_loss) as profit_variance
                FROM ml_training_data
                WHERE timestamp > datetime('now', '-7 days')
            ''')
            
            row = cursor.fetchone()
            conn.close()
            
            return {
                'total_trades': row[0] or 0,
                'win_rate': row[1] or 0.5,
                'recent_trades': row[2] or 0,
                'avg_profit': row[3] or 0.0,
                'data_variance': row[4] or 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting data quality stats: {str(e)}")
            return {
                'total_trades': 0,
                'win_rate': 0.5,
                'recent_trades': 0,
                'avg_profit': 0.0,
                'data_variance': 0.0
            }
    
    def _log_training_event(self, training_type: str, data_count: int):
        """Log training event to database"""
        try:
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            
            # Create training log table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_training_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    training_type TEXT,
                    data_count INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT,
                    notes TEXT
                )
            ''')
            
            cursor.execute('''
                INSERT INTO ml_training_log (training_type, data_count, status)
                VALUES (?, ?, 'completed')
            ''', (training_type, data_count))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging training event: {str(e)}")
    
    def _log_performance_analysis(self, performance: dict):
        """Log performance analysis results"""
        try:
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            
            # Create performance log table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_performance_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    performance_data TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    average_accuracy REAL,
                    best_model TEXT,
                    worst_model TEXT
                )
            ''')
            
            # Calculate summary metrics
            accuracies = [metrics.get('accuracy', 0) for metrics in performance.values()]
            avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
            
            best_model = max(performance.items(), key=lambda x: x[1].get('accuracy', 0))[0] if performance else 'N/A'
            worst_model = min(performance.items(), key=lambda x: x[1].get('accuracy', 0))[0] if performance else 'N/A'
            
            cursor.execute('''
                INSERT INTO ml_performance_log 
                (performance_data, average_accuracy, best_model, worst_model)
                VALUES (?, ?, ?, ?)
            ''', (json.dumps(performance), avg_accuracy, best_model, worst_model))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging performance analysis: {str(e)}")
    
    def _log_data_quality_issues(self, issues: list):
        """Log data quality issues"""
        try:
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            
            # Create data quality log table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_quality_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    issues TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    severity TEXT
                )
            ''')
            
            cursor.execute('''
                INSERT INTO data_quality_log (issues, severity)
                VALUES (?, 'warning')
            ''', (json.dumps(issues),))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging data quality issues: {str(e)}")
    
    def get_training_status(self) -> dict:
        """Get current training status and recent activity"""
        try:
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            
            # Get recent training activity
            cursor.execute('''
                SELECT training_type, data_count, timestamp, status
                FROM ml_training_log
                ORDER BY timestamp DESC
                LIMIT 10
            ''')
            
            recent_training = [
                {
                    'type': row[0],
                    'data_count': row[1],
                    'timestamp': row[2],
                    'status': row[3]
                }
                for row in cursor.fetchall()
            ]
            
            # Get recent performance
            cursor.execute('''
                SELECT average_accuracy, best_model, worst_model, timestamp
                FROM ml_performance_log
                ORDER BY timestamp DESC
                LIMIT 1
            ''')
            
            performance_row = cursor.fetchone()
            recent_performance = None
            if performance_row:
                recent_performance = {
                    'average_accuracy': performance_row[0],
                    'best_model': performance_row[1],
                    'worst_model': performance_row[2],
                    'timestamp': performance_row[3]
                }
            
            conn.close()
            
            return {
                'is_running': self.is_running,
                'recent_training': recent_training,
                'recent_performance': recent_performance,
                'next_incremental': self._get_next_scheduled_time('incremental'),
                'next_full': self._get_next_scheduled_time('full')
            }
            
        except Exception as e:
            logger.error(f"Error getting training status: {str(e)}")
            return {
                'is_running': self.is_running,
                'recent_training': [],
                'recent_performance': None,
                'next_incremental': None,
                'next_full': None
            }
    
    def _get_next_scheduled_time(self, training_type: str) -> str:
        """Get next scheduled time for training type"""
        try:
            now = datetime.now()
            
            if training_type == 'incremental':
                # Next 6-hour interval
                next_time = now + timedelta(hours=6 - (now.hour % 6))
                next_time = next_time.replace(minute=0, second=0, microsecond=0)
            elif training_type == 'full':
                # Next 2 AM
                next_time = now + timedelta(days=1)
                next_time = next_time.replace(hour=2, minute=0, second=0, microsecond=0)
            else:
                return None
            
            return next_time.isoformat()
            
        except Exception as e:
            logger.error(f"Error calculating next scheduled time: {str(e)}")
            return None

# Global scheduler instance
ml_scheduler = MLTrainingScheduler()
