import numpy as np
import pandas as pd
import logging
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

logger = logging.getLogger(__name__)

class SelfLearningEngine:
    """Advanced self-learning ML engine that continuously improves trading strategies"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.performance_history = {}
        self.feature_importance = {}
        self.model_versions = {}
        
        # Learning parameters
        self.min_samples_for_training = 100
        self.retrain_threshold_accuracy = 0.6
        self.feature_selection_threshold = 0.01
        
        # Initialize models directory
        self.models_dir = 'ml_models'
        os.makedirs(self.models_dir, exist_ok=True)
    
    def continuous_learning_cycle(self, contract_type: str) -> Dict:
        """Execute complete learning cycle for a contract type"""
        try:
            logger.info(f"Starting continuous learning cycle for {contract_type}")
            
            # 1. Collect and prepare training data
            training_data = self._collect_recent_training_data(contract_type)
            
            if len(training_data) < self.min_samples_for_training:
                return {
                    'status': 'insufficient_data',
                    'samples_collected': len(training_data),
                    'min_required': self.min_samples_for_training
                }
            
            # 2. Feature engineering and selection
            features, targets = self._engineer_features(training_data)
            selected_features = self._select_best_features(features, targets, contract_type)
            
            # 3. Model training with cross-validation
            model_results = self._train_ensemble_models(selected_features, targets, contract_type)
            
            # 4. Model validation and comparison
            validation_results = self._validate_model_performance(model_results, contract_type)
            
            # 5. Deploy best model if improvement found
            deployment_result = self._deploy_best_model(validation_results, contract_type)
            
            # 6. Update performance tracking
            self._update_performance_history(contract_type, validation_results)
            
            # 7. Generate learning insights
            insights = self._generate_learning_insights(contract_type, validation_results)
            
            return {
                'status': 'success',
                'contract_type': contract_type,
                'samples_processed': len(training_data),
                'model_performance': validation_results,
                'deployment': deployment_result,
                'insights': insights,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in continuous learning cycle: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _collect_recent_training_data(self, contract_type: str, hours_back: int = 24) -> List[Dict]:
        """Collect recent trading data for training"""
        try:
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            
            # Get recent trades with outcomes
            cursor.execute('''
                SELECT 
                    market_data_json,
                    technical_indicators_json,
                    outcome,
                    profit_loss,
                    confidence_score,
                    entry_price,
                    exit_price,
                    duration,
                    timestamp
                FROM ml_training_data 
                WHERE contract_type = ? 
                AND timestamp > datetime('now', '-{} hours')
                ORDER BY timestamp DESC
            '''.format(hours_back), (contract_type,))
            
            rows = cursor.fetchall()
            conn.close()
            
            training_data = []
            for row in rows:
                try:
                    data = {
                        'market_data': json.loads(row[0]) if row[0] else {},
                        'technical_indicators': json.loads(row[1]) if row[1] else {},
                        'outcome': bool(row[2]),
                        'profit_loss': float(row[3]) if row[3] else 0,
                        'confidence_score': float(row[4]) if row[4] else 0.5,
                        'entry_price': float(row[5]) if row[5] else 0,
                        'exit_price': float(row[6]) if row[6] else 0,
                        'duration': int(row[7]) if row[7] else 300,
                        'timestamp': row[8]
                    }
                    training_data.append(data)
                except Exception as e:
                    logger.error(f"Error parsing training data: {str(e)}")
                    continue
            
            logger.info(f"Collected {len(training_data)} training samples for {contract_type}")
            return training_data
            
        except Exception as e:
            logger.error(f"Error collecting training data: {str(e)}")
            return []
    
    def _engineer_features(self, training_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Advanced feature engineering"""
        try:
            features_list = []
            targets = []
            
            for data in training_data:
                # Extract market features
                market_features = self._extract_market_features(data)
                
                # Extract technical features
                technical_features = self._extract_technical_features(data)
                
                # Extract behavioral features
                behavioral_features = self._extract_behavioral_features(data)
                
                # Extract time-based features
                time_features = self._extract_time_features(data)
                
                # Combine all features
                combined_features = np.concatenate([
                    market_features,
                    technical_features,
                    behavioral_features,
                    time_features
                ])
                
                features_list.append(combined_features)
                targets.append(1 if data['outcome'] else 0)
            
            return np.array(features_list), np.array(targets)
            
        except Exception as e:
            logger.error(f"Error engineering features: {str(e)}")
            return np.array([]), np.array([])
    
    def _extract_market_features(self, data: Dict) -> np.ndarray:
        """Extract market-based features"""
        market_data = data.get('market_data', {})
        
        features = [
            market_data.get('current_price', 1.0),
            market_data.get('volatility', 0.01),
            market_data.get('trend', 0),
            market_data.get('volume', 1000),
            market_data.get('bid_ask_spread', 0.001),
            market_data.get('market_depth', 0.5),
            market_data.get('price_momentum', 0),
            market_data.get('volatility_momentum', 0),
        ]
        
        return np.array(features)
    
    def _extract_technical_features(self, data: Dict) -> np.ndarray:
        """Extract technical indicator features"""
        tech_data = data.get('technical_indicators', {})
        
        features = [
            tech_data.get('rsi', 50) / 100,  # Normalize
            tech_data.get('macd', 0),
            tech_data.get('bollinger_position', 0.5),
            tech_data.get('sma_ratio', 1.0),
            tech_data.get('ema_ratio', 1.0),
            tech_data.get('atr', 0.01),
            tech_data.get('stochastic', 50) / 100,
            tech_data.get('williams_r', -50) / 100,
        ]
        
        return np.array(features)
    
    def _extract_behavioral_features(self, data: Dict) -> np.ndarray:
        """Extract trading behavior features"""
        features = [
            data.get('confidence_score', 0.5),
            1.0 if data.get('profit_loss', 0) > 0 else 0.0,  # Previous trade success
            data.get('duration', 300) / 3600,  # Normalize to hours
            abs(data.get('profit_loss', 0)) / max(data.get('entry_price', 1), 1),  # Relative P&L
        ]
        
        return np.array(features)
    
    def _extract_time_features(self, data: Dict) -> np.ndarray:
        """Extract time-based features"""
        try:
            timestamp = datetime.fromisoformat(data['timestamp'])
            
            features = [
                timestamp.hour / 24,  # Hour of day
                timestamp.weekday() / 6,  # Day of week
                timestamp.day / 31,  # Day of month
                np.sin(2 * np.pi * timestamp.hour / 24),  # Cyclic hour
                np.cos(2 * np.pi * timestamp.hour / 24),  # Cyclic hour
            ]
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting time features: {str(e)}")
            return np.array([0, 0, 0, 0, 0])
    
    def _select_best_features(self, features: np.ndarray, targets: np.ndarray, contract_type: str) -> np.ndarray:
        """Select most important features using various methods"""
        try:
            if len(features) == 0 or len(targets) == 0:
                return features
            
            # Use Random Forest for feature importance
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(features, targets)
            
            importances = rf.feature_importances_
            
            # Select features above threshold
            important_indices = np.where(importances > self.feature_selection_threshold)[0]
            
            if len(important_indices) == 0:
                # If no features meet threshold, select top 10
                important_indices = np.argsort(importances)[-10:]
            
            # Store feature importance for this contract type
            self.feature_importance[contract_type] = {
                'importances': importances.tolist(),
                'selected_indices': important_indices.tolist(),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Selected {len(important_indices)} features for {contract_type}")
            return features[:, important_indices]
            
        except Exception as e:
            logger.error(f"Error selecting features: {str(e)}")
            return features
    
    def _train_ensemble_models(self, features: np.ndarray, targets: np.ndarray, contract_type: str) -> Dict:
        """Train ensemble of models with time series validation"""
        try:
            results = {}
            
            # Define models to train
            models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    random_state=42
                ),
                'gradient_boost': GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                ),
                'ensemble': None  # Will be created later
            }
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            for model_name, model in models.items():
                if model is None:
                    continue
                    
                cv_scores = []
                cv_precisions = []
                cv_recalls = []
                
                for train_idx, val_idx in tscv.split(features):
                    X_train, X_val = features[train_idx], features[val_idx]
                    y_train, y_val = targets[train_idx], targets[val_idx]
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Predict and evaluate
                    y_pred = model.predict(X_val_scaled)
                    
                    cv_scores.append(accuracy_score(y_val, y_pred))
                    cv_precisions.append(precision_score(y_val, y_pred, zero_division=0))
                    cv_recalls.append(recall_score(y_val, y_pred, zero_division=0))
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'scaler': scaler,
                    'cv_accuracy': np.mean(cv_scores),
                    'cv_precision': np.mean(cv_precisions),
                    'cv_recall': np.mean(cv_recalls),
                    'cv_std': np.std(cv_scores)
                }
                
                logger.info(f"{contract_type} - {model_name}: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training ensemble models: {str(e)}")
            return {}
    
    def _validate_model_performance(self, model_results: Dict, contract_type: str) -> Dict:
        """Validate model performance and select best model"""
        try:
            if not model_results:
                return {}
            
            # Find best model based on accuracy
            best_model_name = max(model_results.keys(), 
                                key=lambda k: model_results[k]['cv_accuracy'])
            
            best_result = model_results[best_model_name]
            
            # Check if performance meets threshold
            performance_adequate = best_result['cv_accuracy'] >= self.retrain_threshold_accuracy
            
            # Compare with existing model if available
            current_model_path = os.path.join(self.models_dir, f"{contract_type}_model.joblib")
            improvement_found = False
            
            if os.path.exists(current_model_path):
                # Load current model performance
                current_performance = self.performance_history.get(contract_type, {}).get('accuracy', 0)
                improvement_found = best_result['cv_accuracy'] > current_performance + 0.02  # 2% improvement threshold
            else:
                improvement_found = True  # No existing model
            
            return {
                'best_model': best_model_name,
                'best_accuracy': best_result['cv_accuracy'],
                'best_precision': best_result['cv_precision'],
                'best_recall': best_result['cv_recall'],
                'performance_adequate': performance_adequate,
                'improvement_found': improvement_found,
                'all_results': model_results
            }
            
        except Exception as e:
            logger.error(f"Error validating model performance: {str(e)}")
            return {}
    
    def _deploy_best_model(self, validation_results: Dict, contract_type: str) -> Dict:
        """Deploy best model if improvement is found"""
        try:
            if not validation_results.get('improvement_found'):
                return {
                    'deployed': False,
                    'reason': 'No significant improvement found'
                }
            
            if not validation_results.get('performance_adequate'):
                return {
                    'deployed': False,
                    'reason': 'Performance below threshold'
                }
            
            best_model_name = validation_results['best_model']
            model_data = validation_results['all_results'][best_model_name]
            
            # Save model and scaler
            model_path = os.path.join(self.models_dir, f"{contract_type}_model.joblib")
            scaler_path = os.path.join(self.models_dir, f"{contract_type}_scaler.joblib")
            
            joblib.dump(model_data['model'], model_path)
            joblib.dump(model_data['scaler'], scaler_path)
            
            # Update version tracking
            version = self.model_versions.get(contract_type, 0) + 1
            self.model_versions[contract_type] = version
            
            # Store in memory for immediate use
            self.models[contract_type] = model_data['model']
            self.scalers[contract_type] = model_data['scaler']
            
            logger.info(f"Deployed new model for {contract_type} v{version} with {validation_results['best_accuracy']:.3f} accuracy")
            
            return {
                'deployed': True,
                'model_type': best_model_name,
                'version': version,
                'accuracy': validation_results['best_accuracy'],
                'precision': validation_results['best_precision'],
                'recall': validation_results['best_recall']
            }
            
        except Exception as e:
            logger.error(f"Error deploying model: {str(e)}")
            return {
                'deployed': False,
                'error': str(e)
            }
    
    def get_learning_status(self) -> Dict:
        """Get current learning system status"""
        try:
            status = {
                'active_models': list(self.models.keys()),
                'model_versions': self.model_versions,
                'performance_history': self.performance_history,
                'feature_importance': self.feature_importance,
                'last_update': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting learning status: {str(e)}")
            return {'error': str(e)}
    
    def predict(self, contract_type: str, features: np.ndarray) -> Dict:
        """Make prediction using trained model"""
        try:
            if contract_type not in self.models:
                return {
                    'prediction': 0.5,
                    'confidence': 0.5,
                    'status': 'no_model'
                }
            
            model = self.models[contract_type]
            scaler = self.scalers[contract_type]
            
            # Scale features
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            confidence = max(model.predict_proba(features_scaled)[0])
            
            return {
                'prediction': int(prediction),
                'confidence': float(confidence),
                'status': 'success',
                'model_version': self.model_versions.get(contract_type, 1)
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {
                'prediction': 0.5,
                'confidence': 0.5,
                'status': 'error',
                'error': str(e)
            }