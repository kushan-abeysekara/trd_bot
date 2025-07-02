import numpy as np
import pandas as pd
import sqlite3
import logging
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class TrainingData:
    """Training data structure"""
    features: np.ndarray
    target: int  # 0 = loss, 1 = win
    contract_type: str
    timestamp: datetime
    profit_loss: float
    market_conditions: Dict
    confidence: float

class PatternRecognition:
    """Advanced pattern recognition for chart analysis"""
    
    def __init__(self):
        self.pattern_models = {}
        self.scaler = StandardScaler()
    
    def detect_chart_patterns(self, price_data: List[float]) -> Dict:
        """Detect chart patterns using ML"""
        if len(price_data) < 50:
            return {}
        
        prices = np.array(price_data)
        patterns = {}
        
        # Trend patterns
        patterns['trend'] = self._detect_trend_patterns(prices)
        
        # Support/Resistance patterns
        patterns['support_resistance'] = self._detect_support_resistance(prices)
        
        # Reversal patterns
        patterns['reversals'] = self._detect_reversal_patterns(prices)
        
        # Continuation patterns
        patterns['continuations'] = self._detect_continuation_patterns(prices)
        
        # Volatility patterns
        patterns['volatility'] = self._detect_volatility_patterns(prices)
        
        return patterns
    
    def _detect_trend_patterns(self, prices: np.ndarray) -> Dict:
        """Detect trend patterns"""
        # Calculate various trend indicators
        short_trend = np.polyfit(range(len(prices[-10:])), prices[-10:], 1)[0]
        medium_trend = np.polyfit(range(len(prices[-20:])), prices[-20:], 1)[0]
        long_trend = np.polyfit(range(len(prices[-50:])), prices[-50:], 1)[0]
        
        # Trend strength
        trend_strength = abs(short_trend) / np.std(prices[-10:]) if np.std(prices[-10:]) > 0 else 0
        
        return {
            'short_trend': float(short_trend),
            'medium_trend': float(medium_trend),
            'long_trend': float(long_trend),
            'trend_strength': float(trend_strength),
            'trend_consistency': float(np.corrcoef(range(len(prices[-20:])), prices[-20:])[0, 1]) if len(prices) >= 20 else 0
        }
    
    def _detect_support_resistance(self, prices: np.ndarray) -> Dict:
        """Detect support and resistance levels"""
        highs = []
        lows = []
        
        # Find local highs and lows
        for i in range(2, len(prices) - 2):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1] and prices[i] > prices[i-2] and prices[i] > prices[i+2]:
                highs.append(prices[i])
            if prices[i] < prices[i-1] and prices[i] < prices[i+1] and prices[i] < prices[i-2] and prices[i] < prices[i+2]:
                lows.append(prices[i])
        
        current_price = prices[-1]
        
        # Calculate support and resistance
        support = np.mean(lows) if lows else current_price * 0.999
        resistance = np.mean(highs) if highs else current_price * 1.001
        
        return {
            'support_level': float(support),
            'resistance_level': float(resistance),
            'support_distance': float(abs(current_price - support) / current_price),
            'resistance_distance': float(abs(resistance - current_price) / current_price),
            'support_strength': len(lows),
            'resistance_strength': len(highs)
        }
    
    def _detect_reversal_patterns(self, prices: np.ndarray) -> Dict:
        """Detect reversal patterns"""
        if len(prices) < 10:
            return {}
        
        # Double top/bottom detection
        recent_highs = []
        recent_lows = []
        
        for i in range(5, len(prices) - 5):
            if all(prices[i] >= prices[i+j] for j in range(-5, 6)):
                recent_highs.append((i, prices[i]))
            if all(prices[i] <= prices[i+j] for j in range(-5, 6)):
                recent_lows.append((i, prices[i]))
        
        # RSI divergence
        rsi_values = self._calculate_rsi(prices)
        price_momentum = np.diff(prices[-10:])
        rsi_momentum = np.diff(rsi_values[-10:]) if len(rsi_values) >= 10 else []
        
        divergence = 0
        if len(price_momentum) > 0 and len(rsi_momentum) > 0:
            divergence = np.corrcoef(price_momentum, rsi_momentum)[0, 1]
        
        return {
            'double_top_probability': self._calculate_double_top_prob(recent_highs),
            'double_bottom_probability': self._calculate_double_bottom_prob(recent_lows),
            'rsi_divergence': float(divergence) if not np.isnan(divergence) else 0,
            'reversal_strength': float(abs(divergence)) if not np.isnan(divergence) else 0
        }
    
    def _detect_continuation_patterns(self, prices: np.ndarray) -> Dict:
        """Detect continuation patterns"""
        if len(prices) < 20:
            return {}
        
        # Flag pattern detection
        trend_strength = np.polyfit(range(len(prices[-20:])), prices[-20:], 1)[0]
        consolidation_volatility = np.std(prices[-10:]) / np.mean(prices[-10:])
        
        # Triangle patterns
        highs_trend = 0
        lows_trend = 0
        
        if len(prices) >= 20:
            # Get recent highs and lows
            highs = [prices[i] for i in range(len(prices)) if i > 0 and i < len(prices)-1 and prices[i] > prices[i-1] and prices[i] > prices[i+1]]
            lows = [prices[i] for i in range(len(prices)) if i > 0 and i < len(prices)-1 and prices[i] < prices[i-1] and prices[i] < prices[i+1]]
            
            if len(highs) >= 3:
                highs_trend = np.polyfit(range(len(highs[-3:])), highs[-3:], 1)[0]
            if len(lows) >= 3:
                lows_trend = np.polyfit(range(len(lows[-3:])), lows[-3:], 1)[0]
        
        return {
            'flag_pattern_strength': float(abs(trend_strength) / consolidation_volatility) if consolidation_volatility > 0 else 0,
            'triangle_convergence': float(abs(highs_trend - lows_trend)),
            'continuation_probability': float(min(1.0, abs(trend_strength) * 100))
        }
    
    def _detect_volatility_patterns(self, prices: np.ndarray) -> Dict:
        """Detect volatility patterns"""
        returns = np.diff(prices) / prices[:-1]
        
        # Volatility clustering
        volatility = np.array([np.std(returns[max(0, i-5):i+1]) for i in range(len(returns))])
        volatility_autocorr = np.corrcoef(volatility[:-1], volatility[1:])[0, 1] if len(volatility) > 1 else 0
        
        # GARCH-like volatility persistence
        high_vol_periods = volatility > np.percentile(volatility, 75)
        vol_persistence = np.mean(high_vol_periods)
        
        return {
            'volatility_clustering': float(volatility_autocorr) if not np.isnan(volatility_autocorr) else 0,
            'volatility_persistence': float(vol_persistence),
            'current_volatility_regime': 'high' if volatility[-1] > np.percentile(volatility, 75) else 'low'
        }
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.array([np.mean(gains[max(0, i-period):i+1]) for i in range(len(gains))])
        avg_losses = np.array([np.mean(losses[max(0, i-period):i+1]) for i in range(len(losses))])
        
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_double_top_prob(self, highs: List[Tuple]) -> float:
        """Calculate double top probability"""
        if len(highs) < 2:
            return 0.0
        
        # Check for similar price levels
        recent_highs = [h[1] for h in highs[-3:]]
        if len(recent_highs) >= 2:
            price_similarity = 1 - abs(recent_highs[-1] - recent_highs[-2]) / recent_highs[-1]
            return min(1.0, price_similarity * 2)
        
        return 0.0
    
    def _calculate_double_bottom_prob(self, lows: List[Tuple]) -> float:
        """Calculate double bottom probability"""
        if len(lows) < 2:
            return 0.0
        
        # Check for similar price levels
        recent_lows = [l[1] for l in lows[-3:]]
        if len(recent_lows) >= 2:
            price_similarity = 1 - abs(recent_lows[-1] - recent_lows[-2]) / recent_lows[-1]
            return min(1.0, price_similarity * 2)
        
        return 0.0

class SelfTrainingMLEngine:
    """Self-training machine learning engine for trading"""
    
    def __init__(self, db_path: str = 'trading_bot.db'):
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.pattern_recognizer = PatternRecognition()
        self.feature_importance = {}
        self.model_performance = {}
        
        # Initialize database
        self._init_training_database()
        
        # Load existing models
        self._load_models()
    
    def _init_training_database(self):
        """Initialize training database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Training data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    contract_type TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    target INTEGER NOT NULL,
                    profit_loss REAL NOT NULL,
                    market_conditions_json TEXT,
                    confidence REAL DEFAULT 0.5,
                    strategy_used TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    duration INTEGER,
                    volatility REAL,
                    trend_strength REAL,
                    rsi REAL,
                    pattern_detected TEXT
                )
            ''')
            
            # Model performance tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    contract_type TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    accuracy REAL NOT NULL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    training_samples INTEGER,
                    validation_samples INTEGER,
                    feature_importance_json TEXT,
                    hyperparameters_json TEXT
                )
            ''')
            
            # Pattern recognition results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pattern_recognition (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    pattern_type TEXT NOT NULL,
                    pattern_data_json TEXT NOT NULL,
                    market_outcome INTEGER,
                    confidence REAL,
                    price_move_direction INTEGER,
                    price_move_magnitude REAL
                )
            ''')
            
            # Market regime classification
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_regimes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    regime_type TEXT NOT NULL,
                    volatility_level TEXT,
                    trend_strength REAL,
                    duration_minutes INTEGER,
                    transition_probability REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Training database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
    
    def add_training_data(self, market_data: Dict, trade_result: Dict):
        """Add new training data from completed trade"""
        try:
            # Extract features
            features = self._extract_comprehensive_features(market_data)
            if features is None:
                return
            
            # Determine target (1 for profitable trade, 0 for loss)
            target = 1 if trade_result.get('profit_loss', 0) > 0 else 0
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ml_training_data (
                    contract_type, features_json, target, profit_loss,
                    market_conditions_json, confidence, strategy_used,
                    entry_price, exit_price, duration, volatility,
                    trend_strength, rsi, pattern_detected
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_result.get('contract_type', 'unknown'),
                json.dumps(features.tolist()),
                target,
                trade_result.get('profit_loss', 0),
                json.dumps(market_data),
                trade_result.get('confidence', 0.5),
                trade_result.get('strategy_used', 'unknown'),
                trade_result.get('entry_price', 0),
                trade_result.get('exit_price', 0),
                trade_result.get('duration', 0),
                market_data.get('volatility', 0),
                market_data.get('trend_strength', 0),
                market_data.get('rsi', 50),
                json.dumps(market_data.get('patterns', {}))
            ))
            
            conn.commit()
            conn.close()
            
            # Trigger retraining if we have enough new data
            self._check_retrain_trigger(trade_result.get('contract_type', 'unknown'))
            
            logger.info(f"Added training data for {trade_result.get('contract_type')} - Target: {target}")
            
        except Exception as e:
            logger.error(f"Error adding training data: {str(e)}")
    
    def _extract_comprehensive_features(self, market_data: Dict) -> Optional[np.ndarray]:
        """Extract comprehensive features for ML training"""
        try:
            # Price history features
            price_history = market_data.get('price_history', [])
            if len(price_history) < 50:
                return None
            
            prices = np.array([p.get('price', p) if isinstance(p, dict) else p for p in price_history[-100:]])
            
            features = []
            
            # Technical indicators
            features.extend([
                market_data.get('volatility', 0),
                market_data.get('trend_strength', 0),
                market_data.get('rsi', 50) / 100.0,  # Normalize
                market_data.get('momentum', 0)
            ])
            
            # Price-based features
            current_price = prices[-1]
            price_changes = np.diff(prices)
            
            # Moving averages
            if len(prices) >= 20:
                sma_5 = np.mean(prices[-5:])
                sma_10 = np.mean(prices[-10:])
                sma_20 = np.mean(prices[-20:])
                
                features.extend([
                    current_price / sma_5 - 1,
                    current_price / sma_10 - 1,
                    current_price / sma_20 - 1,
                    (sma_5 - sma_10) / sma_10,
                    (sma_10 - sma_20) / sma_20
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
            
            # Volatility features
            if len(price_changes) > 0:
                returns = price_changes / prices[:-1]
                vol_5 = np.std(returns[-5:]) if len(returns) >= 5 else 0
                vol_10 = np.std(returns[-10:]) if len(returns) >= 10 else 0
                vol_20 = np.std(returns[-20:]) if len(returns) >= 20 else 0
                
                features.extend([vol_5, vol_10, vol_20])
            else:
                features.extend([0, 0, 0])
            
            # Pattern features
            patterns = market_data.get('patterns', {})
            if patterns:
                features.extend([
                    patterns.get('trend', {}).get('trend_strength', 0),
                    patterns.get('support_resistance', {}).get('support_distance', 0),
                    patterns.get('support_resistance', {}).get('resistance_distance', 0),
                    patterns.get('reversals', {}).get('reversal_strength', 0),
                    patterns.get('continuations', {}).get('continuation_probability', 0),
                    patterns.get('volatility', {}).get('volatility_clustering', 0)
                ])
            else:
                features.extend([0, 0, 0, 0, 0, 0])
            
            # Time-based features
            current_time = datetime.now()
            features.extend([
                current_time.hour / 24.0,
                current_time.weekday() / 6.0,
                current_time.minute / 60.0
            ])
            
            # Market microstructure
            if len(prices) >= 10:
                price_acceleration = np.mean(np.diff(np.diff(prices[-10:])))
                features.append(price_acceleration)
            else:
                features.append(0)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return None
    
    def train_models(self, contract_type: str, min_samples: int = 100):
        """Train ML models for specific contract type"""
        try:
            # Load training data
            X, y = self._load_training_data(contract_type, min_samples)
            if X is None or len(X) < min_samples:
                logger.warning(f"Not enough training data for {contract_type}: {len(X) if X is not None else 0} samples")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train multiple models
            models = {
                'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
                'gradient_boost': GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42),
                'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
                'svm': SVC(probability=True, kernel='rbf', random_state=42)
            }
            
            best_model = None
            best_score = 0
            model_results = {}
            
            for name, model in models.items():
                try:
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    model_results[name] = {
                        'accuracy': accuracy,
                        'model': model,
                        'predictions': y_pred
                    }
                    
                    if accuracy > best_score:
                        best_score = accuracy
                        best_model = name
                    
                    logger.info(f"{contract_type} - {name}: {accuracy:.3f} accuracy")
                    
                except Exception as e:
                    logger.error(f"Error training {name} for {contract_type}: {str(e)}")
            
            if best_model:
                # Save best model
                self.models[contract_type] = model_results[best_model]['model']
                self.scalers[contract_type] = scaler
                
                # Calculate feature importance
                if hasattr(self.models[contract_type], 'feature_importances_'):
                    self.feature_importance[contract_type] = self.models[contract_type].feature_importances_
                
                # Save performance metrics
                self._save_model_performance(contract_type, model_results, X_train.shape[0], X_test.shape[0])
                
                # Save model to disk
                self._save_model_to_disk(contract_type)
                
                logger.info(f"Successfully trained {contract_type} model - Best: {best_model} ({best_score:.3f})")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error training models for {contract_type}: {str(e)}")
            return False
    
    def predict(self, market_data: Dict, contract_type: str) -> Dict:
        """Make prediction using trained model"""
        try:
            if contract_type not in self.models:
                return {'confidence': 0.5, 'direction': 'neutral', 'reasoning': 'Model not trained'}
            
            # Extract features
            features = self._extract_comprehensive_features(market_data)
            if features is None:
                return {'confidence': 0.5, 'direction': 'neutral', 'reasoning': 'Insufficient data'}
            
            # Scale features
            scaler = self.scalers.get(contract_type)
            if scaler is None:
                return {'confidence': 0.5, 'direction': 'neutral', 'reasoning': 'Scaler not available'}
            
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Make prediction
            model = self.models[contract_type]
            prediction = model.predict(features_scaled)[0]
            confidence = max(model.predict_proba(features_scaled)[0])
            
            # Generate reasoning based on feature importance
            reasoning = self._generate_prediction_reasoning(features, contract_type, market_data)
            
            return {
                'prediction': int(prediction),
                'confidence': float(confidence),
                'direction': 'up' if prediction == 1 else 'down',
                'reasoning': reasoning,
                'model_type': type(model).__name__
            }
            
        except Exception as e:
            logger.error(f"Prediction error for {contract_type}: {str(e)}")
            return {'confidence': 0.5, 'direction': 'neutral', 'reasoning': f'Prediction error: {str(e)}'}
    
    def _generate_prediction_reasoning(self, features: np.ndarray, contract_type: str, market_data: Dict) -> str:
        """Generate human-readable reasoning for prediction"""
        try:
            reasoning_parts = []
            
            # Market conditions
            volatility = market_data.get('volatility', 0)
            trend_strength = market_data.get('trend_strength', 0)
            rsi = market_data.get('rsi', 50)
            
            if volatility > 0.02:
                reasoning_parts.append("High volatility detected")
            elif volatility < 0.005:
                reasoning_parts.append("Low volatility environment")
            
            if abs(trend_strength) > 0.5:
                direction = "upward" if trend_strength > 0 else "downward"
                reasoning_parts.append(f"Strong {direction} trend")
            
            if rsi > 70:
                reasoning_parts.append("Overbought conditions (RSI)")
            elif rsi < 30:
                reasoning_parts.append("Oversold conditions (RSI)")
            
            # Pattern-based reasoning
            patterns = market_data.get('patterns', {})
            if patterns:
                if patterns.get('reversals', {}).get('reversal_strength', 0) > 0.3:
                    reasoning_parts.append("Reversal pattern detected")
                if patterns.get('continuations', {}).get('continuation_probability', 0) > 0.6:
                    reasoning_parts.append("Trend continuation likely")
            
            # Feature importance reasoning
            if contract_type in self.feature_importance:
                top_features = np.argsort(self.feature_importance[contract_type])[-3:]
                feature_names = ['volatility', 'trend', 'rsi', 'momentum', 'sma_ratio', 'pattern_strength']
                
                if len(top_features) > 0 and len(feature_names) > max(top_features):
                    important_feature = feature_names[top_features[-1]] if top_features[-1] < len(feature_names) else 'technical indicator'
                    reasoning_parts.append(f"Key factor: {important_feature}")
            
            if not reasoning_parts:
                reasoning_parts.append("Based on comprehensive technical analysis")
            
            return "; ".join(reasoning_parts)
            
        except Exception as e:
            return "ML model analysis"
    
    def get_model_performance(self, contract_type: str) -> Dict:
        """Get model performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT accuracy, precision_score, recall_score, f1_score, training_samples
                FROM model_performance 
                WHERE contract_type = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            ''', (contract_type,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'accuracy': result[0],
                    'precision': result[1],
                    'recall': result[2],
                    'f1_score': result[3],
                    'training_samples': result[4]
                }
            
            return {'accuracy': 0, 'training_samples': 0}
            
        except Exception as e:
            logger.error(f"Error getting performance for {contract_type}: {str(e)}")
            return {'accuracy': 0, 'training_samples': 0}
    
    def _load_training_data(self, contract_type: str, min_samples: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load training data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT features_json, target 
                FROM ml_training_data 
                WHERE contract_type = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (contract_type, min_samples * 10))  # Get more data than minimum
            
            results = cursor.fetchall()
            conn.close()
            
            if len(results) < min_samples:
                return None, None
            
            X = []
            y = []
            
            for features_json, target in results:
                try:
                    features = json.loads(features_json)
                    X.append(features)
                    y.append(target)
                except json.JSONDecodeError:
                    continue
            
            if len(X) < min_samples:
                return None, None
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error loading training data for {contract_type}: {str(e)}")
            return None, None
    
    def _save_model_performance(self, contract_type: str, model_results: Dict, train_samples: int, test_samples: int):
        """Save model performance to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for model_name, results in model_results.items():
                cursor.execute('''
                    INSERT INTO model_performance (
                        contract_type, model_type, accuracy, training_samples, validation_samples
                    ) VALUES (?, ?, ?, ?, ?)
                ''', (contract_type, model_name, results['accuracy'], train_samples, test_samples))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving performance: {str(e)}")
    
    def _save_model_to_disk(self, contract_type: str):
        """Save trained model to disk"""
        try:
            import os
            model_dir = 'models'
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            with open(f'{model_dir}/{contract_type}_model.pkl', 'wb') as f:
                pickle.dump(self.models[contract_type], f)
            
            # Save scaler
            with open(f'{model_dir}/{contract_type}_scaler.pkl', 'wb') as f:
                pickle.dump(self.scalers[contract_type], f)
            
            logger.info(f"Saved {contract_type} model to disk")
            
        except Exception as e:
            logger.error(f"Error saving model to disk: {str(e)}")
    
    def _load_models(self):
        """Load existing models from disk"""
        try:
            import os
            model_dir = 'models'
            
            if not os.path.exists(model_dir):
                return
            
            for filename in os.listdir(model_dir):
                if filename.endswith('_model.pkl'):
                    contract_type = filename.replace('_model.pkl', '')
                    
                    try:
                        # Load model
                        with open(f'{model_dir}/{filename}', 'rb') as f:
                            self.models[contract_type] = pickle.load(f)
                        
                        # Load scaler
                        scaler_file = f'{model_dir}/{contract_type}_scaler.pkl'
                        if os.path.exists(scaler_file):
                            with open(scaler_file, 'rb') as f:
                                self.scalers[contract_type] = pickle.load(f)
                        
                        logger.info(f"Loaded {contract_type} model from disk")
                        
                    except Exception as e:
                        logger.error(f"Error loading {contract_type} model: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    def _check_retrain_trigger(self, contract_type: str):
        """Check if we should retrain the model"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Count samples since last training
            cursor.execute('''
                SELECT COUNT(*) FROM ml_training_data 
                WHERE contract_type = ? 
                AND timestamp > COALESCE(
                    (SELECT MAX(timestamp) FROM model_performance WHERE contract_type = ?),
                    '1900-01-01'
                )
            ''', (contract_type, contract_type))
            
            new_samples = cursor.fetchone()[0]
            conn.close()
            
            # Retrain if we have 50+ new samples
            if new_samples >= 50:
                logger.info(f"Triggering retrain for {contract_type} - {new_samples} new samples")
                self.train_models(contract_type)
            
        except Exception as e:
            logger.error(f"Error checking retrain trigger: {str(e)}")

# Global instance
self_training_engine = SelfTrainingMLEngine()
