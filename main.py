"""
Main Trading Bot orchestrator
"""
import asyncio
import logging
import json
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import pandas as pd

# Import bot components
from deriv_api import DerivAPI
from technical_analysis import TechnicalAnalyzer
from ai_analyzer import AIAnalyzer
from martingale_system import SmartMartingaleSystem
from database import Trade, TradingSession, AIAnalysis, MarketData, SessionLocal, create_tables
import config

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOGGING_CONFIG['level']),
    format=config.LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(config.LOGGING_CONFIG['file'], encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DerivTradingBot:
    def __init__(self):
        self.api = DerivAPI()
        self.technical_analyzer = TechnicalAnalyzer()
        self.ai_analyzer = AIAnalyzer()
        self.martingale_system = SmartMartingaleSystem()
        
        self.is_running = False
        self.current_session_id = None
        self.account_balance = 0.0
        self.active_trades = {}
        self.market_data_buffer = {}
        self.last_analysis_time = datetime.now()
        self.daily_profit_loss = 0.0
        self.trading_enabled = True
        
        # Initialize database
        create_tables()
        
    async def start(self):
        """Start the trading bot"""
        logger.info("🚀 Starting Deriv AI Trading Bot...")
        
        try:
            # Connect to Deriv API
            await self.api.connect()
            
            if not self.api.is_connected:
                logger.error("Failed to connect to Deriv API")
                return
                
            # Get account information
            await self._initialize_account()
            
            # Start new trading session
            await self._start_new_session()
            
            # Start main trading loop
            self.is_running = True
            await self._main_trading_loop()
            
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
        finally:
            await self._cleanup()
            
    async def stop(self):
        """Stop the trading bot"""
        logger.info("🛑 Stopping trading bot...")
        self.is_running = False
        self.trading_enabled = False
        
        # Close any open positions
        await self._close_all_positions()
        
        # End current session
        await self._end_current_session()
        
    async def _initialize_account(self):
        """Initialize account information"""
        try:
            # Get account balance
            balance_info = await self.api.get_balance()
            if balance_info and 'balance' in balance_info:
                self.account_balance = float(balance_info['balance']['balance'])
                logger.info(f"💰 Account Balance: ${self.account_balance:.2f}")
            else:
                # Fallback for demo mode - use demo balance
                if config.DEMO_MODE:
                    self.account_balance = 10000.0  # Default demo balance
                    logger.info(f"💰 Demo Account Balance: ${self.account_balance:.2f}")
                else:
                    logger.error("Could not retrieve account balance")
                    self.account_balance = 0.0
                
            # Get account details
            account_info = await self.api.get_account_info()
            if account_info and 'get_account_status' in account_info:
                status = account_info['get_account_status']['status']
                logger.info(f"📋 Account Status: {status}")
                
        except Exception as e:
            logger.error(f"Error initializing account: {e}")
            # Set demo balance as fallback
            if config.DEMO_MODE:
                self.account_balance = 10000.0
                logger.info(f"💰 Using Demo Balance: ${self.account_balance:.2f}")
            else:
                self.account_balance = 0.0
            
    async def _start_new_session(self):
        """Start a new trading session"""
        try:
            self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Reset martingale system
            self.martingale_system.reset_session()
            
            # Create session record
            db = SessionLocal()
            try:
                session = TradingSession(
                    session_id=self.current_session_id,
                    start_time=datetime.now()
                )
                db.add(session)
                db.commit()
                logger.info(f"📊 Started new trading session: {self.current_session_id}")
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error starting new session: {e}")
            
    async def _main_trading_loop(self):
        """Main trading loop"""
        logger.info("🔄 Starting main trading loop...")
        
        while self.is_running:
            try:
                # Check if trading should continue
                if not await self._should_continue_trading():
                    logger.info("Trading stopped due to risk management")
                    break
                    
                # Get market data and analyze
                market_analysis = await self._analyze_market()
                
                if market_analysis and market_analysis.get('should_trade', False):
                    # Execute trade based on analysis
                    await self._execute_trade(market_analysis)
                    
                # Check existing trades
                await self._monitor_active_trades()
                
                # Update session statistics
                await self._update_session_stats()
                
                # Wait before next iteration
                await asyncio.sleep(10)  # 10 second intervals
                
            except Exception as e:
                logger.error(f"Error in main trading loop: {e}")
                await asyncio.sleep(30)  # Wait longer on error
                
    async def _should_continue_trading(self) -> bool:
        """Check if trading should continue based on risk management"""
        try:
            # Check martingale system limits
            stop_check = self.martingale_system.should_stop_trading(
                self.account_balance, 
                config.MAX_DAILY_LOSS
            )
            
            if stop_check['should_stop']:
                logger.warning(f"Trading stopped: {', '.join(stop_check['reasons'])}")
                return False
                
            # Check daily loss limit
            if self.daily_profit_loss < -config.MAX_DAILY_LOSS:
                logger.warning(f"Daily loss limit reached: ${self.daily_profit_loss:.2f}")
                return False
                
            # Check account balance
            if self.account_balance < config.INITIAL_STAKE * 5:
                logger.warning(f"Account balance too low: ${self.account_balance:.2f}")
                return False
                
            # Check if demo mode and balance is sufficient
            if config.DEMO_MODE and self.account_balance < 10:
                logger.warning("Demo account balance too low")
                return False
                
            return self.trading_enabled
            
        except Exception as e:
            logger.error(f"Error checking trading conditions: {e}")
            return False
            
    async def _analyze_market(self) -> Optional[Dict[str, Any]]:
        """Analyze market conditions and generate trading signals"""
        try:
            symbol = config.DEFAULT_SYMBOL
            
            # Get current market data
            market_data = await self._get_market_data(symbol)
            if not market_data:
                return None
                
            # Get technical analysis
            technical_indicators = await self._get_technical_analysis(symbol)
            
            # Get AI analysis
            ai_analysis = await self.ai_analyzer.analyze_market_data(
                market_data, 
                technical_indicators,
                await self._get_historical_performance()
            )
            
            # Combine all analysis
            combined_analysis = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'market_data': market_data,
                'technical_indicators': technical_indicators,
                'ai_analysis': ai_analysis,
                'should_trade': self._should_execute_trade(ai_analysis, technical_indicators),
                'recommended_direction': self._get_trade_direction(ai_analysis, technical_indicators),
                'confidence_score': self._calculate_confidence_score(ai_analysis, technical_indicators)
            }
            
            # Store analysis in database
            await self._store_analysis(combined_analysis)
            
            return combined_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market: {e}")
            return None
            
    async def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current market data for symbol"""
        try:
            # Get current tick data
            tick_data = await self.api.get_ticks(symbol)
            if 'tick' not in tick_data:
                return None
                
            tick = tick_data['tick']
            
            # Get candlestick data for analysis
            candles = await self.api.get_candles(symbol, "60", 100)  # 1-minute candles
            
            market_data = {
                'symbol': symbol,
                'current_price': float(tick['quote']),
                'timestamp': datetime.fromtimestamp(tick['epoch']),
                'candles': candles.get('candles', []) if 'candles' in candles else [],
                'bid': float(tick.get('bid', tick['quote'])),
                'ask': float(tick.get('ask', tick['quote']))
            }
            
            # Store in buffer for technical analysis
            if symbol not in self.market_data_buffer:
                self.market_data_buffer[symbol] = []
                
            self.market_data_buffer[symbol].append({
                'timestamp': market_data['timestamp'],
                'price': market_data['current_price'],
                'bid': market_data['bid'],
                'ask': market_data['ask']
            })
            
            # Keep only recent data
            if len(self.market_data_buffer[symbol]) > 1000:
                self.market_data_buffer[symbol] = self.market_data_buffer[symbol][-1000:]
                
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
            
    async def _get_technical_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get technical analysis for symbol"""
        try:
            # Get recent candle data
            candles_data = await self.api.get_candles(symbol, "60", 200)
            
            if 'candles' not in candles_data or not candles_data['candles']:
                logger.warning(f"No candle data available for {symbol}")
                return {}
                
            # Convert to DataFrame
            candles = candles_data['candles']
            df = pd.DataFrame(candles)
            
            # Ensure we have the required columns
            if 'open' in df.columns:
                df['open'] = pd.to_numeric(df['open'])
                df['high'] = pd.to_numeric(df['high'])
                df['low'] = pd.to_numeric(df['low'])
                df['close'] = pd.to_numeric(df['close'])
                
                # Calculate technical indicators
                indicators = self.technical_analyzer.calculate_indicators(df)
                return indicators
            else:
                logger.warning("Invalid candle data format")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting technical analysis: {e}")
            return {}
            
    async def _get_historical_performance(self) -> Dict[str, Any]:
        """Get historical trading performance"""
        try:
            db = SessionLocal()
            try:
                # Get recent trades
                recent_trades = db.query(Trade).order_by(Trade.created_at.desc()).limit(50).all()
                
                if not recent_trades:
                    return {'total_trades': 0, 'win_rate': 0, 'average_profit': 0}
                    
                total_trades = len(recent_trades)
                winning_trades = len([t for t in recent_trades if t.profit_loss and t.profit_loss > 0])
                total_profit = sum(t.profit_loss or 0 for t in recent_trades)
                
                return {
                    'total_trades': total_trades,
                    'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                    'average_profit': total_profit / total_trades if total_trades > 0 else 0,
                    'recent_trend': 'WINNING' if winning_trades > total_trades * 0.6 else 'LOSING'
                }
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error getting historical performance: {e}")
            return {'total_trades': 0, 'win_rate': 0, 'average_profit': 0}
            
    def _should_execute_trade(self, ai_analysis: Dict[str, Any], technical_indicators: Dict[str, Any]) -> bool:
        """Determine if a trade should be executed"""
        try:
            # Check AI confidence
            ai_confidence = ai_analysis.get('confidence', 0)
            if ai_confidence < 0.6:
                return False
                
            # Check AI prediction
            ai_prediction = ai_analysis.get('prediction', 'NEUTRAL')
            if ai_prediction == 'NEUTRAL':
                return False
                
            # Check technical signals
            tech_signal = self.technical_analyzer.get_trading_signal(technical_indicators)
            tech_confidence = tech_signal.get('confidence', 0)
            
            if tech_confidence < 0.5:
                return False
                
            # Check signal alignment
            if ai_prediction != tech_signal.get('signal', 'NEUTRAL'):
                return False  # Signals must agree
                
            # Check risk level
            risk_level = ai_analysis.get('risk_level', 'HIGH')
            if risk_level == 'HIGH' and ai_confidence < 0.8:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error determining trade execution: {e}")
            return False
            
    def _get_trade_direction(self, ai_analysis: Dict[str, Any], technical_indicators: Dict[str, Any]) -> str:
        """Get recommended trade direction"""
        ai_prediction = ai_analysis.get('prediction', 'NEUTRAL')
        tech_signal = self.technical_analyzer.get_trading_signal(technical_indicators)
        tech_prediction = tech_signal.get('signal', 'NEUTRAL')
        
        # Both signals should agree
        if ai_prediction == tech_prediction and ai_prediction != 'NEUTRAL':
            return ai_prediction
            
        return 'NEUTRAL'
        
    def _calculate_confidence_score(self, ai_analysis: Dict[str, Any], technical_indicators: Dict[str, Any]) -> float:
        """Calculate combined confidence score"""
        ai_confidence = ai_analysis.get('confidence', 0)
        tech_signal = self.technical_analyzer.get_trading_signal(technical_indicators)
        tech_confidence = tech_signal.get('confidence', 0)
        
        # Weighted average
        combined_confidence = (ai_confidence * 0.6) + (tech_confidence * 0.4)
        return combined_confidence
        
    async def _execute_trade(self, analysis: Dict[str, Any]):
        """Execute a trade based on analysis"""
        try:
            symbol = analysis['symbol']
            direction = analysis['recommended_direction']
            confidence = analysis['confidence_score']
            
            if direction == 'NEUTRAL':
                return
                
            # Calculate trade parameters using martingale system
            market_volatility = analysis['technical_indicators'].get('volatility', {}).get('atr', 1.0)
            stake_info = self.martingale_system.calculate_next_stake(
                'PENDING',  # No previous result yet
                self.account_balance,
                market_volatility,
                confidence
            )
            
            stake = stake_info['stake']
            duration = self.martingale_system.get_recommended_duration(market_volatility, confidence)
            
            # Convert direction to contract type
            contract_type = 'CALL' if direction == 'BUY' else 'PUT'
            
            logger.info(f"🎯 Executing {contract_type} trade on {symbol}")
            logger.info(f"💰 Stake: ${stake:.2f} | Duration: {duration} ticks | Confidence: {confidence:.2f}")
            
            # Place the trade
            trade_result = await self.api.buy_contract(
                symbol=symbol,
                contract_type=contract_type,
                amount=stake,
                duration=duration,
                duration_unit='t'  # ticks
            )
            
            if 'buy' in trade_result and 'contract_id' in trade_result['buy']:
                contract_id = trade_result['buy']['contract_id']
                
                # Store trade in database
                await self._store_trade(
                    contract_id,
                    symbol,
                    contract_type,
                    stake,
                    duration,
                    analysis
                )
                
                # Add to active trades
                self.active_trades[contract_id] = {
                    'symbol': symbol,
                    'contract_type': contract_type,
                    'stake': stake,
                    'start_time': datetime.now(),
                    'analysis': analysis
                }
                
                logger.info(f"✅ Trade placed successfully: {contract_id}")
                
            else:
                logger.error(f"Failed to place trade: {trade_result}")
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            
    async def _monitor_active_trades(self):
        """Monitor active trades and update their status"""
        try:
            completed_trades = []
            
            for contract_id, trade_info in self.active_trades.items():
                # Get contract information
                contract_info = await self.api.get_contract_info(contract_id)
                
                if 'proposal_open_contract' in contract_info:
                    contract = contract_info['proposal_open_contract']
                    
                    # Check if trade is finished
                    if contract.get('is_expired') or contract.get('is_sold'):
                        profit_loss = float(contract.get('profit', 0))
                        
                        # Update trade in database
                        await self._update_trade_result(contract_id, contract, profit_loss)
                        
                        # Update martingale system
                        result = 'WON' if profit_loss > 0 else 'LOST'
                        self.martingale_system.calculate_next_stake(
                            result,
                            self.account_balance,
                            1.0,  # Default volatility
                            0.5   # Default confidence
                        )
                        
                        # Update balances
                        self.daily_profit_loss += profit_loss
                        self.account_balance += profit_loss
                        
                        logger.info(f"📊 Trade {contract_id} completed: {result} (${profit_loss:.2f})")
                        completed_trades.append(contract_id)
                        
            # Remove completed trades
            for contract_id in completed_trades:
                del self.active_trades[contract_id]
                
        except Exception as e:
            logger.error(f"Error monitoring trades: {e}")
            
    async def _store_trade(self, contract_id: str, symbol: str, contract_type: str, 
                          stake: float, duration: int, analysis: Dict[str, Any]):
        """Store trade information in database"""
        try:
            db = SessionLocal()
            try:
                trade = Trade(
                    contract_id=contract_id,
                    symbol=symbol,
                    trade_type=contract_type,
                    stake=stake,
                    duration=duration,
                    ai_confidence=analysis['confidence_score'],
                    technical_signals=json.dumps(analysis['technical_indicators']),
                    martingale_level=self.martingale_system.consecutive_losses
                )
                
                db.add(trade)
                db.commit()
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error storing trade: {e}")
            
    async def _update_trade_result(self, contract_id: str, contract_info: Dict[str, Any], profit_loss: float):
        """Update trade result in database"""
        try:
            db = SessionLocal()
            try:
                trade = db.query(Trade).filter(Trade.contract_id == contract_id).first()
                if trade:
                    trade.profit_loss = profit_loss
                    trade.status = 'WON' if profit_loss > 0 else 'LOST'
                    trade.end_time = datetime.now()
                    trade.exit_price = float(contract_info.get('exit_tick', 0))
                    trade.entry_price = float(contract_info.get('entry_tick', 0))
                    
                    db.commit()
                    
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error updating trade result: {e}")
            
    async def _store_analysis(self, analysis: Dict[str, Any]):
        """Store analysis in database"""
        try:
            db = SessionLocal()
            try:
                ai_analysis = AIAnalysis(
                    symbol=analysis['symbol'],
                    timeframe='1m',
                    analysis_text=json.dumps(analysis['ai_analysis']),
                    prediction=analysis['recommended_direction'],
                    confidence=analysis['confidence_score'],
                    technical_data=json.dumps(analysis['technical_indicators'])
                )
                
                db.add(ai_analysis)
                db.commit()
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error storing analysis: {e}")
            
    async def _update_session_stats(self):
        """Update current session statistics"""
        try:
            if not self.current_session_id:
                return
                
            db = SessionLocal()
            try:
                session = db.query(TradingSession).filter(
                    TradingSession.session_id == self.current_session_id
                ).first()
                
                if session:
                    # Get session trades
                    trades = db.query(Trade).filter(
                        Trade.created_at >= session.start_time
                    ).all()
                    
                    total_trades = len(trades)
                    winning_trades = len([t for t in trades if t.profit_loss and t.profit_loss > 0])
                    total_profit = sum(t.profit_loss or 0 for t in trades)
                    
                    # Update session
                    session.total_trades = total_trades
                    session.winning_trades = winning_trades
                    session.losing_trades = total_trades - winning_trades
                    session.total_profit_loss = total_profit
                    session.current_consecutive_losses = self.martingale_system.consecutive_losses
                    
                    db.commit()
                    
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error updating session stats: {e}")
            
    async def _end_current_session(self):
        """End the current trading session"""
        try:
            if not self.current_session_id:
                return
                
            db = SessionLocal()
            try:
                session = db.query(TradingSession).filter(
                    TradingSession.session_id == self.current_session_id
                ).first()
                
                if session:
                    session.end_time = datetime.now()
                    session.is_active = False
                    db.commit()
                    
                    logger.info(f"📊 Session ended: {self.current_session_id}")
                    logger.info(f"Total trades: {session.total_trades}")
                    logger.info(f"Win rate: {(session.winning_trades/session.total_trades*100):.1f}%" if session.total_trades > 0 else "No trades")
                    logger.info(f"Total P&L: ${session.total_profit_loss:.2f}")
                    
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error ending session: {e}")
            
    async def _close_all_positions(self):
        """Close all open positions"""
        try:
            for contract_id in list(self.active_trades.keys()):
                try:
                    # Get current contract price and sell
                    contract_info = await self.api.get_contract_info(contract_id)
                    if 'proposal_open_contract' in contract_info:
                        contract = contract_info['proposal_open_contract']
                        if not contract.get('is_expired') and not contract.get('is_sold'):
                            bid_price = float(contract.get('bid_price', 0))
                            if bid_price > 0:
                                await self.api.sell_contract(contract_id, bid_price)
                                logger.info(f"🔄 Closed position: {contract_id}")
                                
                except Exception as e:
                    logger.error(f"Error closing position {contract_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            
    async def _cleanup(self):
        """Cleanup resources"""
        try:
            await self.api.disconnect()
            logger.info("🧹 Bot cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}")
    asyncio.create_task(bot.stop())

# Main execution
if __name__ == "__main__":
    # Create bot instance
    bot = DerivTradingBot()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the bot
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
    finally:
        logger.info("🏁 Bot shutdown complete")
