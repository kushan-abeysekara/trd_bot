import time
import random
import threading
from datetime import datetime
from typing import Dict, Any, List
from deriv_api import DerivAPI
from strategy_engine import StrategyEngine, TradeSignal


class TradingBot:
    def __init__(self, api_token: str):
        self.api = DerivAPI(api_token)
        self.is_running = False
        self.trade_amount = 1.0
        self.duration_ticks = 1  # Changed from 5 to 1 for single-tick trading
        self.trade_history = []
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit_loss': 0.0,
            'winning_rate': 0.0
        }
        self.callbacks = {
            'balance_update': None,
            'trade_update': None,
            'stats_update': None,
            'connection_status': None,
            'strategy_signal': None
        }
        
        # Initialize strategy engine
        self.strategy_engine = StrategyEngine()
        self.strategy_scanning = False
        self.last_signal_time = 0
        self.min_signal_interval = 2  # Reduced to 2 seconds for faster trading
        self.last_strategy_signals = {}  # Track last signal time per strategy
        self.strategy_cooldown = 8  # Reduced to 8 seconds per individual strategy
        self.signal_count = 0  # Track total signals received
        
        # Trading mode: 'random' or 'strategy'
        self.trading_mode = 'strategy'
        
        # Take Profit and Stop Loss settings
        self.take_profit_enabled = False
        self.stop_loss_enabled = False
        self.take_profit_amount = 0.0
        self.stop_loss_amount = 0.0
        self.initial_balance = 0.0
        self.session_profit_loss = 0.0
        
    def set_callback(self, event: str, callback):
        """Set callback for events"""
        self.callbacks[event] = callback
        
    def connect(self):
        """Connect to Deriv API"""
        def connection_callback(success, error=None):
            if self.callbacks['connection_status']:
                self.callbacks['connection_status'](success, error)
                
        # Set up balance update callback
        def balance_update_callback(new_balance):
            if self.callbacks['balance_update']:
                self.callbacks['balance_update'](new_balance)
                
        self.api.set_balance_callback(balance_update_callback)
        self.api.connect(connection_callback)
        
    def start_trading(self, amount: float, duration: int = 1):  # Default to 1 tick
        """Start automated trading"""
        self.trade_amount = amount
        self.duration_ticks = 1  # Always use 1 tick regardless of input
        self.is_running = True
        
        # Reset session tracking
        self.reset_session_tracking()
        
        # Start strategy scanning
        self.strategy_scanning = True
        self.strategy_engine.start_scanning(self._handle_strategy_signal)
        
        # Start price feed simulation (in real implementation, this would be live data)
        price_feed_thread = threading.Thread(target=self._simulate_price_feed)
        price_feed_thread.daemon = True
        price_feed_thread.start()
        
        # Start trading in a separate thread
        trading_thread = threading.Thread(target=self._trading_loop)
        trading_thread.daemon = True
        trading_thread.start()
        
    def stop_trading(self):
        """Stop automated trading"""
        self.is_running = False
        self.strategy_scanning = False
        self.strategy_engine.stop_scanning()
        
    def _trading_loop(self):
        """Main trading loop - now strategy-driven"""
        while self.is_running:
            if not self.api.is_connected:
                time.sleep(1)
                continue
                
            # Check take profit and stop loss limits
            if self.check_profit_loss_limits():
                break  # Stop trading if limits are hit
                
            # Strategy-based trading is handled by signal callbacks
            # This loop now just keeps the bot alive
            time.sleep(1)
            
    def _simulate_price_feed(self):
        """Simulate live price feed for strategy analysis - ENHANCED for more signals"""
        base_price = 1000.0
        price = base_price
        tick_counter = 0
        
        while self.strategy_scanning:
            tick_counter += 1
            
            # Enhanced price movement simulation for better strategy triggering
            
            # Base volatility with more variation
            change = random.uniform(-0.4, 0.4)  # ±0.4% change (increased)
            
            # Add volatility spikes (25% chance - increased)
            if random.random() < 0.25:
                spike_intensity = random.uniform(0.6, 2.0)  # 0.6-2.0% spike (increased)
                spike_direction = random.choice([-1, 1])
                change += spike_direction * spike_intensity
                print(f"📈 Price spike: {spike_direction * spike_intensity:.2f}%")
                
            # Add micro-trends (20% chance - increased)
            if random.random() < 0.20:
                trend_strength = random.uniform(0.15, 0.5)  # 0.15-0.5% trend
                trend_direction = random.choice([-1, 1])
                trend_length = random.randint(3, 8)  # 3-8 ticks in same direction
                print(f"📊 Micro-trend: {trend_direction} direction, {trend_length} ticks")
                
                for i in range(trend_length):
                    if not self.strategy_scanning:
                        break
                    price = price * (1 + (trend_direction * trend_strength) / 100)
                    self.strategy_engine.add_tick(price)
                    time.sleep(random.uniform(0.2, 0.6))  # Faster tick intervals
                continue
                
            # Add compression periods (12% chance)
            if random.random() < 0.12:
                compression_length = random.randint(5, 12)
                print(f"📉 Compression period: {compression_length} ticks")
                
                for i in range(compression_length):
                    if not self.strategy_scanning:
                        break
                    mini_change = random.uniform(-0.08, 0.08)  # Very small movements
                    price = price * (1 + mini_change / 100)
                    self.strategy_engine.add_tick(price)
                    time.sleep(random.uniform(0.6, 1.0))
                continue
            
            # Add divergence patterns (8% chance) - good for divergence strategies
            if random.random() < 0.08:
                print(f"🔄 Divergence pattern starting...")
                # Create price pattern that goes one way while momentum goes another
                divergence_direction = random.choice([-1, 1])
                for i in range(4):
                    if not self.strategy_scanning:
                        break
                    # Price goes one way
                    price_change = divergence_direction * 0.2
                    price = price * (1 + price_change / 100)
                    self.strategy_engine.add_tick(price)
                    time.sleep(random.uniform(0.4, 0.8))
                continue
            
            # Apply normal price movement
            price = price * (1 + change / 100)
            
            # Ensure price doesn't go too far from base
            if price < base_price * 0.93:  # Allow more range
                price = base_price * 0.93
            elif price > base_price * 1.07:
                price = base_price * 1.07
                
            # Feed price to strategy engine
            self.strategy_engine.add_tick(price)
            
            # Log every 50 ticks for monitoring
            if tick_counter % 50 == 0:
                indicators = self.strategy_engine.get_current_indicators()
                print(f"📊 Tick #{tick_counter}: Price={price:.2f}, RSI={indicators.get('rsi', 0):.1f}, "
                      f"Vol={indicators.get('volatility', 0):.2f}%, MACD={indicators.get('macd', 0):.3f}")
            
            # Variable tick intervals for realistic market simulation (faster)
            time.sleep(random.uniform(0.15, 0.8))  # 150ms to 800ms intervals
            
    def _handle_strategy_signal(self, signal: TradeSignal):
        """Handle trade signal from strategy engine - OPTIMIZED FOR FAST TRADING"""
        current_time = time.time()
        self.signal_count += 1
        
        print(f"📡 Signal #{self.signal_count} received: {signal.strategy_name} ({signal.confidence:.2f})")
        
        # Global signal interval check - much shorter now
        if current_time - self.last_signal_time < self.min_signal_interval:
            remaining = self.min_signal_interval - (current_time - self.last_signal_time)
            print(f"⏰ Global cooldown - {remaining:.1f}s remaining")
            return
        
        # Individual strategy cooldown check - OPTIMIZED based on confidence
        strategy_name = signal.strategy_name
        dynamic_cooldown = self.strategy_cooldown
        
        # High-confidence signals get SHORTER cooldowns, not longer!
        if signal.confidence > 0.85:
            dynamic_cooldown = 4  # Only 4 seconds for high-confidence signals
            print(f"🔥 High-confidence signal - reduced cooldown to {dynamic_cooldown}s")
        elif signal.confidence > 0.75:
            dynamic_cooldown = 6  # 6 seconds for good signals
        # else use default 8 seconds
        
        if strategy_name in self.last_strategy_signals:
            time_since_last = current_time - self.last_strategy_signals[strategy_name]
            if time_since_last < dynamic_cooldown:
                remaining_cooldown = dynamic_cooldown - time_since_last
                print(f"❄️  Strategy {strategy_name} cooldown - {remaining_cooldown:.1f}s remaining")
                return
        
        # EMERGENCY BYPASS: If no trades for 30 seconds, accept any signal
        if current_time - self.last_signal_time > 30:
            print(f"🚨 EMERGENCY BYPASS: No trades for 30s, accepting signal!")
            
        # Update timing trackers
        self.last_signal_time = current_time
        self.last_strategy_signals[strategy_name] = current_time
        
        # Execute trade based on signal
        contract_type = signal.direction  # 'CALL' or 'PUT'
        
        print(f"🎯 EXECUTING TRADE #{self.stats['total_trades'] + 1}: {signal.strategy_name}")
        print(f"   Direction: {signal.direction} | Confidence: {signal.confidence:.2f} | Hold: {signal.hold_time}s")
        print(f"   Reason: {signal.entry_reason}")
        print(f"   Conditions: {', '.join(signal.conditions_met)}")
        print(f"   📊 Cooldown used: {dynamic_cooldown}s | Total signals: {self.signal_count}")
        
        # Place the trade asynchronously
        trade_thread = threading.Thread(target=self._place_strategy_trade_async, args=(signal,))
        trade_thread.daemon = True
        trade_thread.start()
        
        # Notify frontend about strategy signal (non-blocking)
        if self.callbacks['strategy_signal']:
            try:
                self.callbacks['strategy_signal']({
                    'strategy_name': signal.strategy_name,
                    'direction': signal.direction,
                    'confidence': signal.confidence,
                    'entry_reason': signal.entry_reason,
                    'conditions_met': signal.conditions_met,
                    'hold_time': signal.hold_time,
                    'indicators': self.strategy_engine.get_current_indicators(),
                    'total_strategies': 35,
                    'trade_number': self.stats['total_trades'] + 1,
                    'cooldown_used': dynamic_cooldown,
                    'signals_received': self.signal_count
                })
            except Exception as e:
                print(f"⚠️  Error sending signal to frontend: {e}")
                
    def _place_strategy_trade_async(self, signal: TradeSignal):
        """Place a trade based on strategy signal (async version)"""
        try:
            self._place_strategy_trade(signal)
        except Exception as e:
            print(f"❌ Error placing strategy trade: {e}")
            
    def _place_strategy_trade(self, signal: TradeSignal):
        """Place a trade based on strategy signal"""
        print(f"📋 Placing trade for {signal.strategy_name}...")
        
        def proposal_callback(response):
            try:
                if response.get("error"):
                    print(f"❌ Proposal error: {response['error']['message']}")
                    return
                    
                proposal = response.get("proposal", {})
                proposal_id = proposal.get("id")
                ask_price = float(proposal.get("ask_price", 0))
                
                if proposal_id and ask_price > 0:
                    print(f"✅ Proposal received: {proposal_id}, Price: ${ask_price}")
                    
                    # Buy the contract
                    def buy_callback(buy_response):
                        try:
                            if buy_response.get("error"):
                                print(f"❌ Buy error: {buy_response['error']['message']}")
                                return
                                
                            buy_data = buy_response.get("buy", {})
                            contract_id = buy_data.get("contract_id")
                            buy_price = float(buy_data.get("buy_price", 0))
                            
                            trade_info = {
                                'id': contract_id,
                                'type': signal.direction,
                                'amount': self.trade_amount,
                                'buy_price': buy_price,
                                'duration': signal.hold_time,  # Use signal hold time
                                'timestamp': datetime.now().isoformat(),
                                'status': 'active',
                                'strategy_name': signal.strategy_name,
                                'strategy_confidence': signal.confidence,
                                'entry_reason': signal.entry_reason,
                                'conditions_met': signal.conditions_met
                            }
                            
                            self.trade_history.append(trade_info)
                            print(f"🎯 Trade placed: {contract_id} for {signal.strategy_name}")
                            
                            # Send trade update (non-blocking)
                            if self.callbacks['trade_update']:
                                try:
                                    self.callbacks['trade_update'](trade_info)
                                except Exception as e:
                                    print(f"⚠️  Error sending trade update: {e}")
                                    
                            # Start trade result simulation in separate thread
                            result_thread = threading.Thread(
                                target=self._simulate_strategy_trade_result, 
                                args=(trade_info, signal)
                            )
                            result_thread.daemon = True
                            result_thread.start()
                            
                        except Exception as e:
                            print(f"❌ Error in buy callback: {e}")
                        
                    self.api.buy_contract(proposal_id, ask_price, buy_callback)
                else:
                    print(f"❌ Invalid proposal response: {response}")
                    
            except Exception as e:
                print(f"❌ Error in proposal callback: {e}")
                
        # Get proposal (non-blocking)
        try:
            self.api.get_proposal(signal.direction, self.duration_ticks, self.trade_amount, proposal_callback)
        except Exception as e:
            print(f"❌ Error getting proposal: {e}")
            
    def _place_trade(self, contract_type: str):
        """Place a single trade"""
        def proposal_callback(response):
            if response.get("error"):
                print(f"Proposal error: {response['error']['message']}")
                return
                
            proposal = response.get("proposal", {})
            proposal_id = proposal.get("id")
            ask_price = float(proposal.get("ask_price", 0))
            
            if proposal_id and ask_price > 0:
                # Buy the contract
                def buy_callback(buy_response):
                    if buy_response.get("error"):
                        print(f"Buy error: {buy_response['error']['message']}")
                        return
                        
                    buy_data = buy_response.get("buy", {})
                    contract_id = buy_data.get("contract_id")
                    buy_price = float(buy_data.get("buy_price", 0))
                    
                    trade_info = {
                        'id': contract_id,
                        'type': contract_type,
                        'amount': self.trade_amount,
                        'buy_price': buy_price,
                        'duration': self.duration_ticks,
                        'timestamp': datetime.now().isoformat(),
                        'status': 'active'
                    }
                    
                    self.trade_history.append(trade_info)
                    
                    if self.callbacks['trade_update']:
                        self.callbacks['trade_update'](trade_info)
                        
                    # Simulate trade result after duration
                    self._simulate_trade_result(trade_info)
                    
                self.api.buy_contract(proposal_id, ask_price, buy_callback)
                
        self.api.get_proposal(contract_type, self.duration_ticks, self.trade_amount, proposal_callback)
        
    def _simulate_trade_result(self, trade_info):
        """Simulate trade result for regular (non-strategy) trades"""
        def delayed_result():
            # Wait for trade duration (in ticks, simulate as seconds for demo)
            duration_seconds = trade_info['duration'] * 2  # 2 seconds per tick for demo
            time.sleep(duration_seconds)
            
            # Win probability for regular trades (60% win rate)
            win_probability = 0.6
            is_win = random.random() < win_probability
            
            if is_win:
                payout = trade_info['buy_price'] * 1.95  # 95% payout
                profit_loss = payout - trade_info['buy_price']
                self.stats['winning_trades'] += 1
            else:
                profit_loss = -trade_info['buy_price']
                self.stats['losing_trades'] += 1
                
            self.stats['total_trades'] += 1
            self.stats['total_profit_loss'] += profit_loss
            self.stats['winning_rate'] = (self.stats['winning_trades'] / self.stats['total_trades']) * 100
            
            # Update session profit/loss tracking
            self.session_profit_loss += profit_loss
            
            # Update actual balance (simulate balance change)
            self.api.update_balance(profit_loss)
            
            # Get the updated balance
            updated_balance = self.get_balance()
            
            # Update trade info with final results
            trade_info['result'] = 'win' if is_win else 'loss'
            trade_info['profit_loss'] = profit_loss
            trade_info['status'] = 'completed'
            trade_info['session_profit_loss'] = self.session_profit_loss
            trade_info['balance_after'] = updated_balance
            
            print(f"💰 Trade Result: {trade_info['result'].upper()} | "
                  f"P&L: ${profit_loss:.2f} | "
                  f"Session P&L: ${self.session_profit_loss:.2f} | "
                  f"New Balance: ${updated_balance:.2f}")
            
            # Emit real-time balance update immediately
            if self.callbacks.get('balance_update'):
                self.callbacks['balance_update']({'balance': updated_balance})
            
            # Emit complete trade update with all final info
            if self.callbacks.get('trade_update'):
                self.callbacks['trade_update'](trade_info)
            
            # Check if take profit or stop loss limits are hit
            if self.check_profit_loss_limits():
                return  # Trading was stopped, exit the method
                
            if self.callbacks['stats_update']:
                self.callbacks['stats_update'](self.stats)
                
        # Run in separate thread
        result_thread = threading.Thread(target=delayed_result)
        result_thread.daemon = True
        result_thread.start()

    def _simulate_strategy_trade_result(self, trade_info, signal: TradeSignal):
        """Simulate trade result based on strategy confidence"""
        def delayed_result():
            # Use shorter wait time for 1-tick trades
            time.sleep(2)  # Wait 2 seconds for 1-tick result
            
            # Win probability based on strategy confidence
            win_probability = 0.5 + (signal.confidence * 0.3)  # 50% base + 30% max bonus
            is_win = random.random() < win_probability
            
            if is_win:
                payout = trade_info['buy_price'] * 1.95  # 95% payout
                profit_loss = payout - trade_info['buy_price']
                self.stats['winning_trades'] += 1
            else:
                profit_loss = -trade_info['buy_price']
                self.stats['losing_trades'] += 1
                
            self.stats['total_trades'] += 1
            self.stats['total_profit_loss'] += profit_loss
            self.stats['winning_rate'] = (self.stats['winning_trades'] / self.stats['total_trades']) * 100
            
            # Update session profit/loss tracking
            self.session_profit_loss += profit_loss
            
            # Update actual balance (simulate balance change)
            self.api.update_balance(profit_loss)
            
            # Get the updated balance
            updated_balance = self.get_balance()
            
            # Update trade info with final results
            trade_info['result'] = 'win' if is_win else 'loss'
            trade_info['profit_loss'] = profit_loss
            trade_info['status'] = 'completed'
            trade_info['actual_win_probability'] = win_probability
            trade_info['strategy_name'] = signal.strategy_name
            trade_info['session_profit_loss'] = self.session_profit_loss
            trade_info['balance_after'] = updated_balance
            
            print(f"💰 Trade Result: {trade_info['result'].upper()} | "
                  f"P&L: ${profit_loss:.2f} | "
                  f"Session P&L: ${self.session_profit_loss:.2f} | "
                  f"New Balance: ${updated_balance:.2f} | "
                  f"Strategy: {signal.strategy_name}")
            
            # Emit real-time balance update immediately
            if self.callbacks.get('balance_update'):
                self.callbacks['balance_update']({'balance': updated_balance})
            
            # Emit complete trade update with all final info
            if self.callbacks.get('trade_update'):
                self.callbacks['trade_update'](trade_info)
            
            # Check if take profit or stop loss limits are hit
            if self.check_profit_loss_limits():
                return  # Trading was stopped, exit the method
                
            if self.callbacks['stats_update']:
                self.callbacks['stats_update'](self.stats)
                
        # Run in separate thread
        result_thread = threading.Thread(target=delayed_result)
        result_thread.daemon = True
        result_thread.start()
        
    def set_trading_mode(self, mode: str):
        """Set trading mode: 'random' or 'strategy'"""
        self.trading_mode = mode
        
    # Methods moved to avoid duplication - see lines 347-357
        
    def get_strategy_indicators(self):
        """Get current technical indicators from strategy engine"""
        return self.strategy_engine.get_current_indicators()
        
    def get_active_strategies(self):
        """Get list of available strategies"""
        return list(self.strategy_engine.strategies.values())
        
    def get_balance(self):
        """Get current balance"""
        return self.api.get_balance_value()
        
    def get_stats(self):
        """Get trading statistics"""
        return self.stats
        
    def get_trade_history(self):
        """Get trade history"""
        return self.trade_history
        
    def disconnect(self):
        """Disconnect from API"""
        self.stop_trading()
        self.api.disconnect()
        
    def set_take_profit(self, enabled: bool, amount: float = 0.0):
        """Set take profit settings"""
        self.take_profit_enabled = enabled
        self.take_profit_amount = amount
        print(f"Take Profit: {'Enabled' if enabled else 'Disabled'} at ${amount}")
        
    def set_stop_loss(self, enabled: bool, amount: float = 0.0):
        """Set stop loss settings"""
        self.stop_loss_enabled = enabled
        self.stop_loss_amount = amount
        print(f"Stop Loss: {'Enabled' if enabled else 'Disabled'} at ${amount}")
        
    def get_bot_status(self):
        """Get current bot status for debugging"""
        current_time = time.time()
        
        # Calculate cooldown status for all strategies that have been used
        strategy_cooldown_status = {}
        for strategy, last_time in self.last_strategy_signals.items():
            time_since = current_time - last_time
            dynamic_cooldown = 8  # default
            # Apply same logic as in _handle_strategy_signal
            strategy_cooldown_status[strategy] = {
                'last_signal_time': last_time,
                'time_since_last': time_since,
                'cooldown_remaining': max(0, dynamic_cooldown - time_since),
                'is_ready': time_since >= dynamic_cooldown
            }
        
        return {
            'is_running': self.is_running,
            'strategy_scanning': self.strategy_scanning,
            'is_connected': self.api.is_connected if self.api else False,
            'last_signal_time': self.last_signal_time,
            'time_since_last_signal': current_time - self.last_signal_time if self.last_signal_time > 0 else 0,
            'global_cooldown_remaining': max(0, self.min_signal_interval - (current_time - self.last_signal_time)) if self.last_signal_time > 0 else 0,
            'total_trades': self.stats['total_trades'],
            'signals_received': self.signal_count,
            'active_strategies': len(self.last_strategy_signals),
            'strategy_cooldowns': strategy_cooldown_status,
            'ready_strategies': len([s for s in strategy_cooldown_status.values() if s['is_ready']]),
            'tick_count': len(self.strategy_engine.tick_history) if hasattr(self.strategy_engine, 'tick_history') else 0,
            'indicators': self.strategy_engine.get_current_indicators() if hasattr(self.strategy_engine, 'get_current_indicators') else {},
            'min_signal_interval': self.min_signal_interval,
            'strategy_cooldown': self.strategy_cooldown,
            'emergency_bypass_active': (current_time - self.last_signal_time > 30) if self.last_signal_time > 0 else False
        }
        
    def reset_session_tracking(self):
        """Reset session profit/loss tracking"""
        self.initial_balance = self.get_balance()
        self.session_profit_loss = 0.0
        print(f"Session tracking reset. Initial balance: ${self.initial_balance}")
        
    def check_profit_loss_limits(self):
        """Check if take profit or stop loss limits are hit"""
        if not (self.take_profit_enabled or self.stop_loss_enabled):
            return False
            
        current_profit_loss = self.session_profit_loss
        
        # Check take profit
        if self.take_profit_enabled and current_profit_loss >= self.take_profit_amount:
            print(f"🎉 TAKE PROFIT HIT! Profit: ${current_profit_loss:.2f} >= Target: ${self.take_profit_amount}")
            self._stop_trading_with_reason("Take Profit reached")
            return True
            
        # Check stop loss
        if self.stop_loss_enabled and current_profit_loss <= -self.stop_loss_amount:
            print(f"⛔ STOP LOSS HIT! Loss: ${current_profit_loss:.2f} <= Limit: ${-self.stop_loss_amount}")
            self._stop_trading_with_reason("Stop Loss reached")
            return True
            
        return False
        
    def _stop_trading_with_reason(self, reason: str):
        """Stop trading with a specific reason"""
        print(f"🛑 STOPPING TRADING: {reason}")
        self.stop_trading()
        
        # Notify frontend about automatic stop
        if self.callbacks['trade_update']:
            self.callbacks['trade_update']({
                'type': 'auto_stop',
                'reason': reason,
                'session_profit_loss': self.session_profit_loss,
                'timestamp': datetime.now().isoformat()
            })
