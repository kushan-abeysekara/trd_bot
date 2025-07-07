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
        self.duration_ticks = 1  # Setting default duration to 1 tick
        self.trade_history = []
        # Enhance trade result verification system
        self.trade_result_verification = {}
        self.verification_errors = 0
        self.last_market_movements = []  # Track recent market movements
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
        self.min_signal_interval = 5  # Reduced to 5 seconds for tick-based trading
        self.last_strategy_signals = {}  # Track last signal time per strategy
        self.strategy_cooldown = 15  # Reduced to 15 seconds for tick-based trading
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
        
        # Double stake pattern settings
        self.trades_since_double = 0
        self.next_double_trade = False
        self.default_trade_amount = 1.0
        self.trades_before_double = random.randint(2, 5)  # Random number between 2-5
        self.double_stake_active = False  # Flag for UI notification
        
        # Risk management settings
        self.risk_management = {
            'max_trades_per_session': 100,  # Increased from 20 to 100 as default max trades per session
            'max_consecutive_losses': 5,   # Default max consecutive losses
            'max_daily_loss': 0.0,         # Default max daily loss (0 means no limit)
            'cooling_period': 0,           # Default cooling period in minutes after hitting limits
            'session_trade_count': 0,      # Track trades in current session
            'consecutive_losses': 0,       # Track consecutive losses
            'daily_loss': 0.0,             # Track daily loss
            'risk_limits_hit': False,      # Flag to indicate if risk limits were hit
            'trading_suspended_until': 0,  # Timestamp until trading is suspended
            'limits_hit_reason': '',       # Reason for hitting limits
            'limits_enabled': True,        # New flag to enable/disable risk limits completely
            'warning_shown': False         # Track if warning was already shown
        }
        
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
            if new_balance > 0:
                # Store initial balance if not already set
                if self.initial_balance <= 0:
                    self.initial_balance = new_balance
                    print(f"Initial balance set in balance callback: ${new_balance}")
                
                if self.callbacks['balance_update']:
                    self.callbacks['balance_update'](new_balance)
            else:
                print(f"Ignored zero balance update")
                
        self.api.set_balance_callback(balance_update_callback)
        self.api.connect(connection_callback)
        
    def start_trading(self, amount: float, duration: int = 1):  # Default to 1 tick
        """Start automated trading with tick-based duration"""
        self.trade_amount = amount
        self.default_trade_amount = amount  # Store the default amount
        
        # Always use 1 tick duration
        self.duration_ticks = 1
        
        self.is_running = True
        
        # Reset session tracking
        self.reset_session_tracking()
        
        # Reset double stake pattern
        self.trades_since_double = 0
        self.next_double_trade = False
        self.trades_before_double = random.randint(2, 5)
        self.double_stake_active = False
        
        # Reset risk management tracking for new session
        self.reset_risk_management(full_reset=True)
        
        # Start strategy scanning
        self.strategy_scanning = True
        self.strategy_engine.start_scanning(self._handle_strategy_signal)
        
        # Make sure strategy engine max trades match our settings
        if hasattr(self.strategy_engine, 'max_session_trades'):
            self.strategy_engine.max_session_trades = self.risk_management['max_trades_per_session']
        
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
        """Simulate live price feed for strategy analysis - MODIFIED for tick trading"""
        base_price = 1000.0
        price = base_price
        tick_counter = 0
        
        while self.strategy_scanning:
            tick_counter += 1
            
            # Enhanced price movement simulation for better strategy triggering
            
            # Base volatility (adjusted for tick-based trading)
            change = random.uniform(-0.3, 0.3)  # ±0.3% change (reduced for tick-based)
            
            # Add volatility spikes (20% chance - adjusted for tick-based)
            if random.random() < 0.20:
                spike_intensity = random.uniform(0.4, 1.2)  # 0.4-1.2% spike (reduced)
                spike_direction = random.choice([-1, 1])
                change += spike_direction * spike_intensity
                print(f"📈 Price spike: {spike_direction * spike_intensity:.2f}%")
            
            # Apply normal price movement
            price = price * (1 + change / 100)
            
            # Ensure price doesn't go too far from base
            if price < base_price * 0.95:
                price = base_price * 0.95
            elif price > base_price * 1.05:
                price = base_price * 1.05
                
            # Feed price to strategy engine
            self.strategy_engine.add_tick(price)
            
            # Log every 30 ticks for monitoring (increased frequency for tick trading)
            if tick_counter % 30 == 0:
                indicators = self.strategy_engine.get_current_indicators()
                print(f"📊 Tick #{tick_counter}: Price={price:.2f}, RSI={indicators.get('rsi', 0):.1f}, "
                      f"Vol={indicators.get('volatility', 0):.2f}%, MACD={indicators.get('macd', 0):.3f}")
            
            # Faster tick intervals for tick-based trading
            time.sleep(random.uniform(0.2, 0.5))  # 200-500ms intervals
    
    def _handle_strategy_signal(self, signal: TradeSignal):
        """Handle trade signal from strategy engine - MODIFIED FOR TICK TRADING"""
        current_time = time.time()
        self.signal_count += 1
        
        print(f"📡 Signal #{self.signal_count} received: {signal.strategy_name} ({signal.confidence:.2f})")
        
        # Check risk management limits before processing signal
        if not self._check_risk_limits():
            return
        
        # Global signal interval check - shorter for tick-based trading
        if current_time - self.last_signal_time < self.min_signal_interval:
            remaining = self.min_signal_interval - (current_time - self.last_signal_time)
            print(f"⏰ Global cooldown - {remaining:.1f}s remaining")
            return
        
        # Individual strategy cooldown check - OPTIMIZED for tick trading
        strategy_name = signal.strategy_name
        dynamic_cooldown = self.strategy_cooldown
        
        # High-confidence signals get SHORTER cooldowns
        if signal.confidence > 0.85:
            dynamic_cooldown = 10  # 10 seconds for high-confidence signals in tick trading
            print(f"🔥 High-confidence signal - reduced cooldown to {dynamic_cooldown}s")
        elif signal.confidence > 0.75:
            dynamic_cooldown = 12  # 12 seconds for good signals
        # else use default 15 seconds
        
        if strategy_name in self.last_strategy_signals:
            time_since_last = current_time - self.last_strategy_signals[strategy_name]
            if time_since_last < dynamic_cooldown:
                remaining_cooldown = dynamic_cooldown - time_since_last
                print(f"❄️  Strategy {strategy_name} cooldown - {remaining_cooldown:.1f}s remaining")
                return
        
        # EMERGENCY BYPASS: If no trades for 30 seconds, accept any signal
        if current_time - self.last_signal_time > 30:
            print(f"🚨 EMERGENCY BYPASS: No trades for 30 seconds, accepting signal!")
            
        # Update timing trackers
        self.last_signal_time = current_time
        self.last_strategy_signals[strategy_name] = current_time
        
        # Execute trade based on signal
        contract_type = signal.direction  # 'CALL' or 'PUT'
        
        # Check if it's time to double the stake
        self.trades_since_double += 1
        if self.trades_since_double >= self.trades_before_double:
            self.next_double_trade = True
            self.double_stake_active = True
            self.trades_since_double = 0
            self.trades_before_double = random.randint(2, 5)  # Reset for next cycle
            print(f"💰 DOUBLE STAKE ACTIVATED for next trade! (after {self.trades_before_double} normal trades)")
        
        # Set the current trade amount (normal or doubled)
        current_trade_amount = self.default_trade_amount * 2 if self.next_double_trade else self.default_trade_amount
        
        print(f"🎯 EXECUTING TRADE #{self.stats['total_trades'] + 1}: {signal.strategy_name}")
        print(f"   Direction: {signal.direction} | Confidence: {signal.confidence:.2f} | Duration: {self.duration_ticks} tick")
        if self.next_double_trade:
            print(f"   💰 DOUBLE STAKE TRADE: ${current_trade_amount}")
        print(f"   Reason: {signal.entry_reason}")
        print(f"   Conditions: {', '.join(signal.conditions_met)}")
        print(f"   📊 Cooldown used: {dynamic_cooldown}s | Total signals: {self.signal_count}")
        
        # Place the trade asynchronously
        trade_thread = threading.Thread(
            target=self._place_strategy_trade_async, 
            args=(signal, current_trade_amount)
        )
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
                    'duration_ticks': self.duration_ticks,  # Use ticks duration
                    'indicators': self.strategy_engine.get_current_indicators(),
                    'total_strategies': 35,
                    'trade_number': self.stats['total_trades'] + 1,
                    'cooldown_used': dynamic_cooldown,
                    'signals_received': self.signal_count,
                    'double_stake': self.next_double_trade
                })
            except Exception as e:
                print(f"⚠️  Error sending signal to frontend: {e}")
    
    def _place_strategy_trade_async(self, signal: TradeSignal, trade_amount=None):
        """Place a trade based on strategy signal (async version)"""
        try:
            self._place_strategy_trade(signal, trade_amount)
        except Exception as e:
            print(f"❌ Error placing strategy trade: {e}")
            
    def _place_strategy_trade(self, signal: TradeSignal, trade_amount=None):
        """Place a trade based on strategy signal"""
        print(f"📋 Placing trade for {signal.strategy_name}...")
        
        # Use the specified trade amount or default
        amount_to_trade = trade_amount if trade_amount is not None else self.trade_amount
        is_double_stake = self.next_double_trade
        
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
                                'amount': amount_to_trade,
                                'buy_price': buy_price,
                                'duration': self.duration_ticks,  # Use tick-based duration
                                'duration_type': 'ticks',  # Specify duration type as ticks
                                'timestamp': datetime.now().isoformat(),
                                'status': 'active',
                                'strategy_name': signal.strategy_name,
                                'strategy_confidence': signal.confidence,
                                'entry_reason': signal.entry_reason,
                                'conditions_met': signal.conditions_met,
                                'double_stake': is_double_stake
                            }
                            
                            self.trade_history.append(trade_info)
                            print(f"🎯 Trade placed: {contract_id} for {signal.strategy_name}")
                            
                            # If this was a double stake trade, reset the flag
                            if self.next_double_trade:
                                self.next_double_trade = False
                                self.double_stake_active = False
                                print(f"💰 Double stake trade complete. Returning to normal stake: ${self.default_trade_amount}")
                            
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
                
        # Get proposal for tick-based trading (non-blocking)
        try:
            self.api.get_proposal_ticks(signal.direction, self.duration_ticks, amount_to_trade, proposal_callback)
        except Exception as e:
            print(f"❌ Error getting proposal: {e}")

    def _simulate_strategy_trade_result(self, trade_info, signal: TradeSignal):
        """Simulate trade result based on strategy confidence for tick-based trades"""
        def delayed_result():
            # Calculate wait time based on tick duration (accelerated for simulation)
            simulation_duration = 2  # 2 seconds per tick trade
            
            print(f"⏳ Simulating {trade_info['duration']} tick trade (waiting {simulation_duration} seconds)...")
            time.sleep(simulation_duration)  # Wait for simulated duration
            
            # Get the contract type and direction for accurate win/loss calculation
            contract_type = trade_info['type']  # 'CALL' or 'PUT'
            contract_id = trade_info.get('id', str(time.time()))
            
            # Generate market movement - this simulates what actually happens in the market
            # Now using a more deterministic approach with bias control
            # We maintain consistent market movements by tracking previous movements
            if len(self.last_market_movements) >= 5:
                # Use recent market pattern to ensure natural distribution
                # If we've had 3+ of the same movement in a row, increase chance of reversal
                up_count = self.last_market_movements.count(True)
                if up_count >= 4:  # Strong uptrend
                    market_move_up = random.random() < 0.3  # 30% chance to continue up
                elif up_count <= 1:  # Strong downtrend
                    market_move_up = random.random() < 0.7  # 70% chance to reverse
                else:
                    market_move_up = random.random() < 0.5  # Normal 50/50
            else:
                market_move_up = random.random() < 0.5  # 50/50 chance for first few trades
                
            # Record this movement for future reference
            self.last_market_movements.append(market_move_up)
            if len(self.last_market_movements) > 10:
                self.last_market_movements.pop(0)  # Keep only last 10 movements
            
            # Determine actual trade result based on market movement and contract type
            # CALL wins when market goes up, PUT wins when market goes down
            is_win = (market_move_up and contract_type == 'CALL') or (not market_move_up and contract_type == 'PUT')
            
            # Store comprehensive verification data
            verification_data = {
                'market_moved_up': market_move_up,
                'contract_type': contract_type, 
                'should_win': is_win,
                'market_direction': 'UP' if market_move_up else 'DOWN',
                'verification_time': datetime.now().isoformat(),
                'contract_id': contract_id
            }
            self.trade_result_verification[contract_id] = verification_data
            
            # Payout for tick-based trading (slightly lower due to shorter duration)
            payout_multiplier = 1.85  # Standard payout for 1-tick trades
            
            if is_win:
                payout = trade_info['buy_price'] * payout_multiplier
                profit_loss = payout - trade_info['buy_price']
                self.stats['winning_trades'] += 1
                print(f"✅ Trade WON: Market moved {'UP' if market_move_up else 'DOWN'}, {contract_type} contract")
                # Reset consecutive losses counter
                self.risk_management['consecutive_losses'] = 0
            else:
                profit_loss = -trade_info['buy_price']
                self.stats['losing_trades'] += 1
                print(f"❌ Trade LOST: Market moved {'UP' if market_move_up else 'DOWN'}, {contract_type} contract")
                # Increment consecutive losses counter
                self.risk_management['consecutive_losses'] += 1
                
            # Update daily loss tracker (negative for losses)
            self.risk_management['daily_loss'] += profit_loss if profit_loss < 0 else 0
            
            self.stats['total_trades'] += 1
            self.stats['total_profit_loss'] += profit_loss
            self.stats['winning_rate'] = (self.stats['winning_trades'] / self.stats['total_trades']) * 100
            
            # Update session profit/loss tracking
            self.session_profit_loss += profit_loss
            
            # Update actual balance (simulate balance change)
            self.api.update_balance(profit_loss)
            
            # Get the updated balance
            updated_balance = self.get_balance()
            
            # Double verification - ensure our trade result is logically consistent
            verified_result = self.verify_trade_outcome(contract_type, market_move_up)
            if verified_result != is_win:
                print(f"⚠️ VERIFICATION ERROR: Inconsistent trade result detected!")
                print(f"   Original: {is_win}, Verified: {verified_result}")
                self.verification_errors += 1
                # Use the verified result for consistency
                is_win = verified_result
                # Recalculate profit/loss based on corrected outcome
                if is_win:
                    profit_loss = trade_info['buy_price'] * payout_multiplier - trade_info['buy_price']
                else:
                    profit_loss = -trade_info['buy_price']
                # Update session profit/loss with corrected value
                self.session_profit_loss += profit_loss - trade_info.get('profit_loss', 0)
            
            # Update trade info with final results
            trade_info['result'] = 'win' if is_win else 'loss'
            trade_info['profit_loss'] = profit_loss
            trade_info['status'] = 'completed'
            trade_info['session_profit_loss'] = self.session_profit_loss
            trade_info['balance_after'] = updated_balance
            trade_info['initial_balance'] = self.initial_balance
            trade_info['real_profit'] = updated_balance - self.initial_balance
            trade_info['payout_multiplier'] = payout_multiplier if is_win else 0
            trade_info['market_movement'] = 'UP' if market_move_up else 'DOWN'
            trade_info['verification'] = verification_data
            trade_info['verification_timestamp'] = datetime.now().timestamp()
            
            print(f"💰 Trade Result: {trade_info['result'].upper()} | "
                  f"P&L: ${profit_loss:.2f} | "
                  f"Session P&L: ${self.session_profit_loss:.2f} | "
                  f"Real Profit: ${trade_info['real_profit']:.2f} | "
                  f"New Balance: ${updated_balance:.2f} | "
                  f"Strategy: {signal.strategy_name}")
            
            # Emit real-time balance update immediately with additional info
            if self.callbacks.get('balance_update'):
                self.callbacks['balance_update']({
                    'balance': updated_balance,
                    'initial_balance': self.initial_balance,
                    'real_profit': updated_balance - self.initial_balance
                })
            
            # Emit complete trade update with all final info
            if self.callbacks.get('trade_update'):
                self.callbacks['trade_update'](trade_info)
            
            # Register trade result with strategy engine for optimization and risk management
            strategy_name = signal.strategy_name
            self.strategy_engine.register_trade_result(
                strategy_name, 
                contract_type,
                is_win, 
                profit_loss
            )
            
            # Increment session trade counters after each completed trade
            self.risk_management['session_trade_count'] += 1
            
            # Sync the session trades count with the strategy engine
            if hasattr(self.strategy_engine, 'sync_session_trades'):
                self.strategy_engine.sync_session_trades(self.risk_management['session_trade_count'])
            
            if hasattr(self.strategy_engine, 'session_trades'):
                self.strategy_engine.session_trades += 1
                print(f"[SYNC] Incremented session_trades in strategy engine: {self.strategy_engine.session_trades}")

            # Check if take profit or stop loss limits are hit
            if self.check_profit_loss_limits():
                return  # Trading was stopped, exit the method
            
            if self.callbacks['stats_update']:
                self.callbacks['stats_update'](self.stats)
                
        # Run in separate thread
        result_thread = threading.Thread(target=delayed_result)
        result_thread.daemon = True
        result_thread.start()
    
    def verify_trade_outcome(self, contract_type, market_movement_up):
        """Verify trade outcome based on contract type and market movement"""
        # CALL contracts win when market moves up, PUT contracts win when market moves down
        return (market_movement_up and contract_type == 'CALL') or (not market_movement_up and contract_type == 'PUT')
    
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
        balance = self.api.get_balance_value()
        
        # If we got a valid balance and initial_balance is not set, set it
        if balance > 0 and self.initial_balance <= 0:
            self.initial_balance = balance
            print(f"Initial balance set in get_balance: ${balance}")
            
        return balance

    def get_stats(self):
        """Get trading statistics"""
        return self.stats
        
    def get_trade_history(self):
        """Get trade history with verification of results"""
        # Verify all trade results one more time before returning
        verified_history = []
        for trade in self.trade_history:
            trade_copy = dict(trade)  # Make a copy to avoid modifying the original
            
            # Double check the result calculation
            if 'type' in trade and 'market_movement' in trade:
                contract_type = trade['type']
                market_moved_up = trade['market_movement'] == 'UP'
                
                # Use our verification method for consistency
                correct_result = self.verify_trade_outcome(contract_type, market_moved_up)
                
                # If there's a mismatch, correct it
                if (correct_result and trade['result'] != 'win') or (not correct_result and trade['result'] != 'loss'):
                    print(f"⚠️ Correcting trade result inconsistency during history retrieval for trade {trade.get('id', '')}")
                    trade_copy['result'] = 'win' if correct_result else 'loss'
                    
                    # Also fix profit_loss to match the result
                    if correct_result:  # Should be a win
                        payout_multiplier = trade.get('payout_multiplier', 1.85)
                        trade_copy['profit_loss'] = trade['buy_price'] * payout_multiplier - trade['buy_price']
                    else:  # Should be a loss
                        trade_copy['profit_loss'] = -trade['buy_price']
                    
                    # Mark as corrected
                    trade_copy['corrected'] = True
                    
                # Add verification data if not present
                if 'verification' not in trade_copy:
                    trade_copy['verification'] = {
                        'market_moved_up': market_moved_up,
                        'contract_type': contract_type,
                        'should_win': correct_result,
                        'verification_added': 'during_retrieval'
                    }
            
            verified_history.append(trade_copy)
            
        return verified_history
    
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
        
    def _check_risk_limits(self):
        """Check if any risk management limits have been hit"""
        current_time = time.time()
        
        # Skip checks if limits are disabled
        if not self.risk_management['limits_enabled']:
            return True
            
        # Check if trading is suspended due to cooling period
        if self.risk_management['trading_suspended_until'] > current_time:
            remaining_time = int(self.risk_management['trading_suspended_until'] - current_time)
            print(f"🧊 Trading suspended for {remaining_time} more seconds due to {self.risk_management['limits_hit_reason']}")
            return False
            
        # Check max trades per session - only if limit is enabled (> 0)
        # NOTE: Don't increment counter here, it should be incremented AFTER successful trade completion
        if (self.risk_management['max_trades_per_session'] > 0 and 
            self.risk_management['session_trade_count'] >= self.risk_management['max_trades_per_session']):
            
            # Don't show the warning repeatedly
            if not self.risk_management['warning_shown']:
                print(f"🛑 Risk management: Maximum {self.risk_management['max_trades_per_session']} trades for this session reached.")
                print(f"To continue trading, either reset the session counter or disable the session limit.")
                self.risk_management['warning_shown'] = True
                
            self.risk_management['risk_limits_hit'] = True
            self.risk_management['limits_hit_reason'] = 'maximum trades per session'
            
            # Apply cooling period if configured
            if self.risk_management['cooling_period'] > 0:
                self.risk_management['trading_suspended_until'] = current_time + (self.risk_management['cooling_period'] * 60)
                print(f"🧊 Trading suspended for {self.risk_management['cooling_period']} minutes")
                
                # Notify frontend about risk limit (non-blocking)
                if self.callbacks['trade_update']:
                    try:
                        self.callbacks['trade_update']({
                            'type': 'risk_limit',
                            'reason': f"Maximum {self.risk_management['max_trades_per_session']} trades reached",
                            'suspended_for': self.risk_management['cooling_period'] * 60,
                            'timestamp': datetime.now().isoformat()
                        })
                    except Exception as e:
                        print(f"⚠️  Error sending risk limit notification: {e}")
            
            return False
            
        # Check consecutive losses limit
        if (self.risk_management['max_consecutive_losses'] > 0 and 
            self.risk_management['consecutive_losses'] >= self.risk_management['max_consecutive_losses']):
            print(f"🛑 Risk management: Maximum {self.risk_management['max_consecutive_losses']} consecutive losses reached.")
            self.risk_management['risk_limits_hit'] = True
            self.risk_management['limits_hit_reason'] = 'maximum consecutive losses'
            
            if self.risk_management['cooling_period'] > 0:
                self.risk_management['trading_suspended_until'] = current_time + (self.risk_management['cooling_period'] * 60)
                
            # Notify frontend about risk limit
            if self.callbacks['trade_update']:
                try:
                    self.callbacks['trade_update']({
                        'type': 'risk_limit',
                        'reason': f"Maximum {self.risk_management['max_consecutive_losses']} consecutive losses reached",
                        'suspended_for': self.risk_management['cooling_period'] * 60,
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    print(f"⚠️  Error sending risk limit notification: {e}")
            
            return False
            
        # Check max daily loss limit
        if (self.risk_management['max_daily_loss'] > 0 and 
            abs(self.risk_management['daily_loss']) >= self.risk_management['max_daily_loss']):
            print(f"🛑 Risk management: Maximum daily loss of ${self.risk_management['max_daily_loss']} reached.")
            self.risk_management['risk_limits_hit'] = True
            self.risk_management['limits_hit_reason'] = 'maximum daily loss'
            
            if self.risk_management['cooling_period'] > 0:
                self.risk_management['trading_suspended_until'] = current_time + (self.risk_management['cooling_period'] * 60)
                
            # Notify frontend about risk limit
            if self.callbacks['trade_update']:
                try:
                    self.callbacks['trade_update']({
                        'type': 'risk_limit',
                        'reason': f"Maximum daily loss of ${self.risk_management['max_daily_loss']} reached",
                        'suspended_for': self.risk_management['cooling_period'] * 60,
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    print(f"⚠️  Error sending risk limit notification: {e}")
            
            return False
        
        # Don't increment counter here - it should be incremented after successful trade completion
        return True
    
    def set_risk_limits(self, max_trades: int = None, max_losses: int = None, 
                       max_daily_loss: float = None, cooling_period: int = None, enabled: bool = None):
        """Set risk management limits"""
        # New parameter to enable/disable all risk limits
        if enabled is not None:
            self.risk_management['limits_enabled'] = enabled
            print(f"Risk management limits {'enabled' if enabled else 'disabled'}")
            
        if max_trades is not None:
            # Allow disabling session trade limit by setting to 0 or negative
            if max_trades <= 0:
                self.risk_management['max_trades_per_session'] = 0
                print(f"Session trade limit disabled (set to {self.risk_management['max_trades_per_session']})")
            else:
                self.risk_management['max_trades_per_session'] = int(max_trades)
                print(f"Set maximum trades per session to {self.risk_management['max_trades_per_session']}")
                
            # Sync with strategy engine
            if hasattr(self.strategy_engine, 'max_session_trades'):
                self.strategy_engine.max_session_trades = self.risk_management['max_trades_per_session']
            
        if max_losses is not None:
            self.risk_management['max_consecutive_losses'] = max(0, int(max_losses))
            print(f"Set maximum consecutive losses to {self.risk_management['max_consecutive_losses']}")
            
        if max_daily_loss is not None:
            self.risk_management['max_daily_loss'] = max(0, float(max_daily_loss))
            print(f"Set maximum daily loss to ${self.risk_management['max_daily_loss']}")
            
        if cooling_period is not None:
            self.risk_management['cooling_period'] = max(0, int(cooling_period))
            print(f"Set cooling period to {self.risk_management['cooling_period']} minutes")
            
    def reset_risk_management(self, full_reset=False):
        """Reset risk management tracking"""
        print(f"Resetting risk management tracking{' (full reset)' if full_reset else ''}")
        
        # Always reset these counters
        self.risk_management['session_trade_count'] = 0
        self.risk_management['consecutive_losses'] = 0
        self.risk_management['risk_limits_hit'] = False
        self.risk_management['trading_suspended_until'] = 0
        self.risk_management['limits_hit_reason'] = ''
        self.risk_management['warning_shown'] = False  # Reset warning flag
        
        # Also reset warning flags in strategy engine
        if hasattr(self.strategy_engine, 'reset_warnings'):
            self.strategy_engine.reset_warnings()
        
        # Only reset daily loss on full reset (typically done once per day)
        if full_reset:
            self.risk_management['daily_loss'] = 0.0
    
    def reset_session_trade_counter(self):
        """Reset only the session trade counter"""
        previous_count = self.risk_management['session_trade_count']
        self.risk_management['session_trade_count'] = 0
        self.risk_management['risk_limits_hit'] = False
        self.risk_management['trading_suspended_until'] = 0
        self.risk_management['warning_shown'] = False  # Reset warning flag
    
        if self.risk_management['limits_hit_reason'] == 'maximum trades per session':
            self.risk_management['limits_hit_reason'] = ''
            
        # Also reset warning flags and session trades in strategy engine
        if hasattr(self.strategy_engine, 'reset_warnings'):
            self.strategy_engine.reset_warnings()
        if hasattr(self.strategy_engine, 'reset_session_trades'):
            self.strategy_engine.reset_session_trades()
    
        print(f"🔄 Reset session trade counter from {previous_count} to 0")
        
        # Notify frontend about reset
        if self.callbacks['trade_update']:
            try:
                self.callbacks['trade_update']({
                    'type': 'risk_reset',
                    'message': f"Session trade counter reset from {previous_count} to 0",
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                print(f"⚠️  Error sending risk reset notification: {e}")
    
        return True
    
    def disable_session_limit(self):
        """Disable session trade limit"""
        previous_limit = self.risk_management['max_trades_per_session']
        self.risk_management['max_trades_per_session'] = 0
        self.risk_management['risk_limits_hit'] = False
        self.risk_management['warning_shown'] = False
        
        if self.risk_management['limits_hit_reason'] == 'maximum trades per session':
            self.risk_management['limits_hit_reason'] = ''
            self.risk_management['trading_suspended_until'] = 0
        
        # Also update the strategy engine
        if hasattr(self.strategy_engine, 'max_session_trades'):
            self.strategy_engine.max_session_trades = 0
        if hasattr(self.strategy_engine, 'reset_warnings'):
            self.strategy_engine.reset_warnings()
        
        print(f"🔓 Session trade limit disabled (was {previous_limit})")
        
        # Notify frontend about the change
        if self.callbacks['trade_update']:
            try:
                self.callbacks['trade_update']({
                    'type': 'risk_update',
                    'message': f"Session trade limit disabled",
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                print(f"⚠️  Error sending risk update notification: {e}")
        
        return True
    
    def get_bot_status(self):
        """Get current bot status for debugging"""
        current_time = time.time()
        
        # Calculate cooldown status for all strategies that have been used
        strategy_cooldown_status = {}
        for strategy, last_time in self.last_strategy_signals.items():
            time_since = current_time - last_time
            dynamic_cooldown = 60  # default
            # Apply same logic as in _handle_strategy_signal
            strategy_cooldown_status[strategy] = {
                'last_signal_time': last_time,
                'time_since_last': time_since,
                'cooldown_remaining': max(0, dynamic_cooldown - time_since),
                'is_ready': time_since >= dynamic_cooldown
            }
        
        status = {
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
            'emergency_bypass_active': (current_time - self.last_signal_time > 30) if self.last_signal_time > 0 else False,
            'double_stake_active': self.double_stake_active,
            'trades_since_double': self.trades_since_double,
            'trades_before_double': self.trades_before_double,
            'verification_errors': self.verification_errors,
            'market_movement_history': [('UP' if m else 'DOWN') for m in self.last_market_movements],
            
            # Add risk management information to status
            'risk_management': {
                'max_trades_per_session': self.risk_management['max_trades_per_session'],
                'current_trades': self.risk_management['session_trade_count'],
                'max_consecutive_losses': self.risk_management['max_consecutive_losses'],
                'current_consecutive_losses': self.risk_management['consecutive_losses'],
                'max_daily_loss': self.risk_management['max_daily_loss'],
                'current_daily_loss': self.risk_management['daily_loss'],
                'cooling_period_minutes': self.risk_management['cooling_period'],
                'trading_suspended': self.risk_management['trading_suspended_until'] > current_time,
                'suspended_remaining': max(0, self.risk_management['trading_suspended_until'] - current_time),
                'limits_hit': self.risk_management['risk_limits_hit'],
                'limits_hit_reason': self.risk_management['limits_hit_reason']
            },
        }
        
        return status
        
    def reset_session_tracking(self):
        """Reset session profit/loss tracking"""
        current_balance = self.get_balance()
        
        # Only set initial balance if we have a valid value
        if current_balance > 0:
            self.initial_balance = current_balance
            self.session_profit_loss = 0.0
            print(f"Session tracking reset. Initial balance: ${self.initial_balance}")
            
            # Add callback to notify frontend of initial balance
            if self.callbacks['balance_update']:
                try:
                    self.callbacks['balance_update']({
                        'balance': self.initial_balance,
                        'initial_balance': self.initial_balance,
                        'is_initial': True
                    })
                except Exception as e:
                    print(f"⚠️  Error sending initial balance update: {e}")
        else:
            print(f"⚠️ Cannot reset session with invalid balance: {current_balance}")
            # Try to refresh balance and retry after a delay
            self.api.refresh_balance()
            threading.Timer(2.0, self.reset_session_tracking).start()

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
