from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import time
import os
from trading_bot import TradingBot
from config import CORS_ORIGINS, FLASK_HOST, FLASK_PORT, FLASK_DEBUG
from cleanup import register_bot  # Import cleanup handler registration

app = Flask(__name__)
app.config['SECRET_KEY'] = 'deriv-trading-bot-secret'

# Configure CORS with environment-specific origins
CORS(app, origins=CORS_ORIGINS)

# Global bot instance
bot_instance = None
update_thread = None

# In-memory storage for updates
latest_balance = 0
latest_trades = []
latest_stats = {}
latest_indicators = {}
latest_session_stats = {}
latest_strategy_signal = None

# Global variables for client activity tracking
last_client_activity = time.time()
client_activity_timeout = 30  # seconds before considering client disconnected
client_activity_thread = None

def start_update_thread():
    """Start a thread to periodically update data from the bot"""
    global update_thread, latest_balance, latest_trades, latest_stats, latest_indicators
    
    def update_loop():
        global bot_instance, latest_balance, latest_trades, latest_stats, latest_indicators
        while bot_instance and bot_instance.api.is_connected:
            try:
                # Update balance with retry mechanism
                retry_count = 0
                max_retries = 3
                
                while retry_count < max_retries:
                    # Refresh balance from API
                    bot_instance.api.refresh_balance()
                    time.sleep(0.5)  # Short wait for balance update
                    
                    # Get current balance
                    current_balance = bot_instance.get_balance()
                    
                    # If we got a valid balance, use it
                    if current_balance > 0:
                        latest_balance = current_balance
                        break
                        
                    retry_count += 1
                    time.sleep(0.5)  # Wait before retry
                
                # Update stats
                latest_stats = bot_instance.get_stats()
                
                # Update trade history with verified trades
                latest_trades = bot_instance.get_trade_history()
                
                # Update indicators
                if hasattr(bot_instance, 'strategy_engine'):
                    latest_indicators = bot_instance.get_strategy_indicators()
                
                # Update session stats
                latest_session_stats.update({
                    'session_profit_loss': bot_instance.session_profit_loss,
                    'initial_balance': bot_instance.initial_balance,
                    'current_balance': latest_balance,
                    'take_profit_enabled': bot_instance.take_profit_enabled,
                    'take_profit_amount': bot_instance.take_profit_amount,
                    'stop_loss_enabled': bot_instance.stop_loss_enabled,
                    'stop_loss_amount': bot_instance.stop_loss_amount
                })
                
                # If initial_balance is still not set but we have a valid balance
                if bot_instance.initial_balance <= 0 and latest_balance > 0:
                    bot_instance.initial_balance = latest_balance
                    print(f"Initial balance set in update loop: {latest_balance}")
                
                time.sleep(2)  # Update every 2 seconds
            except Exception as e:
                print(f"Update thread error: {e}")
                time.sleep(5)  # Pause longer on error
    
    if update_thread is None or not update_thread.is_alive():
        update_thread = threading.Thread(target=update_loop)
        update_thread.daemon = True
        update_thread.start()


def start_client_activity_monitor():
    """Start a thread to monitor client activity and stop bot if client disconnects"""
    global client_activity_thread
    
    def monitor_loop():
        global last_client_activity, bot_instance
        while bot_instance and bot_instance.api.is_connected:
            try:
                # Check if client has been inactive
                time_since_last_activity = time.time() - last_client_activity
                if time_since_last_activity > client_activity_timeout and bot_instance.is_running:
                    print(f"⚠️ Client inactivity detected ({time_since_last_activity:.1f}s). Stopping bot for safety.")
                    bot_instance.stop_trading()
                    # Don't disconnect, just stop trading for safety
                
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                print(f"Client activity monitor error: {e}")
                time.sleep(5)
    
    if client_activity_thread is None or not client_activity_thread.is_alive():
        client_activity_thread = threading.Thread(target=monitor_loop)
        client_activity_thread.daemon = True
        client_activity_thread.start()


@app.route('/api/beacon', methods=['POST'])
def client_beacon():
    """Endpoint for client to signal it's still active"""
    global last_client_activity, bot_instance
    
    # Update last client activity timestamp
    last_client_activity = time.time()
    
    # Start monitoring thread if not already started
    start_client_activity_monitor()
    
    # Return bot status info
    is_connected = False
    is_trading = False
    
    if bot_instance:
        is_connected = bot_instance.api.is_connected
        is_trading = bot_instance.is_running
        
    return jsonify({
        'received': True,
        'timestamp': last_client_activity,
        'bot_connected': is_connected,
        'bot_trading': is_trading
    }), 200


@app.route('/api/connect', methods=['POST'])
def connect_api():
    """Connect to Deriv API with provided token"""
    global bot_instance, latest_balance, last_client_activity
    
    # Update client activity time
    last_client_activity = time.time()
    
    # Start client activity monitor
    start_client_activity_monitor()
    
    data = request.get_json()
    api_token = data.get('api_token')
    
    if not api_token:
        return jsonify({'error': 'API token is required'}), 400
        
    try:
        # Create new bot instance
        bot_instance = TradingBot(api_token)
        
        # Register bot with cleanup handler
        register_bot(bot_instance)
        
        # Set up callbacks
        def on_connection_status(success, error=None):
            global latest_balance
            if success:
                print("Connection successful, initializing balance...")
                # Force multiple balance requests with increasing delays for reliability
                threading.Timer(0.5, bot_instance.api.refresh_balance).start()
                threading.Timer(1.5, bot_instance.api.refresh_balance).start()
                threading.Timer(3.0, bot_instance.api.refresh_balance).start()
                
                # Start the update thread after a short delay
                threading.Timer(3.0, start_update_thread).start()
            
        def on_trade_update(trade_info):
            global latest_trades, latest_balance
            # Add trade to history
            if trade_info not in latest_trades:
                latest_trades.insert(0, trade_info)
            # Update balance if provided
            if 'balance_after' in trade_info and trade_info['balance_after'] > 0:
                latest_balance = trade_info['balance_after']
            
        def on_stats_update(stats):
            global latest_stats
            latest_stats = stats
            
        def on_strategy_signal(signal_data):
            global latest_strategy_signal
            latest_strategy_signal = signal_data
            
        def on_balance_update(balance_data):
            global latest_balance, bot_instance
            
            # Process different balance update formats
            if isinstance(balance_data, dict):
                new_balance = balance_data.get('balance')
                if new_balance and new_balance > 0:
                    latest_balance = new_balance
                    
                    # Also update initial_balance if this is the first balance update
                    if balance_data.get('is_initial', False) and bot_instance and bot_instance.initial_balance <= 0:
                        bot_instance.initial_balance = new_balance
                        print(f"Initial balance set from callback: {new_balance}")
            elif isinstance(balance_data, (int, float)) and balance_data > 0:
                latest_balance = balance_data
                
                # If this is the first valid balance and initial_balance is not set
                if bot_instance and bot_instance.initial_balance <= 0:
                    bot_instance.initial_balance = balance_data
                    print(f"Initial balance set from numeric callback: {balance_data}")
            
        bot_instance.set_callback('connection_status', on_connection_status)
        bot_instance.set_callback('trade_update', on_trade_update)
        bot_instance.set_callback('stats_update', on_stats_update)
        bot_instance.set_callback('strategy_signal', on_strategy_signal)
        bot_instance.set_callback('balance_update', on_balance_update)
        
        # Connect to API
        bot_instance.connect()
        
        return jsonify({'message': 'Connecting to Deriv API...'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/connection-status', methods=['GET'])
def get_connection_status():
    """Check connection status"""
    global bot_instance, latest_balance
    
    if not bot_instance:
        return jsonify({'connected': False}), 200
    
    # Try to get a fresh balance value
    try:
        # Force refresh balance if it's 0
        if latest_balance <= 0:
            bot_instance.api.refresh_balance()
            time.sleep(0.5)  # Short wait for balance update
            fresh_balance = bot_instance.get_balance()
            if fresh_balance > 0:
                latest_balance = fresh_balance
                print(f"Updated zero balance to: {latest_balance}")
    except Exception as e:
        print(f"Error refreshing balance: {e}")
    
    initial_bal = getattr(bot_instance, 'initial_balance', 0)
    return jsonify({
        'connected': bot_instance.api.is_connected,
        'balance': latest_balance,
        'initial_balance': initial_bal
    }), 200


@app.route('/api/start-trading', methods=['POST'])
def start_trading():
    """Start automated trading"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    data = request.get_json()
    amount = float(data.get('amount', 1.0))
    
    try:
        # Always use 1 tick for duration
        bot_instance.start_trading(amount, 1)
        return jsonify({'message': 'Trading started'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stop-trading', methods=['POST'])
def stop_trading():
    """Stop automated trading"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        bot_instance.stop_trading()
        return jsonify({'message': 'Trading stopped'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/balance', methods=['GET'])
def get_balance():
    """Get current balance"""
    global bot_instance, latest_balance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        # Refresh balance from server before returning
        bot_instance.api.refresh_balance()
        latest_balance = bot_instance.get_balance()
        return jsonify({'balance': latest_balance}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/refresh-balance', methods=['POST'])
def refresh_balance():
    """Manually refresh balance from Deriv API"""
    global bot_instance, latest_balance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        # Force multiple refresh attempts with increasing delays
        bot_instance.api.refresh_balance()
        threading.Timer(1.0, bot_instance.api.refresh_balance).start()
        threading.Timer(2.0, bot_instance.api.refresh_balance).start()
        
        # Wait a moment for the first response
        time.sleep(1)
        
        # Try up to 5 times to get a valid balance
        retry_count = 0
        max_retries = 5
        fresh_balance = 0
        
        while retry_count < max_retries and fresh_balance <= 0:
            fresh_balance = bot_instance.get_balance()
            if fresh_balance > 0:
                break
            time.sleep(0.5)
            retry_count += 1
            print(f"Refresh retry {retry_count}/{max_retries}: {fresh_balance}")
        
        if fresh_balance > 0:
            latest_balance = fresh_balance
            # Also update initial_balance if not yet set
            if bot_instance.initial_balance <= 0:
                bot_instance.initial_balance = fresh_balance
                print(f"Initial balance set during refresh: {fresh_balance}")
        
        return jsonify({'balance': latest_balance, 'message': 'Balance refreshed'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get trading statistics"""
    global bot_instance, latest_stats
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        latest_stats = bot_instance.get_stats()
        return jsonify(latest_stats), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trade-history', methods=['GET'])
def get_trade_history():
    """Get trade history with forced refresh"""
    global bot_instance, latest_trades
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        # IMPORTANT FIX: FORCE refresh trade history directly from bot and verify status counts
        fresh_trades = bot_instance.get_trade_history()
        
        # Force check for active trades that should be completed
        if fresh_trades:
            active_trades = [t for t in fresh_trades if t.get('status') == 'active']
            for trade in active_trades:
                # Check if this trade should already be completed (15 sec timeout)
                if 'timestamp' in trade:
                    try:
                        from datetime import datetime, timedelta
                        trade_time = datetime.fromisoformat(trade['timestamp'])
                        time_passed = datetime.now() - trade_time
                        
                        # If more than 15 seconds passed, trade should be completed
                        if time_passed > timedelta(seconds=15):
                            trade['status'] = 'completed'
                            trade['result'] = 'timeout'
                            trade['profit_loss'] = -float(trade.get('amount', 1.0))
                            trade['outcome_source'] = 'forced_timeout'
                            print(f"⚠️ Forced completion of stale trade {trade.get('id')}")
                    except Exception as e:
                        print(f"Error checking trade timestamp: {e}")
        
        latest_trades = fresh_trades  # Update global cache
        
        # Debug information
        active_trades = [t for t in fresh_trades if t.get('status') == 'active']
        completed_trades = [t for t in fresh_trades if t.get('status') == 'completed']
        
        return jsonify({
            'trades': fresh_trades,
            'total_count': len(fresh_trades),
            'active_count': len(active_trades),
            'completed_count': len(completed_trades),
            'debug_info': {
                'last_5_trades': fresh_trades[:5] if fresh_trades else [],
                'active_trade_ids': [t.get('id') for t in active_trades],
                'completed_trade_ids': [t.get('id') for t in completed_trades]
            },
            'verification_errors': getattr(bot_instance, 'verification_errors', 0),
            'verified': True,
            'timestamp': time.time()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/verify-trades', methods=['POST'])
def verify_trades():
    """Manually trigger verification of all trade results"""
    global bot_instance, latest_trades
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        # Fetch fresh, verified trade history
        verified_trades = bot_instance.get_trade_history()
        
        # Count corrections
        corrections = sum(1 for trade in verified_trades if trade.get('corrected', False))
        
        # Update latest trades
        latest_trades = verified_trades
        
        return jsonify({
            'message': f'Trade verification complete. {corrections} trades corrected.',
            'trades': latest_trades,
            'verification_errors': getattr(bot_instance, 'verification_errors', 0),
            'corrections': corrections
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/disconnect', methods=['POST'])
def disconnect():
    """Disconnect from API"""
    global bot_instance, update_thread
    
    if bot_instance:
        bot_instance.disconnect()
        bot_instance = None
        
    return jsonify({'message': 'Disconnected'}), 200


@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    """Get list of available strategies"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        strategies = bot_instance.get_active_strategies()
        return jsonify({'strategies': strategies}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/strategy-status', methods=['GET'])
def get_strategy_status():
    """Get real-time status of all 35 strategies"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        status = bot_instance.strategy_engine.get_strategy_status()
        return jsonify(status), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/strategy-performance', methods=['GET'])
def get_strategy_performance():
    """Get performance summary of all strategies"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        performance = bot_instance.strategy_engine.get_performance_summary()
        return jsonify(performance), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reset-strategy-stats', methods=['POST'])
def reset_strategy_stats():
    """Reset strategy performance statistics"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        bot_instance.strategy_engine.reset_performance_stats()
        return jsonify({'message': 'Strategy statistics reset'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/indicators', methods=['GET'])
def get_indicators():
    """Get current technical indicators"""
    global bot_instance, latest_indicators
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        latest_indicators = bot_instance.get_strategy_indicators()
        return jsonify(latest_indicators), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trading-mode', methods=['POST'])
def set_trading_mode():
    """Set trading mode (random or strategy)"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    data = request.get_json()
    mode = data.get('mode', 'strategy')
    
    if mode not in ['random', 'strategy']:
        return jsonify({'error': 'Invalid mode. Use "random" or "strategy"'}), 400
        
    try:
        bot_instance.set_trading_mode(mode)
        return jsonify({'message': f'Trading mode set to {mode}'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/set-take-profit', methods=['POST'])
def set_take_profit():
    """Set take profit settings"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    data = request.get_json()
    enabled = data.get('enabled', False)
    amount = float(data.get('amount', 0.0))
    
    try:
        bot_instance.set_take_profit(enabled, amount)
        return jsonify({
            'message': f'Take profit {"enabled" if enabled else "disabled"}',
            'enabled': enabled,
            'amount': amount
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/set-stop-loss', methods=['POST'])
def set_stop_loss():
    """Set stop loss settings"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    data = request.get_json()
    enabled = data.get('enabled', False)
    amount = float(data.get('amount', 0.0))
    
    try:
        bot_instance.set_stop_loss(enabled, amount)
        return jsonify({
            'message': f'Stop loss {"enabled" if enabled else "disabled"}',
            'enabled': enabled,
            'amount': amount
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bot-status', methods=['GET'])
def get_bot_status():
    """Get current bot status for debugging"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        status = bot_instance.get_bot_status()
        return jsonify(status), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reset-cooldowns', methods=['POST'])
def reset_cooldowns():
    """Reset all strategy cooldowns (emergency function)"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        # Reset all cooldown timers
        bot_instance.last_signal_time = 0
        bot_instance.last_strategy_signals = {}
        bot_instance.signal_count = 0
        
        return jsonify({
            'message': 'All cooldowns reset', 
            'timestamp': time.time()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/session-stats', methods=['GET'])
def get_session_stats():
    """Get session trading statistics"""
    global bot_instance, latest_session_stats
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        latest_session_stats = {
            'session_profit_loss': bot_instance.session_profit_loss,
            'initial_balance': bot_instance.initial_balance,
            'current_balance': bot_instance.get_balance(),
            'take_profit_enabled': bot_instance.take_profit_enabled,
            'take_profit_amount': bot_instance.take_profit_amount,
            'stop_loss_enabled': bot_instance.stop_loss_enabled,
            'stop_loss_amount': bot_instance.stop_loss_amount,
            'is_running': bot_instance.is_running
        }
        return jsonify(latest_session_stats), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/latest-signal', methods=['GET'])
def get_latest_signal():
    """Get latest strategy signal"""
    global latest_strategy_signal
    
    if latest_strategy_signal:
        return jsonify(latest_strategy_signal), 200
    else:
        return jsonify({'message': 'No signals generated yet'}), 200


@app.route('/api/updates', methods=['GET'])
def get_updates():
    """Get all latest updates in one call (efficient polling)"""
    global latest_balance, latest_trades, latest_stats, latest_indicators, latest_session_stats, latest_strategy_signal, bot_instance, last_client_activity
    
    # Update client activity time
    last_client_activity = time.time()
    
    is_connected = bot_instance and bot_instance.api.is_connected if bot_instance else False
    is_trading = bot_instance and bot_instance.is_running if bot_instance else False
    initial_balance = getattr(bot_instance, 'initial_balance', 0) if bot_instance else 0
    
    # FORCE REFRESH trade history to get latest status
    if bot_instance:
        fresh_trades = bot_instance.get_trade_history()
        # Update latest_trades if we got fresh data
        if fresh_trades:
            latest_trades = fresh_trades
    
    return jsonify({
        'connected': is_connected,
        'trading': is_trading,
        'balance': latest_balance,
        'initial_balance': initial_balance,
        'stats': latest_stats,
        'recent_trades': latest_trades[:10] if latest_trades else [],  # Return more trades for better visibility
        'trade_history_full': latest_trades,  # Include full history for debugging
        'indicators': latest_indicators,
        'session_stats': latest_session_stats,
        'latest_signal': latest_strategy_signal,
        'timestamp': time.time(),
        'trade_count_debug': {
            'total': len(latest_trades) if latest_trades else 0,
            'active': len([t for t in latest_trades if t.get('status') == 'active']) if latest_trades else 0,
            'completed': len([t for t in latest_trades if t.get('status') == 'completed']) if latest_trades else 0
        }
    }), 200


@app.route('/api/strategy-optimizer', methods=['GET'])
def get_optimizer_status():
    """Get strategy optimizer status and best parameters"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        best_params = {}
        if hasattr(bot_instance, 'strategy_engine') and hasattr(bot_instance.strategy_engine, 'optimizer'):
            optimizer = bot_instance.strategy_engine.optimizer
            best_params = optimizer.best_parameters
            
        return jsonify({
            'optimizer_active': True,
            'best_parameters': best_params,
            'strategies_optimized': len(best_params)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/risk-management', methods=['GET'])
def get_risk_management():
    """Get current risk management status"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        if hasattr(bot_instance, 'strategy_engine'):
            consecutive_losses = bot_instance.strategy_engine.consecutive_losses
            max_consecutive_losses = bot_instance.strategy_engine.max_consecutive_losses
            session_trades = bot_instance.strategy_engine.session_trades
            max_session_trades = bot_instance.strategy_engine.max_session_trades
            
            return jsonify({
                'consecutive_losses': consecutive_losses,
                'max_consecutive_losses': max_consecutive_losses,
                'session_trades': session_trades,
                'max_session_trades': max_session_trades,
                'trading_blocked': consecutive_losses >= max_consecutive_losses or session_trades >= max_session_trades
            }), 200
        else:
            return jsonify({'error': 'Strategy engine not initialized'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/risk-management/reset-session-counter', methods=['POST'])
def reset_session_trade_counter():
    """Reset the session trade counter to allow more trades"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        if bot_instance.reset_session_trade_counter():
            return jsonify({
                'success': True,
                'message': 'Session trade counter reset successfully',
                'current_count': 0
            }), 200
        else:
            return jsonify({'error': 'Failed to reset session trade counter'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/risk-management/disable-session-limit', methods=['POST'])
def disable_session_trade_limit():
    """Disable the session trade limit completely"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        if bot_instance.disable_session_limit():
            return jsonify({
                'success': True,
                'message': 'Session trade limit disabled successfully'
            }), 200
        else:
            return jsonify({'error': 'Failed to disable session trade limit'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/risk-management/set-limits', methods=['POST'])
def set_risk_limits():
    """Set risk management limits"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
    
    data = request.get_json()
    max_trades = data.get('max_trades')
    max_losses = data.get('max_losses')
    max_daily_loss = data.get('max_daily_loss')
    cooling_period = data.get('cooling_period')
    enabled = data.get('enabled')
    
    try:
        bot_instance.set_risk_limits(
            max_trades=max_trades,
            max_losses=max_losses,
            max_daily_loss=max_daily_loss,
            cooling_period=cooling_period,
            enabled=enabled
        )
        
        return jsonify({
            'success': True,
            'message': 'Risk limits updated successfully',
            'current_settings': {
                'max_trades': bot_instance.risk_management['max_trades_per_session'],
                'max_losses': bot_instance.risk_management['max_consecutive_losses'],
                'max_daily_loss': bot_instance.risk_management['max_daily_loss'],
                'cooling_period': bot_instance.risk_management['cooling_period'],
                'enabled': bot_instance.risk_management['limits_enabled']
            }
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/risk-management/reset', methods=['POST'])
def reset_risk_management():
    """Reset risk management counters"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    data = request.get_json() or {}
    full_reset = data.get('full_reset', False)
    
    try:
        # Use the bot's native risk management reset function
        bot_instance.reset_risk_management(full_reset=full_reset)
        
        return jsonify({
            'success': True,
            'message': f'Risk management counters reset {"(full reset)" if full_reset else ""}',
            'risk_management': bot_instance.get_bot_status().get('risk_management', {})
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trade-verification', methods=['GET'])
def get_trade_verification():
    """Get trade verification showing real vs simulated results"""
    global bot_instance, latest_trades
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        # Get verified trades with outcome source information
        verified_trades = bot_instance.get_trade_history()
        
        # Categorize trades by outcome source
        real_outcomes = [t for t in verified_trades if t.get('outcome_source') == 'deriv_api']
        fallback_outcomes = [t for t in verified_trades if t.get('outcome_source') == 'fallback_simulation']
        
        # Calculate accuracy metrics
        total_trades = len(verified_trades)
        real_count = len(real_outcomes)
        fallback_count = len(fallback_outcomes)
        
        return jsonify({
            'total_trades': total_trades,
            'real_outcomes': real_count,
            'fallback_outcomes': fallback_count,
            'real_percentage': (real_count / max(1, total_trades)) * 100,
            'trades_by_source': {
                'deriv_api': real_outcomes[-10:],  # Last 10 real outcomes
                'fallback_simulation': fallback_outcomes[-10:]  # Last 10 fallback outcomes
            },
            'verification_summary': {
                'using_real_deriv_outcomes': real_count > 0,
                'fallback_rate': (fallback_count / max(1, total_trades)) * 100,
                'outcome_sources_available': ['deriv_api', 'fallback_simulation']
            }
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Register cleanup handler
    import atexit
    
    def cleanup_on_exit():
        global bot_instance
        print("Server shutting down, cleaning up resources...")
        if bot_instance:
            if bot_instance.is_running:
                print("Stopping trading bot...")
                bot_instance.stop_trading()
            print("Disconnecting from API...")
            bot_instance.disconnect()
    
    atexit.register(cleanup_on_exit)
    
    app.run(debug=FLASK_DEBUG, host=FLASK_HOST, port=FLASK_PORT)
