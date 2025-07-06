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
                # Update balance
                bot_instance.api.refresh_balance()
                latest_balance = bot_instance.get_balance()
                
                # Update stats
                latest_stats = bot_instance.get_stats()
                
                # Update trade history
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
                # Store initial balance right away when connected
                initial_balance = bot_instance.get_balance()
                bot_instance.initial_balance = initial_balance
                latest_balance = initial_balance
                print(f"Initial balance set to: {initial_balance}")  # Debug log
                # Start the update thread
                start_update_thread()
            
        def on_trade_update(trade_info):
            global latest_trades, latest_balance
            # Add trade to history
            if trade_info not in latest_trades:
                latest_trades.insert(0, trade_info)
            # Update balance if provided
            if 'balance_after' in trade_info:
                latest_balance = trade_info['balance_after']
            
        def on_stats_update(stats):
            global latest_stats
            latest_stats = stats
            
        def on_strategy_signal(signal_data):
            global latest_strategy_signal
            latest_strategy_signal = signal_data
            
        def on_balance_update(balance_data):
            global latest_balance
            if isinstance(balance_data, dict):
                latest_balance = balance_data.get('balance', latest_balance)
            else:
                latest_balance = balance_data
            
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
        # Force refresh balance from server
        bot_instance.api.refresh_balance()
        
        # Wait a moment for the response
        time.sleep(1)
        
        latest_balance = bot_instance.get_balance()
        
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
    """Get trade history"""
    global bot_instance, latest_trades
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        latest_trades = bot_instance.get_trade_history()
        return jsonify({'trades': latest_trades}), 200
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
    
    return jsonify({
        'connected': is_connected,
        'trading': is_trading,
        'balance': latest_balance,
        'initial_balance': initial_balance,
        'stats': latest_stats,
        'recent_trades': latest_trades[:5] if latest_trades else [],
        'indicators': latest_indicators,
        'session_stats': latest_session_stats,
        'latest_signal': latest_strategy_signal,
        'timestamp': time.time()
    }), 200


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
