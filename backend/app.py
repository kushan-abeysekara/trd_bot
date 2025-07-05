from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import time
from trading_bot import TradingBot
import config

app = Flask(__name__)
app.config['SECRET_KEY'] = 'deriv-trading-bot-secret'
CORS(app, origins="*")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=True, engineio_logger=True)

# Global bot instance
bot_instance = None
balance_update_thread = None


@app.route('/api/connect', methods=['POST'])
def connect_api():
    """Connect to Deriv API with provided token"""
    global bot_instance
    
    data = request.get_json()
    api_token = data.get('api_token')
    
    if not api_token:
        return jsonify({'error': 'API token is required'}), 400
        
    try:
        # Create new bot instance
        bot_instance = TradingBot(api_token)
        
        # Set up callbacks
        def on_connection_status(success, error=None):
            if success:
                socketio.emit('connection_status', {'connected': True, 'balance': bot_instance.get_balance()})
                start_balance_updates()
            else:
                socketio.emit('connection_status', {'connected': False, 'error': error})
                
        def on_trade_update(trade_info):
            # Emit trade update first
            socketio.emit('trade_update', trade_info)
            
            # Emit balance update if included in trade info
            if 'balance_after' in trade_info:
                socketio.emit('balance_update', {'balance': trade_info['balance_after']})
            else:
                # Fallback to getting current balance
                current_balance = bot_instance.get_balance()
                socketio.emit('balance_update', {'balance': current_balance})
            
            # Also emit session stats update
            if 'session_pnl' in trade_info:
                session_stats = {
                    'session_profit_loss': trade_info['session_pnl'],
                    'initial_balance': bot_instance.initial_balance,
                    'current_balance': trade_info.get('balance_after', bot_instance.get_balance()),
                    'take_profit_enabled': bot_instance.take_profit_enabled,
                    'take_profit_amount': bot_instance.take_profit_amount,
                    'stop_loss_enabled': bot_instance.stop_loss_enabled,
                    'stop_loss_amount': bot_instance.stop_loss_amount
                }
                socketio.emit('session_stats_update', session_stats)
            
        def on_stats_update(stats):
            socketio.emit('stats_update', stats)
            
        def on_strategy_signal(signal_data):
            socketio.emit('strategy_signal', signal_data)
            
        def on_balance_update(balance_data):
            """Handle real-time balance updates"""
            if isinstance(balance_data, dict):
                socketio.emit('balance_update', balance_data)
            else:
                socketio.emit('balance_update', {'balance': balance_data})
            
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


@app.route('/api/start-trading', methods=['POST'])
def start_trading():
    """Start automated trading"""
    global bot_instance
    from config import MIN_TRADE_AMOUNT, DEFAULT_TRADE_AMOUNT
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    data = request.get_json()
    amount = float(data.get('amount', DEFAULT_TRADE_AMOUNT))
    
    # Validate minimum trade amount
    if amount < MIN_TRADE_AMOUNT:
        return jsonify({'error': f'Trade amount must be at least ${MIN_TRADE_AMOUNT}'}), 400
        
    duration = int(data.get('duration', 5))
    
    try:
        bot_instance.start_trading(amount, duration)
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
    global bot_instance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        # Refresh balance from server before returning
        bot_instance.api.refresh_balance()
        balance = bot_instance.get_balance()
        return jsonify({'balance': balance}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/refresh-balance', methods=['POST'])
def refresh_balance():
    """Manually refresh balance from Deriv API"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        # Force refresh balance from server
        bot_instance.api.refresh_balance()
        
        # Wait a moment for the response
        time.sleep(1)
        
        balance = bot_instance.get_balance()
        
        # Emit to all connected clients
        socketio.emit('balance_update', {'balance': balance})
        
        return jsonify({'balance': balance, 'message': 'Balance refreshed'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get trading statistics"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        stats = bot_instance.get_stats()
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trade-history', methods=['GET'])
def get_trade_history():
    """Get trade history"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        history = bot_instance.get_trade_history()
        return jsonify({'trades': history}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/disconnect', methods=['POST'])
def disconnect():
    """Disconnect from API"""
    global bot_instance, balance_update_thread
    
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
    global bot_instance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        indicators = bot_instance.get_strategy_indicators()
        return jsonify(indicators), 200
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
    """Get detailed bot status for debugging"""
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
    global bot_instance
    
    if not bot_instance:
        return jsonify({'error': 'Not connected to API'}), 400
        
    try:
        stats = {
            'session_profit_loss': bot_instance.session_profit_loss,
            'initial_balance': bot_instance.initial_balance,
            'current_balance': bot_instance.get_balance(),
            'take_profit_enabled': bot_instance.take_profit_enabled,
            'take_profit_amount': bot_instance.take_profit_amount,
            'stop_loss_enabled': bot_instance.stop_loss_enabled,
            'stop_loss_amount': bot_instance.stop_loss_amount,
            'is_running': bot_instance.is_running
        }
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def start_balance_updates():
    """Start real-time balance and strategy updates"""
    global balance_update_thread, bot_instance
    
    def update_balance():
        while bot_instance and bot_instance.api.is_connected:
            try:
                # Request fresh balance from Deriv API
                bot_instance.api.refresh_balance()
                
                # Also emit current balance
                balance = bot_instance.get_balance()
                socketio.emit('balance_update', {'balance': balance})
                
                # Get strategy status every cycle
                if hasattr(bot_instance, 'strategy_engine'):
                    strategy_status = bot_instance.strategy_engine.get_strategy_status()
                    socketio.emit('strategy_status_update', strategy_status)
                
                time.sleep(3)  # Update every 3 seconds (more frequent)
            except Exception as e:
                print(f"Balance update error: {e}")
                break
                
    if balance_update_thread is None or not balance_update_thread.is_alive():
        balance_update_thread = threading.Thread(target=update_balance)
        balance_update_thread.daemon = True
        balance_update_thread.start()


@socketio.on('connect')
def handle_connect():
    """Handle socket connection"""
    print('Client connected')
    emit('message', {'data': 'Connected to trading bot server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle socket disconnection"""
    print('Client disconnected')


@socketio.on('request_strategy_update')
def handle_strategy_update_request():
    """Handle client request for strategy update"""
    global bot_instance
    if bot_instance and hasattr(bot_instance, 'strategy_engine'):
        try:
            # Send strategy status
            strategy_status = bot_instance.strategy_engine.get_strategy_status()
            emit('strategy_status_update', strategy_status)
            
            # Send performance summary
            performance = bot_instance.strategy_engine.get_performance_summary()
            emit('strategy_performance_update', performance)
            
            # Send current indicators
            indicators = bot_instance.strategy_engine.get_current_indicators()
            emit('indicators_update', indicators)
            
        except Exception as e:
            emit('error', {'message': f'Strategy update error: {str(e)}'})


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
